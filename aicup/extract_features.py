"""在本機 CPU 抽取棋力預測特徵，輸出 bit 打包的 .npz 分片（格式見 packed.py）

產出 rank-features/{train,val}/*.npz。特徵內容與教學 baseline 一致，但檔案格式不同，
教學 notebook 讀不了。預設以玩家切分驗證集，val 的玩家不會出現在 train。

用法（在 repo 根目錄）：
    python -m aicup.extract_features                 # 全量，以玩家切分
    python -m aicup.extract_features --frac 0.1      # 先抽 10% 測速度
    python -m aicup.extract_features --val-by game   # 逐局切分，與教學 baseline 一致
    python -m aicup.extract_features --val-size 180000 --out rank-features-v180000   # 擴大依玩家切分的 val

中斷後重跑同一個指令會跳過已完成的分片。換參數前要先刪除輸出目錄。
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from aicup import packed

# 與教學 baseline 一樣
LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
RANK_TO_ID = {lv: i for i, lv in enumerate(LEVELS)}
COLOR_TO_ID = {'B': 0, 'W': 1}
SEED = 42
VAL_SIZE = 100_000
FORMAT = 'packed-v1'


def load_dataframe(train_dir):
    paths = sorted(glob.glob(os.path.join(train_dir, 'train_*.csv')))
    if len(paths) != len(LEVELS):
        raise SystemExit('%s 下找到 %d 個 train_*.csv，預期 %d 個'
                         % (train_dir, len(paths), len(LEVELS)))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def split_by_game(df, val_size):
    """逐局切分，與教學 baseline 一樣：打亂後取前 val_size 局，同一位玩家可能同時在 train 與 val。"""
    return df.iloc[:val_size], df.iloc[val_size:]


def split_by_player(df, val_size):
    """整個玩家的棋局只會落在同一邊，驗證集約 val_size 局。

    player_id 可能只在單一等級的 CSV 內唯一，所以用 (rank, player_id) 當玩家。
    """
    player = df['rank'].astype(str) + '/' + df['player_id'].astype(str)
    counts = player.value_counts(sort=False)
    order = counts.index.to_numpy().copy()  # to_numpy 和 index 共用記憶體，原地 shuffle 會打亂 counts
    np.random.default_rng(SEED).shuffle(order)
    cum = counts.loc[order].cumsum().to_numpy()
    val_players = set(order[:int(np.searchsorted(cum, val_size)) + 1])
    is_val = player.isin(val_players).to_numpy()
    return df[is_val], df[~is_val]


def extract_worker(rows, outdir, tag, wid, per_shard):
    """抽一份切片的特徵。分片依輸入列切，解析失敗的局只會讓該分片少幾筆。"""
    from utils import SGFParseRankPrediction  # spawn 子進程才載入 sgfmill

    parser = SGFParseRankPrediction()
    written = bad = skipped = 0
    t0 = time.time()
    n_shards = (len(rows) + per_shard - 1) // per_shard
    for n in range(n_shards):
        path = os.path.join(outdir, '%s-w%d-%06d.npz' % (tag, wid, n))
        chunk = rows[n * per_shard:(n + 1) * per_shard]
        if os.path.exists(path):
            skipped += len(chunk)
            continue
        out = []
        for rank, game_id, player_id, color, sgf_content in chunk:
            feat, err = parser.features_from_sgf_content(sgf_content)
            if err is not None:
                bad += 1
                continue
            out.append({
                'features': packed.pack(feat),
                'rank_idx': RANK_TO_ID[rank],
                'color': COLOR_TO_ID[color],
                # game_id 只在單一等級內唯一，key 加等級前綴避免撞號
                'key': '%s_%s' % (rank, game_id),
                'player_id': player_id,
            })
        packed.save_shard(path, out)
        written += len(out)
        if (n + 1) % 5 == 0 or n + 1 == n_shards:
            done = written + bad
            eta = (time.time() - t0) / done * (len(rows) - skipped - done) if done else 0
            print('  [%s w%d] 分片 %d/%d，剩約 %.0f 分'
                  % (tag, wid, n + 1, n_shards, eta / 60), flush=True)
    return written, bad, skipped


def extract_split(df_part, out_root, tag, nproc, per_shard):
    outdir = os.path.join(out_root, tag)
    os.makedirs(outdir, exist_ok=True)
    bad_color = set(df_part['color'].astype(str)) - set(COLOR_TO_ID)
    if bad_color:
        raise SystemExit('color 欄位有非預期的值：%s' % sorted(bad_color))
    rows = list(zip(df_part['rank'].astype(str), df_part['game_id'].astype(str),
                    df_part['player_id'].astype(str), df_part['color'].astype(str),
                    df_part['sgf_content']))
    jobs = [(rows[i::nproc], outdir, tag, i, per_shard) for i in range(nproc)]
    t0 = time.time()
    with Pool(nproc) as pool:
        results = pool.starmap(extract_worker, jobs)
    written, bad, skipped = (sum(r[k] for r in results) for k in range(3))
    print('%s：新寫入 %d 筆、解析失敗 %d、沿用既有分片 %d 筆，%.1f 分'
          % (tag, written, bad, skipped, (time.time() - t0) / 60), flush=True)


def check_config(out_root, config):
    """同一個輸出目錄只能用同一組參數，否則分片的切法對不上。"""
    path = os.path.join(out_root, 'extract_config.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            old = json.load(f)
        if old != config:
            raise SystemExit('%s 的參數與這次不同：\n  舊 %s\n  新 %s\n請換輸出目錄或刪除舊目錄'
                             % (out_root, old, config))
    else:
        os.makedirs(out_root, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train-dir', default='training')
    ap.add_argument('--out', default='rank-features')
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--per-shard', type=int, default=5000)
    ap.add_argument('--frac', type=float, default=1.0, help='只取打亂後的前一部分，用來測速度')
    ap.add_argument('--val-by', choices=('game', 'player'), default='player')
    ap.add_argument('--val-size', type=int, default=VAL_SIZE,
                    help='val 的局數，預設 %d；以玩家切分時較大的 val 包含較小 val 的全部玩家' % VAL_SIZE)
    args = ap.parse_args()
    if args.val_size <= 0:
        ap.error('--val-size 要是正整數')

    config = {'format': FORMAT, 'frac': args.frac, 'val_by': args.val_by, 'nproc': args.nproc,
              'per_shard': args.per_shard, 'seed': SEED, 'val_size': args.val_size}
    check_config(args.out, config)

    df = load_dataframe(args.train_dir)
    df = df.iloc[:int(len(df) * args.frac)]
    val_size = int(args.val_size * args.frac)
    split = split_by_player if args.val_by == 'player' else split_by_game
    df_val, df_train = split(df, val_size)
    print('共 %d 局：train %d / val %d，%d 個進程'
          % (len(df), len(df_train), len(df_val), args.nproc), flush=True)

    t0 = time.time()
    extract_split(df_val, args.out, 'val', args.nproc, args.per_shard)
    extract_split(df_train, args.out, 'train', args.nproc, args.per_shard)
    print('完成，總耗時 %.1f 分' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
