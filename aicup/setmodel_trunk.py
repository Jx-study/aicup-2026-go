"""humanv0 主幹向量抽取：每局目標玩家的 30 手各一個 768 維向量，給 set 模型探針使用

向量是 act_trunkfinal 之後的主幹輸出（384 通道）在 19x19 棋盤上每通道的平均與最大值，串成 768 維，存 float16。
模型看得到前幾手（history），metadata 固定 rank_1d，只用原方向（對稱 0）。
每局取目標玩家在 SGF 手數 FIRST_MOVE～LAST_MOVE（含，從 1 起算）之間的手，從中等距取 PER_GAME 手，
範圍內不足時全取，範圍內沒有任何手的局不產生向量。

兩種局集合：
    train  split_by_player(df, SPLIT_VAL_SIZE) 的 train 玩家中局數 >= 20 者，每級 --players-per-rank 位，每人抽 20 局
    eval   --keys-from 指定的 jsonl 的局（開發組 20 局）

輸出目錄每 --chunk 局一個 part-XXXXXX.npz，重跑同一指令會跳過已完成的 part；timing.json 記錄起訖時間。
每個 part 的欄位：key、player_id、rank_idx、count（每局向量數）、vec（所有局串接，(sum(count), 768) float16）。

用法（在 repo 根目錄，需要 KataGo python/ 原始碼，見 --katago-python）：
    python -m aicup.setmodel_trunk bench --set train --positions 1000
    python -m aicup.setmodel_trunk extract --set train --out humansl-analysis/setmodel/trunk-train
    python -m aicup.setmodel_trunk extract --set eval --keys-from humansl-analysis/val-v180000-dev-p3.jsonl --out humansl-analysis/setmodel/trunk-eval
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from aicup.analyze_humansl import parse_game, sample_listed_players, target_turns
from aicup.extract_features import SEED, load_dataframe, split_by_player
from aicup.scoring import LEVELS

FIRST_MOVE, LAST_MOVE, PER_GAME = 31, 180, 30
PROFILE = 'rank_1d'
N_CHANNELS = 384
DIM = 2 * N_CHANNELS
MIN_GAMES, GAMES_PER_PLAYER = 20, 20
PLAYERS_PER_RANK = 200
SPLIT_VAL_SIZE = 180_000  # 開發組與封存組都在這個切分的 val 內，train 玩家不能和它們重疊
EVAL_PLAYERS = 986
BENCH_PLAYERS_PER_RANK = (200, 150, 100, 50)
SELECT_SEED = [42, 2000]


def pick_turns(turns, first=FIRST_MOVE, last=LAST_MOVE, n=PER_GAME):
    """turns 是 target_turns 的回合（回合 t 是 SGF 的第 t + 1 手）。範圍內等距取 n 手，不足 n 手全取，沒有則回傳空 list。"""
    inside = [t for t in turns if first <= t + 1 <= last]
    if len(inside) <= n:
        return inside
    return [inside[i] for i in np.round(np.linspace(0, len(inside) - 1, n)).astype(int)]


def select_train_players(df_train, players_per_rank=PLAYERS_PER_RANK, min_games=MIN_GAMES, seed=SELECT_SEED):
    """每個等級抽 players_per_rank 位局數 >= min_games 的玩家，DataFrame：player_id、rank。

    每級依 player_id 排序後以同一個 rng 依 LEVELS 順序洗牌，取前 players_per_rank 位，
    所以縮減人數時取同一份名單的前綴，不會換人。局數不足的等級以 SystemExit 拒絕。
    """
    rng = np.random.default_rng(seed)
    counts = df_train.groupby(['rank', 'player_id']).size()
    counts = counts[counts >= min_games]
    picked = []
    for rank in LEVELS:
        ids = sorted(counts.xs(rank, level='rank').index.astype(str)) if rank in counts.index.get_level_values('rank') else []
        if len(ids) < players_per_rank:
            raise SystemExit('等級 %s 局數 >= %d 的玩家只有 %d 位，不足 %d' % (rank, min_games, len(ids), players_per_rank))
        order = rng.permutation(len(ids))[:players_per_rank]
        picked += [(ids[i], rank) for i in order]
    return pd.DataFrame(picked, columns=['player_id', 'rank'])


def pool_trunk(out):
    """主幹輸出 (B, C, H, W) 在整張棋盤上取每通道平均與最大值，(B, 2C) float32。19x19 沒有棋盤外的位置，不需要 mask。"""
    import torch
    out = out.float()
    return torch.cat([out.mean(dim=(2, 3)), out.amax(dim=(2, 3))], dim=1)


def to_game(row):
    return {'key': '%s_%s' % (row['rank'], row['game_id']), 'rank_idx': LEVELS.index(row['rank']),
            'player_id': str(row['player_id']), 'color': row['color'], 'sgf': row['sgf_content']}


def train_games(train_dir, players_per_rank):
    """train 集合：每局一個 dict（key、rank_idx、player_id、color、sgf）。"""
    df = load_dataframe(train_dir)
    df['player_id'] = df['player_id'].astype(str)
    _, df_train = split_by_player(df, SPLIT_VAL_SIZE)
    players = select_train_players(df_train, players_per_rank)
    sub = sample_listed_players(df_train, players, GAMES_PER_PLAYER, SEED)
    return [to_game(r) for _, r in sub.iterrows()]


def eval_games(keys_from, train_dir):
    """eval 集合：keys_from 每行的局，順序相同，從 train_dir 的 CSV 取棋譜。"""
    from aicup.strength_se import load_keyed_games
    return load_keyed_games(keys_from, os.path.join(train_dir, 'train_*.csv'))


def encode_one(game):
    """worker：一局的 (key, spatial, global)，沒有可用的手或編碼失敗時 spatial 為 None。"""
    from aicup.katago_torch import encode_game
    try:
        parsed = parse_game(game['sgf'])
        turns = pick_turns(target_turns(parsed['moves'], game['color']))
        if not turns:
            return game['key'], None, None
        enc = encode_game(parsed, turns, history=True)
        return game['key'], enc['spatial'], enc['global']
    except Exception:  # 壞 SGF 或不支援的局，該局沒有向量
        return game['key'], None, None


def embed(model, meta_row, spatial, glob_, batch, dtype):
    """輸入編碼 spatial (n, 22, 19, 19) uint8 與 glob_ (n, 19)，回傳 act_trunkfinal 輸出的平均加最大值池化向量，(n, 768) float16。"""
    import torch
    from model_pytorch import ExtraOutputs
    device = next(model.parameters()).device
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype, enabled=device.type == 'cuda' and dtype != torch.float32):
        for s in range(0, len(spatial), batch):
            sp = torch.from_numpy(spatial[s:s + batch]).to(device).float()
            gl = torch.from_numpy(glob_[s:s + batch]).to(device)
            extra = ExtraOutputs(['trunkfinal'])
            model(sp, gl, input_meta=meta_row.expand(len(gl), -1), extra_outputs=extra)
            vec = pool_trunk(extra.returned['trunkfinal']).half().cpu().numpy()
            if not np.isfinite(vec).all():
                raise ValueError('主幹向量轉 float16 後出現 inf 或 nan')
            out.append(vec)
    return np.concatenate(out) if out else np.zeros((0, DIM), np.float16)


def part_path(out_dir, n):
    return os.path.join(out_dir, 'part-%06d.npz' % n)


def write_part(path, games, results, vecs):
    """results 是和 games 同序的 (key, spatial, global)，vecs 是這個 part 全部局向量依序串接的結果。"""
    counts = np.array([0 if r[1] is None else len(r[1]) for r in results])
    tmp = os.path.join(os.path.dirname(path), 'tmp-' + os.path.basename(path))
    np.savez_compressed(tmp, key=np.array([g['key'] for g in games]),
                        player_id=np.array([g['player_id'] for g in games]),
                        rank_idx=np.array([g['rank_idx'] for g in games]), count=counts, vec=vecs)
    os.replace(tmp, path)


def load_vectors(out_dir):
    """讀 extract 的輸出，dict：key -> (count, 768) float16，沒有向量的局不在內。"""
    vectors = {}
    for path in sorted(glob.glob(os.path.join(out_dir, 'part-*.npz'))):
        with np.load(path) as z:
            vec, count = z['vec'], z['count']
            ends = np.cumsum(count)
            for key, end, c in zip(z['key'], ends, count):
                if c:
                    vectors[str(key)] = vec[end - c:end]
    return vectors


def setup_model(args):
    import torch
    from aicup import katago_torch as kt
    if args.katago_python:
        os.environ['KATAGO_PYTHON'] = args.katago_python
    kt.katago()  # 讓 model_pytorch 可以 import
    kt.ensure_humanv0(args.checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = kt.load_humanv0(args.checkpoint, device=device)
    meta_row = torch.from_numpy(kt.profile_row(PROFILE)).to(device)
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    return model, meta_row, dtype


def get_games(args):
    if args.set == 'train':
        return train_games(args.train_dir, args.players_per_rank)
    if not args.keys_from:
        raise SystemExit('--set eval 要給 --keys-from')
    return eval_games(args.keys_from, args.train_dir)


def check_config(args):
    """輸出目錄的 extract_config.json 和這次的參數不同時拒絕，沒有則寫入。part 編號只在同一組參數下對得到同一批局。"""
    config = {'set': args.set, 'players_per_rank': args.players_per_rank if args.set == 'train' else None,
              'chunk': args.chunk, 'keys_from': os.path.basename(args.keys_from) if args.keys_from else None}
    path = os.path.join(args.out, 'extract_config.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            old = json.load(f)
        if old != config:
            raise SystemExit('%s 的參數和這次不同：%s，這次 %s' % (args.out, old, config))
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f)


def extract(args):
    games = get_games(args)
    os.makedirs(args.out, exist_ok=True)
    check_config(args)
    chunks = [games[i:i + args.chunk] for i in range(0, len(games), args.chunk)]
    todo = [n for n in range(len(chunks)) if not os.path.exists(part_path(args.out, n))]
    print('%d 局，%d 個 part，待做 %d 個' % (len(games), len(chunks), len(todo)), flush=True)
    if not todo:
        return
    model, meta_row, dtype = setup_model(args)
    t0 = time.time()
    timing = os.path.join(args.out, 'timing.json')
    log = {'runs': []}
    if os.path.exists(timing):
        with open(timing, encoding='utf-8') as f:
            log = json.load(f)
    log['runs'].append({'start': time.strftime('%Y-%m-%d %H:%M:%S'), 'players_per_rank': args.players_per_rank,
                        'set': args.set, 'dtype': args.dtype})
    positions = 0
    with ProcessPoolExecutor(args.workers) as pool:
        def submit(n):
            return [pool.submit(encode_one, g) for g in chunks[n]]
        pending = submit(todo[0])
        for i, n in enumerate(todo):
            futures = pending
            pending = submit(todo[i + 1]) if i + 1 < len(todo) else None
            results = [f.result() for f in futures]
            ok = [r for r in results if r[1] is not None]
            if ok:
                vecs = embed(model, meta_row, np.concatenate([r[1] for r in ok]), np.concatenate([r[2] for r in ok]),
                             args.batch, dtype)
            else:
                vecs = np.zeros((0, DIM), np.float16)
            write_part(part_path(args.out, n), chunks[n], results, vecs)
            positions += len(vecs)
            el = time.time() - t0
            print('part %d/%d 完成，%d 局無向量，累計 %d 個局面，%.0f 秒，%.1f 個/秒'
                  % (i + 1, len(todo), len(results) - len(ok), positions, el, positions / el), flush=True)
    log['runs'][-1].update(end=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=round(time.time() - t0, 1),
                           positions=positions)
    with open(timing, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=1)


def bench(args):
    """用前 args.positions 個局面分別量編碼與主幹前向的速度，換算整份 train 與 eval 要多久。"""
    import torch
    games = get_games(args)
    model, meta_row, dtype = setup_model(args)
    n = 0
    with ProcessPoolExecutor(args.workers) as pool:
        t0 = time.time()
        results = []
        for g in games:
            if n >= args.positions:
                break
            results.append(pool.submit(encode_one, g))
            n += PER_GAME
        encoded = [r.result() for r in results]
        enc_s = time.time() - t0
    ok = [r for r in encoded if r[1] is not None]
    if not ok:
        raise SystemExit('前 %d 局都沒有可用的局面' % len(encoded))
    spatial = np.concatenate([r[1] for r in ok])[:args.positions]
    glob_ = np.concatenate([r[2] for r in ok])[:args.positions]
    embed(model, meta_row, spatial[:64], glob_[:64], args.batch, dtype)  # 預熱
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    embed(model, meta_row, spatial, glob_, args.batch, dtype)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_s = time.time() - t0
    n_pos = len(spatial)
    enc_rate = sum(len(r[1]) for r in ok) / enc_s
    fwd_rate = n_pos / fwd_s
    rate = min(enc_rate, fwd_rate)  # 編碼與前向重疊執行，取慢的那個
    print('%d 個局面：編碼 %.1f 個/秒（%d workers），前向 %.1f 個/秒，重疊後約 %.1f 個/秒'
          % (n_pos, enc_rate, args.workers, fwd_rate, rate))
    per_player = GAMES_PER_PLAYER * PER_GAME
    eval_pos = EVAL_PLAYERS * per_player
    print('eval %d 位約 %d 個局面，%.2f 小時' % (EVAL_PLAYERS, eval_pos, eval_pos / rate / 3600))
    for ppr in BENCH_PLAYERS_PER_RANK:
        pos = len(LEVELS) * ppr * per_player
        print('train 每級 %d 位約 %d 個局面，%.2f 小時，加 eval 共 %.2f 小時'
              % (ppr, pos, pos / rate / 3600, (pos + eval_pos) / rate / 3600))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=['bench', 'extract'])
    ap.add_argument('--set', choices=['train', 'eval'], required=True)
    ap.add_argument('--train-dir', default='training-v1.1/training', help='含 train_*.csv 的目錄')
    ap.add_argument('--keys-from', help='--set eval 用的 jsonl')
    ap.add_argument('--players-per-rank', type=int, default=PLAYERS_PER_RANK)
    ap.add_argument('--out', help='extract 的輸出目錄')
    ap.add_argument('--katago-python', help='KataGo 原始碼的 python/ 目錄，缺省時讀環境變數 KATAGO_PYTHON')
    ap.add_argument('--checkpoint')
    ap.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=200, help='每個 part 的局數')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--positions', type=int, default=1000, help='bench 用的局面數')
    args = ap.parse_args()
    from aicup import katago_torch as kt
    args.checkpoint = args.checkpoint or kt.CHECKPOINT
    if args.mode == 'extract' and not args.out:
        ap.error('extract 要給 --out')
    if args.mode == 'bench':
        bench(args)
    else:
        extract(args)


if __name__ == '__main__':
    main()
