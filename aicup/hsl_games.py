"""從官方訓練 CSV 抽出 Human SL 分析要跑的局，寫成 analyze_humansl --csv 讀的 <out-root>/hist6d-<run>/games.csv。

每位玩家抽 20 局，抽樣同 analyze_humansl.sample_listed_players（seed 42）。
groupby.sample 的結果取決於整份抽樣清單，newdev 要對整份新增玩家清單抽樣後，再留 newdev 的玩家。
games.csv 的列順序就是分析順序。

可用的 run（玩家清單由 aicup.val_players 產生）：
    trainpool  train-pool-v180000.csv
    val648     val-players.csv
    newdev     抽樣用 val-players-new-v180000.csv，只留 val-players-newdev-v180000.csv

用法（在 repo 根目錄）：
    python -m aicup.hsl_games --run trainpool val648 newdev
"""
import argparse
import glob
import os

import pandas as pd

from aicup.analyze_humansl import sample_listed_players

SEED = 42
PER_PLAYER = 20
GAME_COLS = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']
RUNS = {  # run: (抽樣清單, 保留清單)
    'trainpool': ('train-pool-v180000.csv', 'train-pool-v180000.csv'),
    'val648': ('val-players.csv', 'val-players.csv'),
    'newdev': ('val-players-new-v180000.csv', 'val-players-newdev-v180000.csv'),
}


def load_train(train_dir):
    """依檔名順序讀 train_*.csv；player_id 保留 read_csv 推斷的型別，和 analyze_humansl.load_games 一致。"""
    paths = sorted(glob.glob(os.path.join(train_dir, 'train_*.csv')))
    if not paths:
        raise SystemExit('%s 下找不到 train_*.csv' % train_dir)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def read_players(path):
    return pd.read_csv(path, dtype=str)


def select_games(df, sample_players, keep_players):
    """對 sample_players 每人抽 PER_PLAYER 局，只留 keep_players 內 (player_id, rank) 的局，回傳 GAME_COLS 欄位。"""
    rows = sample_listed_players(df, sample_players, PER_PLAYER, SEED)
    keep = set(zip(keep_players['player_id'], keep_players['rank']))
    mask = [(str(p), str(r)) in keep for p, r in zip(rows['player_id'], rows['rank'])]
    return rows[mask][GAME_COLS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', nargs='+', choices=sorted(RUNS), required=True)
    ap.add_argument('--train-dir', default='training-v1.1/training')
    ap.add_argument('--players-dir', default='humansl-analysis', help='玩家清單所在目錄')
    ap.add_argument('--out-root', default='humansl-analysis', help='輸出到 <out-root>/hist6d-<run>/games.csv')
    args = ap.parse_args()

    outs = {run: os.path.join(args.out_root, 'hist6d-' + run, 'games.csv') for run in args.run}
    existing = [p for p in outs.values() if os.path.exists(p)]
    if existing:
        raise SystemExit('以下檔案已存在，不覆寫：%s' % ', '.join(existing))
    df = load_train(args.train_dir)
    for run in args.run:
        sample_file, keep_file = RUNS[run]
        games = select_games(df, read_players(os.path.join(args.players_dir, sample_file)),
                             read_players(os.path.join(args.players_dir, keep_file)))
        os.makedirs(os.path.dirname(outs[run]), exist_ok=True)
        games.to_csv(outs[run], index=False, lineterminator='\n')
        print('%-10s %6d 局 -> %s' % (run, len(games), outs[run]))


if __name__ == '__main__':
    main()
