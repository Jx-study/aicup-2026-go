"""從官方訓練 CSV 抽出 Human SL 分析要跑的局，寫成 analyze_humansl --csv 讀的 games.csv。

6d profile 的 run 寫到 <out-root>/hist6d-<run>/games.csv，2d profile 的 run（--2d）寫到 <out-root>/hist2d-<run>/games.csv。
games.csv 的列順序就是分析順序，目標側與對手側用同一份。

依玩家清單抽局的 run 每人抽 20 局，抽樣同 analyze_humansl.sample_listed_players（seed 42）。
groupby.sample 的結果取決於整份抽樣清單，newdev、holdout 要對整份新增玩家清單抽樣後，再留各自的玩家。

6d 的 run（玩家清單由 aicup.val_players 產生）：
    trainpool  train-pool-v180000.csv
    val648     val-players.csv
    newdev     抽樣用 val-players-new-v180000.csv，只留 val-players-newdev-v180000.csv
2d 的 run：
    val648     同 6d
    train_p1   train-p1-v180000.csv，train_p2、train_p3 同理
    holdout    抽樣用 val-players-new-v180000.csv，只留 holdout-v180000.csv
    train_p4   train-p4.csv 每人取 HD < 2 局中多數色（同數取黑）的 10 局
    train_p4b  train_p4 的玩家中，和 train_p4 同色、HD < 2、未抽過的局至少 10 局者再抽 10 局
2d 的測試題不需要 games.csv，分析時直接讀測試題目錄。

用法（在 repo 根目錄）：
    python -m aicup.hsl_games --run trainpool val648 newdev
    python -m aicup.hsl_games --2d --run val648 train_p1 train_p2 train_p3 holdout train_p4 train_p4b
"""
import argparse
import glob
import os

import pandas as pd

from aicup.analyze_humansl import sample_listed_players
from aicup.val_players import hd_of

SEED = 42
PER_PLAYER = 20
SAME_COLOR = 10
GAME_COLS = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']
RUNS = {  # run: (抽樣清單, 保留清單)
    'trainpool': ('train-pool-v180000.csv', 'train-pool-v180000.csv'),
    'val648': ('val-players.csv', 'val-players.csv'),
    'newdev': ('val-players-new-v180000.csv', 'val-players-newdev-v180000.csv'),
}
RUNS_2D = {  # run: (抽樣清單, 保留清單)；None 表示另有抽法
    'val648': RUNS['val648'],
    'train_p1': ('train-p1-v180000.csv', 'train-p1-v180000.csv'),
    'train_p2': ('train-p2-v180000.csv', 'train-p2-v180000.csv'),
    'train_p3': ('train-p3-v180000.csv', 'train-p3-v180000.csv'),
    'holdout': ('val-players-new-v180000.csv', 'holdout-v180000.csv'),
    'train_p4': None,
    'train_p4b': None,
}
P4_PLAYERS = 'train-p4.csv'


def load_train(train_dir):
    """依檔名順序讀 train_*.csv；player_id 保留 read_csv 推斷的型別，和 analyze_humansl.load_games 一致。"""
    paths = sorted(glob.glob(os.path.join(train_dir, 'train_*.csv')))
    if not paths:
        raise SystemExit('%s 下找不到 train_*.csv' % train_dir)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df['key'] = df['rank'].astype(str) + '_' + df['game_id'].astype(str)
    df['hd'] = df['sgf_content'].map(hd_of)
    return df


def read_players(path):
    return pd.read_csv(path, dtype=str)


def select_games(df, sample_players, keep_players):
    """對 sample_players 每人抽 PER_PLAYER 局，只留 keep_players 內 (player_id, rank) 的局，回傳 GAME_COLS 欄位。"""
    rows = sample_listed_players(df, sample_players, PER_PLAYER, SEED)
    keep = set(zip(keep_players['player_id'], keep_players['rank']))
    mask = [(str(p), str(r)) in keep for p, r in zip(rows['player_id'], rows['rank'])]
    return rows[mask][GAME_COLS]


def _pairs(d):
    return list(zip(d['player_id'].astype(str), d['rank'].astype(str)))


def same_color_games(df, players, k=SAME_COLOR):
    """players 每人取 HD < 2 局中多數色（同數取黑）的 k 局，依 (player_id, rank) 排序輸出；df 要有 key、hd 欄。"""
    want = set(_pairs(players))
    d = df[[pk in want for pk in _pairs(df)]]
    d = d[d['hd'] < 2].sort_values('key')
    parts = []
    for pk, g in d.groupby(['player_id', 'rank'], sort=True):
        nb, nw = (g['color'] == 'B').sum(), (g['color'] == 'W').sum()
        side = g[g['color'] == ('B' if nb >= nw else 'W')]
        if len(side) < k:
            raise SystemExit('%s 的多數色只有 %d 局' % (pk, len(side)))
        parts.append(side.sample(n=k, random_state=SEED))
    if len(parts) != len(want):
        raise SystemExit('清單 %d 位，找到 %d 位' % (len(want), len(parts)))
    return pd.concat(parts, ignore_index=True)[GAME_COLS]


def topup_games(df, prev, k=SAME_COLOR):
    """prev 的每位玩家取和 prev 同色、HD < 2、不在 prev 的局，剩至少 k 局的玩家抽 k 局，依 (player_id, rank) 排序輸出。"""
    prev = prev.astype({'player_id': str, 'rank': str})
    colors = prev.groupby(['player_id', 'rank'])['color']
    if (colors.nunique() != 1).any():
        raise SystemExit('prev 有玩家用了兩種顏色')
    want = dict(colors.first().items())
    used = set(prev['rank'] + '_' + prev['game_id'].astype(str))
    d = df[~df['key'].isin(used) & (df['hd'] < 2)].copy()
    d['pk'] = _pairs(d)
    d = d[[want.get(pk) == c for pk, c in zip(d['pk'], d['color'])]].sort_values('key')
    ok = {pk for pk, n in d.groupby('pk').size().items() if n >= k}
    if not ok:
        raise SystemExit('沒有玩家剩 %d 局以上可補抽' % k)
    d = d[d['pk'].isin(ok)]
    return pd.concat([g.sample(n=k, random_state=SEED) for _, g in d.groupby('pk', sort=True)],
                     ignore_index=True)[GAME_COLS]


def run_games(run, df, players_dir, two_d, cache):
    """回傳 run 的 GAME_COLS 棋譜列；cache 存同一次執行已算出的 run，train_p4b 從 train_p4 的局補抽。"""
    if run in cache:
        return cache[run]
    if two_d and run == 'train_p4':
        games = same_color_games(df, read_players(os.path.join(players_dir, P4_PLAYERS)))
    elif two_d and run == 'train_p4b':
        games = topup_games(df, run_games('train_p4', df, players_dir, two_d, cache))
    else:
        sample_file, keep_file = (RUNS_2D if two_d else RUNS)[run]
        games = select_games(df, read_players(os.path.join(players_dir, sample_file)),
                             read_players(os.path.join(players_dir, keep_file)))
    cache[run] = games
    return games


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', nargs='+', required=True, help='6d：%s；--2d：%s' % (' '.join(RUNS), ' '.join(RUNS_2D)))
    ap.add_argument('--2d', dest='two_d', action='store_true', help='2d profile 的 run，寫到 hist2d-<run>/')
    ap.add_argument('--train-dir', default='training-v1.1/training')
    ap.add_argument('--players-dir', default='humansl-analysis', help='玩家清單所在目錄')
    ap.add_argument('--out-root', default='humansl-analysis', help='輸出到 <out-root>/hist6d-<run>/ 或 hist2d-<run>/')
    args = ap.parse_args()

    runs = RUNS_2D if args.two_d else RUNS
    bad = [r for r in args.run if r not in runs]
    if bad:
        ap.error('不認得的 run：%s，可用 %s' % (' '.join(bad), ' '.join(runs)))
    prefix = 'hist2d-' if args.two_d else 'hist6d-'
    outs = {run: os.path.join(args.out_root, prefix + run, 'games.csv') for run in args.run}
    existing = [p for p in outs.values() if os.path.exists(p)]
    if existing:
        raise SystemExit('以下檔案已存在，不覆寫：%s' % ', '.join(existing))
    df = load_train(args.train_dir)
    cache = {}
    for run in args.run:
        games = run_games(run, df, args.players_dir, args.two_d, cache)
        os.makedirs(os.path.dirname(outs[run]), exist_ok=True)
        games.to_csv(outs[run], index=False, lineterminator='\n')
        print('%-10s %6d 局 -> %s' % (run, len(games), outs[run]))


if __name__ == '__main__':
    main()
