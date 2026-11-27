"""從官方訓練 CSV 產生玩家切分清單，寫成 analyze_humansl --players-file 的 CSV（欄位 player_id、rank）。

玩家由 rank 與 player_id 一起區分，切分沿用 extract_features.split_by_player。
寫出的清單（val_size 為 --val-size）：
    val-players.csv                 split_by_player(df, --base-size) 的 val 中局數至少 --min-games 的玩家，依 player_id 排序
    val-players-v<val_size>.csv     split_by_player(df, --val-size) 的 val 中同樣條件的玩家，依 rank、player_id 排序
    val-players-new-v<val_size>.csv 上一份減去 val-players.csv
    holdout-v<val_size>.csv         從新增玩家依等級比例抽 --n-holdout 位，作為封存組
    val-players-newdev-v<val_size>.csv  新增玩家扣掉封存組
    val-players-dev-v<val_size>.csv     val-players.csv 加上 newdev
    train-pool-v<val_size>.csv      split_by_player(df, --val-size) 的 train 中局數至少 --min-games 的玩家，每級抽 --pool-per-level 位
以下 4 份不切分，從全部訓練局選，供 2d profile 的 Human SL 分析：
    train-p1-v<val_size>.csv        排除 dev、封存組、train 池、val-players.csv 的 player_id，每級依 player_id 排序取第 1～100 位
    train-p2-v<val_size>.csv        同上的第 101～200 位
    train-p3-v<val_size>.csv        排除 val-players-v<val_size>、封存組、train-p1、train-p2，每級用 random.Random(44) 抽 100 位
    train-p4.csv                    排除以上清單與基準切分 val 中局數至少 5 的 player_id，HD < 2 的局中同一色至少 10 局的玩家
另外印出新增玩家與基準玩家等級分布的卡方同質性檢定。

用法（在 repo 根目錄）：
    python -m aicup.val_players --train-dir training-v1.1/training --val-size 180000 --out-dir humansl-analysis
"""
import argparse
import csv
import os
import random
import re

import numpy as np
import pandas as pd

from aicup.extract_features import SEED, VAL_SIZE, load_dataframe, split_by_player
from aicup.scoring import LEVELS

MIN_GAMES = 20
N_HOLDOUT = 160
POOL_PER_LEVEL = 100
PART_PER_LEVEL = 100
P3_SEED = 44
SAME_COLOR_GAMES = 10
WIDE_MIN_GAMES = 5  # train-p4 另外排除基準切分 val 中局數至少 5 的玩家


def eligible_players(df, val_size, min_games=MIN_GAMES):
    """split_by_player(df, val_size) 的 val 中局數至少 min_games 的玩家，DataFrame：player_id、rank，依 rank、player_id 排序。"""
    val, _ = split_by_player(df, val_size)
    counts = val.groupby(['rank', 'player_id']).size()
    kept = counts[counts >= min_games].reset_index()[['player_id', 'rank']]
    kept['player_id'] = kept['player_id'].astype(str)
    return kept.sort_values(['rank', 'player_id']).reset_index(drop=True)


def player_set(players):
    return set(zip(players['rank'].astype(str), players['player_id'].astype(str)))


def new_players(df, base_size, val_size, min_games=MIN_GAMES):
    """回傳 (基準清單, 全部清單, 新增清單)；val_size 不大於 base_size 或全部清單不含基準清單時丟 ValueError。"""
    if val_size <= base_size:
        raise ValueError('--val-size 要大於 --base-size')
    base = eligible_players(df, base_size, min_games)
    full = eligible_players(df, val_size, min_games)
    if not player_set(base) <= player_set(full):
        raise ValueError('全部清單沒有包含基準清單')
    base_keys = player_set(base)
    is_new = [k not in base_keys for k in zip(full['rank'], full['player_id'])]
    return base, full, full[is_new].reset_index(drop=True)


def quotas(counts, total):
    """最大餘數法把 total 依 counts 比例分配，小數部分相同時 index 小的優先。"""
    exact = [c * total / sum(counts) for c in counts]
    base = [int(e) for e in exact]
    rest = total - sum(base)
    order = sorted(range(len(counts)), key=lambda i: (-(exact[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return base


def holdout_split(new, base, n_holdout=N_HOLDOUT):
    """從新增清單依等級比例抽封存組，回傳 (封存, newdev, dev)。

    每級內對 player_id 排序後依 LEVELS 順序抽，newdev 保留新增清單的順序，dev 是基準清單接 newdev。
    """
    by_level = {lv: sorted(new.loc[new['rank'] == lv, 'player_id']) for lv in LEVELS}
    quota = quotas([len(by_level[lv]) for lv in LEVELS], n_holdout)
    rng = np.random.default_rng([SEED, n_holdout])
    hold = []
    for lv, q in zip(LEVELS, quota):
        pool = by_level[lv]
        hold += [(pool[i], lv) for i in sorted(rng.choice(len(pool), q, replace=False))]
    hold = pd.DataFrame(hold, columns=['player_id', 'rank'])
    newdev = new[~new['player_id'].isin(set(hold['player_id']))].reset_index(drop=True)
    dev = pd.concat([base, newdev], ignore_index=True)
    ids = [set(x['player_id']) for x in (hold, newdev, base)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError('封存、newdev、基準清單有交集')
    return hold, newdev, dev


def train_pool(df, val_size, per_level=POOL_PER_LEVEL, min_games=MIN_GAMES):
    """split_by_player(df, val_size) 的 train 中局數至少 min_games 的玩家，每級內對 player_id 排序後抽 per_level 位。"""
    _, train = split_by_player(df, val_size)
    counts = train.groupby(['rank', 'player_id']).size()
    kept = counts[counts >= min_games].reset_index()
    kept['player_id'] = kept['player_id'].astype(str)
    rng = np.random.default_rng([SEED, per_level * len(LEVELS)])
    rows = []
    for lv in LEVELS:
        pool = sorted(kept.loc[kept['rank'] == lv, 'player_id'])
        if len(pool) < per_level:
            raise ValueError('%s 只有 %d 位' % (lv, len(pool)))
        rows += [(pool[i], lv) for i in sorted(rng.choice(len(pool), per_level, replace=False))]
    return pd.DataFrame(rows, columns=['player_id', 'rank'])


def game_counts(df):
    """全部訓練局中每位玩家的局數，Series，index 為 (rank, player_id 字串)。"""
    return df.groupby([df['rank'].astype(str), df['player_id'].astype(str)]).size()


def first_parts(df, exclude_ids, per_level=PART_PER_LEVEL, n_parts=2, min_games=MIN_GAMES):
    """全部訓練局中局數至少 min_games、player_id 不在 exclude_ids 的玩家，每級依 player_id 排序取前 per_level × n_parts 位，依序切成 n_parts 份。"""
    counts = game_counts(df)
    parts = [[] for _ in range(n_parts)]
    for lv in LEVELS:
        pool = sorted(p for (r, p), n in counts.items() if r == lv and n >= min_games and p not in exclude_ids)
        if len(pool) < per_level * n_parts:
            raise ValueError('%s 只有 %d 位' % (lv, len(pool)))
        for i in range(n_parts):
            parts[i] += [(p, lv) for p in pool[i * per_level:(i + 1) * per_level]]
    return [pd.DataFrame(rows, columns=['player_id', 'rank']) for rows in parts]


def random_part(df, exclude, seed, per_level=PART_PER_LEVEL, min_games=MIN_GAMES):
    """全部訓練局中局數至少 min_games、(rank, player_id) 不在 exclude 的玩家，每級用 random.Random(seed) 抽 per_level 位。"""
    counts = game_counts(df)
    rng = random.Random(seed)
    rows = []
    for lv in LEVELS:
        pool = sorted(p for (r, p), n in counts.items() if r == lv and n >= min_games and (r, p) not in exclude)
        rows += [(p, lv) for p in sorted(rng.sample(pool, per_level))]
    return pd.DataFrame(rows, columns=['player_id', 'rank'])


def hd_of(sgf):
    m = re.search(r'HD\[(\d+)\]', sgf)
    return int(m.group(1)) if m else 0  # 沒有 HD 標記視為互先


def same_color_players(df, exclude_ids, k=SAME_COLOR_GAMES):
    """HD < 2 的局中，同一色至少 k 局且 player_id 不在 exclude_ids 的玩家，依 rank、player_id 排序。"""
    ok = df[df['sgf_content'].map(hd_of) < 2]
    cnt = ok.groupby([ok['player_id'].astype(str), ok['rank'].astype(str), ok['color']]).size().unstack(fill_value=0)
    cnt = cnt[cnt.max(axis=1) >= k].reset_index()
    cnt = cnt[~cnt['player_id'].isin(exclude_ids)]
    return cnt[['player_id', 'rank']].sort_values(['rank', 'player_id']).reset_index(drop=True)


def ids_of(*lists):
    return set().union(*(set(x['player_id'].astype(str)) for x in lists))


def write_players(path, players):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['player_id', 'rank'])
        w.writerows(zip(players['player_id'], players['rank']))


def level_counts(players):
    return np.array([(players['rank'] == lv).sum() for lv in LEVELS])


def homogeneity(base, new):
    """等級分布的卡方同質性檢定，回傳 (卡方, p 值)。"""
    from scipy.stats import chi2_contingency
    table = np.vstack([level_counts(base), level_counts(new)])
    table = table[:, table.sum(axis=0) > 0]
    chi2, p, _, _ = chi2_contingency(table)
    return float(chi2), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train-dir', default='training-v1.1/training')
    ap.add_argument('--base-size', type=int, default=VAL_SIZE)
    ap.add_argument('--val-size', type=int, required=True)
    ap.add_argument('--min-games', type=int, default=MIN_GAMES)
    ap.add_argument('--n-holdout', type=int, default=N_HOLDOUT)
    ap.add_argument('--pool-per-level', type=int, default=POOL_PER_LEVEL)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    df = load_dataframe(args.train_dir)
    base, full, new = new_players(df, args.base_size, args.val_size, args.min_games)
    hold, newdev, dev = holdout_split(new, base.sort_values('player_id').reset_index(drop=True), args.n_holdout)
    pool = train_pool(df, args.val_size, args.pool_per_level, args.min_games)
    if player_set(pool) & player_set(full):
        raise SystemExit('train 池和 val 清單有交集')
    p1, p2 = first_parts(df, ids_of(dev, hold, pool, base), min_games=args.min_games)
    p3 = random_part(df, player_set(full) | player_set(hold) | player_set(p1) | player_set(p2), P3_SEED,
                     min_games=args.min_games)
    wide = eligible_players(df, args.base_size, WIDE_MIN_GAMES)
    p4 = same_color_players(df, ids_of(base, wide, full, pool, p1, p2, p3))

    tag = 'v%d' % args.val_size
    lists = {
        'val-players.csv': base.sort_values('player_id'),
        'val-players-%s.csv' % tag: full,
        'val-players-new-%s.csv' % tag: new,
        'holdout-%s.csv' % tag: hold,
        'val-players-newdev-%s.csv' % tag: newdev,
        'val-players-dev-%s.csv' % tag: dev,
        'train-pool-%s.csv' % tag: pool,
        'train-p1-%s.csv' % tag: p1,
        'train-p2-%s.csv' % tag: p2,
        'train-p3-%s.csv' % tag: p3,
        'train-p4.csv': p4,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {name: os.path.join(args.out_dir, name) for name in lists}
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing:
        raise SystemExit('以下檔案已存在，不覆寫：%s' % ', '.join(existing))
    for name, players in lists.items():
        write_players(paths[name], players)
        print('%-32s %5d 位' % (name, len(players)))

    chi2, p = homogeneity(base, new)
    b, n = level_counts(base), level_counts(new)
    print('| 等級 | ' + ' | '.join(LEVELS) + ' |')
    print('|---|' + '---|' * len(LEVELS))
    print('| 基準 | ' + ' | '.join(map(str, b)) + ' |')
    print('| 新增 | ' + ' | '.join(map(str, n)) + ' |')
    print('卡方同質性 %.2f，p = %.4f' % (chi2, p))


if __name__ == '__main__':
    main()
