"""擴大 held-out 的玩家清單：從官方訓練 CSV 重新切分，寫成 analyze_humansl --players-file 的 CSV

基準清單 = split_by_player(df, --base-size) 的 val 中局數至少 --min-games 的玩家。
新增清單 = split_by_player(df, --val-size) 的 val 中局數至少 --min-games 的玩家，減去基準清單。
玩家由 rank 與 player_id 一起區分。--check 給定時，基準清單要和該 CSV 的玩家集合完全相同。
另外印出新增玩家與基準玩家等級分布的卡方同質性檢定。

用法（在 repo 根目錄）：
    python -m aicup.val_players --val-size 180000 --check humansl-analysis/val-players.csv --out-dir humansl-analysis
"""
import argparse
import os

import numpy as np
import pandas as pd

from aicup.extract_features import VAL_SIZE, load_dataframe, split_by_player
from aicup.scoring import LEVELS

MIN_GAMES = 20


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
    ap.add_argument('--train-dir', default='training')
    ap.add_argument('--base-size', type=int, default=VAL_SIZE)
    ap.add_argument('--val-size', type=int, required=True)
    ap.add_argument('--min-games', type=int, default=MIN_GAMES)
    ap.add_argument('--check', help='基準清單要相同的 CSV，例如 humansl-analysis/val-players.csv')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    df = load_dataframe(args.train_dir)
    base, full, new = new_players(df, args.base_size, args.val_size, args.min_games)
    if args.check:
        ref = pd.read_csv(args.check, dtype=str)
        if player_set(ref) != player_set(base):
            raise SystemExit('基準清單和 %s 不同：只在基準 %d 位、只在檔案 %d 位'
                             % (args.check, len(player_set(base) - player_set(ref)),
                                len(player_set(ref) - player_set(base))))
        print('基準清單和 %s 相同，%d 位' % (args.check, len(base)))
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {name: os.path.join(args.out_dir, 'val-players%s-v%d.csv' % (suffix, args.val_size))
             for name, suffix in (('full', ''), ('new', '-new'))}
    for path in paths.values():
        if os.path.exists(path):
            raise SystemExit('%s 已存在，不覆寫' % path)
    full.to_csv(paths['full'], index=False)
    new.to_csv(paths['new'], index=False)

    chi2, p = homogeneity(base, new)
    b, n, f = level_counts(base), level_counts(new), level_counts(full)
    print('基準 %d 位、新增 %d 位、全部 %d 位' % (len(base), len(new), len(full)))
    print('| 等級 | ' + ' | '.join(LEVELS) + ' |')
    print('|---|' + '---|' * len(LEVELS))
    print('| 基準 | ' + ' | '.join(map(str, b)) + ' |')
    print('| 新增 | ' + ' | '.join(map(str, n)) + ' |')
    gap = np.abs(f / f.sum() - b / b.sum())
    print('合併後與基準的等級比例最大差 %.4f（%s）' % (gap.max(), LEVELS[int(gap.argmax())]))
    print('卡方同質性 %.2f，p = %.4f' % (chi2, p))
    print('寫入 %s、%s' % (paths['full'], paths['new']))


if __name__ == '__main__':
    main()
