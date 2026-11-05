"""val 切分的玩家清單：CNN 的 val_probs.npz 中局數至少 --min-games 的玩家，寫成 analyze_humansl --players-file 的 CSV。

val_probs.npz 涵蓋 val 切分的每一局，玩家由 player_id 與等級一起區分。

用法（在 repo 根目錄）：
    python -m aicup.val_players rank-trained-models/<commit hash>/val_probs.npz --min-games 5 --out humansl-analysis/val-players-798.csv
"""
import argparse

import numpy as np
import pandas as pd

from aicup.scoring import LEVELS


def val_players(path, min_games):
    """回傳 DataFrame：player_id、rank、games，只留局數至少 min_games 的玩家。

    依 rank、player_id 的字串排序（1D 排在 A 前面），不是 LEVELS 的順序。
    """
    z = np.load(path, allow_pickle=True)
    if 'player_id' not in z.files:
        raise ValueError('%s 沒有 player_id 欄位' % path)
    df = pd.DataFrame({'player_id': z['player_id'].astype(str),
                       'rank': [LEVELS[i] for i in z['rank_idx']]})
    counts = df.groupby(['rank', 'player_id']).size().rename('games').reset_index()
    kept = counts[counts['games'] >= min_games].sort_values(['rank', 'player_id'])
    return kept[['player_id', 'rank', 'games']].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('val_probs', help='CNN 的 val_probs.npz')
    ap.add_argument('--min-games', type=int, required=True, help='局數少於此值的玩家不列入')
    ap.add_argument('--out', required=True, help='輸出的 CSV，欄位是 player_id、rank')
    args = ap.parse_args()
    players = val_players(args.val_probs, args.min_games)
    players[['player_id', 'rank']].to_csv(args.out, index=False)
    print('%d 位玩家，%d 局，寫入 %s' % (len(players), players['games'].sum(), args.out))


if __name__ == '__main__':
    main()
