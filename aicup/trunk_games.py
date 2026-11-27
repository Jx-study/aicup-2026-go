"""寫出 trunk_cls 在 Colab 上用的 3 份棋譜 CSV，Colab 不需要完整訓練集。

    train-games.csv     train 切分每位玩家最多 cap × epochs 局（load_split 已打亂，取前面的局），每 epoch 從中抽 cap 局
    newdev-games.csv    newdev 子集（trunk_cls.newdev_subset）
    question-games.csv  題目檔 SC_keys、MX_keys 用到的全部局

用法（在 repo 根目錄）：
    python -m aicup.trunk_games --questions humansl-analysis/questions/questions-s42.npz
"""
import argparse
import os

import numpy as np
import pandas as pd

from aicup import trunk_cls as tc

COLS = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']


def player_keys(df):
    return set(df['rank'].astype(str) + '/' + df['player_id'])


def question_games(questions, df):
    """題目檔 SC_keys、MX_keys 用到的局，順序同 df；訓練集缺局時丟 SystemExit。"""
    keys = sorted({k for t in ('SC', 'MX') for k in questions[t + '_keys'].ravel() if k})
    sub = df[(df['rank'].astype(str) + '_' + df['game_id'].astype(str)).isin(keys)]
    if len(sub) != len(keys):
        raise SystemExit('題目檔有 %d 局，訓練集只找到 %d 局' % (len(keys), len(sub)))
    return sub


def subsets(val_df, tr_df, questions, cap, epochs, newdev_csv=tc.NEWDEV_CSV):
    """回傳 {檔名: DataFrame}；評估玩家不在 val 切分、train 與 val 玩家有交集時丟 SystemExit。"""
    vk, tk = player_keys(val_df), player_keys(tr_df)
    if vk & tk:
        raise SystemExit('train 與 val 切分有共同玩家')
    pool = tr_df.groupby(['rank', 'player_id'], sort=False).head(cap * epochs)
    nd = tc.newdev_subset(val_df, newdev_csv)
    qg = question_games(questions, pd.concat([val_df, tr_df]))
    if not (player_keys(qg) <= vk and player_keys(nd) <= vk):
        raise SystemExit('評估玩家不在 val 切分')
    return {'train-games.csv': pool[COLS], 'newdev-games.csv': nd[COLS], 'question-games.csv': qg[COLS]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--questions', required=True, help='aicup.export_questions 寫出的題目 npz')
    ap.add_argument('--train-dir', default='training-v1.1/training')
    ap.add_argument('--newdev-csv', default=tc.NEWDEV_CSV)
    ap.add_argument('--cap', type=int, default=20, help='每個 epoch 每位玩家最多幾局，同 trunk_cls train --cap')
    ap.add_argument('--epochs', type=int, default=3, help='同 trunk_cls train --epochs；每人局池 = cap × epochs')
    ap.add_argument('--out-dir', default='humansl-analysis/trunk-games')
    args = ap.parse_args()
    paths = {n: os.path.join(args.out_dir, n) for n in ('train-games.csv', 'newdev-games.csv', 'question-games.csv')}
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing:
        raise SystemExit('以下檔案已存在，不覆寫：%s' % ', '.join(existing))
    val_df, tr_df = tc.load_split(args.train_dir)
    out = subsets(val_df, tr_df, np.load(args.questions), args.cap, args.epochs, args.newdev_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, d in out.items():
        d.to_csv(paths[name], index=False, lineterminator='\n')
        print('%-20s %7d 局 -> %s' % (name, len(d), paths[name]))


if __name__ == '__main__':
    main()
