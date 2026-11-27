"""產生 val648 的固定評估題，供 tf_run.py 的 pool、cv 模式與 trunk_cls 的 infer 在相同題目上預測。

題目只由局的順序、玩家、等級、顏色與 HD 決定，從 hsl_games 寫出的 val648 games.csv 重算，不需要 Human SL 分析結果。
每個 seed 的切折同 stack_humansl.make_eval_splits（5 折、每折 10 輪），只有 val-players.csv 的玩家出題。
每折的題目：
    SC  同色、HD < 2 的題，同 fit_humansl.same_color_questions，rng [seed, 折, 2]
    MX  HD < 2 的局混色出題，同 fit_humansl.questions_of，rng [seed, 折, 5]
輸出 <out-dir>/questions-s<seed>.npz：
    {SC,MX}_keys  (題數, 20) 局 key，不足 20 局補空字串
    {SC,MX}_n     局數
    {SC,MX}_y     真實等級
    {SC,MX}_p     玩家 key（rank_idx|player_id）
    {SC,MX}_fold  題目所在的折

用法（在 repo 根目錄）：
    python -m aicup.export_questions --seeds 42 47 48 49 50 51
"""
import argparse
import os

import numpy as np
import pandas as pd

from aicup.eval_posthoc import SIZES
from aicup.fit_humansl import questions_of, same_color_questions
from aicup.scoring import LEVELS
from aicup.stack_humansl import load_eval_players, make_eval_splits
from aicup.val_players import hd_of

FOLDS, REPEATS, MAX_GAMES = 5, 10, 20


def load_games(path):
    """把 games.csv 轉成出題用的 data：key、player_id、rank_idx、black、hd，順序同 games.csv。"""
    g = pd.read_csv(path, dtype={'player_id': str, 'game_id': str})
    return {'key': (g['rank'] + '_' + g['game_id']).to_numpy(),
            'player_id': g['player_id'].to_numpy(),
            'rank_idx': g['rank'].map(LEVELS.index).to_numpy(),
            'black': (g['color'] == 'B').to_numpy(),
            'hd': g['sgf_content'].map(hd_of).to_numpy()}


def export(data, eval_players, seed):
    """回傳一個 seed 的題目 dict，欄位見模組說明。"""
    out = {t: {'q': [], 'y': [], 'p': [], 'fold': []} for t in ('SC', 'MX')}
    for f, (train, _, _, _) in enumerate(make_eval_splits(data, SIZES, REPEATS, eval_players, FOLDS, seed)):
        test = ~train
        tq = {'SC': same_color_questions(data, test, REPEATS, np.random.default_rng([seed, f, 2]), SIZES),
              'MX': questions_of(data, test & (data['hd'] < 2), REPEATS, np.random.default_rng([seed, f, 5]), SIZES)}
        for t, (q, y, p) in tq.items():
            out[t]['q'] += list(q)
            out[t]['y'].append(y)
            out[t]['p'].append(p)
            out[t]['fold'].append(np.full(len(q), f))
    save = {}
    for t, o in out.items():
        keys = np.full((len(o['q']), MAX_GAMES), '', dtype='<U16')
        for i, q in enumerate(o['q']):
            keys[i, :len(q)] = data['key'][q]
        save.update({t + '_keys': keys, t + '_n': np.array([len(q) for q in o['q']]),
                     t + '_y': np.concatenate(o['y']), t + '_p': np.concatenate(o['p']),
                     t + '_fold': np.concatenate(o['fold'])})
    return save


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--games', default='humansl-analysis/hist6d-val648/games.csv', help='hsl_games 的 val648 games.csv')
    ap.add_argument('--eval-players', default='humansl-analysis/val-players.csv')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out-dir', default='humansl-analysis/questions')
    args = ap.parse_args()
    data = load_games(args.games)
    eval_players = load_eval_players(args.eval_players)
    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seeds:
        path = os.path.join(args.out_dir, 'questions-s%d.npz' % seed)
        if os.path.exists(path):
            raise SystemExit('%s 已存在，不覆寫' % path)
        save = export(data, eval_players, seed)
        np.savez_compressed(path, **save)
        print('seed %d：SC %d 題、MX %d 題 -> %s' % (seed, len(save['SC_n']), len(save['MX_n']), path))


if __name__ == '__main__':
    main()
