"""第一段 LR 單獨的模擬題評估，讀 feature_cache 的快取。

出題、切折、train 池出題與 C 的選法同 stack_hgb 的 pool_s1、pool_lrloss_s1，
同樣的特徵群與完整 train 池時，逐題分數和 stack_hgb run 相同；不跑 out-of-fold 與第二段。
特徵群有 NaN 時用擬合題的欄平均填補（整欄都是 NaN 時填 0）。

方案用 --arm 名稱=特徵群@train 池，train 池是 --pool 的名稱，以逗號連接，留空表示不用 train 池：
    --arm s1_p0=hsl,hslo@ --arm s1_p12=hsl,hslo@p1,p2 --arm loss_p12=hsl,hslo,loss,losso@p1,p2
train 池之間、train 池和開發組、train 池和 --exclude-players 的玩家都不可重複。
輸出 .npz 的欄位同 stack_hgb run，可用 stack_hgb compare 比較。

用法（在 repo 根目錄）：
    python -m aicup.eval_hsl_lr --dev humansl-analysis/cache/val648.npz --eval-players humansl-analysis/val-players.csv --pool p1=humansl-analysis/cache/train_p1.npz --pool p2=humansl-analysis/cache/train_p2.npz --arm s1_p12=hsl,hslo@p1,p2 --seed 42 --out Note/ref/r14/curve-s42.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from aicup import feature_cache
from aicup.eval_posthoc import SIZES, parse_sizes
from aicup.fit_humansl import CS, level_probs, new_model, player_keys, question_points, questions_of
from aicup.scoring import pick_level
from aicup.stack_hgb import (POOL_RNG, check_disjoint, fill_nan, question_matrix, read_player_ids, stack_groups,
                             summarize)
from aicup.stack_humansl import load_eval_players, make_eval_splits, parse_cs, select_c, train_rng


def parse_arm(text):
    """'名稱=群1,群2@池1,池2' 回傳 (名稱, 特徵群 tuple, train 池名稱 tuple)。"""
    name, rest = text.split('=', 1)
    groups, pools = rest.split('@', 1)
    return name, tuple(g for g in groups.split(',') if g), tuple(p for p in pools.split(',') if p)


def pool_questions(pools, names, seed, train_repeats, sizes, temp):
    """names 的 train 池接起來出題，回傳 ({特徵群: 題特徵}, 標籤, 玩家)；names 為空時回傳 None。

    出題用 rng [seed, POOL_RNG]，和 stack_hgb 相同。
    """
    if not names:
        return None
    data, games = feature_cache.concat([pools[n] for n in names])
    q, y, g = questions_of(data, np.ones(len(data['key']), bool), train_repeats,
                           np.random.default_rng([seed, POOL_RNG]), sizes)
    return question_matrix(games, None, q, temp), y, g


def evaluate(dev, pools, arms, splits, seed, sizes=SIZES, train_repeats=10, temp=3.0, cs=CS, inner_folds=5, log=print):
    """回傳 ({方案: 依 splits 順序合併的預測等級}, {方案: 各折選到的 C})。"""
    data, games = dev
    pool_q = {}
    for _, _, names in arms:
        if names not in pool_q:
            pool_q[names] = pool_questions(pools, names, seed, train_repeats, sizes, temp)
    preds = {a[0]: [] for a in arms}
    params = {a[0]: [] for a in arms}
    for f, (train, q, _, _) in enumerate(splits):
        tr_q, tr_y, tr_g = questions_of(data, train, train_repeats, train_rng(seed, f), sizes)
        test_players = set(player_keys(data)[[i[0] for i in q]])
        assert not (test_players & set(tr_g)), '測試玩家出現在訓練題'
        m_tr = question_matrix(games, None, tr_q, temp)
        m_te = question_matrix(games, None, q, temp)
        for name, groups, names in arms:
            f_all, y_all, g_all = stack_groups(m_tr, groups), tr_y, tr_g
            if pool_q[names] is not None:
                m_pool, p_y, p_g = pool_q[names]
                f_all = np.vstack([f_all, stack_groups(m_pool, groups)])
                y_all, g_all = np.concatenate([y_all, p_y]), np.concatenate([g_all, p_g])
            assert not (test_players & set(g_all)), '擬合玩家含測試玩家'
            f_all, f_te = fill_nan(f_all, stack_groups(m_te, groups))
            c = select_c(f_all, y_all, g_all, cs, inner_folds)
            preds[name].append(pick_level(level_probs(new_model(c).fit(f_all, y_all), f_te)))
            params[name].append(c)
        log('折 %d 完成：%s' % (f, json.dumps({a: params[a][-1] for a in params})))
    return {a: np.concatenate(v) for a, v in preds.items()}, params


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dev', required=True, help='開發組的 feature_cache 快取')
    ap.add_argument('--eval-players', required=True, help='出測試題的玩家清單 CSV')
    ap.add_argument('--pool', action='append', default=[], help='train 池，名稱=快取路徑，可重複')
    ap.add_argument('--exclude-players', nargs='*', default=[], help='不可出現在 train 池的玩家清單 CSV，例如封存組')
    ap.add_argument('--arm', action='append', required=True, help='名稱=特徵群@train 池，可重複')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sizes', type=parse_sizes, default=SIZES)
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--test-repeats', type=int, default=10)
    ap.add_argument('--inner-folds', type=int, default=5)
    ap.add_argument('--cs', type=parse_cs, default=CS)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    t0 = time.time()
    arms = [parse_arm(a) for a in args.arm]
    dev = feature_cache.load(args.dev)
    pools = {}
    for spec in args.pool:
        name, path = spec.split('=', 1)
        pools[name] = feature_cache.load(path)
    others = {'開發組': set(dev[0]['player_id'])}
    others.update({Path(p).name: read_player_ids(p) for p in args.exclude_players})
    for name, pool in pools.items():
        check_disjoint(set(pool[0]['player_id']), others)
    names = list(pools)
    for i, a in enumerate(names):
        check_disjoint(set(pools[a][0]['player_id']), {n: set(pools[n][0]['player_id']) for n in names[i + 1:]})
    unknown = sorted({n for _, _, names in arms for n in names} - set(pools))
    if unknown:
        ap.error('未定義的 train 池：%s' % ', '.join(unknown))
    data = dev[0]
    splits = make_eval_splits(data, args.sizes, args.test_repeats, load_eval_players(args.eval_players), seed=args.seed)
    questions = [q for s in splits for q in s[1]]
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    print('%d 局，%d 位玩家出測試題，%d 題，seed %d，train 池 %s'
          % (len(data['key']), len(set(players)), len(questions), args.seed,
             '、'.join('%s %d 位' % (n, len(set(player_keys(p[0])))) for n, p in pools.items()) or '無'))
    preds, params = evaluate(dev, pools, arms, splits, args.seed, args.sizes, args.train_repeats, cs=args.cs,
                             inner_folds=args.inner_folds)
    scores = {a: question_points(labels, p) for a, p in preds.items()}
    n_games = np.array([len(q) for q in questions])
    summarize(scores, players, n_games, arms[0][0], args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, player_id=players, keys=np.array([','.join(data['key'][q]) for q in questions]),
             rank_idx=labels, n_games=n_games, params=json.dumps(params),
             **{'score_' + a: s for a, s in scores.items()})
    print('逐題分數 -> %s（%.0f 秒）' % (out, time.time() - t0))


if __name__ == '__main__':
    main()
