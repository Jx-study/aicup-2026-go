"""第一段 LR 單獨的模擬題評估，讀 feature_cache 的快取。

出題、切折、train 池出題與 C 的選法同 stack_hgb 的 pool_s1、pool_lrloss_s1，
同樣的特徵群與完整 train 池時，逐題分數和 stack_hgb run 相同；不跑 out-of-fold 與第二段。
特徵群含 co 的方案用同色、HD < 2 的擬合題。
--test-type sc 出同色、HD < 2 的測試題，mx 出混色、HD < 2 的測試題，預設沿用 make_eval_splits 的題。
sc、mx 的測試玩家是每一折以外的全部玩家，rng 為 [seed, 折, 2]（sc）與 [seed, 折, 5]（mx）。
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
from aicup.fit_humansl import (CS, level_probs, new_model, player_keys, question_points, questions_of,
                               same_color_questions)
from aicup.scoring import pick_level
from aicup.stack_hgb import (POOL_RNG, check_disjoint, fill_nan, question_matrix, read_player_ids, stack_groups,
                             summarize)
from aicup.stack_humansl import load_eval_players, make_eval_splits, parse_cs, select_c, train_rng


def parse_arm(text):
    """'名稱=群1,群2@池1,池2' 回傳 (名稱, 特徵群 tuple, train 池名稱 tuple)。"""
    name, rest = text.split('=', 1)
    groups, pools = rest.split('@', 1)
    return name, tuple(g for g in groups.split(',') if g), tuple(p for p in pools.split(',') if p)


def pool_questions(pools, names, seed, train_repeats, sizes, temp, same_color=False):
    """names 的 train 池接起來出題，回傳 ({特徵群: 題特徵}, 標籤, 玩家)；names 為空時回傳 None。

    出題用 rng [seed, POOL_RNG]，和 stack_hgb 相同；same_color 時出同色、HD < 2 的題，rng 為 [seed, POOL_RNG, 1]。
    """
    if not names:
        return None
    data, games = feature_cache.concat([pools[n] for n in names])
    mask = np.ones(len(data['key']), bool)
    if same_color:
        q, y, g = same_color_questions(data, mask, train_repeats, np.random.default_rng([seed, POOL_RNG, 1]), sizes)
    else:
        q, y, g = questions_of(data, mask, train_repeats, np.random.default_rng([seed, POOL_RNG]), sizes)
    return question_matrix(games, None, q, temp), y, g


def retest_splits(data, splits, seed, sizes, test_type, repeats=10):
    """test_type 為 sc、mx 時，每折的測試題改成依 train 以外的玩家重新出題，train 遮罩不變；default 時原樣回傳。

    sc 是同色、HD < 2，rng 為 [seed, 折, 2]；mx 是混色、HD < 2，rng 為 [seed, 折, 5]。每位玩家出 repeats 輪。
    """
    if test_type == 'default':
        return splits
    out = []
    for f, (train, _, _, _) in enumerate(splits):
        if test_type == 'sc':
            q, y, p = same_color_questions(data, ~train, repeats, np.random.default_rng([seed, f, 2]), sizes)
        else:
            q, y, p = questions_of(data, ~train & (data['hd'] < 2), repeats, np.random.default_rng([seed, f, 5]), sizes)
        out.append((train, q, y, p))
    return out


def evaluate(dev, pools, arms, splits, seed, sizes=SIZES, train_repeats=10, temp=3.0, cs=CS, inner_folds=5, log=print,
             probs_out=None):
    """回傳 ({方案: 依 splits 順序合併的預測等級}, {方案: 各折選到的 C})。

    probs_out 給 dict 時，每個方案的逐題 10 級機率（依 splits 順序合併）寫進 probs_out[方案]。

    特徵群含 co 的方案用同色、HD < 2 的擬合題，其餘方案用混色擬合題；同一份測試題給所有方案。
    """
    data, games = dev
    sc_of = {a[0]: 'co' in a[1] for a in arms}
    kinds = sorted(set(sc_of.values()))
    pool_q = {}
    for _, groups, names in arms:
        key = (names, 'co' in groups)
        if key not in pool_q:
            pool_q[key] = pool_questions(pools, names, seed, train_repeats, sizes, temp, key[1])
    preds = {a[0]: [] for a in arms}
    params = {a[0]: [] for a in arms}
    probs_of = {a[0]: [] for a in arms}
    for f, (train, q, _, _) in enumerate(splits):
        test_players = set(player_keys(data)[[i[0] for i in q]])
        m_te = question_matrix(games, None, q, temp)
        fit_q, m_tr = {}, {}
        for sc in kinds:
            if sc:
                fit_q[sc] = same_color_questions(data, train, train_repeats, np.random.default_rng([seed, f, 3]), sizes)
            else:
                fit_q[sc] = questions_of(data, train, train_repeats, train_rng(seed, f), sizes)
            assert not (test_players & set(fit_q[sc][2])), '測試玩家出現在訓練題'
            m_tr[sc] = question_matrix(games, None, fit_q[sc][0], temp)
        for name, groups, names in arms:
            sc = sc_of[name]
            f_all, y_all, g_all = stack_groups(m_tr[sc], groups), fit_q[sc][1], fit_q[sc][2]
            if pool_q[(names, sc)] is not None:
                m_pool, p_y, p_g = pool_q[(names, sc)]
                f_all = np.vstack([f_all, stack_groups(m_pool, groups)])
                y_all, g_all = np.concatenate([y_all, p_y]), np.concatenate([g_all, p_g])
            assert not (test_players & set(g_all)), '擬合玩家含測試玩家'
            f_all, f_te = fill_nan(f_all, stack_groups(m_te, groups))
            c = select_c(f_all, y_all, g_all, cs, inner_folds)
            probs = level_probs(new_model(c).fit(f_all, y_all), f_te)
            preds[name].append(pick_level(probs))
            probs_of[name].append(probs)
            params[name].append(c)
        log('折 %d 完成：%s' % (f, json.dumps({a: params[a][-1] for a in params})))
    if probs_out is not None:
        probs_out.update({a: np.concatenate(v) for a, v in probs_of.items()})
    return {a: np.concatenate(v) for a, v in preds.items()}, params


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dev', required=True, help='開發組的 feature_cache 快取')
    ap.add_argument('--eval-players', required=True, help='出測試題的玩家清單 CSV')
    ap.add_argument('--pool', action='append', default=[], help='train 池，名稱=快取路徑，可重複')
    ap.add_argument('--exclude-players', nargs='*', default=[], help='不可出現在 train 池的玩家清單 CSV，例如封存組')
    ap.add_argument('--arm', action='append', required=True, help='名稱=特徵群@train 池，可重複')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test-type', choices=('default', 'sc', 'mx'), default='default', help='測試題型，見檔頭')
    ap.add_argument('--sizes', type=parse_sizes, default=SIZES)
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--test-repeats', type=int, default=10)
    ap.add_argument('--inner-folds', type=int, default=5)
    ap.add_argument('--cs', type=parse_cs, default=CS)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save-probs', action='store_true', help='輸出 .npz 另存每個方案的逐題 10 級機率（probs_方案）')
    args = ap.parse_args()

    t0 = time.time()
    arms = [parse_arm(a) for a in args.arm]
    dev = feature_cache.load(args.dev)
    pools = {}
    for spec in args.pool:
        name, path = spec.split('=', 1)
        pools[name] = feature_cache.load(path)
    for name, pool in pools.items():
        if pool[0].get('extra') != dev[0].get('extra'):
            ap.error('train 池 %s 的額外 profile 和開發組不同：%s、%s' % (name, pool[0].get('extra'), dev[0].get('extra')))
    others = {'開發組': set(dev[0]['player_id'])}
    others.update({Path(p).name: read_player_ids(p) for p in args.exclude_players})
    for name, pool in pools.items():
        check_disjoint(set(pool[0]['player_id']), others)
    names = list(pools)
    for i, a in enumerate(names):
        check_disjoint(set(pools[a][0]['player_id']), {n: set(pools[n][0]['player_id']) for n in names[i + 1:]})
    missing = sorted({g for _, groups, _ in arms for g in groups} - set(dev[1]) - {'within', 'logn'})
    if missing:
        ap.error('開發組快取沒有特徵群：%s' % ', '.join(missing))
    unknown = sorted({n for _, _, names in arms for n in names} - set(pools))
    if unknown:
        ap.error('未定義的 train 池：%s' % ', '.join(unknown))
    data = dev[0]
    splits = make_eval_splits(data, args.sizes, args.test_repeats, load_eval_players(args.eval_players), seed=args.seed)
    if args.test_type != 'default' and 'hd' not in data:
        ap.error('--test-type %s 需要快取有 color、hd 欄位' % args.test_type)
    splits = retest_splits(data, splits, args.seed, args.sizes, args.test_type, args.test_repeats)
    questions = [q for s in splits for q in s[1]]
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    print('%d 局，%d 位玩家出測試題，%d 題，seed %d，train 池 %s'
          % (len(data['key']), len(set(players)), len(questions), args.seed,
             '、'.join('%s %d 位' % (n, len(set(player_keys(p[0])))) for n, p in pools.items()) or '無'))
    probs = {} if args.save_probs else None
    preds, params = evaluate(dev, pools, arms, splits, args.seed, args.sizes, args.train_repeats, cs=args.cs,
                             inner_folds=args.inner_folds, probs_out=probs)
    scores = {a: question_points(labels, p) for a, p in preds.items()}
    n_games = np.array([len(q) for q in questions])
    summarize(scores, players, n_games, arms[0][0], args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, player_id=players, keys=np.array([','.join(data['key'][q]) for q in questions]),
             rank_idx=labels, n_games=n_games, params=json.dumps(params),
             **{'score_' + a: s for a, s in scores.items()},
             **{'probs_' + a: v.astype(np.float32) for a, v in (probs or {}).items()})
    print('逐題分數 -> %s（%.0f 秒）' % (out, time.time() - t0))


if __name__ == '__main__':
    main()
