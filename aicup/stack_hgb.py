"""非線性第二層與題內分布特徵，以及加 train 池的兩段式第二層。

出題、切折、訓練題沿用 aicup.stack_humansl，所有方案在同一批測試題上比較。
每題特徵只用該題自己的局計算，不使用任何標籤。特徵群：
    cnn      CNN 合併後的 10 個 log 機率，同 stack_humansl
    hsl      Human SL 基本特徵組 base 的題內平均，同 stack_humansl
    dist     每個 profile 逐手 log 機率的 10、25、50、75、90 分位數與 log 機率 < log(0.01) 的比例，題內 nanmean
    stats    每個 profile 的 move_rank == 0 比例、log(1 + move_rank) 平均，加 rank_1d 的 pmax、entropy 平均，題內 nanmean
    hslo、disto、statso  同上，評的是對手的手
    within   每局指標 s、r、d 的題內 20、50、80 分位數與標準差
    logn     log 題內局數
每局指標：s 是目標側逐手 log P(rank_6d) - log P(rank_11k) 的平均，r 是目標側 rank_6d 的 log(1 + move_rank) 平均，
d 是同一局目標側的 s 減對手側的 s。

方案：
    lr_base       cnn+hsl+hslo 的 logistic regression，用 --jsonl、--opp
    lr_hist       同 lr_base，改用 --hist-jsonl、--hist-opp
    hgb_all       全部特徵群的 HistGradientBoostingClassifier，用 history 版
    hgb_all_lr    同樣特徵，第二層用 lr_base 的 LR 與 C 選法，NaN 用訓練題的欄平均填補
    hgb_nodist    hgb_all 去掉 dist、stats、disto、statso
    hgb_nowithin  hgb_all 去掉 within
    pool、pool_s1  兩段式：第一段是 hgb_all 去掉 cnn（--stage1 lr 時只用 hsl+hslo 的 LR），擬合玩家是 train 池加外層訓練玩家；
                  第二段是 cnn 加第一段 out-of-fold 的 log 機率的 LR。pool_s1 是第一段在測試題的輸出直接選等級
    nopool、nopool_s1  train 池為空的 pool、pool_s1
HGB 的 max_depth 與 max_iter 在外層訓練玩家內依玩家切折選，max_iter 的候選用同一個模型的前幾輪預測。

run 的 .npz 欄位：
    player_id    rank_idx|player_id
    keys         每題的局 key，以逗號連接
    rank_idx     每題的等級
    n_games      每題局數
    score_<方案>   各方案的逐題分數
    params       各方案各折選到的超參數，JSON 字串

用法（在 repo 根目錄）：
    python -m aicup.stack_hgb run --jsonl humansl-analysis/val-v180000-dev-p3.jsonl --opp humansl-analysis/val-v180000-dev-p3-opp.jsonl --hist-jsonl humansl-analysis/val-v180000-dev-p3-hist.jsonl --hist-opp humansl-analysis/val-v180000-dev-p3-opp-hist.jsonl --cnn-probs rank-trained-models/59844b9/val_probs_dev.npz --eval-players humansl-analysis/val-players.csv --seed 42 --out runs/hgb-s42.npz
    python -m aicup.stack_hgb run ... --arms lr_base,pool,pool_s1,nopool --pool-jsonl <train 池目標側> --pool-opp <train 池對手側> --exclude-players humansl-analysis/holdout-v180000.csv --out runs/pool-s42.npz
    python -m aicup.stack_hgb compare --files "runs/hgb-s{seed}.npz" --main hgb_all --base lr_base
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from aicup.eval_humansl import PROB_FLOOR
from aicup.eval_posthoc import SIZES, parse_sizes
from aicup.fit_humansl import (CS, bootstrap_mean, level_probs, load_games, new_model, player_keys, player_means,
                               question_points, questions_of)
from aicup.scoring import LEVELS, pick_level
from aicup.stack_humansl import (EPS, hsl_columns, level_weighted, load_cnn_probs, load_eval_players, load_hsl_opp,
                                 make_eval_splits, parse_cs, predict_inputs, question_features, select_c, train_rng)

PROFILES = ('rank_11k', 'rank_1d', 'rank_6d')
QUANTILES = (10, 25, 50, 75, 90)
LOW_LOGP = np.log(0.01)
WITHIN_QUANTILES = (20, 50, 80)
INDICATORS = ('s', 'r', 'd')
DEPTHS = (2, 3)
ITERS = (150, 300)
HGB_FIXED = dict(learning_rate=0.05, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False)
POOL_RNG = 1000  # train 池不參與切折，各折共用同一批題

GROUP_ORDER = ('cnn', 'hsl', 'dist', 'stats', 'hslo', 'disto', 'statso', 'within', 'logn')
GROUP_WIDTH = {'cnn': len(LEVELS), 'hsl': len(PROFILES) + 1, 'dist': len(PROFILES) * (len(QUANTILES) + 1),
               'stats': len(PROFILES) * 2 + 2, 'within': len(INDICATORS) * (len(WITHIN_QUANTILES) + 1), 'logn': 1}
for _g in ('hsl', 'dist', 'stats'):
    GROUP_WIDTH[_g + 'o'] = GROUP_WIDTH[_g]
DIST_GROUPS = ('dist', 'stats', 'disto', 'statso')
K_GROUPS = GROUP_ORDER
ARM_GROUPS = {'hgb_all': K_GROUPS, 'hgb_all_lr': K_GROUPS,
              'hgb_nodist': tuple(g for g in K_GROUPS if g not in DIST_GROUPS),
              'hgb_nowithin': tuple(g for g in K_GROUPS if g != 'within')}
STAGE1_GROUPS = {'hgb': tuple(g for g in K_GROUPS if g != 'cnn'), 'lr': ('hsl', 'hslo')}
LR_ARMS = ('lr_base', 'lr_hist')
HGB_ARMS = ('hgb_all', 'hgb_nodist', 'hgb_nowithin')
POOL_ARMS = ('pool', 'pool_s1')
EMPTY_POOL_ARMS = ('nopool', 'nopool_s1')
ALL_ARMS = LR_ARMS + HGB_ARMS + ('hgb_all_lr',) + POOL_ARMS + EMPTY_POOL_ARMS
DEFAULT_ARMS = ('lr_base', 'lr_hist', 'hgb_all', 'hgb_all_lr', 'hgb_nodist', 'hgb_nowithin')


def _nanmean(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _log_moves(values):
    """逐手機率轉 log，NaN 保留，下限同 eval_humansl。"""
    p = np.array(values, dtype=np.float64)
    return np.log(np.maximum(p, PROB_FLOOR))


def _finite(v):
    v = np.asarray(v, dtype=np.float64)
    return v[np.isfinite(v)]


def game_side_features(row, profiles=PROFILES):
    """一局一側的 (dist (18,), stats (8,), s, r)。

    每項先去掉 NaN 的手；沒有可用的手時為 NaN。沒有 move_rank 欄位（舊版 jsonl）時 stats 與 r 全為 NaN。
    """
    names = list(row['profiles'])
    idx = [names.index(p) for p in profiles]
    logp = [_log_moves(row['probs'][i]) for i in idx]
    dist = []
    for lp in logp:
        v = _finite(lp)
        if len(v):
            dist += list(np.quantile(v, np.array(QUANTILES) / 100)) + [float((v < LOW_LOGP).mean())]
        else:
            dist += [np.nan] * (len(QUANTILES) + 1)
    diff = _finite(logp[profiles.index('rank_6d')] - logp[profiles.index('rank_11k')])
    s = float(diff.mean()) if len(diff) else np.nan
    stats = [np.nan] * GROUP_WIDTH['stats']
    r = np.nan
    if 'move_rank' in row:
        stats = []
        for i in idx:
            v = _finite(row['move_rank'][i])
            stats += [float((v == 0).mean()), float(np.log1p(v).mean())] if len(v) else [np.nan, np.nan]
        mid = idx[profiles.index('rank_1d')]
        for name in ('pmax', 'entropy'):
            v = _finite(row[name][mid])
            stats.append(float(v.mean()) if len(v) else np.nan)
        v = _finite(row['move_rank'][idx[profiles.index('rank_6d')]])
        r = float(np.log1p(v).mean()) if len(v) else np.nan
    return np.array(dist), np.array(stats), s, r


def load_side(path, data, side):
    """jsonl 依 data['key'] 對齊，回傳 {'dist', 'stats', 's', 'r'}，每項的列對應 data 的局。

    有 key 不在檔案內、等級不一致、side 不符，或 profiles 不含 rank_11k、rank_1d、rank_6d 時丟 ValueError。
    沒有 side 欄位的列視為 target。
    """
    with open(path, encoding='utf-8') as f:
        rows = {r['key']: r for r in map(json.loads, (line for line in f if line.strip()))}
    missing = [k for k in data['key'] if k not in rows]
    if missing:
        raise ValueError('%s 缺少 %d 局，例如 %s' % (path, len(missing), missing[0]))
    out = {'dist': [], 'stats': [], 's': [], 'r': []}
    for key, y in zip(data['key'], data['rank_idx']):
        row = rows[key]
        if row['rank_idx'] != y:
            raise ValueError('%s 的等級和目標側 jsonl 不一致，例如 %s' % (path, key))
        if row.get('side', 'target') != side:
            raise ValueError('%s 的 side 不是 %s，例如 %s' % (path, side, key))
        if not set(PROFILES) <= set(row['profiles']):
            raise ValueError('%s 的 profiles 缺少 %s' % (path, ', '.join(PROFILES)))
        dist, stats, s, r = game_side_features(row)
        out['dist'].append(dist)
        out['stats'].append(stats)
        out['s'].append(s)
        out['r'].append(r)
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def load_game_features(target_path, opp_path):
    """讀兩側 jsonl，回傳 (data, games)。

    data 是 load_games 的目標側結果；games 是 {特徵群: 每局特徵}，含 hsl、dist、stats、hslo、disto、statso，
    以及每局指標 s、r、d（各為 (N,)）。
    """
    data = load_games(target_path)
    tgt = load_side(target_path, data, 'target')
    opp = load_side(opp_path, data, 'opponent')
    games = {'hsl': data['x'][:, hsl_columns(data)], 'dist': tgt['dist'], 'stats': tgt['stats'],
             'hslo': load_hsl_opp(opp_path, data), 'disto': opp['dist'], 'statso': opp['stats'],
             's': tgt['s'], 'r': tgt['r'], 'd': tgt['s'] - opp['s']}
    return data, games


def within_stats(values):
    """一題各局的指標值去掉 NaN 後的 20、50、80 分位數與標準差（ddof=0）；全為 NaN 時為 NaN。"""
    v = _finite(values)
    if not len(v):
        return [np.nan] * (len(WITHIN_QUANTILES) + 1)
    return list(np.quantile(v, np.array(WITHIN_QUANTILES) / 100)) + [float(v.std())]


def question_matrix(games, cnn_probs, questions, temp):
    """每題的特徵，回傳 {特徵群: (題數, 寬度)}；cnn_probs 為 None 時沒有 cnn。

    只用每題自己的局，不使用標籤。hsl、hslo 用平均（同 stack_humansl），其餘逐局特徵用 nanmean。
    """
    out = {}
    if cnn_probs is not None:
        out['cnn'] = question_features(None, cnn_probs, questions, temp)
    for g in ('hsl', 'hslo'):
        out[g] = np.array([games[g][q].mean(axis=0) for q in questions])
    for g in DIST_GROUPS:
        out[g] = np.array([_nanmean(games[g][q], axis=0) for q in questions])
    out['within'] = np.array([sum((within_stats(games[i][q]) for i in INDICATORS), []) for q in questions])
    out['logn'] = np.log(np.array([[len(q)] for q in questions], dtype=np.float64))
    return out


def stack_groups(matrix, groups):
    """依 groups 的順序接起特徵群，回傳 (題數, 欄數)。"""
    return np.hstack([matrix[g] for g in groups])


def hgb_model(depth, iters, seed):
    return HistGradientBoostingClassifier(max_depth=depth, max_iter=iters, random_state=seed, **HGB_FIXED)


def expand_probs(classes, p):
    """staged_predict_proba 每輪的輸出展開成 10 欄，欄位是等級 index。"""
    out = np.zeros((len(p), len(LEVELS)))
    out[:, classes] = p
    return out


def select_hgb(f, y, groups, seed, inner_folds=5, depths=None, iters=None):
    """依 groups 切內層折，回傳 ((max_depth, max_iter), {候選: 平均 pick 得分})。

    每個 max_depth 只擬合 max(iters) 輪，較少輪數用 staged_predict_proba 的前幾輪；
    同分依 (max_depth, max_iter) 字典序取最小。depths、iters 為 None 時用 DEPTHS、ITERS。
    """
    depths = DEPTHS if depths is None else depths
    iters = ITERS if iters is None else iters
    points = {(d, n): [] for d in depths for n in iters}
    for tr, te in GroupKFold(n_splits=inner_folds).split(f, y, groups):
        for d in depths:
            model = hgb_model(d, max(iters), seed).fit(f[tr], y[tr])
            for n, p in enumerate(model.staged_predict_proba(f[te]), 1):
                if n in iters:
                    points[(d, n)].append(question_points(y[te], pick_level(expand_probs(model.classes_, p))))
    scores = {k: float(np.concatenate(v).mean()) for k, v in points.items()}
    best = None
    for k in sorted(scores):
        if best is None or scores[k] > scores[best]:
            best = k
    return best, scores


def fit_hgb(f, y, groups, seed, inner_folds=5):
    """選超參數後擬合，回傳 (model, (max_depth, max_iter))。"""
    params, _ = select_hgb(f, y, groups, seed, inner_folds)
    return hgb_model(*params, seed).fit(f, y), params


def fill_nan(f_train, f_test):
    """NaN 用 f_train 的欄平均填補；整欄都是 NaN 時填 0。"""
    mean = np.nan_to_num(_nanmean(f_train, axis=0), nan=0.0)
    return (np.where(np.isnan(f_train), mean, f_train), np.where(np.isnan(f_test), mean, f_test))


def fit_stage1(kind, params, f, y, seed):
    """第一段模型：hgb 時 params 是 (max_depth, max_iter)，lr 時是 C。"""
    if kind == 'hgb':
        return hgb_model(*params, seed).fit(f, y)
    return new_model(params).fit(f, y)


def stage1_log_probs(model, f):
    return np.log(np.maximum(level_probs(model, f), EPS))


def two_stage(kind, dev_tr, dev_te, pool, cnn_tr, cnn_te, seed, cs=CS, inner_folds=5):
    """兩段式第二層，回傳 (第二段的測試題機率, 第一段的測試題機率, info)。

    dev_tr 是 (特徵, 標籤, 玩家) 的開發組訓練題，dev_te 是測試題特徵，pool 是 train 池的 (特徵, 標籤, 玩家) 或 None。
    cnn_tr、cnn_te 是 CNN 的每題 log 機率。第一段的超參數在 train 池加開發組訓練玩家上選一次；
    第二段用的訓練題第一段輸出是 out-of-fold，開發組訓練玩家依玩家切 inner_folds 折。
    info 有 stage1（第一段超參數）、stage2_c 與 fits（每次第一段擬合的 (用途, 擬合玩家的 set)）。
    """
    f_tr, y_tr, g_tr = dev_tr
    if pool is not None:
        overlap = set(pool[2]) & set(g_tr)
        if overlap:
            raise ValueError('train 池和開發組有 %d 位重複玩家' % len(overlap))

    def with_pool(rows):
        f, y, g = f_tr[rows], y_tr[rows], g_tr[rows]
        if pool is None:
            return f, y, g
        return np.vstack([f, pool[0]]), np.concatenate([y, pool[1]]), np.concatenate([g, pool[2]])

    every = np.arange(len(y_tr))
    f_all, y_all, g_all = with_pool(every)
    if kind == 'hgb':
        params, _ = select_hgb(f_all, y_all, g_all, seed, inner_folds)
    else:
        params = select_c(f_all, y_all, g_all, cs, inner_folds)
    fits = [('test', set(g_all))]
    s1_test = stage1_log_probs(fit_stage1(kind, params, f_all, y_all, seed), dev_te)
    oof = np.full((len(y_tr), len(LEVELS)), np.nan)
    for fit_i, pred_i in GroupKFold(n_splits=inner_folds).split(f_tr, y_tr, g_tr):
        f, y, g = with_pool(fit_i)
        fits.append(('oof', set(g)))
        if set(g) & set(g_tr[pred_i]):
            raise AssertionError('第一段的擬合玩家含被預測的玩家')
        oof[pred_i] = stage1_log_probs(fit_stage1(kind, params, f, y, seed), f_tr[pred_i])
    assert not np.isnan(oof).any(), '訓練題的第一段輸出不是每題都有 out-of-fold 預測'
    f2_tr, f2_te = np.hstack([cnn_tr, oof]), np.hstack([cnn_te, s1_test])
    c2 = select_c(f2_tr, y_tr, g_tr, cs, inner_folds)
    probs = level_probs(new_model(c2).fit(f2_tr, y_tr), f2_te)
    return probs, np.exp(s1_test), {'stage1': params, 'stage2_c': c2, 'fits': fits}


def read_player_ids(path):
    """CSV 的 player_id 欄位，回傳 set。"""
    with open(path, encoding='utf-8', newline='') as f:
        return {r['player_id'] for r in csv.DictReader(f)}


def check_disjoint(pool_ids, others):
    """pool_ids 和 others（{名稱: player_id 的 set}）逐一比對，有交集時丟 ValueError。只比 player_id。"""
    for name, ids in others.items():
        common = set(pool_ids) & set(ids)
        if common:
            raise ValueError('train 池和 %s 有 %d 位相同的 player_id，例如 %s' % (name, len(common), sorted(common)[0]))


def same_games(a, b):
    """兩份 load_games 的結果局、等級、玩家與順序都相同。"""
    return all(np.array_equal(a[k], b[k]) for k in ('key', 'rank_idx', 'player_id'))


def evaluate(old, hist, cnn, splits, arms, seed, sizes=SIZES, train_repeats=10, temp=3.0, cs=CS, inner_folds=5,
             pool=None, stage1='hgb', log=print):
    """在 splits 的測試題上跑各方案，回傳 ({方案: 依 splits 順序合併的預測等級}, {方案: 各折的超參數})。

    old、hist 是 (data, games)，局與順序要相同；cnn 是每局 CNN 機率；pool 是 train 池的 (data, games) 或 None。
    訓練題用 train_rng(seed, 折)，和 stack_humansl 相同；train 池的題用 rng [seed, POOL_RNG]，每折相同。
    """
    unknown = sorted(set(arms) - set(ALL_ARMS))
    if unknown:
        raise ValueError('未知的方案：%s' % ', '.join(unknown))
    if not same_games(old[0], hist[0]):
        raise ValueError('舊版與 history 版 jsonl 的局或順序不同')
    if any(a in POOL_ARMS for a in arms) and pool is None:
        raise ValueError('pool、pool_s1 需要 train 池')
    data_old, games_old = old
    data, games = hist
    two_stage_kinds = [p for p in (POOL_ARMS, EMPTY_POOL_ARMS) if set(p) & set(arms)]
    pool_q = None
    if pool is not None and set(POOL_ARMS) & set(arms):
        check_disjoint(set(pool[0]['player_id']), {'開發組': set(data['player_id'])})
        q, y, g = questions_of(pool[0], np.ones(len(pool[0]['key']), bool), train_repeats,
                               np.random.default_rng([seed, POOL_RNG]), sizes)
        pool_q = (stack_groups(question_matrix(pool[1], None, q, temp), STAGE1_GROUPS[stage1]), y, g)
    preds = {a: [] for a in arms}
    params = {a: [] for a in arms}
    x_old = np.hstack([games_old['hsl'], games_old['hslo']])
    x_hist = np.hstack([games['hsl'], games['hslo']])
    for f, (train, q, _, _) in enumerate(splits):
        for arm, d, x in (('lr_base', data_old, x_old), ('lr_hist', data, x_hist)):
            if arm in arms:
                p, c = predict_inputs(x, cnn, d, train, q, train_rng(seed, f), sizes, train_repeats, temp, cs=cs,
                                      inner_folds=inner_folds)
                preds[arm].append(pick_level(p))
                params[arm].append(c)
        tr_q, tr_y, tr_g = questions_of(data, train, train_repeats, train_rng(seed, f), sizes)
        test_players = set(player_keys(data)[[i[0] for i in q]])
        assert not (test_players & set(tr_g)), '測試玩家出現在訓練題'
        m_tr = question_matrix(games, cnn, tr_q, temp)
        m_te = question_matrix(games, cnn, q, temp)
        for arm in HGB_ARMS:
            if arm in arms:
                f_tr, f_te = stack_groups(m_tr, ARM_GROUPS[arm]), stack_groups(m_te, ARM_GROUPS[arm])
                model, best = fit_hgb(f_tr, tr_y, tr_g, seed, inner_folds)
                preds[arm].append(pick_level(level_probs(model, f_te)))
                params[arm].append(list(best))
        if 'hgb_all_lr' in arms:
            f_tr, f_te = fill_nan(stack_groups(m_tr, ARM_GROUPS['hgb_all_lr']), stack_groups(m_te, ARM_GROUPS['hgb_all_lr']))
            c = select_c(f_tr, tr_y, tr_g, cs, inner_folds)
            preds['hgb_all_lr'].append(pick_level(level_probs(new_model(c).fit(f_tr, tr_y), f_te)))
            params['hgb_all_lr'].append(c)
        for arm_pair in two_stage_kinds:
            p2, p1, info = two_stage(stage1, (stack_groups(m_tr, STAGE1_GROUPS[stage1]), tr_y, tr_g),
                                     stack_groups(m_te, STAGE1_GROUPS[stage1]),
                                     pool_q if arm_pair == POOL_ARMS else None, m_tr['cnn'], m_te['cnn'], seed, cs,
                                     inner_folds)
            for used, fit_players in info['fits']:
                assert not (fit_players & test_players), '第一段的擬合玩家含測試玩家（%s）' % used
            for arm, p in zip(arm_pair, (p2, p1)):
                if arm in arms:
                    preds[arm].append(pick_level(p))
                    stage1_params = list(info['stage1']) if stage1 == 'hgb' else info['stage1']
                    params[arm].append({'stage1': stage1_params, 'stage2_c': info['stage2_c']})
        log('折 %d 完成：%s' % (f, ', '.join('%s %s' % (a, json.dumps(params[a][-1])) for a in arms)))
    return {a: np.concatenate(v) for a, v in preds.items()}, params


def player_vector(score, players, keys):
    """每位玩家的平均分數，依 keys 的順序。"""
    m = player_means(score, players)
    return np.array([m[k] for k in keys])


def summarize(scores, players, n_games, base, seed):
    """印出各方案的不加權、每級等權分數與減 base 的差，以及依每題局數分開的差。"""
    keys = sorted(set(players))
    levels = np.array([int(k.split('|')[0]) for k in keys])
    vec = {a: player_vector(s, players, keys) for a, s in scores.items()}
    print('| 方案 | 不加權 | 95%% 區間 | 每級等權 | 減 %s | 95%% 區間 | 每級等權差 |' % base)
    print('|---|---|---|---|---|---|---|')
    for a, v in vec.items():
        mean, lo, hi = bootstrap_mean(v, seed=seed)
        d, dlo, dhi = bootstrap_mean(v - vec[base], seed=seed)
        print('| %s | %.4f | %.4f～%.4f | %.4f | %+.4f | %+.4f～%+.4f | %+.4f |'
              % (a, mean, lo, hi, level_weighted(v, levels), d, dlo, dhi, level_weighted(v - vec[base], levels)))
    sizes = sorted(set(n_games.tolist()))
    print('\n依每題局數，減 %s（逐題平均）\n' % base)
    print('| 方案 | %s |' % ' | '.join('%d 局' % n for n in sizes))
    print('|' + '---|' * (1 + len(sizes)))
    for a, s in scores.items():
        print('| %s | %s |' % (a, ' | '.join('%+.4f' % (s[n_games == n] - scores[base][n_games == n]).mean()
                                             for n in sizes)))


def load_pool(args):
    """--pool-jsonl、--pool-opp 讀成 (data, games)，並和 --exclude-players 的清單比對 player_id。"""
    pool = load_game_features(args.pool_jsonl, args.pool_opp)
    others = {Path(p).name: read_player_ids(p) for p in args.exclude_players}
    check_disjoint(set(pool[0]['player_id']), others)
    return pool


def cmd_run(args):
    arms = tuple(a.strip() for a in args.arms.split(',') if a.strip())
    old = load_game_features(args.jsonl, args.opp)
    hist = load_game_features(args.hist_jsonl, args.hist_opp)
    data = hist[0]
    cnn = load_cnn_probs(args.cnn_probs, data['key'], data['rank_idx'])
    pool = load_pool(args) if args.pool_jsonl else None
    eval_players = load_eval_players(args.eval_players) if args.eval_players else None
    splits = make_eval_splits(data, args.sizes, args.test_repeats, eval_players, seed=args.seed)
    questions = [q for s in splits for q in s[1]]
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    print('%d 局，%d 位玩家出測試題，%d 題，seed %d%s'
          % (len(data['key']), len(set(players)), len(questions), args.seed,
             '' if pool is None else '，train 池 %d 局、%d 位' % (len(pool[0]['key']), len(set(player_keys(pool[0]))))))
    preds, params = evaluate(old, hist, cnn, splits, arms, args.seed, args.sizes, args.train_repeats, args.cnn_temp,
                             args.cs, args.inner_folds, pool, args.stage1)
    scores = {a: question_points(labels, p) for a, p in preds.items()}
    n_games = np.array([len(q) for q in questions])
    summarize(scores, players, n_games, arms[0], args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, player_id=players, keys=np.array([','.join(data['key'][q]) for q in questions]),
             rank_idx=labels, n_games=n_games, params=json.dumps(params),
             **{'score_' + a: s for a, s in scores.items()})
    print('逐題分數 ->', out)


def load_seed_scores(paths):
    """讀同一個 seed 的多個 run 輸出並合併 score 欄位，assert 逐題 keys 與玩家相同。

    回傳 ({方案: 逐題分數}, player_id, n_games)；同一個方案出現在多個檔案時分數要相同。
    """
    scores, ref = {}, None
    for path in paths:
        z = np.load(path, allow_pickle=False)
        if ref is None:
            ref = {k: z[k] for k in ('keys', 'player_id', 'rank_idx', 'n_games')}
        for k in ref:
            assert np.array_equal(z[k], ref[k]), '%s 的 %s 和 %s 不同' % (path, k, paths[0])
        for name in z.files:
            if name.startswith('score_'):
                arm = name[len('score_'):]
                assert arm not in scores or np.array_equal(scores[arm], z[name]), '%s 的 %s 和其他檔案不同' % (path, arm)
                scores[arm] = z[name]
    return scores, ref['player_id'], ref['n_games']


def cmd_compare(args):
    """多個 seed 合併的配對比較，第一項是主要比較。

    合併區間：每個 seed 先算每位測試玩家的平均配對差，再對每位玩家取各 seed 的平均，依玩家重抽 5,000 次取 95% 區間。
    主要比較的通過條件：合併區間下限 > 0、各 seed 平均增量 ≥ +0.005、每級等權的各 seed 平均增量 ≥ 0。
    """
    seeds = [int(s) for s in args.seeds.split(',')]
    pairs = [(args.main, args.base)] + [tuple(p.split('-', 1)) for p in args.refs.split(',') if p.strip()]
    pv, size_diff, ref_keys, levels = {}, {}, None, None
    for seed in seeds:
        scores, players, n_games = load_seed_scores([p.format(seed=seed) for p in args.files])
        keys = sorted(set(players))
        if ref_keys is None:
            ref_keys, levels = keys, np.array([int(k.split('|')[0]) for k in keys])
        assert keys == ref_keys, 'seed %d 的玩家集合不同' % seed
        pv[seed] = {a: player_vector(s, players, keys) for a, s in scores.items()}
        size_diff[seed] = {(a, b): {int(n): float((scores[a] - scores[b])[n_games == n].mean())
                                    for n in np.unique(n_games)} for a, b in pairs}
    print('# 配對比較：seed %s，測試玩家 %d 位\n' % (','.join(map(str, seeds)), len(ref_keys)))
    arms = sorted({a for p in pairs for a in p})
    print('| seed | %s |' % ' | '.join(arms))
    print('|' + '---|' * (1 + len(arms)))
    for s in seeds:
        print('| %d | %s |' % (s, ' | '.join('%.4f' % pv[s][a].mean() for a in arms)))
    for i, (a, b) in enumerate(pairs):
        diffs = {s: pv[s][a] - pv[s][b] for s in seeds}
        print('\n## %s：%s 減 %s\n' % ('主要比較' if i == 0 else '參考', a, b))
        print('| seed | 差 | 95% 區間 | 每級等權差 |')
        print('|---|---|---|---|')
        lwd = []
        for s in seeds:
            d, lo, hi = bootstrap_mean(diffs[s], seed=s)
            lwd.append(level_weighted(diffs[s], levels))
            print('| %d | %+.4f | %+.4f～%+.4f | %+.4f |' % (s, d, lo, hi, lwd[-1]))
        d, lo, hi = bootstrap_mean(np.mean([diffs[s] for s in seeds], axis=0), n=5000, seed=42)
        mean_lw = float(np.mean(lwd))
        print('| 合併 | %+.4f | %+.4f～%+.4f | %+.4f |' % (d, lo, hi, mean_lw))
        sizes = sorted(size_diff[seeds[0]][(a, b)])
        print('\n依每題局數：%s' % '，'.join('%d 局 %+.4f' % (n, np.mean([size_diff[s][(a, b)][n] for s in seeds]))
                                         for n in sizes))
        if i == 0:
            ok = [lo > 0, d >= 0.005, mean_lw >= 0]
            print('\n判定：合併區間下限 > 0 %s；平均增量 ≥ +0.005 %s；每級等權平均增量 ≥ 0 %s -> %s'
                  % (ok[0], ok[1], ok[2], '通過' if all(ok) else '未通過'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    run = sub.add_parser('run', help='跑一個 seed 的各方案，寫出逐題分數')
    run.add_argument('--jsonl', required=True, help='lr_base 用的目標側 jsonl')
    run.add_argument('--opp', required=True, help='lr_base 用的對手側 jsonl')
    run.add_argument('--hist-jsonl', required=True, help='hgb_all、pool 與 lr_hist 用的目標側 history 版 jsonl，局與順序同 --jsonl')
    run.add_argument('--hist-opp', required=True, help='hgb_all、pool 與 lr_hist 用的對手側 history 版 jsonl')
    run.add_argument('--cnn-probs', required=True, help='CNN 的 val_probs.npz')
    run.add_argument('--eval-players', help='含 player_id、rank 欄位的 CSV，只有這些玩家出測試題')
    run.add_argument('--arms', default=','.join(DEFAULT_ARMS),
                     help='逗號分隔，第一個是印出時的基準，可用 %s' % ', '.join(ALL_ARMS))
    run.add_argument('--pool-jsonl', help='train 池的目標側 history 版 jsonl，pool、pool_s1 需要')
    run.add_argument('--pool-opp', help='train 池的對手側 history 版 jsonl')
    run.add_argument('--exclude-players', nargs='*', default=[],
                     help='含 player_id 欄位的 CSV，train 池的玩家不能出現在其中')
    run.add_argument('--stage1', choices=('hgb', 'lr'), default='hgb', help='lr 時第一段只用 hsl+hslo')
    run.add_argument('--seed', type=int, default=42, help='切折、出題、HGB 與 bootstrap 的 seed')
    run.add_argument('--out', required=True)
    run.add_argument('--cnn-temp', type=float, default=3.0)
    run.add_argument('--sizes', type=parse_sizes, default=SIZES)
    run.add_argument('--train-repeats', type=int, default=10)
    run.add_argument('--test-repeats', type=int, default=10)
    run.add_argument('--inner-folds', type=int, default=5)
    run.add_argument('--cs', type=parse_cs, default=CS)
    cmp_ = sub.add_parser('compare', help='多個 seed 合併的配對比較')
    cmp_.add_argument('--files', nargs='+', required=True, help='run 的輸出，路徑中的 {seed} 會代入各 seed')
    cmp_.add_argument('--seeds', default='42,43,44,45,46')
    cmp_.add_argument('--main', required=True, help='主要比較的左方案')
    cmp_.add_argument('--base', required=True, help='主要比較的右方案')
    cmp_.add_argument('--refs', default='', help='逗號分隔的參考比較，例如 hgb_all_lr-lr_base,lr_hist-lr_base')
    args = ap.parse_args()
    if args.cmd == 'run':
        if bool(args.pool_jsonl) != bool(args.pool_opp):
            ap.error('--pool-jsonl 和 --pool-opp 要同時給')
        if args.pool_jsonl and not args.exclude_players:
            ap.error('給了 --pool-jsonl 就要用 --exclude-players 給 train 池不能重疊的玩家清單')
        cmd_run(args)
    else:
        cmd_compare(args)


if __name__ == '__main__':
    main()
