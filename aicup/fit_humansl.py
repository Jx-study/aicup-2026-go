"""校正 Human SL 分類器：用 analyze_humansl 的 jsonl 訓練 logistic regression，依玩家交叉驗證。

每局特徵是各 profile 的每手平均 log 機率、手數與前 N 手的損失統計，一題的特徵是各局特徵的算術平均。
profile 數從 jsonl 推得；沒有損失欄位（analyze_humansl 不跑損失 query）時只有基本特徵組 base。
訓練與驗證都用模擬題，出題方式與 eval_posthoc 相同，同一位玩家不跨折。
方式 avg 先平均特徵再分類，方式 game 逐局分類後 log 合併。

用法（在 repo 根目錄）：
    python -m aicup.fit_humansl humansl-analysis/pilot-6000-hd.jsonl
    python -m aicup.fit_humansl pilot-6000-hd.jsonl --cnn-probs rank-trained-models/<commit hash>/val_probs.npz
    python -m aicup.fit_humansl pilot-6000-hd.jsonl --config "avg base C=0.1" --cnn-probs ...   # 固定組合，避免選擇偏差
"""
import argparse
import itertools
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aicup.eval_humansl import game_loglik
from aicup.eval_posthoc import SIZES, make_questions, parse_sizes
from aicup.scoring import LEVELS, NEAR_SCORE, aggregate_games, apply_temperature, pick_level

CUTS = (50, 100)  # 損失統計只取目標玩家的前 N 手
CS = (0.01, 0.1, 1.0, 10.0)  # logistic regression 的正則化強度候選
GAME_TEMPS = (1.0, 2.0, 4.0)  # 逐局機率彼此相關，log 合併前要降低信心
N_STATS = 6  # 每個 cut 的目數損失與勝率損失各取平均、中位數、標準差


def game_features(row, cuts=CUTS):
    """一局的特徵向量：每手平均 log 機率、log 手數，有損失欄位時再接各 cut 的損失統計。"""
    valid = np.isfinite(np.array(row['probs'], dtype=np.float64)).sum(axis=1)
    parts = [game_loglik(row['probs']) / np.maximum(valid, 1), [np.log(len(row['turns']))]]
    if 'score_loss' not in row:
        return np.concatenate(parts)
    for cut in cuts:
        for name in ('score_loss', 'winrate_loss'):
            x = np.array(row[name][:cut], dtype=np.float64)
            parts.append([x.mean(), np.median(x), x.std()])
    return np.concatenate(parts)


def load_games(path, cuts=CUTS):
    """讀 jsonl，回傳 dict：x (N, D)、rank_idx、player_id、key、n_profiles、profiles、has_loss。

    各局的 profile 數、profile 名稱與是否有損失欄位必須一致，否則拒絕。
    color 只能是 B 或 W；每局都有 color 與 hd 欄位時另有 black（目標玩家執黑）與 hd（讓子數）兩個每局陣列。
    profiles 是 profile 名稱的 tuple，jsonl 沒有 profiles 欄位時為 None。
    """
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError('%s 沒有任何局' % path)
    n_profiles = {len(r['probs']) for r in rows}
    if len(n_profiles) > 1:
        raise ValueError('%s 各局的 profile 數不同：%s' % (path, sorted(n_profiles)))
    names = {tuple(r['profiles']) if 'profiles' in r else None for r in rows}
    if len(names) > 1:
        raise ValueError('%s 各局的 profile 名稱不同' % path)
    has_loss = {'score_loss' in r for r in rows}
    if len(has_loss) > 1:
        raise ValueError('%s 有些局有損失欄位、有些沒有' % path)
    data = {'x': np.array([game_features(r, cuts) for r in rows]),
            'rank_idx': np.array([r['rank_idx'] for r in rows], np.int64),
            'player_id': np.array([r['player_id'] for r in rows]),
            'key': np.array([r['key'] for r in rows]),
            'n_profiles': n_profiles.pop(), 'profiles': names.pop(), 'has_loss': has_loss.pop()}
    has_color = {'color' in r and 'hd' in r for r in rows}
    if len(has_color) > 1:
        raise ValueError('%s 有些局有 color、hd 欄位，有些沒有' % path)
    if has_color.pop():
        colors = {r['color'] for r in rows}
        if not colors <= {'B', 'W'}:
            raise ValueError('%s 的 color 只能是 B 或 W，看到 %s' % (path, sorted(map(str, colors - {'B', 'W'}))))
        data['black'] = np.array([r['color'] == 'B' for r in rows])
        data['hd'] = np.array([r['hd'] for r in rows], np.int64)
    return data


def feature_sets(n_profiles=10, has_loss=True, cuts=CUTS):
    """特徵組名稱對應欄位 index：base 是各 profile 每手平均 log 機率與 log 手數，有損失欄位時 loss<cut> 是 base 加上該 cut 的損失統計。"""
    n_base = n_profiles + 1  # 各 profile 加手數
    base = list(range(n_base))
    sets = {'base': base}
    if not has_loss:
        return sets
    for i, cut in enumerate(cuts):
        start = n_base + i * N_STATS
        sets['loss%d' % cut] = base + list(range(start, start + N_STATS))
    return sets


def question_points(labels, preds):
    """每題的官方得分。"""
    d = np.abs(np.asarray(labels) - np.asarray(preds))
    return (d == 0) + NEAR_SCORE * (d == 1)


def new_model(c, class_weight=None):
    """class_weight 為 'balanced' 時各等級的總權重相同，不帶訓練題的等級先驗。"""
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight=class_weight))


def level_probs(model, x):
    """predict_proba 展開成 10 欄，欄位是等級 index，訓練資料沒有的等級機率為 0。"""
    out = np.zeros((len(x), len(LEVELS)))
    out[:, model.classes_] = model.predict_proba(x)
    return out


def player_keys(data):
    """每局的玩家識別字串，玩家由 player_id 與 rank_idx 一起區分。"""
    return np.array(['%d|%s' % (r, p) for r, p in zip(data['rank_idx'], data['player_id'])])


def questions_of(data, mask, repeats, rng, sizes):
    """回傳 (questions, labels, players)，questions 的局 index 對應整份資料，players 是 player_keys 的值。"""
    idx = np.flatnonzero(mask)
    keys = player_keys(data)
    qs, labels = [], []
    for _ in range(repeats):
        q, y = make_questions(data['player_id'][idx], data['rank_idx'][idx], sizes, rng)
        qs += [idx[i] for i in q]
        labels.append(y)
    if not qs:
        raise ValueError('沒有玩家有 %d 局以上，無法出題' % min(sizes))
    return qs, np.concatenate(labels), keys[[q[0] for q in qs]]


def same_color_questions(data, mask, repeats, rng, sizes=SIZES):
    """同色、HD < 2 的題目，回傳 (questions, labels, players)，格式同 questions_of。

    每位玩家每輪出 1 題：從 HD < 2 的局有 min(sizes) 局以上的顏色中隨機選一種，
    局數從 sizes 中不超過該顏色可用局數的值隨機選。兩種顏色都不夠的玩家不出題。
    """
    if 'black' not in data:
        raise ValueError('資料沒有 color、hd 欄位，無法出同色題')
    keys = player_keys(data)
    by_player = {}
    for i in np.flatnonzero(mask & (data['hd'] < 2)):
        by_player.setdefault(keys[i], ([], []))[0 if data['black'][i] else 1].append(i)
    qs, labels, players = [], [], []
    for _ in range(repeats):
        for k in sorted(by_player):
            opts = [c for c in (0, 1) if len(by_player[k][c]) >= min(sizes)]
            if not opts:
                continue
            games = np.array(by_player[k][opts[rng.integers(len(opts))]])
            usable = [s for s in sizes if s <= len(games)]
            q = rng.choice(games, usable[rng.integers(len(usable))], replace=False)
            qs.append(q)
            labels.append(data['rank_idx'][q[0]])
            players.append(k)
    if not qs:
        raise ValueError('沒有玩家有 %d 局以上同色、HD < 2 的局，無法出題' % min(sizes))
    return qs, np.array(labels), np.array(players)


def color_features(data):
    """每局 3 欄 (N, 3)：執黑、HD 為 1、HD 為 1 且執黑；題內平均後是顏色、HD1 比例與兩者乘積的比例。"""
    black = data['black'].astype(np.float64)
    h1 = (data['hd'] == 1).astype(np.float64)
    return np.c_[black, h1, h1 * black]


def question_x(x, questions):
    return np.array([x[q].mean(axis=0) for q in questions])


def predict_avg(data, columns, c, train, questions, rng, sizes, train_repeats):
    """各局特徵平均後分類。sizes 是每題可用的局數，見 eval_posthoc.make_questions。"""
    x = data['x'][:, columns]
    tr_q, tr_y, _ = questions_of(data, train, train_repeats, rng, sizes)
    model = new_model(c).fit(question_x(x, tr_q), tr_y)
    return level_probs(model, question_x(x, questions))


def predict_game(data, columns, c, temp, train, questions):
    """逐局分類後 log 合併。"""
    x = data['x'][:, columns]
    model = new_model(c).fit(x[train], data['rank_idx'][train])
    probs = apply_temperature(level_probs(model, x), temp)
    return np.array([aggregate_games(probs[q], 'log') for q in questions])


def player_folds(data, folds, seed):
    """依玩家分層切折，回傳每局的折編號。"""
    keys = player_keys(data)
    players = sorted(set(zip(keys, data['rank_idx'])))
    y = np.array([r for _, r in players])
    fold_of = {}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_of.update({players[i][0]: f for i in te})
    return np.array([fold_of[k] for k in keys])


def make_splits(data, sizes, repeats, folds=5, seed=42):
    """先切好各折的測試題，回傳 [(train 遮罩, questions, labels, players)]。

    測試題只由 seed 與折編號決定，所有組合都在同一批題目上比較。
    """
    fold = player_folds(data, folds, seed)
    splits = []
    for f in range(folds):
        q, y, p = questions_of(data, fold == f, repeats, np.random.default_rng([seed, f]), sizes)
        splits.append((fold != f, q, y, p))
    return splits


def cross_validate(splits, predict, seed=42):
    """predict(train, questions, rng) 回傳每題機率；回傳依 splits 順序合併的機率。"""
    rng = np.random.default_rng(seed)
    return np.concatenate([predict(train, q, rng) for train, q, _, _ in splits])


def player_means(scores, players):
    """{玩家: 該玩家各題的平均得分}。"""
    return {p: float(scores[players == p].mean()) for p in np.unique(players)}


def bootstrap_mean(values, n=1000, seed=42):
    """重抽玩家的平均與 95% 區間。"""
    v = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def configs(data, sizes, train_repeats):
    """回傳 [(組合名稱, predict)]。"""
    out = []
    for (name, cols), c in itertools.product(feature_sets(data['n_profiles'], data['has_loss']).items(), CS):
        out.append(('avg %s C=%g' % (name, c), lambda tr, q, rng, cols=cols, c=c: predict_avg(
            data, cols, c, tr, q, rng, sizes, train_repeats)))
        for t in GAME_TEMPS:
            out.append(('game %s C=%g T=%g' % (name, c, t), lambda tr, q, rng, cols=cols, c=c, t=t: predict_game(
                data, cols, c, t, tr, q)))
    return out


def evaluate_grid(data, splits, sizes, train_repeats, seed, only=None):
    """跑全部組合，回傳 [(名稱, 每題 pick 得分, 每題 argmax 得分)]，依玩家平均的 pick 分數排序。

    only 是組合名稱時只跑這一個，名稱不存在時拒絕。
    """
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    grid = configs(data, sizes, train_repeats)
    if only is not None:
        names = [name for name, _ in grid]
        if only not in names:
            raise ValueError('沒有組合 %r，可用的有：%s' % (only, ', '.join(names)))
        grid = [g for g in grid if g[0] == only]
    results = []
    for name, predict in grid:
        probs = cross_validate(splits, predict, seed)
        results.append((name, question_points(labels, pick_level(probs)), question_points(labels, probs.argmax(1))))
    return sorted(results, key=lambda r: -np.mean(list(player_means(r[1], players).values())))


def read_val_probs(path):
    """讀 val_probs.npz，回傳 (key 對應列號的 dict, 每局機率, rank_idx 或 None)。"""
    with np.load(path, allow_pickle=False) as z:
        val_keys, probs = z['key'], z['probs']
        val_rank = z['rank_idx'] if 'rank_idx' in z.files else None
    return {k: i for i, k in enumerate(val_keys)}, probs, val_rank


def cnn_question_points(path, keys, questions, labels, temp):
    """CNN 在同一批題目上的 log + pick 得分。回傳 (題目遮罩, 得分)，遮罩標出全部局都在 val_probs 的題目。

    val_probs 有 rank_idx 時，檢查涵蓋的局和題目的等級一致，避免檔案版本對不上。
    """
    index, probs, val_rank = read_val_probs(path)
    covered = np.array([all(keys[g] in index for g in q) for q in questions])
    if val_rank is not None:
        for q, y in zip(np.array(questions, dtype=object)[covered], np.asarray(labels)[covered]):
            if any(val_rank[index[keys[g]]] != y for g in q):
                raise ValueError('%s 的等級和 jsonl 不一致，例如 %s' % (path, keys[q[0]]))
    scaled = apply_temperature(probs, temp)
    dist = [aggregate_games(scaled[[index[keys[g]] for g in q]], 'log')
            for q, ok in zip(questions, covered) if ok]
    if not dist:
        return covered, np.empty(0)
    return covered, question_points(labels[covered], pick_level(np.array(dist)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('jsonl', help='analyze_humansl 的輸出')
    ap.add_argument('--sizes', type=parse_sizes, default=SIZES, help='每題局數，逗號分隔，預設 5,10,15,20')
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--test-repeats', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--top', type=int, default=12, help='印出前幾個組合')
    ap.add_argument('--cnn-probs', help='CNN 的 val_probs.npz，用來在同一批題目上比較')
    ap.add_argument('--cnn-temp', type=float, default=3.0, help='CNN 每局機率的溫度，預設同 predict_rank')
    ap.add_argument('--config', help='只跑這個組合，例如 "avg base C=0.1"；和 CNN 比較時用來避免選擇偏差')
    args = ap.parse_args()

    data = load_games(args.jsonl)
    splits = make_splits(data, args.sizes, args.test_repeats, seed=args.seed)
    results = evaluate_grid(data, splits, args.sizes, args.train_repeats, args.seed, args.config)
    questions = [q for s in splits for q in s[1]]
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    print('%d 局，%d 位玩家，%d 題，%d 個組合' % (len(data['x']), len(set(players)), len(questions), len(results)))
    print('| 組合 | pick | 95% 區間 | argmax |')
    print('|---|---|---|---|')
    for name, pick, argmax in results[:args.top]:
        mean, lo, hi = bootstrap_mean(player_means(pick, players).values(), seed=args.seed)
        print('| %s | %.4f | %.4f～%.4f | %.4f |' % (name, mean, lo, hi, np.mean(
            list(player_means(argmax, players).values()))))

    if args.cnn_probs:
        covered, cnn = cnn_question_points(args.cnn_probs, data['key'], questions, labels, args.cnn_temp)
        name, best, _ = results[0]
        p = players[covered]
        print('CNN 比較：%d 位玩家，%d 題' % (len(set(p)), covered.sum()))
        if args.config is None:
            print('  警告：Human SL 用的是同一批題目上挑出的最佳組合，分數偏樂觀；固定組合請加 --config')
        if covered.any():
            a, b = player_means(best[covered], p), player_means(cnn, p)
            shared = sorted(a)
            x = np.array([a[k] for k in shared])
            y = np.array([b[k] for k in shared])
            for label, v in (('Human SL ' + name, x), ('CNN', y), ('差', x - y)):
                print('  %s：%.4f，95%% 區間 %.4f～%.4f' % ((label,) + bootstrap_mean(v, seed=args.seed)))


if __name__ == '__main__':
    main()
