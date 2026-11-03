"""CNN 與 Human SL 的 stacking：每題的 CNN log 機率加 Human SL 特徵組 A，用 logistic regression 輸出 10 類機率。

只用 CNN 沒訓練過的 val 玩家擬合。切題與依玩家的 5 折沿用 fit_humansl，
CNN 單獨、Human SL 單獨、stacking 在同一批題目上比較，差用以玩家重抽的配對 bootstrap。
正則化強度用巢狀交叉驗證選：只在外層折的訓練玩家內再依玩家切內層折。

用法（在 repo 根目錄）：
    python -m aicup.stack_humansl humansl-analysis/val648-p3.jsonl --cnn-probs rank-trained-models/<commit hash>/val_probs.npz
"""
import argparse

import numpy as np
from sklearn.model_selection import GroupKFold

from aicup.fit_humansl import (CS, bootstrap_mean, feature_sets, level_probs, load_games, make_splits,
                               new_model, player_means, predict_avg, question_points, question_x,
                               questions_of)
from aicup.scoring import aggregate_games, apply_temperature, pick_level

PROB_FLOOR = 1e-12  # 合併後的機率可能下溢成 0，取 log 前的下限
HSL_C = 0.1  # Human SL 單獨的組合事先固定為 avg A C=0.1


def load_cnn_probs(path, keys, rank_idx):
    """val_probs.npz 中對應 keys 的每局機率，形狀 (len(keys), 10)。

    有 key 不在檔案內，或檔案有 rank_idx 且和 rank_idx 不一致時丟 ValueError。
    """
    with np.load(path, allow_pickle=False) as z:
        val_keys, probs = z['key'], z['probs']
        val_rank = z['rank_idx'] if 'rank_idx' in z.files else None
    index = {k: i for i, k in enumerate(val_keys)}
    missing = [k for k in keys if k not in index]
    if missing:
        raise ValueError('%s 缺少 %d 局，例如 %s' % (path, len(missing), ', '.join(missing[:3])))
    rows = np.array([index[k] for k in keys])
    if val_rank is not None and (val_rank[rows] != rank_idx).any():
        raise ValueError('%s 的等級和 jsonl 不一致，例如 %s' % (path, keys[np.flatnonzero(val_rank[rows] != rank_idx)[0]]))
    return probs[rows].astype(np.float64)


def cnn_question_probs(probs, questions, temp):
    """每題的 CNN 機率，溫度與 log 合併同 predict_rank。"""
    scaled = apply_temperature(probs, temp)
    return np.array([aggregate_games(scaled[q], 'log') for q in questions])


def question_features(x, cnn_probs, questions, temp):
    """每題的特徵：CNN 合併後的 10 個 log 機率，接各局 Human SL 特徵的平均；x 為 None 時只有 CNN。"""
    logp = np.log(np.maximum(cnn_question_probs(cnn_probs, questions, temp), PROB_FLOOR))
    if x is None:
        return logp
    return np.hstack([logp, question_x(x, questions)])


def select_c(x, y, groups, cs=CS, inner_folds=5):
    """依 groups 切內層折，回傳平均 pick 得分最高的 C，同分取 cs 中較前者。"""
    splitter = GroupKFold(n_splits=inner_folds)
    scores = []
    for c in cs:
        points = []
        for tr, te in splitter.split(x, y, groups):
            model = new_model(c).fit(x[tr], y[tr])
            points.append(question_points(y[te], pick_level(level_probs(model, x[te]))))
        scores.append(np.concatenate(points).mean())
    return cs[int(np.argmax(scores))]


def predict_stack(data, cnn_probs, train, questions, rng, games, train_repeats, temp, cs=CS, inner_folds=5,
                  use_hsl=True):
    """用 train 遮罩的玩家切訓練題、選 C、擬合，回傳 (questions 的 10 類機率, 選到的 C)。

    use_hsl 為 False 時只用 CNN 特徵，用來分開「在 val 上重新擬合」與「加入 Human SL」的貢獻。
    """
    x = data['x'][:, feature_sets(data['n_profiles'], data['has_loss'])['A']] if use_hsl else None
    tr_q, tr_y, tr_players = questions_of(data, train, train_repeats, rng, *games)
    f_tr = question_features(x, cnn_probs, tr_q, temp)
    c = select_c(f_tr, tr_y, tr_players, cs, inner_folds)
    model = new_model(c).fit(f_tr, tr_y)
    return level_probs(model, question_features(x, cnn_probs, questions, temp)), c


def report(label, values, seed):
    print('  %s：%.4f，95%% 區間 %.4f～%.4f' % ((label,) + bootstrap_mean(values, seed=seed)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('jsonl', help='analyze_humansl 的輸出，玩家要是 CNN 沒訓練過的 val 玩家')
    ap.add_argument('--cnn-probs', required=True, help='CNN 的 val_probs.npz')
    ap.add_argument('--cnn-temp', type=float, default=3.0, help='CNN 每局機率的溫度，預設同 predict_rank')
    ap.add_argument('--min-games', type=int, default=5)
    ap.add_argument('--max-games', type=int, default=20)
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--test-repeats', type=int, default=10)
    ap.add_argument('--inner-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42, help='切折、切題與 bootstrap 的 seed')
    args = ap.parse_args()

    data = load_games(args.jsonl)
    cnn = load_cnn_probs(args.cnn_probs, data['key'], data['rank_idx'])
    games = (args.min_games, args.max_games)
    splits = make_splits(data, games, args.test_repeats, seed=args.seed)
    hsl_cols = feature_sets(data['n_profiles'], data['has_loss'])['A']
    stack, refit, hsl, chosen = [], [], [], []
    for f, (train, q, _, _) in enumerate(splits):
        # 每種方法各自從同一個 seed 起算，訓練題相同，不受執行順序影響
        def rng():
            return np.random.default_rng([args.seed, f])
        p, c = predict_stack(data, cnn, train, q, rng(), games, args.train_repeats, args.cnn_temp,
                             inner_folds=args.inner_folds)
        stack.append(p)
        p, c_refit = predict_stack(data, cnn, train, q, rng(), games, args.train_repeats, args.cnn_temp,
                                   inner_folds=args.inner_folds, use_hsl=False)
        refit.append(p)
        chosen.append((c, c_refit))
        hsl.append(predict_avg(data, hsl_cols, HSL_C, train, q, rng(), games, args.train_repeats))
    questions = [q for s in splits for q in s[1]]
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    probs = {'stacking': np.concatenate(stack), 'CNN 重新擬合': np.concatenate(refit),
             'Human SL avg A C=%g' % HSL_C: np.concatenate(hsl),
             'CNN': cnn_question_probs(cnn, questions, args.cnn_temp)}
    print('%d 局，%d 位玩家，%d 題，切折 seed %d' % (len(data['x']), len(set(players)), len(questions), args.seed))
    print('各折選到的 C（stacking / CNN 重新擬合）：%s' % ', '.join('%g/%g' % c for c in chosen))
    means = {}
    for name, p in probs.items():
        m = player_means(question_points(labels, pick_level(p)), players)
        means[name] = np.array([m[k] for k in sorted(m)])
        report(name, means[name], args.seed)
    for other in list(probs)[1:]:
        report('stacking - %s' % other, means['stacking'] - means[other], args.seed)
    _, lo, _ = bootstrap_mean(means['stacking'] - means['CNN'], seed=args.seed)
    print('暫定門檻（stacking - CNN 的區間下限 > 0）：%s' % ('通過' if lo > 0 else '未通過'))


if __name__ == '__main__':
    main()
