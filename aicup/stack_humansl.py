"""CNN 與 Human SL 的 stacking：每題的 CNN log 機率加 Human SL 特徵組 A，用 logistic regression 輸出 10 類機率。

只用 CNN 沒訓練過的 val 玩家擬合。出題方式與依玩家的 5 折沿用 fit_humansl，
CNN 單獨、Human SL 單獨、stacking 在同一批題目上比較，差用以玩家重抽的配對 bootstrap。
正則化強度用巢狀交叉驗證選：只在外層折的訓練玩家內再依玩家切內層折。

用法（在 repo 根目錄）：
    python -m aicup.stack_humansl humansl-analysis/val648-p3.jsonl --cnn-probs rank-trained-models/<commit hash>/val_probs.npz
"""
import argparse

import numpy as np
from sklearn.model_selection import GroupKFold

from aicup.eval_posthoc import SIZES, parse_sizes
from aicup.fit_humansl import (CS, bootstrap_mean, feature_sets, level_probs, load_games, make_splits,
                               new_model, player_means, predict_avg, question_points, question_x,
                               questions_of, read_val_probs)
from aicup.scoring import aggregate_games, apply_temperature, pick_level

EPS = 1e-12  # log 前的下限，與 scoring.aggregate_games 一致
HSL_C = 0.1  # Human SL 單獨比較用的固定組合 (avg A, C=0.1)


def load_cnn_probs(path, keys, rank_idx):
    """val_probs.npz 中對應 keys 的每局機率，形狀 (len(keys), 10)。

    有 key 不在檔案內，或檔案有 rank_idx 且和 rank_idx 不一致時丟 ValueError。
    """
    index, probs, val_rank = read_val_probs(path)
    missing = [k for k in keys if k not in index]
    if missing:
        raise ValueError('%s 缺少 %d 局，例如 %s' % (path, len(missing), ', '.join(missing[:3])))
    rows = np.array([index[k] for k in keys])
    if val_rank is not None:
        bad = np.flatnonzero(val_rank[rows] != rank_idx)
        if len(bad):
            raise ValueError('%s 的等級和 jsonl 不一致，例如 %s' % (path, keys[bad[0]]))
    return probs[rows].astype(np.float64)


def cnn_question_probs(probs, questions, temp):
    """每題的 CNN 機率，先 apply_temperature 再 aggregate_games 以 log 合併，同 predict_rank。"""
    scaled = apply_temperature(probs, temp)
    return np.array([aggregate_games(scaled[q], 'log') for q in questions])


def question_features(x, cnn_probs, questions, temp):
    """每題的特徵：CNN 合併後的 10 個 log 機率，接各局 Human SL 特徵的平均；x 為 None 時只有 CNN。"""
    logp = np.log(np.maximum(cnn_question_probs(cnn_probs, questions, temp), EPS))
    if x is None:
        return logp
    return np.hstack([logp, question_x(x, questions)])


def hsl_columns(data):
    """Human SL 特徵組 A 在 data['x'] 的欄位。"""
    return feature_sets(data['n_profiles'], data['has_loss'])['A']


def train_rng(seed, fold):
    """第 fold 折抽訓練題的 rng，三種方法各自取一個，抽到的訓練題相同。"""
    return np.random.default_rng([seed, fold])


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


def predict_stack(data, cnn_probs, train, questions, rng, sizes, train_repeats, temp, cs=CS, inner_folds=5,
                  use_hsl=True):
    """用 train 遮罩的玩家切訓練題、選 C、擬合，回傳 (questions 的 10 類機率, 選到的 C)。

    sizes 為每題可用的局數，temp 為 CNN 每局機率的溫度，cs 為候選 C，inner_folds 為內層折數。
    use_hsl 為 False 時只用 CNN 特徵，用來分開「在 val 上重新擬合」與「加入 Human SL」的貢獻。
    """
    x = data['x'][:, hsl_columns(data)] if use_hsl else None
    tr_q, tr_y, tr_players = questions_of(data, train, train_repeats, rng, sizes)
    f_tr = question_features(x, cnn_probs, tr_q, temp)
    c_best = select_c(f_tr, tr_y, tr_players, cs, inner_folds)
    model = new_model(c_best).fit(f_tr, tr_y)
    return level_probs(model, question_features(x, cnn_probs, questions, temp)), c_best


def print_interval(label, values, seed):
    """印出平均與 95% bootstrap 區間，回傳 (平均, 下限, 上限)。"""
    interval = bootstrap_mean(values, seed=seed)
    print('  %s：%.4f，95%% 區間 %.4f～%.4f' % ((label,) + interval))
    return interval


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('jsonl', help='analyze_humansl 的輸出，玩家要是 CNN 沒訓練過的 val 玩家')
    ap.add_argument('--cnn-probs', required=True, help='CNN 的 val_probs.npz')
    ap.add_argument('--cnn-temp', type=float, default=3.0, help='CNN 每局機率的溫度，預設同 predict_rank')
    ap.add_argument('--sizes', type=parse_sizes, default=SIZES, help='每題局數，逗號分隔，預設 5,10,15,20')
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--test-repeats', type=int, default=10)
    ap.add_argument('--inner-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42, help='切折、出題與 bootstrap 的 seed')
    args = ap.parse_args()

    data = load_games(args.jsonl)
    cnn = load_cnn_probs(args.cnn_probs, data['key'], data['rank_idx'])
    splits = make_splits(data, args.sizes, args.test_repeats, seed=args.seed)
    hsl_cols = hsl_columns(data)
    stack, refit, hsl, chosen = [], [], [], []
    for f, (train, q, _, _) in enumerate(splits):
        # 每種方法各自從同一個 seed 起算，訓練題相同，不受執行順序影響
        p_stack, c_stack = predict_stack(data, cnn, train, q, train_rng(args.seed, f), args.sizes,
                                         args.train_repeats, args.cnn_temp, inner_folds=args.inner_folds)
        p_refit, c_refit = predict_stack(data, cnn, train, q, train_rng(args.seed, f), args.sizes,
                                         args.train_repeats, args.cnn_temp, inner_folds=args.inner_folds,
                                         use_hsl=False)
        stack.append(p_stack)
        refit.append(p_refit)
        chosen.append((c_stack, c_refit))
        hsl.append(predict_avg(data, hsl_cols, HSL_C, train, q, train_rng(args.seed, f), args.sizes,
                               args.train_repeats))
    questions = [q for s in splits for q in s[1]]
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    probs = {'stacking': np.concatenate(stack), 'CNN 重新擬合': np.concatenate(refit),
             'Human SL avg A C=%g' % HSL_C: np.concatenate(hsl),
             'CNN': cnn_question_probs(cnn, questions, args.cnn_temp)}
    print('%d 局，%d 位玩家，%d 題，seed %d' % (len(data['x']), len(set(players)), len(questions), args.seed))
    print('各折選到的 C（stacking / CNN 重新擬合）：%s' % ', '.join('%g/%g' % c for c in chosen))
    means = {}
    for name, p in probs.items():
        m = player_means(question_points(labels, pick_level(p)), players)
        means[name] = np.array([m[k] for k in sorted(m)])
        print_interval(name, means[name], args.seed)
    diffs = {other: print_interval('stacking - %s' % other, means['stacking'] - means[other], args.seed)
             for other in list(probs)[1:]}
    lo = diffs['CNN'][1]
    print('暫定門檻（stacking - CNN 的區間下限 > 0）：%s' % ('通過' if lo > 0 else '未通過'))


if __name__ == '__main__':
    main()
