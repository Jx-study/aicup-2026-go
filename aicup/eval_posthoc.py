"""事後分析：讀 train_rank 存下的 val_probs.npz，在 val 上模擬多局題目，比較合併方式與選法。

同一位玩家（player_id 與 rank_idx 相同）的局打亂後依序切成多題，每題局數在 --min-games 到 --max-games 之間隨機。
切題有隨機性，重複 --repeats 次取平均與標準差。只用 numpy，在本機 CPU 上跑。

用法（在 repo 根目錄）：
    python -m aicup.eval_posthoc rank-trained-models/<commit hash>/val_probs.npz
    python -m aicup.eval_posthoc val_probs.npz --min-games 5 --max-games 20 --repeats 10
"""
import argparse
import collections
import itertools

import numpy as np

from aicup.scoring import aggregate_games, pick_level, score_strength

AGGS = ('mean', 'log')
SELECTS = ('argmax', 'pick')


def load_val_probs(path):
    """讀 val_probs.npz；檔案有 epoch 欄位時一併讀出為 int。"""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in ('probs', 'rank_idx', 'key', 'player_id')}
        if 'epoch' in z.files:
            data['epoch'] = int(z['epoch'])
    return data


def make_questions(player_id, rank_idx, min_games, max_games, rng):
    """把每位玩家的局切成多題，每題局數在 min_games 到 max_games 之間（兩端都包含）。

    回傳 (questions, labels)：questions 是每題的局 index 陣列 list，labels 是每題的等級 index。
    每位玩家的局切完後，剩下不足 min_games 的局捨棄。
    """
    if not 1 <= min_games <= max_games:
        raise ValueError('需要 1 <= min_games <= max_games，收到 %d、%d' % (min_games, max_games))
    groups = collections.defaultdict(list)
    for i, key in enumerate(zip(player_id, rank_idx)):
        groups[key].append(i)
    questions, labels = [], []
    for (_, rank), idx in sorted(groups.items()):
        idx = rng.permutation(idx)
        start = 0
        while len(idx) - start >= min_games:
            n = int(rng.integers(min_games, max_games + 1))
            questions.append(idx[start:start + n])
            labels.append(rank)
            start += n
    return questions, np.array(labels, np.int64)


def predict_questions(probs, questions, agg, select):
    preds = np.empty(len(questions), np.int64)
    for q, idx in enumerate(questions):
        dist = aggregate_games(probs[idx], agg)
        preds[q] = dist.argmax() if select == 'argmax' else pick_level(dist)
    return preds


def evaluate(data, min_games, max_games, repeats, seed):
    """回傳 {'single': {select: 分數}, 'questions': {(agg, select): (平均, 標準差)}, 'n_questions': 平均題數}。"""
    probs, labels = data['probs'], data['rank_idx'].astype(np.int64)
    single = {'argmax': score_strength(labels, probs.argmax(1)),
              'pick': score_strength(labels, pick_level(probs))}
    rng = np.random.default_rng(seed)
    scores = collections.defaultdict(list)
    n_questions = []
    for _ in range(repeats):
        questions, q_labels = make_questions(data['player_id'], labels, min_games, max_games, rng)
        if not questions:
            raise ValueError('沒有玩家在 val 中有 %d 局以上，無法切題' % min_games)
        n_questions.append(len(questions))
        for agg, select in itertools.product(AGGS, SELECTS):
            preds = predict_questions(probs, questions, agg, select)
            scores[(agg, select)].append(score_strength(q_labels, preds))
    return {'single': single,
            'questions': {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()},
            'n_questions': float(np.mean(n_questions))}


def games_per_player(data):
    """回傳 (玩家數, 局數最小, 局數中位, 局數最大)。"""
    counts = collections.Counter(zip(data['player_id'], data['rank_idx']))
    if not counts:
        raise ValueError('val_probs 沒有任何局')
    c = np.array(list(counts.values()))
    return len(c), int(c.min()), float(np.median(c)), int(c.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('probs', help='train_rank 存下的 val_probs.npz')
    ap.add_argument('--min-games', type=int, default=5)
    ap.add_argument('--max-games', type=int, default=20)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    data = load_val_probs(args.probs)
    n_players, lo, med, hi = games_per_player(data)
    if 'epoch' in data:
        print('val_probs 來自 epoch %d' % data['epoch'])
    print('val %d 局，%d 位玩家，每位局數 最小 %d / 中位 %g / 最大 %d'
          % (len(data['probs']), n_players, lo, med, hi))
    res = evaluate(data, args.min_games, args.max_games, args.repeats, args.seed)
    print('單局分數：argmax %.4f，pick %.4f' % (res['single']['argmax'], res['single']['pick']))
    print('模擬題（每題 %d～%d 局，平均 %.0f 題，重複 %d 次）：'
          % (args.min_games, args.max_games, res['n_questions'], args.repeats))
    print('| agg | select | 分數 | 標準差 |')
    print('|---|---|---|---|')
    for (agg, select), (mean, std) in res['questions'].items():
        print('| %s | %s | %.4f | %.4f |' % (agg, select, mean, std))


if __name__ == '__main__':
    main()
