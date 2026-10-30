"""事後分析：讀 train_rank 存下的 val_probs.npz，在 val 上模擬多局題目，比較合併方式與選法。

模擬題：
    玩家以 (player_id, rank_idx) 區分，每位玩家的局打亂後依序切成多題。
    每題局數在 --min-games 到 --max-games 之間隨機，切剩不足 --min-games 的局捨棄。
    切題重複 --repeats 次，分數取平均與標準差。

溫度校準（--calibrate）：
    依玩家把 val 分成兩半，一半挑溫度，另一半評分，再交換。
    每種合併方式與選法各自挑溫度，評分取兩半的平均。

用法（在 repo 根目錄）：
    python -m aicup.eval_posthoc rank-trained-models/<commit hash>/val_probs.npz
    python -m aicup.eval_posthoc val_probs.npz --min-games 5 --max-games 20 --repeats 10
    python -m aicup.eval_posthoc val_probs.npz --calibrate
"""
import argparse
import collections
import itertools

import numpy as np

from aicup.scoring import aggregate_games, apply_temperature, pick_level, score_strength

AGGS = ('mean', 'log')
SELECTS = ('argmax', 'pick')
TEMPS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)  # 溫度校準的候選值，大於 1 時縮小各等級機率的差距


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


def split_players(player_id, rank_idx, rng):
    """把玩家（player_id 與 rank_idx 相同）隨機分成兩半，回傳每局是否屬於第一半的 bool 陣列。"""
    keys = sorted(set(zip(player_id, rank_idx)))
    first = {keys[i] for i in rng.permutation(len(keys))[:len(keys) // 2]}
    return np.array([k in first for k in zip(player_id, rank_idx)])


def subset(data, mask):
    return {k: (v[mask] if k in ('probs', 'rank_idx', 'key', 'player_id') else v) for k, v in data.items()}


def question_scores(data, temps, combos, min_games, max_games, repeats, seed):
    """回傳 {(agg, select, temp): 模擬題平均分數}，同一次切題共用於所有溫度與組合。"""
    labels = data['rank_idx'].astype(np.int64)
    scaled = {t: apply_temperature(data['probs'], t) for t in temps}
    rng = np.random.default_rng(seed)
    scores = collections.defaultdict(list)
    for _ in range(repeats):
        questions, q_labels = make_questions(data['player_id'], labels, min_games, max_games, rng)
        if not questions:
            raise ValueError('沒有玩家在 val 中有 %d 局以上，無法切題' % min_games)
        for (agg, select), t in itertools.product(combos, temps):
            preds = predict_questions(scaled[t], questions, agg, select)
            scores[(agg, select, t)].append(score_strength(q_labels, preds))
    return {k: float(np.mean(v)) for k, v in scores.items()}


def calibrate(data, temps, min_games, max_games, repeats, seed):
    """兩折溫度校準，回傳 {(agg, select): {'before': 分數, 'after': 分數, 'temps': [各折挑出的溫度]}}。

    before、after 都是兩折評分半的平均；before 用溫度 1。
    """
    combos = list(itertools.product(AGGS, SELECTS))
    rng = np.random.default_rng(seed)
    first = split_players(data['player_id'], data['rank_idx'], rng)
    res = {c: {'before': [], 'after': [], 'temps': []} for c in combos}
    for fit_mask in (first, ~first):
        fit = question_scores(subset(data, fit_mask), temps, combos, min_games, max_games, repeats, seed)
        best = {c: max(temps, key=lambda t: fit[c + (t,)]) for c in combos}
        test = question_scores(subset(data, ~fit_mask), sorted(set(best.values()) | {1.0}), combos,
                               min_games, max_games, repeats, seed)
        for c in combos:
            res[c]['before'].append(test[c + (1.0,)])
            res[c]['after'].append(test[c + (best[c],)])
            res[c]['temps'].append(best[c])
    return {c: {'before': float(np.mean(v['before'])), 'after': float(np.mean(v['after'])),
                'temps': v['temps']} for c, v in res.items()}


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
    ap.add_argument('--calibrate', action='store_true', help='依玩家分兩半做溫度校準')
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
    if args.calibrate:
        cal = calibrate(data, TEMPS, args.min_games, args.max_games, args.repeats, args.seed)
        print('溫度校準（兩折，各折挑出的溫度 / 評分半的平均分數）：')
        print('| agg | select | 溫度 | 校準前 | 校準後 |')
        print('|---|---|---|---|---|')
        for (agg, select), r in cal.items():
            print('| %s | %s | %s | %.4f | %.4f |' % (agg, select, ', '.join('%g' % t for t in r['temps']),
                                                   r['before'], r['after']))


if __name__ == '__main__':
    main()
