"""不訓練的 Human SL 分類器：讀 analyze_humansl 的 jsonl，用各 profile 的 log 機率總和預測等級。

每局對每個 profile 加總目標玩家各手的 log 機率，總和最高的 profile 就是預測等級。
模擬題的切法與 eval_posthoc 相同，同一題把各局的 log 機率總和相加。
log 機率直接相加，不經 aggregate_games，避免單局差距被機率下限截斷。

用法（在 repo 根目錄）：
    python -m aicup.eval_humansl humansl-analysis/pilot.jsonl
    python -m aicup.eval_humansl pilot.jsonl --min-games 5 --max-games 20 --repeats 10
"""
import argparse
import json

import numpy as np

from aicup.eval_posthoc import make_questions
from aicup.scoring import pick_level, score_strength

PROB_FLOOR = 1e-6  # jsonl 的機率四捨五入到小數 6 位，0 代表小於這個精度


def game_loglik(probs):
    """一局的 (10,) log 機率總和。probs 形狀為 (10, 手數)，不合法的手是 NaN，不計入。"""
    p = np.array(probs, dtype=np.float64)
    return np.nansum(np.log(np.maximum(p, PROB_FLOOR)), axis=1)


def load_rows(path):
    """讀 jsonl，回傳 dict：loglik (N, 10)、rank_idx (N,)、player_id (N,)。"""
    loglik, rank_idx, player_id = [], [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            loglik.append(game_loglik(r['probs']))
            rank_idx.append(r['rank_idx'])
            player_id.append(r['player_id'])
    if not loglik:
        raise ValueError('%s 沒有任何局' % path)
    return {'loglik': np.array(loglik), 'rank_idx': np.array(rank_idx, np.int64),
            'player_id': np.array(player_id)}


def to_probs(loglik):
    """log 機率總和轉成機率分布，先驗是均勻分布。形狀 (..., 10)。"""
    logp = loglik - loglik.max(axis=-1, keepdims=True)  # 避免 exp 下溢
    q = np.exp(logp)
    return q / q.sum(axis=-1, keepdims=True)


def predict(loglik, select):
    """回傳 (N,) 等級 index。loglik 形狀 (N, 10)，select 是 argmax 或 pick。"""
    return loglik.argmax(-1) if select == 'argmax' else pick_level(to_probs(loglik))


def evaluate(data, min_games, max_games, repeats, seed):
    """回傳 {'single': {select: 分數}, 'questions': {select: (平均, 標準差)}, 'n_questions': 平均題數}。"""
    loglik, labels = data['loglik'], data['rank_idx']
    selects = ('argmax', 'pick')
    single = {s: score_strength(labels, predict(loglik, s)) for s in selects}
    rng = np.random.default_rng(seed)
    scores = {s: [] for s in selects}
    n_questions = []
    for _ in range(repeats):
        questions, q_labels = make_questions(data['player_id'], labels, min_games, max_games, rng)
        if not questions:
            raise ValueError('沒有玩家有 %d 局以上，無法切題' % min_games)
        n_questions.append(len(questions))
        summed = np.array([loglik[idx].sum(axis=0) for idx in questions])
        for s in selects:
            scores[s].append(score_strength(q_labels, predict(summed, s)))
    return {'single': single,
            'questions': {s: (float(np.mean(v)), float(np.std(v))) for s, v in scores.items()},
            'n_questions': float(np.mean(n_questions))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('jsonl', help='analyze_humansl 的輸出')
    ap.add_argument('--min-games', type=int, default=5)
    ap.add_argument('--max-games', type=int, default=20)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    data = load_rows(args.jsonl)
    res = evaluate(data, args.min_games, args.max_games, args.repeats, args.seed)
    print('%d 局，%d 位玩家' % (len(data['loglik']), len(set(data['player_id']))))
    print('單局分數：argmax %.4f，pick %.4f' % (res['single']['argmax'], res['single']['pick']))
    print('模擬題（每題 %d～%d 局，平均 %.0f 題，重複 %d 次）：'
          % (args.min_games, args.max_games, res['n_questions'], args.repeats))
    print('| select | 分數 | 標準差 |')
    print('|---|---|---|')
    for select, (mean, std) in res['questions'].items():
        print('| %s | %.4f | %.4f |' % (select, mean, std))


if __name__ == '__main__':
    main()
