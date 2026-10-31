"""配對 bootstrap：比較兩份 val_probs.npz 的分數差（B 減 A），判斷改動是否超過雜訊。

做法：
    兩份 val_probs 的局必須相同（以 key 對齊）。
    每次重抽 val 玩家（玩家以 (player_id, rank_idx) 區分，可重複抽到），兩個版本用同一批玩家與同一組切題。
    單局分數：重抽後的各局得分平均。
    模擬題分數：重抽後的玩家各自切題（切法同 eval_posthoc），兩版預測同一批題目。

用法（在 repo 根目錄）：
    python -m aicup.bootstrap_compare rank-trained-models/<A>/val_probs.npz rank-trained-models/<B>/val_probs.npz
    python -m aicup.bootstrap_compare A.npz B.npz --repeats 1000 --agg log --select pick --temp 3.0
    python -m aicup.bootstrap_compare A.npz B.npz --history-a A/history.csv --history-b B/history.csv
"""
import argparse
import csv

import numpy as np

from aicup.eval_posthoc import load_val_probs
from aicup.scoring import NEAR_SCORE, apply_temperature, pick_level

EPS = 1e-12  # log 前的下限，與 scoring.aggregate_games 一致


def align(a, b):
    """依 key 把 b 的順序對到 a，回傳重排後的 b；兩邊局集合不同就報錯。"""
    if len(a['key']) != len(b['key']) or set(a['key'].tolist()) != set(b['key'].tolist()):
        raise ValueError('兩份 val_probs 的局不同，無法配對')
    if (a['key'] == b['key']).all():
        return b
    sb = np.argsort(b['key'])
    order = sb[np.searchsorted(b['key'][sb], a['key'])]
    return {k: (v[order] if k in ('probs', 'rank_idx', 'key', 'player_id') else v) for k, v in b.items()}


def player_groups(player_id, rank_idx):
    """回傳 (members, labels)：members[c] 是第 c 位玩家的局 index，labels[c] 是該玩家的等級。"""
    groups = {}
    for i, key in enumerate(zip(player_id.tolist(), rank_idx.tolist())):
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    return [np.array(groups[k]) for k in keys], np.array([k[1] for k in keys], np.int64)


def game_scores(labels, preds):
    """每局得分：猜中 1，差一級 1/e，其餘 0。"""
    diff = np.abs(labels - preds)
    return np.where(diff == 0, 1.0, np.where(diff == 1, NEAR_SCORE, 0.0))


def cut_questions(members, draw, min_games, max_games, rng):
    """對重抽到的玩家各自切題，回傳 (局 index 串接陣列, 每題起點, 每題所屬玩家的編號)。"""
    flat, starts, owner = [], [], []
    pos = 0
    for c in draw:
        idx = rng.permutation(members[c])
        s = 0
        while len(idx) - s >= min_games:
            n = int(rng.integers(min_games, max_games + 1))
            chunk = idx[s:s + n]  # 最後一題可能不足 n 局，與 eval_posthoc 相同
            flat.append(chunk)
            starts.append(pos)
            owner.append(c)
            pos += len(chunk)
            s += n
    if not flat:
        raise ValueError('重抽的玩家都不足 %d 局，無法切題' % min_games)
    return np.concatenate(flat), np.array(starts), np.array(owner)


def predict_cut(probs, logp, flat, starts, agg, select):
    """對切好的題目合併各局並選等級，回傳每題預測。"""
    sizes = np.diff(np.append(starts, len(flat)))
    if agg == 'mean':
        dist = np.add.reduceat(probs[flat], starts, axis=0) / sizes[:, None]
    else:
        s = np.add.reduceat(logp[flat], starts, axis=0)
        s -= s.max(axis=1, keepdims=True)
        dist = np.exp(s)
        dist /= dist.sum(axis=1, keepdims=True)
    return dist.argmax(1) if select == 'argmax' else pick_level(dist)


def single_preds(probs, select):
    return probs.argmax(1) if select == 'argmax' else pick_level(probs)


def paired_bootstrap(probs_a, probs_b, player_id, rank_idx, repeats, min_games, max_games, agg, select, seed,
                     temp=1.0):
    """回傳 {'single': (A, B) 各次分數, 'questions': (A, B) 各次分數}，每個元素是長度 repeats 的陣列。

    單局分數用原始機率；模擬題先對每局機率做溫度縮放再合併，與 predict_rank 的推論一致。
    """
    labels = rank_idx.astype(np.int64)
    members, member_labels = player_groups(player_id, rank_idx)
    n_players = len(members)
    counts = np.array([len(m) for m in members])

    # 單局：每位玩家的得分總和，重抽時以權重加總
    single_sum = []
    for probs in (probs_a, probs_b):
        g = game_scores(labels, single_preds(probs, select))
        single_sum.append(np.array([g[m].sum() for m in members]))
    q_probs = [apply_temperature(p, temp) for p in (probs_a, probs_b)]
    logp = [np.log(np.clip(p, EPS, None)) for p in q_probs]

    rng = np.random.default_rng(seed)
    single = np.empty((2, repeats))
    questions = np.empty((2, repeats))
    for r in range(repeats):
        draw = rng.integers(0, n_players, n_players)
        w = np.bincount(draw, minlength=n_players)
        for k in range(2):
            single[k, r] = (single_sum[k] * w).sum() / (counts * w).sum()
        flat, starts, owner = cut_questions(members, draw, min_games, max_games, rng)
        q_labels = member_labels[owner]
        for k, probs in enumerate(q_probs):
            preds = predict_cut(probs, logp[k], flat, starts, agg, select)
            questions[k, r] = game_scores(q_labels, preds).mean()
    return {'single': tuple(single), 'questions': tuple(questions)}


def summarize(a, b):
    """回傳 B 減 A 的差：平均、標準差、95% 區間、差大於 0 的比例。"""
    d = b - a
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'mean_a': float(a.mean()), 'mean_b': float(b.mean()), 'diff': float(d.mean()),
            'sd': float(d.std(ddof=1)) if len(d) > 1 else 0.0, 'lo': float(lo), 'hi': float(hi),
            'p_pos': float((d > 0).mean())}


def tail_mean(path, column='val_score_pick', n=4):
    """讀 history.csv，回傳最後 n 個 epoch 的 column 平均。"""
    with open(path, newline='', encoding='utf-8') as f:
        values = [float(row[column]) for row in csv.DictReader(f)]
    if len(values) < n:
        raise ValueError('%s 只有 %d 個 epoch，不足 %d' % (path, len(values), n))
    return float(np.mean(values[-n:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('probs_a', help='比較對象的 val_probs.npz')
    ap.add_argument('probs_b', help='新版本的 val_probs.npz，差值為 B 減 A')
    ap.add_argument('--repeats', type=int, default=1000)
    ap.add_argument('--min-games', type=int, default=5)
    ap.add_argument('--max-games', type=int, default=20)
    ap.add_argument('--agg', choices=('mean', 'log'), default='log')
    ap.add_argument('--select', choices=('argmax', 'pick'), default='pick')
    ap.add_argument('--temp', type=float, default=3.0, help='模擬題每局機率的溫度，預設同 predict_rank')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--history-a', help='A 的 history.csv，用來比最後 4 個 epoch 的 val_score_pick 平均')
    ap.add_argument('--history-b', help='B 的 history.csv')
    args = ap.parse_args()
    if not 1 <= args.min_games <= args.max_games:
        ap.error('需要 1 <= min-games <= max-games')
    if bool(args.history_a) != bool(args.history_b):
        ap.error('--history-a 與 --history-b 要一起給')

    a = load_val_probs(args.probs_a)
    b = align(a, load_val_probs(args.probs_b))
    print('val_probs 來自 epoch：A %s，B %s' % (a.get('epoch', '?'), b.get('epoch', '?')))
    res = paired_bootstrap(a['probs'], b['probs'], a['player_id'], a['rank_idx'], args.repeats,
                           args.min_games, args.max_games, args.agg, args.select, args.seed, args.temp)
    print('配對 bootstrap（重抽玩家 %d 次，模擬題 %d～%d 局，%s + %s，溫度 %g，差為 B 減 A）：'
          % (args.repeats, args.min_games, args.max_games, args.agg, args.select, args.temp))
    print('| 指標 | A | B | 差 | 標準差 | 95% 區間 | 差大於 0 |')
    print('|---|---|---|---|---|---|---|')
    for name, key in (('單局 %s' % args.select, 'single'), ('模擬題', 'questions')):
        s = summarize(*res[key])
        print('| %s | %.4f | %.4f | %+.4f | %.4f | [%+.4f, %+.4f] | %.3f |'
              % (name, s['mean_a'], s['mean_b'], s['diff'], s['sd'], s['lo'], s['hi'], s['p_pos']))
    if args.history_a and args.history_b:
        ta, tb = tail_mean(args.history_a), tail_mean(args.history_b)
        print('最後 4 個 epoch 的 val_score_pick 平均：A %.4f，B %.4f，差 %+.4f' % (ta, tb, tb - ta))


if __name__ == '__main__':
    main()
