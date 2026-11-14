"""選手辨識評估與計分模組。"""

import math
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def score_player_ranks(ranks: Sequence[int]) -> float:
    """計算一組名次的官方平均得分。

    正解排第 r 名（1-based）得 exp(-(r - 1))，大於 5 名或未命中得 0。
    """
    if not ranks:
        return 0.0
    scores = []
    for r in ranks:
        if 1 <= r <= 5:
            scores.append(math.exp(-(r - 1)))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def evaluate_predictions(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, str]
) -> Dict[str, float]:
    """評估預測結果並回傳得分指標。

    回傳包含 score、top1_acc、top5_acc 與 total 的字典。
    """
    ranks = []
    top1 = 0
    top5 = 0
    total = len(ground_truth)

    for qid, true_pid in ground_truth.items():
        preds = predictions.get(qid, [])
        top5_preds = list(preds[:5])
        if true_pid in top5_preds:
            r = top5_preds.index(true_pid) + 1
            ranks.append(r)
            top5 += 1
            if r == 1:
                top1 += 1
        else:
            ranks.append(0)

    avg_score = score_player_ranks(ranks)
    return {
        'score': avg_score,
        'top1_acc': top1 / total if total > 0 else 0.0,
        'top5_acc': top5 / total if total > 0 else 0.0,
        'total': float(total),
    }


def load_benchmark_players(
    train_dir: str = 'training',
    n_players: int = 400,
    min_games: int = 25,
    levels: Sequence[str] = ('6D', '5D', '4D', '3D', '2D', '1D')
) -> Dict[str, List[str]]:
    """載入固定數量的選手棋譜供本地驗證。

    回傳字典：{player_id: [sgf_content, ...]}。
    """
    players = {}
    for lv in levels:
        path = os.path.join(train_dir, f'train_{lv}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df_b = df[df['color'] == 'B']
        counts = df_b.groupby('player_id').size()
        for pid, cnt in counts.items():
            if cnt >= min_games and pid not in players:
                sgfs = df_b[df_b['player_id'] == pid]['sgf_content'].tolist()
                players[pid] = sgfs
                if len(players) >= n_players:
                    break
        if len(players) >= n_players:
            break
    return players
