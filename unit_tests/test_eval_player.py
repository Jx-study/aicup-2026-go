"""選手辨識評估模組的單元測試。"""
import math
import pytest
from aicup.eval_player import score_player_ranks, evaluate_predictions


def test_score_player_ranks():
    # 驗證前 5 名指數遞減與未命中得 0 分
    ranks = [1, 2, 3, 4, 5, 6, 0]
    expected = [
        1.0,
        math.exp(-1),
        math.exp(-2),
        math.exp(-3),
        math.exp(-4),
        0.0,
        0.0
    ]
    assert pytest.approx(score_player_ranks(ranks), rel=1e-5) == sum(expected) / len(ranks)


def test_evaluate_predictions():
    ground_truth = {
        'q1': 'p1',
        'q2': 'p2',
        'q3': 'p3',
    }
    predictions = {
        'q1': ['p1', 'p2', 'p3', 'p4', 'p5'],  # 第 1 名得分 1.0
        'q2': ['px', 'p2', 'py', 'pz', 'pw'],  # 第 2 名得分 exp(-1)
        'q3': ['pa', 'pb', 'pc', 'pd', 'pe'],  # 未入前 5 名得 0 分
    }
    res = evaluate_predictions(predictions, ground_truth)
    assert res['total'] == 3
    assert res['top1_acc'] == 1 / 3
    assert res['top5_acc'] == 2 / 3
    expected_score = (1.0 + math.exp(-1) + 0.0) / 3
    assert pytest.approx(res['score'], rel=1e-5) == expected_score
