"""aicup.fuse_player 雙軌融合模組之單元測試。"""

import math
import os
import tempfile
import numpy as np
import pytest

from aicup.fuse_player import (
    rank_overlap,
    rank_gaussian_similarity,
    compute_rank_compatibility,
    predict_split_fused,
    TOP_N
)

SAMPLE_SGF = "(;SZ[19]KM[6.5]RU[Japanese];B[dp];W[pd];B[qp];W[dd])"


def test_rank_overlap():
    p1 = [0.0] * 10
    p1[4] = 1.0  # 1D
    p2 = [0.0] * 10
    p2[4] = 1.0  # 1D
    p3 = [0.0] * 10
    p3[9] = 1.0  # 6D

    assert pytest.approx(rank_overlap(p1, p2), rel=1e-5) == 1.0
    assert pytest.approx(rank_overlap(p1, p3), rel=1e-5) == 0.0


def test_rank_gaussian_similarity():
    # 期望段位相同
    assert pytest.approx(rank_gaussian_similarity(5.0, 5.0, sigma=1.5), rel=1e-5) == 1.0

    # 段位差 3.0，sigma=1.5（-> exp(-9 / 4.5) = exp(-2)）
    expected = math.exp(-2.0)
    assert pytest.approx(rank_gaussian_similarity(5.0, 2.0, sigma=1.5), rel=1e-5) == expected


def test_compute_rank_compatibility_mask():
    c_rank = {"expected_rank": 8.0, "probs": [0.0] * 8 + [0.5, 0.5]}  # 高段 5D/6D
    q_rank_high = {"expected_rank": 7.5, "probs": [0.0] * 7 + [0.5, 0.5, 0.0]}  # 4D/5D
    q_rank_low = {"expected_rank": 1.0, "probs": [0.5, 0.5] + [0.0] * 8}  # 級位 D/C

    # 高段題目：段位差 0.5 <= 3.0，通過遮罩
    score, passed = compute_rank_compatibility(c_rank, q_rank_high, delta_thresh=3.0)
    assert passed is True
    assert not math.isinf(score)
    assert score >= 0.0

    # 級位題目：段位差 7.0 > 3.0，硬性遮罩剔除
    score_low, passed_low = compute_rank_compatibility(c_rank, q_rank_low, delta_thresh=3.0)
    assert passed_low is False
    assert math.isinf(score_low)
    assert score_low < 0


def test_compute_rank_compatibility_missing():
    # 缺候選人或題目棋力時，應優雅放行且不加權
    score, passed = compute_rank_compatibility(None, {"expected_rank": 5.0})
    assert passed is True
    assert score == 0.0

    score2, passed2 = compute_rank_compatibility({"expected_rank": 5.0}, None)
    assert passed2 is True
    assert score2 == 0.0


def test_predict_split_fused_mock():
    # 建立 6 位測試候選人特徵與棋力
    cand_profiles = {}
    cand_ranks = {}
    idf_weights = {}

    for i in range(1, 7):
        pid = f"pub_p_{i:04d}"
        # 開局特徵：候選人 1 與題目完全相同，但棋力差距極大（8.5 vs 1.0）
        # 候選人 2 開局特徵次佳，且棋力完全吻合（1.2 vs 1.0）
        cand_profiles[pid] = {("b", 0, 3, 3): 1.0 if i == 1 else (0.8 if i == 2 else 0.1)}
        if i == 1:
            cand_ranks[pid] = {"expected_rank": 8.5, "probs": [0.0] * 8 + [0.5, 0.5]}
        elif i == 2:
            cand_ranks[pid] = {"expected_rank": 1.2, "probs": [0.5, 0.5] + [0.0] * 8}
        else:
            # 其他候選人段位與題目接近（2.0 vs 1.0），皆在容許閥值內
            cand_ranks[pid] = {"expected_rank": 2.0, "probs": [0.2] * 5 + [0.0] * 5}

    query_ranks = {
        "pub_q_0001": {"expected_rank": 1.0, "probs": [0.6, 0.4] + [0.0] * 8}
    }

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        temp_test_csv = f.name
        f.write("question_id,num_games,sgf_1,color_1\n")
        f.write(f"pub_q_0001,1,\"{SAMPLE_SGF}\",B\n")

    try:
        results = predict_split_fused(
            temp_test_csv,
            cand_profiles,
            idf_weights,
            cand_ranks=cand_ranks,
            query_ranks=query_ranks,
            top_n=TOP_N,
            alpha=1.0,
            delta_thresh=3.0,
            rank_metric="overlap"
        )
        assert len(results) == 1
        row = results[0]
        qid = row[0]
        top5 = row[1:]

        assert qid == "pub_q_0001"
        assert len(top5) == TOP_N
        # 候選人 1 雖然開局特徵最高，但因段位差 7.5 > 3.0 被硬性剔除
        # 候選人 2 棋力吻合且特徵佳，排在第一名
        assert top5[0] == "pub_p_0002"
        assert "pub_p_0001" not in top5[:3]
    finally:
        if os.path.exists(temp_test_csv):
            os.remove(temp_test_csv)
