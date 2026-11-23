"""aicup.fuse_player 雙軌融合模組之單元測試。"""

import math
import os
import tempfile
import numpy as np
import pytest

from aicup.fuse_player import (
    apply_csls_column_penalty,
    rank_overlap,
    rank_gaussian_similarity,
    compute_rank_compatibility,
    predict_split_fused,
    TOP_N
)
from aicup.pattern_player import build_chi2_pattern_index, extract_game_local_patterns, score_query_patterns

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


def test_predict_split_fused_with_pattern_track():
    # 所有候選人開局與棋力完全相同，僅局部棋形區分目標候選人 3
    query_sgf = "(;SZ[19];B[aa];W[ab];B[ba];W[bb];B[ca])"
    other_sgf = "(;SZ[19];B[jj];W[jk];B[kj];W[kk];B[lj])"
    q_pat = extract_game_local_patterns(query_sgf, "b")
    o_pat = extract_game_local_patterns(other_sgf, "b")

    cand_profiles = {f"pub_p_{i:04d}": {} for i in range(1, 7)}
    cand_counters = {
        f"pub_p_{i:04d}": (q_pat if i == 3 else o_pat)
        for i in range(1, 7)
    }
    pat_index = build_chi2_pattern_index(cand_counters, min_df=1, eps=1e-3)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        temp_test_csv = f.name
        f.write("question_id,num_games,sgf_1,color_1\n")
        f.write(f'pub_q_0001,1,"{query_sgf}",B\n')

    try:
        # pattern_beta=0 時不啟用棋形軌，字母序第一名為 pub_p_0001
        res_off = predict_split_fused(
            temp_test_csv, cand_profiles, {},
            pattern_index=pat_index, pattern_beta=0.0,
        )
        assert res_off[0][1] == "pub_p_0001"

        # pattern_beta=0.7 時棋形完全吻合的 pub_p_0003 應排第一名
        res_on = predict_split_fused(
            temp_test_csv, cand_profiles, {},
            pattern_index=pat_index, pattern_beta=0.7,
        )
        assert res_on[0][1] == "pub_p_0003"

        # 所有候選人都被硬遮罩剔除時，兜底排序亦應採納 pattern_beta * pat_sim
        cand_ranks = {f"pub_p_{i:04d}": {"expected_rank": 9.0, "probs": [0.0] * 9 + [1.0]} for i in range(1, 7)}
        query_ranks = {"pub_q_0001": {"expected_rank": 0.0, "probs": [1.0] + [0.0] * 9}}
        res_pruned = predict_split_fused(
            temp_test_csv, cand_profiles, {},
            cand_ranks=cand_ranks, query_ranks=query_ranks,
            delta_thresh=3.0, pattern_index=pat_index, pattern_beta=0.7,
        )
        assert res_pruned[0][1] == "pub_p_0003"
    finally:
        if os.path.exists(temp_test_csv):
            os.remove(temp_test_csv)


def test_apply_csls_column_penalty_and_hub_suppression():
    # 候選 0 為全場高分樞紐，候選 1 僅對第 0 題高分
    mat = np.array([
        [0.80, 0.78],
        [0.80, 0.10],
        [0.80, 0.10],
    ], dtype=np.float64)
    adj = apply_csls_column_penalty(mat, csls_k=2)
    # 候選 0 前 2 大平均為 0.80（-> 扣 0.40 -> 0.40）；候選 1 前 2 大平均為 0.44（-> 扣 0.22 -> 0.56）
    assert adj[0, 0] == pytest.approx(0.40, abs=1e-12)
    assert adj[0, 1] == pytest.approx(0.56, abs=1e-12)
    assert adj[0, 1] > adj[0, 0]

    # 含 -inf 遮罩元素時只對有限值計算欄位樞紐分並保留 -inf
    mat_mask = np.array([
        [0.80, float("-inf")],
        [0.60, 0.50],
        [float("-inf"), 0.30],
    ], dtype=np.float64)
    adj_mask = apply_csls_column_penalty(mat_mask, csls_k=2)
    assert math.isinf(adj_mask[0, 1]) and adj_mask[0, 1] < 0
    assert math.isinf(adj_mask[2, 0]) and adj_mask[2, 0] < 0
    assert adj_mask[0, 0] == pytest.approx(0.80 - 0.5 * 0.70, abs=1e-12)
    assert adj_mask[1, 1] == pytest.approx(0.50 - 0.5 * 0.40, abs=1e-12)



def test_predict_split_fused_csls_matches_matrix_ranking():
    # CSLS 路徑的排序應等於整批分數矩陣扣欄樞紐分後逐列排序
    sgfs = [
        "(;SZ[19];B[aa];W[ab];B[ba];W[bb];B[ca])",
        "(;SZ[19];B[jj];W[jk];B[kj];W[kk];B[lj])",
        "(;SZ[19];B[dd];W[dc];B[cd];W[cc];B[ed])",
    ]
    pats = [extract_game_local_patterns(s, "b") for s in sgfs]
    cand_counters = {
        "pub_p_0001": pats[0] + pats[1],
        "pub_p_0002": pats[0] + pats[0] + pats[2],
        "pub_p_0003": pats[2],
        "pub_p_0004": pats[1] + pats[2],
        "pub_p_0005": pats[1],
        "pub_p_0006": pats[0] + pats[2] + pats[2],
    }
    pat_index = build_chi2_pattern_index(cand_counters, min_df=1, eps=1e-3)
    cand_names = sorted(cand_counters)
    cand_profiles = {pid: {} for pid in cand_names}

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        temp_test_csv = f.name
        f.write("question_id,num_games,sgf_1,color_1\n")
        for qi, s in enumerate(sgfs, start=1):
            f.write(f'pub_q_{qi:04d},1,"{s}",B\n')

    try:
        res = predict_split_fused(
            temp_test_csv, cand_profiles, {},
            pattern_index=pat_index, pattern_beta=1.0, csls_k=2,
        )
        mat = np.array([
            [score_query_patterns(pat_index, p)[pid] for pid in cand_names] for p in pats
        ])
        adj = apply_csls_column_penalty(mat, 2)
        for qi, row in enumerate(res):
            expected = sorted(range(len(cand_names)), key=lambda i: (-adj[qi, i], cand_names[i]))[:TOP_N]
            assert row[1:] == [cand_names[i] for i in expected]
    finally:
        if os.path.exists(temp_test_csv):
            os.remove(temp_test_csv)


def test_predict_split_fused_seq_track():
    # 序列軌關閉時結果不變，開啟時依序列相似度排序
    cand_names = [f"pub2_p_{i:04d}" for i in range(1, 7)]
    cand_profiles = {pid: {} for pid in cand_names}
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(8, 8)))[0][:6]
    cand_seq = {pid: np.vstack([basis[i]] * 3) for i, pid in enumerate(cand_names)}
    query_seq = {"pub2_q_0001": basis[4][None, :], "pub2_q_0002": basis[1][None, :]}

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        temp_test_csv = f.name
        f.write("question_id,num_games,sgf_1,color_1\n")
        f.write(f'pub2_q_0001,1,"{SAMPLE_SGF}",B\n')
        f.write(f'pub2_q_0002,1,"{SAMPLE_SGF}",B\n')

    try:
        base = predict_split_fused(temp_test_csv, cand_profiles, {}, csls_k=5)
        off = predict_split_fused(
            temp_test_csv, cand_profiles, {}, csls_k=5,
            seq_tracks=[(cand_seq, query_seq, 1.0)], seq_gamma=0.0,
        )
        on = predict_split_fused(
            temp_test_csv, cand_profiles, {}, csls_k=5, emb_top2_weight=0.5,
            seq_tracks=[(cand_seq, query_seq, 1.0)], seq_gamma=1.0,
        )
        assert off == base
        assert on[0][1] == "pub2_p_0005"
        assert on[1][1] == "pub2_p_0002"
    finally:
        if os.path.exists(temp_test_csv):
            os.remove(temp_test_csv)


def test_predict_split_fused_open_weight(monkeypatch):
    # 開局權重 1.0 與預設相同；權重 0 時只依序列軌排序
    from aicup import fuse_player
    cand_names = [f"pub2_p_{i:04d}" for i in range(1, 7)]
    cand_profiles = {pid: {"style": 6 - i} for i, pid in enumerate(cand_names)}
    monkeypatch.setattr(fuse_player, "compute_style_similarity",
                        lambda q, prof, color_cond=False: 10.0 * prof["style"])
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(8, 8)))[0][:6]
    cand_seq = {pid: np.vstack([basis[i]] * 3) for i, pid in enumerate(cand_names)}
    query_seq = {"pub2_q_0001": basis[4][None, :]}

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        temp_test_csv = f.name
        f.write("question_id,num_games,sgf_1,color_1\n")
        f.write(f'pub2_q_0001,1,"{SAMPLE_SGF}",B\n')

    try:
        kw = dict(csls_k=0, alpha=0.0, seq_tracks=[(cand_seq, query_seq, 1.0)], seq_gamma=1.0)
        default = predict_split_fused(temp_test_csv, cand_profiles, {}, **kw)
        one = predict_split_fused(temp_test_csv, cand_profiles, {}, open_weight=1.0, **kw)
        zero = predict_split_fused(temp_test_csv, cand_profiles, {}, open_weight=0.0, **kw)
        assert one == default
        assert default[0][1] == "pub2_p_0001"
        assert zero[0][1] == "pub2_p_0005"
    finally:
        if os.path.exists(temp_test_csv):
            os.remove(temp_test_csv)


def test_run_full_fused_pipeline_rejects_bad_pub2_weights():
    from aicup.fuse_player import run_full_fused_pipeline
    with pytest.raises(ValueError, match="pub2_seq_weights"):
        run_full_fused_pipeline(seq_cache_dir=["a", "b"], seq_gamma=1.0, pub2_seq_weights=[1.0])
    with pytest.raises(ValueError, match="pub2_seq_weights"):
        run_full_fused_pipeline(seq_cache_dir=["a"], seq_gamma=1.0, pub2_seq_weights=[-1.0])
    with pytest.raises(ValueError, match="pub2_open_weight"):
        run_full_fused_pipeline(pub2_open_weight=-0.5)
