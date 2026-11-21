"""執色分離開局指紋與雙軌融合的單元測試。"""

import math
import pandas as pd

from aicup.opening_player import (
    build_candidate_profiles_and_idf,
    build_candidate_profiles_from_moves,
    build_query_vectors,
    compute_style_similarity,
    get_canonical_moves,
)
from aicup.fuse_player import predict_split_fused


SGF_B_STAR = "(;SZ[19]HD[0];B[pd];W[dd];B[pp];W[dp];B[qf];W[nc])"
SGF_B_KOMOKU = "(;SZ[19]HD[0];B[qd];W[dc];B[pq];W[cp];B[oc];W[pe])"
SGF_W_STAR = "(;SZ[19]HD[0];B[qd];W[dd];B[pq];W[dp];B[oc];W[qf])"


def test_all_black_color_cond_matches_flat():
    """全黑候選與全黑題目下，分色模式與原版結果完全一致。"""
    m1 = get_canonical_moves(SGF_B_STAR, "b")
    m2 = get_canonical_moves(SGF_B_KOMOKU, "b")
    games = {"p1": [("b", m1)], "p2": [("b", m2)]}

    prof_flat, idf_flat = build_candidate_profiles_from_moves(games, color_cond=False)
    prof_cc, idf_cc = build_candidate_profiles_from_moves(games, color_cond=True)
    assert idf_flat == idf_cc

    q_flat = build_query_vectors([("b", m1)], idf_flat, color_cond=False)
    q_cc = build_query_vectors([("b", m1)], idf_cc, color_cond=True)

    for pid in ("p1", "p2"):
        s_flat = compute_style_similarity(q_flat, prof_flat[pid], color_cond=False)
        s_cc = compute_style_similarity(q_cc, prof_cc[pid], color_cond=True)
        assert math.isclose(s_flat, s_cc, rel_tol=1e-9, abs_tol=1e-9)


def test_missing_color_fallback_to_all():
    """候選人缺少白棋盤面時，白棋向量自動回退至全局向量。"""
    m1 = get_canonical_moves(SGF_B_STAR, "b")
    m2 = get_canonical_moves(SGF_W_STAR, "w")
    games = {"p_only_b": [("b", m1)], "p_both": [("b", m1), ("w", m2)]}

    prof_cc, idf = build_candidate_profiles_from_moves(games, color_cond=True)
    assert prof_cc["p_only_b"]["w"] == prof_cc["p_only_b"]["all"]
    assert prof_cc["p_both"]["w"] != prof_cc["p_both"]["all"]

    q_w = build_query_vectors([("w", m2)], idf, color_cond=True)
    sim_both = compute_style_similarity(q_w, prof_cc["p_both"], color_cond=True)
    sim_only_b = compute_style_similarity(q_w, prof_cc["p_only_b"], color_cond=True)
    assert sim_both > sim_only_b


def test_mixed_query_color_weighting():
    """混合顏色題目依黑、白局數比例加權合併相似度。"""
    mb = get_canonical_moves(SGF_B_STAR, "b")
    mw = get_canonical_moves(SGF_W_STAR, "w")
    games = {"p1": [("b", mb), ("w", mw)], "p2": [("b", get_canonical_moves(SGF_B_KOMOKU, "b"))]}

    prof_cc, idf = build_candidate_profiles_from_moves(games, color_cond=True)
    q_mix = build_query_vectors([("b", mb), ("b", mb), ("w", mw)], idf, color_cond=True)
    q_b = build_query_vectors([("b", mb)], idf, color_cond=True)
    q_w = build_query_vectors([("w", mw)], idf, color_cond=True)

    s_mix = compute_style_similarity(q_mix, prof_cc["p1"], color_cond=True)
    s_expected = (2 / 3) * compute_style_similarity(q_b, prof_cc["p1"], color_cond=True) + (
        1 / 3
    ) * compute_style_similarity(q_w, prof_cc["p1"], color_cond=True)
    assert math.isclose(s_mix, s_expected, rel_tol=1e-9, abs_tol=1e-9)


def test_empty_color_vector_renormalizes_weight():
    """某一執色特徵為空 dict 時，有效權重應重新正規化而不壓低相似度。"""
    mb = get_canonical_moves(SGF_B_STAR, "b")
    games = {"p1": [("b", mb)]}
    prof_cc, idf = build_candidate_profiles_from_moves(games, color_cond=True)

    q_b_only = build_query_vectors([("b", mb)], idf, color_cond=True)
    q_with_empty_w = build_query_vectors([("b", mb), ("w", [])], idf, color_cond=True)

    s_b = compute_style_similarity(q_b_only, prof_cc["p1"], color_cond=True)
    s_empty = compute_style_similarity(q_with_empty_w, prof_cc["p1"], color_cond=True)
    assert math.isclose(s_b, s_empty, rel_tol=1e-9, abs_tol=1e-9)


def test_predict_split_fused_color_cond(tmp_path):
    """雙軌融合推論開啟分色與純高斯模式時能正確排序且無缺失。"""
    cand_df = pd.DataFrame([
        {"player_id": "p1", "sgf_content": SGF_B_STAR, "color": "B"},
        {"player_id": "p1", "sgf_content": SGF_W_STAR, "color": "W"},
        {"player_id": "p2", "sgf_content": SGF_B_KOMOKU, "color": "B"},
        {"player_id": "p3", "sgf_content": SGF_B_KOMOKU, "color": "W"},
        {"player_id": "p4", "sgf_content": SGF_B_STAR, "color": "W"},
        {"player_id": "p5", "sgf_content": SGF_W_STAR, "color": "B"},
    ])
    test_df = pd.DataFrame([
        {"question_id": "q1", "num_games": 2, "sgf_1": SGF_B_STAR, "color_1": "B", "sgf_2": SGF_W_STAR, "color_2": "W"},
    ])
    cand_csv = tmp_path / "cands.csv"
    test_csv = tmp_path / "test.csv"
    cand_df.to_csv(cand_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    profs, idf = build_candidate_profiles_and_idf(str(cand_csv), color_cond=True)
    cand_ranks = {f"p{i}": {"expected_rank": float(i), "probs": [0.1] * 10} for i in range(1, 6)}
    query_ranks = {"q1": {"expected_rank": 1.0, "probs": [0.1] * 10}}

    rows = predict_split_fused(
        str(test_csv), profs, idf,
        cand_ranks=cand_ranks, query_ranks=query_ranks,
        top_n=5, alpha=0.2, delta_thresh=0.0, rank_metric="gaussian", sigma=1.5,
        color_cond=True
    )
    assert len(rows) == 1
    assert rows[0][0] == "q1"
    assert rows[0][1] == "p1"
    assert len(set(rows[0][1:])) == 5
