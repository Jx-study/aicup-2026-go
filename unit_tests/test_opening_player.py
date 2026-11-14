"""開局指紋選手辨識的單元測試。"""
import os
import tempfile
import pytest

from aicup.opening_player import (
    TOP_N,
    extract_raw_counts,
    get_canonical_moves,
    predict_split,
)

SAMPLE_SGF = "(;SZ[19]KM[6.5]RU[Japanese];B[dp];W[pd];B[qp];W[dd];B[oq];W[qf];B[fc];W[cf])"


def test_get_canonical_moves_black():
    moves = get_canonical_moves(SAMPLE_SGF, target_color='b', max_moves=10)
    assert len(moves) == 4  # 黑棋共 4 手
    col, step, r, c, opp = moves[0]
    assert col == 'b'
    assert step == 0
    # 第一手旋轉至標準八分區
    assert 0 <= r <= 9
    assert 0 <= c <= 9
    assert r <= c
    assert opp is None  # 黑棋第一手無前手對手落子


def test_get_canonical_moves_white():
    moves = get_canonical_moves(SAMPLE_SGF, target_color='w', max_moves=10)
    assert len(moves) == 4  # 白棋共 4 手
    col, step, r, c, opp = moves[0]
    assert col == 'w'
    assert step == 0
    assert 0 <= r <= 9
    assert 0 <= c <= 9
    assert r <= c
    assert opp is not None  # 白棋第一手前有黑棋首手


def test_get_canonical_moves_empty_and_invalid():
    assert get_canonical_moves("", target_color='b') == []
    assert get_canonical_moves("invalid sgf content", target_color='w') == []
    assert get_canonical_moves("(;SZ[19])", target_color='b') == []


def test_get_canonical_moves_short():
    short_sgf = "(;SZ[19];B[pd];W[dp])"
    moves_b = get_canonical_moves(short_sgf, target_color='b', max_moves=10)
    assert len(moves_b) == 1
    moves_w = get_canonical_moves(short_sgf, target_color='w', max_moves=10)
    assert len(moves_w) == 1


def test_extract_raw_counts():
    moves = get_canonical_moves(SAMPLE_SGF, target_color='b', max_moves=10)
    cnt = extract_raw_counts([moves])
    assert len(cnt) > 0


def test_predict_split_mock():
    # 建立 6 位測試候選人特徵與 IDF
    cand_profiles = {}
    idf_weights = {}
    for i in range(1, 7):
        pid = f"pub_p_{i:04d}"
        cand_profiles[pid] = {('b', 0, 3, 3 + (i % 3)): 1.0}

    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
        temp_test_csv = f.name
        f.write("question_id,num_games,sgf_1,color_1\n")
        f.write(f"pub_q_0001,1,\"{SAMPLE_SGF}\",B\n")

    try:
        results = predict_split(temp_test_csv, cand_profiles, idf_weights, top_n=TOP_N)
        assert len(results) == 1
        row = results[0]
        assert row[0] == "pub_q_0001"
        top5 = row[1:]
        assert len(top5) == TOP_N
        assert len(set(top5)) == TOP_N
        for p in top5:
            assert p.startswith("pub_p_")
    finally:
        if os.path.exists(temp_test_csv):
            os.remove(temp_test_csv)
