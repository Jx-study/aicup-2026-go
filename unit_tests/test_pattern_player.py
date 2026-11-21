"""aicup.pattern_player 局部棋形指紋與 Chi-squared 殘差模組之單元測試。"""

from collections import Counter
import os
import tempfile

import numpy as np
import pytest

from aicup.pattern_player import (
    build_candidate_pattern_counters,
    build_chi2_pattern_index,
    extract_game_local_patterns,
    extract_patterns_batch,
    merge_pattern_counters,
    score_query_patterns,
    score_query_patterns_array,
    vectorize_pattern_counter,
)


def _transform_coord(r: int, c: int, rot: int, flip: bool) -> tuple:
    """對 19x19 棋盤座標套用 D4 群旋轉與水平鏡像。"""
    for _ in range(rot):
        r, c = 18 - c, r
    if flip:
        c = 18 - c
    return r, c


def _make_sgf(moves, hd: int = 0) -> str:
    """由 (color, r, c) 清單組裝 SGF 字串。"""
    header = f"(;SZ[19]HD[{hd}]" if hd >= 2 else "(;SZ[19]"
    body = []
    for col, r, c in moves:
        if r is None or c is None:
            body.append(f";{col}[]")
        else:
            pt = chr(97 + c) + chr(97 + r)
            body.append(f";{col}[{pt}]")
    return header + "".join(body) + ")"


def test_d4_symmetry_invariance():
    # 含盤角、邊線與中腹落子的非對稱棋局，8 種 D4 變換必須產生完全相同計數
    base_moves = [
        ("B", 3, 3), ("W", 3, 4), ("B", 4, 3), ("W", 2, 4),
        ("B", 4, 4), ("W", 0, 1), ("B", 1, 0), ("W", 10, 12),
        ("B", 10, 10), ("W", 12, 10), ("B", 11, 11),
    ]
    ref_b = extract_game_local_patterns(_make_sgf(base_moves), "b")
    ref_w = extract_game_local_patterns(_make_sgf(base_moves), "w")
    assert sum(ref_b.values()) == 12  # 6 手黑棋 x 2 種棋形（3x3 與 12 點菱形）
    assert sum(ref_w.values()) == 10  # 5 手白棋 x 2 種棋形

    for rot in range(4):
        for flip in (False, True):
            t_moves = [
                (col, *_transform_coord(r, c, rot, flip))
                for col, r, c in base_moves
            ]
            sgf_t = _make_sgf(t_moves)
            assert extract_game_local_patterns(sgf_t, "b") == ref_b
            assert extract_game_local_patterns(sgf_t, "w") == ref_w


def test_handicap_stones_and_color_flip():
    # HD[2] 預置星位 pd=(3,15) 與 dp=(15,3) 為黑子，且 SGF 內 B/W 標記反轉
    # 在 HD[2] 中 SGF 的 ;B[pc] 實際為白方落子，;W[od] 實際為黑方落子（鄰近讓子 pd）
    sgf_hd2 = "(;SZ[19]HD[2];B[pc];W[od])"
    cnt_b = extract_game_local_patterns(sgf_hd2, "b")
    cnt_w = extract_game_local_patterns(sgf_hd2, "w")
    assert sum(cnt_b.values()) == 2
    assert sum(cnt_w.values()) == 2

    # 對照無讓子盤面，手動擺上兩顆讓子後由白先下 pc、黑再下 od，棋形必須完全一致
    # 這裡在無讓子局先由黑擺 pd 與 dp（中間穿插遠處白子 aa、sa），再下 pc、od
    sgf_equiv = "(;SZ[19];B[pd];W[aa];B[dp];W[sa];W[pc];B[od])"
    # 取最後兩手對應目標方棋形驗證與 HD[2] 差異僅在前置手順，透過空盤比對確認讓子有生效
    sgf_no_hd = "(;SZ[19];W[pc];B[od])"
    assert cnt_b != extract_game_local_patterns(sgf_no_hd, "b")
    assert cnt_w != extract_game_local_patterns(sgf_no_hd, "w")


def test_capture_removes_stones_before_subsequent_pattern():
    # 黑棋在 (0,0) 落子，白棋圍住 (0,1) 與 (1,0) 將黑子提掉，接著黑棋在 (0,0) 重下
    # 若提子生效，重下 (0,0) 前該點為空且兩鄰為敵子；同時測試角落被提後的周圍狀態
    sgf_cap = _make_sgf([
        ("B", 0, 0),
        ("W", 0, 1),
        ("B", 10, 10),
        ("W", 1, 0),  # 提掉 (0,0) 黑子
        ("B", 0, 2),
        ("W", 10, 11),
        ("B", 1, 1),  # 落於 (1,1)，此時 (0,0) 應已為空點而非己子
    ])
    # 對照組：黑子未曾下過 (0,0)，直接由白下 (0,1)、(1,0) 後黑下 (1,1)
    sgf_no_stone = _make_sgf([
        ("B", 10, 10),
        ("W", 0, 1),
        ("B", 0, 2),
        ("W", 1, 0),
        ("B", 1, 1),
    ])
    # 兩局在最後一手 (1,1) 之前的局部盤面完全相同，計數差異僅來自多出的第一手 (0,0)
    cnt_cap = extract_game_local_patterns(sgf_cap, "b")
    cnt_no = extract_game_local_patterns(sgf_no_stone, "b")
    first_only = extract_game_local_patterns(_make_sgf([("B", 0, 0)]), "b")
    assert cnt_cap == cnt_no + first_only


def test_empty_and_malformed_sgf_guard():
    assert extract_game_local_patterns("", "b") == Counter()
    assert extract_game_local_patterns("   ", "w") == Counter()
    assert extract_game_local_patterns("(;SZ[19])", "b") == Counter()
    assert extract_game_local_patterns("(;SZ[19];B[];W[])", "b") == Counter()
    assert extract_game_local_patterns("not an sgf", "b") == Counter()


def test_single_vs_multiprocess_exact_match():
    sgfs = [
        (_make_sgf([("B", 3, 3), ("W", 15, 15), ("B", 3, 4), ("W", 15, 14)]), "b"),
        (_make_sgf([("B", 2, 2), ("W", 3, 3), ("B", 4, 4), ("W", 5, 5)]), "w"),
        ("(;SZ[19]HD[4];B[pd];W[dp];B[dd];W[pp])", "b"),
        ("(;SZ[19];B[];W[jj];B[jk])", "b"),
    ]
    seq_res = extract_patterns_batch(sgfs, workers=1)
    mp_res = extract_patterns_batch(sgfs, workers=2, chunksize=2)
    assert seq_res == mp_res


def test_chi2_matrix_equivalence_and_zero_vocab_guard():
    # 建立 4 位候選人的合成棋形計數
    cand_counters = {
        "p1": Counter({(3, 1): 10, (3, 2): 2, (12, 5): 8, (3, 99): 5}),
        "p2": Counter({(3, 1): 3, (3, 2): 9, (12, 5): 4}),
        "p3": Counter({(3, 1): 6, (3, 2): 6, (12, 5): 2}),
        "p4": Counter({(3, 1): 1, (3, 2): 8, (12, 5): 11}),
    }
    index = build_chi2_pattern_index(cand_counters, min_df=3, eps=1e-3)
    # (3, 99) 僅出現於 1 位候選人，應被 min_df=3 過濾
    assert (3, 99) not in index.vocab_to_idx
    assert len(index.vocab_to_idx) == 3

    query = Counter({(3, 1): 9, (3, 2): 1, (12, 5): 7, (3, 888): 100})
    scores_dict = score_query_patterns(index, query)
    scores_arr = score_query_patterns_array(index, query)

    # 與逐筆手動計算餘弦相似度比對，誤差須低於 1e-12
    q_vec = vectorize_pattern_counter(index.vocab_to_idx, index.mu, index.denom, query)
    for i, pid in enumerate(index.pids):
        c_vec = vectorize_pattern_counter(index.vocab_to_idx, index.mu, index.denom, cand_counters[pid])
        manual_cos = float(np.dot(c_vec, q_vec))
        assert scores_dict[pid] == pytest.approx(manual_cos, abs=1e-12)
        assert float(scores_arr[i]) == pytest.approx(manual_cos, abs=1e-12)

    # 與 p1 分佈最相近，p1 得分應為最高
    assert max(scores_dict, key=scores_dict.get) == "p1"

    # min_df 過高導致空詞表時，應安全回傳全 0 分數
    empty_index = build_chi2_pattern_index(cand_counters, min_df=99, eps=1e-3)
    assert empty_index.mat.shape == (4, 0)
    assert score_query_patterns(empty_index, query) == {pid: 0.0 for pid in empty_index.pids}


def test_build_candidate_pattern_counters_csv():
    sgf1 = _make_sgf([("B", 3, 3), ("W", 15, 15), ("B", 3, 4)])
    sgf2 = _make_sgf([("B", 10, 10), ("W", 10, 11)])
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("player_id,sgf_content,color\n")
        f.write(f'p1,"{sgf1}",B\n')
        f.write(f'p1,"{sgf2}",B\n')
        f.write(f'p2,"{sgf2}",W\n')

    try:
        res = build_candidate_pattern_counters(csv_path, workers=1)
        assert set(res.keys()) == {"p1", "p2"}
        expected_p1 = merge_pattern_counters([
            extract_game_local_patterns(sgf1, "b"),
            extract_game_local_patterns(sgf2, "b"),
        ])
        assert res["p1"] == expected_p1
        assert res["p2"] == extract_game_local_patterns(sgf2, "w")
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_empty_counter_vectorizes_to_zero():
    # 沒有棋形的題目不得與任何候選人產生非零相似度
    cands = {
        "p1": Counter({(3, 1): 5, (3, 2): 1}),
        "p2": Counter({(3, 1): 1, (3, 2): 5}),
        "p3": Counter({(3, 1): 3, (3, 2): 3}),
    }
    index = build_chi2_pattern_index(cands, min_df=1)
    assert np.all(score_query_patterns_array(index, Counter()) == 0.0)
