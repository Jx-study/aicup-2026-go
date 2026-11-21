"""競賽一 pub2/priv2 Human SL 清單與 analyze_humansl --manifest 載入測試。"""

import pandas as pd
import pytest

from aicup.analyze_humansl import load_manifest_games
from aicup.prep_player_hsl import build_pub2_manifest, extract_hd


SGF0 = "(;SZ[19]HD[0];B[pd];W[dd];B[pp];W[dp])"
SGF1 = "(;SZ[19]HD[1];B[pd];W[dd];B[pp];W[dp])"


def test_extract_hd():
    """能正確解析 HD 標籤與缺漏情形。"""
    assert extract_hd(SGF0) == 0
    assert extract_hd(SGF1) == 1
    assert extract_hd("(;SZ[19];B[pd])") == -1


def test_build_pub2_manifest_and_load(tmp_path):
    """產出的清單 key 唯一、欄位完整，且可被 load_manifest_games 讀取。"""
    cand_df = pd.DataFrame([
        {"game_id": "g1", "player_id": "p1", "sgf_content": SGF0, "color": "B"},
        {"game_id": "g2", "player_id": "p1", "sgf_content": SGF1, "color": "W"},
    ])
    test_df = pd.DataFrame([
        {"question_id": "pub2_q1", "num_games": 2, "sgf_1": SGF0, "color_1": "B", "sgf_2": SGF1, "color_2": "W"},
    ])
    for fname in ("player_identification_candidates2_public.csv", "player_identification_candidates2_private.csv"):
        cand_df.to_csv(tmp_path / fname, index=False)
    for fname, qid in (
        ("player_identification_test2_public.csv", "pub2_q1"),
        ("player_identification_test2_private.csv", "priv2_q1"),
    ):
        df = test_df.copy()
        df["question_id"] = qid
        df.to_csv(tmp_path / fname, index=False)

    out_df, hd_counts = build_pub2_manifest(str(tmp_path))
    assert len(out_df) == 8
    assert out_df["key"].nunique() == 8
    assert hd_counts == {0: 4, 1: 4}

    m_csv = tmp_path / "manifest.csv"
    out_df.to_csv(m_csv, index=False)
    games = load_manifest_games(str(m_csv), limit=3)
    assert len(games) == 3
    assert set(games[0].keys()) == {"key", "rank_idx", "player_id", "color", "sgf"}
    assert games[0]["rank_idx"] == -1


def test_load_manifest_games_rejects_invalid_color(tmp_path):
    """清單含非法顏色時應於讀檔階段拒絕。"""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([
        {"key": "k1", "entity_id": "p1", "color": "X", "sgf_content": SGF0},
    ]).to_csv(bad_csv, index=False)
    with pytest.raises(SystemExit):
        load_manifest_games(str(bad_csv))
