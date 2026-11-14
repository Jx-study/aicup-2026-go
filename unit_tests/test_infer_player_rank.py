"""aicup.infer_player_rank 棋力推論模組之單元測試。"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from aicup.infer_player_rank import (
    compute_expected_rank,
    aggregate_probs,
    predict_sgf_batch,
    infer_candidates_from_df,
    infer_questions_from_df,
    save_rank_cache,
    load_rank_cache,
    RANK_CLASSES
)

SAMPLE_SGF = "(;SZ[19]KM[6.5]RU[Japanese];B[dp];W[pd];B[qp];W[dd])"


class DummyRankModel(torch.nn.Module):
    """用於測試的輕量 Dummy 棋力模型。"""
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.size(0)
        # 固定輸出 logits: 傾向於 3D (index 6) 與 4D (index 7)
        logits = torch.zeros((batch_size, RANK_CLASSES), dtype=torch.float32)
        logits[:, 6] = 2.0
        logits[:, 7] = 2.0
        return logits


def test_compute_expected_rank_1d():
    # 類別 D (index 0) 獨佔
    p0 = np.zeros(10)
    p0[0] = 1.0
    assert compute_expected_rank(p0) == 0.0

    # 類別 6D (index 9) 獨佔
    p9 = np.zeros(10)
    p9[9] = 1.0
    assert compute_expected_rank(p9) == 9.0

    # 均勻分佈（0+1+...+9 = 45 -> 4.5）
    pu = np.full(10, 0.1)
    assert pytest.approx(compute_expected_rank(pu), rel=1e-5) == 4.5


def test_compute_expected_rank_2d():
    p = np.zeros((2, 10))
    p[0, 1] = 1.0  # index 1
    p[1, 5] = 1.0  # index 5
    res = compute_expected_rank(p)
    assert len(res) == 2
    assert res[0] == 1.0
    assert res[1] == 5.0


def test_aggregate_probs():
    p1 = np.zeros(10)
    p1[4] = 1.0  # 1D
    p2 = np.zeros(10)
    p2[6] = 1.0  # 3D

    agg_mean = aggregate_probs([p1, p2], agg="mean")
    assert pytest.approx(agg_mean.sum(), rel=1e-5) == 1.0
    assert pytest.approx(agg_mean[4], rel=1e-5) == 0.5
    assert pytest.approx(agg_mean[6], rel=1e-5) == 0.5

    # 測試空陣列 fallback
    empty_agg = aggregate_probs([])
    assert len(empty_agg) == 10
    assert pytest.approx(empty_agg.sum(), rel=1e-5) == 1.0


def test_predict_sgf_batch():
    model = DummyRankModel()
    device = torch.device("cpu")
    sgfs = [SAMPLE_SGF, "", "invalid sgf content", SAMPLE_SGF]

    probs = predict_sgf_batch(model, sgfs, device, batch_size=2)
    assert probs.shape == (4, 10)
    for p in probs:
        assert pytest.approx(p.sum(), rel=1e-4) == 1.0

    # 無效局預設為均勻分佈
    assert pytest.approx(probs[1, 0], rel=1e-4) == 0.1
    assert pytest.approx(probs[2, 0], rel=1e-4) == 0.1

    # 有效局應由 DummyModel 預測出 index 6 與 7 機率偏高
    assert probs[0, 6] > probs[0, 0]


def test_infer_candidates_from_df():
    model = DummyRankModel()
    device = torch.device("cpu")

    df = pd.DataFrame({
        "game_id": [1, 2, 3],
        "player_id": ["p_001", "p_001", "p_002"],
        "sgf_content": [SAMPLE_SGF, SAMPLE_SGF, SAMPLE_SGF],
        "color": ["B", "B", "W"]
    })

    cands = infer_candidates_from_df(model, df, device, max_games_per_player=1, batch_size=2, verbose=False)
    assert "p_001" in cands
    assert "p_002" in cands
    assert cands["p_001"]["num_games"] == 1
    assert len(cands["p_001"]["probs"]) == 10
    assert 0.0 <= cands["p_001"]["expected_rank"] <= 9.0


def test_infer_questions_from_df():
    model = DummyRankModel()
    device = torch.device("cpu")

    df = pd.DataFrame({
        "question_id": ["q_001", "q_002"],
        "num_games": [2, 1],
        "sgf_1": [SAMPLE_SGF, SAMPLE_SGF],
        "sgf_2": [SAMPLE_SGF, ""]
    })

    queries = infer_questions_from_df(model, df, device, batch_size=2, verbose=False)
    assert "q_001" in queries
    assert "q_002" in queries
    assert queries["q_001"]["num_games"] == 2
    assert queries["q_002"]["num_games"] == 1
    assert len(queries["q_001"]["probs"]) == 10


def test_save_and_load_cache():
    cache_data = {
        "p_001": {
            "probs": [0.1] * 10,
            "expected_rank": 4.5,
            "pred_rank": "1D",
            "num_games": 5
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_cache.json")
        save_rank_cache(cache_data, json_path)
        assert os.path.exists(json_path)

        loaded = load_rank_cache(json_path)
        assert loaded == cache_data
