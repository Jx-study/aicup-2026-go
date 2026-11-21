"""競賽一目標側 Human SL 棋力推論與快取建置之單元測試。"""

import json
import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aicup.infer_player_hsl import (
    EXPECTED_PROFILES,
    aggregate_entity_probs,
    build_target_features,
    combine_log_mean,
    fit_target_hsl_model,
    infer_from_jsonl,
)


def _make_dummy_model():
    """建立可重現的合成 7 欄目標側邏輯回歸模型。"""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 7))
    y = np.tile(np.arange(10), 20)
    sc = StandardScaler()
    clf = LogisticRegression(C=0.5, max_iter=500)
    clf.fit(sc.fit_transform(X), y)
    return sc, clf


def test_aggregate_entity_probs_log_mean():
    """多局機率對數平均合併應回傳合法分佈、期望段位與局數。"""
    p1 = np.array([0.01] * 4 + [0.46, 0.46] + [0.01] * 4)
    p2 = np.array([0.01] * 5 + [0.91] + [0.01] * 4)
    rec = aggregate_entity_probs(np.vstack([p1, p2]))
    assert rec["num_games"] == 2
    assert rec["pred_rank"] == "2D"
    assert math.isclose(sum(rec["probs"]), 1.0, abs_tol=1e-4)
    assert 4.0 < rec["expected_rank"] < 6.0


def test_infer_from_jsonl_splits_cand_and_query(tmp_path):
    """jsonl 推論後應依 _p_ 正確拆分候選人與題目快取。"""
    sc, clf = _make_dummy_model()
    rows = [
        {
            "key": "cand2_pub_g1",
            "rank_idx": -1,
            "player_id": "pub2_p_0001",
            "color": "B",
            "hd": 0,
            "profiles": list(EXPECTED_PROFILES),
            "turns": [0, 2, 4],
            "probs": [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.1, 0.2, 0.3]],
        },
        {
            "key": "cand2_pub_g2",
            "rank_idx": -1,
            "player_id": "pub2_p_0001",
            "color": "W",
            "hd": 1,
            "profiles": list(EXPECTED_PROFILES),
            "turns": [1, 3, 5],
            "probs": [[0.25, 0.35, 0.45], [0.35, 0.45, 0.55], [0.15, 0.25, 0.35]],
        },
        {
            "key": "pub2_q_0001_g01",
            "rank_idx": -1,
            "player_id": "pub2_q_0001",
            "color": "B",
            "hd": 0,
            "profiles": list(EXPECTED_PROFILES),
            "turns": [0, 2, 4],
            "probs": [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.1, 0.2, 0.3]],
        },
    ]
    p = tmp_path / "pub2.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cand_ranks, query_ranks = infer_from_jsonl(str(p), sc, clf)
    assert list(cand_ranks.keys()) == ["pub2_p_0001"]
    assert list(query_ranks.keys()) == ["pub2_q_0001"]
    assert cand_ranks["pub2_p_0001"]["num_games"] == 2
    assert query_ranks["pub2_q_0001"]["num_games"] == 1


def test_infer_from_jsonl_rejects_wrong_profiles(tmp_path):
    """profiles 順序或名稱不符時應丟出 ValueError。"""
    sc, clf = _make_dummy_model()
    p = tmp_path / "bad.jsonl"
    row = {
        "key": "k1",
        "rank_idx": -1,
        "player_id": "pub2_p_0001",
        "color": "B",
        "hd": 0,
        "profiles": ["rank_11k", "rank_1d", "rank_2d"],
        "turns": [0, 2],
        "probs": [[0.2, 0.3], [0.3, 0.4], [0.1, 0.2]],
    }
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        infer_from_jsonl(str(p), sc, clf)


def test_fit_target_hsl_model_filters_hd(tmp_path):
    """擬合時應排除 HD >= 2 的局並統計玩家數。"""
    rng = np.random.default_rng(0)
    n = 300
    hd = np.where(np.arange(n) % 7 == 0, 2, np.arange(n) % 2)
    path = tmp_path / "c.npz"
    np.savez(
        path,
        g_hsl=rng.normal(size=(n, 4)),
        d_black=rng.integers(0, 2, n),
        d_hd=hd,
        d_rank_idx=np.arange(n) % 10,
        d_player_id=np.array(["p%d" % (i % 50) if hd[i] < 2 else "skip" for i in range(n)]),
    )
    scaler, clf, n_players = fit_target_hsl_model([str(path)])
    assert n_players == len({i % 50 for i in range(n) if hd[i] < 2})
    assert clf.coef_.shape == (10, 7)


def test_combine_log_mean_matches_single_game():
    """單局合併應還原該局機率。"""
    p = np.array([[0.05] * 5 + [0.15] * 3 + [0.3 / 2] * 2])
    p = p / p.sum()
    out, exp_r = combine_log_mean(p)
    assert np.allclose(out, p[0])
    assert math.isclose(exp_r, float(np.dot(p[0], np.arange(10))))
