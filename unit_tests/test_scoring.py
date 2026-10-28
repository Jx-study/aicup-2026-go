import numpy as np
import pytest

from aicup.scoring import (LEVELS, NEAR_SCORE, aggregate_games, expected_scores,
                     pick_level, score_strength)


def test_score_exact_near_far():
    assert score_strength(['3D'], ['3D']) == 1.0
    assert score_strength(['3D'], ['4D']) == pytest.approx(NEAR_SCORE)
    assert score_strength(['3D'], ['5D']) == 0.0


def test_score_kyu_dan_boundary_is_adjacent():
    # A 與 1D 在 LEVELS 中相鄰
    assert score_strength(['A'], ['1D']) == pytest.approx(NEAR_SCORE)


def test_score_average_and_index_input():
    # 猜中、差一級、差兩級各一題
    expected = (1.0 + NEAR_SCORE + 0.0) / 3
    assert score_strength([0, 5, 9], [0, 4, 7]) == pytest.approx(expected)
    assert score_strength(['D', '2D', '6D'], ['D', '1D', '4D']) == pytest.approx(expected)


def test_score_length_mismatch():
    with pytest.raises(ValueError):
        score_strength(['D', 'C'], ['D'])


def test_expected_scores_hand_computed():
    # 0.40 / 0.35 / 0.25 落在 index 3–5
    p = np.zeros(10)
    p[3:6] = [0.40, 0.35, 0.25]
    e = expected_scores(p)
    assert e[3] == pytest.approx(0.40 + NEAR_SCORE * 0.35)
    assert e[4] == pytest.approx(0.35 + NEAR_SCORE * (0.40 + 0.25))
    assert e[5] == pytest.approx(0.25 + NEAR_SCORE * 0.35)


def test_pick_level_prefers_middle_over_argmax():
    p = np.zeros(10)
    p[3:6] = [0.40, 0.35, 0.25]
    assert int(np.argmax(p)) == 3
    assert int(pick_level(p)) == 4


def test_expected_scores_edges_have_one_neighbor():
    p = np.zeros(10)
    p[0], p[1] = 0.6, 0.4
    e = expected_scores(p)
    assert e[0] == pytest.approx(0.6 + NEAR_SCORE * 0.4)
    p = np.zeros(10)
    p[9], p[8] = 0.6, 0.4
    assert expected_scores(p)[9] == pytest.approx(0.6 + NEAR_SCORE * 0.4)


def test_pick_level_batch_shape():
    probs = np.random.default_rng(0).dirichlet(np.ones(10), size=7)
    assert pick_level(probs).shape == (7,)


def test_pick_level_matches_simulated_score():
    # 期望得分最大的選法，在依分布抽樣的真實等級上平均分數不應低於 argmax
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.ones(10) * 0.5, size=2000)
    truth = np.array([rng.choice(10, p=q) for q in probs])
    assert score_strength(truth, pick_level(probs)) >= score_strength(truth, probs.argmax(1))


def test_aggregate_mean_and_log():
    g = np.array([[0.7, 0.3] + [0.0] * 8,
                  [0.5, 0.5] + [0.0] * 8])
    assert aggregate_games(g, 'mean')[:2] == pytest.approx([0.6, 0.4])
    q = aggregate_games(g, 'log')
    assert q[:2] == pytest.approx([0.7, 0.3])  # 0.35 : 0.15
    assert q.sum() == pytest.approx(1.0)


def test_levels_count():
    assert len(LEVELS) == 10
