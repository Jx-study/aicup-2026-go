import numpy as np
import pytest

from aicup import stack_humansl as sh
from aicup.fit_humansl import make_splits, question_points
from aicup.scoring import aggregate_games, apply_temperature, pick_level


def toy_data(n_players=50, games=20, seed=0):
    """每級 5 位玩家，Human SL 特徵與 CNN 機率都帶有等級訊號。"""
    rng = np.random.default_rng(seed)
    rank = np.repeat(np.arange(n_players) % 10, games)
    player = np.repeat(['p%d' % i for i in range(n_players)], games)
    x = rank[:, None] + rng.normal(0, 2, (len(rank), 4))
    logits = -0.5 * (np.arange(10)[None, :] - rank[:, None] - rng.normal(0, 2, (len(rank), 1))) ** 2
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    keys = np.array(['k%d' % i for i in range(len(rank))])
    data = {'x': x, 'rank_idx': rank.astype(np.int64), 'player_id': player, 'key': keys,
            'n_profiles': 3, 'has_loss': False}
    return data, probs


def write_npz(path, keys, probs, rank_idx):
    np.savez(path, key=keys, probs=probs, rank_idx=rank_idx)


def test_load_cnn_probs_aligns_by_key(tmp_path):
    probs = np.eye(10)[[1, 2, 3]]
    write_npz(tmp_path / 'v.npz', np.array(['a', 'b', 'c']), probs, np.array([1, 2, 3]))
    out = sh.load_cnn_probs(str(tmp_path / 'v.npz'), np.array(['c', 'a']), np.array([3, 1]))
    assert np.array_equal(out, probs[[2, 0]])


def test_load_cnn_probs_rejects_missing_key(tmp_path):
    write_npz(tmp_path / 'v.npz', np.array(['a']), np.eye(10)[[1]], np.array([1]))
    with pytest.raises(ValueError, match='b'):
        sh.load_cnn_probs(str(tmp_path / 'v.npz'), np.array(['a', 'b']), np.array([1, 1]))


def test_load_cnn_probs_rejects_rank_mismatch(tmp_path):
    write_npz(tmp_path / 'v.npz', np.array(['a']), np.eye(10)[[1]], np.array([1]))
    with pytest.raises(ValueError, match='等級'):
        sh.load_cnn_probs(str(tmp_path / 'v.npz'), np.array(['a']), np.array([2]))


def test_cnn_question_probs_matches_predict_rank_aggregation():
    _, probs = toy_data()
    questions = [np.array([0, 1, 2]), np.array([5])]
    out = sh.cnn_question_probs(probs, questions, temp=3.0)
    scaled = apply_temperature(probs, 3.0)
    assert np.allclose(out[0], aggregate_games(scaled[[0, 1, 2]], 'log'))
    assert np.allclose(out[1], aggregate_games(scaled[[5]], 'log'))


def test_question_features_concat_cnn_logp_and_mean_hsl():
    data, probs = toy_data()
    questions = [np.array([0, 1]), np.array([3, 4, 5])]
    f = sh.question_features(data['x'], probs, questions, temp=3.0)
    assert f.shape == (2, 10 + 4)
    assert np.allclose(f[:, :10], np.log(np.maximum(sh.cnn_question_probs(probs, questions, 3.0), 1e-12)))
    assert np.isfinite(f).all()
    assert np.allclose(f[1, 10:], data['x'][[3, 4, 5]].mean(0))


def test_predict_stack_without_hsl_uses_only_cnn_features():
    # Human SL 特徵換成雜訊時，只用 CNN 的結果不變
    data, probs = toy_data()
    splits = make_splits(data, (5, 20), 2, seed=42)
    train, questions, _, _ = splits[0]
    noisy = dict(data, x=np.random.default_rng(9).normal(size=data['x'].shape))
    a, _ = sh.predict_stack(data, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                            temp=3.0, cs=(0.1, 1.0), inner_folds=3, use_hsl=False)
    b, _ = sh.predict_stack(noisy, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                            temp=3.0, cs=(0.1, 1.0), inner_folds=3, use_hsl=False)
    assert np.allclose(a, b)


def test_select_c_rejects_over_regularized_model():
    # C 太小時機率接近均勻，選不出正確等級
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 30)
    x = y[:, None] + rng.normal(0, 0.3, (len(y), 2))
    groups = np.arange(len(y)) % 15
    assert sh.select_c(x, y, groups, cs=(1e-6, 10.0), inner_folds=3) == 10.0


def test_predict_stack_ignores_test_player_labels():
    data, probs = toy_data()
    splits = make_splits(data, (5, 20), 2, seed=42)
    train, questions, _, _ = splits[0]
    a, c_a = sh.predict_stack(data, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                              temp=3.0, cs=(0.1, 1.0), inner_folds=3)
    leaked = dict(data, rank_idx=np.where(train, data['rank_idx'], 0))
    b, c_b = sh.predict_stack(leaked, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                              temp=3.0, cs=(0.1, 1.0), inner_folds=3)
    assert c_a == c_b and np.allclose(a, b)
    assert a.shape == (len(questions), 10) and np.allclose(a.sum(1), 1)


def test_predict_stack_beats_chance_on_informative_features():
    data, probs = toy_data()
    splits = make_splits(data, (5, 20), 2, seed=42)
    points = []
    for train, questions, labels, _ in splits:
        p, _ = sh.predict_stack(data, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                                temp=3.0, cs=(0.1, 1.0), inner_folds=3)
        points.append(question_points(labels, pick_level(p)))
    assert np.concatenate(points).mean() > 0.5
