import re
import sys

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from aicup import stack_humansl as sh
from aicup.fit_humansl import make_splits, question_points
from aicup.scoring import aggregate_games, apply_temperature, pick_level
from stack_helpers import toy_data, write_npz


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
    assert np.allclose(f[:, :10], np.log(np.maximum(sh.cnn_question_probs(probs, questions, 3.0), sh.EPS)))
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


def test_select_c_never_shares_a_group_across_inner_folds(monkeypatch):
    overlaps = []
    expected = np.arange(300) % 15

    class SpySplitter(GroupKFold):
        def split(self, X, y=None, groups=None):
            assert np.array_equal(groups, expected)
            for tr, te in super().split(X, y, groups):
                overlaps.append(set(groups[tr]) & set(groups[te]))
                yield tr, te

    monkeypatch.setattr(sh, 'GroupKFold', SpySplitter)
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 30)
    x = y[:, None] + rng.normal(0, 0.3, (len(y), 2))
    sh.select_c(x, y, expected, cs=(0.1, 1.0), inner_folds=3)
    assert len(overlaps) == 6 and not any(overlaps)


def test_train_rng_is_reproducible_per_seed_and_fold():
    a = sh.train_rng(42, 0).random(5)
    assert np.array_equal(a, sh.train_rng(42, 0).random(5))
    assert not np.array_equal(a, sh.train_rng(43, 0).random(5))
    assert not np.array_equal(a, sh.train_rng(42, 1).random(5))


def test_make_splits_depends_on_seed():
    data, _ = toy_data()
    a = make_splits(data, (5, 20), 2, seed=1)
    b = make_splits(data, (5, 20), 2, seed=1)
    c = make_splits(data, (5, 20), 2, seed=2)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(a, c))


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
    # 隨機猜約 0.2，特徵有訊號時應明顯高於此
    assert np.concatenate(points).mean() > 0.5


def test_main_runs_end_to_end_on_toy_data(tmp_path, monkeypatch, capsys):
    data, probs = toy_data()
    write_npz(tmp_path / 'v.npz', data['key'], probs, data['rank_idx'])
    monkeypatch.setattr(sh, 'load_games', lambda path: data)
    monkeypatch.setattr(sys, 'argv', ['stack_humansl', 'unused.jsonl', '--cnn-probs', str(tmp_path / 'v.npz'),
                                      '--train-repeats', '2', '--test-repeats', '2', '--inner-folds', '3'])
    sh.main()
    out = capsys.readouterr().out
    for name in ('stacking', 'CNN 重新擬合', 'Human SL avg A C=0.1', 'CNN', 'stacking - CNN', '暫定門檻'):
        assert name in out
    # 門檻判定要和印出的區間下限一致
    lo = float(re.search(r'stacking - CNN：\S+，95% 區間 (\S+)～', out).group(1))
    verdict = out.split('暫定門檻')[1].split('：')[1].strip()
    assert verdict == ('通過' if lo > 0 else '未通過')
