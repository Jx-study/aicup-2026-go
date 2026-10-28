import numpy as np
import pytest

from aicup import eval_posthoc, train_rank
from aicup.scoring import LEVELS, NEAR_SCORE


def _data(n_players=6, games_each=12, seed=0):
    rng = np.random.default_rng(seed)
    pid, rank = [], []
    for p in range(n_players):
        pid += ['p%d' % p] * games_each
        rank += [p % len(LEVELS)] * games_each
    rank = np.array(rank, np.int8)
    # 正確等級機率偏高的假預測
    logits = rng.normal(size=(len(rank), len(LEVELS)))
    logits[np.arange(len(rank)), rank] += 1.5
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return {'probs': probs.astype(np.float32), 'rank_idx': rank,
            'key': np.array(['g%d' % i for i in range(len(rank))]), 'player_id': np.array(pid)}


def test_make_questions_sizes_and_labels():
    d = _data()
    rng = np.random.default_rng(0)
    questions, labels = eval_posthoc.make_questions(d['player_id'], d['rank_idx'], 5, 7, rng)
    assert all(5 <= len(q) <= 7 for q in questions)
    for q, lab in zip(questions, labels):
        assert len(set(d['player_id'][q])) == 1  # 一題只含同一位玩家
        assert (d['rank_idx'][q] == lab).all()
    used = np.concatenate(questions)
    assert len(used) == len(set(used.tolist()))  # 一局不會出現在兩題


def test_same_player_id_different_rank_is_different_player():
    pid = np.array(['p'] * 10)
    rank = np.array([0] * 5 + [3] * 5)
    _, labels = eval_posthoc.make_questions(pid, rank, 5, 5, np.random.default_rng(0))
    assert sorted(labels.tolist()) == [0, 3]


def test_evaluate_multi_game_beats_single():
    res = eval_posthoc.evaluate(_data(n_players=20), 5, 10, 3, 0)
    assert set(res['questions']) == {(a, s) for a in eval_posthoc.AGGS for s in eval_posthoc.SELECTS}
    assert res['questions'][('mean', 'argmax')][0] > res['single']['argmax']


def test_evaluate_fails_when_too_few_games():
    with pytest.raises(ValueError):
        eval_posthoc.evaluate(_data(games_each=3), 5, 10, 1, 0)


def test_make_questions_rejects_bad_sizes():
    pid, rank = np.array(['p'] * 4), np.zeros(4, np.int64)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        eval_posthoc.make_questions(pid, rank, 0, 3, rng)
    with pytest.raises(ValueError):
        eval_posthoc.make_questions(pid, rank, 4, 3, rng)


def test_evaluate_hand_computed():
    # A 等級 0 全猜中；B 等級 5 全猜 4，差一級
    probs = np.zeros((4, len(LEVELS)), np.float32)
    probs[:2, 0] = 1
    probs[2:, 4] = 1
    data = {'probs': probs, 'rank_idx': np.array([0, 0, 5, 5], np.int8),
            'key': np.array(['g%d' % i for i in range(4)]), 'player_id': np.array(['A', 'A', 'B', 'B'])}
    res = eval_posthoc.evaluate(data, 2, 2, 3, 0)
    for mean, std in res['questions'].values():
        assert mean == pytest.approx((1 + NEAR_SCORE) / 2) and std == pytest.approx(0)


def test_load_val_probs_roundtrip(tmp_path):
    d = _data()
    meta = {'rank_idx': d['rank_idx'].astype(np.int8), 'key': np.array(d['key'], dtype=str),
            'player_id': np.array(d['player_id'], dtype=str)}
    path = str(tmp_path / 'val_probs.npz')
    train_rank.save_val_probs(path, d['probs'], meta, 3)
    loaded = eval_posthoc.load_val_probs(path)
    np.testing.assert_array_equal(loaded['probs'], d['probs'])
    for k in meta:
        np.testing.assert_array_equal(loaded[k], meta[k])
    assert loaded['epoch'] == 3
