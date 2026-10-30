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


def test_apply_temperature():
    p = _data()['probs']
    np.testing.assert_allclose(eval_posthoc.apply_temperature(p, 1.0), p, rtol=1e-5)
    flat = eval_posthoc.apply_temperature(p, 1e6)
    np.testing.assert_allclose(flat, 1 / len(LEVELS), rtol=1e-4)
    sharp = eval_posthoc.apply_temperature(p, 0.5)
    np.testing.assert_array_equal(sharp.argmax(1), p.argmax(1))  # 溫度不改變單局 argmax
    with pytest.raises(ValueError):
        eval_posthoc.apply_temperature(p, 0)


def test_split_players_keeps_player_together():
    d = _data(n_players=10)
    first = eval_posthoc.split_players(d['player_id'], d['rank_idx'], np.random.default_rng(0))
    for p in set(d['player_id']):
        assert len(set(first[d['player_id'] == p].tolist())) == 1
    assert first.sum() == len(first) // 2


def test_calibrate_log_argmax_unaffected():
    cal = eval_posthoc.calibrate(_data(n_players=20), (0.5, 1.0, 3.0), 5, 10, 2, 0)
    assert set(cal) == {(a, s) for a in eval_posthoc.AGGS for s in eval_posthoc.SELECTS}
    r = cal[('log', 'argmax')]  # log 相加整體除以溫度，argmax 不變
    assert r['after'] == pytest.approx(r['before'])
    assert all(len(r['temps']) == 2 for r in cal.values())


def test_calibrate_flattens_overconfident_pick():
    # 真實等級都是 4，每局都很有把握地猜 3 或 5；log 合併在溫度 1 時被多數方帶走
    lean_low = np.zeros(len(LEVELS))
    lean_low[[3, 4, 5]] = [0.8, 0.15, 0.05]
    lean_high = lean_low.copy()
    lean_high[[3, 5]] = [0.05, 0.8]
    rows = [lean_low, lean_low, lean_low, lean_high, lean_high] * 40
    data = {'probs': np.clip(np.array(rows), 1e-6, None).astype(np.float32),
            'rank_idx': np.full(len(rows), 4, np.int8), 'key': np.array(['g%d' % i for i in range(len(rows))]),
            'player_id': np.repeat(['p%d' % i for i in range(40)], 5)}
    cal = eval_posthoc.calibrate(data, eval_posthoc.TEMPS, 5, 5, 1, 0)
    r = cal[('log', 'pick')]
    assert r['before'] < 0.5 and r['after'] == pytest.approx(1.0)
    assert all(t > 1 for t in r['temps'])
