import json

import numpy as np
import pytest

from aicup import eval_humansl as eh


def row(player, rank, best, turns=3):
    """best 這個 profile 的機率是 0.5，其他 0.1。"""
    probs = [[0.5 if i == best else 0.1] * turns for i in range(10)]
    return {'key': '%s_%d' % (player, turns), 'rank_idx': rank, 'player_id': player, 'probs': probs}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_game_loglik_sums_turns():
    ll = eh.game_loglik(row('p', 0, 2, turns=4)['probs'])
    assert ll.shape == (10,)
    assert ll[2] == pytest.approx(4 * np.log(0.5)) and ll[0] == pytest.approx(4 * np.log(0.1))


def test_game_loglik_skips_nan_turn_and_floors_zero():
    probs = [[0.5, float('nan'), 0.0]] * 10
    assert eh.game_loglik(probs)[0] == pytest.approx(np.log(0.5) + np.log(eh.PROB_FLOOR))


def test_to_probs_handles_gap_that_would_underflow():
    ll = np.array([-5000.0] + [-6000.0] * 9)
    q = eh.to_probs(ll)
    assert q[0] == pytest.approx(1.0) and q.sum() == pytest.approx(1.0)


def test_predict_argmax_and_pick():
    ll = np.log(np.array([[0.4, 0.0001, 0.3, 0.3, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9]]))
    assert eh.predict(ll, 'argmax')[0] == 0
    assert eh.predict(ll, 'pick')[0] == 2  # 2 的期望得分含 3 的 1/e


def test_load_rows_shapes(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_rows(p, [row('p1', 0, 0), row('p2', 3, 3, turns=2)])
    d = eh.load_rows(str(p))
    assert d['loglik'].shape == (2, 10)
    assert d['rank_idx'].tolist() == [0, 3] and d['player_id'].tolist() == ['p1', 'p2']


def test_load_rows_rejects_empty_file(tmp_path):
    p = tmp_path / 'a.jsonl'
    p.write_text('', encoding='utf-8')
    with pytest.raises(ValueError):
        eh.load_rows(str(p))


def make_data(wrong_games=0):
    # 兩位玩家各 10 局；wrong_games 局投給錯的等級，但多局相加後仍是正確等級
    rows = []
    for player, rank in (('p1', 0), ('p2', 5)):
        for g in range(10):
            rows.append(row(player, rank, 9 if g < wrong_games else rank))
    return {'loglik': np.array([eh.game_loglik(r['probs']) for r in rows]),
            'rank_idx': np.array([r['rank_idx'] for r in rows], np.int64),
            'player_id': np.array([r['player_id'] for r in rows])}


def test_evaluate_perfect_when_all_games_vote_correctly():
    res = eh.evaluate(make_data(), (5,), repeats=3, seed=42)
    assert res['single']['argmax'] == pytest.approx(1.0)
    assert res['questions']['argmax'] == pytest.approx((1.0, 0.0))
    assert res['n_questions'] == 2  # 每位玩家每輪 1 題


def test_evaluate_question_sum_fixes_single_game_errors():
    res = eh.evaluate(make_data(wrong_games=3), (10,), repeats=1, seed=42)
    assert res['single']['argmax'] == pytest.approx(0.7)
    assert res['questions']['argmax'][0] == pytest.approx(1.0)


def test_evaluate_rejects_when_no_player_has_enough_games():
    with pytest.raises(ValueError):
        eh.evaluate(make_data(), (11, 20), repeats=1, seed=42)


def test_load_rows_roundtrip_with_analyze_humansl_fields(tmp_path):
    # analyze_humansl 的 NaN 以 JSON 的 NaN 寫出，json.loads 要讀得回來
    r = row('p1', 0, 0)
    r['probs'][1][0] = float('nan')
    r.update({'color': 'B', 'turns': [0, 2, 4], 'eval_visits': 1,
              'score_loss': [0.1, 0.2, 0.3], 'winrate_loss': [0.01, 0.02, 0.03]})
    p = tmp_path / 'a.jsonl'
    write_rows(p, [r])
    assert np.isfinite(eh.load_rows(str(p))['loglik']).all()
