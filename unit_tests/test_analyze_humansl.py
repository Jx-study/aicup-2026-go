import numpy as np
import pytest

from aicup import analyze_humansl as ah


def test_to_gtp_skips_letter_i():
    assert ah.to_gtp((3, 3)) == 'D4'
    assert ah.to_gtp((0, 8)) == 'J1'
    assert ah.to_gtp((18, 18)) == 'T19'
    assert ah.to_gtp(None) == 'pass'


def test_policy_index_is_top_left_row_major():
    assert ah.policy_index((18, 0)) == 0           # 左上角 A19
    assert ah.policy_index((0, 18)) == 18 * 19 + 18  # 右下角 T1
    assert ah.policy_index((3, 15)) == 300         # Q4
    assert ah.policy_index(None) == 361


def test_parse_game_converts_moves():
    g = ah.parse_game('(;GM[1]SZ[19]KM[6.5];B[dd];W[pp];B[];W[dp])')
    assert g['komi'] == 6.5
    assert [c for c, _ in g['moves']] == ['B', 'W', 'B', 'W']
    assert [ah.to_gtp(p) for _, p in g['moves']] == ['D16', 'Q4', 'pass', 'D4']
    assert g['initial_stones'] == []


def test_parse_game_default_komi_and_handicap():
    g = ah.parse_game('(;GM[1]SZ[19]HA[2]AB[dd][pp];W[dp])', default_komi=0.5)
    assert g['komi'] == 0.5
    assert sorted((c, ah.to_gtp(p)) for c, p in g['initial_stones']) == [('B', 'D16'), ('B', 'Q4')]
    assert len(g['moves']) == 1


def test_parse_game_rejects_other_board_size():
    with pytest.raises(ValueError):
        ah.parse_game('(;GM[1]SZ[9];B[dd])')


def test_target_turns():
    moves = [('B', (0, 0)), ('W', (1, 1)), ('B', (2, 2)), ('W', (3, 3)), ('B', (4, 4))]
    assert ah.target_turns(moves, 'B') == [0, 2, 4]
    assert ah.target_turns(moves, 'W') == [1, 3]
    assert ah.target_turns(moves, 'B', max_turns=2) == [0, 2]


def test_build_query_fields():
    g = ah.parse_game('(;GM[1]SZ[19]KM[7.5];B[dd];W[pp])')
    q = ah.build_query('k|0', g, [0, 2], 'rank_5k', 'chinese')
    assert q['moves'] == [['B', 'D16'], ['W', 'Q4']]
    assert q['analyzeTurns'] == [0, 2]
    assert q['overrideSettings'] == {'humanSLProfile': 'rank_5k'}
    assert q['includePolicy'] and q['maxVisits'] == 1


class FakeEngine:
    """每個 turn 回傳一個 humanPolicy，actual 落子位置的機率由 prob 決定。"""

    def __init__(self, prob, fail_ids=()):
        self.prob, self.fail_ids = prob, set(fail_ids)

    def run(self, queries):
        out = {}
        for q in queries:
            if q['id'] in self.fail_ids:
                out[q['id']] = None
                continue
            hp = [0.0] * 362
            for t in q['analyzeTurns']:
                hp_t = list(hp)
                hp_t[0] = self.prob
                out.setdefault(q['id'], {})[t] = hp_t
        return out


def test_analyze_group_collects_probs(monkeypatch):
    # 兩手都落在左上角 A19，政策機率固定為 0.25
    g = {'key': 'D_1', 'rank_idx': 0, 'color': 'B',
         'parsed': {'moves': [('B', (18, 0)), ('W', (18, 1)), ('B', (18, 0))],
                    'initial_stones': [], 'komi': 7.5}}
    rows = ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0)
    assert len(rows) == 1
    r = rows[0]
    assert r['turns'] == [0, 2]
    assert np.allclose(r['probs'], 0.25) and np.array(r['probs']).shape == (10, 2)


def test_analyze_group_skips_failed_game():
    g = {'key': 'D_1', 'rank_idx': 0, 'color': 'B',
         'parsed': {'moves': [('B', (18, 0))], 'initial_stones': [], 'komi': 7.5}}
    assert ah.analyze_group(FakeEngine(0.5, fail_ids=['D_1|3']), [g], 'chinese', 0) == []


def test_analyze_group_marks_illegal_as_nan():
    g = {'key': 'D_1', 'rank_idx': 0, 'color': 'B',
         'parsed': {'moves': [('B', (18, 0))], 'initial_stones': [], 'komi': 7.5}}
    rows = ah.analyze_group(FakeEngine(-1.0), [g], 'chinese', 0)
    assert np.isnan(np.array(rows[0]['probs'], dtype=float)).all()
