import json
import queue
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def test_sgf_point_matches_sgfmill():
    from sgfmill import sgf
    node = sgf.Sgf_game.from_string('(;SZ[19];B[pd])').get_main_sequence()[1]
    assert ah.sgf_point('pd') == node.get_move()[1]


def test_parse_game_even_game_keeps_colors_and_komi():
    g = ah.parse_game('(;SZ[19]HD[0];B[pd];W[dp])')
    assert g['hd'] == 0 and g['komi'] == 6.5
    assert [c for c, _ in g['moves']] == ['B', 'W'] and g['initial_stones'] == []


def test_parse_game_no_hd_uses_default_komi():
    g = ah.parse_game('(;SZ[19];B[pd])')
    assert g['hd'] is None and g['komi'] == 6.5


def test_parse_game_hd1_no_komi_keeps_colors():
    g = ah.parse_game('(;SZ[19]HD[1];B[pd];W[dp])')
    assert g['komi'] == 0.0
    assert [c for c, _ in g['moves']] == ['B', 'W'] and g['initial_stones'] == []


def test_parse_game_handicap_matches_official_example():
    g = ah.parse_game('(;SZ[19]HD[2];B[pp];W[dd])')
    assert g['komi'] == 0.0
    assert [(c, ah.to_gtp(p)) for c, p in g['moves']] == [('W', 'Q4'), ('B', 'D16')]
    assert [(c, ah.to_gtp(p)) for c, p in g['initial_stones']] == [('B', 'Q16'), ('B', 'D4')]


@pytest.mark.parametrize('hd', range(2, 10))
def test_handicap_stone_counts(hd):
    g = ah.parse_game('(;SZ[19]HD[%d];B[aa])' % hd)
    assert len({p for _, p in g['initial_stones']}) == hd


def test_handicap_target_turns_use_actual_color():
    # color 欄位是實際顏色，目標執白時是棋譜標 B 的著手
    g = ah.parse_game('(;SZ[19]HD[2];B[pp];W[dd];B[qq])')
    assert ah.target_turns(g['moves'], 'W') == [0, 2]


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


def test_build_eval_query_has_no_profile_and_covers_next_turn():
    g = ah.parse_game('(;GM[1]SZ[19]KM[7.5];B[dd];W[pp];B[dp])')
    q = ah.build_eval_query('k|eval', g, [0, 2], 'chinese', 16)
    assert q['analyzeTurns'] == [0, 1, 2, 3]
    assert q['maxVisits'] == 16
    assert 'overrideSettings' not in q and 'includePolicy' not in q


def test_eval_turns_merges_adjacent_turns():
    assert ah.eval_turns([0, 2, 4]) == [0, 1, 2, 3, 4, 5]
    assert ah.eval_turns([0, 1]) == [0, 1, 2]


# 黑方視角的評估：第 0 手後黑領先少 2 目，第 1 手後黑領先多 3 目
ROOTS = {0: {'scoreLead': 5.0, 'winrate': 0.7}, 1: {'scoreLead': 3.0, 'winrate': 0.6},
         2: {'scoreLead': 6.0, 'winrate': 0.8}, 3: {'scoreLead': 6.0, 'winrate': 0.8}}


def test_move_losses_black_is_lead_decrease():
    score, winrate = ah.move_losses(ROOTS, [0], 'B')
    assert score == pytest.approx([2.0]) and winrate == pytest.approx([0.1])


def test_move_losses_white_is_lead_increase():
    score, winrate = ah.move_losses(ROOTS, [1], 'W')
    assert score == pytest.approx([3.0]) and winrate == pytest.approx([0.2])


def test_move_losses_negative_when_position_improves():
    score, winrate = ah.move_losses(ROOTS, [1], 'B')
    assert score == pytest.approx([-3.0]) and winrate == pytest.approx([-0.2])


class FakeEngine:
    """每個 turn 回傳一個 humanPolicy，actual 落子位置的機率由 prob 決定；eval query 回傳 ROOTS。"""

    def __init__(self, prob, fail_ids=()):
        self.prob, self.fail_ids = prob, set(fail_ids)
        self.queries = []

    def run(self, queries):
        self.queries += queries
        out = {}
        for q in queries:
            if q['id'] in self.fail_ids:
                out[q['id']] = None
                continue
            if q['id'].endswith('|eval'):
                out[q['id']] = {t: {'rootInfo': ROOTS[t]} for t in q['analyzeTurns']}
                continue
            hp = [0.0] * 362
            for t in q['analyzeTurns']:
                hp_t = list(hp)
                hp_t[0] = self.prob
                out.setdefault(q['id'], {})[t] = {'humanPolicy': hp_t}
        return out


def test_analyze_group_collects_probs():
    # 兩手都落在左上角 A19，政策機率固定為 0.25
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B',
         'parsed': {'moves': [('B', (18, 0)), ('W', (18, 1)), ('B', (18, 0))],
                    'initial_stones': [], 'komi': 7.5, 'hd': 0}}
    rows = ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0)
    assert len(rows) == 1
    r = rows[0]
    assert r['turns'] == [0, 2]
    assert np.allclose(r['probs'], 0.25) and np.array(r['probs']).shape == (10, 2)


THREE_MOVES = {'moves': [('B', (18, 0)), ('W', (18, 1)), ('B', (18, 2))],
               'initial_stones': [], 'komi': 7.5, 'hd': 0}


def test_analyze_group_records_losses_aligned_with_turns():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    engine = FakeEngine(0.25)
    r = ah.analyze_group(engine, [g], 'chinese', 0, eval_visits=8)[0]
    assert r['turns'] == [0, 2] and r['eval_visits'] == 8 and r['player_id'] == 'p1'
    assert r['hd'] == 0
    assert r['score_loss'] == pytest.approx([2.0, 0.0])
    assert r['winrate_loss'] == pytest.approx([0.1, 0.0])
    assert [q['maxVisits'] for q in engine.queries if q['id'] == 'D_1|eval'] == [8]


def test_analyze_group_white_loss_sign():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'W', 'parsed': THREE_MOVES}
    r = ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0)[0]
    assert r['turns'] == [1] and r['score_loss'] == pytest.approx([3.0])


def test_analyze_group_max_turns_limits_evals():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    engine = FakeEngine(0.25)
    r = ah.analyze_group(engine, [g], 'chinese', 1)[0]
    assert r['turns'] == [0] and len(r['score_loss']) == 1
    assert [q['analyzeTurns'] for q in engine.queries if q['id'] == 'D_1|eval'] == [[0, 1]]


def make_engine(responses):
    """不啟動 KataGo，回應預先放進 queue，讓 Engine.run 的迴圈可以直接測。"""
    engine = ah.Engine.__new__(ah.Engine)
    engine.proc = SimpleNamespace(stdin=SimpleNamespace(write=lambda s: None, flush=lambda: None))
    engine.out = queue.Queue()
    for r in responses:
        engine.out.put(r)
    return engine


def test_engine_run_error_marks_query_failed_and_finishes():
    engine = make_engine([{'id': 'a', 'error': 'Illegal move', 'field': 'moves'},
                          {'id': 'b', 'turnNumber': 0, 'rootInfo': {'winrate': 0.5}}])
    got = engine.run([{'id': 'a', 'analyzeTurns': [0, 1]}, {'id': 'b', 'analyzeTurns': [0]}])
    assert got == {'a': None, 'b': {0: {'rootInfo': {'winrate': 0.5}}}}


def test_engine_run_ignores_warning():
    engine = make_engine([{'id': 'a', 'warning': 'unused', 'field': 'rules'},
                          {'id': 'a', 'turnNumber': 0, 'humanPolicy': [0.5], 'policy': [0.1]}])
    assert engine.run([{'id': 'a', 'analyzeTurns': [0]}]) == {'a': {0: {'humanPolicy': [0.5]}}}


def player_df():
    # 每級 p*_big 有 5 局、p*_small 有 2 局
    rows = []
    for rank in 'DC':
        for i in range(3):
            rows += [(rank, '%s%d_big' % (rank, i))] * 5 + [(rank, '%s%d_small' % (rank, i))] * 2
    return pd.DataFrame(rows, columns=['rank', 'player_id'])


def test_sample_players_counts_and_skips_short_players():
    out = ah.sample_players(player_df(), 2, 4, seed=42)
    per_player = out.groupby('player_id').size()
    assert (per_player == 4).all() and len(per_player) == 4
    assert all(p.endswith('_big') for p in per_player.index)
    assert (out.groupby('rank')['player_id'].nunique() == 2).all()


def test_sample_players_is_deterministic():
    a = ah.sample_players(player_df(), 2, 4, seed=42)
    b = ah.sample_players(player_df(), 2, 4, seed=42)
    assert a.index.tolist() == b.index.tolist()


def test_sample_players_rejects_rank_with_too_few_players():
    with pytest.raises(SystemExit, match='C'):
        df = player_df()
        ah.sample_players(df[df['player_id'] != 'C0_big'], 3, 4, seed=42)


def test_sample_players_rejects_rank_with_no_qualified_player():
    df = player_df()
    with pytest.raises(SystemExit, match='C'):
        ah.sample_players(df[(df['rank'] == 'D') | df['player_id'].str.endswith('_small')],
                          1, 4, seed=42)


def test_load_games_by_player_is_json_serializable(tmp_path):
    df = pd.DataFrame({'player_id': [7] * 3, 'game_id': [1, 2, 3], 'rank': ['D'] * 3,
                       'color': ['B'] * 3, 'sgf_content': ['(;SZ[19])'] * 3})
    df.to_csv(tmp_path / 'train_D.csv', index=False)
    games = ah.load_games(str(tmp_path / 'train_*.csv'), 0, 42, 0, 1, 2)
    assert len(games) == 2 and games[0]['player_id'] == '7'
    json.dumps(games)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_done_keys_returns_keys_for_same_visits(tmp_path):
    p = tmp_path / 'a.jsonl'
    assert ah.done_keys(str(p), 1) == set()
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 1}])
    assert ah.done_keys(str(p), 1) == {'D_1'}


@pytest.mark.parametrize('row', [{'key': 'D_1', 'eval_visits': 8}, {'key': 'D_1'}])
def test_done_keys_rejects_mismatched_or_old_rows(tmp_path, row):
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, [row])
    with pytest.raises(SystemExit):
        ah.done_keys(str(p), 1)


def test_analyze_group_skips_failed_game():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B',
         'parsed': {'moves': [('B', (18, 0))], 'initial_stones': [], 'komi': 7.5, 'hd': 0}}
    assert ah.analyze_group(FakeEngine(0.5, fail_ids=['D_1|3']), [g], 'chinese', 0) == []


def test_analyze_group_skips_game_when_eval_fails():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B',
         'parsed': {'moves': [('B', (18, 0))], 'initial_stones': [], 'komi': 7.5, 'hd': 0}}
    assert ah.analyze_group(FakeEngine(0.5, fail_ids=['D_1|eval']), [g], 'chinese', 0) == []


def test_analyze_group_marks_illegal_as_nan():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B',
         'parsed': {'moves': [('B', (18, 0))], 'initial_stones': [], 'komi': 7.5, 'hd': 0}}
    rows = ah.analyze_group(FakeEngine(-1.0), [g], 'chinese', 0)
    assert np.isnan(np.array(rows[0]['probs'], dtype=float)).all()
