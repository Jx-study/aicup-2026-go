import json
import queue
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aicup import analyze_humansl as ah
from aicup import fit_humansl
from stack_helpers import write_rank_tests


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


P3 = ['rank_11k', 'rank_1d', 'rank_6d']


def test_analyze_group_sends_only_selected_profiles():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    engine = FakeEngine(0.25)
    r = ah.analyze_group(engine, [g], 'chinese', 0, profiles=P3)[0]
    sent = [q['overrideSettings']['humanSLProfile'] for q in engine.queries if 'overrideSettings' in q]
    assert sent == P3
    assert np.array(r['probs']).shape == (3, 2) and r['profiles'] == P3


def test_analyze_group_default_records_all_profiles():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    assert ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0)[0]['profiles'] == ah.PROFILES


def test_analyze_group_no_eval_skips_eval_query_and_loss_fields():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    engine = FakeEngine(0.25)
    r = ah.analyze_group(engine, [g], 'chinese', 0, eval_visits=0)[0]
    assert not any(q['id'].endswith('|eval') for q in engine.queries)
    assert 'score_loss' not in r and 'winrate_loss' not in r and r['eval_visits'] == 0


def test_check_profiles_rejects_unknown_name():
    ah.check_profiles(P3)
    with pytest.raises(SystemExit, match='rank_7d.*rank_11k'):
        ah.check_profiles(['rank_1d', 'rank_7d'])


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


def listed_df():
    # a 在 D 有 5 局、在 C 有 2 局，b 在 C 有 4 局，c 在 D 有 6 局
    rows = [('a', 'D')] * 5 + [('a', 'C')] * 2 + [('b', 'C')] * 4 + [('c', 'D')] * 6
    df = pd.DataFrame(rows, columns=['player_id', 'rank'])
    df['game_id'] = range(len(df))
    return df


LISTED = pd.DataFrame({'player_id': ['b', 'a'], 'rank': ['C', 'D']})


def test_sample_listed_players_keeps_only_listed_rank():
    out = ah.sample_listed_players(listed_df(), LISTED, 4, seed=42)
    assert out.groupby(['player_id', 'rank']).size().to_dict() == {('a', 'D'): 4, ('b', 'C'): 4}


def test_sample_listed_players_order_ignores_list_order():
    a = ah.sample_listed_players(listed_df(), LISTED, 3, seed=42)
    b = ah.sample_listed_players(listed_df(), LISTED.iloc[::-1], 3, seed=42)
    assert a['game_id'].tolist() == b['game_id'].tolist()


def test_sample_listed_players_rejects_short_player():
    players = pd.DataFrame({'player_id': ['a', 'b', 'x'], 'rank': ['C', 'C', 'D']})
    with pytest.raises(SystemExit, match="'a', 'C'.*'x', 'D'"):
        ah.sample_listed_players(listed_df(), players, 4, seed=42)


def write_train_csv(tmp_path):
    df = listed_df()
    df['color'], df['sgf_content'] = 'B', '(;SZ[19])'
    for rank, part in df.groupby('rank'):
        part.to_csv(tmp_path / ('train_%s.csv' % rank), index=False)
    LISTED.to_csv(tmp_path / 'players.csv', index=False)
    return str(tmp_path / 'train_*.csv'), str(tmp_path / 'players.csv')


def test_load_games_players_file(tmp_path):
    csv_glob, players = write_train_csv(tmp_path)
    games = ah.load_games(csv_glob, 0, 42, 0, games_per_player=4, players_file=players)
    again = ah.load_games(csv_glob, 0, 42, 0, games_per_player=4, players_file=players)
    assert len(games) == 8 and [g['key'] for g in games] == [g['key'] for g in again]
    assert {(g['player_id'], ah.LEVELS[g['rank_idx']]) for g in games} == {('a', 'D'), ('b', 'C')}


@pytest.mark.parametrize('per_rank, players_per_rank', [(1, 0), (0, 1)])
def test_load_games_players_file_rejects_other_sampling(tmp_path, per_rank, players_per_rank):
    csv_glob, players = write_train_csv(tmp_path)
    with pytest.raises(SystemExit, match='--players-file'):
        ah.load_games(csv_glob, per_rank, 42, 0, players_per_rank=players_per_rank,
                      games_per_player=4, players_file=players)


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


def test_done_keys_resumes_same_profiles(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 0, 'profiles': P3}])
    assert ah.done_keys(str(p), 0, P3) == {'D_1'}


def test_done_keys_rejects_different_profiles(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 0, 'profiles': P3}])
    with pytest.raises(SystemExit, match='--out'):
        ah.done_keys(str(p), 0, ['rank_11k', 'rank_6d'])


def test_done_keys_treats_missing_profiles_as_all(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 1}])
    assert ah.done_keys(str(p), 1) == {'D_1'}
    with pytest.raises(SystemExit):
        ah.done_keys(str(p), 1, P3)


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


def test_no_eval_rows_load_in_fit_humansl(tmp_path):
    games = [{'key': 'D_%d' % i, 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
             for i in range(2)]
    rows = ah.analyze_group(FakeEngine(0.25), games, 'chinese', 0, eval_visits=0, profiles=P3)
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, rows)
    out = fit_humansl.load_games(str(p))
    assert out['n_profiles'] == 3 and out['has_loss'] is False


def test_load_test_games_keys_colors_and_order(tmp_path):
    sgf = '(;SZ[19]HD[0];B[pd];W[dp])'
    write_rank_tests(tmp_path / 't', {'public': [([sgf, sgf], 'W'), ([sgf], 'B')], 'private': [([sgf], 'B')]})
    games = ah.load_test_games(str(tmp_path / 't'))
    assert [g['key'] for g in games] == ['pub_q_0001_g01', 'pub_q_0001_g02', 'pub_q_0002_g01', 'priv_q_0001_g01']
    assert [g['color'] for g in games] == ['W', 'W', 'B', 'B']
    assert [g['player_id'] for g in games] == ['pub_q_0001', 'pub_q_0001', 'pub_q_0002', 'priv_q_0001']
    assert all(g['rank_idx'] == -1 and g['sgf'] == sgf for g in games)
    assert len(ah.load_test_games(str(tmp_path / 't'), limit=3)) == 3


def test_test_game_keys_count_within_question():
    games = [(0, 's', 'B'), (1, 's', 'B'), (0, 's', 'B')]
    assert ah.test_game_keys(['a', 'b'], games) == ['a_g01', 'b_g01', 'a_g02']


def test_side_color():
    assert ah.side_color('B', 'target') == 'B' and ah.side_color('B', 'opponent') == 'W'
    assert ah.side_color('W', 'opponent') == 'B'


def test_analyze_group_opponent_side_scores_other_color():
    # 目標執黑，對手是白方的第 1 手，損失同 test_analyze_group_white_loss_sign
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    engine = FakeEngine(0.25)
    r = ah.analyze_group(engine, [g], 'chinese', 0, side='opponent')[0]
    assert r['turns'] == [1] and r['score_loss'] == pytest.approx([3.0])
    assert r['color'] == 'B' and r['side'] == 'opponent'
    assert all(q['analyzeTurns'] == [1] for q in engine.queries if 'overrideSettings' in q)


def test_analyze_group_records_target_side_by_default():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    assert ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0)[0]['side'] == 'target'


def test_done_keys_rejects_different_side(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 0, 'profiles': P3, 'side': 'opponent'}])
    assert ah.done_keys(str(p), 0, P3, 'opponent') == {'D_1'}
    with pytest.raises(SystemExit, match='side'):
        ah.done_keys(str(p), 0, P3)


def test_done_keys_treats_missing_side_as_target(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 0, 'profiles': P3}])
    assert ah.done_keys(str(p), 0, P3) == {'D_1'}
    with pytest.raises(SystemExit, match='side'):
        ah.done_keys(str(p), 0, P3, 'opponent')


def test_build_query_pre_root_history_only_when_enabled():
    g = ah.parse_game('(;GM[1]SZ[19]KM[7.5];B[dd];W[pp])')
    base = ah.build_query('k|0', g, [0], 'rank_5k', 'chinese')
    assert base['overrideSettings'] == {'humanSLProfile': 'rank_5k'}
    assert base == ah.build_query('k|0', g, [0], 'rank_5k', 'chinese', pre_root_history=False)
    q = ah.build_query('k|0', g, [0], 'rank_5k', 'chinese', pre_root_history=True)
    assert q['overrideSettings'] == {'humanSLProfile': 'rank_5k', 'ignorePreRootHistory': False}


def test_analyze_group_pre_root_history_marks_rows_and_profile_queries():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    assert 'pre_root_history' not in ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0)[0]
    engine = FakeEngine(0.25)
    r = ah.analyze_group(engine, [g], 'chinese', 0, profiles=P3, pre_root_history=True)[0]
    assert r['pre_root_history'] is True
    prof = [q for q in engine.queries if 'overrideSettings' in q]
    assert len(prof) == 3
    assert all(q['overrideSettings']['ignorePreRootHistory'] is False for q in prof)
    assert all('overrideSettings' not in q for q in engine.queries if q['id'].endswith('|eval'))


def test_done_keys_rejects_different_pre_root_history(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 0, 'profiles': P3}])
    with pytest.raises(SystemExit, match='pre_root_history'):
        ah.done_keys(str(p), 0, P3, pre_root_history=True)
    write_jsonl(p, [{'key': 'D_1', 'eval_visits': 0, 'profiles': P3, 'pre_root_history': True}])
    assert ah.done_keys(str(p), 0, P3, pre_root_history=True) == {'D_1'}
    with pytest.raises(SystemExit, match='pre_root_history'):
        ah.done_keys(str(p), 0, P3)


def test_policy_stats_rank_pmax_entropy():
    hp = [0.0] * 362
    hp[0], hp[1], hp[2], hp[3] = 0.5, 0.25, 0.25, -1.0  # 最後一點不合法
    assert ah.policy_stats(hp, 0)[:2] == (0.0, 0.5)
    rank, pmax, ent = ah.policy_stats(hp, 1)
    assert rank == 1.0 and pmax == 0.5
    assert ent == pytest.approx(-(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25)))
    assert ah.policy_stats(hp, 4)[0] == 3.0  # 機率 0 的合法點，前面有 3 個更大
    assert all(np.isnan(ah.policy_stats(hp, 3)))


def test_analyze_group_move_stats_columns_and_default_absent():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    assert not {'move_rank', 'pmax', 'entropy'} & set(ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0)[0])
    r = ah.analyze_group(FakeEngine(0.25), [g], 'chinese', 0, profiles=P3, move_stats=True)[0]
    for name in ('move_rank', 'pmax', 'entropy'):
        assert np.array(r[name]).shape == np.array(r['probs']).shape == (3, 2)
    # FakeEngine 的 policy 只有 A19 為 0.25；第一手落在 A19，第二手落在機率 0 的合法點
    assert np.allclose(r['move_rank'], [[0, 1]] * 3) and np.allclose(r['pmax'], 0.25)
    assert np.allclose(r['entropy'], -0.25 * np.log(0.25))


def test_analyze_group_move_stats_nan_when_illegal():
    g = {'key': 'D_1', 'rank_idx': 0, 'player_id': 'p1', 'color': 'B', 'parsed': THREE_MOVES}
    r = ah.analyze_group(FakeEngine(-1.0), [g], 'chinese', 0, move_stats=True)[0]
    for name in ('move_rank', 'pmax', 'entropy'):
        assert np.isnan(np.array(r[name], dtype=float)[:, 0]).all()  # 第一手的 A19 不合法
        assert not np.isnan(np.array(r[name], dtype=float)[:, 1]).any()


def test_check_profiles_accepts_preaz_of_same_levels():
    ah.check_profiles(['rank_11k', 'preaz_1d', 'preaz_11k'])
    for bad in ('preaz_7d', 'proyear_2020', 'xrank_1d'):
        with pytest.raises(SystemExit):
            ah.check_profiles([bad])


def test_main_rejects_test_dir_with_keys_from(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['analyze_humansl', '--test-dir', 'tests', '--keys-from', 'k.jsonl',
                                     '--out', str(tmp_path / 'o.jsonl')])
    with pytest.raises(SystemExit, match='--keys-from'):
        ah.main()
