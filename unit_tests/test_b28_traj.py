import numpy as np
import pandas as pd
import pytest

from aicup import analyze_humansl as ah
from aicup import b28_traj as bt

N = 361


def resp(lead=0.0, wr=0.5, stdev=10.0, own=None, pol=None):
    own = np.zeros(N) if own is None else own
    pol = np.full(N + 1, 1.0 / (N + 1)) if pol is None else pol
    return {'rootInfo': {'scoreLead': lead, 'winrate': wr, 'scoreStdev': stdev},
            'ownership': own.tolist(), 'policy': pol.tolist()}


def test_triplet_starts_needs_next_move_and_spaces_evenly():
    moves = [('B', None), ('W', None)] * 3  # B 在 0、2、4
    assert bt.triplet_starts(moves, 'B', 16) == [0, 2, 4]
    assert bt.triplet_starts(moves[:5], 'B', 16) == [0, 2]  # 第 4 手後面沒有對手手
    long = [('B', None), ('W', None)] * 100
    s = bt.triplet_starts(long, 'W', 16)
    assert len(s) == 16 and s[0] == 1 and s[-1] == 197
    assert all(long[t][0] == 'W' for t in s)


def test_triplet_starts_skips_same_color_runs():
    moves = [('B', None), ('B', None), ('W', None), ('B', None), ('W', None)]
    assert bt.triplet_starts(moves, 'B', 16) == [1, 3]


def test_illegal_move_prior_is_nan():
    point = ah.sgf_point('dd')
    pol = np.full(N + 1, 0.001); pol[ah.policy_index(point)] = -1
    f = dict(zip(bt.FEATURES, bt.move_features(resp(pol=pol), resp(), point, 'B', 3)))
    assert np.isnan(f['log_prior']) and np.isnan(f['is_top']) and not np.isnan(f['own_at_move'])


def test_save_part_never_overwrites_after_delete(tmp_path):
    rows = bt.analyze_group(FakeEngine(), [game()], 'japanese')
    for _ in range(3):
        bt.save_part(str(tmp_path), rows)
    (tmp_path / 'part-00001.npz').unlink()
    assert bt.save_part(str(tmp_path), rows).endswith('part-00003.npz')
    assert len(bt.load_parts(str(tmp_path))['key']) == 3
    (tmp_path / 'part-00000.npz').unlink()  # load_parts 讀完已關閉檔案，Windows 上也能刪


def test_query_turns_dedup_and_cap():
    assert bt.query_turns([0, 2]) == [0, 1, 2, 3, 4]
    long = [('B', None), ('W', None)] * 100
    assert len(bt.query_turns(bt.triplet_starts(long, 'B'))) <= 48


def test_traj_query_flags():
    g = ah.parse_game('(;SZ[19]HD[0];B[dd];W[pp];B[dp])')
    q = bt.build_traj_query('x', g, [0, 1, 2], 'japanese')
    assert q['maxVisits'] == 1 and q['includeOwnership'] and q['includePolicy']
    assert q['analyzeTurns'] == [0, 1, 2]
    assert q['overrideSettings'] == {'ignorePreRootHistory': False}
    assert 'humanSLProfile' not in q['overrideSettings']


def test_ownership_index_matches_policy_index():
    # A19（SGF aa）是左上角，ownership 與 policy 的第 0 格
    assert ah.policy_index(ah.sgf_point('aa')) == 0
    assert ah.policy_index(ah.sgf_point('sa')) == 18
    assert ah.policy_index(ah.sgf_point('as')) == 18 * 19


@pytest.mark.parametrize('color,s', [('B', 1.0), ('W', -1.0)])
def test_move_features_signs_follow_mover(color, s):
    point = ah.sgf_point('dd')
    idx = ah.policy_index(point)
    ob = np.zeros(N); ob[:20] = 0.95; ob[idx] = 0.3  # 黑方為正
    oa = ob.copy(); oa[100:105] = 1.0  # 落子後黑方多 5 點
    pol = np.full(N + 1, 0.001); pol[idx] = 0.5; pol[0] = -1  # 第 0 格不合法
    f = dict(zip(bt.FEATURES, bt.move_features(resp(2.0, 0.6, 9.0, ob, pol), resp(1.0, 0.55, 11.0, oa),
                                               point, color, 7)))
    assert f['score_loss'] == pytest.approx(s * 1.0)
    assert f['winrate_loss'] == pytest.approx(s * 0.05)
    assert f['own_at_move'] == pytest.approx(s * 0.3)
    assert f['own_gain_signed'] == pytest.approx(s * 5.0)
    assert f['own_flip_l1'] == pytest.approx(5.0)
    assert f['dead_zone'] == 0.0
    assert idx == 60
    assert f['unsettled'] == pytest.approx((N - 20) / N)
    assert f['unsettled_delta'] == pytest.approx(-5 / N)
    assert f['score_stdev'] == 9.0 and f['stdev_delta'] == pytest.approx(2.0)
    assert f['norm_score_loss'] == pytest.approx(s * 1.0 / 10.0)
    assert f['log_prior'] == pytest.approx(np.log(0.5)) and f['is_top'] == 1.0
    assert f['turn'] == 7
    assert f['wr_mover'] == pytest.approx(0.6 if color == 'B' else 0.4)
    assert f['lead_mover'] == pytest.approx(s * 2.0)
    legal = pol[pol >= 0]
    assert f['policy_entropy'] == pytest.approx(-(legal * np.log(legal)).sum())


def test_move_features_dead_zone_and_pass():
    point = ah.sgf_point('dd')
    ob = np.zeros(N); ob[ah.policy_index(point)] = -0.95
    f = dict(zip(bt.FEATURES, bt.move_features(resp(own=ob), resp(own=ob), point, 'W', 3)))
    assert f['dead_zone'] == 1.0 and f['own_at_move'] == pytest.approx(0.95)
    f = dict(zip(bt.FEATURES, bt.move_features(resp(), resp(), None, 'B', 3)))
    assert np.isnan(f['own_at_move']) and np.isnan(f['dead_zone']) and np.isnan(f['log_prior'])
    assert len(bt.FEATURES) == 17


def game(sgf='(;SZ[19]HD[0];B[dd];W[pp];B[dp];W[pd];B[qq])', color='B'):
    return {'key': 'A_g1', 'rank_idx': 3, 'player_id': '0012', 'color': color, 'parsed': ah.parse_game(sgf)}


class FakeEngine:
    def __init__(self):
        self.queries = []

    def run(self, queries, keep=()):
        self.queries += queries
        return {q['id']: {t: resp(lead=float(t)) for t in q['analyzeTurns']} for q in queries}


def test_analyze_group_sides_and_padding():
    engine = FakeEngine()
    r = bt.analyze_group(engine, [game()], 'japanese', k=4)[0]
    assert engine.queries[0]['analyzeTurns'] == [0, 1, 2, 3, 4]
    assert r['turns'].tolist() == [[0, 2, -1, -1], [1, 3, -1, -1]]
    assert r['feats'].shape == (2, 4, 17) and np.isnan(r['feats'][:, 2:]).all()
    # 目標黑方第 0 手：lead 0 -> 1，黑方變好，損失 -1；對手白方第 1 手：lead 1 -> 2，白方損失 +1
    assert r['feats'][0, 0, 0] == pytest.approx(-1.0) and r['feats'][1, 0, 0] == pytest.approx(1.0)
    assert r['hd'] == 0 and r['player_id'] == '0012'


def test_analyze_group_white_target():
    r = bt.analyze_group(FakeEngine(), [game(color='W')], 'japanese', k=4)[0]
    assert r['turns'].tolist() == [[1, 3, -1, -1], [2, 4, -1, -1]]


def test_analyze_group_handicap_colors():
    # HD 2：棋譜標記和實際相反，parse_game 對調後第 0 手是白
    g = game('(;SZ[19]HD[2];B[qq];W[cc];B[qc];W[cq])', color='W')
    assert g['parsed']['moves'][0][0] == 'W'
    r = bt.analyze_group(FakeEngine(), [g], 'japanese', k=4)[0]
    assert r['turns'].tolist() == [[0, 2, -1, -1], [1, 3, -1, -1]]
    assert r['hd'] == 2


def test_save_and_load_parts(tmp_path):
    rows = bt.analyze_group(FakeEngine(), [game()], 'japanese')
    bt.save_part(str(tmp_path), rows)
    bt.save_part(str(tmp_path), rows)
    z = bt.load_parts(str(tmp_path))
    assert z['feats'].shape == (2, 2, 16, 17) and list(z['key']) == ['A_g1', 'A_g1']
    assert bt.load_parts(str(tmp_path / 'none')) is None
    assert not list(tmp_path.glob('*.tmp'))


def test_load_csv_games_keeps_string_ids(tmp_path):
    p = tmp_path / 'g.csv'
    pd.DataFrame({'player_id': ['0012'], 'game_id': ['g_0001'], 'rank': ['1D'], 'color': ['W'],
                  'sgf_content': ['(;SZ[19])']}).to_csv(p, index=False)
    g = bt.load_csv_games(str(p))[0]
    assert g['player_id'] == '0012' and g['key'] == '1D_g_0001' and g['rank_idx'] == 4


def test_sign_check_queries_and_check():
    qs = bt.sign_check_queries('japanese')
    assert [q['initialPlayer'] for q in qs] == ['B', 'W'] and qs[0]['moves'] == []
    assert len(qs[0]['initialStones']) == 13 and all(c == 'B' for c, _ in qs[0]['initialStones'])
    inside, _ = bt.corner_points()
    good = np.zeros(N); good[[ah.policy_index(p) for p in inside]] = 0.9
    ok = {'sign|B': {0: {'ownership': good}}, 'sign|W': {0: {'ownership': good}}}
    assert bt.check_sign(ok)['W'] == pytest.approx(0.9)
    flipped = {'sign|B': {0: {'ownership': good}}, 'sign|W': {0: {'ownership': -good}}}
    with pytest.raises(RuntimeError):
        bt.check_sign(flipped)


def test_engine_reports_black_and_skips_human_model(monkeypatch, tmp_path):
    seen = {}

    class P:
        def __init__(self, cmd, **kw):
            seen['cmd'] = cmd
            self.stdout = iter(())
            self.stdin = None

    monkeypatch.setattr(ah.subprocess, 'Popen', P)
    ah.Engine(str(tmp_path), 'b28.bin.gz', None, override='numAnalysisThreads=8')
    cmd = seen['cmd']
    assert '-human-model' not in cmd
    assert 'reportAnalysisWinratesAs=BLACK' in cmd[cmd.index('-override-config') + 1]
    ah.Engine(str(tmp_path), 'b28.bin.gz', 'h.bin.gz')
    assert seen['cmd'][seen['cmd'].index('-human-model') + 1] == 'h.bin.gz'
