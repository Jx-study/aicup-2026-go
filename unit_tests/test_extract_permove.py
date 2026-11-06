import argparse
import json
import os

import numpy as np
import pandas as pd
import pytest

from aicup import extract_permove as ep
from aicup import katago_torch as kt
from aicup.analyze_humansl import parse_game, target_turns
from aicup.scoring import LEVELS

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.environ.get('KATAGO_PYTHON') or kt.DEFAULT_KATAGO_PYTHON, 'features.py')),
    reason='沒有 KataGo python/ 原始碼')

SGF = '(;SZ[19]HD[0];B[pd];W[dp];B[pp];W[dd];B[fc];W[cf];B[qj];W[jq];B[nq];W[qn])'
SGF_HD = '(;SZ[19]HD[2];B[qc];W[cc];B[qq];W[ce])'


def game(key='D_g_1', color='B', sgf=SGF, player='p1'):
    return {'key': key, 'rank_idx': 0, 'player_id': player, 'color': color, 'sgf': sgf}


def toy_df(players_per_rank=4, games_per_player=5):
    rows = []
    for rank in LEVELS:
        for p in range(players_per_rank):
            for g in range(games_per_player):
                rows.append({'player_id': 'p%d' % p, 'game_id': 'g_%s_%d_%d' % (rank, p, g), 'rank': rank,
                             'color': 'B', 'sgf_content': SGF})
    return pd.DataFrame(rows)


def test_pack_roundtrip():
    parsed = parse_game(SGF_HD)
    enc = kt.encode_game(parsed, [0, 1, 2, 3])
    packed = ep.pack_positions(enc)
    assert packed['spatial'].shape == (4, ep.SPATIAL_BYTES) and packed['legal'].shape == (4, ep.LEGAL_BYTES)
    np.testing.assert_array_equal(ep.unpack_spatial(packed['spatial']), enc['spatial'])
    np.testing.assert_array_equal(ep.unpack_legal(packed['legal']), enc['legal'])
    np.testing.assert_array_equal(packed['move'], enc['move'])
    np.testing.assert_array_equal(packed['global'], enc['global'])


def test_sample_spike_games_caps():
    df = toy_df(players_per_rank=4, games_per_player=5)
    picked, caps = ep.sample_spike_games(df, per_rank=8, max_games=2)
    assert len(picked) == 8 * len(LEVELS)
    assert set(caps.values()) == {2}
    per_player = picked.groupby(['rank', 'player_id']).size()
    assert per_player.max() <= 2
    # 2 局上限湊不到 12 局時放寬成 4 局
    picked, caps = ep.sample_spike_games(df, per_rank=12, max_games=2)
    assert set(caps.values()) == {4} and picked.groupby(['rank', 'player_id']).size().max() <= 4
    with pytest.raises(SystemExit):
        ep.sample_spike_games(df, per_rank=21, max_games=2)
    # 同一個 seed 結果相同
    a, _ = ep.sample_spike_games(df, per_rank=8, max_games=2)
    b, _ = ep.sample_spike_games(df, per_rank=8, max_games=2)
    assert a['game_id'].tolist() == b['game_id'].tolist()


def test_split_holdout_by_player():
    df = toy_df(players_per_rank=10, games_per_player=3)
    train, holdout = ep.split_holdout(df, 0.05)
    assert not ep.player_set(train) & ep.player_set(holdout)
    assert len(ep.player_set(holdout)) == int(np.ceil(100 * 0.05))
    assert len(train) + len(holdout) == len(df)


def test_sampled_moves_are_target_side():
    for color in ('B', 'W'):
        g = ep.game_job(game(color=color), {0: ('sample', 3, (42, 0))})
        out, why = ep.encode_job(g)
        assert why is None
        moves = parse_game(SGF)['moves']
        assert all(moves[t][0] == color for t in out['turn'])
        assert len(out['turn']) == 3 and set(out['side']) == {0}
        assert set(out['turn']) <= set(target_turns(moves, color))


def test_both_sides_and_metadata():
    moves = parse_game(SGF_HD)['moves']
    g = ep.game_job(game(color='W', sgf=SGF_HD), {0: target_turns(moves, 'W'), 1: target_turns(moves, 'B')})
    out, why = ep.encode_job(g)
    assert why is None
    assert out['turn'].tolist() == [0, 1, 2, 3]
    assert out['side'].tolist() == [0, 1, 0, 1]  # 讓子局第一手是白
    assert set(out['hd']) == {2} and set(out['color']) == {1}
    assert out['key'].tolist() == ['D_g_1'] * 4


def test_bad_game_dropped():
    out, why = ep.encode_job(ep.game_job(game(sgf='not sgf'), {0: ('sample', 3, (42, 0))}))
    assert out is None and why.startswith('parse')
    bad = '(;SZ[19];B[pd];B[dp];W[pp])'
    out, why = ep.encode_job(ep.game_job(game(sgf=bad, color='W'), {0: [2]}))
    assert out is None and why.startswith('encode')


def test_extract_split_resume_and_summary(tmp_path):
    games = [ep.game_job(game(key='D_g_%d' % i, player='p%d' % (i % 2)), {0: ('sample', 2, (42, i))})
             for i in range(5)]
    games.append(ep.game_job(game(key='D_g_bad', sgf='bad'), {0: ('sample', 2, (42, 9))}))
    summary = ep.extract_split(games, str(tmp_path / 'train'), 'train', nproc=1, per_shard=2)
    assert summary['shards'] == 3 and summary['games'] == 5 and summary['players'] == 2
    assert summary['positions'] == 10 and summary['dropped_games'] == 1
    assert summary['positions_target'] == 10 and summary['positions_opponent'] == 0
    first = tmp_path / 'train' / 'train-00000.npz'
    mtime = first.stat().st_mtime
    ep.extract_split(games, str(tmp_path / 'train'), 'train', nproc=1, per_shard=2)
    assert first.stat().st_mtime == mtime  # 已存在的分片跳過
    with np.load(first) as z:
        assert set(z.files) == set(ep.FIELDS)
    assert json.load(open(tmp_path / 'train' / 'summary.json'))['positions'] == 10


def test_resume_after_crash_between_writes(tmp_path):
    games = [ep.game_job(game(key='D_g_%d' % i), {0: ('sample', 2, (42, i))}) for i in range(2)]
    split_dir = tmp_path / 'train'
    ep.extract_split(games, str(split_dir), 'train', nproc=1, per_shard=2)
    # 模擬寫完丟棄紀錄、改名前中斷：只留 .dropped.json 與 .tmp
    npz = split_dir / 'train-00000.npz'
    os.replace(npz, str(npz) + '.tmp')
    summary = ep.extract_split(games, str(split_dir), 'train', nproc=1, per_shard=2)
    assert summary['positions'] == 4 and summary['dropped_games'] == 0


def test_read_turns_skips_blank_lines(tmp_path):
    p = tmp_path / 'a.jsonl'
    p.write_text('{"key": "a", "turns": [1]}\n\n{"key": "b", "turns": [2]}\n', encoding='utf-8')
    assert list(ep.read_turns(str(p))) == ['a', 'b']


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def val648_args(tmp_path, **kw):
    args = {'keys_from': str(tmp_path / 'target.jsonl'), 'opp_from': str(tmp_path / 'opp.jsonl'),
            'test_dir': None, 'csv': 'unused', 'out': str(tmp_path / 'out'), 'nproc': 1, 'games_per_shard': 2}
    args.update(kw)
    return argparse.Namespace(**args)


def setup_val648(tmp_path, monkeypatch, sgf_of=None):
    """兩局的 jsonl 與假的 load_keyed_games；回傳 games。"""
    sgf_of = sgf_of or {}
    games = [game(key='D_g_%d' % i, color=c, player='p%d' % i, sgf=sgf_of.get(i, SGF))
             for i, c in enumerate('BW')]
    moves = parse_game(SGF)['moves']
    write_jsonl(tmp_path / 'target.jsonl',
                [{'key': g['key'], 'turns': target_turns(moves, g['color'])} for g in games])
    write_jsonl(tmp_path / 'opp.jsonl',
                [{'key': g['key'], 'turns': target_turns(moves, ep.OPP[g['color']])} for g in games])
    import aicup.strength_se
    monkeypatch.setattr(aicup.strength_se, 'load_keyed_games', lambda path, csv: [dict(g) for g in games])
    return games


def test_run_val648_both_sides(tmp_path, monkeypatch):
    setup_val648(tmp_path, monkeypatch)
    ep.run_val648(val648_args(tmp_path))
    summary = json.load(open(tmp_path / 'out' / 'val' / 'summary.json'))
    assert summary['positions_target'] == 10 and summary['positions_opponent'] == 10  # 10 手各 5 手
    assert summary['games'] == 2 and summary['players'] == 2


def test_run_val648_count_mismatch(tmp_path, monkeypatch):
    setup_val648(tmp_path, monkeypatch, sgf_of={1: 'bad'})
    with pytest.raises(SystemExit):
        ep.run_val648(val648_args(tmp_path))


def test_run_val648_key_mismatch(tmp_path, monkeypatch):
    setup_val648(tmp_path, monkeypatch)
    write_jsonl(tmp_path / 'opp.jsonl', [{'key': 'other', 'turns': [1]}])
    with pytest.raises(SystemExit):
        ep.run_val648(val648_args(tmp_path))


def test_run_val648_test_dir(tmp_path, monkeypatch):
    games = setup_val648(tmp_path, monkeypatch)
    import aicup.analyze_humansl
    test_games = [dict(g, rank_idx=-1) for g in games] + [dict(games[0], key='extra')]
    monkeypatch.setattr(aicup.analyze_humansl, 'load_test_games', lambda d: test_games)
    ep.run_val648(val648_args(tmp_path, test_dir='tests'))
    with np.load(tmp_path / 'out' / 'val' / 'val-00000.npz') as z:
        assert set(z['rank_idx'].tolist()) == {-1}
        assert set(z['key'].tolist()) == {'D_g_0', 'D_g_1'}  # 只抽 jsonl 有的局
    monkeypatch.setattr(aicup.analyze_humansl, 'load_test_games', lambda d: test_games[:1])
    with pytest.raises(SystemExit):  # jsonl 的局在測試題找不到
        ep.run_val648(val648_args(tmp_path, test_dir='tests', out=str(tmp_path / 'out2')))
