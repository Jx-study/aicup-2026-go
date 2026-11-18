import json

import numpy as np
import pytest

from aicup import feature_cache

P2 = ('rank_11k', 'rank_1d', 'rank_2d')
P6 = ('rank_11k', 'rank_1d', 'rank_6d')


def fake(keys, players, profiles=P2, feature_profiles=P2):
    n = len(keys)
    data = {'x': np.arange(n * 2, dtype=float).reshape(n, 2), 'rank_idx': np.zeros(n, np.int64),
            'player_id': np.array(players), 'key': np.array(keys),
            'n_profiles': 3, 'profiles': profiles, 'has_loss': True, 'feature_profiles': feature_profiles}
    games = {'hsl': np.ones((n, 4)), 'loss': np.full((n, 8), 2.0)}
    return data, games


def test_cache_roundtrip(tmp_path, monkeypatch):
    t, o = tmp_path / 't.jsonl', tmp_path / 'o.jsonl'
    t.write_text('x', encoding='utf-8')
    o.write_text('y', encoding='utf-8')
    seen = []
    monkeypatch.setattr(feature_cache, 'load_game_features',
                        lambda a, b, profiles: seen.append(profiles) or fake(['k1', 'k2'], ['p', 'p']))
    out = tmp_path / 'c.npz'
    feature_cache.build(str(t), str(o), str(out), P6)
    assert seen == [P6]
    data, games = feature_cache.load(str(out))
    assert list(data['key']) == ['k1', 'k2'] and data['profiles'] == P2 and data['feature_profiles'] == P6
    assert data['has_loss'] is True and set(games) == {'hsl', 'loss'}
    t.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError):
        feature_cache.load(str(out))


def test_concat_rejects_duplicates_and_profiles():
    a, b = fake(['k1'], ['p']), fake(['k2'], ['q'])
    data, games = feature_cache.concat([a, b])
    assert list(data['key']) == ['k1', 'k2'] and games['hsl'].shape == (2, 4)
    assert data['feature_profiles'] == P2
    with pytest.raises(ValueError):
        feature_cache.concat([a, fake(['k1'], ['q'])])
    with pytest.raises(ValueError):
        feature_cache.concat([a, fake(['k3'], ['q'], P6, P6)])
    with pytest.raises(ValueError):
        feature_cache.concat([a, fake(['k3'], ['q'], P2, P6)])


def with_color(pair):
    data, games = pair
    n = len(data['key'])
    data = dict(data, black=np.arange(n) % 2 == 0, hd=np.arange(n, dtype=np.int64))
    return data, games


def test_color_arrays_roundtrip_and_concat(tmp_path, monkeypatch):
    t, o = tmp_path / 't.jsonl', tmp_path / 'o.jsonl'
    t.write_text('x', encoding='utf-8')
    o.write_text('y', encoding='utf-8')
    monkeypatch.setattr(feature_cache, 'load_game_features', lambda a, b, profiles: with_color(fake(['k1', 'k2'], ['p', 'p'])))
    out = tmp_path / 'c.npz'
    feature_cache.build(str(t), str(o), str(out))
    data, games = feature_cache.load(str(out))
    assert data['black'].tolist() == [True, False] and data['hd'].tolist() == [0, 1]
    both, _ = feature_cache.concat([(data, games), with_color(fake(['k3'], ['q']))])
    assert both['black'].tolist() == [True, False, True] and both['hd'].tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        feature_cache.concat([(data, games), fake(['k3'], ['q'])])


def test_cache_without_color_has_no_color_arrays(tmp_path, monkeypatch):
    t, o = tmp_path / 't.jsonl', tmp_path / 'o.jsonl'
    t.write_text('x', encoding='utf-8')
    o.write_text('y', encoding='utf-8')
    monkeypatch.setattr(feature_cache, 'load_game_features', lambda a, b, profiles: fake(['k1'], ['p']))
    out = tmp_path / 'c.npz'
    feature_cache.build(str(t), str(o), str(out))
    assert 'black' not in feature_cache.load(str(out))[0]


def write_jsonl(path, rows):
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')


def extra_row(key, rank_idx=0, side='opponent', profiles=('rank_11k', 'rank_5k'), history=True):
    # 每手機率：第一個 profile 0.5，第二個 0.25
    return {'key': key, 'rank_idx': rank_idx, 'player_id': 'p', 'side': side, 'turns': [0, 2],
            'profiles': list(profiles), 'probs': [[0.5, 0.5], [0.25, 0.25]], 'pre_root_history': history}


def main_opp(tmp_path, keys=('k1', 'k2'), **kw):
    path = tmp_path / 'main.jsonl'
    write_jsonl(path, [extra_row(k, profiles=P2[:2], **kw) for k in keys])
    return str(path)


def test_load_opp_extra_takes_new_profiles_in_data_order(tmp_path):
    data = {'key': np.array(['k2', 'k1']), 'rank_idx': np.array([0, 0]), 'profiles': P2}
    path = tmp_path / 'x.jsonl'
    write_jsonl(path, [extra_row('k1'), extra_row('k2')])
    x, new = feature_cache.load_opp_extra(str(path), data, main_opp(tmp_path))
    assert new == ['rank_5k'] and x.shape == (2, 1)
    assert np.allclose(x[:, 0], np.log(0.25))


@pytest.mark.parametrize('rows', [
    [extra_row('k1')],
    [extra_row('k1'), extra_row('k2', rank_idx=1)],
    [extra_row('k1'), extra_row('k2', side='target')],
    [extra_row('k1', profiles=P2[:2]), extra_row('k2', profiles=P2[:2])],
    [extra_row('k1'), extra_row('k2', history=False)],
    [extra_row('k1'), dict(extra_row('k2'), turns=[0, 2, 4])],
    [extra_row('k1'), extra_row('k2', profiles=('rank_5k', 'rank_11k'))],
])
def test_load_opp_extra_rejects(tmp_path, rows):
    data = {'key': np.array(['k1', 'k2']), 'rank_idx': np.array([0, 0]), 'profiles': P2}
    path = tmp_path / 'x.jsonl'
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        feature_cache.load_opp_extra(str(path), data, main_opp(tmp_path))


def test_build_with_extra_group(tmp_path, monkeypatch):
    t, o, e = tmp_path / 't.jsonl', tmp_path / 'o.jsonl', tmp_path / 'e.jsonl'
    t.write_text('x', encoding='utf-8')
    o.write_text('y', encoding='utf-8')
    e.write_text('z', encoding='utf-8')
    monkeypatch.setattr(feature_cache, 'load_game_features', lambda a, b, profiles: fake(['k1', 'k2'], ['p', 'p']))
    monkeypatch.setattr(feature_cache, 'load_opp_extra',
                        lambda path, data, history: (np.full((2, 3), -1.0), ['a', 'b', 'c']))
    out = tmp_path / 'c.npz'
    feature_cache.build(str(t), str(o), str(out), opp_extra=str(e), extra_group='hslx')
    data, games = feature_cache.load(str(out))
    assert games['hslx'].shape == (2, 3) and data['extra'] == {'group': 'hslx', 'profiles': ['a', 'b', 'c']}
    e.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError):
        feature_cache.load(str(out))
    with pytest.raises(ValueError):
        feature_cache.build(str(t), str(o), str(out), opp_extra=str(e), extra_group='bad')
    with pytest.raises(ValueError):
        feature_cache.build(str(t), str(o), str(out), opp_extra=str(e))


def test_concat_rejects_different_extra():
    a, b = fake(['k1'], ['p']), fake(['k2'], ['q'])
    a[0]['extra'] = {'group': 'hslx', 'profiles': ['rank_2d']}
    b[0]['extra'] = {'group': 'hslx', 'profiles': ['rank_6d']}
    with pytest.raises(ValueError):
        feature_cache.concat([a, b])
    b[0]['extra'] = dict(a[0]['extra'])
    assert feature_cache.concat([a, b])[0]['extra'] == a[0]['extra']


def test_load_opp_extra_rejects_history_mismatch_per_row(tmp_path):
    data = {'key': np.array(['k1', 'k2']), 'rank_idx': np.array([0, 0]), 'profiles': P2}
    path = tmp_path / 'x.jsonl'
    write_jsonl(path, [extra_row('k1'), extra_row('k2')])
    with pytest.raises(ValueError, match='pre_root_history'):
        feature_cache.load_opp_extra(str(path), data, main_opp(tmp_path, history=False))


def test_check_extra_args():
    feature_cache.check_extra_args(None, None)
    feature_cache.check_extra_args('e.jsonl', 'hslx2')
    for args in (('e.jsonl', None), (None, 'hslx'), ('e.jsonl', 'bad')):
        with pytest.raises(ValueError):
            feature_cache.check_extra_args(*args)
