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
