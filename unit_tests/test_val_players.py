import pandas as pd
import pytest

from aicup.scoring import LEVELS
from aicup.val_players import (eligible_players, holdout_split, homogeneity, new_players, player_set, quotas,
                               train_pool, write_players)


def _fake_df():
    rows = [{'player_id': 'p%d' % p, 'game_id': 'g_%d_%d' % (p, g), 'rank': ('D', '3D', 'A')[p % 3],
             'color': 'B', 'sgf_content': ''}
            for p in range(300) for g in range(5 + p % 9 * 5)]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_eligible_players_min_games():
    df = _fake_df()
    players = eligible_players(df, 3000, min_games=20)
    counts = df.groupby(['rank', 'player_id']).size()
    assert all(counts[(r, p)] >= 20 for p, r in zip(players['player_id'], players['rank']))
    assert list(players.columns) == ['player_id', 'rank']


def test_new_players_disjoint_and_union():
    df = _fake_df()
    base, full, new = new_players(df, 2000, 5000, min_games=20)
    assert len(new) > 0
    assert not player_set(base) & player_set(new)
    assert player_set(base) | player_set(new) == player_set(full)


def test_new_players_rejects_smaller_val():
    with pytest.raises(ValueError):
        new_players(_fake_df(), 5000, 2000)


def test_homogeneity_same_distribution():
    a = pd.DataFrame({'player_id': ['x%d' % i for i in range(30)], 'rank': ['D', 'A', '3D'] * 10})
    b = pd.DataFrame({'player_id': ['y%d' % i for i in range(30)], 'rank': ['D', 'A', '3D'] * 10})
    _, p = homogeneity(a, b)
    assert p > 0.99


def _level_df(n_per_level, prefix):
    return pd.DataFrame([(f'{prefix}{lv}_{i}', lv) for lv in LEVELS for i in range(n_per_level)],
                        columns=['player_id', 'rank'])


def test_quotas_sum_and_tie_break():
    assert quotas([1, 1, 1], 2) == [1, 1, 0]
    assert sum(quotas([57, 56, 60, 71], 30)) == 30


def test_holdout_split_disjoint_and_stable():
    new, base = _level_df(10, 'n'), _level_df(5, 'b')
    hold, newdev, dev = holdout_split(new, base, n_holdout=20)
    assert len(hold) == 20 and len(newdev) == len(new) - 20
    assert not set(hold['player_id']) & set(newdev['player_id'])
    assert list(dev['player_id']) == list(base['player_id']) + list(newdev['player_id'])
    hold2, _, _ = holdout_split(new, base, n_holdout=20)
    assert hold.equals(hold2)


def test_train_pool_per_level_and_excludes_val():
    df = pd.DataFrame([{'player_id': 'p%d' % p, 'game_id': 'g_%d_%d' % (p, g), 'rank': LEVELS[p % 10],
                        'color': 'B', 'sgf_content': ''} for p in range(300) for g in range(20 + p % 3)])
    pool = train_pool(df, 1000, per_level=5, min_games=20)
    assert (pool.groupby('rank').size() == 5).all()
    assert not player_set(pool) & player_set(eligible_players(df, 1000, min_games=20))


def test_write_players_lf(tmp_path):
    path = tmp_path / 'p.csv'
    write_players(path, pd.DataFrame({'player_id': ['a', 'b'], 'rank': ['D', '1D']}))
    assert path.read_bytes() == b'player_id,rank\na,D\nb,1D\n'
