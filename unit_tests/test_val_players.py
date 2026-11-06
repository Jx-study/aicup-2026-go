import pandas as pd
import pytest

from aicup.val_players import eligible_players, homogeneity, new_players, player_set


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
