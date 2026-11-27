import pandas as pd

from aicup.analyze_humansl import sample_listed_players
from aicup.hsl_games import GAME_COLS, PER_PLAYER, RUNS, SEED, select_games


def _fake_df():
    rows = [{'player_id': 'p%d' % p, 'game_id': 'g_%d_%d' % (p, g), 'rank': ('D', 'A')[p % 2],
             'color': 'BW'[g % 2], 'sgf_content': '(;SZ[19])'}
            for p in range(12) for g in range(25 + p)]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


def _players(ids):
    return pd.DataFrame({'player_id': ['p%d' % p for p in ids], 'rank': [('D', 'A')[p % 2] for p in ids]})


def test_select_games_columns_and_counts():
    games = select_games(_fake_df(), _players(range(12)), _players(range(12)))
    assert list(games.columns) == GAME_COLS
    assert len(games) == 12 * PER_PLAYER
    assert games.groupby('player_id').size().eq(PER_PLAYER).all()


def test_select_games_keep_is_subset_of_full_sample():
    df = _fake_df()
    full = sample_listed_players(df, _players(range(12)), PER_PLAYER, SEED)
    kept = select_games(df, _players(range(12)), _players([1, 4, 7]))
    expected = full[full['player_id'].isin(['p1', 'p4', 'p7'])]
    assert list(kept['game_id']) == list(expected['game_id'])


def test_select_games_depends_on_sample_list():
    # 抽樣清單不同時同一位玩家抽到的局不同，所以 newdev 要對整份新增清單抽樣
    df = _fake_df()
    wide = select_games(df, _players(range(12)), _players([4]))
    narrow = select_games(df, _players([4]), _players([4]))
    assert set(wide['player_id']) == set(narrow['player_id']) == {'p4'}
    assert len(wide) == len(narrow) == PER_PLAYER
    assert list(wide['game_id']) != list(narrow['game_id'])


def test_newdev_samples_over_new_players():
    assert RUNS['newdev'] == ('val-players-new-v180000.csv', 'val-players-newdev-v180000.csv')
