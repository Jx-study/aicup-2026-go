import pandas as pd

from aicup.analyze_humansl import sample_listed_players
from aicup.hsl_games import (GAME_COLS, PER_PLAYER, RUNS, RUNS_2D, SEED, same_color_games, select_games,
                              topup_games)
from aicup.val_players import hd_of


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


def _color_df():
    # p0 黑 14 局、白 3 局；p1 白 12 局；p2 黑白各 11 局，其中黑 1 局讓 2 子
    rows = []
    for pid, rank, color, n in (('p0', 'D', 'B', 14), ('p0', 'D', 'W', 3), ('p1', 'A', 'W', 12),
                                ('p2', '1D', 'B', 11), ('p2', '1D', 'W', 11)):
        rows += [{'player_id': pid, 'game_id': '%s_%s_%d' % (pid, color, i), 'rank': rank, 'color': color,
                  'sgf_content': 'HD[2]' if (pid, color, i) == ('p2', 'B', 0) else ''} for i in range(n)]
    df = pd.DataFrame(rows)
    df['key'] = df['rank'] + '_' + df['game_id']
    df['hd'] = df['sgf_content'].map(hd_of)
    return df


def test_same_color_games_majority_color():
    games = same_color_games(_color_df(), _players_pairs([('p0', 'D'), ('p1', 'A'), ('p2', '1D')]), k=10)
    assert list(games.columns) == GAME_COLS
    by = games.groupby('player_id')['color'].agg(set).to_dict()
    assert by == {'p0': {'B'}, 'p1': {'W'}, 'p2': {'W'}}  # p2 黑只剩 10 局可用，白 11 局較多
    assert (games.groupby('player_id').size() == 10).all()


def test_topup_games_same_color_unused():
    df = _color_df()
    first = same_color_games(df, _players_pairs([('p0', 'D'), ('p1', 'A')]), k=4)
    more = topup_games(df, first, k=8)
    assert set(more['player_id']) == {'p0', 'p1'}  # p0 黑剩 10 局、p1 白剩 8 局
    assert not set(more['game_id']) & set(first['game_id'])
    assert set(zip(more['player_id'], more['color'])) == {('p0', 'B'), ('p1', 'W')}
    assert set(topup_games(df, first, k=9)['player_id']) == {'p0'}


def _players_pairs(pairs):
    return pd.DataFrame(pairs, columns=['player_id', 'rank'])


def test_runs_2d_lists():
    assert RUNS_2D['holdout'] == ('val-players-new-v180000.csv', 'holdout-v180000.csv')
    assert RUNS_2D['val648'] == RUNS['val648']
