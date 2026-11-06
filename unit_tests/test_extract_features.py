import pandas as pd

from aicup.extract_features import split_by_game, split_by_player


def _fake_df(n_players=50, games_each=20):
    rows = []
    for lv in ('D', '3D'):
        for p in range(n_players):
            for g in range(games_each):
                rows.append({'player_id': 'p%d' % p, 'game_id': 'g_%d_%d' % (p, g),
                             'rank': lv, 'color': 'B', 'sgf_content': ''})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_split_by_game_matches_notebook():
    df = _fake_df()
    val, train = split_by_game(df, 300)
    assert len(val) == 300 and len(train) == len(df) - 300
    assert val.index.tolist() == list(range(300))


def test_split_by_player_no_overlap():
    df = _fake_df()
    val, train = split_by_player(df, 300)
    key = lambda d: set(d['rank'] + '/' + d['player_id'])
    assert not key(val) & key(train)
    assert len(val) + len(train) == len(df)
    assert 300 <= len(val) < 300 + 20  # 最多多出一位玩家的局數


def test_split_by_player_size_with_uneven_players():
    # 每位玩家局數不同時，累計局數要對到正確的玩家
    rows = [{'player_id': 'p%d' % p, 'game_id': 'g_%d_%d' % (p, g),
             'rank': 'D', 'color': 'B', 'sgf_content': ''}
            for p in range(200) for g in range(1 + p % 7 * 30)]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    val, _ = split_by_player(df, 2000)
    assert 2000 <= len(val) < 2000 + 181  # 最多多出一位玩家的局數


def test_split_by_player_larger_val_is_superset():
    # 玩家順序與 val_size 無關，較大的 val 只會在後面多加玩家
    rows = [{'player_id': 'p%d' % p, 'game_id': 'g_%d_%d' % (p, g),
             'rank': ('D', '3D')[p % 2], 'color': 'B', 'sgf_content': ''}
            for p in range(300) for g in range(1 + p % 11 * 4)]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    key = lambda d: set(d['rank'] + '/' + d['player_id'])
    sizes = (500, 1500, 3000, 5000)
    vals = [key(split_by_player(df, n)[0]) for n in sizes]
    for small, large in zip(vals, vals[1:]):
        assert small < large


def test_split_by_player_whole_players():
    df = _fake_df()
    val, _ = split_by_player(df, 300)
    # 同一 player_id 在不同等級是不同玩家，val 只會包含完整玩家
    assert len(val) % 20 == 0
