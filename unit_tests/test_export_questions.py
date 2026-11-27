import numpy as np
import pandas as pd

from aicup.export_questions import MAX_GAMES, export, load_games
from aicup.scoring import LEVELS


def _games_csv(tmp_path):
    rows = []
    for p in range(100):
        rank = LEVELS[p % 10]
        for g in range(20):
            rows.append({'player_id': '%04d' % p, 'game_id': '%d' % (p * 100 + g), 'rank': rank,
                         'color': 'BW'[g % 2], 'sgf_content': '(;HD[2])' if g == 0 else '(;KM[6.5])'})
    path = tmp_path / 'games.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, {'%d|%04d' % (p % 10, p) for p in range(0, 100, 2)}


def test_load_games_keeps_string_ids_and_hd(tmp_path):
    path, _ = _games_csv(tmp_path)
    d = load_games(path)
    assert d['player_id'][0] == '0000' and d['key'][0] == 'D_0'
    assert d['hd'][0] == 2 and d['hd'][1] == 0
    assert d['black'][0] and not d['black'][1]


def test_export_only_eval_players_and_consistent(tmp_path):
    path, players = _games_csv(tmp_path)
    data = load_games(path)
    q = export(data, players, 42)
    for t in ('SC', 'MX'):
        assert q[t + '_keys'].shape == (len(q[t + '_n']), MAX_GAMES)
        assert set(q[t + '_p']) <= players
        assert set(q[t + '_fold']) <= set(range(5))
        for keys, n, y in zip(q[t + '_keys'], q[t + '_n'], q[t + '_y']):
            assert (keys[:n] != '').all() and (keys[n:] == '').all()
            assert {LEVELS.index(k.split('_')[0]) for k in keys[:n]} == {y}
    hd2 = set(data['key'][data['hd'] >= 2])
    assert not hd2 & set(q['SC_keys'].ravel()) and not hd2 & set(q['MX_keys'].ravel())
    again = export(data, players, 42)
    assert all(np.array_equal(q[k], again[k]) for k in q)
