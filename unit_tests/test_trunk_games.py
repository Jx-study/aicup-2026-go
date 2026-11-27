import numpy as np
import pandas as pd
import pytest

from aicup.trunk_games import COLS, question_games, subsets


def _df(prefix, n_players, n_games):
    return pd.DataFrame([{'player_id': '%s%d' % (prefix, p), 'game_id': '%s%d_%d' % (prefix, p, g), 'rank': 'D',
                          'color': 'B', 'sgf_content': ''} for p in range(n_players) for g in range(n_games)])


def _questions(keys):
    k = np.array([keys + [''] * (20 - len(keys))])
    return {'SC_keys': k, 'MX_keys': np.full((1, 20), '')}


def _newdev(tmp_path, ids):
    path = tmp_path / 'newdev.csv'
    pd.DataFrame({'player_id': ids, 'rank': 'D'}).to_csv(path, index=False)
    return str(path)


def test_subsets_caps_pool_and_keeps_eval_in_val(tmp_path):
    val, tr = _df('v', 3, 25), _df('t', 4, 10)
    out = subsets(val, tr, _questions(['D_v0_0', 'D_v1_3']), cap=2, epochs=3, newdev_csv=_newdev(tmp_path, ['v2']))
    assert list(out['train-games.csv'].columns) == COLS
    assert out['train-games.csv'].groupby('player_id').size().eq(6).all()
    assert set(out['newdev-games.csv']['player_id']) == {'v2'} and len(out['newdev-games.csv']) == 20
    assert set(out['question-games.csv']['game_id']) == {'v0_0', 'v1_3'}


def test_subsets_rejects_overlap_and_eval_outside_val(tmp_path):
    val, tr = _df('v', 2, 20), _df('t', 2, 20)
    with pytest.raises(SystemExit):
        subsets(val, pd.concat([tr, val.head(1)]), _questions(['D_v0_0']), 2, 3, _newdev(tmp_path, ['v1']))
    with pytest.raises(SystemExit):
        subsets(val, tr, _questions(['D_t0_0']), 2, 3, _newdev(tmp_path, ['v1']))


def test_question_games_missing_key():
    with pytest.raises(SystemExit):
        question_games(_questions(['D_v0_0', 'D_x_9']), _df('v', 1, 5))
