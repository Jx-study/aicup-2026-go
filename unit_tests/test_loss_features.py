import numpy as np

from aicup.loss_features import LOSS_COLUMNS, N_LOSS, game_loss_features


def col(name):
    return LOSS_COLUMNS.index(name)


def test_basic_values():
    row = {'score_loss': [-1.0, 0.0, 3.0, 6.0, 20.0], 'winrate_loss': [0.0, -0.1, 0.2, 0.05, 0.3]}
    f = game_loss_features(row)
    assert f.shape == (N_LOSS,)
    assert np.isclose(f[col('mean_loss')], 28.0 / 5)
    assert np.isclose(f[col('mean_pos_loss')], 29.0 / 5)
    assert np.isclose(f[col('blunder_rate_2')], 3 / 5)
    assert np.isclose(f[col('blunder_rate_5')], 2 / 5)
    assert np.isclose(f[col('blunder_rate_15')], 1 / 5)
    assert np.isclose(f[col('winrate_drop_mean')], 0.55 / 5)
    assert np.isclose(f[col('winrate_blunder_rate')], 2 / 5)


def test_none_and_nan_moves_skipped():
    f = game_loss_features({'score_loss': [None, float('nan'), 4.0], 'winrate_loss': [None, 0.2, float('nan')]})
    assert np.isclose(f[col('mean_loss')], 4.0)
    assert np.isclose(f[col('winrate_blunder_rate')], 1.0)


def test_missing_or_empty_is_nan():
    assert np.isnan(game_loss_features({})).all()
    assert np.isnan(game_loss_features({'score_loss': [], 'winrate_loss': []})).all()
    f = game_loss_features({'score_loss': [1.0]})
    assert np.isfinite(f[:col('winrate_drop_mean')]).all()
    assert np.isnan(f[col('winrate_drop_mean'):]).all()
