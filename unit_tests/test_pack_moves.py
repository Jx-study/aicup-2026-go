import numpy as np

from aicup.pack_moves import parse_move_features


def _row(T, color, hd):
    return {
        'turns': list(range(T)),
        'probs': [[0.5] * T, [0.25] * T, [0.1] * T],
        'pmax': [[0.9] * T, [0.8] * T, [0.7] * T],
        'entropy': [[1.0] * T, [2.0] * T, [3.0] * T],
        'color': color,
        'hd': hd,
    }


def test_parse_move_features_shape_and_meta():
    for T, color, hd in ((120, 'B', 1), (60, 'W', 0)):
        feats, mask = parse_move_features(_row(T, color, hd), _row(T, 'W', 0))
        assert feats.shape == (200, 14)
        assert mask.shape == (200,) and mask.sum() == T
        assert (feats[:T, 11] == float(color == 'B')).all()
        assert (feats[:T, 12] == float(hd == 1)).all()
        assert (feats[T:] == 0).all()
        np.testing.assert_allclose(feats[0, 13], 1 / np.log(T + 2), rtol=1e-6)
        np.testing.assert_allclose(feats[0, 1], np.log(0.25), rtol=1e-6)


def test_parse_move_features_truncates_and_missing_opp():
    feats, mask = parse_move_features(_row(300, 'B', 1), None)
    assert mask.all() and feats.shape == (200, 14)
    assert np.isfinite(feats).all()
