import numpy as np
import pytest
import torch

from aicup import symmetry


def _board():
    return np.random.default_rng(0).integers(0, 2, (2, 3, 19, 19)).astype(np.float32)


def test_eight_transforms_are_distinct():
    x = _board()
    outs = [symmetry.apply(x, k).tobytes() for k in range(symmetry.N_SYMMETRIES)]
    assert len(set(outs)) == symmetry.N_SYMMETRIES


def test_identity_is_k0():
    x = _board()
    np.testing.assert_array_equal(symmetry.apply(x, 0), x)


def test_invert_restores_numpy_and_torch():
    x = _board()
    t = torch.from_numpy(x)
    for k in range(symmetry.N_SYMMETRIES):
        np.testing.assert_array_equal(symmetry.invert(symmetry.apply(x, k), k), x)
        assert torch.equal(symmetry.invert(symmetry.apply(t, k), k), t)


def test_torch_matches_numpy():
    x = _board()
    t = torch.from_numpy(x)
    for k in range(symmetry.N_SYMMETRIES):
        np.testing.assert_array_equal(symmetry.apply(t, k).numpy(), symmetry.apply(x, k))


def test_point_moves_with_board():
    # 單一點的位置要隨變換移動，且只作用在最後兩維
    x = np.zeros((1, 19, 19), np.float32)
    x[0, 0, 3] = 1
    seen = {tuple(np.argwhere(symmetry.apply(x, k))[0]) for k in range(symmetry.N_SYMMETRIES)}
    assert len(seen) == symmetry.N_SYMMETRIES
    assert all(p[0] == 0 for p in seen)


def test_numpy_output_works_with_from_numpy():
    x = _board()
    for k in range(symmetry.N_SYMMETRIES):
        torch.from_numpy(symmetry.apply(x, k))


def test_rejects_out_of_range_k():
    x = _board()
    for k in (8, -1):
        with pytest.raises(ValueError):
            symmetry.apply(x, k)
        with pytest.raises(ValueError):
            symmetry.invert(x, k)


def test_closed_under_composition():
    x = _board()
    outs = [symmetry.apply(x, c).tobytes() for c in range(symmetry.N_SYMMETRIES)]
    for a in range(symmetry.N_SYMMETRIES):
        for b in range(symmetry.N_SYMMETRIES):
            assert symmetry.apply(symmetry.apply(x, a), b).tobytes() in outs
