import numpy as np
import pytest

from aicup.val_players import val_players


def test_val_players_counts_by_player_and_rank(tmp_path):
    # p1 在 D 有 3 局、在 C 有 1 局，p2 在 C 有 2 局
    np.savez(tmp_path / 'v.npz', player_id=np.array(['p1', 'p1', 'p2', 'p1', 'p2', 'p1']),
             rank_idx=np.array([0, 0, 1, 0, 1, 1]), key=np.arange(6), probs=np.zeros((6, 10)))
    out = val_players(str(tmp_path / 'v.npz'), 2)
    assert out.values.tolist() == [['p2', 'C', 2], ['p1', 'D', 3]]


def test_val_players_requires_player_id(tmp_path):
    np.savez(tmp_path / 'v.npz', rank_idx=np.array([0]), key=np.arange(1))
    with pytest.raises(ValueError, match='player_id'):
        val_players(str(tmp_path / 'v.npz'), 1)
