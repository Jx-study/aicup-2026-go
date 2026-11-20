import numpy as np
import pandas as pd
import pytest
import torch

from aicup import trunk_cls as tc


def _df():
    return pd.DataFrame({'sgf_content': ['(;SZ[19]HD[0];B[aa])', '(;SZ[19]HD[0];B[bb];W[cc])', '(;SZ[19]中)'],
                         'rank': ['D', '6D', 'D'], 'player_id': ['p1', 'p1', 'p2'], 'color': ['B', 'W', 'B'],
                         'game_id': ['g1', 'g2', 'g3']})


def test_game_store_roundtrip():
    s = tc.GameStore(_df())
    assert [s.sgf(i) for i in range(3)] == list(_df()['sgf_content'])
    assert s.rank_idx.tolist() == [0, 9, 0]
    assert s.player.tolist() == [0, 1, 2]  # 同一 player_id 在不同等級是不同玩家
    assert s.color.tolist() == [0, 1, 0]
    assert s.key.tolist() == ['D_g1', '6D_g2', 'D_g3']


def test_epoch_games_cap_and_determinism():
    player = np.repeat(np.arange(5), [1, 3, 10, 25, 2])
    a = tc.epoch_games(player, 4, 42, 0)
    assert np.bincount(player[a], minlength=5).tolist() == [1, 3, 4, 4, 2]
    assert len(set(a.tolist())) == len(a)
    assert np.array_equal(a, tc.epoch_games(player, 4, 42, 0))
    assert not np.array_equal(a, tc.epoch_games(player, 4, 42, 1))


def test_class_weights():
    w = tc.class_weights(np.array([0, 0, 0, 1]), n=2)
    assert w.mean() == pytest.approx(1.0)
    assert w[1] / w[0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        tc.class_weights(np.array([0, 0]), n=2)


def test_game_mean():
    logits = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 5.0]])
    out = tc.game_mean(logits, torch.tensor([0, 0, 1]), 2)
    assert out.tolist() == [[2.0, 1.0], [5.0, 5.0]]


def test_spaced():
    assert tc.spaced([1, 3, 5], 8) == [1, 3, 5]
    assert tc.spaced(list(range(100)), 3) == [0, 50, 99]  # 等距含頭尾


def test_level_equal_nll():
    logp = np.log(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.25, 0.75]]))
    y = np.array([0, 0, 0, 1])
    # 0 級平均 ln2，1 級 -ln0.75，再取兩級平均
    assert tc.level_equal_nll(logp, y, n=2) == pytest.approx((np.log(2) - np.log(0.75)) / 2)


def test_collate_skips_none():
    a = (np.zeros((2, 22, 19, 19), np.uint8), np.zeros((2, 19), np.float32), 3, 0)
    b = (np.ones((1, 22, 19, 19), np.uint8), np.ones((1, 19), np.float32), 5, 2)
    out = tc.collate([a, None, b])
    assert out['game'].tolist() == [0, 0, 1]
    assert out['label'].tolist() == [3, 5]
    assert out['index'].tolist() == [0, 2]
    assert tc.collate([None]) is None
