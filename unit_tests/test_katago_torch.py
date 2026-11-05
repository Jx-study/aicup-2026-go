import os

import numpy as np
import pytest

from aicup import katago_torch as kt
from aicup import symmetry
from aicup.analyze_humansl import parse_game, policy_index, sgf_point, target_turns

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get('KATAGO_PYTHON') or kt.DEFAULT_KATAGO_PYTHON, 'features.py')),
                                reason='沒有 KataGo python/ 原始碼')


def pos(s):
    """SGF 座標在 KataGo 張量中的 index（y * 19 + x，aa 在左上角）。"""
    return (ord(s[1]) - ord('a')) * 19 + (ord(s[0]) - ord('a'))


def plane(enc, i, ch):
    return set(np.flatnonzero(enc['spatial'][i, ch].reshape(-1)))


def test_inverse_rank():
    assert kt.inverse_rank('rank_9d') == 1
    assert kt.inverse_rank('rank_1d') == 9
    assert kt.inverse_rank('rank_6d') == 4
    assert kt.inverse_rank('rank_1k') == 10
    assert kt.inverse_rank('rank_11k') == 20
    for bad in ('rank_0d', 'rank_10d', 'rank_21k', 'preaz_1d', 'rank_d'):
        with pytest.raises(ValueError):
            kt.inverse_rank(bad)


def test_profile_row():
    row = kt.profile_row('rank_1d')
    assert row.shape == (kt.N_META,)
    assert row[0] == 1 and row[1] == 1  # 雙方都是人類
    assert row[6:15].sum() == 9 and row[15:40].sum() == 0  # 自己的 inverse rank 9
    assert row[40:49].sum() == 9 and row[49:74].sum() == 0  # 對手相同
    assert row[74] == 0.5  # 是否計分未知
    assert row[75] == 0 and row[79] == 1  # 讀秒
    assert row[151 + 2] == 1  # 來源 KGS
    np.testing.assert_array_equal(row, kt.profile_row('rank_1d', 'W'))


def test_handicap_stones_and_colors():
    parsed = parse_game('(;SZ[19]HD[2];B[qc];W[cc];B[qq])')
    turns = target_turns(parsed['moves'], 'W')
    assert turns == [0, 2]
    enc = kt.encode_game(parsed, turns)
    assert plane(enc, 0, 1) == set()
    assert plane(enc, 0, 2) == {pos('pd'), pos('dp')}
    assert plane(enc, 1, 1) == {pos('qc')}
    assert plane(enc, 1, 2) == {pos('pd'), pos('dp'), pos('cc')}
    # 日本規則：白的 selfKomi = 貼目 0 + 黑子數 - 白子數
    assert enc['global'][0, 5] == pytest.approx(2 / 20)
    assert enc['global'][1, 5] == pytest.approx(2 / 20)
    assert list(enc['move']) == [pos('qc'), pos('qq')]


def test_komi_even_game():
    parsed = parse_game('(;SZ[19];B[pd];W[dp];B[pp];W[dd])')
    enc = kt.encode_game(parsed, [0, 1, 2])
    assert enc['global'][0, 5] == pytest.approx(-6.5 / 20)  # 黑方視角
    assert enc['global'][1, 5] == pytest.approx((6.5 + 1) / 20)
    assert enc['global'][2, 5] == pytest.approx(-(6.5 + 1 - 1) / 20)
    assert enc['global'][0, 9] == 1 and enc['global'][0, 10] == 1  # 數地、TAX_SEKI


def test_pass():
    parsed = parse_game('(;SZ[19];B[pd];W[];B[dp];W[dd])')
    assert parsed['moves'][1] == ('W', None)
    enc = kt.encode_game(parsed, [1, 2, 3])
    assert enc['move'][0] == policy_index(None) == 361
    assert enc['global'][1, 14] == 1  # 對手剛 pass
    assert enc['global'][0, 14] == 0 and enc['global'][2, 14] == 0
    assert enc['global'][1, 0] == 0  # 沒有 history 時不標前一手 pass
    with_hist = kt.encode_game(parsed, [1, 2, 3], history=True)
    assert with_hist['global'][1, 0] == 1
    assert with_hist['global'][2, 1] == 1  # 前二手是自己的 pass
    assert plane(with_hist, 2, 9) == {pos('dp')}


def test_alternation_checked_only_up_to_last_turn():
    parsed = parse_game('(;SZ[19];B[pd];W[dp];B[pp];B[dd])')  # 第 3 手連續同色
    assert len(kt.encode_game(parsed, [0, 2])['move']) == 2
    with pytest.raises(ValueError):
        kt.encode_game(parsed, [3])


def test_encore_rejected():
    parsed = parse_game('(;SZ[19];B[pd];W[];B[];W[dd])')
    kt.encode_game(parsed, [1])
    with pytest.raises(ValueError):
        kt.encode_game(parsed, [3])


def test_capture_and_legal():
    # 黑 ba、ab 提掉白 aa，白 aa 再下是自殺
    parsed = parse_game('(;SZ[19];B[ba];W[aa];B[ab];W[ss])')
    enc = kt.encode_game(parsed, [1, 3])
    assert plane(enc, 0, 2) == {pos('ba')}
    assert plane(enc, 1, 1) == set()
    assert plane(enc, 1, 2) == {pos('ba'), pos('ab')}
    assert not enc['legal'][1, pos('aa')]
    assert not enc['legal'][1, pos('ba')]  # 有子
    assert enc['legal'][1, pos('ss')] and enc['legal'][1, 361]
    assert plane(enc, 0, 5) == {pos('ba')}  # 白下之前黑 ba 有 aa、ca、bb 三口氣
    assert plane(enc, 1, 5) == {pos('ba'), pos('ab')}  # 提子後 aa 空出，各有三口氣


def test_ko_point_illegal():
    # 黑 bc、cb、cd 與白 db、ec、dd 圍出劫，黑 dc 提白 cc 後白不能馬上回提
    sgf = '(;SZ[19];B[bc];W[db];B[cb];W[ec];B[cd];W[dd];B[ss];W[cc];B[dc];W[rr])'
    parsed = parse_game(sgf)
    enc = kt.encode_game(parsed, [9])
    assert not enc['legal'][0, pos('cc')]
    assert plane(enc, 0, 6) == {pos('cc')}


def test_move_index_matches_katago():
    kg = kt.katago()
    board = kg['board'].Board(size=19)
    feats = kg['features'].Features({'version': kt.INPUT_VERSION}, 19)
    for s in ('aa', 'sa', 'as', 'ss', 'pd', 'dp', 'jj', 'qc'):
        p = sgf_point(s)
        assert feats.loc_to_tensor_pos(kt.to_loc(board, p), board) == policy_index(p) == pos(s)
    assert feats.loc_to_tensor_pos(kt.to_loc(board, None), board) == 361


@pytest.mark.parametrize('k', range(8))
def test_policy_symmetry(k):
    rng = np.random.default_rng(k)
    x = rng.normal(size=(3, 362)).astype(np.float32)
    y = kt.policy_symmetry(x, k)
    assert y[:, 361].tolist() == x[:, 361].tolist()
    np.testing.assert_array_equal(y[:, :361].reshape(3, 19, 19), symmetry.apply(x[:, :361].reshape(3, 19, 19), k))
    np.testing.assert_array_equal(kt.policy_symmetry(y, k, inverse=True), x)
    # 盤面與落點一起轉：輸入平面上的子和 policy 上的同一點對齊
    p = pos('qc')
    one_hot = np.zeros((1, 362), np.float32)
    one_hot[0, p] = 1
    grid = np.zeros((19, 19), np.float32)
    grid.reshape(-1)[p] = 1
    assert kt.policy_symmetry(one_hot, k)[0, :361].argmax() == symmetry.apply(grid, k).argmax()


@pytest.mark.skipif(not os.path.exists(kt.CHECKPOINT), reason='沒有 humanv0 checkpoint')
def test_move_log_probs_model():
    import torch
    model = kt.load_humanv0()
    parsed = parse_game('(;SZ[19];B[pd];W[dp];B[pp];W[dd];B[fc])')
    enc = kt.encode_game(parsed, [0, 1, 2, 3, 4])
    logp = kt.move_log_probs(model, enc, 'rank_1d')
    assert logp.shape == (5,) and np.all(logp < 0) and np.all(np.isfinite(logp))
    # 開局星位在 1d profile 下機率不低
    assert np.exp(logp[:4]).min() > 0.05
    with torch.no_grad():
        logits = kt.policy_logits(model, torch.from_numpy(enc['spatial']).float(),
                                  torch.from_numpy(enc['global']),
                                  torch.from_numpy(kt.profile_row('rank_1d')).expand(5, -1))
    assert logits.shape == (5, 362)
    # 對稱下結果接近原方向
    logp_sym = kt.move_log_probs(model, enc, 'rank_1d', sym=5)
    assert np.abs(logp_sym - logp).max() < 0.5
