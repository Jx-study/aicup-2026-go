import numpy as np
import pandas as pd
import pytest
import torch

from aicup import color


def _feats():
    """第 t 個盤面的黑面全為 2t+1、白面全為 2t+2，3 局 × 2 個盤面。"""
    x = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1, 1)
    return x.repeat(3, 1, 19, 19)


def test_parse_hd():
    assert color.parse_hd('(;GM[1]SZ[19]HD[3];B[pd])') == 3
    assert color.parse_hd('(;GM[1]SZ[19];B[pd])') is None


def test_feature_color_flips_only_real_handicap():
    c = np.array([0, 1, 0, 1, 0, 1], np.uint8)
    hd = np.array([0, 0, 1, 1, 2, 9])
    assert color.feature_color(c, hd).tolist() == [0, 1, 0, 1, 1, 0]


def test_apply_none_keeps_features():
    x = _feats()
    assert torch.equal(color.apply_color(x.clone(), torch.tensor([0, 1, 1]), 'none'), x)


def test_apply_swap_exchanges_planes_of_white_targets():
    x = color.apply_color(_feats(), torch.tensor([0, 1, 0]), 'swap')
    assert x[:, :, 0, 0].tolist() == [[1, 2, 3, 4], [2, 1, 4, 3], [1, 2, 3, 4]]


def test_apply_plane_marks_black_target_as_one():
    x = color.apply_color(_feats(), torch.tensor([0, 1, 0]), 'plane')
    assert x.shape == (3, 5, 19, 19)
    assert x[:, 4].sum((1, 2)).tolist() == [361, 0, 361]
    assert x[1, 0, 0, 0] == 1  # plane 不對調


def test_apply_both_swaps_and_adds_plane():
    x = color.apply_color(_feats(), torch.tensor([0, 1, 0]), 'both')
    assert x[1, :, 0, 0].tolist() == [2, 1, 4, 3, 0]
    assert x[0, :, 0, 0].tolist() == [1, 2, 3, 4, 1]


def test_in_channels():
    assert [color.in_channels(m) for m in ('none', 'swap', 'plane', 'both')] == [500, 500, 501, 501]


def _write_csv(path, rank, rows):
    pd.DataFrame([{'player_id': 'p', 'game_id': g, 'rank': rank, 'color': 'B', 'sgf_content': s}
                  for g, s in rows]).to_csv(path / ('train_%s.csv' % rank), index=False)


def test_hd_map_round_trip(tmp_path):
    _write_csv(tmp_path, '6D', [(1, '(;SZ[19]HD[0];B[aa])'), (2, '(;SZ[19]HD[4];B[aa])')])
    _write_csv(tmp_path, 'D', [(1, '(;SZ[19]HD[1];B[aa])')])
    out = tmp_path / 'hd.npz'
    color.build_hd_map(str(tmp_path), str(out))
    hd = color.load_hd(str(out), np.array(['D_1', '6D_2', '6D_1']))
    assert hd.tolist() == [1, 4, 0]


def test_hd_map_rejects_csv_without_hd(tmp_path):
    _write_csv(tmp_path, 'D', [(1, '(;SZ[19];B[aa])')])
    with pytest.raises(SystemExit, match='v1.1'):
        color.build_hd_map(str(tmp_path), str(tmp_path / 'hd.npz'))


def test_load_hd_rejects_missing_key(tmp_path):
    np.savez(tmp_path / 'hd.npz', key=np.array(['D_1']), hd=np.array([0], np.int8))
    with pytest.raises(SystemExit, match='D_2'):
        color.load_hd(str(tmp_path / 'hd.npz'), np.array(['D_1', 'D_2']))
