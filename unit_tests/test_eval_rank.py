import sys

import numpy as np
import pytest
import torch

from aicup import color, eval_rank, packed
from aicup.scoring import LEVELS
from network import GoRankResNet
from test_train_rank import _load_probs, _main, _setup, _write_hd
from tta_helpers import tta_expect


def _eval(monkeypatch, feat, ckpt, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'eval_rank', '--feat-dir', str(feat), '--ckpt-dir', str(ckpt),
        '--batch-size', '4', '--channels', '8', '--blocks', '1', '--no-amp', *extra])
    eval_rank.main()


def _read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _trained(tmp_path, monkeypatch, *extra):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', *extra)
    return feat, ckpt


def test_without_tta_matches_train_val_probs(tmp_path, monkeypatch):
    feat, ckpt = _trained(tmp_path, monkeypatch)
    _eval(monkeypatch, feat, ckpt)
    got, ref = _read(ckpt / 'val_probs_eval.npz'), _load_probs(ckpt)
    np.testing.assert_allclose(got['probs'], ref['probs'], atol=1e-5)
    for k in ('rank_idx', 'key', 'player_id', 'epoch'):
        np.testing.assert_array_equal(got[k], ref[k])


def _val_input(feat, mode='none'):
    val = packed.load_split(str(feat / 'val'))
    x = packed.unpack_torch(torch.from_numpy(val['features']))
    if mode == 'none':
        return x
    hd = color.load_hd(str(feat / 'hd.npz'), val['key'])
    return color.apply_color(x, torch.from_numpy(color.feature_color(val['color'], hd)).long(), mode)


def _best(ckpt, mode='none'):
    model = GoRankResNet(in_channels=color.in_channels(mode), num_channels=8, num_blocks=1,
                         num_classes=len(LEVELS))
    model.load_state_dict(torch.load(ckpt / 'best_model.pth', weights_only=True))
    return model.eval()


def test_tta_averages_symmetries(tmp_path, monkeypatch):
    feat, ckpt = _trained(tmp_path, monkeypatch)
    _eval(monkeypatch, feat, ckpt, '--tta')
    got = _read(ckpt / 'val_probs_tta.npz')['probs']
    np.testing.assert_allclose(got, tta_expect(_best(ckpt), _val_input(feat)).numpy(), atol=1e-5)
    assert not np.allclose(got, _load_probs(ckpt)['probs'], atol=1e-6)


def test_tta_with_color(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'both')
    _eval(monkeypatch, feat, ckpt, '--tta')
    expect = tta_expect(_best(ckpt, 'both'), _val_input(feat, 'both')).numpy()
    np.testing.assert_allclose(_read(ckpt / 'val_probs_tta.npz')['probs'], expect, atol=1e-5)


def test_follows_train_config_color(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'both')
    _eval(monkeypatch, feat, ckpt)
    np.testing.assert_allclose(_read(ckpt / 'val_probs_eval.npz')['probs'], _load_probs(ckpt)['probs'], atol=1e-5)


def test_rejects_missing_checkpoint(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    ckpt.mkdir()
    with pytest.raises(SystemExit, match='best_model.pth'):
        _eval(monkeypatch, feat, ckpt)


def test_custom_out_path(tmp_path, monkeypatch):
    feat, ckpt = _trained(tmp_path, monkeypatch)
    out = tmp_path / 'x.npz'
    _eval(monkeypatch, feat, ckpt, '--out', str(out))
    assert out.exists() and not (ckpt / 'val_probs_eval.npz').exists()


def test_refuses_to_overwrite_val_probs(tmp_path, monkeypatch):
    feat, ckpt = _trained(tmp_path, monkeypatch)
    before = _load_probs(ckpt)['probs']
    with pytest.raises(SystemExit, match='val_probs.npz'):
        _eval(monkeypatch, feat, ckpt, '--tta', '--out', str(ckpt / 'val_probs.npz'))
    np.testing.assert_array_equal(_load_probs(ckpt)['probs'], before)
