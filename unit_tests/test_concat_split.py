import os

import numpy as np
import pytest

from aicup import concat_split
from aicup.concat_split import build_dir, concat_npz


def _write(path, keys, seq):
    np.savez(path, key=np.array(keys), seq=np.array(seq, dtype=np.int16), patch_size=np.int64(7))


def test_concat_npz_expected_arrays(tmp_path):
    a, b = tmp_path / 'train.npz', tmp_path / 'val648.npz'
    _write(a, ['k0', 'k1'], [[1, 2], [3, 4]])
    _write(b, ['v0'], [[5, 6]])
    got = concat_npz(str(a), str(b))
    assert sorted(got) == ['key', 'patch_size', 'seq']
    assert got['key'].tolist() == ['k0', 'k1', 'v0']
    assert got['seq'].dtype == np.int16 and got['seq'].tolist() == [[1, 2], [3, 4], [5, 6]]
    assert got['patch_size'].ndim == 0 and int(got['patch_size']) == 7


def test_concat_npz_rejects_different_fields(tmp_path):
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    _write(a, ['k0'], [[1, 2]])
    np.savez(b, key=np.array(['v0']))
    with pytest.raises(ValueError):
        concat_npz(str(a), str(b))


def test_build_dir_writes_train_and_links_others(tmp_path, monkeypatch):
    # 用假的 symlink 記錄呼叫，Windows 沒有建立 symlink 的權限也能測
    calls = []
    monkeypatch.setattr(concat_split.os, 'symlink', lambda src, dst: calls.append((src, dst)))
    src, out = tmp_path / 'src', tmp_path / 'out'
    src.mkdir()
    _write(src / 'train.npz', ['k0', 'k1'], [[1, 2], [3, 4]])
    _write(src / 'val648.npz', ['v0'], [[5, 6]])
    _write(src / 'newdev.npz', ['d0'], [[7, 8]])
    assert build_dir(str(src), str(out), 'train.npz', 'val648.npz') == 3
    assert [os.path.basename(d) for _, d in calls] == ['newdev.npz', 'val648.npz']
    assert all(s == os.path.abspath(str(src / os.path.basename(d))) for s, d in calls)
    assert not (out / 'train.npz.tmp').exists()
    with np.load(out / 'train.npz') as z:
        assert z['key'].tolist() == ['k0', 'k1', 'v0']
