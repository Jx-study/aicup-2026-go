from types import SimpleNamespace
import numpy as np
import torch

from aicup import colab_setup, packed


def test_feature_gb_parallel_and_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_setup, 'REPO', str(tmp_path))
    feat_dir = tmp_path / 'rank-features-test'
    (feat_dir / 'train').mkdir(parents=True)
    (feat_dir / 'val').mkdir(parents=True)

    assert colab_setup.feature_gb('rank-features-test') is None

    np.savez(feat_dir / 'train' / 'shard0.npz', rank_idx=np.zeros(10, dtype=np.int8))
    np.savez(feat_dir / 'val' / 'shard1.npz', rank_idx=np.zeros(5, dtype=np.int8))

    expected_gb = 15 * packed.PACKED_LEN / 1024 ** 3
    assert colab_setup.feature_gb('rank-features-test', workers=4) == expected_gb


def test_check_gpu_skip_size_check(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        lambda idx: SimpleNamespace(name='MockGPU', total_memory=24 * 1024 ** 3),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError('feature_gb should not be called when skip_size_check=True')

    monkeypatch.setattr(colab_setup, 'feature_gb', fail_if_called)
    colab_setup.check_gpu('rank-features-v180000', skip_size_check=True)
    out = capsys.readouterr().out
    assert 'MockGPU' in out
    assert '已跳過 rank-features-v180000 特徵大小檢查' in out
