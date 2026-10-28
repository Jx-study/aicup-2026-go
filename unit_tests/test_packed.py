import numpy as np
import pytest

from aicup import packed


def _feat(seed):
    rng = np.random.default_rng(seed)
    return (rng.random(packed.FEATURE_SHAPE) > 0.9).astype(np.uint8)


def test_packed_len():
    assert packed.PACKED_LEN == 22563  # 180,500 bit 向上取整到 byte


def test_pack_unpack_numpy_roundtrip():
    feats = np.stack([_feat(i) for i in range(3)])
    rows = np.stack([packed.pack(f) for f in feats])
    assert rows.shape == (3, packed.PACKED_LEN) and rows.dtype == np.uint8
    assert np.array_equal(packed.unpack_numpy(rows), feats)


def test_pack_rejects_wrong_shape():
    with pytest.raises(ValueError):
        packed.pack(np.zeros((2, 19, 19), np.uint8))


def test_unpack_torch_matches_numpy():
    torch = pytest.importorskip('torch')
    feats = np.stack([_feat(i) for i in range(4)])
    rows = np.stack([packed.pack(f) for f in feats])
    got = packed.unpack_torch(torch.from_numpy(rows))
    assert got.dtype == torch.float32 and tuple(got.shape) == (4, *packed.FEATURE_SHAPE)
    assert np.array_equal(got.numpy().astype(np.uint8), feats)


def _row(i, rank_idx, color):
    return {'features': packed.pack(_feat(i)), 'rank_idx': rank_idx, 'color': color,
            'key': '1D_g_%07d' % i, 'player_id': 'p%d' % i}


def test_save_and_load_split(tmp_path):
    d = tmp_path / 'train'
    d.mkdir()
    packed.save_shard(str(d / 'train-w0-000000.npz'), [_row(0, 4, 0), _row(1, 9, 1)])
    packed.save_shard(str(d / 'train-w1-000000.npz'), [_row(2, 0, 1)])
    assert not list(d.glob('*.tmp'))
    out = packed.load_split(str(d))
    assert out['features'].shape == (3, packed.PACKED_LEN)
    assert out['rank_idx'].tolist() == [4, 9, 0]
    assert out['color'].tolist() == [0, 1, 1]
    assert out['key'].tolist() == ['1D_g_0000000', '1D_g_0000001', '1D_g_0000002']
    assert out['player_id'].tolist() == ['p0', 'p1', 'p2']
    assert np.array_equal(packed.unpack_numpy(out['features'][2:3])[0], _feat(2))


def test_save_empty_shard(tmp_path):
    # 整片都解析失敗時仍要寫出檔案，重跑才會跳過
    path = str(tmp_path / 'val-w0-000000.npz')
    packed.save_shard(path, [])
    out = packed.load_split(str(tmp_path))
    assert out['features'].shape == (0, packed.PACKED_LEN)


def test_load_split_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        packed.load_split(str(tmp_path))
