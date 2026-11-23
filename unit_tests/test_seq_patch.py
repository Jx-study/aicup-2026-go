import numpy as np
import pytest
import torch

from aicup import pack_move_seq, train_seq_emb


def _encode_grid(grid):
    """將 (P, P) 0..3 整數網格依每格 2 bit 打包成 uint8 陣列。"""
    flat = np.asarray(grid, dtype=np.uint8).ravel()
    nb = pack_move_seq.patch_nbytes(grid.shape[0])
    out = np.zeros(nb, dtype=np.uint8)
    for i, val in enumerate(flat.tolist()):
        out[i >> 2] |= (val & 3) << (2 * (3 - (i & 3)))
    return out


def test_patch_encoder_valid_mask_matches_unmasked_and_zeros_pad():
    torch.manual_seed(0)
    enc = train_seq_emb.PatchEncoder(patch_size=7, out_dim=64).eval()
    nb = pack_move_seq.patch_nbytes(7)
    patch = torch.randint(0, 256, (3, 10, nb), dtype=torch.uint8)
    valid_mask = torch.zeros((3, 10), dtype=torch.bool)
    valid_mask[0, :6] = True
    valid_mask[1, :3] = True
    valid_mask[2, :1] = True

    with torch.no_grad():
        out_unmasked = enc(patch)
        out_masked = enc(patch, valid_mask)
        # 竄改 PAD 位置的原始 bytes 不得影響任何有效位置輸出
        patch_mut = patch.clone()
        patch_mut[~valid_mask] = 255
        out_mut = enc(patch_mut, valid_mask)

    assert out_masked.shape == (3, 10, 64)
    torch.testing.assert_close(out_masked[valid_mask], out_unmasked[valid_mask], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_masked, out_mut, atol=1e-6, rtol=1e-6)
    assert float(out_masked[~valid_mask].abs().max().item()) == 0.0

    # 全部為 PAD 時輸出全 0 且形狀正確
    all_pad = torch.zeros((2, 8), dtype=torch.bool)
    with torch.no_grad():
        out_empty = enc(torch.randint(0, 256, (2, 8, nb), dtype=torch.uint8), all_pad)
    assert out_empty.shape == (2, 8, 64)
    assert float(out_empty.abs().max().item()) == 0.0


@pytest.mark.parametrize('patch_size', (3, 5, 7))
def test_patch_encoder_valid_conv_reduces_to_1x1(patch_size):
    enc = train_seq_emb.PatchEncoder(patch_size=patch_size, out_dim=64).eval()
    x = torch.zeros((4, 4, patch_size, patch_size), dtype=torch.float32)
    with torch.no_grad():
        feat = enc.conv(x)
    assert feat.shape == (4, 64, 1, 1)


@pytest.mark.parametrize('bad_size', (1, 2, 4, 6))
def test_patch_encoder_rejects_invalid_patch_size(bad_size):
    with pytest.raises(ValueError):
        train_seq_emb.PatchEncoder(patch_size=bad_size, out_dim=64)


def test_patch_encoder_distinguishes_stone_positions():
    torch.manual_seed(7)
    enc = train_seq_emb.PatchEncoder(patch_size=7, out_dim=64).eval()
    coords = [
        (3, 2),  # 落子點正左方緊鄰敵子
        (3, 4),  # 落子點正右方對稱位置
        (2, 3),  # 落子點正上方鄰接位置
        (0, 0),  # 局部邊角遠距敵子
    ]
    outs = []
    with torch.no_grad():
        for r, c in coords:
            grid = np.zeros((7, 7), dtype=np.uint8)
            grid[r, c] = 2
            p = torch.from_numpy(_encode_grid(grid)).view(1, 1, -1)
            outs.append(enc(p))
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            diff = float((outs[i] - outs[j]).abs().max().item())
            assert diff > 1e-3


def test_seq_transformer_patch_size_zero_state_dict_unchanged():
    m_plain = train_seq_emb.GoMoveSeqTransformer(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64, embed_dim=16, max_len=24,
    )
    m_state_conv = train_seq_emb.GoMoveSeqTransformer(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64, embed_dim=16, max_len=24,
        state_feats=True, conv_kernel=9, patch_size=0,
    )
    for model in (m_plain, m_state_conv):
        assert model.patch_enc is None
        assert not any(k.startswith('patch_enc.') for k in model.state_dict())
    assert m_plain.in_proj.in_features == 272
    assert m_state_conv.in_proj.in_features == 336
