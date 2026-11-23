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


def _tiny_model(**kw):
    return train_seq_emb.GoMoveSeqTransformer(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64, embed_dim=16, max_len=200,
        pool_mode='all', **kw,
    )


def _random_games(bsz, seq_len, n_valid, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randint(0, pack_move_seq.PASS_POS, (bsz, seq_len), generator=g)
    pos[:, 3] = pack_move_seq.PASS_POS
    pos[:, n_valid:] = pack_move_seq.PAD_POS
    is_target = (torch.arange(seq_len) % 2).expand(bsz, seq_len).clone()
    is_target[:, n_valid:] = 2
    return pos, is_target


def test_new_flags_off_keep_state_dict_keys():
    base = set(_tiny_model().state_dict())
    off = set(_tiny_model(spatial_bias=False, stage_pool=False).state_dict())
    assert base == off
    assert not any(k.startswith(('spatial_table', 'stage_pools')) for k in base)


def test_spatial_pair_buckets_values():
    p = pack_move_seq.PASS_POS
    pos = torch.tensor([[0, 1, 20, 4 * 19 + 4, 6 * 19, 10 * 19, 18 * 19 + 18, p, pack_move_seq.PAD_POS]])
    b = train_seq_emb.spatial_pair_buckets(pos)[0]
    assert b.shape == (9, 9)
    # 與第 0 手 (0,0) 的距離：0、1、2、8、6、10、36
    assert b[0, :7].tolist() == [0, 1, 2, 6, 5, 6, 7]
    assert b[3, 3] == 0 and torch.equal(b, b.T)
    last = train_seq_emb.SPATIAL_NUM - 1
    assert (b[7] == last).all() and (b[:, 8] == last).all()


def test_spatial_bias_zero_table_matches_plain_model():
    torch.manual_seed(0)
    plain = _tiny_model().eval()
    biased = _tiny_model(spatial_bias=True).eval()
    biased.load_state_dict(plain.state_dict(), strict=False)
    pos, is_target = _random_games(3, 40, 30)
    with torch.no_grad():
        torch.testing.assert_close(plain(pos, is_target), biased(pos, is_target), atol=1e-5, rtol=1e-5)
        biased.spatial_table.data[0] = 3.0
        assert not torch.allclose(plain(pos, is_target), biased(pos, is_target), atol=1e-4)


@pytest.mark.parametrize('kw', [{'spatial_bias': True}, {'stage_pool': True}])
def test_new_flags_ignore_pad_tail(kw):
    # 尾端補 pad 的結果須與只含有效手數的序列相同
    torch.manual_seed(0)
    model = _tiny_model(**kw).eval()
    if model.spatial_table is not None:
        model.spatial_table.data.uniform_(-2.0, 2.0)
    pos, is_target = _random_games(2, 180, 160)
    with torch.no_grad():
        torch.testing.assert_close(model(pos, is_target), model(pos[:, :160], is_target[:, :160]),
                                   atol=1e-5, rtol=1e-5)


def test_stage_pool_masks_split_at_bounds():
    model = _tiny_model(stage_pool=True).eval()
    pos, is_target = _random_games(1, 200, 170)
    masks = {}
    for j, sp in enumerate(model.stage_pools):
        sp.register_forward_hook(lambda m, inp, out, j=j: masks.__setitem__(j, inp[1][0].nonzero().flatten().tolist()))
    with torch.no_grad():
        model(pos, is_target)
    assert masks[0] == list(range(0, 50))
    assert masks[1] == list(range(50, 150))
    assert masks[2] == list(range(150, 170))


def test_stage_pool_short_game_zero_later_stages():
    torch.manual_seed(0)
    model = _tiny_model(stage_pool=True).eval()
    pos, is_target = _random_games(2, 180, 40)
    seen = {}
    model.stage_pools[1].register_forward_hook(lambda m, i, o: seen.__setitem__(1, o))
    model.stage_pools[2].register_forward_hook(lambda m, i, o: seen.__setitem__(2, o))
    with torch.no_grad():
        emb = model(pos, is_target)
    assert torch.isfinite(emb).all()
    assert torch.count_nonzero(seen[1]) == 0 and torch.count_nonzero(seen[2]) == 0
    assert model.bn_pool.num_features == 32 * 2 * 4


def test_new_flags_backward_reaches_table():
    torch.manual_seed(0)
    model = _tiny_model(spatial_bias=True, stage_pool=True).train()
    pos, is_target = _random_games(4, 60, 55)
    model(pos, is_target).sum().backward()
    assert model.spatial_table.grad is not None and model.spatial_table.grad.abs().sum() > 0


def test_spatial_bias_shared_contiguous_across_blocks():
    model = _tiny_model(spatial_bias=True).eval()
    seen = []
    for blk in model.blocks:
        blk.register_forward_pre_hook(lambda m, args: seen.append(args[2]))
    pos, is_target = _random_games(2, 30, 25)
    with torch.no_grad():
        model(pos, is_target)
    assert len(seen) == len(model.blocks)
    assert all(b is seen[0] for b in seen) and seen[0].is_contiguous()
