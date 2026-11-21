import json

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import fuse_player, infer_seq_emb, pack_move_seq, packed, symmetry, train_seq_emb
from aicup.scoring import LEVELS


def _sgf_with_moves(n_moves, hd=0, offset=0, add_pass=False):
    coords = [(r, c) for r in range(19) for c in range(19)]
    letters = 'abcdefghijklmnopqrs'
    parts = [
        ';%s[%s%s]' % ('B' if i % 2 == 0 else 'W', letters[c], letters[r])
        for i, (r, c) in enumerate(coords[offset:offset + n_moves])
    ]
    if add_pass:
        parts.append(';B[]' if n_moves % 2 == 0 else ';W[]')
    return '(;GM[1]FF[4]SZ[19]HD[%d]%s)' % (hd, ''.join(parts))


def test_d4_pos_table_matches_symmetry_apply():
    table = pack_move_seq.D4_POS_TABLE
    assert table.shape == (8, pack_move_seq.NUM_POS)
    # 驗證全部 8 種對稱與全部 361 格棋盤座標皆與 symmetry.apply 完全一致
    for k in range(8):
        assert table[k, pack_move_seq.PASS_POS] == pack_move_seq.PASS_POS
        assert table[k, pack_move_seq.PAD_POS] == pack_move_seq.PAD_POS
        for r in range(19):
            for c in range(19):
                board = np.zeros((19, 19), dtype=np.uint8)
                board[r, c] = 1
                transformed = symmetry.apply(board, k)
                nr, nc = np.argwhere(transformed == 1)[0]
                assert int(table[k, r * 19 + c]) == int(nr * 19 + nc)


def test_parse_sgf_move_seq_hd_flip_pass_truncate_and_empty():
    sgf_hd0 = '(;GM[1]FF[4]SZ[19]HD[0];B[pd];W[dd];B[];W[pp])'
    pos, role, length, hd, ok = pack_move_seq.parse_sgf_move_seq(sgf_hd0, actual_color_id=0, max_moves=8)
    assert ok and length == 4 and hd == 0
    assert pos[2] == pack_move_seq.PASS_POS
    assert np.all(pos[4:] == pack_move_seq.PAD_POS)
    np.testing.assert_array_equal(role[:4], [1, 0, 1, 0])
    assert np.all(role[4:] == pack_move_seq.ROLE_PAD)

    # HD >= 2 時 SGF 標記黑白相反：實際執黑（0）應對應 SGF 的 W 手
    sgf_hd2 = '(;GM[1]FF[4]SZ[19]HD[2];B[pd];W[dd];B[pp];W[dp])'
    _, role_hd2_b, _, hd2, ok2 = pack_move_seq.parse_sgf_move_seq(sgf_hd2, actual_color_id=0, max_moves=8)
    _, role_hd2_w, _, _, _ = pack_move_seq.parse_sgf_move_seq(sgf_hd2, actual_color_id=1, max_moves=8)
    assert ok2 and hd2 == 2
    np.testing.assert_array_equal(role_hd2_b[:4], [0, 1, 0, 1])
    np.testing.assert_array_equal(role_hd2_w[:4], [1, 0, 1, 0])

    # HD == 1 維持原色
    sgf_hd1 = '(;GM[1]FF[4]SZ[19]HD[1];B[pd];W[dd])'
    _, role_hd1_b, _, hd1, _ = pack_move_seq.parse_sgf_move_seq(sgf_hd1, actual_color_id=0, max_moves=8)
    assert hd1 == 1
    np.testing.assert_array_equal(role_hd1_b[:2], [1, 0])

    # 截斷超過 max_moves 的長局
    long_sgf = _sgf_with_moves(20)
    pos_t, role_t, len_t, _, ok_t = pack_move_seq.parse_sgf_move_seq(long_sgf, actual_color_id=0, max_moves=10)
    assert ok_t and len_t == 10
    assert np.all(pos_t < pack_move_seq.PASS_POS)
    assert np.all(role_t != pack_move_seq.ROLE_PAD)

    # 空局與壞 SGF 防呆
    for bad in ('', '(;GM[1]FF[4]SZ[19]HD[0])', 'not_an_sgf'):
        p_b, r_b, l_b, _, ok_b = pack_move_seq.parse_sgf_move_seq(bad, actual_color_id=0, max_moves=8)
        assert not ok_b and l_b == 0
        assert np.all(p_b == pack_move_seq.PAD_POS)
        assert np.all(r_b == pack_move_seq.ROLE_PAD)


def _make_synthetic_dataset(tmp_path):
    feat_dir = tmp_path / 'rank-features-v180000'
    train_dir = tmp_path / 'training-v1.1' / 'training'
    train_dir.mkdir(parents=True)

    # 6 位訓練玩家（每人 6 局）、4 位 newdev 玩家（每人 16 局）、2 位 val648 玩家（每人 4 局）
    train_rows, val_rows, csv_by_rank = [], [], {lv: [] for lv in LEVELS[:3]}
    gid = 0
    for split, pid_prefix, n_p, n_g in (
        ('train', 'tr_', 6, 6),
        ('newdev', 'nd_', 4, 16),
        ('val648', 'v6_', 2, 4),
    ):
        for p in range(n_p):
            rk_idx = p % 3
            rk = LEVELS[rk_idx]
            pid = '%s%d' % (pid_prefix, p)
            for g in range(n_g):
                col = 'B' if (gid % 2 == 0) else 'W'
                hd = 2 if (gid % 5 == 0) else 0
                sgf_s = _sgf_with_moves(10 + (gid % 6), hd=hd, offset=(gid * 3) % 100)
                key = '%s_g%d' % (rk, gid)
                csv_by_rank[rk].append({
                    'rank': rk, 'game_id': 'g%d' % gid, 'player_id': pid,
                    'color': col, 'sgf_content': sgf_s,
                })
                row = {
                    'features': np.zeros(packed.PACKED_LEN, dtype=np.uint8),
                    'rank_idx': rk_idx,
                    'color': 0 if col == 'B' else 1,
                    'key': key,
                    'player_id': pid,
                }
                if split == 'train':
                    train_rows.append(row)
                else:
                    val_rows.append(row)
                gid += 1

    (feat_dir / 'train').mkdir(parents=True)
    (feat_dir / 'val').mkdir(parents=True)
    packed.save_shard(str(feat_dir / 'train' / 's0.npz'), train_rows)
    packed.save_shard(str(feat_dir / 'val' / 's0.npz'), val_rows)
    for rk, rows in csv_by_rank.items():
        pd.DataFrame(rows).to_csv(train_dir / ('train_%s.csv' % rk), index=False)

    newdev_csv = tmp_path / 'newdev.csv'
    val648_csv = tmp_path / 'val648.csv'
    pd.DataFrame({'player_id': ['nd_%d' % i for i in range(4)], 'rank': ['D', 'C', 'B', 'D']}).to_csv(newdev_csv, index=False)
    pd.DataFrame({'player_id': ['v6_%d' % i for i in range(2)], 'rank': ['D', 'C']}).to_csv(val648_csv, index=False)
    return train_dir, feat_dir, newdev_csv, val648_csv


def test_pack_splits_and_disjoint_check(tmp_path):
    train_dir, feat_dir, newdev_csv, val648_csv = _make_synthetic_dataset(tmp_path)
    out_dir = tmp_path / 'seq'
    summary = pack_move_seq.pack_splits(
        train_dir=str(train_dir),
        feat_dir=str(feat_dir),
        out_dir=str(out_dir),
        newdev_csv=str(newdev_csv),
        val648_csv=str(val648_csv),
        max_moves=24,
        nproc=1,
    )
    assert summary['train']['games'] == 36
    assert summary['newdev']['games'] == 64
    assert summary['val648']['games'] == 8

    with np.load(out_dir / 'train.npz', allow_pickle=False) as ztr, \
            np.load(out_dir / 'newdev.npz', allow_pickle=False) as znd, \
            np.load(out_dir / 'val648.npz', allow_pickle=False) as zv6:
        assert ztr['pos'].shape == (36, 24) and ztr['pos'].dtype == np.uint16
        assert ztr['is_target'].shape == (36, 24) and ztr['is_target'].dtype == np.uint8
        tr_p = set(ztr['player_id'].tolist())
        nd_p = set(znd['player_id'].tolist())
        v6_p = set(zv6['player_id'].tolist())
        assert tr_p.isdisjoint(nd_p) and tr_p.isdisjoint(v6_p) and nd_p.isdisjoint(v6_p)

    # 玩家重疊時必須拒絕執行
    bad_nd = tmp_path / 'bad_nd.csv'
    pd.DataFrame({'player_id': ['tr_0', 'nd_0'], 'rank': ['D', 'D']}).to_csv(bad_nd, index=False)
    with pytest.raises(ValueError, match='重疊'):
        pack_move_seq.pack_splits(
            train_dir=str(train_dir),
            feat_dir=str(feat_dir),
            out_dir=str(tmp_path / 'seq_bad'),
            newdev_csv=str(bad_nd),
            val648_csv=str(val648_csv),
            max_moves=24,
            nproc=1,
        )


def test_seq_transformer_forward_pool_modes_and_empty_game():
    torch.manual_seed(0)
    pos = torch.full((4, 16), pack_move_seq.PAD_POS, dtype=torch.long)
    role = torch.full((4, 16), pack_move_seq.ROLE_PAD, dtype=torch.long)
    # 前 3 局正常，第 4 局為全 pad 空局
    pos[:3, :8] = torch.randint(0, 361, (3, 8))
    role[:3, :8] = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0]).unsqueeze(0).expand(3, 8)
    pos[1, 3] = pack_move_seq.PASS_POS

    for pool_mode in ('target', 'all'):
        model = train_seq_emb.GoMoveSeqTransformer(
            d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
            embed_dim=16, max_len=16, pool_mode=pool_mode,
        )
        model.eval()
        emb = model(pos, role)
        assert emb.shape == (4, 16)
        assert torch.isfinite(emb).all()
        torch.testing.assert_close(emb[:3].norm(dim=1), torch.ones(3), atol=1e-5, rtol=1e-5)
        assert float(emb[3].abs().max().item()) == 0.0


def test_train_seq_emb_default_no_d4_smoke_and_resume(tmp_path):
    # 確認 CLI 預設不做 D4 增強
    parsed_default = train_seq_emb.parse_args(['--ckpt-dir', str(tmp_path / 'c')])
    assert parsed_default.augment_d4 is False
    parsed_d4 = train_seq_emb.parse_args(['--ckpt-dir', str(tmp_path / 'c'), '--augment-d4'])
    assert parsed_d4.augment_d4 is True

    train_dir, feat_dir, newdev_csv, val648_csv = _make_synthetic_dataset(tmp_path)
    seq_dir = tmp_path / 'seq'
    pack_move_seq.pack_splits(
        train_dir=str(train_dir), feat_dir=str(feat_dir), out_dir=str(seq_dir),
        newdev_csv=str(newdev_csv), val648_csv=str(val648_csv), max_moves=24, nproc=1,
    )

    ckpt_dir = tmp_path / 'ckpt'
    base_args = [
        '--seq-dir', str(seq_dir),
        '--ckpt-dir', str(ckpt_dir),
        '--d-model', '32', '--nhead', '4', '--num-layers', '2',
        '--dim-feedforward', '64', '--embed-dim', '16', '--max-len', '24',
        '--p-players', '3', '--k-games', '2', '--min-games', '5',
        '--epochs', '2', '--max-steps', '2', '--bn-batches', '2',
        '--warmup-steps', '1', '--eval-seeds', '42', '43',
        '--no-amp', '--log-every', '0',
    ]
    train_seq_emb.main(base_args)
    assert (ckpt_dir / 'best_model.pth').exists()
    assert (ckpt_dir / 'last.pt').exists()
    cfg = json.loads((ckpt_dir / 'train_config.json').read_text(encoding='utf-8'))
    assert cfg['augment_d4'] is False and cfg['pool_mode'] == 'target'

    hist = pd.read_csv(ckpt_dir / 'history.csv')
    assert len(hist) == 2
    assert 'seen_games' in hist.columns and int(hist['seen_games'].iloc[0]) == 12
    assert {'dev_top5', 'dev_top5_csls', 'dev_csls_q1'} <= set(hist.columns)

    # 改設定後拒絕從 last.pt 接續
    with pytest.raises(SystemExit):
        train_seq_emb.main(base_args + ['--augment-d4'])


def test_infer_seq_emb_packed_and_test_consistency_and_fuse_compat(tmp_path):
    cases = [
        ('D_g0', 'p0', _sgf_with_moves(12, hd=0, offset=0), 'b', 0),
        ('D_g1', 'p0', _sgf_with_moves(10, hd=0, offset=20), 'w', 0),
        ('D_g2', 'p1', _sgf_with_moves(14, hd=2, offset=40), 'b', 2),
        ('D_g3', 'p1', _sgf_with_moves(8, hd=3, offset=60, add_pass=True), 'w', 3),
    ]
    max_len = 24
    pos_np, role_np, len_np, hd_np, ok_np = pack_move_seq.parse_sgf_batch(
        [c[2] for c in cases],
        [0 if c[3] == 'b' else 1 for c in cases],
        max_moves=max_len,
        nproc=1,
    )
    assert bool(ok_np.all())

    seq_dir = tmp_path / 'seq'
    pack_move_seq.save_seq_npz(
        str(seq_dir / 'newdev.npz'),
        pos=pos_np, is_target=role_np, length=len_np,
        key=[c[0] for c in cases],
        player_id=[c[1] for c in cases],
        rank_idx=[0] * len(cases),
        col=[0 if c[3] == 'b' else 1 for c in cases],
        hd=hd_np,
    )

    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    torch.manual_seed(77)
    model = train_seq_emb.GoMoveSeqTransformer(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        embed_dim=16, max_len=max_len, pool_mode='target',
    )
    torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
    (ckpt_dir / 'train_config.json').write_text(json.dumps({
        'd_model': 32, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 64,
        'embed_dim': 16, 'max_len': max_len, 'dropout': 0.1, 'head_dropout': 0.2,
        'pool_mode': 'target', 'augment_d4': False, 'raw_color': False,
    }), encoding='utf-8')

    tests_dir = tmp_path / 'tests'
    tests_dir.mkdir()
    pd.DataFrame([
        {'game_id': k, 'player_id': pid, 'sgf_content': s, 'color': col}
        for k, pid, s, col, _ in cases
    ]).to_csv(tests_dir / 'player_identification_candidates2_public.csv', index=False)

    for tta in (1, 4):
        packed_out = tmp_path / ('packed_tta%d.npz' % tta)
        infer_seq_emb.main([
            '--mode', 'packed',
            '--ckpt', str(ckpt_dir / 'best_model.pth'),
            '--seq-dir', str(seq_dir), '--split', 'newdev',
            '--tta', str(tta), '--batch-size', '2', '--no-amp',
            '--out', str(packed_out),
        ])
        cache_dir = tmp_path / ('cache_tta%d' % tta)
        infer_seq_emb.main([
            '--mode', 'test',
            '--ckpt', str(ckpt_dir / 'best_model.pth'),
            '--tests-dir', str(tests_dir), '--groups', 'b',
            '--tta', str(tta), '--batch-size', '2', '--nproc', '1', '--no-amp',
            '--out-dir', str(cache_dir),
        ])

        with np.load(packed_out, allow_pickle=False) as zp, \
                np.load(cache_dir / 'pub2_cand_emb.npz', allow_pickle=False) as zt:
            assert {'emb', 'key', 'player_id', 'rank_idx', 'color', 'tta', 'ckpt_sha256'} <= set(zp.files)
            assert {'emb', 'entity_id', 'color', 'valid', 'tta', 'ckpt_sha256'} <= set(zt.files)
            assert int(zp['tta']) == tta and int(zt['tta']) == tta
            assert bool(zt['valid'].all())
            np.testing.assert_allclose(zp['emb'].astype(np.float32), zt['emb'].astype(np.float32), atol=1e-3)

        # 驗證 fuse_player 可直接載入推論快取
        loaded = fuse_player._load_emb_cache_npz(str(cache_dir / 'pub2_cand_emb.npz'))
        assert set(loaded.keys()) == {'p0', 'p1'}
        assert loaded['p0'].shape == (2, 16)
