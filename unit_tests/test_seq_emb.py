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
    pos, role, length, hd, ok = pack_move_seq.parse_sgf_move_seq(sgf_hd0, actual_color_id=0, max_moves=8)[:5]
    assert ok and length == 4 and hd == 0
    assert pos[2] == pack_move_seq.PASS_POS
    assert np.all(pos[4:] == pack_move_seq.PAD_POS)
    np.testing.assert_array_equal(role[:4], [1, 0, 1, 0])
    assert np.all(role[4:] == pack_move_seq.ROLE_PAD)

    # HD >= 2 時 SGF 標記黑白相反：實際執黑（0）應對應 SGF 的 W 手
    sgf_hd2 = '(;GM[1]FF[4]SZ[19]HD[2];B[pd];W[dd];B[pp];W[dp])'
    _, role_hd2_b, _, hd2, ok2 = pack_move_seq.parse_sgf_move_seq(sgf_hd2, actual_color_id=0, max_moves=8)[:5]
    _, role_hd2_w, _, _, _ = pack_move_seq.parse_sgf_move_seq(sgf_hd2, actual_color_id=1, max_moves=8)[:5]
    assert ok2 and hd2 == 2
    np.testing.assert_array_equal(role_hd2_b[:4], [0, 1, 0, 1])
    np.testing.assert_array_equal(role_hd2_w[:4], [1, 0, 1, 0])

    # HD == 1 維持原色
    sgf_hd1 = '(;GM[1]FF[4]SZ[19]HD[1];B[pd];W[dd])'
    _, role_hd1_b, _, hd1, _ = pack_move_seq.parse_sgf_move_seq(sgf_hd1, actual_color_id=0, max_moves=8)[:5]
    assert hd1 == 1
    np.testing.assert_array_equal(role_hd1_b[:2], [1, 0])

    # 截斷超過 max_moves 的長局
    long_sgf = _sgf_with_moves(20)
    pos_t, role_t, len_t, _, ok_t = pack_move_seq.parse_sgf_move_seq(long_sgf, actual_color_id=0, max_moves=10)[:5]
    assert ok_t and len_t == 10
    assert np.all(pos_t < pack_move_seq.PASS_POS)
    assert np.all(role_t != pack_move_seq.ROLE_PAD)

    # 空局與壞 SGF 防呆
    for bad in ('', '(;GM[1]FF[4]SZ[19]HD[0])', 'not_an_sgf'):
        p_b, r_b, l_b, _, ok_b = pack_move_seq.parse_sgf_move_seq(bad, actual_color_id=0, max_moves=8)[:5]
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
    )[:5]
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


def _rc(r, c):
    return r * 19 + c


def test_replay_move_states_capture_atari_ko_and_pass():
    B, W = 1, 2
    # 角落：黑 (0,2) 叫吃白 (0,1)，黑 (1,1) 提掉
    corner = [(B, _rc(0, 0)), (W, _rc(0, 1)), (B, _rc(0, 2)), (W, _rc(18, 18)), (B, _rc(1, 1)),
              (W, pack_move_seq.PASS_POS)]
    st = pack_move_seq.replay_move_states(corner, hd=0)
    assert st[2][2] == 1  # 叫吃
    assert st[4][1] == 1 and st[4][0] == 3 and st[4][3] == 0  # 提 1 子、提後 4 氣、非打劫
    assert st[5] == (pack_move_seq.LIBS_NONE, pack_move_seq.CAP_NONE, pack_move_seq.FLAG_NONE,
                     pack_move_seq.FLAG_NONE, pack_move_seq.PAT3_RAW_NONE)

    # 打劫：白 (4,4) 只剩 1 氣，黑 (4,5) 提 1 子後單子 1 氣
    ko = [(B, _rc(3, 4)), (W, _rc(3, 5)), (B, _rc(5, 4)), (W, _rc(5, 5)), (B, _rc(4, 3)),
          (W, _rc(4, 6)), (B, _rc(10, 10)), (W, _rc(4, 4)), (B, _rc(4, 5))]
    st = pack_move_seq.replay_move_states(ko, hd=0)
    assert st[7][0] == 0 and st[7][1] == 0  # 白入子 1 氣、未提子
    assert st[8][1] == 1 and st[8][3] == 1 and st[8][0] == 0


def test_replay_move_states_d4_invariant():
    B, W = 1, 2
    moves = [(B, _rc(3, 4)), (W, _rc(3, 5)), (B, _rc(5, 4)), (W, _rc(5, 5)), (B, _rc(4, 3)),
             (W, _rc(4, 6)), (B, _rc(0, 0)), (W, _rc(4, 4)), (B, _rc(4, 5)), (W, _rc(0, 1)),
             (B, _rc(0, 2)), (W, pack_move_seq.PASS_POS), (B, _rc(1, 1))]
    ref = pack_move_seq.replay_move_states(moves, hd=0)
    for k in range(8):
        tm = [(c, int(pack_move_seq.D4_POS_TABLE[k, p])) for c, p in moves]
        assert pack_move_seq.replay_move_states(tm, hd=0) == ref


def test_parse_state_places_handicap_stones():
    # pc 緊鄰讓子 pd；讓子屬於 SGF 標記為白的一方，pc 剩 3 氣
    sgf_hd2 = '(;GM[1]FF[4]SZ[19]HD[2];B[pc];W[aa])'
    sgf_hd0 = '(;GM[1]FF[4]SZ[19]HD[0];B[pc];W[aa])'
    r2 = pack_move_seq.parse_sgf_move_seq(sgf_hd2, 0, max_moves=8, state_feats=True)
    r0 = pack_move_seq.parse_sgf_move_seq(sgf_hd0, 0, max_moves=8, state_feats=True)
    assert r2.ok and r2.patch is None
    assert int(r2.st4[0, 0]) == 2 and int(r0.st4[0, 0]) == 3
    assert int(r2.st4[2, 0]) == pack_move_seq.LIBS_NONE and int(r2.pat3[2]) == pack_move_seq.PAT3_RAW_NONE
    # 不開 state_feats 時沒有狀態欄位
    plain = pack_move_seq.parse_sgf_move_seq(sgf_hd2, 0, max_moves=8)
    assert plain.st4 is None and plain.pat3 is None


def test_pack_splits_state_feats(tmp_path):
    train_dir, feat_dir, newdev_csv, val648_csv = _make_synthetic_dataset(tmp_path)
    kw = dict(train_dir=str(train_dir), feat_dir=str(feat_dir), newdev_csv=str(newdev_csv),
              val648_csv=str(val648_csv), max_moves=24, nproc=1)
    pack_move_seq.pack_splits(out_dir=str(tmp_path / 'plain'), **kw)
    pack_move_seq.pack_splits(out_dir=str(tmp_path / 'state'), state_feats=True, **kw)
    vocab = np.load(tmp_path / 'state' / pack_move_seq.PAT3_VOCAB_FILE)
    assert 0 < len(vocab) <= pack_move_seq.PAT3_TOP
    for split in ('train', 'newdev', 'val648'):
        with np.load(tmp_path / 'plain' / ('%s.npz' % split)) as zp, \
                np.load(tmp_path / 'state' / ('%s.npz' % split)) as zs:
            assert (zp['key'] == zs['key']).all() and (zp['pos'] == zs['pos']).all()
            assert set(pack_move_seq.STATE_FIELDS) <= set(zs.files)
            assert not set(pack_move_seq.STATE_FIELDS) & set(zp.files)
            pad = zs['pos'] == pack_move_seq.PAD_POS
            assert (zs['st_libs'][pad] == pack_move_seq.LIBS_NONE).all()
            assert (zs['pat3'][pad] == pack_move_seq.PAT3_PAD).all()
            assert int(zs['pat3'].max()) <= pack_move_seq.PAT3_PAD


def test_pack_splits_extra_split_reuses_vocab(tmp_path):
    train_dir, feat_dir, newdev_csv, val648_csv = _make_synthetic_dataset(tmp_path)
    kw = dict(train_dir=str(train_dir), feat_dir=str(feat_dir), newdev_csv=str(newdev_csv),
              max_moves=24, nproc=1, state_feats=True)
    full_dir = tmp_path / 'full'
    pack_move_seq.pack_splits(out_dir=str(full_dir), val648_csv=str(val648_csv), **kw)

    # v6_1 移到 hold，只寫出 hold 並沿用 full 的詞表
    v6_only0 = tmp_path / 'v6_0.csv'
    hold_csv = tmp_path / 'hold.csv'
    pd.DataFrame({'player_id': ['v6_0'], 'rank': ['D']}).to_csv(v6_only0, index=False)
    pd.DataFrame({'player_id': ['v6_1'], 'rank': ['C']}).to_csv(hold_csv, index=False)
    hold_dir = tmp_path / 'hold'
    summary = pack_move_seq.pack_splits(
        out_dir=str(hold_dir), val648_csv=str(v6_only0), extra_splits={'hold': str(hold_csv)},
        splits=['hold'], pat3_vocab_path=str(full_dir / pack_move_seq.PAT3_VOCAB_FILE), **kw)
    assert list(summary) == ['hold'] and summary['hold']['games'] == 4
    assert sorted(p.name for p in hold_dir.iterdir()) == ['hold.npz', pack_move_seq.PAT3_VOCAB_FILE]
    with np.load(full_dir / 'val648.npz') as zf, np.load(hold_dir / 'hold.npz') as zh:
        sel = zf['player_id'] == 'v6_1'
        assert set(zf.files) == set(zh.files)
        for f in zf.files:
            if zf[f].ndim:
                np.testing.assert_array_equal(zf[f][sel], zh[f])

    # hold 與 newdev 有交集、或不寫 train 又沒給詞表時拒絕執行
    bad_hold = tmp_path / 'bad_hold.csv'
    pd.DataFrame({'player_id': ['nd_0'], 'rank': ['D']}).to_csv(bad_hold, index=False)
    with pytest.raises(ValueError, match='交集'):
        pack_move_seq.pack_splits(out_dir=str(tmp_path / 'bad'), val648_csv=str(v6_only0),
                                  extra_splits={'hold': str(bad_hold)}, splits=['hold'],
                                  pat3_vocab_path=str(full_dir / pack_move_seq.PAT3_VOCAB_FILE), **kw)
    with pytest.raises(ValueError, match='詞表'):
        pack_move_seq.pack_splits(out_dir=str(tmp_path / 'bad2'), val648_csv=str(v6_only0),
                                  extra_splits={'hold': str(hold_csv)}, splits=['hold'], **kw)
    args = pack_move_seq.parse_args(['--extra-split', 'hold=a.csv', '--splits', 'hold'])
    assert args.extra_split == {'hold': 'a.csv'} and args.splits == ['hold']


def _state_inputs(n_moves, total_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.full((1, total_len), pack_move_seq.PAD_POS, dtype=torch.long)
    role = torch.full((1, total_len), pack_move_seq.ROLE_PAD, dtype=torch.long)
    st = torch.tensor([pack_move_seq.LIBS_NONE, pack_move_seq.CAP_NONE, pack_move_seq.FLAG_NONE,
                       pack_move_seq.FLAG_NONE, pack_move_seq.PAT3_PAD]).repeat(1, total_len, 1)
    pos[0, :n_moves] = torch.randint(0, 361, (n_moves,), generator=g)
    role[0, :n_moves] = torch.arange(n_moves) % 2
    st[0, :n_moves, 0] = torch.randint(0, 5, (n_moves,), generator=g)
    st[0, :n_moves, 4] = torch.randint(0, 50, (n_moves,), generator=g)
    return pos, role, st


def test_state_conv_model_pad_invariance_and_default_unchanged():
    default = train_seq_emb.GoMoveSeqTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                                                 embed_dim=16, max_len=24)
    assert not any(k.startswith(('libs_emb', 'cap_emb', 'atari_emb', 'ko_emb', 'pat3_emb', 'convs'))
                   for k in default.state_dict())
    assert default.in_proj.in_features == 272

    torch.manual_seed(1)
    model = train_seq_emb.GoMoveSeqTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                                               embed_dim=16, max_len=24, state_feats=True, conv_kernel=9)
    model.eval()
    assert model.in_proj.in_features == 336
    p16, r16, s16 = _state_inputs(8, 16)
    p24, r24, s24 = _state_inputs(8, 24)
    with torch.no_grad():
        e16 = model(p16, r16, s16)
        e24 = model(p24, r24, s24)
    torch.testing.assert_close(e16, e24, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        model(p16, r16)
    with pytest.raises(ValueError):
        train_seq_emb.GoMoveSeqTransformer(d_model=32, nhead=4, num_layers=1, conv_kernel=4)


def test_train_seq_emb_state_conv_smoke(tmp_path):
    train_dir, feat_dir, newdev_csv, val648_csv = _make_synthetic_dataset(tmp_path)
    seq_dir = tmp_path / 'seq_state'
    pack_move_seq.pack_splits(
        train_dir=str(train_dir), feat_dir=str(feat_dir), out_dir=str(seq_dir),
        newdev_csv=str(newdev_csv), val648_csv=str(val648_csv), max_moves=24, nproc=1, state_feats=True,
    )
    ckpt_dir = tmp_path / 'ckpt_state'
    base = [
        '--d-model', '32', '--nhead', '4', '--num-layers', '2',
        '--dim-feedforward', '64', '--embed-dim', '16', '--max-len', '24',
        '--p-players', '3', '--k-games', '2', '--min-games', '5',
        '--epochs', '1', '--max-steps', '2', '--bn-batches', '2',
        '--warmup-steps', '1', '--eval-seeds', '42',
        '--no-amp', '--log-every', '0', '--state-feats',
    ]
    train_seq_emb.main(base + ['--seq-dir', str(seq_dir), '--ckpt-dir', str(ckpt_dir), '--conv-kernel', '3'])
    cfg = json.loads((ckpt_dir / 'train_config.json').read_text(encoding='utf-8'))
    assert cfg['state_feats'] is True and cfg['conv_kernel'] == 3
    assert (ckpt_dir / pack_move_seq.PAT3_VOCAB_FILE).exists()
    model, _ = infer_seq_emb.load_seq_emb_model(str(ckpt_dir / 'best_model.pth'), torch.device('cpu'))
    assert model.state_feats and model.conv_kernel == 3

    # 沒有狀態欄位的序列不能訓練 state 模型
    plain_dir = tmp_path / 'seq_plain'
    pack_move_seq.pack_splits(
        train_dir=str(train_dir), feat_dir=str(feat_dir), out_dir=str(plain_dir),
        newdev_csv=str(newdev_csv), val648_csv=str(val648_csv), max_moves=24, nproc=1,
    )
    with pytest.raises(SystemExit):
        train_seq_emb.main(base + ['--seq-dir', str(plain_dir), '--ckpt-dir', str(tmp_path / 'ckpt_bad')])


def test_infer_seq_emb_state_packed_and_test_consistency(tmp_path):
    cases = [
        ('D_g0', 'p0', _sgf_with_moves(12, hd=0, offset=0), 'b'),
        ('D_g1', 'p0', _sgf_with_moves(10, hd=0, offset=20), 'w'),
        ('D_g2', 'p1', _sgf_with_moves(14, hd=2, offset=40), 'b'),
        ('D_g3', 'p1', _sgf_with_moves(8, hd=3, offset=60, add_pass=True), 'w'),
    ]
    max_len = 24
    res = pack_move_seq.parse_sgf_batch(
        [c[2] for c in cases], [0 if c[3] == 'b' else 1 for c in cases], max_moves=max_len, nproc=1,
        state_feats=True,
    )
    vocab = pack_move_seq.build_pat3_vocab(res.pat3)
    state = pack_move_seq.map_state_fields(res.st4, res.pat3, vocab)
    meta = dict(key=[c[0] for c in cases], player_id=[c[1] for c in cases], rank_idx=[0] * 4,
                col=[0 if c[3] == 'b' else 1 for c in cases], hd=res.hd)
    seq_dir = tmp_path / 'seq'
    pack_move_seq.save_seq_npz(str(seq_dir / 'newdev.npz'), pos=res.pos, is_target=res.is_target,
                               length=res.length, state=state, **meta)
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    np.save(ckpt_dir / pack_move_seq.PAT3_VOCAB_FILE, vocab)
    torch.manual_seed(5)
    model = train_seq_emb.GoMoveSeqTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                                               embed_dim=16, max_len=max_len, state_feats=True, conv_kernel=3)
    torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
    (ckpt_dir / 'train_config.json').write_text(json.dumps({
        'd_model': 32, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 64,
        'embed_dim': 16, 'max_len': max_len, 'dropout': 0.1, 'head_dropout': 0.2,
        'pool_mode': 'target', 'augment_d4': False, 'raw_color': False,
        'state_feats': True, 'conv_kernel': 3,
    }), encoding='utf-8')
    tests_dir = tmp_path / 'tests'
    tests_dir.mkdir()
    pd.DataFrame([
        {'game_id': k, 'player_id': pid, 'sgf_content': s, 'color': col} for k, pid, s, col in cases
    ]).to_csv(tests_dir / 'player_identification_candidates2_public.csv', index=False)

    ckpt = str(ckpt_dir / 'best_model.pth')
    packed_out = tmp_path / 'packed.npz'
    infer_seq_emb.main(['--mode', 'packed', '--ckpt', ckpt, '--seq-dir', str(seq_dir), '--split', 'newdev',
                        '--batch-size', '2', '--no-amp', '--out', str(packed_out)])
    infer_seq_emb.main(['--mode', 'test', '--ckpt', ckpt, '--tests-dir', str(tests_dir), '--groups', 'b',
                        '--batch-size', '2', '--nproc', '1', '--no-amp', '--out-dir', str(tmp_path / 'cache')])
    with np.load(packed_out) as zp, np.load(tmp_path / 'cache' / 'pub2_cand_emb.npz') as zt:
        assert bool(zt['valid'].all())
        np.testing.assert_allclose(zp['emb'].astype(np.float32), zt['emb'].astype(np.float32), atol=1e-3)

    # 序列沒有狀態欄位時，state 模型的 packed 推論必須拒絕
    pack_move_seq.save_seq_npz(str(tmp_path / 'plain' / 'newdev.npz'), pos=res.pos, is_target=res.is_target,
                               length=res.length, **meta)
    with pytest.raises(SystemExit):
        infer_seq_emb.main(['--mode', 'packed', '--ckpt', ckpt, '--seq-dir', str(tmp_path / 'plain'),
                            '--split', 'newdev', '--no-amp', '--out', str(tmp_path / 'bad.npz')])


def _decode_patch(b, size):
    arr = np.frombuffer(bytes(b), dtype=np.uint8)
    cells = np.stack([(arr >> (6 - 2 * k)) & 3 for k in range(4)], -1).ravel()[:size * size]
    return cells.reshape(size, size)


def test_replay_move_patches_relative_colors_and_edges():
    B, W = 1, 2
    moves = [(B, _rc(0, 0)), (W, _rc(1, 1)), (B, pack_move_seq.PASS_POS), (W, _rc(9, 9))]
    states, patches = pack_move_seq.replay_move_states(moves, hd=0, patch_size=7)
    assert states == pack_move_seq.replay_move_states(moves, hd=0)
    p1 = _decode_patch(patches[1], 7)  # 白 (1,1) 落子前，黑 (0,0) 在左上 (-1,-1)
    assert p1[3, 3] == 0 and p1[2, 2] == 2
    assert (p1[:2, :] == 3).all() and (p1[:, :2] == 3).all()
    assert patches[2] == bytes(pack_move_seq.patch_nbytes(7))
    p3 = _decode_patch(patches[3], 7)  # 白 (9,9) 周圍沒有子
    assert (p3 == 0).all()


def test_pack_patches_aligns_keys_and_train_infer_patch(tmp_path):
    from aicup import pack_move_patch
    train_dir, feat_dir, newdev_csv, val648_csv = _make_synthetic_dataset(tmp_path)
    seq_dir = tmp_path / 'seq_state'
    pack_move_seq.pack_splits(
        train_dir=str(train_dir), feat_dir=str(feat_dir), out_dir=str(seq_dir),
        newdev_csv=str(newdev_csv), val648_csv=str(val648_csv), max_moves=24, nproc=1, state_feats=True,
    )
    patch_dir = tmp_path / 'patch'
    summary = pack_move_patch.pack_patches(str(seq_dir), str(train_dir), str(patch_dir), patch_size=5, nproc=1)
    assert summary == {'train': 36, 'newdev': 64, 'val648': 8}
    with np.load(seq_dir / 'newdev.npz') as zs:
        patch, ps = pack_move_patch.load_patch_npz(str(patch_dir / 'newdev_patch.npz'), zs['key'])
        assert ps == 5 and patch.shape == (64, 24, pack_move_seq.patch_nbytes(5))
        with pytest.raises(SystemExit):
            pack_move_patch.load_patch_npz(str(patch_dir / 'newdev_patch.npz'), zs['key'][::-1])
    assert not list(patch_dir.glob('*.tmp*'))

    ckpt_dir = tmp_path / 'ckpt_patch'
    train_seq_emb.main([
        '--seq-dir', str(seq_dir), '--ckpt-dir', str(ckpt_dir), '--patch-dir', str(patch_dir),
        '--d-model', '32', '--nhead', '4', '--num-layers', '2',
        '--dim-feedforward', '64', '--embed-dim', '16', '--max-len', '24',
        '--p-players', '3', '--k-games', '2', '--min-games', '5',
        '--epochs', '1', '--max-steps', '2', '--bn-batches', '2',
        '--warmup-steps', '1', '--eval-seeds', '42',
        '--no-amp', '--log-every', '0', '--state-feats', '--fp16',
    ])
    cfg = json.loads((ckpt_dir / 'train_config.json').read_text(encoding='utf-8'))
    assert cfg['patch_size'] == 5 and cfg['state_feats'] is True
    with pytest.raises(SystemExit):
        train_seq_emb.parse_args(['--ckpt-dir', 'x', '--patch-dir', 'p', '--augment-d4'])

    # packed（局部盤面檔）與 test（從 SGF 重算）推論一致
    with np.load(seq_dir / 'newdev.npz') as zs:
        keys = zs['key'].tolist()
    sgf_by_key = {}
    for f in train_dir.glob('train_*.csv'):
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            sgf_by_key['%s_%s' % (r['rank'], r['game_id'])] = (r['player_id'], r['sgf_content'], r['color'].lower())
    tests_dir = tmp_path / 'tests'
    tests_dir.mkdir()
    pd.DataFrame([
        {'game_id': k, 'player_id': sgf_by_key[k][0], 'sgf_content': sgf_by_key[k][1], 'color': sgf_by_key[k][2]}
        for k in keys
    ]).to_csv(tests_dir / 'player_identification_candidates2_public.csv', index=False)
    ckpt = str(ckpt_dir / 'best_model.pth')
    packed_out = tmp_path / 'packed.npz'
    infer_seq_emb.main(['--mode', 'packed', '--ckpt', ckpt, '--seq-dir', str(seq_dir), '--split', 'newdev',
                        '--patch-dir', str(patch_dir), '--batch-size', '8', '--no-amp', '--out', str(packed_out)])
    infer_seq_emb.main(['--mode', 'test', '--ckpt', ckpt, '--tests-dir', str(tests_dir), '--groups', 'b',
                        '--batch-size', '8', '--nproc', '1', '--no-amp', '--out-dir', str(tmp_path / 'cache')])
    with np.load(packed_out) as zp, np.load(tmp_path / 'cache' / 'pub2_cand_emb.npz') as zt:
        assert bool(zt['valid'].all())
        np.testing.assert_allclose(zp['emb'].astype(np.float32), zt['emb'].astype(np.float32), atol=1e-3)
    with pytest.raises(SystemExit):
        infer_seq_emb.main(['--mode', 'packed', '--ckpt', ckpt, '--seq-dir', str(seq_dir), '--split', 'newdev',
                            '--no-amp', '--out', str(tmp_path / 'bad.npz')])
    model, _ = infer_seq_emb.load_seq_emb_model(ckpt, torch.device('cpu'))
    pos = torch.full((1, 24), pack_move_seq.PAD_POS, dtype=torch.long)
    with pytest.raises(ValueError):
        train_seq_emb.forward_seq_embeddings(model, pos, pos.clone(), False, tta=8,
                                             patch=torch.zeros((1, 24, 7), dtype=torch.uint8))


def test_replay_multi_capture_and_suicide_buckets():
    def at(r, c):
        return r * 19 + c

    # 黑 (0,2) 提掉角上白兩子，提子桶 2-3 對應 2
    moves = [(1, at(1, 0)), (2, at(0, 0)), (1, at(1, 1)), (2, at(0, 1)), (1, at(0, 2))]
    out = pack_move_seq.replay_move_states(moves, hd=0)
    assert out[4][1] == 2 and out[4][3] == 0
    # 白 (0,0) 落入黑子包圍且沒有提子，自殺手氣數 0 歸第 0 桶
    moves = [(1, at(0, 1)), (2, at(5, 5)), (1, at(1, 0)), (2, at(0, 0))]
    out = pack_move_seq.replay_move_states(moves, hd=0)
    assert out[3][0] == 0 and out[3][1] == 0
    # 自殺子留在盤上，黑 (0,2) 連 (0,1) 只剩 3 氣，桶 2
    out = pack_move_seq.replay_move_states(moves + [(1, at(0, 2))], hd=0)
    assert out[4][0] == 2


def test_parse_sgf_batch_multiproc_matches_single():
    sgfs = [_sgf_with_moves(10 + i, hd=i % 3, offset=7 * i, add_pass=bool(i % 2)) for i in range(6)]
    sgfs.append('not_an_sgf')
    cols = [i % 2 for i in range(len(sgfs))]
    one = pack_move_seq.parse_sgf_batch(sgfs, cols, max_moves=16, nproc=1, state_feats=True, patch_size=5)
    two = pack_move_seq.parse_sgf_batch(sgfs, cols, max_moves=16, nproc=2, state_feats=True, patch_size=5)
    for name in pack_move_seq.SeqParse._fields:
        np.testing.assert_array_equal(getattr(one, name), getattr(two, name))
    plain = pack_move_seq.parse_sgf_batch(sgfs, cols, max_moves=16, nproc=2)
    assert plain.st4 is None and plain.pat3 is None and plain.patch is None
    empty = pack_move_seq.parse_sgf_batch([], [], max_moves=16, state_feats=True)
    assert empty.pos.shape == (0, 16) and empty.st4.shape == (0, 16, 4) and empty.patch is None


def test_seq_files_bytes_uses_packed_shapes(tmp_path):
    seq_dir, patch_dir = tmp_path / 'seq', tmp_path / 'patch'
    seq_dir.mkdir()
    patch_dir.mkdir()
    np.savez_compressed(seq_dir / 'train.npz', pos=np.zeros((3, 24), dtype=np.uint16))
    np.savez_compressed(seq_dir / 'newdev.npz', pos=np.zeros((2, 24), dtype=np.uint16))
    np.savez_compressed(patch_dir / 'train_patch.npz', patch=np.zeros((3, 24, 7), dtype=np.uint8))
    n_moves = 5 * 24
    assert train_seq_emb._seq_files_bytes(str(seq_dir)) == n_moves * 3
    assert train_seq_emb._seq_files_bytes(str(seq_dir), state_feats=True) == n_moves * 13
    assert train_seq_emb._seq_files_bytes(str(seq_dir), patch_dir=str(patch_dir)) == n_moves * 3 + 3 * 24 * 7


def test_pack_patches_rejects_mismatched_splits(tmp_path):
    from aicup import pack_move_patch
    seq_dir = tmp_path / 'seq'
    seq_dir.mkdir()
    for split, length in (('train', 24), ('newdev', 20)):
        np.savez_compressed(seq_dir / ('%s.npz' % split), key=np.array(['D_%s' % split]),
                            pos=np.zeros((1, length), dtype=np.uint16), color=np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError):
        pack_move_patch.pack_patches(str(seq_dir), str(tmp_path), str(tmp_path / 'out'), splits=('train', 'newdev'))


def test_as_dir_list_accepts_str_sequence_and_none():
    assert fuse_player._as_dir_list(None) == []
    assert fuse_player._as_dir_list('a/b') == ['a/b']
    assert fuse_player._as_dir_list(('a', 'b')) == ['a', 'b']
