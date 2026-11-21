import numpy as np
import pandas as pd
import pytest
import torch

from aicup import infer_player_emb, packed, predict_rank, train_player_emb
from aicup.scoring import LEVELS
from network import GoRankResNet


def _write_packed_split(split_dir, n_players, games_per_player, seed, pid_prefix='p', rank_offset=0):
    rng = np.random.default_rng(seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    gid = 0
    for p in range(n_players):
        rk = (p + rank_offset) % 3
        for _ in range(games_per_player):
            rows.append({
                'features': rng.integers(0, 256, packed.PACKED_LEN, dtype=np.uint8),
                'rank_idx': rk,
                'color': gid % 2,
                'key': '%s_g%d' % (LEVELS[rk], gid),
                'player_id': '%s%d' % (pid_prefix, p),
            })
            gid += 1
    mid = len(rows) // 2
    packed.save_shard(str(split_dir / 's0.npz'), rows[:mid])
    packed.save_shard(str(split_dir / 's1.npz'), rows[mid:])
    return rows


def _write_hd(feat_dir):
    keys = np.concatenate([packed.load_split(str(feat_dir / s))['key'] for s in ('train', 'val')])
    hd = np.array([3 if i % 4 == 0 else 0 for i in range(len(keys))], dtype=np.int8)
    np.savez(feat_dir / 'hd.npz', key=keys, hd=hd)


def _setup_feat(tmp_path):
    feat = tmp_path / 'feat'
    _write_packed_split(feat / 'train', n_players=6, games_per_player=6, seed=0, pid_prefix='tr_')
    _write_packed_split(feat / 'val', n_players=4, games_per_player=16, seed=1, pid_prefix='va_')
    _write_hd(feat)
    dev_csv = tmp_path / 'dev.csv'
    pd.DataFrame({'player_id': ['va_0', 'va_1', 'va_2'], 'rank': ['D', 'C', 'B']}).to_csv(dev_csv, index=False)
    return feat, dev_csv


def test_model_output_unit_norm_and_param_partition():
    model = train_player_emb.GoStyleEmbedNet(in_channels=500, num_channels=16, num_blocks=2, embed_dim=32)
    x = torch.rand(4, 500, 19, 19)
    emb = model(x)
    assert emb.shape == (4, 32)
    torch.testing.assert_close(emb.norm(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)

    bb_ids = {id(p) for p in model.backbone_parameters()}
    hd_ids = {id(p) for p in model.head_parameters()}
    all_ids = {id(p) for p in model.parameters()}
    assert bb_ids.isdisjoint(hd_ids)
    assert bb_ids | hd_ids == all_ids


def test_load_rank_backbone_exact_and_501_channels(tmp_path):
    rank_model = GoRankResNet(in_channels=500, num_channels=16, num_blocks=2, num_classes=len(LEVELS))
    ckpt = tmp_path / 'rank.pth'
    torch.save(rank_model.state_dict(), ckpt)

    emb_500 = train_player_emb.GoStyleEmbedNet(in_channels=500, num_channels=16, num_blocks=2, embed_dim=32)
    loaded = train_player_emb.load_rank_backbone(emb_500, str(ckpt))
    assert loaded > 0
    torch.testing.assert_close(emb_500.conv_in.weight, rank_model.conv_in.weight)
    torch.testing.assert_close(emb_500.layers[1].conv2.weight, rank_model.layers[1].conv2.weight)

    emb_501 = train_player_emb.GoStyleEmbedNet(in_channels=501, num_channels=16, num_blocks=2, embed_dim=32)
    train_player_emb.load_rank_backbone(emb_501, str(ckpt))
    torch.testing.assert_close(emb_501.conv_in.weight[:, :500], rank_model.conv_in.weight)
    assert float(emb_501.conv_in.weight[:, 500].detach().abs().max().item()) == 0.0


def test_subcenter_arcface_and_supcon_loss():
    torch.manual_seed(0)
    arc = train_player_emb.SubCenterArcFace(embed_dim=16, num_classes=5, num_sub_centers=2, scale=20.0, margin=0.25)
    emb = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss_arc, corr = arc(emb, labels)
    assert torch.isfinite(loss_arc) and 0 <= corr <= 8
    loss_sup = train_player_emb.supcon_loss(emb, labels, temperature=0.1)
    assert torch.isfinite(loss_sup) and float(loss_sup.detach().item()) > 0
    (loss_arc + 0.3 * loss_sup).backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()

    # 同一玩家向量越接近時 SupCon 損失應顯著較低
    clustered = torch.zeros(4, 8)
    clustered[0, 0] = clustered[1, 0] = 1.0
    clustered[2, 1] = clustered[3, 1] = 1.0
    y_pair = torch.tensor([0, 0, 1, 1])
    l_good = train_player_emb.supcon_loss(clustered, y_pair, temperature=0.1)
    l_bad = train_player_emb.supcon_loss(clustered, torch.tensor([0, 1, 0, 1]), temperature=0.1)
    assert float(l_good) < float(l_bad)


def test_tail_crop_shifts_and_zero_pads():
    x = torch.arange(1, 501, dtype=torch.float32).view(1, 500, 1, 1).expand(2, 500, 19, 19).clone()
    gen = torch.Generator().manual_seed(7)
    out = train_player_emb.apply_tail_crop(x, max_crop=10, gen=gen)
    # 找第一個非零通道，確認平移量為偶數且內容與原始前端一致
    nz = (out[0, :, 0, 0] != 0).nonzero(as_tuple=False)
    shift = int(nz[0].item())
    assert shift % 2 == 0 and 0 <= shift <= 20
    if shift > 0:
        assert float(out[:, :shift].abs().max()) == 0.0
        torch.testing.assert_close(out[:, shift:], x[:, :500 - shift])


def test_build_player_index_and_within_rank_pk_sampler():
    # 建立 4 個段位、每段位 3 位玩家以驗證每個 batch 段位數不超過 2
    ranks, pids = [], []
    for r in range(4):
        for p in range(3):
            ranks.extend([r] * 6)
            pids.extend(['p%d_%d' % (r, p)] * 6)
    # 另加同 player_id 跨段位、局數不足與被排除者
    ranks.extend([0] * 6 + [1] * 6 + [1] * 3 + [0] * 6)
    pids.extend(['dup'] * 6 + ['dup'] * 6 + ['short'] * 3 + ['excl'] * 6)
    meta = {'rank_idx': np.array(ranks, dtype=np.int8), 'player_id': np.array(pids, dtype=str)}

    pg, rp, keys = train_player_emb.build_player_index(meta, min_games=5, exclude_pids={'excl'})
    # (0, 'dup') 與 (1, 'dup') 視為兩位獨立玩家，'short' 局數不足被排除，'excl' 被排除
    assert (0, 'dup') in keys and (1, 'dup') in keys
    assert all(k[1] not in ('short', 'excl') for k in keys)

    cid_to_rank = {cid: key[0] for cid, key in enumerate(keys)}
    rng = np.random.default_rng(42)
    batches = train_player_emb.sample_within_rank_pk_batches(pg, rp, n_batches=10, p_players=3, k_games=2, rng=rng)
    assert len(batches) == 10
    for g_idx, labels in batches:
        assert len(g_idx) == 6 and len(labels) == 6
        uniq, cnts = np.unique(labels, return_counts=True)
        assert len(uniq) == 3 and np.all(cnts == 2)
        batch_ranks = {cid_to_rank[int(c)] for c in uniq}
        assert len(batch_ranks) <= 2


def test_recompute_bn_stats_and_extract_preserves_mode():
    torch.manual_seed(0)
    model = train_player_emb.GoStyleEmbedNet(in_channels=500, num_channels=8, num_blocks=1, embed_dim=16)
    x = torch.from_numpy(np.random.default_rng(0).integers(0, 256, (8, packed.PACKED_LEN), dtype=np.uint8))
    bn = train_player_emb.recompute_bn_stats(model, x, batch_size=4, device=torch.device('cpu'), amp=False, n_batches=2)
    assert not torch.allclose(bn.bn_in.running_mean, model.bn_in.running_mean)
    assert not torch.allclose(bn.bn_emb.running_mean, model.bn_emb.running_mean)
    assert float(model.bn_in.running_mean.abs().max()) == 0.0

    # extract_split_embeddings 須保留呼叫前的 train / eval 模式
    model.eval()
    train_player_emb.extract_split_embeddings(model, x, np.arange(4), 4, torch.device('cpu'), False)
    assert not model.training
    model.train()
    train_player_emb.extract_split_embeddings(model, x, np.arange(4), 4, torch.device('cpu'), False)
    assert model.training


def test_evaluate_b_group_embeddings_perfect_separation():
    rng = np.random.default_rng(0)
    # 4 位玩家各自正交方向，每人 16 局
    embs = []
    pids = []
    for p in range(4):
        v = np.zeros(8, dtype=np.float32)
        v[p] = 1.0
        for _ in range(16):
            embs.append(v + rng.normal(0, 0.01, size=8).astype(np.float32))
            pids.append('p%d' % p)
    embs = np.stack(embs)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    res = train_player_emb.evaluate_b_group_embeddings(embs, pids, seeds=(42, 43), n_cands=4, n_queries=4, csls_k=2)
    assert res['dev_top5'] == pytest.approx(1.0)
    assert res['dev_top5_csls'] == pytest.approx(1.0)
    assert 'dev_csls_q1' in res


def test_train_player_emb_rejects_missing_dev_players(tmp_path):
    feat, _ = _setup_feat(tmp_path)
    missing_csv = tmp_path / 'not_found.csv'
    with pytest.raises(SystemExit, match='dev-players'):
        train_player_emb.main([
            '--feat-dir', str(feat),
            '--ckpt-dir', str(tmp_path / 'ckpt'),
            '--dev-players', str(missing_csv),
            '--epochs', '1', '--no-amp',
        ])


def test_train_player_emb_smoke_and_resume(tmp_path):
    feat, dev_csv = _setup_feat(tmp_path)
    rank_model = GoRankResNet(in_channels=500, num_channels=8, num_blocks=1, num_classes=len(LEVELS))
    init_ckpt = tmp_path / 'rank_init.pth'
    torch.save(rank_model.state_dict(), init_ckpt)

    ckpt_dir = tmp_path / 'ckpt'
    base_args = [
        '--feat-dir', str(feat),
        '--ckpt-dir', str(ckpt_dir),
        '--init', str(init_ckpt),
        '--dev-players', str(dev_csv),
        '--channels', '8', '--blocks', '1', '--embed-dim', '16',
        '--p-players', '3', '--k-games', '2', '--min-games', '5',
        '--steps-per-epoch', '2', '--epochs', '2', '--bn-batches', '2',
        '--warmup-steps', '1', '--tail-crop', '4',
        '--eval-seeds', '42', '43', '--no-amp', '--log-every', '0',
    ]
    train_player_emb.main(base_args)
    assert (ckpt_dir / 'best_model.pth').exists()
    assert (ckpt_dir / 'last.pt').exists()
    assert (ckpt_dir / 'train_config.json').exists()
    hist = pd.read_csv(ckpt_dir / 'history.csv')
    assert len(hist) == 2
    assert hist['dev_players'].iloc[0] == 3
    assert {'raw_dev_top5_csls', 'dev_top5_csls', 'dev_csls_q1'} <= set(hist.columns)

    # 測試設定改變時拒絕接續
    with pytest.raises(SystemExit):
        train_player_emb.main(base_args[:-4] + ['--epochs', '3', '--no-amp', '--log-every', '0'])


def test_compute_sgf_feature_colors_handicap_flip():
    sgfs = [
        '(;GM[1]FF[4]SZ[19]HD[0];B[pd];W[dd])',
        '(;GM[1]FF[4]SZ[19]HD[1];B[pd];W[dd])',
        '(;GM[1]FF[4]SZ[19]HD[2];B[pd];W[dd])',
        '(;GM[1]FF[4]SZ[19]HD[3];B[pd];W[dd])',
    ]
    # HD 0 與 1 維持原色（0 黑 1 白），HD >= 2 黑白反轉
    got_b = infer_player_emb.compute_sgf_feature_colors(sgfs, ['b', 'b', 'b', 'b'])
    got_w = infer_player_emb.compute_sgf_feature_colors(sgfs, ['w', 'w', 'w', 'w'])
    np.testing.assert_array_equal(got_b, [0, 0, 1, 1])
    np.testing.assert_array_equal(got_w, [1, 1, 0, 0])


def test_csv_readers_reject_invalid_or_missing_color(tmp_path):
    sgf_ok = '(;GM[1]FF[4]SZ[19]HD[0];B[pd];W[dd])'
    bad_cand = tmp_path / 'bad_cand.csv'
    pd.DataFrame([{'game_id': 'g1', 'player_id': 'p1', 'sgf_content': sgf_ok, 'color': np.nan}]).to_csv(bad_cand, index=False)
    with pytest.raises(ValueError, match='顏色'):
        infer_player_emb._read_cand_csv(str(bad_cand))

    bad_test = tmp_path / 'bad_test.csv'
    pd.DataFrame([{'question_id': 'q1', 'num_games': 1, 'sgf_1': sgf_ok, 'color_1': 'x'}]).to_csv(bad_test, index=False)
    with pytest.raises(ValueError, match='顏色'):
        infer_player_emb._read_test_csv(str(bad_test))


def test_packed_and_sgf_inference_consistency(tmp_path):
    # 建立同一組含讓子與非讓子、黑白兩色的真實 SGF，驗證 packed 與 test 兩條路徑輸出一致
    cases = [
        ('D_g0', 'p0', '(;GM[1]FF[4]SZ[19]HD[0];B[pd];W[dd];B[pp];W[dp])', 'b', 0),
        ('D_g1', 'p0', '(;GM[1]FF[4]SZ[19]HD[0];B[pd];W[dd];B[pp];W[dp])', 'w', 0),
        ('D_g2', 'p1', '(;GM[1]FF[4]SZ[19]HD[2];B[pd];W[dd];B[pp];W[dp])', 'b', 2),
        ('D_g3', 'p1', '(;GM[1]FF[4]SZ[19]HD[3];B[pd];W[dd];B[pp];W[dp])', 'w', 3),
    ]
    feat_dir = tmp_path / 'feat'
    val_dir = feat_dir / 'val'
    val_dir.mkdir(parents=True)
    rows = []
    for key, pid, sgf_str, col_str, _ in cases:
        f_arr = predict_rank.sgf_to_features(sgf_str)
        rows.append({
            'features': packed.pack(f_arr),
            'rank_idx': 0,
            'color': 0 if col_str == 'b' else 1,
            'key': key,
            'player_id': pid,
        })
    packed.save_shard(str(val_dir / 's0.npz'), rows)
    np.savez(feat_dir / 'hd.npz', key=np.array([c[0] for c in cases]), hd=np.array([c[4] for c in cases], dtype=np.int8))

    # 儲存固定權重模型
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    torch.manual_seed(123)
    model = train_player_emb.GoStyleEmbedNet(in_channels=500, num_channels=8, num_blocks=1, embed_dim=16)
    torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
    (ckpt_dir / 'train_config.json').write_text(
        '{"channels": 8, "blocks": 1, "embed_dim": 16, "dropout": 0.2, "color": "swap", "raw_color": false}',
        encoding='utf-8',
    )

    packed_out = tmp_path / 'packed.npz'
    infer_player_emb.main([
        '--mode', 'packed',
        '--ckpt', str(ckpt_dir / 'best_model.pth'),
        '--feat-dir', str(feat_dir),
        '--tta', '4', '--batch-size', '4', '--no-amp',
        '--out', str(packed_out),
    ])

    tests_dir = tmp_path / 'tests'
    tests_dir.mkdir()
    pd.DataFrame([
        {'game_id': key, 'player_id': pid, 'sgf_content': sgf_str, 'color': col_str}
        for key, pid, sgf_str, col_str, _ in cases
    ]).to_csv(tests_dir / 'player_identification_candidates2_public.csv', index=False)

    cache_dir = tmp_path / 'cache'
    infer_player_emb.main([
        '--mode', 'test',
        '--ckpt', str(ckpt_dir / 'best_model.pth'),
        '--tests-dir', str(tests_dir),
        '--groups', 'b', '--tta', '4', '--batch-size', '4', '--nproc', '1', '--no-amp',
        '--out-dir', str(cache_dir),
    ])

    with np.load(packed_out, allow_pickle=False) as zp, np.load(cache_dir / 'pub2_cand_emb.npz', allow_pickle=False) as zt:
        np.testing.assert_allclose(zp['emb'].astype(np.float32), zt['emb'].astype(np.float32), atol=1e-3)


def test_aggregate_and_center_similarity_edge_cases():
    # 空字典不應噴 StopIteration
    empty = infer_player_emb.aggregate_and_center_similarity({}, {}, ['c1'], ['q1'])
    assert empty.shape == (1, 1) and np.isneginf(empty[0, 0])

    # 無效候選欄位應為 -inf，不排在負相似度候選之前
    cands = {
        'c1': np.array([[1.0, 0.0, 0.0]]),
        'c2': np.array([[0.0, 1.0, 0.0]]),
        'c_empty': np.zeros((0, 3)),
    }
    queries = {
        'q1': np.array([[1.0, 0.0, 0.0]]),
        'q2': np.array([[0.0, 1.0, 0.0]]),
    }
    sim = infer_player_emb.aggregate_and_center_similarity(cands, queries, ['c1', 'c2', 'c_empty'], ['q1', 'q2'])
    assert sim.shape == (2, 3)
    assert sim[0, 0] > sim[0, 1]
    assert sim[1, 1] > sim[1, 0]
    assert np.all(np.isneginf(sim[:, 2]))
