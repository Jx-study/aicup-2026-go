import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import packed, train_player
from aicup.scoring import LEVELS
from network import GoRankResNet


def _tiny(in_channels=4):
    return GoRankResNet(in_channels=in_channels, num_channels=8, num_blocks=1, num_classes=len(LEVELS))


def test_question_logits_equal_mean_of_game_logits():
    torch.manual_seed(0)
    model = _tiny().eval()
    x = torch.randn(7, 4, 19, 19)
    qid = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    with torch.no_grad():
        got = train_player.question_logits(model, x, qid, 3)
        games = model(x)
    expect = torch.stack([games[qid == i].mean(0) for i in range(3)])
    torch.testing.assert_close(got, expect, atol=1e-5, rtol=1e-5)


def test_game_embed_matches_forward_before_fc_2():
    torch.manual_seed(0)
    model = _tiny().eval()
    x = torch.randn(3, 4, 19, 19)
    with torch.no_grad():
        torch.testing.assert_close(model.fc_2(train_player.game_embed(model, x)), model(x))


def _players():
    # a 23 局、b 4 局（不足 5）、c 5 局、d 40 局
    return np.array(['a'] * 23 + ['b'] * 4 + ['c'] * 5 + ['d'] * 40)


def test_make_epoch_questions_invariants():
    players = _players()
    questions = train_player.make_epoch_questions(players, np.random.default_rng(0))
    used = np.concatenate(questions)
    assert len(used) == len(set(used))  # 每局最多一次
    assert all(len(set(players[q])) == 1 for q in questions)  # 同一位玩家
    assert all(len(q) in train_player.SIZES for q in questions)
    assert 'b' not in {players[q][0] for q in questions}  # 少於 5 局沒有題目
    assert {players[q][0] for q in questions} == {'a', 'c', 'd'}
    # 剩下不到 5 局才丟：a 最多丟 4 局，c 全用，d 最多丟 4 局
    assert 23 - 4 <= sum(len(q) for q in questions if players[q[0]] == 'a') <= 23
    assert sum(len(q) for q in questions if players[q[0]] == 'c') == 5
    assert 40 - 4 <= sum(len(q) for q in questions if players[q[0]] == 'd') <= 40


def test_make_epoch_questions_deterministic():
    a = train_player.make_epoch_questions(_players(), np.random.default_rng(3))
    b = train_player.make_epoch_questions(_players(), np.random.default_rng(3))
    c = train_player.make_epoch_questions(_players(), np.random.default_rng(4))
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(a) != len(c) or not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_pack_batches_keeps_questions_whole():
    rng = np.random.default_rng(0)
    questions = train_player.make_epoch_questions(_players(), rng)
    batches = train_player.pack_batches(questions, max_games=30)
    assert all(sum(len(q) for q in b) <= 30 for b in batches)
    flat = [q for b in batches for q in b]
    assert len(flat) == len(questions)
    assert all(x is y for x, y in zip(flat, questions))  # 順序不變，每題一次且不拆開


def test_pack_batches_rejects_oversized_question():
    with pytest.raises(ValueError):
        train_player.pack_batches([np.arange(20)], max_games=10)


def test_set_train_mode_freezes_bn_statistics():
    torch.manual_seed(0)
    model = _tiny()
    train_player.set_train_mode(model)
    assert model.dropout.training and model.conv_in.training and model.layers.training
    before = {k: v.clone() for k, v in model.state_dict().items() if 'running' in k or 'num_batches' in k}
    model(torch.randn(4, 4, 19, 19))
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())


def _write_split(split_dir, n_players, games, seed):
    rng = np.random.default_rng(seed)
    split_dir.mkdir(parents=True)
    rows = [{'features': rng.integers(0, 256, packed.PACKED_LEN, dtype=np.uint8),
             'rank_idx': p % len(LEVELS), 'color': g % 2,
             'key': 'r%d_p%d_g%d' % (p % len(LEVELS), p, g), 'player_id': 'p%d' % p}
            for p in range(n_players) for g in range(games)]
    packed.save_shard(str(split_dir / 'a.npz'), rows)


def _group_ids(n_players, games_per_rank):
    """每級 n 位玩家、每位 games 局的 (rank_idx, group_id)，玩家 id 跨等級可重複。"""
    rank_idx, group_id = [], []
    for r in range(3):
        for p in range(n_players):
            for g in range(games_per_rank):
                rank_idx.append(r)
                group_id.append('%d|p%d' % (r, p))
    return np.array(rank_idx), np.array(group_id)


def test_select_monitor_players():
    rank_idx, group_id = _group_ids(6, 20)
    got = train_player.select_monitor_players(rank_idx, group_id, per_rank=2, min_games=20, seed=1)
    assert len(got) == 6 and all(sum(g.startswith('%d|' % r) for g in got) == 2 for r in range(3))
    assert np.array_equal(got, train_player.select_monitor_players(rank_idx, group_id, 2, 20, 1))
    assert not np.array_equal(got, train_player.select_monitor_players(rank_idx, group_id, 2, 20, 2))
    with pytest.raises(ValueError):
        train_player.select_monitor_players(rank_idx, group_id, per_rank=2, min_games=21)


def test_monitor_players_absent_from_training_and_questions_nested():
    rank_idx, group_id = _group_ids(6, 20)
    monitor = train_player.select_monitor_players(rank_idx, group_id, per_rank=2, min_games=20)
    games = np.flatnonzero(~np.isin(group_id, monitor))
    args = SimpleNamespace(epochs=3, seed=42, batch_size=30, accum=2, max_steps=0)
    plans = train_player.epoch_plans(group_id[games], args, games)
    used = {g for plan in plans for w in plan for b in w for q in b for g in q}
    assert not used & set(np.flatnonzero(np.isin(group_id, monitor)))
    assert used <= set(games)

    questions, sizes = train_player.monitor_questions(group_id, monitor)
    assert len(questions) == 4 * len(monitor) and list(sizes[:4]) == list(train_player.SIZES)
    assert np.array_equal(questions[3][:5], questions[0]) and np.array_equal(questions[3][:10], questions[1])
    assert all(len({group_id[g] for g in q}) == 1 for q in questions)


def test_total_updates_are_ceil_of_batches_over_accum():
    players = np.array(['a'] * 23 + ['c'] * 5 + ['d'] * 40)
    args = SimpleNamespace(epochs=3, seed=7, batch_size=30, accum=3, max_steps=0)
    plans = train_player.epoch_plans(players, args)
    for epoch, plan in enumerate(plans):
        rng = np.random.default_rng([args.seed, epoch])
        n_batches = len(train_player.pack_batches(train_player.make_epoch_questions(players, rng), 30))
        assert len(plan) == -(-n_batches // 3)
        assert all(len(w) <= 3 for w in plan)
    args.max_steps = 1
    assert all(len(p) == 1 for p in train_player.epoch_plans(players, args))


def test_accumulated_gradient_equals_single_batch_gradient():
    torch.manual_seed(0)
    model = GoRankResNet(in_channels=packed.FEATURE_SHAPE[0], num_channels=8, num_blocks=1, num_classes=len(LEVELS))
    train_player.set_train_mode(model)
    model.dropout.p = 0.0  # 關閉 dropout，兩邊的亂數才相同
    x = torch.from_numpy(np.random.default_rng(0).integers(0, 256, (12, packed.PACKED_LEN), dtype=np.uint8))
    y = torch.tensor([0, 0, 0, 3, 3, 3, 3, 7, 7, 7, 7, 7])
    q = [np.arange(0, 3), np.arange(3, 7), np.arange(7, 10), np.arange(10, 12)]
    device = torch.device('cpu')

    def grads(window):
        model.zero_grad()
        loss, _, n = train_player.window_backward(model, x, y, window, device, False)
        return [p.grad.clone() for p in model.parameters()], loss, n

    g_acc, loss_acc, n_acc = grads([[q[0], q[1]], [q[2], q[3]]])
    g_one, loss_one, n_one = grads([q])
    assert n_acc == n_one == 4
    assert loss_acc == pytest.approx(loss_one, rel=1e-5)
    for a, b in zip(g_acc, g_one):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)


def test_main_writes_loadable_best_model(tmp_path, monkeypatch):
    feat = tmp_path / 'feat'
    _write_split(feat / 'train', 16, 22, 0)  # 玩家 0～9 每級一位，當 monitor；10～15 訓練
    _write_split(feat / 'val', 2, 5, 1)
    init = tmp_path / 'init.pth'
    torch.manual_seed(0)
    start = GoRankResNet(in_channels=packed.FEATURE_SHAPE[0], num_channels=16, num_blocks=2,
                         num_classes=len(LEVELS))
    torch.save(start.state_dict(), init)
    ckpt = tmp_path / 'ckpt'
    monkeypatch.setattr(sys, 'argv', [
        'train_player', '--feat-dir', str(feat), '--init', str(init), '--ckpt-dir', str(ckpt),
        '--epochs', '1', '--batch-size', '40', '--channels', '16', '--blocks', '2', '--max-steps', '2',
        '--accum', '2', '--monitor-per-rank', '1', '--monitor-every', '1', '--log-every', '1',
        '--lr', '1e-2', '--warmup-steps', '1', '--augment', '--no-amp'])
    train_player.main()

    model = GoRankResNet(in_channels=packed.FEATURE_SHAPE[0], num_channels=16, num_blocks=2,
                         num_classes=len(LEVELS))
    model.load_state_dict(torch.load(ckpt / 'best_model.pth', weights_only=True))
    after = model.state_dict()
    before = start.state_dict()
    bn = [k for k in before if 'running' in k or 'num_batches' in k]
    assert bn and all(torch.equal(before[k], after[k]) for k in bn)  # BN 統計量不變
    assert not torch.equal(before['fc_2.weight'], after['fc_2.weight'])  # 其他參數有更新

    with np.load(ckpt / 'val_probs.npz', allow_pickle=False) as z:
        assert z['probs'].shape == (10, len(LEVELS)) and int(z['epoch']) == 1
    history = pd.read_csv(ckpt / 'history.csv')
    assert len(history) == 1 and history['steps'][0] == 2
    assert (ckpt / 'last.pt').exists() and (ckpt / 'epoch_1.pth').exists()
    mon = pd.read_csv(ckpt / 'monitor.csv')
    assert {'lr', 'train_loss', 'grad_norm', 'players', 'monitor_loss', 'monitor_loss_5', 'monitor_loss_20',
            'monitor_top_frac'} <= set(mon.columns)
    assert list(mon['step']) == [0, 1, 2]  # 訓練前一列，之後每個更新步一列
    assert mon['monitor_loss'].notna().all() and mon['train_loss'].iloc[1:].notna().all()
    assert (mon['grad_norm'].dropna() > 0).all() and mon['monitor_top_frac'].between(0.1, 1).all()


def test_monitor_rows_every_n_steps_and_epoch_end(tmp_path, monkeypatch):
    feat = tmp_path / 'feat'
    _write_split(feat / 'train', 16, 22, 0)
    _write_split(feat / 'val', 2, 5, 1)
    init = tmp_path / 'init.pth'
    torch.save(GoRankResNet(in_channels=packed.FEATURE_SHAPE[0], num_channels=8, num_blocks=1,
                            num_classes=len(LEVELS)).state_dict(), init)
    ckpt = tmp_path / 'ckpt'
    monkeypatch.setattr(sys, 'argv', [
        'train_player', '--feat-dir', str(feat), '--init', str(init), '--ckpt-dir', str(ckpt),
        '--epochs', '1', '--batch-size', '20', '--channels', '8', '--blocks', '1', '--max-steps', '3',
        '--accum', '1', '--monitor-per-rank', '1', '--monitor-every', '2', '--no-amp'])
    train_player.main()
    mon = pd.read_csv(ckpt / 'monitor.csv')
    assert list(mon['step']) == [0, 1, 2, 3]
    assert list(mon['monitor_loss'].notna()) == [True, False, True, True]  # 第 0、2 步與 epoch 結束
