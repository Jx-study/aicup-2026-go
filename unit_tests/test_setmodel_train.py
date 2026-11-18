import argparse

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from aicup import setmodel_train as st  # noqa: E402


def toy_games(n_players=40, games_per_player=8, signal=1.0, seed=0):
    """等級 r 的玩家，向量第 40r～40r+39 維加 signal。"""
    rng = np.random.default_rng(seed)
    n = n_players * games_per_player
    rank = np.repeat(np.arange(n_players) % st.N_LEVELS, games_per_player)
    vec = rng.normal(size=(n, st.PER_GAME, st.DIM)).astype(np.float16)
    for r in range(st.N_LEVELS):
        vec[rank == r, :, 40 * r:40 * r + 40] += signal
    return {'key': np.array(['g%d' % i for i in range(n)]),
            'player_id': np.repeat(['p%d' % i for i in range(n_players)], games_per_player),
            'rank_idx': rank, 'vec': vec, 'mask': np.ones((n, st.PER_GAME), bool)}


def toy_model(games):
    mean, std = st.fit_standardizer(games['vec'], games['mask'], np.random.default_rng(0))
    torch.manual_seed(0)
    return st.build_model(mean, std, d=16, dropout=0.0).eval()


def test_logits_ignore_game_order_and_other_questions():
    games = toy_games()
    model = toy_model(games)
    q = [np.array([0, 1, 2]), np.array([8, 9])]
    base = st.question_logits(model, games['vec'], games['mask'], q).detach()
    swapped = st.question_logits(model, games['vec'], games['mask'], [np.array([2, 0, 1]), np.array([9, 8])]).detach()
    alone = st.question_logits(model, games['vec'], games['mask'], q[1:]).detach()
    assert torch.allclose(base, swapped, atol=1e-5)
    assert torch.allclose(base[1], alone[0], atol=1e-5)


def test_masked_moves_do_not_contribute():
    games = toy_games()
    model = toy_model(games)
    mask = games['mask'].copy()
    mask[0, 10:] = False
    vec = games['vec'].copy()
    a = st.embed_games(model, vec[:1], mask[:1])
    vec[0, 10:] += 100
    b = st.embed_games(model, vec[:1], mask[:1])
    assert np.allclose(a, b)


def test_q_fn_matches_logits_and_falls_back_to_prior():
    games = toy_games()
    model = toy_model(games)
    has = np.ones(len(games['key']), bool)
    has[[0, 1]] = False
    gv = st.embed_games(model, games['vec'], games['mask'])
    prior = np.log(np.full(st.N_LEVELS, 0.1))
    fn = st.make_q_fn(model, gv, has, prior)
    out = fn([np.array([0, 1]), np.array([0, 2, 3]), np.array([2, 3])])
    assert np.allclose(out[0], prior)
    assert np.allclose(out[1], out[2], atol=1e-5)  # 沒有向量的局不影響
    want = torch.log_softmax(st.question_logits(model, games['vec'], games['mask'], [np.array([2, 3])]), dim=1)
    assert np.allclose(out[2], want[0].detach().numpy(), atol=1e-5)
    assert np.allclose(np.exp(out[2]).sum(), 1.0)


def test_split_players_keeps_player_whole():
    pid = np.repeat(['a', 'b', 'c', 'd', 'e'], 4)
    rank = np.repeat([0, 1, 2, 3, 4], 4)
    train, val = st.split_players(pid, rank, 0.4, np.random.default_rng(0))
    assert len(val) == 2 and len(train) == 3
    assert not set(np.concatenate(train)) & set(np.concatenate(val))
    assert all(len(set(pid[g])) == 1 for g in train + val)


def test_split_players_rejects_player_across_ranks():
    with pytest.raises(ValueError):
        st.split_players(np.array(['a', 'a']), np.array([3, 4]), 0.5, np.random.default_rng(0))


def test_training_learns_separable_signal():
    games = toy_games(n_players=60, games_per_player=8)
    args = argparse.Namespace(seed=0, dim=16, dropout=0.0, weight_decay=0.0, batch=16, questions_per_player=20)
    rng = np.random.default_rng(0)
    train_p, val_p = st.split_players(games['player_id'], games['rank_idx'], 0.3, rng)
    mean, std = st.fit_standardizer(games['vec'], games['mask'], rng)
    val_q, owner = st.player_questions(val_p, np.random.default_rng(1), 4)
    _, score, epoch, history = st.train_one(games['vec'], games['mask'], games['rank_idx'], train_p, val_p, val_q, owner,
                                            lr=3e-3, epochs=5, args=args, mean=mean, std=std,
                                            rng=np.random.default_rng(2), log=lambda row: None)
    assert history[-1]['train_loss'] < history[0]['train_loss']
    assert score > 0.5 and 1 <= epoch <= 5


def test_fit_standardizer_rows_matches_copy():
    games = toy_games(n_players=20, games_per_player=4)
    games['mask'][::3, 20:] = False
    rows = np.arange(5, 60, 2)
    a = st.fit_standardizer(games['vec'][rows], games['mask'][rows], np.random.default_rng(7), n=500)
    b = st.fit_standardizer(games['vec'], games['mask'], np.random.default_rng(7), n=500, rows=rows)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_class_weights_equalize_levels():
    labels = np.array([0] * 6 + [1] * 2 + [2] * 2)
    w = st.class_weights(labels)
    assert w.shape == (st.N_LEVELS,)
    total = np.bincount(labels, minlength=st.N_LEVELS) * w
    assert np.allclose(total[:3], total[0]) and np.all(w[3:] == 0)
    assert np.isclose(total[0], len(labels) / st.N_LEVELS)


def test_training_with_balance_runs():
    games = toy_games(n_players=60, games_per_player=8)
    args = argparse.Namespace(seed=0, dim=16, dropout=0.0, weight_decay=0.0, batch=16, questions_per_player=4,
                              balance=True)
    rng = np.random.default_rng(0)
    train_p, val_p = st.split_players(games['player_id'], games['rank_idx'], 0.3, rng)
    mean, std = st.fit_standardizer(games['vec'], games['mask'], rng)
    val_q, owner = st.player_questions(val_p, np.random.default_rng(1), 4)
    _, _, _, history = st.train_one(games['vec'], games['mask'], games['rank_idx'], train_p, val_p, val_q, owner,
                                    lr=3e-3, epochs=2, args=args, mean=mean, std=std,
                                    rng=np.random.default_rng(2), log=lambda row: None)
    assert len(history) == 2 and np.isfinite(history[-1]['train_loss'])


def test_load_games_pads_and_aligns(tmp_path):
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(5, st.DIM)).astype(np.float16)
    np.savez(tmp_path / 'part-000000.npz', key=np.array(['a', 'b', 'c']), player_id=np.array(['p', 'p', 'q']),
             rank_idx=np.array([0, 0, 1]), count=np.array([3, 0, 2]), vec=vec)
    games = st.load_games(str(tmp_path))
    assert games['mask'].sum(axis=1).tolist() == [3, 0, 2]
    assert np.array_equal(games['vec'][2, :2], vec[3:5]) and not games['vec'][1].any()
    v, m = st.align(games, ['c', 'b'])
    assert m.sum(axis=1).tolist() == [2, 0] and np.array_equal(v[0, :2], vec[3:5])
    with pytest.raises(ValueError):
        st.align(games, ['c', 'zzz'])


def test_load_games_multiple_dirs(tmp_path):
    rng = np.random.default_rng(0)
    for name, keys in [('a', ['k1', 'k2']), ('b', ['k3'])]:
        d = tmp_path / name
        d.mkdir()
        np.savez(d / 'part-000000.npz', key=np.array(keys), player_id=np.array(['p'] * len(keys)),
                 rank_idx=np.zeros(len(keys), np.int64), count=np.ones(len(keys), np.int64),
                 vec=rng.normal(size=(len(keys), st.DIM)).astype(np.float16))
    games = st.load_games([str(tmp_path / 'a'), str(tmp_path / 'b')])
    assert list(games['key']) == ['k1', 'k2', 'k3']
    with pytest.raises(SystemExit):  # 同一個目錄給兩次，key 重複
        st.load_games([str(tmp_path / 'a'), str(tmp_path / 'a')])
