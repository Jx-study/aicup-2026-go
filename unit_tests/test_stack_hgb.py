import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aicup import stack_hgb as sh
from aicup import stack_humansl
from aicup.fit_humansl import load_games, player_keys, question_points
from aicup.scoring import pick_level
from aicup.stack_humansl import make_eval_splits

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'humansl-analysis'


def toy_rows(n_players=50, games=20, moves=30, seed=0, move_stats=True, prefix='p',
             profiles=('rank_11k', 'rank_1d', 'rank_2d'), loss=False):
    """目標側與對手側的 jsonl 列，機率帶有等級訊號。loss 為 True 時加 score_loss、winrate_loss，數值隨等級下降，部分手為 None。"""
    rng = np.random.default_rng(seed)
    target, opp = [], []
    for i in range(n_players):
        rank = i % 10
        for g in range(games):
            key = '%s%d_g%d' % (prefix, i, g)
            for side, rows in (('target', target), ('opponent', opp)):
                base = rng.uniform(0.01, 0.9, moves) * (0.5 + rank / 20)
                probs = [np.clip(base * s + rng.normal(0, 0.02, moves), 0, 1) for s in (0.6, 0.8, 1.0)]
                row = {'key': key, 'rank_idx': rank, 'player_id': '%s%d' % (prefix, i), 'hd': 0, 'color': 'B',
                       'side': side, 'turns': list(range(1, 2 * moves, 2)), 'eval_visits': 0,
                       'profiles': list(profiles),
                       'probs': [[round(float(v), 6) for v in p] for p in probs]}
                row['probs'][0][0] = None  # 不合法的手
                if move_stats:
                    row['pre_root_history'] = True
                    row['move_rank'] = [[float(v) for v in rng.integers(0, 10, moves)] for _ in range(3)]
                    row['pmax'] = [[float(v) for v in rng.uniform(0.1, 1, moves)] for _ in range(3)]
                    row['entropy'] = [[float(v) for v in rng.uniform(0, 4, moves)] for _ in range(3)]
                if loss:
                    row['score_loss'] = [None if m % 11 == 0 else float(v)
                                         for m, v in enumerate(rng.exponential(5 - rank / 3, moves))]
                    row['winrate_loss'] = [None if m % 13 == 0 else float(v)
                                           for m, v in enumerate(rng.exponential(0.1 - rank / 150, moves))]
                rows.append(row)
    return target, opp


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return str(path)


def toy_files(tmp_path, name='dev', **kw):
    target, opp = toy_rows(**kw)
    return (write_jsonl(tmp_path / ('%s.jsonl' % name), target),
            write_jsonl(tmp_path / ('%s-opp.jsonl' % name), opp))


@pytest.fixture
def fast_hgb(monkeypatch):
    monkeypatch.setattr(sh, 'ITERS', (3, 6))


def toy_cnn(data, seed=1):
    rng = np.random.default_rng(seed)
    logits = -0.5 * (np.arange(10)[None, :] - data['rank_idx'][:, None] - rng.normal(0, 2, (len(data['key']), 1))) ** 2
    return np.exp(logits) / np.exp(logits).sum(1, keepdims=True)


def test_game_side_features_widths_and_nan():
    target, _ = toy_rows(n_players=1, games=1)
    dist, stats, s, r = sh.game_side_features(target[0])
    assert dist.shape == (18,) and stats.shape == (8,)
    assert np.isfinite(dist).all() and np.isfinite(stats).all() and np.isfinite([s, r]).all()
    row = dict(target[0])
    for k in ('move_rank', 'pmax', 'entropy'):
        row.pop(k)
    dist2, stats2, s2, r2 = sh.game_side_features(row)
    assert np.array_equal(dist, dist2) and s == s2
    assert np.isnan(stats2).all() and np.isnan(r2)
    row['probs'] = [[None] * 30 for _ in range(3)]
    dist3, _, s3, _ = sh.game_side_features(row)
    assert np.isnan(dist3).all() and np.isnan(s3)


def test_game_side_features_values():
    row = {'profiles': ['rank_11k', 'rank_1d', 'rank_2d'],
           'probs': [[0.5, 0.001, None], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
           'move_rank': [[0, 1, None], [0, 0, 3], [0, 2, 2]],
           'pmax': [[1, 1, 1], [0.4, 0.6, None], [1, 1, 1]], 'entropy': [[0, 0, 0], [1, 2, 3], [0, 0, 0]]}
    dist, stats, s, r = sh.game_side_features(row)
    assert dist[5] == pytest.approx(0.5)  # rank_11k 兩手中一手 < 0.01
    assert dist[2] == pytest.approx(np.log(0.5 * 0.001) / 2)  # 兩手的中位數
    assert stats[0] == pytest.approx(0.5) and stats[2] == pytest.approx(2 / 3)
    assert stats[6] == pytest.approx(0.5) and stats[7] == pytest.approx(2.0)
    assert s == pytest.approx(((np.log(0.5) - np.log(0.5)) + (np.log(0.5) - np.log(0.001))) / 2)
    assert r == pytest.approx(np.mean(np.log1p([0, 2, 2])))


def test_question_matrix_widths(tmp_path):
    data, games = sh.load_game_features(*toy_files(tmp_path, n_players=10))
    q = [np.arange(5), np.arange(20, 40)]
    m = sh.question_matrix(games, toy_cnn(data), q, 3.0)
    assert {g: m[g].shape[1] for g in m} == {**sh.GROUP_WIDTH, 'co': 3}
    assert sh.stack_groups(m, sh.K_GROUPS).shape == (2, 83)
    assert sh.stack_groups(m, sh.STAGE1_GROUPS['hgb']).shape == (2, 73)
    assert m['logn'][:, 0] == pytest.approx(np.log([5, 20]))


def test_within_stats_drops_nan():
    out = sh.within_stats([1.0, np.nan, 3.0])
    assert out == pytest.approx([1.4, 2.0, 2.6, 1.0])
    assert np.isnan(sh.within_stats([np.nan, np.nan])).all()


def test_question_features_use_only_own_games(tmp_path):
    data, games = sh.load_game_features(*toy_files(tmp_path, n_players=10))
    cnn = toy_cnn(data)
    q = np.array([3, 7, 11, 15, 19])
    before = sh.question_matrix(games, cnn, [q], 3.0)
    rng = np.random.default_rng(5)
    others = np.setdiff1d(np.arange(len(data['key'])), q)
    perm = np.arange(len(data['key']))
    perm[others] = rng.permutation(others)
    shuffled = {k: v[perm] for k, v in games.items()}
    after = sh.question_matrix(shuffled, cnn[perm], [q], 3.0)
    for g in before:
        assert np.array_equal(before[g], after[g], equal_nan=True), g


def test_features_unchanged_when_labels_shuffled(tmp_path):
    target, opp = toy_rows(n_players=20)
    data, games = sh.load_game_features(write_jsonl(tmp_path / 't.jsonl', target), write_jsonl(tmp_path / 'o.jsonl', opp))
    new_rank = dict(zip(['p%d' % i for i in range(20)], np.random.default_rng(3).permutation(np.arange(20) % 10)))
    relabel = [[dict(r, rank_idx=int(new_rank[r['player_id']])) for r in rows] for rows in (target, opp)]
    data2, games2 = sh.load_game_features(write_jsonl(tmp_path / 't2.jsonl', relabel[0]),
                                          write_jsonl(tmp_path / 'o2.jsonl', relabel[1]))
    assert not np.array_equal(data['rank_idx'], data2['rank_idx'])
    q = [np.arange(i * 20, i * 20 + 10) for i in range(20)]
    cnn = toy_cnn(data)
    m1, m2 = sh.question_matrix(games, cnn, q, 3.0), sh.question_matrix(games2, cnn, q, 3.0)
    for g in m1:
        assert np.array_equal(m1[g], m2[g], equal_nan=True), g


def test_staged_prediction_matches_shorter_fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5))
    y = (x[:, 0] * 2 + rng.normal(size=300)).round().clip(-2, 2).astype(int) + 2
    full = sh.hgb_model(2, 12, 0).fit(x, y)
    staged = list(full.staged_predict_proba(x))[5]
    short = sh.hgb_model(2, 6, 0).fit(x, y)
    assert np.allclose(staged, short.predict_proba(x))


def test_select_hgb_tie_takes_smallest():
    x = np.zeros((100, 2))
    y = np.repeat(np.arange(5), 20)
    groups = np.repeat(np.arange(25), 4)
    best, scores = sh.select_hgb(x, y, groups, 0, inner_folds=5, depths=(2, 3), iters=(2, 4))
    assert len(set(scores.values())) == 1
    assert best == (2, 2)


def run_combos_preds(monkeypatch, target, opp, cnn, splits, seed, sizes, train_repeats, cs, inner_folds):
    """執行 stack_humansl.run_combos 的 cnn+hsl+hslo，攔截最後一次 question_points 取得逐題預測。"""
    calls = []
    original = stack_humansl.question_points

    def record(labels, preds):
        calls.append(np.asarray(preds))
        return original(labels, preds)

    monkeypatch.setattr(stack_humansl, 'question_points', record)
    args = SimpleNamespace(se=None, se_opp=None, hsl_opp=opp, pm=None, combos='cnn+hsl+hslo', seed=seed,
                           sizes=sizes, train_repeats=train_repeats, cnn_temp=3.0, cs=cs, inner_folds=inner_folds,
                           prior='keep', set_model=None, set_dir=None, out=None)
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    stack_humansl.run_combos(args, load_games(target), cnn, splits, labels, players)
    return calls[-1]


def test_lr_base_matches_stack_humansl(tmp_path, monkeypatch, capsys):
    target, opp = toy_files(tmp_path, n_players=60)
    old = sh.load_game_features(target, opp)
    data = old[0]
    cnn = toy_cnn(data)
    eval_players = set(player_keys(data)[:50 * 20:20])  # 其餘 10 位只當訓練
    splits = make_eval_splits(data, (5, 10), 2, eval_players, seed=7)
    preds, _ = sh.evaluate(old, old, cnn, splits, ('lr_base',), 7, sizes=(5, 10), train_repeats=2, cs=(0.1, 1.0),
                           inner_folds=3, log=lambda *_: None)
    ref = run_combos_preds(monkeypatch, target, opp, cnn, splits, 7, (5, 10), 2, (0.1, 1.0), 3)
    assert len(ref) == len(preds['lr_base'])
    assert np.array_equal(preds['lr_base'], ref)


def test_evaluate_runs_all_arms(tmp_path, fast_hgb):
    dev = sh.load_game_features(*toy_files(tmp_path))
    pool = sh.load_game_features(*toy_files(tmp_path, name='pool', n_players=20, prefix='q', seed=9))
    cnn = toy_cnn(dev[0])
    splits = make_eval_splits(dev[0], (5, 10), 1, None, seed=1)
    labels = np.concatenate([s[2] for s in splits])
    preds, params = sh.evaluate(dev, dev, cnn, splits, sh.ALL_ARMS, 1, sizes=(5, 10), train_repeats=2,
                                cs=(0.1,), inner_folds=2, pool=pool, log=lambda *_: None)
    assert set(preds) == set(sh.ALL_ARMS)
    for arm, p in preds.items():
        assert len(p) == len(labels)
        assert question_points(labels, p).mean() > 0.2, arm
    assert all(len(v) == len(splits) for v in params.values())


def test_evaluate_rejects_mismatched_order(tmp_path):
    dev = sh.load_game_features(*toy_files(tmp_path, n_players=10))
    order = np.random.default_rng(0).permutation(len(dev[0]['key']))
    other = ({k: (v[order] if isinstance(v, np.ndarray) else v) for k, v in dev[0].items()}, dev[1])
    with pytest.raises(ValueError, match='順序'):
        sh.evaluate(dev, other, toy_cnn(dev[0]), [], ('lr_base',), 0)


def toy_stage_inputs(n_players=30, per_player=4, seed=0, prefix='p'):
    rng = np.random.default_rng(seed)
    g = np.repeat(['9|%s%d' % (prefix, i) for i in range(n_players)], per_player)
    y = np.repeat(np.arange(n_players) % 5, per_player)
    f = y[:, None] + rng.normal(0, 1, (len(y), 3))
    return f, y, g


def test_two_stage_never_fits_on_predicted_players(fast_hgb):
    f, y, g = toy_stage_inputs()
    pool = toy_stage_inputs(20, seed=1, prefix='q')
    f_te = toy_stage_inputs(10, seed=2, prefix='t')[0]
    cnn_tr, cnn_te = np.zeros((len(y), 10)), np.zeros((len(f_te), 10))
    for kind in ('lr', 'hgb', 'hgb_hsl'):
        probs, s1, info = sh.two_stage(kind, (f, y, g), f_te, pool, cnn_tr, cnn_te, 0, cs=(1.0,), inner_folds=3)
        assert probs.shape == s1.shape == (len(f_te), 10)
        assert np.allclose(s1.sum(1), 1)
        test_fit = [s for used, s in info['fits'] if used == 'test']
        oof_fits = [s for used, s in info['fits'] if used == 'oof']
        assert len(test_fit) == 1 and test_fit[0] == set(g) | set(pool[2])
        assert len(oof_fits) == 3
        for s in oof_fits:
            assert set(pool[2]) <= s
            assert len(set(g) - s) > 0  # 每折都留下被預測的開發組玩家


def test_two_stage_rejects_pool_overlap():
    f, y, g = toy_stage_inputs()
    with pytest.raises(ValueError, match='重複'):
        sh.two_stage('lr', (f, y, g), f[:3], (f, y, g), np.zeros((len(y), 10)), np.zeros((3, 10)), 0)


def test_evaluate_rejects_pool_overlapping_dev(tmp_path):
    dev = sh.load_game_features(*toy_files(tmp_path))
    splits = make_eval_splits(dev[0], (5,), 1, None, seed=0)
    with pytest.raises(ValueError, match='player_id'):
        sh.evaluate(dev, dev, toy_cnn(dev[0]), splits, ('pool',), 0, sizes=(5,), pool=dev, log=lambda *_: None)


def test_check_disjoint():
    sh.check_disjoint({'a', 'b'}, {'x': {'c'}})
    with pytest.raises(ValueError, match='x'):
        sh.check_disjoint({'a', 'b'}, {'x': {'b'}})


def test_real_player_lists_disjoint():
    names = ('val-players.csv', 'val-players-newdev-v180000.csv', 'holdout-v180000.csv')
    paths = [ANALYSIS / 'train-pool-v180000.csv'] + [ANALYSIS / n for n in names]
    if not all(p.exists() for p in paths):
        pytest.skip('沒有本機的玩家清單')
    pool = sh.read_player_ids(paths[0])
    assert len(pool) == 1000
    lists = {p.name: sh.read_player_ids(p) for p in paths[1:]}
    sh.check_disjoint(pool, lists)
    for a in lists:
        for b in lists:
            if a < b:
                assert not lists[a] & lists[b], (a, b)


def test_load_side_rejects_wrong_side(tmp_path):
    target, opp = toy_files(tmp_path, n_players=2)
    data, _ = sh.load_game_features(target, opp)
    with pytest.raises(ValueError, match='side'):
        sh.load_side(opp, data, 'target')


def test_run_and_compare_cli(tmp_path, monkeypatch, capsys):
    target, opp = toy_files(tmp_path)
    data = sh.load_game_features(target, opp)[0]
    np.savez(tmp_path / 'cnn.npz', key=data['key'], probs=toy_cnn(data), rank_idx=data['rank_idx'])
    for seed in (1, 2):
        monkeypatch.setattr('sys.argv', ['stack_hgb', 'run', '--jsonl', target, '--opp', opp, '--hist-jsonl', target,
                                         '--hist-opp', opp, '--cnn-probs', str(tmp_path / 'cnn.npz'),
                                         '--arms', 'lr_base,lr_hist,hgb_all_lr', '--sizes', '5,10', '--train-repeats', '1',
                                         '--test-repeats', '1', '--inner-folds', '2', '--cs', '1',
                                         '--seed', str(seed), '--out', str(tmp_path / ('r-s%d.npz' % seed))])
        sh.main()
    z = np.load(tmp_path / 'r-s1.npz')
    assert np.array_equal(z['score_lr_base'], z['score_lr_hist'])  # 兩組 jsonl 相同
    assert set(json.loads(str(z['params']))) == {'lr_base', 'lr_hist', 'hgb_all_lr'}
    monkeypatch.setattr('sys.argv', ['stack_hgb', 'compare', '--files', str(tmp_path / 'r-s{seed}.npz'),
                                     '--seeds', '1,2', '--main', 'hgb_all_lr', '--base', 'lr_base', '--refs', 'lr_hist-lr_base'])
    sh.main()
    out = capsys.readouterr().out
    assert '判定：' in out and 'lr_hist 減 lr_base' in out


def test_load_game_features_d_is_target_minus_opponent(tmp_path):
    target_rows, opp_rows = toy_rows(n_players=10)
    data, games = sh.load_game_features(write_jsonl(tmp_path / 't.jsonl', target_rows),
                                        write_jsonl(tmp_path / 'o.jsonl', opp_rows[::-1]))  # 對手側列順序不同
    opp_by_key = {r['key']: r for r in opp_rows}
    for i in (0, 57, 199):
        _, _, s_t, r_t = sh.game_side_features(target_rows[i])
        dist_o, stats_o, s_o, _ = sh.game_side_features(opp_by_key[data['key'][i]])
        assert games['s'][i] == pytest.approx(s_t) and games['r'][i] == pytest.approx(r_t)
        assert games['d'][i] == pytest.approx(s_t - s_o)
        assert np.array_equal(games['disto'][i], dist_o) and np.array_equal(games['statso'][i], stats_o)


def test_load_real_history_jsonl(tmp_path):
    target = ANALYSIS / 'hist-val648' / 'val648-p3-hist.jsonl'
    opp = ANALYSIS / 'hist-val648_opp' / 'val648-p3-opp-hist.jsonl'
    if not (target.exists() and opp.exists()):
        pytest.skip('沒有本機的 history 版 jsonl')
    with open(target, encoding='utf-8') as f:
        lines = [next(f) for _ in range(200)]
    keys = {json.loads(line)['key'] for line in lines}
    with open(opp, encoding='utf-8') as f:
        opp_lines = [line for line in f if json.loads(line)['key'] in keys]
    (tmp_path / 't.jsonl').write_text(''.join(lines), encoding='utf-8')
    (tmp_path / 'o.jsonl').write_text(''.join(opp_lines), encoding='utf-8')
    data, games = sh.load_game_features(str(tmp_path / 't.jsonl'), str(tmp_path / 'o.jsonl'))
    assert len(data['key']) == 200
    widths = {g: games[g].shape[1] for g in ('hsl', 'dist', 'stats', 'hslo', 'disto', 'statso')}
    assert widths == {g: sh.GROUP_WIDTH[g] for g in widths}
    for g in ('hsl', 'dist', 'stats', 'hslo', 'disto', 'statso', 's', 'r', 'd'):
        assert np.isfinite(games[g]).all(), g
    assert ((games['stats'][:, [0, 2, 4]] >= 0) & (games['stats'][:, [0, 2, 4]] <= 1)).all()


def test_pool_requires_exclude_players(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['stack_hgb', 'run', '--jsonl', 'a', '--opp', 'b', '--hist-jsonl', 'c',
                                     '--hist-opp', 'd', '--cnn-probs', 'e', '--pool-jsonl', 'f', '--pool-opp', 'g',
                                     '--out', 'h'])
    with pytest.raises(SystemExit):
        sh.main()
    assert '--exclude-players' in capsys.readouterr().err


def test_hgb_hsl_stage1_width_and_model(tmp_path, fast_hgb):
    assert sh.STAGE1_GROUPS['hgb_hsl'] == ('hsl', 'hslo')
    data, games = sh.load_game_features(*toy_files(tmp_path, n_players=50))
    m = sh.question_matrix(games, None, [np.arange(5)], 3.0)
    feat = sh.stack_groups(m, sh.STAGE1_GROUPS['hgb_hsl'])
    assert feat.shape == (1, 8)

    pool = sh.load_game_features(*toy_files(tmp_path, name='pool', n_players=20, prefix='q', seed=3))
    cnn = toy_cnn(data)
    splits = make_eval_splits(data, (5,), 1, None, seed=2)
    preds, params = sh.evaluate(
        (data, games), (data, games), cnn, splits,
        arms=('pool',), seed=2, sizes=(5,), train_repeats=1,
        cs=(1.0,), inner_folds=2, pool=pool, stage1='hgb_hsl', log=lambda *_: None
    )
    assert 'pool_hgbhsl' in preds
    assert len(preds['pool_hgbhsl']) == sum(len(s[2]) for s in splits)
    p = params['pool_hgbhsl'][0]['stage1']
    assert isinstance(p, list) and len(p) == 2


def test_merge_two_stage_outputs_no_conflict(tmp_path, fast_hgb, monkeypatch):
    target, opp = toy_files(tmp_path)
    data = sh.load_game_features(target, opp)[0]
    p_target, p_opp = toy_files(tmp_path, name='pool', n_players=10, prefix='q', seed=8)
    np.savez(tmp_path / 'cnn.npz', key=data['key'], probs=toy_cnn(data), rank_idx=data['rank_idx'])
    holdout = tmp_path / 'holdout.csv'
    holdout.write_text('player_id\nnone\n', encoding='utf-8')

    # Run 1: --stage1 lr
    monkeypatch.setattr('sys.argv', [
        'stack_hgb', 'run', '--jsonl', target, '--opp', opp, '--hist-jsonl', target, '--hist-opp', opp,
        '--cnn-probs', str(tmp_path / 'cnn.npz'),
        '--pool-jsonl', p_target, '--pool-opp', p_opp, '--exclude-players', str(holdout),
        '--stage1', 'lr', '--arms', 'lr_base,lr_hist,pool,pool_s1',
        '--sizes', '5', '--train-repeats', '1', '--test-repeats', '1', '--inner-folds', '2', '--cs', '1',
        '--seed', '42', '--out', str(tmp_path / 'r-lr-s42.npz')
    ])
    sh.main()

    # Run 2: --stage1 hgb_hsl
    monkeypatch.setattr('sys.argv', [
        'stack_hgb', 'run', '--jsonl', target, '--opp', opp, '--hist-jsonl', target, '--hist-opp', opp,
        '--cnn-probs', str(tmp_path / 'cnn.npz'),
        '--pool-jsonl', p_target, '--pool-opp', p_opp, '--exclude-players', str(holdout),
        '--stage1', 'hgb_hsl', '--arms', 'pool,pool_s1',
        '--sizes', '5', '--train-repeats', '1', '--test-repeats', '1', '--inner-folds', '2', '--cs', '1',
        '--seed', '42', '--out', str(tmp_path / 'r-hgbhsl-s42.npz')
    ])
    sh.main()

    # 合併讀取兩個檔案，確認沒有衝突
    scores, players, n_games = sh.load_seed_scores([
        str(tmp_path / 'r-lr-s42.npz'),
        str(tmp_path / 'r-hgbhsl-s42.npz')
    ])
    assert 'lr_base' in scores
    assert 'pool' in scores
    assert 'pool_hgbhsl' in scores


def test_parse_profiles():
    assert sh.parse_profiles('rank_11k,rank_1d,rank_6d') == ('rank_11k', 'rank_1d', 'rank_6d')
    assert sh.parse_profiles(' rank_11k, rank_1d, rank_2d ') == sh.PROFILES
    for bad in ('rank_11k,rank_1d', 'rank_1d,rank_11k,rank_6d', 'rank_11k,rank_1d,rank_2d,rank_6d'):
        with pytest.raises(argparse.ArgumentTypeError, match='rank_11k,rank_1d'):
            sh.parse_profiles(bad)


def test_load_game_features_profiles_param(tmp_path):
    p6 = ('rank_11k', 'rank_1d', 'rank_6d')
    files_6d = toy_files(tmp_path, name='d6', n_players=10, profiles=p6)
    with pytest.raises(ValueError, match='rank_2d'):
        sh.load_game_features(*files_6d)
    _, g6 = sh.load_game_features(*files_6d, profiles=p6)
    _, g2 = sh.load_game_features(*toy_files(tmp_path, name='d2', n_players=10))
    # 兩份 toy 只有 profile 名稱不同，特徵要相同
    for k in ('hsl', 'hslo', 'dist', 'stats', 's', 'r', 'd'):
        np.testing.assert_array_equal(g6[k], g2[k])


def test_game_side_features_uses_top_profile():
    probs = [[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]]
    row = {'profiles': list(sh.PROFILES), 'probs': probs,
           'move_rank': [[0, 1], [0, 0], [3, 3]], 'pmax': [[0.5, 0.5]] * 3, 'entropy': [[1.0, 1.0]] * 3}
    _, _, s, r = sh.game_side_features(row)
    assert np.isclose(s, np.log(4.0))
    assert np.isclose(r, np.log(4.0))


def test_two_stage_lr_loss_fills_nan():
    rng = np.random.default_rng(0)
    n = 200
    y = np.repeat(np.arange(10), n // 10)
    g = np.array(['%d|p%d' % (c, i % 4) for i, c in enumerate(y)])
    f = rng.normal(size=(n, 4)) + y[:, None] * 0.3
    f[::7, 2] = np.nan
    cnn = np.log(np.full((n, 10), 0.1))
    probs, s1, info = sh.two_stage('lr_loss', (f, y, g), f[:20], None, cnn, cnn[:20], seed=0, inner_folds=2)
    assert np.isfinite(probs).all() and np.isfinite(s1).all()


def test_fill_nan_uses_fit_mean():
    f = np.array([[1.0, np.nan], [3.0, np.nan]])
    g = np.array([[np.nan, 5.0]])
    a, b = sh.fill_nan(f, g)
    assert np.array_equal(a, [[1.0, 0.0], [3.0, 0.0]])
    assert np.array_equal(b, [[2.0, 5.0]])
