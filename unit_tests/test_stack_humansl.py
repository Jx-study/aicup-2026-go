import json
import re
import sys

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from aicup import stack_humansl as sh
from aicup.fit_humansl import make_splits, question_points
from aicup.scoring import aggregate_games, apply_temperature, pick_level
from stack_helpers import toy_data, write_npz


def test_load_cnn_probs_aligns_by_key(tmp_path):
    probs = np.eye(10)[[1, 2, 3]]
    write_npz(tmp_path / 'v.npz', np.array(['a', 'b', 'c']), probs, np.array([1, 2, 3]))
    out = sh.load_cnn_probs(str(tmp_path / 'v.npz'), np.array(['c', 'a']), np.array([3, 1]))
    assert np.array_equal(out, probs[[2, 0]])


def test_load_cnn_probs_rejects_missing_key(tmp_path):
    write_npz(tmp_path / 'v.npz', np.array(['a']), np.eye(10)[[1]], np.array([1]))
    with pytest.raises(ValueError, match='b'):
        sh.load_cnn_probs(str(tmp_path / 'v.npz'), np.array(['a', 'b']), np.array([1, 1]))


def test_load_cnn_probs_rejects_rank_mismatch(tmp_path):
    write_npz(tmp_path / 'v.npz', np.array(['a']), np.eye(10)[[1]], np.array([1]))
    with pytest.raises(ValueError, match='等級'):
        sh.load_cnn_probs(str(tmp_path / 'v.npz'), np.array(['a']), np.array([2]))


def test_cnn_question_probs_matches_predict_rank_aggregation():
    _, probs = toy_data()
    questions = [np.array([0, 1, 2]), np.array([5])]
    out = sh.cnn_question_probs(probs, questions, temp=3.0)
    scaled = apply_temperature(probs, 3.0)
    assert np.allclose(out[0], aggregate_games(scaled[[0, 1, 2]], 'log'))
    assert np.allclose(out[1], aggregate_games(scaled[[5]], 'log'))


def test_question_features_concat_cnn_logp_and_mean_hsl():
    data, probs = toy_data()
    questions = [np.array([0, 1]), np.array([3, 4, 5])]
    f = sh.question_features(data['x'], probs, questions, temp=3.0)
    assert f.shape == (2, 10 + 4)
    assert np.allclose(f[:, :10], np.log(np.maximum(sh.cnn_question_probs(probs, questions, 3.0), sh.EPS)))
    assert np.isfinite(f).all()
    assert np.allclose(f[1, 10:], data['x'][[3, 4, 5]].mean(0))


def test_predict_stack_without_hsl_uses_only_cnn_features():
    # Human SL 特徵換成雜訊時，只用 CNN 的結果不變
    data, probs = toy_data()
    splits = make_splits(data, (5, 20), 2, seed=42)
    train, questions, _, _ = splits[0]
    noisy = dict(data, x=np.random.default_rng(9).normal(size=data['x'].shape))
    a, _ = sh.predict_stack(data, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                            temp=3.0, cs=(0.1, 1.0), inner_folds=3, use_hsl=False)
    b, _ = sh.predict_stack(noisy, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                            temp=3.0, cs=(0.1, 1.0), inner_folds=3, use_hsl=False)
    assert np.allclose(a, b)


def test_select_c_rejects_over_regularized_model():
    # C 太小時機率接近均勻，選不出正確等級
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 30)
    x = y[:, None] + rng.normal(0, 0.3, (len(y), 2))
    groups = np.arange(len(y)) % 15
    assert sh.select_c(x, y, groups, cs=(1e-6, 10.0), inner_folds=3) == 10.0


def test_select_c_never_shares_a_group_across_inner_folds(monkeypatch):
    overlaps = []
    expected = np.arange(300) % 15

    class SpySplitter(GroupKFold):
        def split(self, X, y=None, groups=None):
            assert np.array_equal(groups, expected)
            for tr, te in super().split(X, y, groups):
                overlaps.append(set(groups[tr]) & set(groups[te]))
                yield tr, te

    monkeypatch.setattr(sh, 'GroupKFold', SpySplitter)
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 30)
    x = y[:, None] + rng.normal(0, 0.3, (len(y), 2))
    sh.select_c(x, y, expected, cs=(0.1, 1.0), inner_folds=3)
    assert len(overlaps) == 6 and not any(overlaps)


def test_train_rng_is_reproducible_per_seed_and_fold():
    a = sh.train_rng(42, 0).random(5)
    assert np.array_equal(a, sh.train_rng(42, 0).random(5))
    assert not np.array_equal(a, sh.train_rng(43, 0).random(5))
    assert not np.array_equal(a, sh.train_rng(42, 1).random(5))


def test_make_splits_depends_on_seed():
    data, _ = toy_data()
    a = make_splits(data, (5, 20), 2, seed=1)
    b = make_splits(data, (5, 20), 2, seed=1)
    c = make_splits(data, (5, 20), 2, seed=2)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(a, c))


def test_predict_stack_ignores_test_player_labels():
    data, probs = toy_data()
    splits = make_splits(data, (5, 20), 2, seed=42)
    train, questions, _, _ = splits[0]
    a, c_a = sh.predict_stack(data, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                              temp=3.0, cs=(0.1, 1.0), inner_folds=3)
    leaked = dict(data, rank_idx=np.where(train, data['rank_idx'], 0))
    b, c_b = sh.predict_stack(leaked, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                              temp=3.0, cs=(0.1, 1.0), inner_folds=3)
    assert c_a == c_b and np.allclose(a, b)
    assert a.shape == (len(questions), 10) and np.allclose(a.sum(1), 1)


def test_predict_stack_beats_chance_on_informative_features():
    data, probs = toy_data()
    splits = make_splits(data, (5, 20), 2, seed=42)
    points = []
    for train, questions, labels, _ in splits:
        p, _ = sh.predict_stack(data, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                                temp=3.0, cs=(0.1, 1.0), inner_folds=3)
        points.append(question_points(labels, pick_level(p)))
    # 隨機猜約 0.2，特徵有訊號時應明顯高於此
    assert np.concatenate(points).mean() > 0.5


def test_main_runs_end_to_end_on_toy_data(tmp_path, monkeypatch, capsys):
    data, probs = toy_data()
    write_npz(tmp_path / 'v.npz', data['key'], probs, data['rank_idx'])
    monkeypatch.setattr(sh, 'load_games', lambda path: data)
    monkeypatch.setattr(sys, 'argv', ['stack_humansl', 'unused.jsonl', '--cnn-probs', str(tmp_path / 'v.npz'),
                                      '--train-repeats', '2', '--test-repeats', '2', '--inner-folds', '3'])
    sh.main()
    out = capsys.readouterr().out
    for name in ('stacking', 'CNN 重新擬合', 'Human SL avg A C=0.1', 'CNN', 'stacking - CNN', '暫定門檻'):
        assert name in out
    # 門檻判定要和印出的區間下限一致
    lo = float(re.search(r'stacking - CNN：\S+，95% 區間 (\S+)～', out).group(1))
    verdict = out.split('暫定門檻')[1].split('：')[1].strip()
    assert verdict == ('通過' if lo > 0 else '未通過')


def write_se(path, keys, rank_idx, n_sym=8, seed=0, side='target'):
    """每局 SE∞ score 隨等級上升，對稱之間加雜訊。"""
    rng = np.random.default_rng(seed)
    with open(path, 'w', encoding='utf-8') as f:
        for k, y in zip(keys, rank_idx):
            mean = (y + rng.normal(0, 1, n_sym)).tolist()
            sl = np.full(11, 0.01)
            sl[min(y + 1, 10)] = 0.9
            f.write(json.dumps({'key': str(k), 'rank_idx': int(y), 'n': 50, 'score_mean': mean,
                                'score_std': [1.0] * n_sym, 'sl_rank': (sl / sl.sum()).tolist(),
                                'hd': int(y % 3 == 0), 'color': 'BW'[int(y) % 2], 'side': side}) + '\n')


def test_load_se_blocks_aligns_by_key_and_averages_symmetries(tmp_path):
    keys, rank = np.array(['a', 'b', 'c']), np.array([3, 1, 2])
    with open(tmp_path / 's.jsonl', 'w', encoding='utf-8') as f:
        for k, y, m, hd, color in (('b', 1, list(range(8)), None, 'B'), ('a', 3, [8.0] * 8, 2, 'W'),
                                   ('c', 2, [0.0] * 8, 1, 'B')):
            f.write(json.dumps({'key': k, 'rank_idx': y, 'n': 5, 'score_mean': m, 'score_std': [2.0] * 8,
                                'sl_rank': [1 / 11] * 11, 'hd': hd, 'color': color}) + '\n')
    b = sh.load_se_blocks(str(tmp_path / 's.jsonl'), keys, rank)
    assert b['se1'][1].tolist() == [0.0, 2.0] and b['se4'][1, 0] == 1.5 and b['se8'][1, 0] == 3.5
    assert b['se8'][0, 0] == 8.0
    assert b['sl'].shape == (3, 11) and np.allclose(b['sl'], np.log(1 / 11))
    assert b['hd'].ravel().tolist() == [-1.0, 0.0, 1.0]  # 執白讓人、分先、執黑被讓


def test_hd_direction():
    assert sh.hd_direction(None, 'B') == 0.0 and sh.hd_direction(0, 'W') == 0.0
    assert sh.hd_direction(1, 'B') == 1.0 and sh.hd_direction(5, 'W') == -1.0


def test_load_se_blocks_opponent_side_and_fewer_symmetries(tmp_path):
    data, _ = toy_data(n_players=10, games=5)
    write_se(tmp_path / 'o.jsonl', data['key'], data['rank_idx'], n_sym=4, side='opponent')
    b = sh.load_se_blocks(str(tmp_path / 'o.jsonl'), data['key'], data['rank_idx'], side='opponent')
    assert sorted(b) == ['se1o', 'se4o', 'slo']  # 4 種對稱的檔案沒有 se8o，對手側不重複產生 hd
    with pytest.raises(ValueError, match='side'):
        sh.load_se_blocks(str(tmp_path / 'o.jsonl'), data['key'], data['rank_idx'])


def test_load_se_blocks_rejects_mixed_symmetry_counts(tmp_path):
    with open(tmp_path / 's.jsonl', 'w', encoding='utf-8') as f:
        for k, n_sym in (('a', 8), ('b', 4)):
            f.write(json.dumps({'key': k, 'rank_idx': 1, 'n': 5, 'score_mean': [0.0] * n_sym,
                                'score_std': [0.0] * n_sym, 'sl_rank': [0.1] * 11, 'hd': 0,
                                'color': 'B'}) + '\n')
    with pytest.raises(ValueError, match='對稱數'):
        sh.load_se_blocks(str(tmp_path / 's.jsonl'), np.array(['a', 'b']), np.array([1, 1]))


@pytest.mark.parametrize('row, match', [
    ({'key': 'x', 'rank_idx': 3, 'n': 5, 'score_mean': [0] * 8, 'score_std': [0] * 8, 'sl_rank': [0.1] * 11}, '缺少'),
    ({'key': 'a', 'rank_idx': 2, 'n': 5, 'score_mean': [0] * 8, 'score_std': [0] * 8, 'sl_rank': [0.1] * 11}, '等級'),
    ({'key': 'a', 'rank_idx': 3, 'n': 0, 'score_mean': None, 'score_std': None, 'sl_rank': None}, '可評的手'),
])
def test_load_se_blocks_rejects_bad_rows(tmp_path, row, match):
    (tmp_path / 's.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match=match):
        sh.load_se_blocks(str(tmp_path / 's.jsonl'), np.array(['a']), np.array([3]))


def test_parse_combo_and_inputs():
    blocks = {'hsl': np.ones((3, 2)), 'se8': np.zeros((3, 2))}
    assert sh.parse_combo('cnn+hsl+se8', set(blocks) | {'cnn'}) == ['cnn', 'hsl', 'se8']
    for bad in ('cnn+foo', 'hsl+hsl'):
        with pytest.raises(ValueError):
            sh.parse_combo(bad, set(blocks) | {'cnn'})
    probs = np.full((3, 10), 0.1)
    x, c = sh.combo_inputs(['cnn'], blocks, probs)
    assert x is None and c is probs
    x, c = sh.combo_inputs(['se8', 'hsl'], blocks, probs)
    assert c is None and x.shape == (3, 4) and x[:, :2].sum() == 0


def test_question_features_without_cnn_is_feature_mean():
    x = np.arange(12, dtype=float).reshape(6, 2)
    f = sh.question_features(x, None, [np.array([0, 1]), np.array([4, 5])], temp=3.0)
    assert f.tolist() == [[1.0, 2.0], [9.0, 10.0]]


def test_divide_prior_removes_label_frequency():
    labels = np.array([0, 0, 0, 1])
    out = sh.divide_prior(np.array([[0.75, 0.25] + [0.0] * 8]), labels)
    assert np.allclose(out[0, :2], [0.5, 0.5]) and np.isclose(out.sum(), 1.0)


def test_level_weighted_averages_levels_not_players():
    means = np.array([1.0, 1.0, 1.0, 0.0])
    assert sh.level_weighted(means, np.array([0, 0, 0, 1])) == 0.5


@pytest.mark.parametrize('prior', sh.PRIORS)
def test_predict_inputs_se_only_beats_chance(tmp_path, prior):
    data, _ = toy_data()
    write_se(tmp_path / 's.jsonl', data['key'], data['rank_idx'])
    blocks = sh.load_se_blocks(str(tmp_path / 's.jsonl'), data['key'], data['rank_idx'])
    splits = make_splits(data, (5, 20), 2, seed=42)
    points = []
    for train, questions, labels, _ in splits:
        p, _ = sh.predict_inputs(blocks['se8'], None, data, train, questions, np.random.default_rng(1), (5, 20), 2,
                                 temp=3.0, cs=(0.1, 1.0), inner_folds=3, prior=prior)
        assert np.allclose(p.sum(axis=1), 1.0)
        points.append(question_points(labels, pick_level(p)))
    assert np.concatenate(points).mean() > 0.5


def test_predict_stack_matches_predict_inputs_keep():
    data, probs = toy_data()
    train, questions, _, _ = make_splits(data, (5, 20), 2, seed=42)[0]
    a, c_a = sh.predict_stack(data, probs, train, questions, np.random.default_rng(1), (5, 20), 2,
                              temp=3.0, cs=(0.1, 1.0), inner_folds=3)
    b, c_b = sh.predict_inputs(data['x'][:, sh.hsl_columns(data)], probs, data, train, questions,
                               np.random.default_rng(1), (5, 20), 2, temp=3.0, cs=(0.1, 1.0), inner_folds=3)
    assert c_a == c_b and np.array_equal(a, b)


def test_main_combos_end_to_end(tmp_path, monkeypatch, capsys):
    data, probs = toy_data()
    write_npz(tmp_path / 'v.npz', data['key'], probs, data['rank_idx'])
    write_se(tmp_path / 's.jsonl', data['key'], data['rank_idx'])
    write_se(tmp_path / 'o.jsonl', data['key'], data['rank_idx'], n_sym=4, side='opponent')
    monkeypatch.setattr(sh, 'load_games', lambda path: data)
    monkeypatch.setattr(sys, 'argv', ['stack_humansl', 'unused.jsonl', '--cnn-probs', str(tmp_path / 'v.npz'),
                                      '--se', str(tmp_path / 's.jsonl'), '--se-opp', str(tmp_path / 'o.jsonl'),
                                      '--combos', 'cnn+hsl,cnn+hsl+se8,se8+sl,cnn+hsl+se4o+hd',
                                      '--prior', 'balanced', '--train-repeats', '2', '--test-repeats', '2',
                                      '--inner-folds', '3'])
    sh.main()
    out = capsys.readouterr().out
    assert 'prior balanced，基準 cnn+hsl' in out
    rows = [line for line in out.splitlines() if line.startswith('| cnn') or line.startswith('| se8')]
    assert len(rows) == 4
    assert '+0.0000' in rows[0]  # 基準減自己


@pytest.mark.parametrize('extra', [['--se', 's.jsonl'], ['--se-opp', 'o.jsonl'], ['--hsl-opp', 'h.jsonl'],
                                   ['--prior', 'balanced']])
def test_main_rejects_combo_only_options_without_combos(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['stack_humansl', 'unused.jsonl', '--cnn-probs', 'v.npz'] + extra)
    with pytest.raises(SystemExit):
        sh.main()


P3 = ['rank_11k', 'rank_1d', 'rank_6d']


def write_hsl(path, keys, rank_idx, side=None, seed=0, players=None):
    """analyze_humansl --no-eval 格式的 jsonl，每局 3 個 profile、4 手；side 為 None 時不寫 side 欄位。"""
    rng = np.random.default_rng(seed)
    players = ['p'] * len(keys) if players is None else players
    rows = []
    for k, y, p in zip(keys, rank_idx, players):
        r = {'key': str(k), 'rank_idx': int(y), 'player_id': str(p), 'hd': 0, 'color': 'B',
             'turns': [0, 2, 4, 6], 'eval_visits': 0, 'profiles': P3,
             'probs': rng.uniform(0.01, 0.9, (3, 4)).round(4).tolist()}
        if side:
            r['side'] = side
        rows.append(r)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_load_hsl_opp_aligns_by_key(tmp_path):
    keys, rank = np.array(['a', 'b', 'c']), np.array([0, 5, 9])
    write_hsl(tmp_path / 't.jsonl', keys, rank)
    write_hsl(tmp_path / 'o.jsonl', keys[::-1], rank[::-1], side='opponent')
    data = sh.load_games(str(tmp_path / 't.jsonl'))
    opp = sh.load_games(str(tmp_path / 'o.jsonl'))
    x = sh.load_hsl_opp(str(tmp_path / 'o.jsonl'), data)
    assert x.shape == (3, len(sh.hsl_columns(data)))
    assert np.allclose(x, opp['x'][::-1][:, sh.hsl_columns(opp)])


@pytest.mark.parametrize('side, keys, rank, match', [
    (None, ['a', 'b'], [0, 5], 'side'),
    ('target', ['a', 'b'], [0, 5], 'side'),
    ('opponent', ['a'], [0], '缺少'),
    ('opponent', ['a', 'b'], [0, 6], '等級'),
])
def test_load_hsl_opp_rejects_bad_file(tmp_path, side, keys, rank, match):
    write_hsl(tmp_path / 't.jsonl', ['a', 'b'], [0, 5])
    write_hsl(tmp_path / 'o.jsonl', keys, rank, side=side)
    with pytest.raises(ValueError, match=match):
        sh.load_hsl_opp(str(tmp_path / 'o.jsonl'), sh.load_games(str(tmp_path / 't.jsonl')))


def test_load_hsl_opp_allow_missing_fills_nan(tmp_path):
    write_hsl(tmp_path / 't.jsonl', ['a', 'b', 'c'], [-1, -1, -1])
    write_hsl(tmp_path / 'o.jsonl', ['c', 'a'], [-1, -1], side='opponent')
    data = sh.load_games(str(tmp_path / 't.jsonl'))
    opp = sh.load_games(str(tmp_path / 'o.jsonl'))
    x = sh.load_hsl_opp(str(tmp_path / 'o.jsonl'), data, allow_missing=True)
    cols = sh.hsl_columns(opp)
    assert np.allclose(x[0], opp['x'][1, cols]) and np.allclose(x[2], opp['x'][0, cols])
    assert np.isnan(x[1]).all()


def test_load_hsl_opp_rejects_different_profiles(tmp_path):
    write_hsl(tmp_path / 't.jsonl', ['a'], [0])
    write_hsl(tmp_path / 'o.jsonl', ['a'], [0], side='opponent')
    rows = [json.loads(line) for line in (tmp_path / 'o.jsonl').read_text(encoding='utf-8').splitlines()]
    rows[0]['profiles'] = ['rank_11k', 'rank_1d', 'rank_9d']
    (tmp_path / 'o.jsonl').write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='profiles'):
        sh.load_hsl_opp(str(tmp_path / 'o.jsonl'), sh.load_games(str(tmp_path / 't.jsonl')))


def test_main_combos_with_hsl_opp(tmp_path, monkeypatch, capsys):
    data, probs = toy_data()
    write_hsl(tmp_path / 't.jsonl', data['key'], data['rank_idx'], players=data['player_id'])
    write_hsl(tmp_path / 'o.jsonl', data['key'], data['rank_idx'], side='opponent', seed=1, players=data['player_id'])
    write_npz(tmp_path / 'v.npz', data['key'], probs, data['rank_idx'])
    monkeypatch.setattr(sys, 'argv', ['stack_humansl', str(tmp_path / 't.jsonl'),
                                      '--cnn-probs', str(tmp_path / 'v.npz'), '--hsl-opp', str(tmp_path / 'o.jsonl'), '--combos', 'cnn+hsl,cnn+hsl+hslo',
                                      '--train-repeats', '2', '--test-repeats', '2', '--inner-folds', '3'])
    sh.main()
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith('| cnn')]
    assert len(rows) == 2 and rows[1].startswith('| cnn+hsl+hslo |')


def with_extra_players(data, probs, n_players=10, games=7):
    """data 後面接上 n_players 位局數較少的玩家，回傳 (合併的 data, 合併的 probs, 原本玩家的 player_keys set)。"""
    extra, extra_probs = toy_data(n_players=n_players, games=games, seed=5)
    merged = dict(data)
    for k, prefix in (('player_id', 'x'), ('key', 'xk')):
        merged[k] = np.concatenate([data[k], np.array([prefix + v for v in extra[k]])])
    for k in ('x', 'rank_idx'):
        merged[k] = np.concatenate([data[k], extra[k]])
    return merged, np.vstack([probs, extra_probs]), set(sh.player_keys(data))


def test_make_eval_splits_without_list_matches_make_splits():
    data, _ = toy_data()
    for a, b in zip(make_splits(data, (5, 20), 2, seed=7), sh.make_eval_splits(data, (5, 20), 2, seed=7)):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
        assert all(np.array_equal(p, q) for p, q in zip(a[1], b[1]))


def test_make_eval_splits_extra_players_only_train():
    data, probs = toy_data()
    merged, _, eval_players = with_extra_players(data, probs)
    base = make_splits(data, (5, 20), 2, seed=7)
    out = sh.make_eval_splits(merged, (5, 20), 2, eval_players, seed=7)
    n = len(data['key'])
    for (tr_a, q_a, y_a, p_a), (tr_b, q_b, y_b, p_b) in zip(base, out):
        assert all(np.array_equal(merged['key'][q], data['key'][r]) for q, r in zip(q_b, q_a))  # 測試題相同
        assert np.array_equal(y_a, y_b) and np.array_equal(p_a, p_b)
        assert np.array_equal(tr_b[:n], tr_a) and tr_b[n:].all()  # 額外玩家每一折都只當訓練


def test_make_eval_splits_with_all_players_matches_make_splits():
    data, _ = toy_data()
    full = sh.make_eval_splits(data, (5, 20), 2, set(sh.player_keys(data)), seed=7)
    for a, b in zip(make_splits(data, (5, 20), 2, seed=7), full):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
        assert all(np.array_equal(p, q) for p, q in zip(a[1], b[1]))


def test_make_eval_splits_rejects_unknown_player():
    data, _ = toy_data()
    with pytest.raises(ValueError, match='不在資料內'):
        sh.make_eval_splits(data, (5, 20), 2, {'3|nobody'}, seed=7)


def test_load_eval_players(tmp_path):
    (tmp_path / 'p.csv').write_text('player_id,rank\nabc,1D\n007,D\n', encoding='utf-8')
    assert sh.load_eval_players(str(tmp_path / 'p.csv')) == {'4|abc', '0|007'}


def test_main_eval_players_end_to_end(tmp_path, monkeypatch, capsys):
    data, probs = toy_data()
    merged, merged_probs, eval_players = with_extra_players(data, probs)
    write_hsl(tmp_path / 't.jsonl', merged['key'], merged['rank_idx'], players=merged['player_id'])
    write_npz(tmp_path / 'v.npz', merged['key'], merged_probs, merged['rank_idx'])
    with open(tmp_path / 'p.csv', 'w', encoding='utf-8') as f:
        f.write('player_id,rank\n' + ''.join('%s,%s\n' % (k.split('|')[1], sh.LEVELS[int(k.split('|')[0])])
                                             for k in sorted(eval_players)))
    monkeypatch.setattr(sys, 'argv', ['stack_humansl', str(tmp_path / 't.jsonl'), '--cnn-probs', str(tmp_path / 'v.npz'),
                                      '--eval-players', str(tmp_path / 'p.csv'), '--combos', 'cnn+hsl',
                                      '--train-repeats', '2', '--test-repeats', '2', '--inner-folds', '3'])
    sh.main()
    out = capsys.readouterr().out
    assert '50 位玩家出測試題，10 位只當訓練' in out
