import json

import numpy as np
import pytest

from aicup import fit_humansl as fh
from aicup.scoring import NEAR_SCORE


def row(key, player, rank, turns=120, loss=1.0, n_profiles=10):
    # 真實等級的 profile 機率最高，手數足以涵蓋兩個 cut；loss 為 None 時模擬不跑損失 query 的輸出
    probs = [[0.4 if i == rank else 0.1] * turns for i in range(n_profiles)]
    r = {'key': key, 'player_id': player, 'rank_idx': rank, 'turns': list(range(turns)), 'probs': probs}
    if loss is not None:
        r.update(score_loss=[loss] * turns, winrate_loss=[loss / 10] * turns)
    return r


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def make_data(players_per_rank=6, games=20, ranks=(0, 4, 9), seed=0):
    rng = np.random.default_rng(seed)
    x, rank_idx, player_id, keys = [], [], [], []
    for r in ranks:
        for p in range(players_per_rank):
            for g in range(games):
                v = rng.normal(0, 0.3, 11 + 2 * fh.N_STATS)
                v[r] += 2.0
                x.append(v)
                rank_idx.append(r)
                player_id.append('p%d_%d' % (r, p))
                keys.append('%d_%d_%d' % (r, p, g))
    return {'x': np.array(x), 'rank_idx': np.array(rank_idx, np.int64),
            'player_id': np.array(player_id), 'key': np.array(keys), 'n_profiles': 10, 'has_loss': True}


def test_game_features_values():
    r = row('k', 'p', 2, turns=120, loss=2.0)
    r['score_loss'] = [1.0] * 50 + [9.0] * 70
    v = fh.game_features(r)
    assert v.shape == (11 + 2 * fh.N_STATS,)
    assert v[2] == pytest.approx(np.log(0.4)) and v[0] == pytest.approx(np.log(0.1))
    assert v[10] == pytest.approx(np.log(120))
    assert v[11] == pytest.approx(1.0)                      # cut 50 的目數損失平均
    assert v[17] == pytest.approx((50 + 9 * 50) / 100)      # cut 100 的目數損失平均


def test_game_features_short_game_uses_available_moves():
    v = fh.game_features(row('k', 'p', 2, turns=30, loss=3.0))
    assert v[11] == pytest.approx(3.0) and v[17] == pytest.approx(3.0)


def test_feature_sets_columns():
    sets = fh.feature_sets()
    assert sets['A'] == list(range(11))
    assert sets['B50'] == list(range(17))
    assert sets['B100'] == list(range(11)) + list(range(17, 23))


def test_load_games_shapes_and_empty(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_rows(p, [row('D_1', 'p1', 0), row('D_2', 'p1', 0)])
    d = fh.load_games(str(p))
    assert d['x'].shape == (2, 23) and d['key'].tolist() == ['D_1', 'D_2']
    p.write_text('', encoding='utf-8')
    with pytest.raises(ValueError):
        fh.load_games(str(p))


def test_question_points():
    s = fh.question_points([3, 3, 3], [3, 4, 9])
    assert s == pytest.approx([1.0, NEAR_SCORE, 0.0])


def test_game_features_skips_nan_moves_in_average():
    r = row('k', 'p', 2, turns=120)
    r['probs'][0][0] = float('nan')
    assert fh.game_features(r)[0] == pytest.approx(np.log(0.1))


def test_player_folds_keeps_player_in_one_fold_and_stratifies():
    d = make_data()
    fold = fh.player_folds(d, 3, seed=42)
    for p in np.unique(d['player_id']):
        assert len(set(fold[d['player_id'] == p])) == 1
    for r in np.unique(d['rank_idx']):
        players = {p: f for p, f, x in zip(d['player_id'], fold, d['rank_idx']) if x == r}
        assert sorted(np.bincount(list(players.values()), minlength=3)) == [2, 2, 2]


def test_player_folds_separates_same_id_across_ranks():
    d = make_data(players_per_rank=3, ranks=(0, 4))
    d['player_id'] = np.char.replace(np.char.replace(d['player_id'], 'p0_', 'x'), 'p4_', 'x')
    fh.player_folds(d, 3, seed=42)
    assert len(set(fh.player_keys(d))) == 6


def test_questions_of_stay_inside_mask_and_report_true_player():
    d = make_data()
    mask = d['rank_idx'] == 4
    qs, labels, players = fh.questions_of(d, mask, 2, np.random.default_rng(0), (5, 20))
    assert all(mask[q].all() for q in qs)
    assert set(labels) == {4} and len(players) == len(qs)
    assert all(len(set(d['player_id'][q])) == 1 for q in qs)
    keys = fh.player_keys(d)
    assert [keys[q[0]] for q in qs] == players.tolist()


def test_splits_never_train_on_test_players():
    d = make_data()
    keys = fh.player_keys(d)
    for train, q, y, p in fh.make_splits(d, (5, 20), 1, folds=3):
        assert not set(keys[train]) & set(p)
        assert not set(np.concatenate(q)) & set(np.flatnonzero(train))


def test_splits_keep_all_players_and_labels_match_players():
    d = make_data()
    splits = fh.make_splits(d, (5, 20), 2, folds=3)
    players = np.concatenate([s[3] for s in splits])
    labels = np.concatenate([s[2] for s in splits])
    assert len(set(players)) == 18
    assert all(p.split('|')[0] == str(y) for p, y in zip(players, labels))


def test_splits_are_independent_of_predict_rng():
    d = make_data()
    a = fh.make_splits(d, (5, 20), 2, folds=3)
    b = fh.make_splits(d, (5, 20), 2, folds=3)
    assert all(np.array_equal(x, y) for s, t in zip(a, b) for x, y in zip(s[1], t[1]))


@pytest.mark.parametrize('mode', ['avg', 'game'])
def test_learned_model_separates_ranks(mode):
    d = make_data()
    cols = fh.feature_sets()['A']
    if mode == 'avg':
        predict = lambda tr, q, rng: fh.predict_avg(d, cols, 1.0, tr, q, rng, (5, 20), 3)
    else:
        predict = lambda tr, q, rng: fh.predict_game(d, cols, 1.0, 2.0, tr, q)
    splits = fh.make_splits(d, (5, 20), 2, folds=3)
    probs = fh.cross_validate(splits, predict)
    labels = np.concatenate([s[2] for s in splits])
    assert probs.shape[1] == 10
    assert (probs.argmax(1) == labels).mean() > 0.95


def test_level_probs_columns_are_level_indices():
    d = make_data()
    train = d['rank_idx'] != 9
    model = fh.new_model(1.0).fit(d['x'][train][:, :11], d['rank_idx'][train])
    probs = fh.level_probs(model, d['x'][:, :11])
    assert probs.shape == (len(d['x']), 10) and (probs[:, 9] == 0).all()
    mask = d['rank_idx'] == 4
    assert (probs[mask].argmax(1) == 4).mean() > 0.95


def test_evaluate_grid_sorted_and_aligned_with_questions():
    d = make_data()
    splits = fh.make_splits(d, (5, 20), 1, folds=3)
    results = fh.evaluate_grid(d, splits, (5, 20), 1, 42)
    assert len(results) == len(fh.configs(d, (5, 20), 1))
    n = sum(len(s[1]) for s in splits)
    assert all(len(r[1]) == n and len(r[2]) == n for r in results)
    players = np.concatenate([s[3] for s in splits])
    means = [np.mean(list(fh.player_means(r[1], players).values())) for r in results]
    assert means == sorted(means, reverse=True)


def test_bootstrap_mean_interval_contains_mean():
    mean, lo, hi = fh.bootstrap_mean([0.2, 0.4, 0.6, 0.8])
    assert mean == pytest.approx(0.5) and lo <= mean <= hi and lo < hi


def test_cnn_question_points_uses_same_questions_and_marks_coverage(tmp_path):
    val_keys = np.array(['a', 'b', 'c', 'd'])
    probs = np.full((4, 10), 0.01)
    probs[:, 3] = 0.91
    p = tmp_path / 'val_probs.npz'
    np.savez(p, key=val_keys, probs=probs)
    keys = np.array(['a', 'b', 'c', 'x'])
    questions = [np.array([0, 1]), np.array([2, 3])]
    covered, points = fh.cnn_question_points(str(p), keys, questions, np.array([3, 3]), 1.0)
    assert covered.tolist() == [True, False] and points == pytest.approx([1.0])


def test_cnn_question_points_empty_when_no_overlap(tmp_path):
    p = tmp_path / 'val_probs.npz'
    np.savez(p, key=np.array(['a']), probs=np.full((1, 10), 0.1))
    covered, points = fh.cnn_question_points(str(p), np.array(['z']), [np.array([0])], np.array([3]), 1.0)
    assert not covered.any() and len(points) == 0


def test_game_features_without_loss_and_three_profiles():
    v = fh.game_features(row('k', 'p', 1, turns=80, loss=None, n_profiles=3))
    assert v.shape == (4,)
    assert v[1] == pytest.approx(np.log(0.4)) and v[3] == pytest.approx(np.log(80))


def test_load_games_reports_profiles_and_loss(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_rows(p, [row('D_1', 'p1', 0, loss=None, n_profiles=3), row('D_2', 'p1', 0, loss=None, n_profiles=3)])
    d = fh.load_games(str(p))
    assert d['x'].shape == (2, 4) and d['n_profiles'] == 3 and not d['has_loss']
    assert fh.feature_sets(d['n_profiles'], d['has_loss']) == {'A': [0, 1, 2, 3]}


def test_load_games_rejects_mixed_rows(tmp_path):
    p = tmp_path / 'a.jsonl'
    write_rows(p, [row('D_1', 'p1', 0), row('D_2', 'p1', 0, loss=None)])
    with pytest.raises(ValueError, match='欄位'):
        fh.load_games(str(p))
    write_rows(p, [row('D_1', 'p1', 0), row('D_2', 'p1', 0, n_profiles=3)])
    with pytest.raises(ValueError, match='profile'):
        fh.load_games(str(p))


def test_load_games_reads_profile_names(tmp_path):
    p = tmp_path / 'a.jsonl'
    names = ['rank_11k', 'rank_1d', 'rank_6d']
    rows = [dict(row('D_%d' % i, 'p1', 0, loss=None, n_profiles=3), profiles=names) for i in range(2)]
    write_rows(p, rows)
    assert fh.load_games(str(p))['profiles'] == tuple(names)
    rows[1]['profiles'] = ['rank_11k', 'rank_6d', 'rank_1d']
    write_rows(p, rows)
    with pytest.raises(ValueError, match='名稱'):
        fh.load_games(str(p))
    write_rows(p, [row('D_1', 'p1', 0)])
    assert fh.load_games(str(p))['profiles'] is None


def test_feature_sets_three_profiles_with_loss():
    sets = fh.feature_sets(3, True)
    assert sets['A'] == [0, 1, 2, 3]
    assert sets['B50'] == list(range(10)) and sets['B100'] == [0, 1, 2, 3] + list(range(10, 16))


def test_evaluate_grid_only_one_config():
    d = make_data()
    splits = fh.make_splits(d, (5, 20), 1, folds=3)
    results = fh.evaluate_grid(d, splits, (5, 20), 1, 42, only='avg A C=1')
    assert [r[0] for r in results] == ['avg A C=1']
    with pytest.raises(ValueError, match='avg A C=0.1'):
        fh.evaluate_grid(d, splits, (5, 20), 1, 42, only='nope')


def test_cnn_question_points_rejects_label_mismatch(tmp_path):
    p = tmp_path / 'val_probs.npz'
    np.savez(p, key=np.array(['a', 'b']), probs=np.full((2, 10), 0.1), rank_idx=np.array([3, 4]))
    with pytest.raises(ValueError, match='等級'):
        fh.cnn_question_points(str(p), np.array(['a', 'b']), [np.array([0, 1])], np.array([3]), 1.0)
