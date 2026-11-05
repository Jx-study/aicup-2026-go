import sys

import numpy as np
import pandas as pd
import pytest

from aicup import predict_stack as ps
from aicup import stack_humansl as sh
from aicup.fit_humansl import level_probs, question_points
from aicup.predict_rank import file_sha256
from aicup.scoring import LEVELS, pick_level
from stack_helpers import toy_data, write_npz, write_rank_tests

SGF = '(;SZ[19]HD[0];B[pd];W[dp])'


def player_questions(data, n=10):
    """每位玩家取前 n 局當一題，回傳 (questions, labels)。"""
    questions = [np.flatnonzero(data['player_id'] == p)[:n] for p in np.unique(data['player_id'])]
    return questions, np.array([data['rank_idx'][q[0]] for q in questions])


def test_fit_stack_matches_stack_humansl_with_all_players():
    data, probs = toy_data()
    questions, _ = player_questions(data)
    model, c = ps.fit_stack(data, probs, (5, 10), 2, 3.0, 7, inner_folds=3)
    x = data['x'][:, sh.hsl_columns(data)]
    ours = level_probs(model, sh.question_features(x, probs, questions, 3.0))
    theirs, c_theirs = sh.predict_stack(data, probs, np.ones(len(x), bool), questions, np.random.default_rng(7),
                                        (5, 10), 2, 3.0, inner_folds=3)
    assert c == c_theirs and np.allclose(ours, theirs)


def test_fit_stack_balanced_matches_stack_humansl_prior_balanced():
    data, probs = toy_data()
    questions, _ = player_questions(data)
    model, c = ps.fit_stack(data, probs, (5, 10), 2, 3.0, 7, inner_folds=3, prior='balanced')
    x = data['x'][:, sh.hsl_columns(data)]
    ours = level_probs(model, sh.question_features(x, probs, questions, 3.0))
    theirs, c_theirs = sh.predict_inputs(x, probs, data, np.ones(len(x), bool), questions, np.random.default_rng(7),
                                         (5, 10), 2, 3.0, inner_folds=3, prior='balanced')
    assert c == c_theirs and np.allclose(ours, theirs)
    assert model[-1].class_weight == 'balanced'


def test_fit_stack_learns_toy_levels():
    data, probs = toy_data()
    questions, labels = player_questions(data)
    model, _ = ps.fit_stack(data, probs, (5, 10), 2, 3.0, 0, inner_folds=3)
    x = data['x'][:, sh.hsl_columns(data)]
    preds = pick_level(level_probs(model, sh.question_features(x, probs, questions, 3.0)))
    assert question_points(labels, preds).mean() > 0.5


def test_fit_stack_with_extra_features_matches_stack_humansl_combo():
    data, probs = toy_data()
    questions, _ = player_questions(data)
    opp = np.random.default_rng(3).normal(size=(len(data['x']), 2))
    x = np.hstack([data['x'][:, sh.hsl_columns(data)], opp])
    model, c = ps.fit_stack(data, probs, (5, 10), 2, 3.0, 7, inner_folds=3, x=x)
    ours = level_probs(model, sh.question_features(x, probs, questions, 3.0))
    theirs, c_theirs = sh.predict_inputs(x, probs, data, np.ones(len(x), bool), questions, np.random.default_rng(7),
                                         (5, 10), 2, 3.0, inner_folds=3)
    assert c == c_theirs and np.allclose(ours, theirs)


def jsonl_data(keys, n_profiles=3, profiles=None):
    x = np.arange(len(keys) * 4, dtype=np.float64).reshape(len(keys), 4)
    return {'x': x, 'key': np.array(keys), 'n_profiles': n_profiles, 'profiles': profiles, 'has_loss': False}


def test_check_profiles_compares_names_then_counts():
    a = jsonl_data([], profiles=('rank_11k', 'rank_1d', 'rank_6d'))
    ps.check_profiles(a, jsonl_data([], profiles=('rank_11k', 'rank_1d', 'rank_6d')))
    with pytest.raises(ValueError, match='profile'):
        ps.check_profiles(a, jsonl_data([], profiles=('rank_11k', 'rank_6d', 'rank_1d')))
    ps.check_profiles(a, jsonl_data([]))  # 測試端沒有名稱時只比對數量
    with pytest.raises(ValueError, match='profile'):
        ps.check_profiles(a, jsonl_data([], n_profiles=10))


def test_align_test_orders_by_key_and_marks_unusable_games():
    td = jsonl_data(['b', 'a'])
    probs = np.full((3, 10), 0.1)
    probs[2, 0] = np.nan
    x, valid = ps.align_test(td, ['a', 'b', 'c'], probs)
    assert np.array_equal(x[:2], td['x'][[1, 0]]) and np.isnan(x[2]).all()
    assert valid.tolist() == [True, True, False]
    probs[2, 0] = 0.1
    probs[0, 0] = np.nan
    _, valid = ps.align_test(td, ['a', 'b', 'c'], probs)
    assert valid.tolist() == [False, True, False]  # a 的 CNN 是 NaN，c 沒有 Human SL


def test_align_test_with_features_marks_nan_rows_unusable():
    td = jsonl_data(['b', 'a'])
    x = np.array([[1.0, np.nan], [2.0, 3.0]])  # b 缺對手側
    out, valid = ps.align_test(td, ['a', 'b'], np.full((2, 10), 0.1), x)
    assert np.array_equal(out[0], [2.0, 3.0]) and valid.tolist() == [True, False]


def test_read_test_probs_checks_order_tta_and_checkpoint(tmp_path):
    ckpt = tmp_path / 'best_model.pth'
    ckpt.write_bytes(b'weights')
    p = tmp_path / 'g.npz'
    np.savez(p, probs=np.full((2, 10), 0.1), game_q=np.array([0, 1]), tta=False, ckpt_sha256=file_sha256(str(ckpt)))
    probs, sha = ps.read_test_probs(str(p), np.array([0, 1]), str(ckpt))
    assert probs.shape == (2, 10) and sha == file_sha256(str(ckpt))
    with pytest.raises(ValueError, match='對不上'):
        ps.read_test_probs(str(p), np.array([0, 0]))
    other = tmp_path / 'other.pth'
    other.write_bytes(b'other')
    with pytest.raises(ValueError, match='不是'):
        ps.read_test_probs(str(p), np.array([0, 1]), str(other))
    np.savez(p, probs=np.full((2, 10), 0.1), game_q=np.array([0, 1]), tta=True)
    with pytest.raises(ValueError, match='TTA'):
        ps.read_test_probs(str(p), np.array([0, 1]))


def test_predict_question_without_usable_games_gets_weakest_level():
    data, probs = toy_data()
    model, _ = ps.fit_stack(data, probs, (5, 10), 2, 3.0, 0, inner_folds=3)
    x = data['x'][:, sh.hsl_columns(data)]
    rows = [180, 181, 182, 0]  # 前 3 局是等級 9 的玩家
    valid = np.array([True, True, True, False])
    preds, empty = ps.predict(model, x[rows], probs[rows], np.array([0, 0, 0, 1]), valid, 2, 3.0)
    assert empty == 1 and preds[1] == 0 and preds[0] >= 8


def test_main_writes_submission(tmp_path, monkeypatch, capsys):
    data, probs = toy_data()
    write_npz(tmp_path / 'v.npz', data['key'], probs, data['rank_idx'])
    # public 2 題、private 2 題；private 第 2 題唯一的局沒有 Human SL 結果
    write_rank_tests(tmp_path / 't', {'public': [([SGF] * 3, 'B'), ([SGF] * 3, 'W')],
                                      'private': [([SGF] * 3, 'B'), ([SGF], 'W')]})
    rows = [60, 61, 62, 140, 141, 142, 100, 101, 102]  # 等級 3、7、5 的玩家
    keys = ['pub_q_0001_g01', 'pub_q_0001_g02', 'pub_q_0001_g03', 'pub_q_0002_g01', 'pub_q_0002_g02',
            'pub_q_0002_g03', 'priv_q_0001_g01', 'priv_q_0001_g02', 'priv_q_0001_g03']
    test = {'x': data['x'][rows], 'key': np.array(keys), 'n_profiles': 3, 'profiles': None, 'has_loss': False}
    np.savez(tmp_path / 'g.npz', probs=probs[rows + [0]], game_q=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3]), tta=False)
    monkeypatch.setattr(ps, 'load_games', lambda path: data if path == 'val.jsonl' else test)
    out = tmp_path / 'sub.csv'
    monkeypatch.setattr(sys, 'argv', ['predict_stack', '--val-jsonl', 'val.jsonl', '--val-probs', str(tmp_path / 'v.npz'),
                                      '--test-jsonl', 'test.jsonl', '--test-probs', str(tmp_path / 'g.npz'),
                                      '--test-dir', str(tmp_path / 't'), '--out', str(out),
                                      '--sizes', '5,10', '--train-repeats', '2', '--inner-folds', '3'])
    ps.main()
    sub = pd.read_csv(out)
    assert sub['question_id'].tolist() == ['pub_q_0001', 'pub_q_0002', 'priv_q_0001', 'priv_q_0002']
    preds = [LEVELS.index(r) for r in sub['pred_rank']]
    assert all(abs(p - t) <= 1 for p, t in zip(preds[:3], (3, 7, 5)))
    assert preds[3] == 0
    out_text = capsys.readouterr().out
    assert '選到 C=' in out_text and '1 題沒有可用的局' in out_text


def test_main_with_hsl_opp_uses_opponent_features(tmp_path, monkeypatch, capsys):
    data, probs = toy_data()
    write_npz(tmp_path / 'v.npz', data['key'], probs, data['rank_idx'])
    write_rank_tests(tmp_path / 't', {'public': [([SGF] * 3, 'B')], 'private': [([SGF], 'W')]})
    rows = [60, 61, 62, 100]
    keys = ['pub_q_0001_g01', 'pub_q_0001_g02', 'pub_q_0001_g03', 'priv_q_0001_g01']
    test = {'x': data['x'][rows], 'key': np.array(keys), 'n_profiles': 3, 'profiles': None, 'has_loss': False}
    np.savez(tmp_path / 'g.npz', probs=probs[rows], game_q=np.array([0, 0, 0, 1]), tta=False)
    monkeypatch.setattr(ps, 'load_games', lambda path: data if path == 'val.jsonl' else test)
    calls = []

    def fake_opp(path, d, allow_missing=False):
        calls.append((path, allow_missing))
        x = d['x'][:, :2].copy()
        if path == 'test-opp.jsonl':
            x[3] = np.nan  # private 唯一的局缺對手側
        return x

    monkeypatch.setattr(ps, 'load_hsl_opp', fake_opp)
    out = tmp_path / 'sub.csv'
    monkeypatch.setattr(sys, 'argv', ['predict_stack', '--val-jsonl', 'val.jsonl', '--val-probs', str(tmp_path / 'v.npz'),
                                      '--test-jsonl', 'test.jsonl', '--test-probs', str(tmp_path / 'g.npz'),
                                      '--test-dir', str(tmp_path / 't'), '--out', str(out),
                                      '--val-hsl-opp', 'val-opp.jsonl', '--test-hsl-opp', 'test-opp.jsonl',
                                      '--sizes', '5,10', '--train-repeats', '2', '--inner-folds', '3'])
    ps.main()
    assert calls == [('val-opp.jsonl', False), ('test-opp.jsonl', True)]
    preds = [LEVELS.index(r) for r in pd.read_csv(out)['pred_rank']]
    assert abs(preds[0] - 3) <= 1 and preds[1] == 0
    text = capsys.readouterr().out
    assert '特徵：cnn+hsl+hslo' in text and '1 題沒有可用的局' in text


@pytest.mark.parametrize('extra', [['--val-hsl-opp', 'v.jsonl'], ['--test-hsl-opp', 't.jsonl']])
def test_main_requires_both_hsl_opp_files(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['predict_stack', '--val-jsonl', 'v', '--val-probs', 'v', '--test-jsonl', 't',
                                      '--test-probs', 't'] + extra)
    with pytest.raises(SystemExit):
        ps.main()
