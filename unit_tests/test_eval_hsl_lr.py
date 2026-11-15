import numpy as np
import pytest

from aicup import eval_hsl_lr


def test_parse_arm():
    assert eval_hsl_lr.parse_arm('a=hsl,hslo@p1,p2') == ('a', ('hsl', 'hslo'), ('p1', 'p2'))
    assert eval_hsl_lr.parse_arm('b=hsl@') == ('b', ('hsl',), ())


def write_players(path, ids):
    path.write_text('player_id\n' + ''.join('%s\n' % i for i in ids), encoding='utf-8')


def test_main_rejects_overlapping_pools_and_excluded_players(tmp_path, monkeypatch):
    caches = {'a.npz': ['p1', 'p2'], 'b.npz': ['p2', 'p3'], 'c.npz': ['p4'], 'dev.npz': ['d1']}
    monkeypatch.setattr(eval_hsl_lr.feature_cache, 'load', lambda path: ({'player_id': np.array(caches[path])}, {}))
    exclude = tmp_path / 'ex.csv'
    write_players(exclude, ['p4'])

    def run(*extra):
        argv = ['eval_hsl_lr', '--dev', 'dev.npz', '--eval-players', 'x', '--out', str(tmp_path / 'o.npz'),
                '--arm', 'm=hsl@'] + list(extra)
        monkeypatch.setattr('sys.argv', argv)
        eval_hsl_lr.main()

    with pytest.raises(ValueError, match='p2'):
        run('--pool', 'a=a.npz', '--pool', 'b=b.npz')
    with pytest.raises(ValueError, match='p4'):
        run('--pool', 'c=c.npz', '--exclude-players', str(exclude))


def load_toy(tmp_path, **kw):
    from test_stack_hgb import toy_files
    from aicup.stack_hgb import load_game_features
    data, games = load_game_features(*toy_files(tmp_path, **kw))
    data = dict(data, feature_profiles=data['profiles'])
    # 偶數局改成白棋，每五局有一局 HD 為 2
    n = len(data['key'])
    data['black'] = np.arange(n) % 2 == 1
    data['hd'] = np.where(np.arange(n) % 5 == 0, 2, 0)
    return data, games


def toy_splits(dev, sizes):
    from aicup.stack_humansl import make_eval_splits
    return make_eval_splits(dev[0], sizes, 2, None, folds=2, seed=3)


@pytest.mark.parametrize('test_type', ['sc', 'mx'])
@pytest.mark.parametrize('repeats', [1, 3])
def test_retest_splits_respects_repeats_and_question_type(tmp_path, test_type, repeats):
    dev = load_toy(tmp_path, name='dev', n_players=20, games=20)
    sizes = (2, 5)
    splits = toy_splits(dev, sizes)
    new = eval_hsl_lr.retest_splits(dev[0], splits, 3, sizes, test_type, repeats)
    data = dev[0]
    for (train, q, y, p), (train0, _, _, _) in zip(new, splits):
        assert np.array_equal(train, train0)
        assert len(q) == len(set(p)) * repeats  # 每位玩家每輪一題
        for g in q:
            assert (data['hd'][g] < 2).all() and not train[g].any()
            assert len(g) in sizes
            if test_type == 'sc':
                assert len(set(data['black'][g])) == 1
    assert eval_hsl_lr.retest_splits(data, splits, 3, sizes, 'default', repeats) is splits


def test_evaluate_runs_color_and_plain_arms(tmp_path):
    dev = load_toy(tmp_path, name='dev', n_players=20, games=20)
    pool = load_toy(tmp_path, name='pool', n_players=20, games=20, prefix='q', seed=7)
    sizes = (2, 5)
    splits = eval_hsl_lr.retest_splits(dev[0], toy_splits(dev, sizes), 3, sizes, 'sc', 2)
    arms = [('A', ('hsl', 'hslo'), ('pl',)), ('E', ('hsl', 'hslo', 'co'), ('pl',))]
    preds, params = eval_hsl_lr.evaluate(dev, {'pl': pool}, arms, splits, 3, sizes, 2, cs=(0.1, 1.0), inner_folds=2,
                                         log=lambda *_: None)
    n = sum(len(s[1]) for s in splits)
    assert preds['A'].shape == preds['E'].shape == (n,) and set(params) == {'A', 'E'}
    only_e = eval_hsl_lr.evaluate(dev, {'pl': pool}, arms[1:], splits, 3, sizes, 2, cs=(0.1, 1.0), inner_folds=2,
                                  log=lambda *_: None)[0]
    assert np.array_equal(only_e['E'], preds['E'])  # 方案之間互不影響
