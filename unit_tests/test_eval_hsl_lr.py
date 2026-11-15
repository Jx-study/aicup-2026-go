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
