import zipfile

from utils import tf_bundle


def test_write_zip_contains_tf_run(tmp_path):
    for d, names in (('d1', ['a_moves.npz', 'val648_moves.npz', 'test_moves.npz']),
                     ('d2', ['b_moves.npz', 'val648_moves.npz', 'test_moves.npz'])):
        (tmp_path / d).mkdir()
        for n in names:
            (tmp_path / d / n).write_bytes(b'x')
    out = tmp_path / 'bundle.zip'
    tf_bundle.write_zip(str(out), str(tmp_path / 'd1') + '/', ['a_moves.npz'], True, str(tmp_path / 'd2') + '/', ['b_moves.npz'])
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert 'tf_run.py' in names
        assert {'data/a_moves.npz', 'data2/b_moves.npz', 'data/test_moves.npz', 'data2/val648_moves.npz'} <= names
        assert len(names) == 7
        with open('aicup/tf_run.py', 'rb') as f:
            assert z.read('tf_run.py') == f.read()
