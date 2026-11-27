import contextlib
import io
import json

import numpy as np

from aicup.pack_moves import build_jobs, parse_move_features


def _row(T, color, hd):
    return {
        'turns': list(range(T)),
        'probs': [[0.5] * T, [0.25] * T, [0.1] * T],
        'pmax': [[0.9] * T, [0.8] * T, [0.7] * T],
        'entropy': [[1.0] * T, [2.0] * T, [3.0] * T],
        'color': color,
        'hd': hd,
    }


def test_parse_move_features_shape_and_meta():
    for T, color, hd in ((120, 'B', 1), (60, 'W', 0)):
        feats, mask = parse_move_features(_row(T, color, hd), _row(T, 'W', 0))
        assert feats.shape == (200, 14)
        assert mask.shape == (200,) and mask.sum() == T
        assert (feats[:T, 11] == float(color == 'B')).all()
        assert (feats[:T, 12] == float(hd == 1)).all()
        assert (feats[T:] == 0).all()
        np.testing.assert_allclose(feats[0, 13], 1 / np.log(T + 2), rtol=1e-6)
        np.testing.assert_allclose(feats[0, 1], np.log(0.25), rtol=1e-6)


def test_parse_move_features_truncates_and_missing_opp():
    feats, mask = parse_move_features(_row(300, 'B', 1), None)
    assert mask.all() and feats.shape == (200, 14)
    assert np.isfinite(feats).all()


def _humansl_outputs():
    """colab_humansl 參數格的 RUNS 輸出路徑，{MODE: Drive 上的路徑}。"""
    with open('colab_humansl.ipynb', encoding='utf-8') as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    g = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells[0], g)
    return {m: out for m, (_, out, _) in g['RUNS'].items()}


def test_build_jobs_reads_colab_humansl_outputs():
    outs = set(_humansl_outputs().values())
    for six_d in (True, False):
        for t, o, npz in build_jobs('humansl-analysis/', 'packed/', six_d):
            paths = (t if isinstance(t, tuple) else (t,)) + (o if isinstance(o, tuple) else (o,))
            assert set(paths) <= outs, paths
            assert npz.startswith('packed/') and npz.endswith('_moves.npz')


def test_build_jobs_p4all_concatenates_p4_and_p4b():
    jobs = {npz: (t, o) for t, o, npz in build_jobs('h/', 'out/')}
    t, o = jobs['out/train_p4all_moves.npz']
    assert t == ('h/hist2d-train_p4/train_p4-p3-hist.jsonl', 'h/hist2d-train_p4b/train_p4b-p3-hist.jsonl')
    assert o == ('h/hist2d-train_p4/train_p4-p3-opp-hist.jsonl', 'h/hist2d-train_p4b/train_p4b-p3-opp-hist.jsonl')
    six = [npz for _, _, npz in build_jobs('h/', 'out/', six_d=True)]
    assert six == ['out/val648_moves.npz', 'out/trainpool_moves.npz', 'out/newdev_moves.npz', 'out/test_moves.npz']
