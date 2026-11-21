import numpy as np
import pytest

from aicup.blend_submit import ORDER, blend, load


def test_blend_formula():
    rng = np.random.default_rng(0)
    a, b, tf = (rng.dirichlet(np.ones(10), 800) for _ in range(3))
    p = blend(a, b, tf, 1 / 3)
    np.testing.assert_allclose(p, (2 / 3) * (a + b) / 2 + (1 / 3) * tf)
    np.testing.assert_allclose(p.sum(1), 1)


def test_load_rejects_wrong_question_order(tmp_path):
    probs = np.full((800, 10), 0.1)
    good, bad = tmp_path / 'good.npz', tmp_path / 'bad.npz'
    np.savez(good, question_id=np.array(ORDER), probs=probs)
    np.savez(bad, question_id=np.array(ORDER[::-1]), probs=probs)
    assert load(good).shape == (800, 10)
    with pytest.raises(ValueError):
        load(bad)
