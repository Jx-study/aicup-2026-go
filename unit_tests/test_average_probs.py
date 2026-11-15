import numpy as np
import pandas as pd
import pytest

from aicup import average_probs as ap
from aicup.scoring import LEVELS, pick_level


def fake_probs(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(len(LEVELS)), n)
    return np.array(['pub_q_%04d' % (i + 1) for i in range(n)]), p


def test_average_rows_sum_to_one():
    q, a = fake_probs(50, 0)
    _, b = fake_probs(50, 1)
    avg = ap.average_probs(q, a, q, b)
    np.testing.assert_allclose(avg.sum(axis=1), 1.0)
    np.testing.assert_allclose(avg, (a + b) / 2)


@pytest.mark.parametrize('change', ['order', 'value', 'length'])
def test_question_id_mismatch_rejected(change):
    q, a = fake_probs(10, 0)
    _, b = fake_probs(10, 1)
    qb = q.copy()
    if change == 'order':
        qb[[0, 1]] = qb[[1, 0]]
    elif change == 'value':
        qb[3] = 'priv_q_9999'
    else:
        qb, b = qb[:9], b[:9]
    with pytest.raises(ValueError, match='question_id'):
        ap.average_probs(q, a, qb, b)


def test_nan_rejected():
    q, a = fake_probs(10, 0)
    _, b = fake_probs(10, 1)
    b[2] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        ap.average_probs(q, a, q, b)


def test_main_writes_800_rows(tmp_path, monkeypatch):
    q, a = fake_probs(800, 0)
    _, b = fake_probs(800, 1)
    np.savez(tmp_path / 'a.npz', question_id=q, probs=a)
    np.savez(tmp_path / 'b.npz', question_id=q, probs=b)
    monkeypatch.setattr('sys.argv', ['average_probs', '--probs', str(tmp_path / 'a.npz'), str(tmp_path / 'b.npz'),
                                     '--out-dir', str(tmp_path / 'out')])
    ap.main()
    sub = pd.read_csv(tmp_path / 'out' / 'submission_rank_avg.csv')
    assert list(sub.columns) == ['question_id', 'pred_rank']
    assert len(sub) == 800
    assert sub['question_id'].tolist() == q.tolist()
    assert sub['pred_rank'].tolist() == [LEVELS[i] for i in pick_level((a + b) / 2)]
