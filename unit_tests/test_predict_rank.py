import numpy as np
import pandas as pd
import pytest
import torch

from aicup import predict_rank
from aicup.scoring import LEVELS
from network import GoRankResNet

SGF = '(;GM[1]SZ[19];B[pd];W[dp];B[dd];W[pp])'


def _write_tests(tmp_path, public, private):
    """public、private 為每題的 sgf 清單，其餘 sgf_ 欄位留空。"""
    for split, questions in (('public', public), ('private', private)):
        rows = []
        for i, sgfs in enumerate(questions):
            row = {'question_id': '%s_%d' % (split[:3] if split == 'public' else 'priv', i)}
            for k in range(1, 21):
                row['sgf_%d' % k] = sgfs[k - 1] if k <= len(sgfs) else ''
            rows.append(row)
        pd.DataFrame(rows).to_csv(tmp_path / ('rank_prediction_test_%s.csv' % split), index=False)


def _tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    model = GoRankResNet(num_channels=8, num_blocks=1, num_classes=len(LEVELS))
    path = tmp_path / 'tiny.pth'
    torch.save(model.state_dict(), path)
    return str(path)


def test_read_questions_skips_empty_columns(tmp_path):
    _write_tests(tmp_path, [[SGF, SGF, SGF]], [[SGF]])
    qids, games = predict_rank.read_questions(str(tmp_path))
    assert qids == ['pub_0', 'priv_0']
    assert [q for q, _ in games] == [0, 0, 0, 1]


def test_decide_aggregates_per_question():
    probs = np.array([[0.0] * 9 + [1.0], [0.0] * 9 + [1.0], [1.0] + [0.0] * 9], np.float32)
    preds = predict_rank.decide(2, [0, 0, 1], probs, agg='mean', select='argmax')
    assert preds.tolist() == [9, 0]


def test_decide_all_failed_falls_back_to_weakest():
    probs = np.full((2, 10), np.nan, np.float32)
    assert predict_rank.decide(1, [0, 0], probs).tolist() == [0]


def test_decide_ignores_failed_games():
    good = np.zeros((1, 10), np.float32)
    good[0, 4] = 1.0
    probs = np.vstack([good, np.full((1, 10), np.nan, np.float32)])
    assert predict_rank.decide(1, [0, 0], probs, select='argmax').tolist() == [4]


def test_predict_games_marks_bad_sgf_as_nan():
    model = GoRankResNet(num_channels=8, num_blocks=1, num_classes=len(LEVELS)).eval()
    probs = predict_rank.predict_games(model, [SGF, 'not an sgf', SGF], torch.device('cpu'),
                                       amp=False, batch_size=2, nproc=1)
    assert probs.shape == (3, 10)
    assert np.isnan(probs[1]).all()
    assert np.allclose(probs[[0, 2]].sum(axis=1), 1.0, atol=1e-5)


def test_main_writes_submission(tmp_path, monkeypatch):
    _write_tests(tmp_path, [[SGF, SGF], ['bad']], [[SGF]])
    out = tmp_path / 'sub.csv'
    monkeypatch.setattr('sys.argv', [
        'predict_rank', '--ckpt', _tiny_ckpt(tmp_path), '--test-dir', str(tmp_path),
        '--out', str(out), '--probs', str(tmp_path / 'probs.npz'),
        '--channels', '8', '--blocks', '1', '--nproc', '1'])
    predict_rank.main()
    sub = pd.read_csv(out)
    assert sub['question_id'].tolist() == ['pub_0', 'pub_1', 'priv_0']
    assert set(sub['pred_rank']) <= set(LEVELS)
    assert sub.loc[1, 'pred_rank'] == LEVELS[0]  # 整題解析失敗

    # 第二次直接讀快取，不需要 --ckpt
    monkeypatch.setattr('sys.argv', [
        'predict_rank', '--test-dir', str(tmp_path), '--out', str(out),
        '--probs', str(tmp_path / 'probs.npz'), '--agg', 'log', '--select', 'argmax'])
    predict_rank.main()
    assert len(pd.read_csv(out)) == 3


def test_main_rejects_stale_cache(tmp_path, monkeypatch):
    _write_tests(tmp_path, [[SGF]], [[SGF]])
    np.savez(tmp_path / 'probs.npz', probs=np.zeros((5, 10), np.float32),
             game_q=np.zeros(5, np.int64))
    monkeypatch.setattr('sys.argv', [
        'predict_rank', '--test-dir', str(tmp_path), '--out', str(tmp_path / 's.csv'),
        '--probs', str(tmp_path / 'probs.npz')])
    with pytest.raises(SystemExit):
        predict_rank.main()
