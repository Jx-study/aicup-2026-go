import json

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import color, predict_rank
from aicup.scoring import LEVELS
from network import GoRankResNet
from tta_helpers import sharp_model, tta_expect

SGF = '(;GM[1]SZ[19];B[pd];W[dp];B[dd];W[pp])'
HD2_SGF = SGF.replace('SZ[19]', 'SZ[19]HD[2]')
_POINTS = np.random.default_rng(0).permutation(361)[:60]
# 60 手不重複的隨機落點，盤面不對稱，8 個方向的輸入差異夠大
ASYM_SGF = '(;GM[1]SZ[19]' + ''.join(';%s[%s%s]' % ('BW'[i % 2], chr(97 + p % 19), chr(97 + p // 19))
                                     for i, p in enumerate(_POINTS)) + ')'
BOTH_CHANNELS = color.in_channels('both')


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


def _tiny_ckpt(tmp_path, in_channels=500):
    torch.manual_seed(0)
    model = GoRankResNet(in_channels=in_channels, num_channels=8, num_blocks=1, num_classes=len(LEVELS))
    path = tmp_path / 'tiny.pth'
    torch.save(model.state_dict(), path)
    return str(path)


def test_read_questions_skips_empty_columns(tmp_path):
    _write_tests(tmp_path, [[SGF, SGF, SGF]], [[SGF]])
    qids, games = predict_rank.read_questions(str(tmp_path))
    assert qids == ['pub_0', 'priv_0']
    assert [q for q, _, _ in games] == [0, 0, 0, 1]


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
             game_q=np.zeros(5, np.int64), ckpt_sha256='x')
    monkeypatch.setattr('sys.argv', [
        'predict_rank', '--test-dir', str(tmp_path), '--out', str(tmp_path / 's.csv'),
        '--probs', str(tmp_path / 'probs.npz')])
    with pytest.raises(SystemExit, match='測試資料'):
        predict_rank.main()


def test_main_rejects_cache_without_ckpt_record(tmp_path, monkeypatch):
    _write_tests(tmp_path, [[SGF]], [[SGF]])
    np.savez(tmp_path / 'probs.npz', probs=np.zeros((2, 10), np.float32), game_q=np.arange(2))
    monkeypatch.setattr('sys.argv', [
        'predict_rank', '--test-dir', str(tmp_path), '--out', str(tmp_path / 's.csv'),
        '--probs', str(tmp_path / 'probs.npz')])
    with pytest.raises(SystemExit, match='刪除後重跑'):
        predict_rank.main()


def test_main_rejects_cache_from_other_ckpt(tmp_path, monkeypatch):
    _write_tests(tmp_path, [[SGF]], [[SGF]])
    ckpt = _tiny_ckpt(tmp_path)
    argv = ['predict_rank', '--test-dir', str(tmp_path), '--out', str(tmp_path / 's.csv'),
            '--probs', str(tmp_path / 'probs.npz'), '--channels', '8', '--blocks', '1', '--nproc', '1']
    monkeypatch.setattr('sys.argv', argv + ['--ckpt', ckpt])
    predict_rank.main()
    predict_rank.main()  # 同一個 checkpoint 可以沿用快取

    other = tmp_path / 'other.pth'
    other.write_bytes(open(ckpt, 'rb').read() + b'x')
    monkeypatch.setattr('sys.argv', argv + ['--ckpt', str(other)])
    with pytest.raises(SystemExit, match='不是'):
        predict_rank.main()


def test_decide_temperature_changes_log_pick():
    # 真實等級 4，每局猜 3 或 5，溫度 1 時被多數方帶走，調高後選中間
    lean_low = np.zeros(10)
    lean_low[[3, 4, 5]] = [0.8, 0.15, 0.05]
    lean_high = lean_low.copy()
    lean_high[[3, 5]] = [0.05, 0.8]
    probs = np.array([lean_low] * 3 + [lean_high] * 2, np.float32)
    assert predict_rank.decide(1, [0] * 5, probs, 'log', 'pick', temp=1.0).tolist() == [3]
    assert predict_rank.decide(1, [0] * 5, probs, 'log', 'pick', temp=8.0).tolist() == [4]


def _write_colored(tmp_path, games):
    """public、private 各一題，games 為 (sgf, color) 清單。"""
    row = {'question_id': 'q_0'}
    for k in range(1, 21):
        sgf, c = games[k - 1] if k <= len(games) else ('', '')
        row['sgf_%d' % k], row['color_%d' % k] = sgf, c
    for split in ('public', 'private'):
        pd.DataFrame([row]).to_csv(tmp_path / ('rank_prediction_test_%s.csv' % split), index=False)


def test_read_questions_pairs_color_with_game(tmp_path):
    _write_colored(tmp_path, [(SGF, 'W'), (SGF, 'B')])
    _, games = predict_rank.read_questions(str(tmp_path))
    assert [c for _, _, c in games[:2]] == ['W', 'B']


def test_game_colors_flip_handicap_only():
    hd1 = SGF.replace('SZ[19]', 'SZ[19]HD[1]')
    games = [(0, HD2_SGF, 'W'), (0, hd1, 'W'), (0, SGF, 'B')]
    assert predict_rank.game_colors(games, raw_color=False).tolist() == [0, 1, 0]
    assert predict_rank.game_colors(games, raw_color=True).tolist() == [1, 1, 0]


def test_game_colors_reject_missing_column():
    with pytest.raises(SystemExit, match='color_'):
        predict_rank.game_colors([(0, SGF, None)], raw_color=False)


def test_predict_games_applies_color():
    torch.manual_seed(0)
    model = GoRankResNet(in_channels=BOTH_CHANNELS, num_channels=8, num_blocks=1,
                         num_classes=len(LEVELS)).eval()
    probs = predict_rank.predict_games(model, [SGF, SGF], torch.device('cpu'), amp=False, batch_size=2,
                                       nproc=1, colors=np.array([0, 1]), color_mode='both')
    feat = torch.from_numpy(predict_rank.sgf_to_features(SGF)[None]).float()
    with torch.no_grad():
        expect = torch.softmax(model(color.apply_color(feat, torch.tensor([1]), 'both')), 1).numpy()
    np.testing.assert_allclose(probs[1], expect[0], atol=1e-5)
    assert not np.allclose(probs[0], probs[1], atol=1e-6)


def test_predict_games_keeps_color_after_bad_sgf():
    torch.manual_seed(0)
    model = GoRankResNet(in_channels=BOTH_CHANNELS, num_channels=8, num_blocks=1,
                         num_classes=len(LEVELS)).eval()
    probs = predict_rank.predict_games(model, [SGF, 'not an sgf', SGF], torch.device('cpu'), amp=False,
                                       batch_size=2, nproc=1, colors=np.array([0, 0, 1]), color_mode='both')
    feat = torch.from_numpy(predict_rank.sgf_to_features(SGF)[None]).float()
    with torch.no_grad():
        expect = torch.softmax(model(color.apply_color(feat, torch.tensor([1]), 'both')), 1).numpy()
    np.testing.assert_allclose(probs[2], expect[0], atol=1e-5)


@pytest.mark.parametrize('raw_color, expected', [(False, [0, 0]), (True, [1, 0])], ids=['fixed', 'raw'])
def test_main_follows_train_config(tmp_path, monkeypatch, raw_color, expected):
    _write_colored(tmp_path, [(HD2_SGF, 'W'), (SGF, 'B')])
    ckpt = _tiny_ckpt(tmp_path, BOTH_CHANNELS)
    (tmp_path / 'train_config.json').write_text(
        json.dumps({'color': 'both', 'raw_color': raw_color}), encoding='utf-8')
    real, seen = predict_rank.predict_games, {}

    def spy(model, sgfs, device, amp, batch_size, nproc, colors, color_mode, **kw):
        seen.update(colors=colors, mode=color_mode)
        return real(model, sgfs, device, amp, batch_size, nproc, colors, color_mode, **kw)

    monkeypatch.setattr(predict_rank, 'predict_games', spy)
    monkeypatch.setattr('sys.argv', [
        'predict_rank', '--ckpt', ckpt, '--test-dir', str(tmp_path), '--out', str(tmp_path / 's.csv'),
        '--probs', str(tmp_path / 'p.npz'), '--channels', '8', '--blocks', '1', '--nproc', '1'])
    predict_rank.main()
    assert seen['mode'] == 'both'
    assert seen['colors'].tolist() == expected * 2  # public、private 各一題
    assert len(pd.read_csv(tmp_path / 's.csv')) == 2


def test_read_train_config_warns_when_missing(tmp_path, capsys):
    cfg = predict_rank.read_train_config(str(tmp_path / 'best_model.pth'))
    assert cfg == {'color': 'none', 'raw_color': False}
    assert 'train_config.json' in capsys.readouterr().out


def test_predict_games_tta_averages_symmetries():
    model = sharp_model()
    probs = predict_rank.predict_games(model, [ASYM_SGF], torch.device('cpu'), amp=False, batch_size=2,
                                       nproc=1, tta=True)
    plain = predict_rank.predict_games(model, [ASYM_SGF], torch.device('cpu'), amp=False, batch_size=2, nproc=1)
    feat = torch.from_numpy(predict_rank.sgf_to_features(ASYM_SGF)[None]).float()
    np.testing.assert_allclose(probs, tta_expect(model, feat).numpy(), atol=1e-5)
    assert np.abs(probs - plain).max() > 1e-3


def test_predict_games_tta_with_color():
    model = sharp_model(BOTH_CHANNELS)
    probs = predict_rank.predict_games(model, [ASYM_SGF], torch.device('cpu'), amp=False, batch_size=2,
                                       nproc=1, colors=np.array([1]), color_mode='both', tta=True)
    feat = torch.from_numpy(predict_rank.sgf_to_features(ASYM_SGF)[None]).float()
    expect = tta_expect(model, color.apply_color(feat, torch.tensor([1]), 'both')).numpy()
    np.testing.assert_allclose(probs, expect, atol=1e-5)


def test_main_rejects_cache_with_other_tta(tmp_path, monkeypatch):
    _write_tests(tmp_path, [[SGF]], [[SGF]])
    argv = ['predict_rank', '--ckpt', _tiny_ckpt(tmp_path), '--test-dir', str(tmp_path),
            '--out', str(tmp_path / 's.csv'), '--probs', str(tmp_path / 'probs.npz'),
            '--channels', '8', '--blocks', '1', '--nproc', '1']
    monkeypatch.setattr('sys.argv', argv + ['--tta'])
    predict_rank.main()
    predict_rank.main()  # 同樣開 TTA 可以沿用快取
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit, match='TTA'):
        predict_rank.main()


def test_main_rejects_old_cache_without_tta_field(tmp_path, monkeypatch):
    _write_tests(tmp_path, [[SGF]], [[SGF]])
    ckpt = _tiny_ckpt(tmp_path)
    np.savez(tmp_path / 'probs.npz', probs=np.full((2, 10), 0.1, np.float32), game_q=np.arange(2),
             ckpt_sha256=predict_rank.file_sha256(ckpt))
    monkeypatch.setattr('sys.argv', [
        'predict_rank', '--ckpt', ckpt, '--test-dir', str(tmp_path), '--out', str(tmp_path / 's.csv'),
        '--probs', str(tmp_path / 'probs.npz'), '--tta'])
    with pytest.raises(SystemExit, match='TTA'):
        predict_rank.main()
