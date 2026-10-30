import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import packed, train_rank
from aicup.scoring import LEVELS
from network import GoRankResNet


def _write_split(split_dir, n, seed):
    rng = np.random.default_rng(seed)
    split_dir.mkdir(parents=True)
    rows = [{'features': rng.integers(0, 256, packed.PACKED_LEN, dtype=np.uint8),
             'rank_idx': i % len(LEVELS), 'color': i % 2,
             'key': '%s_g%d' % (LEVELS[i % len(LEVELS)], i), 'player_id': 'p%d' % (i // 3)}
            for i in range(n)]
    # 分兩片，驗證多分片時順序仍對得上
    packed.save_shard(str(split_dir / 'a.npz'), rows[:n // 2])
    packed.save_shard(str(split_dir / 'b.npz'), rows[n // 2:])


def _setup(tmp_path):
    feat = tmp_path / 'feat'
    _write_split(feat / 'train', 16, 0)
    _write_split(feat / 'val', 10, 1)
    return feat, tmp_path / 'ckpt'


def _main(monkeypatch, feat, ckpt, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'train_rank', '--feat-dir', str(feat), '--ckpt-dir', str(ckpt),
        '--batch-size', '4', '--channels', '8', '--blocks', '1', *extra])
    train_rank.main()


def _load_probs(ckpt):
    with np.load(ckpt / 'val_probs.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _recompute(weights, feat):
    model = GoRankResNet(num_channels=8, num_blocks=1, num_classes=len(LEVELS))
    model.load_state_dict(torch.load(weights, weights_only=True))
    model.eval()
    val = packed.load_split(str(feat / 'val'))
    with torch.no_grad():
        x = packed.unpack_torch(torch.from_numpy(val['features']))
        return torch.softmax(model(x), 1).numpy()


def test_val_probs_written_with_best_model(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp')
    saved = _load_probs(ckpt)
    val = packed.load_split(str(feat / 'val'))

    assert saved['probs'].dtype == np.float32 and saved['probs'].shape == (10, len(LEVELS))
    np.testing.assert_allclose(saved['probs'].sum(1), 1, atol=1e-5)
    for k in ('rank_idx', 'key', 'player_id'):
        np.testing.assert_array_equal(saved[k], val[k])
    assert int(saved['epoch']) == 1
    np.testing.assert_allclose(saved['probs'], _recompute(ckpt / 'best_model.pth', feat), atol=1e-5)


def test_val_probs_follow_best_epoch(tmp_path, monkeypatch):
    # validate 每個 epoch 先算 argmax 再算 pick；讓 epoch 1 的 pick 最高
    scores = iter([0.1, 0.9, 0.1, 0.2, 0.1, 0.3])
    monkeypatch.setattr(train_rank, 'score_strength', lambda *a: next(scores))
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '3', '--no-amp')
    saved = _load_probs(ckpt)

    assert int(saved['epoch']) == 1
    np.testing.assert_allclose(saved['probs'], _recompute(ckpt / 'epoch_1.pth', feat), atol=1e-5)
    assert not np.allclose(saved['probs'], _recompute(ckpt / 'epoch_3.pth', feat), atol=1e-5)


def test_val_probs_backfilled_on_resume(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp')
    (ckpt / 'val_probs.npz').unlink()
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp')
    saved = _load_probs(ckpt)

    assert int(saved['epoch']) == 1
    np.testing.assert_allclose(saved['probs'], _recompute(ckpt / 'best_model.pth', feat), atol=1e-5)


def test_val_probs_with_amp(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1')
    saved = _load_probs(ckpt)

    assert saved['probs'].dtype == np.float32 and saved['probs'].shape == (10, len(LEVELS))
    np.testing.assert_allclose(saved['probs'].sum(1), 1, atol=1e-3)


def test_lr_at_cosine():
    lr = lambda s: train_rank.lr_at(s, 100, 1.0, 'cosine', 10)
    assert lr(0) == pytest.approx(0.1)
    assert lr(9) == pytest.approx(1.0)
    assert lr(10) == pytest.approx(1.0)
    assert lr(55) == pytest.approx(0.5)
    assert lr(100) == pytest.approx(0.0)
    assert all(lr(s) >= lr(s + 1) for s in range(10, 100))
    assert train_rank.lr_at(50, 100, 1e-3, 'constant', 10) == 1e-3


@pytest.mark.parametrize('extra', [(), ('--augment',)], ids=['plain', 'augment'])
def test_cosine_resume_matches_single_run(tmp_path, monkeypatch, extra):
    feat, _ = _setup(tmp_path)
    once, twice = tmp_path / 'once', tmp_path / 'twice'
    args = ('--epochs', '2', '--no-amp', '--schedule', 'cosine', '--warmup-steps', '2', *extra)
    _main(monkeypatch, feat, once, *args)

    validate, calls = train_rank.validate, []

    def fail_second(*a, **k):  # 第 2 個 epoch 存檔前中斷
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return validate(*a, **k)

    monkeypatch.setattr(train_rank, 'validate', fail_second)
    with pytest.raises(KeyboardInterrupt):
        _main(monkeypatch, feat, twice, *args)
    monkeypatch.setattr(train_rank, 'validate', validate)
    _main(monkeypatch, feat, twice, *args)

    a, b = pd.read_csv(once / 'history.csv'), pd.read_csv(twice / 'history.csv')
    cols = ['train_loss', 'lr', 'val_loss']
    pd.testing.assert_frame_equal(a[cols], b[cols])
    # 16 局、batch 4、2 個 epoch 共 8 步，最後一步是第 7 步
    assert a['lr'].iloc[-1] == pytest.approx(train_rank.lr_at(7, 8, 1e-3, 'cosine', 2))


def test_cosine_rejects_changed_epochs(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    args = ('--no-amp', '--schedule', 'cosine', '--warmup-steps', '2')
    _main(monkeypatch, feat, ckpt, '--epochs', '1', *args)
    with pytest.raises(SystemExit):
        _main(monkeypatch, feat, ckpt, '--epochs', '2', *args)


def test_patience_exhausted():
    h = [{'val_score_pick': s} for s in (0.1, 0.3, 0.2, 0.25)]
    assert not train_rank.patience_exhausted(h, 0)
    assert not train_rank.patience_exhausted(h, 3)
    assert train_rank.patience_exhausted(h, 2)
    assert not train_rank.patience_exhausted([], 1)
    tie = [{'val_score_pick': s} for s in (0.3, 0.3)]
    assert train_rank.patience_exhausted(tie, 1)  # 平手不算超過，和 best_model.pth 的更新條件一致


def test_patience_stops_and_stays_stopped(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    scores = iter([0.5, 0.4, 0.3, 0.2, 0.1])
    validate = train_rank.validate

    def fake(*a, **k):  # val 分數逐 epoch 下降
        metrics, probs = validate(*a, **k)
        metrics['val_score_pick'] = next(scores)
        return metrics, probs

    monkeypatch.setattr(train_rank, 'validate', fake)
    args = ('--epochs', '5', '--no-amp', '--patience', '2', '--min-epochs', '0')
    _main(monkeypatch, feat, ckpt, *args)
    assert len(pd.read_csv(ckpt / 'history.csv')) == 3
    _main(monkeypatch, feat, ckpt, *args)  # 重跑不會繼續訓練
    assert len(pd.read_csv(ckpt / 'history.csv')) == 3


def test_augment_changes_training_and_is_deterministic(tmp_path, monkeypatch):
    feat, _ = _setup(tmp_path)
    runs = {}
    for name, extra in (('plain', ()), ('aug1', ('--augment',)), ('aug2', ('--augment',))):
        _main(monkeypatch, feat, tmp_path / name, '--epochs', '2', '--no-amp', *extra)
        runs[name] = pd.read_csv(tmp_path / name / 'history.csv')['train_loss']
    pd.testing.assert_series_equal(runs['aug1'], runs['aug2'])
    assert not runs['plain'].equals(runs['aug1'])


def test_augment_rejects_resume_without_augment(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--augment')
    with pytest.raises(SystemExit):
        _main(monkeypatch, feat, ckpt, '--epochs', '2', '--no-amp')


def test_patience_off_by_default(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    scores = iter([0.5, 0.4, 0.3, 0.2, 0.1])
    validate = train_rank.validate

    def fake(*a, **k):
        metrics, probs = validate(*a, **k)
        metrics['val_score_pick'] = next(scores)
        return metrics, probs

    monkeypatch.setattr(train_rank, 'validate', fake)
    _main(monkeypatch, feat, ckpt, '--epochs', '5', '--no-amp', '--min-epochs', '0')
    assert len(pd.read_csv(ckpt / 'history.csv')) == 5


def test_runs_log_records_each_start(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp')
    _main(monkeypatch, feat, ckpt, '--epochs', '2', '--no-amp')
    _main(monkeypatch, feat, ckpt, '--epochs', '2', '--no-amp')  # 已跑完，沒有訓練就不記錄
    runs = [json.loads(s) for s in (ckpt / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['start_epoch'] for r in runs] == [0, 1]
    assert runs[1]['argv'][-3:] == ['--epochs', '2', '--no-amp'] and runs[1]['commit']


def test_min_epochs_delays_patience():
    # 前 3 個 epoch 下跌，第 4 個回升，對應前期 val 跳動
    h = [{'val_score_pick': s} for s in (0.31, 0.29, 0.27, 0.35)]
    assert train_rank.patience_exhausted(h[:3], 2)
    assert not train_rank.patience_exhausted(h[:3], 2, min_epochs=6)
    late = [{'val_score_pick': s} for s in (0.1, 0.2, 0.3, 0.3, 0.3, 0.3)]
    assert train_rank.patience_exhausted(late, 3, min_epochs=6)
