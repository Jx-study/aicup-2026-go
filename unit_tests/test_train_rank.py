import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from aicup import color, packed, symmetry, train_rank
from aicup.scoring import LEVELS
from network import GoRankResNet
from tta_helpers import sharp_model, tta_expect


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


def _recompute(weights, feat, mode='none', hd=None):
    model = GoRankResNet(in_channels=color.in_channels(mode), num_channels=8, num_blocks=1,
                         num_classes=len(LEVELS))
    model.load_state_dict(torch.load(weights, weights_only=True))
    model.eval()
    val = packed.load_split(str(feat / 'val'))
    c = val['color'] if hd is None else color.feature_color(val['color'], hd)
    with torch.no_grad():
        x = packed.unpack_torch(torch.from_numpy(val['features']))
        return torch.softmax(model(color.apply_color(x, torch.from_numpy(c).long(), mode)), 1).numpy()


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


def _write_hd(feat):
    """給 train、val 兩邊的局寫 hd.npz，每 4 局有 1 局讓子。"""
    keys = np.concatenate([packed.load_split(str(feat / s))['key'] for s in ('train', 'val')])
    hd = np.array([3 if int(k.split('_g')[1]) % 4 == 0 else 0 for k in keys], np.int8)
    np.savez(feat / 'hd.npz', key=keys, hd=hd)


def test_color_requires_hd_map(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    with pytest.raises(SystemExit, match='hd.npz'):
        _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'both')


def test_color_val_uses_feature_color(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'both')
    val = packed.load_split(str(feat / 'val'))
    hd = color.load_hd(str(feat / 'hd.npz'), val['key'])
    saved = _load_probs(ckpt)['probs']
    np.testing.assert_allclose(saved, _recompute(ckpt / 'best_model.pth', feat, 'both', hd), atol=1e-5)
    assert not np.allclose(saved, _recompute(ckpt / 'best_model.pth', feat, 'both'), atol=1e-5)


def test_raw_color_ignores_handicap(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'swap', '--raw-color')
    np.testing.assert_allclose(_load_probs(ckpt)['probs'],
                               _recompute(ckpt / 'best_model.pth', feat, 'swap'), atol=1e-5)


def test_color_written_to_train_config(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'plane')
    cfg = json.loads((ckpt / 'train_config.json').read_text(encoding='utf-8'))
    assert cfg == {'color': 'plane', 'raw_color': False}


def test_color_rejects_resume_with_other_mode(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'swap')
    written = (ckpt / 'train_config.json').read_text(encoding='utf-8')
    with pytest.raises(SystemExit, match='設定'):
        _main(monkeypatch, feat, ckpt, '--epochs', '2', '--no-amp')
    assert (ckpt / 'train_config.json').read_text(encoding='utf-8') == written  # 被拒絕時不能改掉設定


def test_color_with_augment_runs(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'both', '--augment')
    assert (ckpt / 'best_model.pth').exists()


def test_use_amp_ignores_emulated_bf16(monkeypatch):
    calls = []

    def supported(including_emulation=True):
        calls.append(including_emulation)
        return including_emulation

    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', supported)
    assert train_rank.use_amp(torch.device('cuda'), no_amp=False) is False
    assert calls == [False]
    assert train_rank.use_amp(torch.device('cpu'), no_amp=False) is True
    assert train_rank.use_amp(torch.device('cpu'), no_amp=True) is False


def test_raw_color_written_to_train_config(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'plane', '--raw-color')
    cfg = json.loads((ckpt / 'train_config.json').read_text(encoding='utf-8'))
    assert cfg == {'color': 'plane', 'raw_color': True}


def test_batches_keep_color_aligned_with_order():
    n = 6
    feats = np.zeros((n, *packed.FEATURE_SHAPE), np.uint8)
    feats[np.arange(n), 0, 0, np.arange(n)] = 1  # 第 0 個黑面的第 i 格標記第 i 局
    x = torch.from_numpy(np.stack([packed.pack(f) for f in feats]))
    y, c = torch.arange(n), torch.tensor([0, 1, 1, 0, 1, 0])
    order = torch.tensor([3, 1, 5, 0, 4, 2])
    seen = []
    for xb, yb in train_rank.batches(x, y, 4, torch.device('cpu'), order, c, 'both'):
        for feat, game in zip(xb, yb.tolist()):
            seen.append(game)
            target = int(c[game])
            assert feat[target, 0, game] == 1  # 黑面或白面取決於這一局自己的顏色
            assert feat[500].sum() == (361 if target == 0 else 0)
    assert seen == order.tolist()


def test_train_batches_get_corrected_color(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    real, seen = train_rank.batches, []

    def spy(x, y, batch_size, device, order=None, c=None, mode='none'):
        if order is not None:
            seen.append(c)
        return real(x, y, batch_size, device, order, c, mode)

    monkeypatch.setattr(train_rank, 'batches', spy)
    _main(monkeypatch, feat, ckpt, '--epochs', '1', '--no-amp', '--color', 'both')
    train = packed.load_split(str(feat / 'train'))
    hd = color.load_hd(str(feat / 'hd.npz'), train['key'])
    np.testing.assert_array_equal(seen[0].numpy(), color.feature_color(train['color'], hd))


def test_val_probs_backfill_uses_color(tmp_path, monkeypatch):
    feat, ckpt = _setup(tmp_path)
    _write_hd(feat)
    args = ('--epochs', '1', '--no-amp', '--color', 'both')
    _main(monkeypatch, feat, ckpt, *args)
    (ckpt / 'val_probs.npz').unlink()
    _main(monkeypatch, feat, ckpt, *args)
    hd = color.load_hd(str(feat / 'hd.npz'), packed.load_split(str(feat / 'val'))['key'])
    np.testing.assert_allclose(_load_probs(ckpt)['probs'],
                               _recompute(ckpt / 'best_model.pth', feat, 'both', hd), atol=1e-5)



def test_forward_probs_tta_averages_symmetries():
    model = sharp_model()
    x = torch.rand(3, 500, 19, 19)
    with torch.no_grad():
        got = train_rank.forward_probs(model, x, amp=False, tta=True)
        logit_mean = torch.softmax(torch.stack([model(symmetry.apply(x, k).contiguous())
                                                for k in range(symmetry.N_SYMMETRIES)]).mean(0), 1)
        plain = train_rank.forward_probs(model, x, amp=False)
        torch.testing.assert_close(plain, torch.softmax(model(x), 1))
    torch.testing.assert_close(got, tta_expect(model, x))
    # 平均機率與平均 logits、單一方向都要分得開，測試才抓得到這兩種錯誤
    assert (got - logit_mean).abs().max() > 1e-3
    assert (got - plain).abs().max() > 1e-3


def test_validate_tta_loss_is_nll_of_averaged_probs(tmp_path):
    feat, _ = _setup(tmp_path)
    val = packed.load_split(str(feat / 'val'))
    x, y = torch.from_numpy(val['features']), torch.from_numpy(val['rank_idx'].astype(np.int64))
    args = SimpleNamespace(batch_size=4, color='none')
    metrics, probs = train_rank.validate(sharp_model(), x, y, args, torch.device('cpu'), False,
                                         torch.nn.CrossEntropyLoss(), tta=True)
    nll = -np.log(probs[np.arange(len(probs)), val['rank_idx']])
    # validate 的 val_loss 是各 batch 平均的平均，batch 4、4、2
    expect = np.mean([nll[:4].mean(), nll[4:8].mean(), nll[8:].mean()])
    assert metrics['val_loss'] == pytest.approx(expect, rel=1e-5)
