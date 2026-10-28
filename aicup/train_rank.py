"""棋力預測訓練：整份打包資料放進 GPU 記憶體，每個 batch 在 GPU 上解包。

模型與預設超參數同 rank-prediction.ipynb。每個 epoch 存 last.pt，重跑自動接續。
best_model.pth 依 val_score_pick 挑選，只存 state_dict。

用法（在 repo 根目錄）：
    python -m aicup.train_rank
    python -m aicup.train_rank --epochs 10   # 已跑完 5 個 epoch 時接續到第 10 個
    python -m aicup.train_rank --fresh       # 忽略既有 checkpoint 從頭訓練
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from aicup import packed
from aicup.scoring import LEVELS, pick_level, score_strength
from network import GoRankResNet

NUM_CLASSES = len(LEVELS)
GPU_MARGIN_GB = 14  # 資料之外留給模型訓練的 VRAM，教學版設定約需 9 GB


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--feat-dir', default='rank-features')
    ap.add_argument('--ckpt-dir', default='rank-trained-models')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto',
                    help='打包資料放哪裡；auto 在 VRAM 足夠時放 GPU')
    ap.add_argument('--no-amp', action='store_true', help='關閉 bf16，改用純 fp32')
    ap.add_argument('--fresh', action='store_true', help='忽略既有 last.pt')
    ap.add_argument('--max-steps', type=int, default=0, help='每個 epoch 最多幾步，0 為不限，用於冒煙測試')
    return ap.parse_args()


def load_to(split_dir, data_device):
    """載入一個 split，打包特徵與標籤搬到 data_device，numpy 副本隨即釋放。"""
    t0 = time.time()
    d = packed.load_split(split_dir)
    x = torch.from_numpy(d.pop('features')).to(data_device)
    y = torch.from_numpy(d['rank_idx'].astype(np.int64)).to(data_device)
    print('  %s：%s 筆，%.1f GB，%.1f 秒' % (split_dir, format(len(y), ','),
                                          x.numel() / 1024 ** 3, time.time() - t0), flush=True)
    return x, y


def pick_data_device(args, device, n_bytes):
    if args.data_device != 'auto':
        return torch.device(args.data_device)
    if device.type != 'cuda':
        return device
    free, _ = torch.cuda.mem_get_info()
    need = n_bytes + GPU_MARGIN_GB * 1024 ** 3
    if free >= need:
        return device
    print('VRAM 剩 %.1f GB，不足 %.1f GB，資料改放 CPU' % (free / 1024 ** 3, need / 1024 ** 3))
    return torch.device('cpu')


def split_bytes(split_dir):
    """依分片筆數估算打包特徵大小，不必先解壓。"""
    n = 0
    for p in packed.list_shards(split_dir):
        with np.load(p, allow_pickle=False) as z:
            n += len(z['rank_idx'])
    return n * packed.PACKED_LEN


def batches(x, y, batch_size, device, order=None):
    """依 order（預設為原順序）切 batch，在 device 上解包成 float。"""
    n = len(y)
    for i in range(0, n, batch_size):
        if order is None:
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        else:
            idx = order[i:i + batch_size]
            xb, yb = x[idx], y[idx]
        xb = xb.to(device, non_blocking=True)
        yield packed.unpack_torch(xb), yb.to(device, non_blocking=True)


def validate(model, x, y, args, device, amp, criterion):
    model.eval()
    loss_sum, batches_n, probs = 0.0, 0, []
    with torch.no_grad():
        for xb, yb in batches(x, y, args.batch_size, device):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(xb)
            loss_sum += criterion(out.float(), yb).item()
            batches_n += 1
            probs.append(torch.softmax(out.float(), 1).cpu().numpy())
    model.train()
    probs, labels = np.concatenate(probs), y.cpu().numpy()
    return {
        'val_loss': loss_sum / max(batches_n, 1),
        'val_acc': float((probs.argmax(1) == labels).mean() * 100),
        'val_score_argmax': score_strength(labels, probs.argmax(1)),
        'val_score_pick': score_strength(labels, pick_level(probs)),
        'val_n': len(labels),
    }


def save_atomic(obj, path):
    """先寫暫存檔再改名，存到一半斷線不會毀掉上一份 checkpoint。"""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = not args.no_amp and (device.type == 'cpu' or torch.cuda.is_bf16_supported())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dir = os.path.join(args.feat_dir, 'train')
    val_dir = os.path.join(args.feat_dir, 'val')
    data_device = pick_data_device(args, device, split_bytes(train_dir) + split_bytes(val_dir))
    print('載入資料到 %s' % data_device, flush=True)
    x_val, y_val = load_to(val_dir, data_device)
    x_train, y_train = load_to(train_dir, data_device)

    model = GoRankResNet(num_channels=args.channels, num_blocks=args.blocks,
                         num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, 'last.pt')
    config = {k: getattr(args, k) for k in ('batch_size', 'lr', 'channels', 'blocks', 'seed')}
    config['amp'] = amp
    start_epoch, history, best_score = 0, [], -1.0
    if os.path.exists(last_path) and not args.fresh:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt['config'] != config:
            raise SystemExit('last.pt 的設定與這次不同：\n  舊 %s\n  新 %s\n'
                             '要用新設定請加 --fresh 或換 --ckpt-dir' % (ckpt['config'], config))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, history, best_score = ckpt['epoch'], ckpt['history'], ckpt['best_score']
        print('從 epoch %d 接續（目前最佳 val_score_pick %.4f）' % (start_epoch, best_score))

    n_params = sum(p.numel() for p in model.parameters())
    print('device %s | data %s | amp %s | train %s / val %s | 參數量 %s'
          % (device, data_device, amp, format(len(y_train), ','), format(len(y_val), ','),
             format(n_params, ',')))
    print('設定', json.dumps(config, ensure_ascii=False))
    if start_epoch >= args.epochs:
        print('已完成 %d 個 epoch，要繼續訓練請加大 --epochs' % start_epoch)
        return

    model.train()
    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)  # dropout 的亂數也隨 epoch 固定
        gen = torch.Generator(device=data_device).manual_seed(args.seed + epoch)
        order = torch.randperm(len(y_train), generator=gen, device=data_device)
        t0 = time.time()
        loss_sum, correct, total, steps = 0.0, 0, 0, 0
        for xb, yb in batches(x_train, y_train, args.batch_size, device, order):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(xb)
            loss = criterion(out.float(), yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
            steps += 1
            if steps % 500 == 0:
                print('  ep%d step %d  loss %.4f  acc %.2f%%  %.1f 分'
                      % (epoch + 1, steps, loss_sum / steps, 100 * correct / total,
                         (time.time() - t0) / 60), flush=True)
            if args.max_steps and steps >= args.max_steps:
                break

        train_min = (time.time() - t0) / 60
        metrics = validate(model, x_val, y_val, args, device, amp, criterion)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / max(steps, 1),
               'train_acc': 100 * correct / max(total, 1), **metrics,
               'train_minutes': train_min, 'total_minutes': (time.time() - t0) / 60}
        history.append(row)
        print('Epoch %d/%d  train acc %.2f%% | val acc %.2f%%  score argmax %.4f  pick %.4f | %.1f 分'
              % (epoch + 1, args.epochs, row['train_acc'], row['val_acc'],
                 row['val_score_argmax'], row['val_score_pick'], row['total_minutes']), flush=True)

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'epoch_%d.pth' % (epoch + 1)))
        if row['val_score_pick'] > best_score:
            best_score = row['val_score_pick']
            save_atomic(model.state_dict(), os.path.join(args.ckpt_dir, 'best_model.pth'))
            print('  更新 best_model.pth')
        pd.DataFrame(history).to_csv(os.path.join(args.ckpt_dir, 'history.csv'), index=False)
        # last.pt 最後寫，確保它記錄的 epoch 之前的檔案都已存好
        save_atomic({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': epoch + 1, 'history': history, 'best_score': best_score,
                     'config': config}, last_path)

    print('完成。最佳 val_score_pick %.4f' % best_score)


if __name__ == '__main__':
    main()
