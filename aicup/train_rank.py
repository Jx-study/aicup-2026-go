"""棋力預測訓練：整份打包資料放進 GPU 記憶體，每個 batch 在 GPU 上解包。

模型與預設超參數同 rank-prediction.ipynb。每個 epoch 存 last.pt，重跑自動接續。
best_model.pth 依 val_score_pick 挑選，只存 state_dict；同時存 val_probs.npz 供事後分析。
--soft-label 以有序 soft label 訓練，val_loss 仍是 hard label 的 NLL。
--ema 追蹤權重的 EMA，best_model.pth 與 val_probs.npz 改用 EMA 權重挑選，原始權重的指標記成 raw_ 欄位，最後一個 epoch 另用 train 資料重算 EMA 的 BN 統計量，指標記成 bn_ 欄位，權重存成 final_ema_bn.pth。
--patience 預設關閉；啟用時跑滿 --min-epochs 後，val_score_pick 連續 --patience 個 epoch 沒超過最佳就提前停止。
每次啟動在 runs.jsonl 追加一筆指令與環境，接續訓練時也會記錄。

用法（在 repo 根目錄）：
    python -m aicup.train_rank
    python -m aicup.train_rank --epochs 10   # 已跑完 5 個 epoch 時接續到第 10 個
    python -m aicup.train_rank --schedule cosine --epochs 15   # warmup 後 cosine 降到 0
    python -m aicup.train_rank --augment      # 每個 batch 隨機套 8 種盤面對稱之一
    python -m aicup.train_rank --color both   # 目標玩家的顏色資訊，需要 <feat-dir>/hd.npz
    python -m aicup.train_rank --soft-label 1 --ema 0.999   # 有序 soft label 與 EMA
    python -m aicup.train_rank --patience 3   # 跑滿 --min-epochs 後，連續 3 個 epoch 沒超過最佳就停
    python -m aicup.train_rank --fresh       # 忽略既有 checkpoint 從頭訓練
"""
import argparse
import copy
import datetime
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from aicup import color, packed, symmetry
from aicup.scoring import LEVELS, pick_level, score_strength
from network import GoRankResNet

NUM_CLASSES = len(LEVELS)
GPU_MARGIN_GB = 14  # 資料之外留給模型訓練的 VRAM，教學版設定約需 9 GB
EPS = 1e-12  # log 前的下限，與 scoring.aggregate_games 一致


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--feat-dir', default='rank-features')
    ap.add_argument('--ckpt-dir', default='rank-trained-models')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--schedule', choices=('constant', 'cosine'), default='constant',
                    help='cosine 在 --warmup-steps 內線性升到 --lr，之後依總步數降到 0')
    ap.add_argument('--warmup-steps', type=int, default=1000)
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto',
                    help='打包資料放哪裡；auto 在 VRAM 足夠時放 GPU')
    ap.add_argument('--no-amp', action='store_true', help='關閉 bf16，改用純 fp32')
    ap.add_argument('--augment', action='store_true', help='每個 batch 隨機套 8 種盤面對稱之一')
    ap.add_argument('--color', choices=color.MODES, default='none',
                    help='swap 依目標玩家對調黑白平面，plane 加 color plane，both 兩者都做')
    ap.add_argument('--raw-color', action='store_true',
                    help='不修正讓子局，直接用 color 欄位，不需要 hd.npz')
    ap.add_argument('--soft-label', type=float, default=0.0, metavar='TAU',
                    help='目標正比於 exp(-|k - y| / TAU)，0 為 one-hot')
    ap.add_argument('--ema', type=float, default=0.0, help='EMA 的 decay，0 為不追蹤')
    ap.add_argument('--bn-batches', type=int, default=200,
                    help='最後一個 epoch 重算 EMA 的 BN 統計量用幾個 train batch，0 為不重算')
    ap.add_argument('--patience', type=int, default=0,
                    help='val_score_pick 連續幾個 epoch 沒超過最佳就停，0 為不停')
    ap.add_argument('--min-epochs', type=int, default=6,
                    help='跑滿幾個 epoch 才檢查 --patience，lr 高的前期 val 跳動大')
    ap.add_argument('--fresh', action='store_true', help='忽略既有 last.pt')
    ap.add_argument('--max-steps', type=int, default=0, help='每個 epoch 最多幾步，0 為不限，用於冒煙測試')
    args = ap.parse_args()
    if args.soft_label < 0 or args.bn_batches < 0 or not 0 <= args.ema < 1:
        ap.error('--soft-label、--bn-batches 要 >= 0，--ema 要在 [0, 1)')
    return args


def load_to(split_dir, data_device):
    """載入一個 split，打包特徵與標籤搬到 data_device，特徵的 numpy 副本隨即釋放。

    回傳 (x, y, meta)，meta 是 key、player_id 等其餘欄位的 numpy 陣列，順序和 x 相同。
    """
    t0 = time.time()
    d = packed.load_split(split_dir)
    x = torch.from_numpy(d.pop('features')).to(data_device)
    y = torch.from_numpy(d['rank_idx'].astype(np.int64)).to(data_device)
    print('  %s：%s 筆，%.1f GB，%.1f 秒' % (split_dir, format(len(y), ','),
                                          x.numel() / 1024 ** 3, time.time() - t0), flush=True)
    return x, y, d


def load_colors(meta, args, data_device):
    """回傳一個 split 的特徵顏色 tensor，--color none 時回傳 None。"""
    if args.color == 'none':
        return None
    c = meta['color']
    if not args.raw_color:
        c = color.feature_color(c, color.load_hd(os.path.join(args.feat_dir, 'hd.npz'), meta['key']))
    return torch.from_numpy(c.astype(np.int64)).to(data_device)


def use_amp(device, no_amp):
    """是否開 bf16；GPU 上只算硬體支援，不算模擬。"""
    if no_amp:
        return False
    return device.type == 'cpu' or torch.cuda.is_bf16_supported(including_emulation=False)


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


def batches(x, y, batch_size, device, order=None, c=None, mode='none'):
    """依 order（預設為原順序，可只取一部分）切 batch，在 device 上解包成 float，c 不為 None 時套用顏色資訊。"""
    n = len(y) if order is None else len(order)
    for i in range(0, n, batch_size):
        idx = slice(i, i + batch_size) if order is None else order[i:i + batch_size]
        xb = packed.unpack_torch(x[idx].to(device, non_blocking=True))
        if c is not None:
            xb = color.apply_color(xb, c[idx].to(device, non_blocking=True), mode)
        yield xb, y[idx].to(device, non_blocking=True)


def lr_at(step, total_steps, base_lr, schedule, warmup_steps):
    """第 step 步（從 0 起算）的 lr；只由步數決定，接續訓練不需要另存 scheduler 狀態。"""
    if schedule == 'constant':
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def soft_label_table(tau, n=NUM_CLASSES):
    """回傳 (n, n) 目標分布，第 y 列正比於 exp(-|k - y| / tau)；tau 為 0 時是 one-hot。"""
    if tau <= 0:
        return torch.eye(n)
    k = torch.arange(n, dtype=torch.float64)
    q = torch.exp(-(k[None] - k[:, None]).abs() / tau)
    return (q / q.sum(1, keepdim=True)).float()


def make_ema(model, decay):
    """回傳追蹤 model 的 EMA，BN 的 running mean/var 也一起平均。"""
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)


def recompute_bn(model, x, y, args, device, amp, c=None):
    """回傳 model 的副本，BN 統計量以 train 中隨機 args.bn_batches 個 batch 的累積平均重算。"""
    bn = copy.deepcopy(model)
    norms = [m for m in bn.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # None 為累積平均，每個 batch 權重相同
    gen = torch.Generator(device=x.device).manual_seed(args.seed)
    order = torch.randperm(len(y), generator=gen, device=x.device)[:args.bn_batches * args.batch_size]
    bn.train()
    with torch.no_grad():
        for xb, _ in batches(x, y, args.batch_size, device, order, c, args.color):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                bn(xb)
    for m in norms:
        m.momentum = 0.1  # 還原 BatchNorm2d 預設值
    return bn


def patience_exhausted(history, patience, min_epochs=0):
    """history 至少有 min_epochs 筆，且最佳 epoch 之後已過 patience 個 epoch 時回傳 True。

    結果由 history 推得，接續訓練時同樣成立。
    """
    if patience <= 0 or len(history) < max(min_epochs, 1):
        return False
    scores = [r['val_score_pick'] for r in history]
    return len(scores) - 1 - int(np.argmax(scores)) >= patience


def log_run(ckpt_dir, start_epoch, device):
    """在 ckpt_dir/runs.jsonl 追加一筆這次執行的時間、commit、參數、GPU 與起始 epoch。"""
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
    run = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
           'commit': git.stdout.strip() or 'unknown', 'argv': sys.argv[1:],
           'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
           'start_epoch': start_epoch}
    with open(os.path.join(ckpt_dir, 'runs.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(run, ensure_ascii=False) + '\n')


def forward_probs(model, x, amp, tta=False):
    """回傳等級機率 (B, 10)；tta 時平均 8 種對稱方向的機率。"""
    views = range(symmetry.N_SYMMETRIES) if tta else (0,)
    total = 0
    for k in views:
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(symmetry.apply(x, k).contiguous() if k else x)
        total = total + torch.softmax(out.float(), 1)
    return total / len(views)


def validate(model, x, y, args, device, amp, criterion, c=None, tta=False):
    """回傳 (指標 dict, 逐局機率 (N, 10) float32)；tta 時機率與 val_loss 都用 8 種對稱方向的平均機率。"""
    model.eval()
    loss_sum, batches_n, probs = 0.0, 0, []
    with torch.no_grad():
        for xb, yb in batches(x, y, args.batch_size, device, c=c, mode=args.color):
            p = forward_probs(model, xb, amp, tta)
            # 機率總和為 1，對 log p 取 cross entropy 等於 p 的 NLL
            loss_sum += criterion(torch.log(p.clamp_min(EPS)), yb).item()
            batches_n += 1
            probs.append(p.cpu().numpy())
    model.train()
    probs, labels = np.concatenate(probs), y.cpu().numpy()
    metrics = {
        'val_loss': loss_sum / max(batches_n, 1),
        'val_acc': float((probs.argmax(1) == labels).mean() * 100),
        'val_score_argmax': score_strength(labels, probs.argmax(1)),
        'val_score_pick': score_strength(labels, pick_level(probs)),
        'val_n': len(labels),
    }
    return metrics, probs.astype(np.float32)


def save_atomic(obj, path):
    """先寫暫存檔再改名，存到一半斷線不會毀掉上一份 checkpoint。"""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_val_probs(path, probs, meta, epoch):
    """寫出 val 逐局機率與對應的 rank_idx、key、player_id，同樣先寫暫存檔。

    epoch 是產生這份機率的 epoch（從 1 起算），和 best_model.pth 對應。
    """
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:  # 傳檔案物件，避免 numpy 自動補 .npz 副檔名
        np.savez(f, probs=probs, rank_idx=meta['rank_idx'], key=meta['key'],
                 player_id=meta['player_id'], epoch=np.int64(epoch))
    os.replace(tmp, path)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = use_amp(device, args.no_amp)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dir = os.path.join(args.feat_dir, 'train')
    val_dir = os.path.join(args.feat_dir, 'val')
    data_device = pick_data_device(args, device, split_bytes(train_dir) + split_bytes(val_dir))
    print('載入資料到 %s' % data_device, flush=True)
    x_val, y_val, val_meta = load_to(val_dir, data_device)
    c_val = load_colors(val_meta, args, data_device)
    x_train, y_train, train_meta = load_to(train_dir, data_device)
    c_train = load_colors(train_meta, args, data_device)

    in_channels = color.in_channels(args.color)
    model = GoRankResNet(in_channels=in_channels, num_channels=args.channels, num_blocks=args.blocks,
                         num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    soft = soft_label_table(args.soft_label).to(device) if args.soft_label > 0 else None
    ema = make_ema(model, args.ema) if args.ema > 0 else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, 'last.pt')
    config = {k: getattr(args, k) for k in ('batch_size', 'lr', 'channels', 'blocks', 'seed')}
    config['amp'] = amp
    if args.schedule != 'constant':  # constant 不加欄位，舊的 last.pt 仍可接續
        config.update(schedule=args.schedule, warmup_steps=args.warmup_steps, epochs=args.epochs)
    if args.augment:
        config['augment'] = True
    if args.color != 'none':
        config.update(color=args.color, raw_color=args.raw_color)
    if soft is not None:
        config['soft_label'] = args.soft_label
    if ema is not None:
        config['ema'] = args.ema
    start_epoch, history, best_score = 0, [], -1.0
    if os.path.exists(last_path) and not args.fresh:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt['config'] != config:
            raise SystemExit('last.pt 的設定與這次不同：\n  舊 %s\n  新 %s\n'
                             '要用新設定請加 --fresh 或換 --ckpt-dir' % (ckpt['config'], config))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema is not None:
            ema.load_state_dict(ckpt['ema'])
        start_epoch, history, best_score = ckpt['epoch'], ckpt['history'], ckpt['best_score']
        print('從 epoch %d 接續（目前最佳 val_score_pick %.4f）' % (start_epoch, best_score))

    # 通過設定比對才寫，被拒絕的執行不能改掉既有 checkpoint 的設定
    with open(os.path.join(args.ckpt_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'color': args.color, 'raw_color': args.raw_color}, f)  # predict_rank 依此設定輸入

    best_path = os.path.join(args.ckpt_dir, 'best_model.pth')
    probs_path = os.path.join(args.ckpt_dir, 'val_probs.npz')
    if history and os.path.exists(best_path) and not os.path.exists(probs_path):
        # 缺 val_probs.npz 時用 best_model.pth 補寫
        best = GoRankResNet(in_channels=in_channels, num_channels=args.channels, num_blocks=args.blocks,
                            num_classes=NUM_CLASSES).to(device)
        best.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        _, probs = validate(best, x_val, y_val, args, device, amp, criterion, c_val)
        best_epoch = max(history, key=lambda r: r['val_score_pick'])['epoch']
        save_val_probs(probs_path, probs, val_meta, best_epoch)
        print('已用 best_model.pth（epoch %d）補寫 val_probs.npz' % best_epoch)
        del best

    n_params = sum(p.numel() for p in model.parameters())
    print('device %s | data %s | amp %s | train %s / val %s | 參數量 %s'
          % (device, data_device, amp, format(len(y_train), ','), format(len(y_val), ','),
             format(n_params, ',')))
    print('設定', json.dumps(config, ensure_ascii=False))
    if start_epoch >= args.epochs:
        print('已完成 %d 個 epoch，要繼續訓練請加大 --epochs' % start_epoch)
        return
    if patience_exhausted(history, args.patience, args.min_epochs):
        print('已在 epoch %d 提前停止（patience %d），要繼續訓練請加 --patience 0' % (start_epoch, args.patience))
        return

    log_run(args.ckpt_dir, start_epoch, device)
    steps_per_epoch = math.ceil(len(y_train) / args.batch_size)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = steps_per_epoch * args.epochs

    model.train()
    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)  # dropout 的亂數也隨 epoch 固定
        gen = torch.Generator(device=data_device).manual_seed(args.seed + epoch)
        order = torch.randperm(len(y_train), generator=gen, device=data_device)
        aug_gen = torch.Generator().manual_seed(args.seed + epoch)  # 對稱的選擇只由 seed 與 epoch 決定
        t0 = time.time()
        loss_sum, correct, total, steps = 0.0, 0, 0, 0
        for xb, yb in batches(x_train, y_train, args.batch_size, device, order, c_train, args.color):
            lr = lr_at(epoch * steps_per_epoch + steps, total_steps, args.lr, args.schedule,
                       args.warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            if args.augment:
                k = int(torch.randint(symmetry.N_SYMMETRIES, (1,), generator=aug_gen))
                xb = symmetry.apply(xb, k).contiguous()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(xb)
            loss = criterion(out.float(), yb if soft is None else soft[yb])
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update_parameters(model)

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
        metrics, val_probs = validate(model, x_val, y_val, args, device, amp, criterion, c_val)
        raw = {}
        if ema is not None:  # val_ 欄位與 best_model.pth 都用 EMA 權重
            raw = {'raw_' + k: v for k, v in metrics.items() if k != 'val_n'}
            metrics, val_probs = validate(ema.module, x_val, y_val, args, device, amp, criterion, c_val)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / max(steps, 1),
               'train_acc': 100 * correct / max(total, 1), 'lr': lr, **metrics, **raw,
               'train_minutes': train_min, 'total_minutes': (time.time() - t0) / 60}
        history.append(row)
        stop = patience_exhausted(history, args.patience, args.min_epochs)
        if ema is not None and args.bn_batches and (stop or epoch + 1 == args.epochs):
            bn = recompute_bn(ema.module, x_train, y_train, args, device, amp, c_train)
            bn_metrics, _ = validate(bn, x_val, y_val, args, device, amp, criterion, c_val)
            row.update({'bn_' + k: v for k, v in bn_metrics.items() if k != 'val_n'})
            save_atomic(bn.state_dict(), os.path.join(args.ckpt_dir, 'final_ema_bn.pth'))
            print('  重算 BN 的 EMA：val pick %.4f' % row['bn_val_score_pick'])
            del bn
        print('Epoch %d/%d  train acc %.2f%% | val acc %.2f%%  score argmax %.4f  pick %.4f | %.1f 分'
              % (epoch + 1, args.epochs, row['train_acc'], row['val_acc'],
                 row['val_score_argmax'], row['val_score_pick'], row['total_minutes']), flush=True)

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'epoch_%d.pth' % (epoch + 1)))
        selected = model if ema is None else ema.module  # module 去掉 AveragedModel 外層與 n_averaged
        if row['val_score_pick'] > best_score:
            best_score = row['val_score_pick']
            save_val_probs(probs_path, val_probs, val_meta, epoch + 1)
            save_atomic(selected.state_dict(), best_path)
            print('  更新 best_model.pth')
        pd.DataFrame(history).to_csv(os.path.join(args.ckpt_dir, 'history.csv'), index=False)
        # last.pt 最後寫，確保它記錄的 epoch 之前的檔案都已存好
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1, 'history': history, 'best_score': best_score, 'config': config}
        if ema is not None:
            state['ema'] = ema.state_dict()
        save_atomic(state, last_path)
        if stop:
            print('val_score_pick 連續 %d 個 epoch 沒超過最佳，提前停止' % args.patience)
            break

    print('完成。最佳 val_score_pick %.4f' % best_score)


if __name__ == '__main__':
    main()
