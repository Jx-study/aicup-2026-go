"""用 checkpoint 對 val 重新推論，輸出與 val_probs.npz 相同格式的機率檔，可開 TTA。

顏色設定讀 checkpoint 目錄的 train_config.json，和 predict_rank 相同。
輸出的 epoch 沿用同目錄 val_probs.npz 的值，沒有這個檔案時為 0。
預設輸出 <ckpt-dir>/val_probs_tta.npz（開 --tta）或 val_probs_eval.npz。

用法（在 repo 根目錄）：
    python -m aicup.eval_rank --ckpt-dir rank-trained-models/<commit hash> --tta
    python -m aicup.bootstrap_compare <ckpt-dir>/val_probs.npz <ckpt-dir>/val_probs_tta.npz
"""
import argparse
import os
import time

import numpy as np
import torch

from aicup import color, train_rank
from aicup.predict_rank import load_model, read_train_config


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--feat-dir', default='rank-features')
    ap.add_argument('--ckpt-dir', required=True, help='含 best_model.pth 的目錄')
    ap.add_argument('--out', help='輸出路徑，預設寫在 --ckpt-dir')
    ap.add_argument('--tta', action='store_true', help='平均 8 種對稱方向的機率')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto')
    ap.add_argument('--no-amp', action='store_true', help='關閉 bf16，改用純 fp32')
    args = ap.parse_args()

    best_path = os.path.join(args.ckpt_dir, 'best_model.pth')
    if not os.path.exists(best_path):
        raise SystemExit('找不到 %s' % best_path)
    out = args.out or os.path.join(args.ckpt_dir, 'val_probs_tta.npz' if args.tta else 'val_probs_eval.npz')
    ref = os.path.join(args.ckpt_dir, 'val_probs.npz')
    if os.path.abspath(out) == os.path.abspath(ref):
        raise SystemExit('--out 不能是訓練時寫的 %s，它是比較的基準' % ref)
    cfg = read_train_config(best_path)
    args.color, args.raw_color = cfg['color'], cfg['raw_color']  # load_colors 與 validate 讀這兩個欄位

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and train_rank.use_amp(device, args.no_amp)  # 同 predict_rank，CPU 不用 bf16
    torch.backends.cudnn.benchmark = True
    val_dir = os.path.join(args.feat_dir, 'val')
    data_device = train_rank.pick_data_device(args, device, train_rank.split_bytes(val_dir))
    x, y, meta = train_rank.load_to(val_dir, data_device)
    c = train_rank.load_colors(meta, args, data_device)
    model = load_model(best_path, args.channels, args.blocks, device, color.in_channels(args.color))
    print('device %s | data %s | amp %s | tta %s | color %s | raw_color %s'
          % (device, data_device, amp, args.tta, args.color, args.raw_color), flush=True)

    t0 = time.time()
    metrics, probs = train_rank.validate(model, x, y, args, device, amp, torch.nn.CrossEntropyLoss(), c=c,
                                         tta=args.tta)
    epoch = 0
    if os.path.exists(ref):
        with np.load(ref, allow_pickle=False) as z:
            epoch = int(z['epoch']) if 'epoch' in z.files else 0
    train_rank.save_val_probs(out, probs, meta, epoch)
    print('val_loss %.4f | val_acc %.2f%% | argmax %.4f | pick %.4f | %.1f 分'
          % (metrics['val_loss'], metrics['val_acc'], metrics['val_score_argmax'], metrics['val_score_pick'],
             (time.time() - t0) / 60))
    print('寫出 %s' % out)


if __name__ == '__main__':
    main()
