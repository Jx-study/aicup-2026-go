"""目標玩家的顏色資訊：讓子局的顏色修正、黑白平面對調與 color plane。

HD>=2 的局 SGF 的 B/W 標記與實際相反，color 欄位是實際顏色，所以要先換成特徵平面上的標記
才能對調平面。對照表 hd.npz 由 v1.1 CSV 產生，分片本身沒有 HD。

用法（在 repo 根目錄）：
    python -m aicup.color --train-dir training-v1.1/training --out rank-features/hd.npz
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

MODES = ('none', 'swap', 'plane', 'both')
MIN_REVERSED_HD = 2  # HD[0]、HD[1] 標記正確
_HD_RE = re.compile(r'HD\[(\d+)\]')


def parse_hd(sgf_content):
    """回傳 SGF 的 HD[n]，沒有時回傳 None。"""
    m = _HD_RE.search(sgf_content)
    return None if m is None else int(m.group(1))


def feature_color(color, hd):
    """實際顏色（0 黑 1 白）換成特徵平面上的標記，讓子局黑白相反。"""
    return color ^ (np.asarray(hd) >= MIN_REVERSED_HD)


def in_channels(mode):
    return 501 if mode in ('plane', 'both') else 500


def apply_color(x, color, mode):
    """對 (B, 500, 19, 19) 套用顏色資訊，color 是 (B,) 的特徵顏色（0 黑 1 白）。

    swap、both 在 x 上原地對調，plane、both 回傳 501 channels 的新 tensor。
    """
    import torch

    if mode in ('swap', 'both'):
        white = color.bool()
        swapped = torch.arange(x.shape[1], device=x.device).view(-1, 2).flip(1).view(-1)
        x[white] = x[white][:, swapped]
    if mode in ('plane', 'both'):
        plane = (1 - color).to(x.dtype).view(-1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
        x = torch.cat([x, plane], 1)
    return x


def build_hd_map(train_dir, out):
    """從 v1.1 的 train_*.csv 寫出 hd.npz，欄位 key、hd，key 與分片相同。"""
    keys, hds = [], []
    for path in sorted(glob.glob(os.path.join(train_dir, 'train_*.csv'))):
        df = pd.read_csv(path, usecols=['rank', 'game_id', 'sgf_content'])
        for rank, game_id, sgf_content in zip(df['rank'], df['game_id'], df['sgf_content']):
            hd = parse_hd(sgf_content)
            if hd is None:
                raise SystemExit('%s 的 %s 沒有 HD，請用 v1.1 訓練集' % (path, game_id))
            keys.append('%s_%s' % (rank, game_id))
            hds.append(hd)
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:  # 傳檔案物件，避免 numpy 自動補 .npz 副檔名
        np.savez(f, key=np.array(keys, dtype=str), hd=np.array(hds, np.int8))
    os.replace(tmp, out)
    print('寫出 %s，%d 局' % (out, len(keys)))


def load_hd(path, keys):
    """回傳 keys 各局的 HD，檔案不存在或缺 key 時拒絕執行。"""
    if not os.path.exists(path):
        raise SystemExit('找不到 %s，用 python -m aicup.color 產生，或加 --raw-color' % path)
    with np.load(path, allow_pickle=False) as z:
        known, hd = z['key'], z['hd']
    order = np.argsort(known)
    known, hd = known[order], hd[order]
    pos = np.searchsorted(known, keys).clip(max=len(known) - 1)
    missing = known[pos] != keys
    if missing.any():
        raise SystemExit('%s 缺 %d 局，例如 %s' % (path, int(missing.sum()), keys[missing][0]))
    return hd[pos]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train-dir', default='training-v1.1/training')
    ap.add_argument('--out', default='rank-features/hd.npz')
    args = ap.parse_args()
    build_hd_map(args.train_dir, args.out)


if __name__ == '__main__':
    main()
