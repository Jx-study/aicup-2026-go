"""Colab 執行環境設定：安裝套件、把資料目錄接到 Google Drive、檢查 GPU

前提：Drive 已掛載、repo 已 clone 且目前目錄在 repo 根目錄。
執行方式：python -m aicup.colab_setup [--feat-dir rank-features-v180000]。可重複執行，已完成的步驟會跳過。
--feat-dir 另外連結一個 Drive 上已存在的特徵目錄，VRAM 檢查改算這個目錄，缺省是 rank-features。

Drive 目錄結構（MyDrive/aicup-2026/）：
    training/             官方訓練 CSV
    tests/                官方測試 CSV
    rank-features/        bit 打包的特徵分片（train/、val/ 下的 .npz）
    rank-trained-models/  checkpoint 與 history.csv
"""
import argparse
import os
import subprocess
import sys

DRIVE_ROOT = '/content/drive/MyDrive/aicup-2026'
REPO = os.getcwd()

MIN_VRAM_GB = 9  # 教學版 BATCH_SIZE=256 之訓練需求
GPU_MARGIN_GB = 14  # 與 train_rank.py 相同：資料之外留給訓練的 VRAM


def install_packages():
    missing = []
    for mod in ('sgfmill',):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print('安裝', missing)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
    else:
        print('套件已齊全')


def link(name, target):
    """把 repo/name 指向 target，已存在的連結先移除"""
    path = os.path.join(REPO, name)
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.exists(path):
        raise RuntimeError('%s 是實體目錄，請手動確認後刪除再重跑' % path)
    os.makedirs(target, exist_ok=True)
    os.symlink(target, path)
    print('%-20s -> %s' % (name, target))


def setup_dirs(feat_dir='rank-features'):
    if not os.path.isdir('/content/drive/MyDrive'):
        raise RuntimeError('Drive 尚未掛載')
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    # 特徵分片直接從 Drive 讀
    for name in ('training', 'tests', 'rank-features', 'rank-trained-models'):
        link(name, os.path.join(DRIVE_ROOT, name))
    if feat_dir != 'rank-features':
        target = os.path.join(DRIVE_ROOT, feat_dir)
        if not os.path.isdir(target):  # 不建立空目錄，避免訓練讀到空的特徵
            raise RuntimeError('Drive 上沒有 %s' % target)
        link(feat_dir, target)


def feature_gb(feat_dir='rank-features'):
    """依分片筆數估算打包特徵大小（GB），沒有分片時回傳 None"""
    try:
        import numpy as np
    except ImportError:
        return None
    from aicup import packed
    n = 0
    for sub in ('train', 'val'):
        for p in packed.list_shards(os.path.join(REPO, feat_dir, sub)):
            with np.load(p, allow_pickle=False) as z:
                n += len(z['rank_idx'])
    return n * packed.PACKED_LEN / 1024 ** 3 if n else None


def check_gpu(feat_dir='rank-features'):
    import torch
    if not torch.cuda.is_available():
        print('警告：沒有 GPU。到「執行階段 -> 變更執行階段類型」選 GPU')
        return
    prop = torch.cuda.get_device_properties(0)
    vram = prop.total_memory / 1024 ** 3
    print('GPU %s，VRAM %.1f GB，CPU %d 核' % (prop.name, vram, os.cpu_count()))
    if vram < MIN_VRAM_GB:
        print('警告：VRAM 小於 %d GB，BATCH_SIZE 需調小' % MIN_VRAM_GB)
    data = feature_gb(feat_dir)
    if data is None:
        print('Drive 上的 %s 還沒有特徵分片' % feat_dir)
    elif data + GPU_MARGIN_GB <= vram:
        print('%s 特徵 %.1f GB，可以整份放進 GPU' % (feat_dir, data))
    else:
        print('%s 特徵 %.1f GB，放不進 GPU，train_rank.py 會改放 CPU，速度較慢' % (feat_dir, data))


def summarize():
    for name in ('training', 'tests'):
        files = sorted(os.listdir(os.path.join(REPO, name)))
        print('%-9s %d 個檔案 %s' % (name, len(files), files[:3]))
        if not files:
            print('警告：Drive 上的 %s/ 是空的，資料還沒上傳' % name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--feat-dir', default='rank-features', help='Drive 上的特徵目錄名稱，缺省 rank-features')
    args = ap.parse_args()
    install_packages()
    setup_dirs(args.feat_dir)
    check_gpu(args.feat_dir)
    summarize()


if __name__ == '__main__':
    main()
