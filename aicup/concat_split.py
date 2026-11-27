"""把一個驗證切分接在序列檔與局部盤面檔的 train 後面，寫到新目錄，供序列模型多加訓練玩家。

新目錄的 train.npz、train_patch.npz 是 train 接上 --extra 切分（每個陣列依第 0 維串接，純量欄位沿用 train），
其他檔案以 symlink 指回原目錄，所以 newdev、val648、hold 的評估與推論照舊。

用法（在 repo 根目錄）：
    python -m aicup.concat_split --seq-dir move-seq-state-v180000 --patch-dir move-seq-patch-v180000 \\
        --extra val648 --out-seq /content/seq-val648 --out-patch /content/patch-val648 --expect 918278
"""
import argparse
import os

import numpy as np


def concat_npz(first, second):
    """回傳 {欄位: 陣列}：first 與 second 的同名陣列依第 0 維串接，0 維欄位取 first 的值。"""
    with np.load(first) as a, np.load(second) as b:
        if set(a.files) != set(b.files):
            raise ValueError('%s 與 %s 的欄位不同' % (first, second))
        return {k: (np.concatenate([a[k], b[k]]) if a[k].ndim else a[k]) for k in a.files}


def build_dir(src, out, train_name, extra_name):
    """在 out 寫出 train_name 串接 extra_name 的結果，src 的其他檔案以 symlink 放進 out，回傳串接後的局數。"""
    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(src)):
        dst = os.path.join(out, name)
        if name == train_name or os.path.lexists(dst):
            continue
        os.symlink(os.path.abspath(os.path.join(src, name)), dst)
    merged = concat_npz(os.path.join(src, train_name), os.path.join(src, extra_name))
    tmp = os.path.join(out, train_name + '.tmp')
    with open(tmp, 'wb') as f:  # 寫完才改名，中斷時不會留下不完整的 train 檔
        np.savez(f, **merged)
    os.replace(tmp, os.path.join(out, train_name))
    return len(merged['key'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seq-dir', required=True)
    ap.add_argument('--patch-dir', required=True)
    ap.add_argument('--extra', required=True, help='接在 train 後面的切分名稱，例如 val648')
    ap.add_argument('--out-seq', required=True)
    ap.add_argument('--out-patch', required=True)
    ap.add_argument('--expect', type=int, default=0, help='串接後應有的局數，0 為不檢查')
    args = ap.parse_args()

    n_seq = build_dir(args.seq_dir, args.out_seq, 'train.npz', args.extra + '.npz')
    n_patch = build_dir(args.patch_dir, args.out_patch, 'train_patch.npz', args.extra + '_patch.npz')
    if n_seq != n_patch:
        raise SystemExit('序列 %d 局、局部盤面 %d 局，數量不同' % (n_seq, n_patch))
    if args.expect and n_seq != args.expect:
        raise SystemExit('串接後 %d 局，預期 %d 局' % (n_seq, args.expect))
    print('串接完成：%d 局 -> %s、%s' % (n_seq, args.out_seq, args.out_patch))


if __name__ == '__main__':
    main()
