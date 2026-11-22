"""依既有落子序列檔的 key 順序，打包每手落子前的局部盤面，供序列模型的局部盤面編碼器使用。

每手以落子點為中心取 patch_size x patch_size 格（空、己、敵、盤外，相對落子方），每格 2 bit。
輸出 {split}_patch.npz，含 patch (N, L, patch_nbytes) uint8、key 與 patch_size，key 順序與
--seq-dir 內同名序列檔完全相同。寫入時經磁碟 memmap，避免整份局部盤面同時放在記憶體。

用法（在 repo 根目錄）：
    python -m aicup.pack_move_patch --seq-dir move-seq-state-v180000 --out-dir move-seq-patch-v180000
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from aicup.pack_move_seq import DEFAULT_PATCH_SIZE, parse_sgf_batch, patch_nbytes

SPLITS = ('train', 'newdev', 'val648')
BATCH_GAMES = 20000


def _flush(buf, mmaps, ref_pos, patch_size, max_moves, raw_color, nproc):
    """解析一批 (split, idx, sgf, color_id)，比對 pos 後寫入對應 memmap。"""
    if not buf:
        return
    res = parse_sgf_batch(
        [b[2] for b in buf], [b[3] for b in buf], max_moves=max_moves, raw_color=raw_color,
        nproc=nproc, patch_size=patch_size,
    )
    pos, patch = res.pos, res.patch
    for j, (split, idx, _, _) in enumerate(buf):
        if not np.array_equal(pos[j], ref_pos[split][idx]):
            raise ValueError('%s 第 %d 局重新解析的 pos 與序列檔不同' % (split, idx))
        mmaps[split][idx] = patch[j]
    buf.clear()


def pack_patches(seq_dir, train_dir, out_dir, patch_size=DEFAULT_PATCH_SIZE, splits=SPLITS, nproc=1):
    """為 seq_dir 內各 split 序列檔產生局部盤面檔，回傳 {split: 局數}。"""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    nb = patch_nbytes(patch_size)
    key_to_loc, ref_pos, mmaps, keys_by_split, filled = {}, {}, {}, {}, {}
    max_moves = raw_color = None
    for split in splits:
        with np.load(os.path.join(seq_dir, '%s.npz' % split), allow_pickle=False) as z:
            keys = z['key']
            ref_pos[split] = z['pos']
            colors = z['color']
            split_raw = bool(z['raw_color']) if 'raw_color' in z.files else False
        split_moves = ref_pos[split].shape[1]
        # 同一批解析共用 max_moves 與 raw_color，各 split 必須一致
        if max_moves is not None and (split_moves, split_raw) != (max_moves, raw_color):
            raise ValueError('%s 的 max_moves 或 raw_color 與其他 split 不同' % split)
        max_moves, raw_color = split_moves, split_raw
        keys_by_split[split] = keys
        for i, (k, col) in enumerate(zip(keys.tolist(), colors.tolist())):
            key_to_loc[str(k)] = (split, i, int(col))
        mmaps[split] = np.lib.format.open_memmap(
            os.path.join(out_dir, '%s_patch.tmp.npy' % split), mode='w+', dtype=np.uint8,
            shape=(len(keys), max_moves, nb),
        )
        filled[split] = 0

    csv_paths = sorted(glob.glob(os.path.join(train_dir, 'train_*.csv')))
    if not csv_paths:
        raise SystemExit('%s 下找不到任何 train_*.csv' % train_dir)
    buf = []
    for path in csv_paths:
        for chunk in pd.read_csv(path, usecols=['rank', 'game_id', 'sgf_content'], chunksize=50000):
            for rk, gid, sgf_s in zip(chunk['rank'].astype(str), chunk['game_id'].astype(str), chunk['sgf_content']):
                loc = key_to_loc.get('%s_%s' % (rk, gid))
                if loc is None:
                    continue
                buf.append((loc[0], loc[1], str(sgf_s), loc[2]))
                filled[loc[0]] += 1
                if len(buf) >= BATCH_GAMES:
                    _flush(buf, mmaps, ref_pos, patch_size, max_moves, raw_color, nproc)
        print('  %s 完成（%.0f 秒）' % (os.path.basename(path), time.time() - t0), flush=True)
    _flush(buf, mmaps, ref_pos, patch_size, max_moves, raw_color, nproc)

    summary = {}
    for split in splits:
        n = len(keys_by_split[split])
        if filled[split] != n:
            raise ValueError('%s 只找到 %d / %d 局的 SGF' % (split, filled[split], n))
        mmaps[split].flush()
        out_path = os.path.join(out_dir, '%s_patch.npz' % split)
        tmp = out_path + '.tmp'
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, patch=mmaps[split], key=keys_by_split[split], patch_size=np.int64(patch_size))
        os.replace(tmp, out_path)
        tmp_npy = mmaps[split].filename
        del mmaps[split]
        os.remove(tmp_npy)
        summary[split] = n
        print('%s：%d 局，%.1f MB，累計 %.0f 秒'
              % (split, n, os.path.getsize(out_path) / 1024 ** 2, time.time() - t0), flush=True)
    return summary


def load_patch_npz(path, keys):
    """載入局部盤面檔並檢查 key 順序與序列檔相同，回傳 (patch uint8 ndarray, patch_size)。"""
    if not os.path.exists(path):
        raise SystemExit('找不到局部盤面檔：%s' % path)
    with np.load(path, allow_pickle=False) as z:
        if len(z['key']) != len(keys) or not (z['key'] == np.asarray(keys)).all():
            raise SystemExit('%s 的 key 順序與序列檔不同' % path)
        return z['patch'], int(z['patch_size'])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seq-dir', default='move-seq-state-v180000')
    ap.add_argument('--train-dir', default=os.path.join('training-v1.1', 'training'))
    ap.add_argument('--out-dir', default='move-seq-patch-v180000')
    ap.add_argument('--patch-size', type=int, default=DEFAULT_PATCH_SIZE)
    ap.add_argument('--splits', nargs='+', default=list(SPLITS))
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    if args.patch_size < 3 or args.patch_size % 2 == 0:
        ap.error('--patch-size 要是 >= 3 的奇數')
    return args


def main(argv=None):
    args = parse_args(argv)
    pack_patches(args.seq_dir, args.train_dir, args.out_dir, patch_size=args.patch_size,
                 splits=args.splits, nproc=args.nproc)


if __name__ == '__main__':
    main()
