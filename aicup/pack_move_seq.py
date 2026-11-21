"""打包整局落子座標序列為緊湊 .npz，供序列 Transformer 風格編碼器訓練與推論。

沿用既有 bit-packed 特徵目錄的 train/val 玩家切分，從 v1.1 訓練 CSV 解析每局前 L 手
落子座標與目標方標記（含 HD >= 2 讓子局執色反轉與 pass 處理），輸出 train、newdev、
val648 三份單檔 .npz，並驗證訓練集與驗證集玩家嚴格互斥。

用法（在 repo 根目錄）：
    python -m aicup.pack_move_seq --train-dir training-v1.1/training \\
        --feat-dir rank-features-v180000 --out-dir move-seq-features-v180000
"""
import argparse
import csv
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from aicup import color, packed
from aicup.extract_features import COLOR_TO_ID, RANK_TO_ID

BOARD_SIZE = 19
MAX_MOVES = 320
PASS_POS = BOARD_SIZE * BOARD_SIZE  # 361
PAD_POS = PASS_POS + 1  # 362
NUM_POS = PAD_POS + 1  # 363

ROLE_OPP = 0
ROLE_TARGET = 1
ROLE_PAD = 2
NUM_ROLES = 3

DEFAULT_NEWDEV_CSV = os.path.join('humansl-analysis', 'val-players-newdev-v180000.csv')
DEFAULT_VAL648_CSV = os.path.join('humansl-analysis', 'val-players.csv')


def build_d4_pos_table():
    """建立 D4 8 種盤面對稱之座標置換表，形狀 (8, 363) uint16。

    前 361 格對應 19x19 棋盤位置（r * 19 + c），與 symmetry.apply 方向一致；
    PASS_POS 與 PAD_POS 維持不變。
    """
    table = np.empty((8, NUM_POS), dtype=np.uint16)
    for k in range(8):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                nr, nc = r, c
                for _ in range(k % 4):
                    nr, nc = BOARD_SIZE - 1 - nc, nr
                if k >= 4:
                    nc = BOARD_SIZE - 1 - nc
                table[k, r * BOARD_SIZE + c] = nr * BOARD_SIZE + nc
        table[k, PASS_POS] = PASS_POS
        table[k, PAD_POS] = PAD_POS
    return table


D4_POS_TABLE = build_d4_pos_table()


def parse_sgf_move_seq(sgf_content, actual_color_id, max_moves=MAX_MOVES, raw_color=False):
    """解析單局 SGF 為定長落子序列，回傳 (pos, is_target, length, hd, ok)。

    actual_color_id 為目標方實際執色（0 黑 1 白）；HD >= 2 時依 color.feature_color
    反轉 SGF 內 B/W 標記。解析失敗或無任何落子時 ok 為 False。
    """
    pos = np.full(max_moves, PAD_POS, dtype=np.uint16)
    is_target = np.full(max_moves, ROLE_PAD, dtype=np.uint8)
    if not isinstance(sgf_content, str) or not sgf_content.strip():
        return pos, is_target, 0, 0, False

    hd_val = color.parse_hd(sgf_content)
    hd = 0 if hd_val is None else int(hd_val)
    target_sgf_col = int(actual_color_id) if raw_color else int(color.feature_color(int(actual_color_id), hd))

    try:
        from sgfmill import sgf  # 延遲載入，避免主行程重複匯入開銷
        game = sgf.Sgf_game.from_string(sgf_content)
        nodes = game.get_main_sequence()
    except Exception:
        return pos, is_target, 0, hd, False

    idx = 0
    for node in nodes[1:]:
        col, mv = node.get_move()
        if col is None:
            continue
        col_l = col.lower()
        if col_l == 'b':
            move_col_id = 0
        elif col_l == 'w':
            move_col_id = 1
        else:
            continue

        if mv is None:
            p = PASS_POS
        else:
            r, c = mv
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                p = r * BOARD_SIZE + c
            else:
                p = PASS_POS

        pos[idx] = p
        is_target[idx] = ROLE_TARGET if move_col_id == target_sgf_col else ROLE_OPP
        idx += 1
        if idx >= max_moves:
            break

    return pos, is_target, idx, hd, idx > 0


def _parse_chunk_worker(args):
    """子進程批次解析落子序列，頂層函式供 multiprocessing 使用。"""
    rows, max_moves, raw_color = args
    n = len(rows)
    pos_arr = np.full((n, max_moves), PAD_POS, dtype=np.uint16)
    role_arr = np.full((n, max_moves), ROLE_PAD, dtype=np.uint8)
    len_arr = np.zeros(n, dtype=np.uint16)
    hd_arr = np.zeros(n, dtype=np.int8)
    ok_arr = np.zeros(n, dtype=bool)

    for i, (sgf_content, col_id) in enumerate(rows):
        p, r, length, hd, ok = parse_sgf_move_seq(sgf_content, col_id, max_moves=max_moves, raw_color=raw_color)
        pos_arr[i] = p
        role_arr[i] = r
        len_arr[i] = length
        hd_arr[i] = min(max(hd, -128), 127)
        ok_arr[i] = ok
    return pos_arr, role_arr, len_arr, hd_arr, ok_arr


def parse_sgf_batch(sgfs, color_ids, max_moves=MAX_MOVES, raw_color=False, nproc=1):
    """批次解析多局 SGF，回傳 (pos, is_target, length, hd, valid)。"""
    n = len(sgfs)
    if n == 0:
        return (
            np.zeros((0, max_moves), dtype=np.uint16),
            np.zeros((0, max_moves), dtype=np.uint8),
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=bool),
        )
    pairs = list(zip(sgfs, color_ids))
    eff_proc = max(1, min(int(nproc), n))
    if eff_proc == 1:
        return _parse_chunk_worker((pairs, max_moves, raw_color))

    chunk_size = (n + eff_proc * 4 - 1) // (eff_proc * 4)
    chunks = [(pairs[i:i + chunk_size], max_moves, raw_color) for i in range(0, n, chunk_size)]
    with Pool(eff_proc) as pool:
        parts = pool.map(_parse_chunk_worker, chunks)
    return tuple(np.concatenate([p[k] for p in parts], axis=0) for k in range(5))


def read_player_ids_csv(path):
    """讀取含 player_id 欄位的 CSV，回傳玩家集合。"""
    if not os.path.exists(path):
        raise SystemExit('找不到玩家清單：%s' % path)
    pids = set()
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pid = str(row['player_id']).strip()
            if pid:
                pids.add(pid)
    return pids


def load_split_keys(split_dir):
    """讀取 bit-packed 分片目錄內所有對局 key 集合。"""
    paths = packed.list_shards(split_dir)
    if not paths:
        raise SystemExit('%s 下沒有 .npz 分片' % split_dir)
    metas = packed.load_meta(paths, fields=['key'])
    keys = set()
    for m in metas:
        for k in m['key'].tolist():
            keys.add(str(k))
    return keys


def save_seq_npz(path, pos, is_target, length, key, player_id, rank_idx, col, hd, raw_color=False):
    """以暫存檔原子寫出單一落子序列 .npz。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez_compressed(
            f,
            pos=pos.astype(np.uint16),
            is_target=is_target.astype(np.uint8),
            length=length.astype(np.uint16),
            key=np.asarray(key, dtype=str),
            player_id=np.asarray(player_id, dtype=str),
            rank_idx=np.asarray(rank_idx, dtype=np.int8),
            color=np.asarray(col, dtype=np.uint8),
            hd=np.asarray(hd, dtype=np.int8),
            raw_color=np.bool_(raw_color),
        )
    os.replace(tmp, path)


def pack_splits(
    train_dir,
    feat_dir,
    out_dir,
    newdev_csv=DEFAULT_NEWDEV_CSV,
    val648_csv=DEFAULT_VAL648_CSV,
    max_moves=MAX_MOVES,
    nproc=1,
    raw_color=False,
):
    """依 feat_dir 的 train/val 切分與玩家清單，打包 train、newdev、val648 三份序列檔。"""
    t0 = time.time()
    train_keys = load_split_keys(os.path.join(feat_dir, 'train'))
    val_keys = load_split_keys(os.path.join(feat_dir, 'val'))
    newdev_pids = read_player_ids_csv(newdev_csv)
    val648_pids = read_player_ids_csv(val648_csv)

    if newdev_pids & val648_pids:
        raise ValueError('newdev 與 val648 玩家有交集：%d 位' % len(newdev_pids & val648_pids))

    # 逐檔解析並立即釋放原始 SGF 字串，降低記憶體峰值
    parts = {
        s: {'pos': [], 'role': [], 'len': [], 'hd': [], 'key': [], 'pid': [], 'rk': [], 'col': [], 'drop': 0}
        for s in ('train', 'newdev', 'val648')
    }
    train_pid_set = set()

    csv_paths = sorted(glob.glob(os.path.join(train_dir, 'train_*.csv')))
    if not csv_paths:
        raise SystemExit('%s 下找不到任何 train_*.csv' % train_dir)

    for path in csv_paths:
        file_buckets = {
            s: {'sgf': [], 'col': [], 'key': [], 'pid': [], 'rk': []}
            for s in ('train', 'newdev', 'val648')
        }
        for chunk in pd.read_csv(path, usecols=['rank', 'game_id', 'player_id', 'color', 'sgf_content'], chunksize=50000):
            for rk, gid, pid, col_s, sgf_s in zip(
                chunk['rank'].astype(str),
                chunk['game_id'].astype(str),
                chunk['player_id'].astype(str),
                chunk['color'].astype(str),
                chunk['sgf_content'],
            ):
                key = '%s_%s' % (rk, gid)
                if key in train_keys:
                    target_split = 'train'
                    train_pid_set.add(pid)
                elif key in val_keys:
                    if pid in newdev_pids:
                        target_split = 'newdev'
                    elif pid in val648_pids:
                        target_split = 'val648'
                    else:
                        continue
                else:
                    continue

                col_u = col_s.strip().upper()
                if col_u not in COLOR_TO_ID:
                    raise ValueError('%s 的 %s 顏色欄位無效：%r' % (path, key, col_s))
                b = file_buckets[target_split]
                b['sgf'].append(str(sgf_s))
                b['col'].append(COLOR_TO_ID[col_u])
                b['key'].append(key)
                b['pid'].append(pid)
                b['rk'].append(RANK_TO_ID[rk])

        for split_name in ('train', 'newdev', 'val648'):
            b = file_buckets[split_name]
            if not b['key']:
                continue
            pos, is_target, length, hd, valid = parse_sgf_batch(
                b['sgf'], b['col'], max_moves=max_moves, raw_color=raw_color, nproc=nproc,
            )
            keep = valid
            p_acc = parts[split_name]
            p_acc['drop'] += int((~keep).sum())
            if keep.any():
                p_acc['pos'].append(pos[keep])
                p_acc['role'].append(is_target[keep])
                p_acc['len'].append(length[keep])
                p_acc['hd'].append(hd[keep])
                p_acc['key'].append(np.asarray(b['key'], dtype=str)[keep])
                p_acc['pid'].append(np.asarray(b['pid'], dtype=str)[keep])
                p_acc['rk'].append(np.asarray(b['rk'], dtype=np.int8)[keep])
                p_acc['col'].append(np.asarray(b['col'], dtype=np.uint8)[keep])

    # 嚴格驗證訓練集與驗證集玩家互斥
    overlap_nd = train_pid_set & newdev_pids
    overlap_v6 = train_pid_set & val648_pids
    if overlap_nd or overlap_v6:
        raise ValueError(
            '訓練集與驗證集玩家重疊：newdev %d 位、val648 %d 位' % (len(overlap_nd), len(overlap_v6))
        )

    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for split_name in ('train', 'newdev', 'val648'):
        t_s = time.time()
        p_acc = parts[split_name]
        if p_acc['pos']:
            pos_all = np.concatenate(p_acc['pos'], axis=0)
            role_all = np.concatenate(p_acc['role'], axis=0)
            len_all = np.concatenate(p_acc['len'], axis=0)
            hd_all = np.concatenate(p_acc['hd'], axis=0)
            key_all = np.concatenate(p_acc['key'], axis=0)
            pid_all = np.concatenate(p_acc['pid'], axis=0)
            rk_all = np.concatenate(p_acc['rk'], axis=0)
            col_all = np.concatenate(p_acc['col'], axis=0)
        else:
            pos_all = np.zeros((0, max_moves), dtype=np.uint16)
            role_all = np.zeros((0, max_moves), dtype=np.uint8)
            len_all = np.zeros(0, dtype=np.uint16)
            hd_all = np.zeros(0, dtype=np.int8)
            key_all = np.zeros(0, dtype=str)
            pid_all = np.zeros(0, dtype=str)
            rk_all = np.zeros(0, dtype=np.int8)
            col_all = np.zeros(0, dtype=np.uint8)

        out_path = os.path.join(out_dir, '%s.npz' % split_name)
        save_seq_npz(
            out_path,
            pos=pos_all,
            is_target=role_all,
            length=len_all,
            key=key_all,
            player_id=pid_all,
            rank_idx=rk_all,
            col=col_all,
            hd=hd_all,
            raw_color=raw_color,
        )
        n_valid = len(key_all)
        n_drop = p_acc['drop']
        n_players = len(set(pid_all.tolist()))
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(
            '%s：%d 局（丟棄 %d），%d 位玩家，%.1f MB，%.1f 秒'
            % (split_name, n_valid, n_drop, n_players, size_mb, time.time() - t_s),
            flush=True,
        )
        summary[split_name] = {'games': n_valid, 'dropped': n_drop, 'players': n_players, 'path': out_path}

    print('全部完成，總耗時 %.1f 秒' % (time.time() - t0), flush=True)
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train-dir', default=os.path.join('training-v1.1', 'training'))
    ap.add_argument('--feat-dir', default='rank-features-v180000')
    ap.add_argument('--out-dir', default='move-seq-features-v180000')
    ap.add_argument('--newdev-csv', default=DEFAULT_NEWDEV_CSV)
    ap.add_argument('--val648-csv', default=DEFAULT_VAL648_CSV)
    ap.add_argument('--max-moves', type=int, default=MAX_MOVES)
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--raw-color', action='store_true', help='不依 HD >= 2 反轉執色')
    args = ap.parse_args(argv)
    if args.max_moves < 1:
        ap.error('--max-moves 要 >= 1')
    return args


def main(argv=None):
    args = parse_args(argv)
    pack_splits(
        train_dir=args.train_dir,
        feat_dir=args.feat_dir,
        out_dir=args.out_dir,
        newdev_csv=args.newdev_csv,
        val648_csv=args.val648_csv,
        max_moves=args.max_moves,
        nproc=args.nproc,
        raw_color=args.raw_color,
    )


if __name__ == '__main__':
    main()
