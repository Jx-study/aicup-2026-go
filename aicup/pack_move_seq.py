"""打包整局落子座標序列為緊湊 .npz，供序列 Transformer 風格編碼器訓練與推論。

沿用既有 bit-packed 特徵目錄的 train/val 玩家切分，從 v1.1 訓練 CSV 解析每局前 L 手
落子座標與目標方標記（含 HD >= 2 讓子局執色反轉與 pass 處理），輸出 train、newdev、
val648 三份單檔 .npz，並驗證訓練集與驗證集玩家嚴格互斥。--extra-split 可從 val 切分另外
打包指定玩家，--splits 只寫出指定切分。

用法（在 repo 根目錄）：
    python -m aicup.pack_move_seq --train-dir training-v1.1/training \\
        --feat-dir rank-features-v180000 --out-dir move-seq-features-v180000
    python -m aicup.pack_move_seq --state-feats --extra-split hold=humansl-analysis/holdout-v180000.csv \\
        --splits hold --pat3-vocab move-seq-state-v180000/pat3_vocab.npy --out-dir move-seq-state-v180000
"""
import argparse
import csv
import glob
import os
import time
from multiprocessing import Pool
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd

from aicup import color, packed
from aicup.analyze_humansl import HANDICAP_STONES
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

# 規則引擎狀態欄位；pass 與 pad 使用各欄最後一格
STATE_FIELDS = ('st_libs', 'st_cap', 'st_atari', 'st_ko', 'pat3')
LIBS_NONE = 5  # 氣數桶 1, 2, 3, 4-5, 6+ 對應 0..4
CAP_NONE = 4  # 提子桶 0, 1, 2-3, 4+ 對應 0..3
FLAG_NONE = 2
PAT3_RAW_NONE = 65535  # 8 鄰點各 2 bit 的原始編碼，盤面上不可能全為盤外
PAT3_TOP = 4095  # 詞表大小，其後依序為 OOV 與 pass/pad
PAT3_OOV = PAT3_TOP
PAT3_PAD = PAT3_TOP + 1
PAT3_NUM = PAT3_TOP + 2
PAT3_VOCAB_FILE = 'pat3_vocab.npy'
DEFAULT_PATCH_SIZE = 7

DEFAULT_NEWDEV_CSV = os.path.join('humansl-analysis', 'val-players-newdev-v180000.csv')
DEFAULT_VAL648_CSV = os.path.join('humansl-analysis', 'val-players.csv')


def patch_nbytes(patch_size):
    """局部盤面每格 2 bit，每 byte 存 4 格。"""
    return (patch_size * patch_size + 3) // 4


class SeqParse(NamedTuple):
    """落子序列解析結果，單局時各欄少掉 N 維，未要求的欄位為 None。

    pos         (N, L) uint16 落子座標
    is_target   (N, L) uint8 目標方標記
    length      (N,) uint16 有效手數
    hd          (N,) int8 讓子數
    ok          (N,) bool 解析成功且至少一手
    st4         (N, L, 4) uint8 規則引擎狀態，state_feats 才有
    pat3        (N, L) uint16 pat3 原始編碼，state_feats 才有
    patch       (N, L, patch_nbytes) uint8 局部盤面，patch_size > 0 才有
    """
    pos: np.ndarray
    is_target: np.ndarray
    length: Any
    hd: Any
    ok: Any
    st4: Optional[np.ndarray] = None
    pat3: Optional[np.ndarray] = None
    patch: Optional[np.ndarray] = None


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


def _build_neighbors4():
    """每個盤面點的上下左右鄰點索引。"""
    nbs = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            nbs.append(tuple(
                rr * BOARD_SIZE + cc
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
            ))
    return tuple(nbs)


_NEIGHBORS4 = _build_neighbors4()
_NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _build_pat3_perms():
    """D4 8 種變換下 8 鄰點的讀取順序，用於取最小編碼作為正規化棋形。"""
    perms = []
    for k in range(8):
        perm = []
        for dr, dc in _NB8:
            tr, tc = dr, dc
            for _ in range(k % 4):
                tr, tc = tc, -tr
            if k >= 4:
                tc = -tc
            perm.append(_NB8.index((tr, tc)))
        perms.append(tuple(perm))
    return tuple(perms)


_PAT3_PERMS = _build_pat3_perms()


def _pat3_code(board, r, c, color):
    """落子前 3x3 鄰域（空 0、己 1、敵 2、盤外 3）的 D4 正規化編碼。"""
    vals = []
    for dr, dc in _NB8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            v = board[rr * BOARD_SIZE + cc]
            vals.append(0 if v == 0 else (1 if v == color else 2))
        else:
            vals.append(3)
    best = PAT3_RAW_NONE
    for perm in _PAT3_PERMS:
        code = 0
        for i in perm:
            code = (code << 2) | vals[i]
        if code < best:
            best = code
    return best


def _group_libs(board, start):
    """回傳 start 所在棋塊的點集合與氣集合。"""
    color_v = board[start]
    stack, group, libs = [start], {start}, set()
    while stack:
        u = stack.pop()
        for w in _NEIGHBORS4[u]:
            b = board[w]
            if b == 0:
                libs.add(w)
            elif b == color_v and w not in group:
                group.add(w)
                stack.append(w)
    return group, libs


def _bucket_libs(n):
    return 0 if n <= 1 else (1 if n == 2 else (2 if n == 3 else (3 if n <= 5 else 4)))


def _bucket_cap(n):
    return 0 if n == 0 else (1 if n == 1 else (2 if n <= 3 else 3))


def _patch_bytes(board, r, c, color, patch_size):
    """落子前以 (r, c) 為中心的局部盤面（空 0、己 1、敵 2、盤外 3），每格 2 bit 依列優先打包。"""
    half = patch_size // 2
    out = bytearray(patch_nbytes(patch_size))
    i = 0
    for rr in range(r - half, r + half + 1):
        for cc in range(c - half, c + half + 1):
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                v = board[rr * BOARD_SIZE + cc]
                code = 0 if v == 0 else (1 if v == color else 2)
            else:
                code = 3
            out[i >> 2] |= code << (2 * (3 - (i & 3)))
            i += 1
    return bytes(out)


def replay_move_states(moves, hd, patch_size=0):
    """依序重放 (SGF 標記色 1 黑 2 白, pos) 落子，回傳每手 (libs, cap, atari, ko, pat3 原始編碼)。

    HD >= 2 讓子局的讓子屬於 SGF 標記為白的一方（資料集讓子局 B/W 標記與實際相反）。
    落在已有子的點視為覆蓋，自殺手保留在盤上且氣數記為 0。
    patch_size > 0 時另外回傳每手落子前局部盤面的打包 bytes，pass 為全 0。
    """
    board = [0] * (BOARD_SIZE * BOARD_SIZE)
    for st in HANDICAP_STONES.get(int(hd), '').split():
        board[(BOARD_SIZE - 1 - (ord(st[1]) - 97)) * BOARD_SIZE + ord(st[0]) - 97] = 2
    out, patches = [], []
    empty_patch = bytes(patch_nbytes(patch_size)) if patch_size else b''
    for col_v, p in moves:
        if p >= PASS_POS:
            out.append((LIBS_NONE, CAP_NONE, FLAG_NONE, FLAG_NONE, PAT3_RAW_NONE))
            if patch_size:
                patches.append(empty_patch)
            continue
        opp = 3 - col_v
        r, c = divmod(p, BOARD_SIZE)
        pat = _pat3_code(board, r, c, col_v)
        if patch_size:
            patches.append(_patch_bytes(board, r, c, col_v, patch_size))
        board[p] = col_v
        captured, seen = 0, set()
        for w in _NEIGHBORS4[p]:
            if board[w] == opp and w not in seen:
                grp, libs = _group_libs(board, w)
                seen |= grp
                if not libs:
                    for u in grp:
                        board[u] = 0
                    captured += len(grp)
        atari, seen = 0, set()
        for w in _NEIGHBORS4[p]:
            if board[w] == opp and w not in seen:
                grp, libs = _group_libs(board, w)
                seen |= grp
                if len(libs) == 1:
                    atari = 1
        own, own_libs = _group_libs(board, p)
        ko = int(captured == 1 and len(own) == 1 and len(own_libs) == 1)
        out.append((_bucket_libs(len(own_libs)), _bucket_cap(captured), atari, ko, pat))
    return (out, patches) if patch_size else out


def build_pat3_vocab(pat3_raw):
    """由 train 的 pat3 原始編碼依頻率取前 PAT3_TOP 個，同頻依編碼排序。"""
    codes = np.asarray(pat3_raw, dtype=np.uint16).ravel()
    codes = codes[codes != PAT3_RAW_NONE]
    if codes.size == 0:
        return np.zeros(0, dtype=np.uint16)
    uniq, cnt = np.unique(codes, return_counts=True)
    order = np.lexsort((uniq, -cnt))[:PAT3_TOP]
    return uniq[order].astype(np.uint16)


def map_state_fields(st4, pat3_raw, vocab):
    """把 (N, L, 4) 狀態與 (N, L) pat3 原始編碼合成 (N, L, 5) int16，pat3 換成詞表索引。"""
    lut = np.full(PAT3_RAW_NONE + 1, PAT3_OOV, dtype=np.int16)
    lut[np.asarray(vocab, dtype=np.int64)] = np.arange(len(vocab), dtype=np.int16)
    lut[PAT3_RAW_NONE] = PAT3_PAD
    out = np.empty(st4.shape[:2] + (5,), dtype=np.int16)
    out[..., :4] = st4
    # 分塊查表，避免全量 intp 索引暫存
    step = max(1, (1 << 24) // max(1, pat3_raw.shape[1]))
    for i in range(0, len(pat3_raw), step):
        out[i:i + step, :, 4] = lut[pat3_raw[i:i + step]]
    return out


def _empty_state(max_moves):
    st4 = np.empty((max_moves, 4), dtype=np.uint8)
    st4[:] = (LIBS_NONE, CAP_NONE, FLAG_NONE, FLAG_NONE)
    return st4, np.full(max_moves, PAT3_RAW_NONE, dtype=np.uint16)


def parse_sgf_move_seq(sgf_content, actual_color_id, max_moves=MAX_MOVES, raw_color=False, state_feats=False,
                       patch_size=0):
    """解析單局 SGF 為定長落子序列，回傳單局 SeqParse。

    actual_color_id 為目標方實際執色（0 黑 1 白）；HD >= 2 時依 color.feature_color
    反轉 SGF 內 B/W 標記。解析失敗或無任何落子時 ok 為 False。
    """
    pos = np.full(max_moves, PAD_POS, dtype=np.uint16)
    is_target = np.full(max_moves, ROLE_PAD, dtype=np.uint8)
    st4, pat3 = _empty_state(max_moves) if state_feats else (None, None)
    patch = np.zeros((max_moves, patch_nbytes(patch_size)), dtype=np.uint8) if patch_size else None

    def result(length, hd, ok):
        return SeqParse(pos, is_target, length, hd, ok, st4, pat3, patch)

    if not isinstance(sgf_content, str) or not sgf_content.strip():
        return result(0, 0, False)

    hd_val = color.parse_hd(sgf_content)
    hd = 0 if hd_val is None else int(hd_val)
    target_sgf_col = int(actual_color_id) if raw_color else int(color.feature_color(int(actual_color_id), hd))

    try:
        from sgfmill import sgf  # 延遲載入，避免主行程重複匯入開銷
        game = sgf.Sgf_game.from_string(sgf_content)
        nodes = game.get_main_sequence()
    except Exception:
        return result(0, hd, False)

    idx = 0
    sgf_moves = []
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
        sgf_moves.append((move_col_id + 1, p))
        idx += 1
        if idx >= max_moves:
            break

    if state_feats or patch_size:
        res = replay_move_states(sgf_moves, hd, patch_size=patch_size)
        states, patches = res if patch_size else (res, None)
        if state_feats:
            for i, st in enumerate(states):
                st4[i] = st[:4]
                pat3[i] = st[4]
        if patch_size:
            for i, b in enumerate(patches):
                patch[i] = np.frombuffer(b, dtype=np.uint8)
    return result(idx, hd, idx > 0)


def _empty_batch(n, max_moves, state_feats, patch_size):
    """建立 n 局的批次 SeqParse，各欄填 pad 值。"""
    return SeqParse(
        pos=np.full((n, max_moves), PAD_POS, dtype=np.uint16),
        is_target=np.full((n, max_moves), ROLE_PAD, dtype=np.uint8),
        length=np.zeros(n, dtype=np.uint16),
        hd=np.zeros(n, dtype=np.int8),
        ok=np.zeros(n, dtype=bool),
        st4=np.empty((n, max_moves, 4), dtype=np.uint8) if state_feats else None,
        pat3=np.empty((n, max_moves), dtype=np.uint16) if state_feats else None,
        patch=np.zeros((n, max_moves, patch_nbytes(patch_size)), dtype=np.uint8) if patch_size else None,
    )


def _parse_chunk_worker(args):
    """子進程批次解析落子序列，頂層函式供 multiprocessing 使用。"""
    rows, max_moves, raw_color, state_feats, patch_size = args
    out = _empty_batch(len(rows), max_moves, state_feats, patch_size)
    for i, (sgf_content, col_id) in enumerate(rows):
        r = parse_sgf_move_seq(
            sgf_content, col_id, max_moves=max_moves, raw_color=raw_color, state_feats=state_feats,
            patch_size=patch_size,
        )
        out.pos[i] = r.pos
        out.is_target[i] = r.is_target
        out.length[i] = r.length
        out.hd[i] = min(max(r.hd, -128), 127)
        out.ok[i] = r.ok
        for name in ('st4', 'pat3', 'patch'):
            arr = getattr(out, name)
            if arr is not None:
                arr[i] = getattr(r, name)
    return out


def parse_sgf_batch(sgfs, color_ids, max_moves=MAX_MOVES, raw_color=False, nproc=1, state_feats=False,
                    patch_size=0):
    """批次解析多局 SGF，回傳批次 SeqParse。"""
    n = len(sgfs)
    if n == 0:
        return _empty_batch(0, max_moves, state_feats, patch_size)
    pairs = list(zip(sgfs, color_ids))
    eff_proc = max(1, min(int(nproc), n))
    if eff_proc == 1:
        return _parse_chunk_worker((pairs, max_moves, raw_color, state_feats, patch_size))

    chunk_size = (n + eff_proc * 4 - 1) // (eff_proc * 4)
    chunks = [(pairs[i:i + chunk_size], max_moves, raw_color, state_feats, patch_size)
              for i in range(0, n, chunk_size)]
    with Pool(eff_proc) as pool:
        parts = pool.map(_parse_chunk_worker, chunks)
    return SeqParse(*(
        None if parts[0][k] is None else np.concatenate([p[k] for p in parts], axis=0)
        for k in range(len(SeqParse._fields))
    ))


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


def save_seq_npz(path, pos, is_target, length, key, player_id, rank_idx, col, hd, raw_color=False, state=None):
    """以暫存檔原子寫出單一落子序列 .npz；state 為 (N, L, 5) 時另存 STATE_FIELDS 各欄。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.tmp'
    extra = {}
    if state is not None:
        for j, name in enumerate(STATE_FIELDS):
            extra[name] = state[..., j].astype(np.uint16 if name == 'pat3' else np.uint8)
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
            **extra,
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
    state_feats=False,
    extra_splits=None,
    splits=None,
    pat3_vocab_path=None,
):
    """依 feat_dir 的 train/val 切分與玩家清單，打包 train、newdev、val648 與額外切分的序列檔。

    extra_splits 為 {切分名: 玩家 CSV}，從 val 切分取出這些玩家，玩家須與其他切分互斥。
    splits 指定要寫出的切分，None 表示全部。
    state_feats 為 True 時另存規則引擎狀態欄位；pat3_vocab_path 有值時沿用該詞表，
    否則只由 train 建立並寫出 PAT3_VOCAB_FILE。
    """
    t0 = time.time()
    train_keys = load_split_keys(os.path.join(feat_dir, 'train'))
    val_keys = load_split_keys(os.path.join(feat_dir, 'val'))
    val_pids = {'newdev': read_player_ids_csv(newdev_csv), 'val648': read_player_ids_csv(val648_csv)}
    for name, path in (extra_splits or {}).items():
        if not name or name in ('train', 'newdev', 'val648'):
            raise ValueError('額外切分名稱無效：%r' % name)
        val_pids[name] = read_player_ids_csv(path)
    names = list(val_pids)
    all_splits = ('train',) + tuple(names)
    out_splits = tuple(splits) if splits else all_splits
    if set(out_splits) - set(all_splits):
        raise ValueError('未知的切分：%s' % sorted(set(out_splits) - set(all_splits)))

    for i, a in enumerate(names):
        for b_name in names[i + 1:]:
            inter = val_pids[a] & val_pids[b_name]
            if inter:
                raise ValueError('%s 與 %s 玩家有交集：%d 位' % (a, b_name, len(inter)))

    vocab = None
    if state_feats and pat3_vocab_path:
        vocab = np.load(pat3_vocab_path).astype(np.uint16)
        dst = os.path.join(out_dir, PAT3_VOCAB_FILE)
        if os.path.exists(dst) and not np.array_equal(np.load(dst), vocab):
            raise ValueError('%s 與 %s 的 pat3 詞表不同' % (pat3_vocab_path, dst))
    elif state_feats and 'train' not in out_splits:
        raise ValueError('不寫出 train 時要用 pat3_vocab_path 指定詞表')

    # 逐檔解析並立即釋放原始 SGF 字串，降低記憶體峰值
    parts = {
        s: {'pos': [], 'role': [], 'len': [], 'hd': [], 'key': [], 'pid': [], 'rk': [], 'col': [],
            'st4': [], 'pat3': [], 'drop': 0}
        for s in out_splits
    }
    train_pid_set = set()

    csv_paths = sorted(glob.glob(os.path.join(train_dir, 'train_*.csv')))
    if not csv_paths:
        raise SystemExit('%s 下找不到任何 train_*.csv' % train_dir)

    for path in csv_paths:
        file_buckets = {
            s: {'sgf': [], 'col': [], 'key': [], 'pid': [], 'rk': []}
            for s in out_splits
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
                    target_split = next((s for s in names if pid in val_pids[s]), None)
                    if target_split is None:
                        continue
                else:
                    continue
                if target_split not in file_buckets:
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

        for split_name in out_splits:
            b = file_buckets[split_name]
            if not b['key']:
                continue
            res = parse_sgf_batch(
                b['sgf'], b['col'], max_moves=max_moves, raw_color=raw_color, nproc=nproc,
                state_feats=state_feats,
            )
            pos, is_target, length, hd, keep = res.pos, res.is_target, res.length, res.hd, res.ok
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
                if state_feats:
                    p_acc['st4'].append(res.st4[keep])
                    p_acc['pat3'].append(res.pat3[keep])

    # 嚴格驗證訓練集與驗證集玩家互斥
    overlap = {s: len(train_pid_set & val_pids[s]) for s in names}
    if any(overlap.values()):
        raise ValueError('訓練集與驗證集玩家重疊：' + '、'.join('%s %d 位' % kv for kv in overlap.items()))

    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    vocab_dst = os.path.join(out_dir, PAT3_VOCAB_FILE)
    if state_feats and vocab is None:
        vocab = build_pat3_vocab(np.concatenate(parts['train']['pat3'], axis=0) if parts['train']['pat3'] else [])
        np.save(vocab_dst, vocab)
        print('pat3 詞表 %d 個（只由 train 建立）' % len(vocab), flush=True)
    elif state_feats and not os.path.exists(vocab_dst):
        np.save(vocab_dst, vocab)
    for split_name in out_splits:
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
            state_all = (
                map_state_fields(np.concatenate(p_acc['st4'], axis=0), np.concatenate(p_acc['pat3'], axis=0), vocab)
                if state_feats else None
            )
            p_acc['st4'], p_acc['pat3'] = [], []
        else:
            pos_all = np.zeros((0, max_moves), dtype=np.uint16)
            role_all = np.zeros((0, max_moves), dtype=np.uint8)
            len_all = np.zeros(0, dtype=np.uint16)
            hd_all = np.zeros(0, dtype=np.int8)
            key_all = np.zeros(0, dtype=str)
            pid_all = np.zeros(0, dtype=str)
            rk_all = np.zeros(0, dtype=np.int8)
            col_all = np.zeros(0, dtype=np.uint8)
            state_all = np.zeros((0, max_moves, 5), dtype=np.int16) if state_feats else None

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
            state=state_all,
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
    ap.add_argument('--state-feats', action='store_true', help='另存規則引擎狀態欄位與 pat3 詞表')
    ap.add_argument('--extra-split', action='append', default=[], metavar='NAME=CSV',
                    help='從 val 切分另外打包 CSV 內的玩家，可重複')
    ap.add_argument('--splits', nargs='+', default=None, help='只寫出這些切分，預設全部')
    ap.add_argument('--pat3-vocab', default='', help='沿用既有 pat3 詞表，不由 train 重建')
    args = ap.parse_args(argv)
    if args.max_moves < 1:
        ap.error('--max-moves 要 >= 1')
    extra = {}
    for item in args.extra_split:
        name, sep, path = item.partition('=')
        if not sep or not name or not path:
            ap.error('--extra-split 格式為 NAME=CSV：%r' % item)
        extra[name] = path
    args.extra_split = extra
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
        state_feats=args.state_feats,
        extra_splits=args.extra_split,
        splits=args.splits,
        pat3_vocab_path=args.pat3_vocab or None,
    )


if __name__ == '__main__':
    main()
