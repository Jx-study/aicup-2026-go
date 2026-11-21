"""全盤 D4-不變局部棋形指紋抽取與 Chi-squared 殘差比對模組。

完整重放 19x19 棋局（含讓子與提子），抽取目標方每手落子前之 D4-不變
3x3 與 12 點菱形局部棋形頻率，並僅由候選池估計背景分佈建立 Chi-squared 殘差索引。
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import re
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

import numpy as np
import pandas as pd

from aicup.analyze_humansl import HANDICAP_STONES

DEFAULT_PATTERN_EPS = 1e-3
DEFAULT_PATTERN_MIN_DF = 3
DEFAULT_PATTERN_BETA = 0.7
DEFAULT_CHUNKSIZE = 200

RE_MOVE = re.compile(r";([BW])\[([a-s]{2})?\]")
RE_HD = re.compile(r"HD\[(\d+)\]")

# 13 點菱形座標偏移：前 9 點為 3x3，後 4 點為距離 2 之正交點
_DIAMOND_PTS: Tuple[Tuple[int, int], ...] = (
    tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    + ((-2, 0), (2, 0), (0, -2), (0, 2))
)
_PT_TO_IDX: Dict[Tuple[int, int], int] = {pt: i for i, pt in enumerate(_DIAMOND_PTS)}


def _build_d4_permutations() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    """預先計算 3x3 與 12 點菱形在 D4 群 8 種對稱變換下的索引置換表。"""
    d4_3x3 = []
    d4_diamond = []
    for rot in range(4):
        for flip in (False, True):
            perm3 = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = dr, dc
                    for _ in range(rot):
                        r, c = -c, r
                    if flip:
                        c = -c
                    perm3.append((r + 1) * 3 + (c + 1))
            d4_3x3.append(tuple(perm3))

            # 落子前中心必為空點，菱形僅編碼外圍 12 點
            perm12 = []
            for dr, dc in _DIAMOND_PTS:
                if dr == 0 and dc == 0:
                    continue
                r, c = dr, dc
                for _ in range(rot):
                    r, c = -c, r
                if flip:
                    c = -c
                perm12.append(_PT_TO_IDX[(r, c)])
            d4_diamond.append(tuple(perm12))
    return tuple(d4_3x3), tuple(d4_diamond)


def _build_neighbors() -> Tuple[Tuple[int, ...], ...]:
    """預先計算 19x19 棋盤各格之四鄰居一維索引。"""
    neighbors = []
    for r in range(19):
        for c in range(19):
            nb = []
            if r > 0:
                nb.append((r - 1) * 19 + c)
            if r < 18:
                nb.append((r + 1) * 19 + c)
            if c > 0:
                nb.append(r * 19 + c - 1)
            if c < 18:
                nb.append(r * 19 + c + 1)
            neighbors.append(tuple(nb))
    return tuple(neighbors)


_D4_3X3, _D4_DIAMOND = _build_d4_permutations()
NEIGHBORS = _build_neighbors()


class Chi2PatternIndex(NamedTuple):
    """候選池 Chi-squared 殘差棋形索引。

    pids: 排序後的候選人 ID 清單
    vocab_to_idx: 棋形鍵值對應至特徵欄位的映射字典
    mu: 候選池平均相對頻率向量，形狀 (V,)
    denom: 變異數穩定分母 sqrt(mu + eps)，形狀 (V,)
    mat: L2 正規化後之候選殘差矩陣，形狀 (N, V)
    """

    pids: List[str]
    vocab_to_idx: Dict[Tuple[int, int], int]
    mu: np.ndarray
    denom: np.ndarray
    mat: np.ndarray


def extract_game_local_patterns(sgf_str: str, target_color: str) -> Counter:
    """完整重放單局 SGF 並抽取目標方每手落子前之 D4-不變 3x3 與 12 點菱形計數。

    棋點相對顏色編碼：空=0、己方=1、對方=2、盤外邊界=3。
    """
    if not isinstance(sgf_str, str) or not sgf_str.strip():
        return Counter()

    col_norm = str(target_color).strip().lower()
    if col_norm not in ("b", "w"):
        col_norm = "b"
    target_val = 1 if col_norm == "b" else 2

    m_hd = RE_HD.search(sgf_str)
    hd = int(m_hd.group(1)) if m_hd else 0
    board = [0] * 361
    if hd in HANDICAP_STONES:
        for s in HANDICAP_STONES[hd].split():
            r, c = ord(s[1]) - 97, ord(s[0]) - 97
            board[r * 19 + c] = 1

    # 讓子局（HD>=2）依資料集慣例反轉 SGF 執色標記
    flip_color = hd >= 2
    raw_moves = RE_MOVE.findall(sgf_str)
    cnt: Counter = Counter()

    for c_ch, pt in raw_moves:
        if not pt:
            continue
        col = (2 if c_ch == "B" else 1) if flip_color else (1 if c_ch == "B" else 2)
        opp = 3 - col
        r, c = ord(pt[1]) - 97, ord(pt[0]) - 97
        idx = r * 19 + c

        if col == target_val:
            patch = [0] * 13
            for k, (dr, dc) in enumerate(_DIAMOND_PTS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 19 and 0 <= cc < 19:
                    v = board[rr * 19 + cc]
                    patch[k] = 0 if v == 0 else (1 if v == col else 2)
                else:
                    patch[k] = 3

            best3 = 1 << 30
            for perm in _D4_3X3:
                code = 0
                for pi in perm:
                    code = (code << 2) | patch[pi]
                if code < best3:
                    best3 = code
            cnt[(3, best3)] += 1

            best12 = 1 << 30
            for perm in _D4_DIAMOND:
                code = 0
                for pi in perm:
                    code = (code << 2) | patch[pi]
                if code < best12:
                    best12 = code
            cnt[(12, best12)] += 1

        # 落子並移除相鄰無氣的對方棋塊
        board[idx] = col
        for nb in NEIGHBORS[idx]:
            if board[nb] == opp:
                stack = [nb]
                grp = {nb}
                has_lib = False
                while stack:
                    u = stack.pop()
                    for w in NEIGHBORS[u]:
                        bw = board[w]
                        if bw == 0:
                            has_lib = True
                            break
                        if bw == opp and w not in grp:
                            grp.add(w)
                            stack.append(w)
                    if has_lib:
                        break
                if not has_lib:
                    for u in grp:
                        board[u] = 0

    return cnt


def merge_pattern_counters(counters: Iterable[Counter]) -> Counter:
    """合併多局局部棋形計數器為單一計數器。"""
    merged: Counter = Counter()
    for cnt in counters:
        merged.update(cnt)
    return merged


def _extract_pattern_worker(task: Tuple[str, str]) -> Counter:
    """供 ProcessPoolExecutor 呼叫之頂層可序列化抽取函式。"""
    sgf_str, target_color = task
    return extract_game_local_patterns(sgf_str, target_color)


def extract_patterns_batch(
    tasks: Sequence[Tuple[str, str]],
    workers: int = 1,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> List[Counter]:
    """批次抽取多局局部棋形，workers 大於 1 時啟用多行程平行處理。"""
    if not tasks:
        return []
    if workers <= 1 or len(tasks) < 2:
        return [extract_game_local_patterns(sgf_str, col) for sgf_str, col in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_extract_pattern_worker, tasks, chunksize=chunksize))


def build_candidate_pattern_counters(
    cand_csv_path: str,
    workers: int = 1,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Dict[str, Counter]:
    """讀取候選庫 CSV 並抽取每位候選人合併後之局部棋形計數。"""
    df = pd.read_csv(cand_csv_path)
    pids_per_row: List[str] = []
    tasks: List[Tuple[str, str]] = []
    has_color = "color" in df.columns

    for _, row in df.iterrows():
        pid = str(row["player_id"]).strip()
        sgf_content = "" if pd.isna(row["sgf_content"]) else str(row["sgf_content"])
        if has_color and pd.notna(row["color"]):
            color = str(row["color"]).strip().lower()
        else:
            color = "b"
        if color not in ("b", "w"):
            color = "b"
        pids_per_row.append(pid)
        tasks.append((sgf_content, color))

    counters = extract_patterns_batch(tasks, workers=workers, chunksize=chunksize)
    by_pid: Dict[str, Counter] = {}
    for pid, cnt in zip(pids_per_row, counters):
        if pid not in by_pid:
            by_pid[pid] = Counter()
        by_pid[pid].update(cnt)
    return by_pid


def vectorize_pattern_counter(
    vocab_to_idx: Dict[Tuple[int, int], int],
    mu: np.ndarray,
    denom: np.ndarray,
    counter: Counter,
) -> np.ndarray:
    """將單一棋形計數器轉換為 L2 正規化之 Chi-squared 殘差向量。"""
    vocab_size = len(mu)
    if vocab_size == 0:
        return np.zeros(0, dtype=np.float64)

    # 沒有任何棋形時回傳零向量，避免 -mu 方向造成假相似度
    tot = float(sum(counter.values()))
    if tot <= 0.0:
        return np.zeros(vocab_size, dtype=np.float64)
    p = np.zeros(vocab_size, dtype=np.float64)
    for k, val in counter.items():
        idx = vocab_to_idx.get(k)
        if idx is not None:
            p[idx] = val / tot

    vec = (p - mu) / denom
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec /= norm
    return vec


def build_chi2_pattern_index(
    cand_counters: Dict[str, Counter],
    min_df: int = DEFAULT_PATTERN_MIN_DF,
    eps: float = DEFAULT_PATTERN_EPS,
) -> Chi2PatternIndex:
    """僅由候選庫建立詞表、背景分佈與正規化 Chi-squared 殘差索引矩陣。"""
    pids = sorted(cand_counters.keys())
    n_cands = len(pids)
    if n_cands == 0:
        empty = np.zeros(0, dtype=np.float64)
        return Chi2PatternIndex([], {}, empty, empty, np.zeros((0, 0), dtype=np.float64))

    df: Counter = Counter()
    mu_sum: Counter = Counter()
    for pid in pids:
        cnt = cand_counters[pid]
        tot = float(sum(cnt.values()))
        if tot <= 0.0:
            continue
        for k, val in cnt.items():
            df[k] += 1
            mu_sum[k] += val / tot

    vocab = sorted(k for k, d in df.items() if d >= min_df)
    vocab_to_idx = {k: i for i, k in enumerate(vocab)}
    vocab_size = len(vocab)

    if vocab_size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return Chi2PatternIndex(pids, vocab_to_idx, empty, empty, np.zeros((n_cands, 0), dtype=np.float64))

    mu = np.array([mu_sum[k] / n_cands for k in vocab], dtype=np.float64)
    denom = np.sqrt(mu + eps)
    mat = np.vstack([
        vectorize_pattern_counter(vocab_to_idx, mu, denom, cand_counters[pid])
        for pid in pids
    ])
    return Chi2PatternIndex(pids, vocab_to_idx, mu, denom, mat)


def score_query_patterns_array(
    index: Chi2PatternIndex,
    query_counter: Counter,
) -> np.ndarray:
    """計算題目棋形計數器與所有候選人之餘弦相似度陣列（順序同 index.pids）。"""
    n_cands = len(index.pids)
    if n_cands == 0 or index.mat.shape[1] == 0:
        return np.zeros(n_cands, dtype=np.float64)
    q_vec = vectorize_pattern_counter(index.vocab_to_idx, index.mu, index.denom, query_counter)
    return index.mat @ q_vec


def score_query_patterns(
    index: Chi2PatternIndex,
    query_counter: Counter,
) -> Dict[str, float]:
    """計算題目棋形計數器與所有候選人之餘弦相似度字典。"""
    scores = score_query_patterns_array(index, query_counter)
    return {pid: float(sc) for pid, sc in zip(index.pids, scores)}
