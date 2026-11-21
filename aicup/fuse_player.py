"""開局風格與 GoRankResNet 棋力分佈之雙軌融合選手辨識評分器。

結合純開局指紋特徵與深度神經網路棋力分佈，
透過「開局風格相似度 + alpha * 棋力相容度」與「硬性階層遮罩」進行雙重過濾與重排序。
支援離線 JSON 快取載入，確保無 GPU 或無大型權重環境下亦能秒級復刻推論。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from aicup.opening_player import (
    DEFAULT_MAX_MOVES,
    IDF_POWER,
    TOP_N,
    W_RESP,
    build_candidate_profiles_and_idf,
    build_query_vectors,
    compute_style_similarity,
    extract_raw_counts,
    get_canonical_moves,
    get_raw_moves,
)
from aicup.pattern_player import (
    Chi2PatternIndex,
    DEFAULT_PATTERN_EPS,
    DEFAULT_PATTERN_MIN_DF,
    build_candidate_pattern_counters,
    build_chi2_pattern_index,
    extract_patterns_batch,
    merge_pattern_counters,
    score_query_patterns,
)

DEFAULT_CAND_CACHE = os.path.join("player-trained-models", "rank-cache", "cand_ranks.json")
DEFAULT_QUERY_CACHE = os.path.join("player-trained-models", "rank-cache", "query_ranks.json")


def rank_overlap(p_c: Sequence[float], p_q: Sequence[float]) -> float:
    """計算兩份 10 級機率分佈的內積交疊度（Overlap / Bhattacharyya）。"""
    if len(p_c) != 10 or len(p_q) != 10:
        return 0.0
    return float(np.dot(np.asarray(p_c, dtype=np.float64), np.asarray(p_q, dtype=np.float64)))


def rank_gaussian_similarity(exp_c: float, exp_q: float, sigma: float = 1.5) -> float:
    """計算兩位選手期望段位差的高斯相容度得分（值域 0.0 至 1.0）。"""
    diff = float(exp_c - exp_q)
    return float(np.exp(-(diff ** 2) / (2.0 * (sigma ** 2))))


def compute_rank_compatibility(
    cand_rank: Optional[dict],
    query_rank: Optional[dict],
    metric: str = "hybrid",
    delta_thresh: float = 3.0,
    sigma: float = 1.5
) -> Tuple[float, bool]:
    """計算候選人與題目的棋力相容度分數，並判定是否通過硬性遮罩。

    回傳：(相容度分數, 是否通過遮罩)。
    若超過容許段位差（delta_thresh），回傳 (-inf, False) 進行硬性剔除。
    """
    if not cand_rank or not query_rank:
        return 0.0, True

    exp_c = cand_rank.get("expected_rank")
    exp_q = query_rank.get("expected_rank")

    # 硬性階層遮罩
    if exp_c is not None and exp_q is not None and delta_thresh > 0:
        if abs(exp_c - exp_q) > delta_thresh:
            return float("-inf"), False

    # 相容度分數計算
    if metric == "overlap":
        p_c = cand_rank.get("probs", [])
        p_q = query_rank.get("probs", [])
        score = rank_overlap(p_c, p_q)
    elif metric == "gaussian":
        if exp_c is not None and exp_q is not None:
            score = rank_gaussian_similarity(exp_c, exp_q, sigma=sigma)
        else:
            score = 0.0
    elif metric == "hybrid":
        p_c = cand_rank.get("probs", [])
        p_q = query_rank.get("probs", [])
        ov = rank_overlap(p_c, p_q)
        gs = rank_gaussian_similarity(exp_c, exp_q, sigma=sigma) if (exp_c is not None and exp_q is not None) else 0.0
        score = 0.5 * ov + 0.5 * gs
    else:
        score = 0.0

    return score, True


def apply_csls_column_penalty(score_mat: np.ndarray, csls_k: int) -> np.ndarray:
    """對分數矩陣 (Q, C) 套用候選側 CSLS 去樞紐化，扣除 0.5 倍各欄前 k 大平均。"""
    if csls_k <= 0 or score_mat.size == 0:
        return score_mat.copy()
    n_q, n_c = score_mat.shape
    if np.all(np.isfinite(score_mat)):
        k = min(csls_k, n_q)
        col_hub = np.partition(score_mat, n_q - k, axis=0)[-k:].mean(axis=0)
        return score_mat - 0.5 * col_hub[None, :]

    out = score_mat.copy()
    for j in range(n_c):
        col = score_mat[:, j]
        finite_mask = np.isfinite(col)
        n_valid = int(finite_mask.sum())
        if n_valid == 0:
            continue
        valid_vals = col[finite_mask]
        k = min(csls_k, n_valid)
        hub = float(np.partition(valid_vals, n_valid - k)[-k:].mean())
        out[finite_mask, j] -= 0.5 * hub
    return out


def compute_emb_similarity(
    cand_embs_by_id: Dict[str, np.ndarray],
    query_embs_by_id: Dict[str, np.ndarray],
    cand_ids: Sequence[str],
    query_ids: Sequence[str],
    topk_weight: float = 0.0,
    topk: int = 2,
) -> np.ndarray:
    """計算候選池中心化質心餘弦，並可選疊加局對局 top-k 相似度矩陣 (Q, C)。

    沒有有效向量的候選欄位回傳 -inf，避免排在負相似度候選之前。
    """
    if not cand_ids or not query_ids:
        return np.zeros((len(query_ids), len(cand_ids)), dtype=np.float64)

    dim = None
    for cid in cand_ids:
        arr = cand_embs_by_id.get(cid)
        if arr is not None and len(arr) > 0:
            dim = np.asarray(arr).shape[-1]
            break
    if dim is None:
        return np.full((len(query_ids), len(cand_ids)), float("-inf"), dtype=np.float64)

    cand_mat = np.zeros((len(cand_ids), dim), dtype=np.float64)
    cand_ok = np.zeros(len(cand_ids), dtype=bool)
    cand_arrays: List[Optional[np.ndarray]] = [None] * len(cand_ids)
    for i, cid in enumerate(cand_ids):
        arr = cand_embs_by_id.get(cid)
        if arr is not None and len(arr) > 0:
            a64 = np.asarray(arr, dtype=np.float64)
            cand_arrays[i] = a64
            cand_mat[i] = a64.mean(axis=0)
            cand_ok[i] = True

    mu_cand = cand_mat[cand_ok].mean(axis=0, keepdims=True)
    cand_centered = cand_mat - mu_cand
    cand_norms = np.linalg.norm(cand_centered, axis=1, keepdims=True)
    cand_unit = np.where(cand_ok[:, None], cand_centered / np.maximum(cand_norms, 1e-12), 0.0)

    q_mat = np.zeros((len(query_ids), dim), dtype=np.float64)
    q_ok = np.zeros(len(query_ids), dtype=bool)
    q_arrays: List[Optional[np.ndarray]] = [None] * len(query_ids)
    for j, qid in enumerate(query_ids):
        arr = query_embs_by_id.get(qid)
        if arr is not None and len(arr) > 0:
            a64 = np.asarray(arr, dtype=np.float64)
            q_arrays[j] = a64
            q_mat[j] = a64.mean(axis=0)
            q_ok[j] = True

    q_centered = q_mat - mu_cand
    q_norms = np.linalg.norm(q_centered, axis=1, keepdims=True)
    q_unit = np.where(q_ok[:, None], q_centered / np.maximum(q_norms, 1e-12), 0.0)
    sim = q_unit @ cand_unit.T

    if topk_weight > 0.0 and topk > 0:
        valid_c_idx = [i for i in range(len(cand_ids)) if cand_ok[i]]
        c_chunks = []
        c_counts = []
        for i in valid_c_idx:
            cg = cand_arrays[i] - mu_cand
            cg_u = cg / np.maximum(np.linalg.norm(cg, axis=1, keepdims=True), 1e-12)
            c_chunks.append(cg_u)
            c_counts.append(len(cg_u))
        c_flat = np.vstack(c_chunks)
        uniform_n = c_counts[0] if len(set(c_counts)) == 1 else 0
        offsets = np.concatenate([[0], np.cumsum(c_counts)])

        s_topk = np.zeros((len(query_ids), len(cand_ids)), dtype=np.float64)
        for j in range(len(query_ids)):
            if not q_ok[j]:
                continue
            qg = q_arrays[j] - mu_cand
            qg_u = qg / np.maximum(np.linalg.norm(qg, axis=1, keepdims=True), 1e-12)
            pw = qg_u @ c_flat.T
            if uniform_n > 0:
                k_eff = min(topk, uniform_n)
                pw3 = pw.reshape(len(qg_u), len(valid_c_idx), uniform_n)
                if k_eff == uniform_n:
                    vals = pw3.mean(axis=(0, 2))
                else:
                    vals = np.partition(pw3, uniform_n - k_eff, axis=2)[:, :, -k_eff:].mean(axis=(0, 2))
                s_topk[j, valid_c_idx] = vals
            else:
                for pos, ci in enumerate(valid_c_idx):
                    sl = pw[:, offsets[pos]:offsets[pos + 1]]
                    n_g = sl.shape[1]
                    k_eff = min(topk, n_g)
                    if k_eff == n_g:
                        s_topk[j, ci] = float(sl.mean())
                    else:
                        s_topk[j, ci] = float(np.partition(sl, n_g - k_eff, axis=1)[:, -k_eff:].mean())
        sim = sim + topk_weight * s_topk

    if not cand_ok.all():
        sim[:, ~cand_ok] = float("-inf")
    return sim


def aggregate_and_center_similarity(
    cand_embs_by_id: Dict[str, np.ndarray],
    query_embs_by_id: Dict[str, np.ndarray],
    cand_ids: Sequence[str],
    query_ids: Sequence[str],
) -> np.ndarray:
    """計算多局平均、候選池中心化並 L2 正規化後之餘弦相似度矩陣 (Q, C)。"""
    return compute_emb_similarity(cand_embs_by_id, query_embs_by_id, cand_ids, query_ids, topk_weight=0.0)


def _load_emb_cache_npz(path: str, head_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """載入 infer_player_emb 產出之 .npz 快取，可選合併前段視角 head 快取。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 Embedding 快取檔：{path}")
    with np.load(path, allow_pickle=False) as z:
        embs = z["emb"].astype(np.float64)
        eids = z["entity_id"].astype(str)
        valid = z["valid"].astype(bool) if "valid" in z.files else np.ones(len(eids), dtype=bool)
        sha = str(z["ckpt_sha256"]) if "ckpt_sha256" in z.files else ""

    if head_path is not None:
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"找不到前段視角 Embedding 快取檔：{head_path}")
        with np.load(head_path, allow_pickle=False) as zh:
            h_embs = zh["emb"].astype(np.float64)
            h_eids = zh["entity_id"].astype(str)
            h_valid = zh["valid"].astype(bool)
            h_sha = str(zh["ckpt_sha256"]) if "ckpt_sha256" in zh.files else ""
        if sha and h_sha and sha != h_sha:
            raise ValueError(f"原快取與 head 快取的 ckpt_sha256 不一致：{path} vs {head_path}")
        if not np.array_equal(eids, h_eids):
            raise ValueError(f"原快取與 head 快取的 entity_id 順序不一致：{path} vs {head_path}")
        merge_mask = valid & h_valid
        merged = 0.5 * (embs + h_embs)
        merged_norms = np.linalg.norm(merged, axis=1, keepdims=True)
        merged_unit = merged / np.maximum(merged_norms, 1e-12)
        embs = np.where(merge_mask[:, None], merged_unit, embs)

    grouped: Dict[str, List[np.ndarray]] = {}
    for eid, vec, ok in zip(eids.tolist(), embs, valid.tolist()):
        if ok:
            grouped.setdefault(eid, []).append(vec)
    return {eid: np.vstack(vs) for eid, vs in grouped.items()}


def predict_split_fused(
    test_csv_path: str,
    cand_profiles: Dict[str, dict],
    idf_weights: Dict[tuple, float],
    cand_ranks: Optional[Dict[str, dict]] = None,
    query_ranks: Optional[Dict[str, dict]] = None,
    top_n: int = TOP_N,
    max_moves: int = DEFAULT_MAX_MOVES,
    w_resp: float = W_RESP,
    alpha: float = 0.2,
    delta_thresh: float = 3.0,
    rank_metric: str = "hybrid",
    sigma: float = 1.5,
    color_cond: bool = False,
    pattern_index: Optional[Chi2PatternIndex] = None,
    pattern_beta: float = 0.0,
    pattern_workers: int = 1,
    csls_k: int = 0,
    cand_embs: Optional[Dict[str, np.ndarray]] = None,
    query_embs: Optional[Dict[str, np.ndarray]] = None,
    emb_gamma: float = 0.0,
    emb_top2_weight: float = 0.0,
    emb_tracks: Optional[Sequence[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float]]] = None,
    raw_open: bool = False,
) -> List[List[str]]:
    """結合開局指紋、局部棋形殘差與棋力相容度對單一測試檔案進行融合預測。"""
    test_df = pd.read_csv(test_csv_path)
    cand_names = sorted(cand_profiles.keys())
    results = []

    cand_ranks = cand_ranks or {}
    query_ranks = query_ranks or {}
    use_pattern = pattern_index is not None and pattern_beta > 0.0

    sgf_cols = [c for c in test_df.columns if c.startswith("sgf_")]

    # 預先收集各題有效 SGF 與執色，啟用局部棋形時可整批平行抽取
    parsed_rows = []
    pat_tasks: List[Tuple[str, str]] = []
    pat_row_indices: List[int] = []

    for row_idx, (_, row) in enumerate(test_df.iterrows()):
        qid = str(row["question_id"]).strip()
        num_games = int(row.get("num_games", len(sgf_cols)))
        raw_games: List[Tuple[str, str]] = []
        for i in range(1, num_games + 1):
            s_col = f"sgf_{i}"
            c_col = f"color_{i}"
            if s_col in row and pd.notna(row[s_col]) and str(row[s_col]).strip():
                sgf_content = str(row[s_col])
                color = str(row[c_col]).strip().lower() if c_col in row and pd.notna(row[c_col]) else "b"
                if color not in ("b", "w"):
                    color = "b"
                raw_games.append((sgf_content, color))
                if use_pattern:
                    pat_tasks.append((sgf_content, color))
                    pat_row_indices.append(row_idx)
        parsed_rows.append((qid, raw_games))

    query_pat_counters: Optional[List[Counter]] = None
    if use_pattern:
        flat_counters = extract_patterns_batch(pat_tasks, workers=pattern_workers)
        grouped: List[List[Counter]] = [[] for _ in range(len(parsed_rows))]
        for r_idx, cnt in zip(pat_row_indices, flat_counters):
            grouped[r_idx].append(cnt)
        query_pat_counters = [merge_pattern_counters(cs) for cs in grouped]

    n_q = len(parsed_rows)
    n_c = len(cand_names)
    tracks: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float]] = []
    if emb_gamma > 0.0:
        if emb_tracks is not None:
            tracks = list(emb_tracks)
        elif cand_embs is not None and query_embs is not None:
            tracks = [(cand_embs, query_embs, 1.0)]

    emb_sim_mat = None
    if tracks:
        qids_order = [qid for qid, _ in parsed_rows]
        w_total = sum(w for _, _, w in tracks)
        if w_total <= 0.0:
            raise ValueError("emb_tracks 權重總和必須 > 0")
        acc_mat = np.zeros((n_q, n_c), dtype=np.float64)
        for c_e, q_e, w in tracks:
            m = compute_emb_similarity(
                c_e, q_e, cand_names, qids_order, topk_weight=emb_top2_weight, topk=2
            )
            # 若少數候選完全缺有效向量，改補 0.0 以保留其餘特徵的兜底排序
            m = np.where(np.isfinite(m), m, 0.0)
            acc_mat += (w / w_total) * m
        emb_sim_mat = acc_mat

    score_mat = np.full((n_q, n_c), float("-inf"), dtype=np.float64) if csls_k > 0 else None
    base_mat = np.zeros((n_q, n_c), dtype=np.float64) if csls_k > 0 else None
    move_fn = get_raw_moves if raw_open else get_canonical_moves

    for row_idx, (qid, raw_games) in enumerate(parsed_rows):
        # 抽取題目開局風格
        query_games = [
            (color, move_fn(sgf_content, color, max_moves=max_moves))
            for sgf_content, color in raw_games
        ]
        q_vecs = build_query_vectors(query_games, idf_weights, w_resp=w_resp, color_cond=color_cond)

        pat_scores = score_query_patterns(pattern_index, query_pat_counters[row_idx]) if use_pattern else None
        q_rank_info = query_ranks.get(qid)

        # 評估所有候選人
        valid_candidates = []
        pruned_candidates = []

        for ci, c_pid in enumerate(cand_names):
            style_sim = compute_style_similarity(q_vecs, cand_profiles[c_pid], color_cond=color_cond)
            base_sim = style_sim + pattern_beta * pat_scores.get(c_pid, 0.0) if pat_scores is not None else style_sim
            if emb_sim_mat is not None:
                base_sim += emb_gamma * float(emb_sim_mat[row_idx, ci])

            c_rank_info = cand_ranks.get(c_pid)
            rank_score, passed_mask = compute_rank_compatibility(
                c_rank_info, q_rank_info,
                metric=rank_metric, delta_thresh=delta_thresh, sigma=sigma
            )

            if score_mat is not None and base_mat is not None:
                base_mat[row_idx, ci] = base_sim
                if passed_mask and not math.isinf(rank_score):
                    score_mat[row_idx, ci] = base_sim + alpha * rank_score
                continue

            if passed_mask and not math.isinf(rank_score):
                final_score = base_sim + alpha * rank_score
                valid_candidates.append((final_score, base_sim, c_pid))
            else:
                # 被硬性遮罩剔除，記錄下來以供極端兜底
                pruned_candidates.append((base_sim, c_pid))

        if score_mat is not None:
            continue

        # 依融合總分排序
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [pid for _, _, pid in valid_candidates[:top_n]]

        # 若通過遮罩的候選人數不足 5 人，使用風格相似度最高之被剔除候選人補足
        if len(top_candidates) < top_n:
            pruned_candidates.sort(key=lambda x: x[0], reverse=True)
            for _, pid in pruned_candidates:
                if pid not in top_candidates:
                    top_candidates.append(pid)
                if len(top_candidates) == top_n:
                    break

        # 極端兜底：若仍不足，由字母序補足
        if len(top_candidates) < top_n:
            for pid in cand_names:
                if pid not in top_candidates:
                    top_candidates.append(pid)
                if len(top_candidates) == top_n:
                    break

        results.append([qid] + top_candidates[:top_n])

    # 啟用 CSLS 時於整批分數矩陣套用候選欄去樞紐化後再排序
    if score_mat is not None and base_mat is not None:
        adj_mat = apply_csls_column_penalty(score_mat, csls_k)
        for row_idx, (qid, _) in enumerate(parsed_rows):
            row_vals = adj_mat[row_idx]
            valid_idx = np.flatnonzero(np.isfinite(row_vals))
            valid_sorted = sorted(valid_idx.tolist(), key=lambda i: (-float(row_vals[i]), cand_names[i]))
            top_candidates = [cand_names[i] for i in valid_sorted[:top_n]]

            if len(top_candidates) < top_n:
                pruned_idx = np.flatnonzero(~np.isfinite(row_vals))
                pruned_sorted = sorted(pruned_idx.tolist(), key=lambda i: (-float(base_mat[row_idx, i]), cand_names[i]))
                for i in pruned_sorted:
                    pid = cand_names[i]
                    if pid not in top_candidates:
                        top_candidates.append(pid)
                    if len(top_candidates) == top_n:
                        break

            if len(top_candidates) < top_n:
                for pid in cand_names:
                    if pid not in top_candidates:
                        top_candidates.append(pid)
                    if len(top_candidates) == top_n:
                        break

            results.append([qid] + top_candidates[:top_n])

    return results


def _load_rank_cache(path: Optional[str], label: str) -> Dict[str, dict]:
    """載入棋力快取 JSON，路徑為空或不存在時回傳空字典。"""
    if not path:
        return {}
    if os.path.exists(path):
        print(f"[雙軌融合] 載入{label}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  - 包含 {len(data)} 筆棋力資料")
        return data
    print(f"[警告] 找不到{label} {path}，該分組將退回預設或純開局風格")
    return {}


def run_full_fused_pipeline(
    tests_dir: str = "tests",
    cand_cache_path: str = DEFAULT_CAND_CACHE,
    query_cache_path: str = DEFAULT_QUERY_CACHE,
    out_csv: str = "submission_player.csv",
    max_moves: int = DEFAULT_MAX_MOVES,
    idf_power: float = IDF_POWER,
    w_resp: float = W_RESP,
    alpha: float = 0.2,
    delta_thresh: float = 3.0,
    rank_metric: str = "hybrid",
    sigma: float = 1.5,
    color_cond: bool = False,
    cand2_cache_path: Optional[str] = None,
    query2_cache_path: Optional[str] = None,
    pub2_alpha: Optional[float] = None,
    pub2_delta_thresh: Optional[float] = None,
    pub2_metric: Optional[str] = None,
    pub2_sigma: Optional[float] = None,
    pattern_beta: float = 0.0,
    pattern_eps: float = DEFAULT_PATTERN_EPS,
    pattern_min_df: int = DEFAULT_PATTERN_MIN_DF,
    pattern_workers: int = 1,
    csls_k: int = 0,
    pub2_csls_k: Optional[int] = None,
    emb_cache_dir: Union[None, str, Sequence[str]] = None,
    emb_gamma: float = 0.0,
    pub2_emb_gamma: Optional[float] = None,
    emb_top2_weight: float = 0.0,
    emb_weights: Optional[Sequence[float]] = None,
    head_emb_dirs: Optional[Sequence[str]] = None,
    raw_open: bool = False,
) -> pd.DataFrame:
    """執行全量四份測試題融合推論並輸出提交 CSV。"""
    cand_ranks = _load_rank_cache(cand_cache_path, "候選人棋力快取")
    query_ranks = _load_rank_cache(query_cache_path, "題目棋力快取")
    if bool(cand2_cache_path) != bool(query2_cache_path):
        raise ValueError("cand2_cache_path 與 query2_cache_path 必須同時指定或同時省略")
    if emb_gamma < 0.0 or (pub2_emb_gamma is not None and pub2_emb_gamma < 0.0):
        raise ValueError("emb_gamma 與 pub2_emb_gamma 必須 >= 0")
    if emb_top2_weight < 0.0:
        raise ValueError("emb_top2_weight 必須 >= 0")

    emb_dirs: List[str] = []
    if isinstance(emb_cache_dir, str):
        emb_dirs = [emb_cache_dir]
    elif emb_cache_dir is not None:
        emb_dirs = list(emb_cache_dir)

    if (emb_gamma > 0.0 or (pub2_emb_gamma is not None and pub2_emb_gamma > 0.0)) and not emb_dirs:
        raise ValueError("emb_gamma 或 pub2_emb_gamma > 0 時必須指定 emb_cache_dir")

    norm_emb_dir_set = {os.path.normpath(d) for d in emb_dirs}
    head_dir_set = {os.path.normpath(d) for d in (head_emb_dirs or ())}
    unknown_head = head_dir_set - norm_emb_dir_set
    if unknown_head:
        raise ValueError(f"head_emb_dirs 包含未列於 emb_cache_dir 的目錄：{sorted(unknown_head)}")

    if emb_dirs:
        if emb_weights is None:
            emb_weight_list = [1.0 / len(emb_dirs)] * len(emb_dirs)
        else:
            if len(emb_weights) != len(emb_dirs):
                raise ValueError("emb_weights 長度必須與 emb_cache_dir 目錄數相同")
            if any(w < 0.0 for w in emb_weights) or sum(emb_weights) <= 0.0:
                raise ValueError("emb_weights 必須皆 >= 0 且總和 > 0")
            emb_weight_list = list(emb_weights)
    else:
        emb_weight_list = []
    # 兩份 pub2/priv2 快取同時省略時沿用 pub_/priv_ 的快取
    cand2_ranks = _load_rank_cache(cand2_cache_path, "pub2/priv2 候選人棋力快取") if cand2_cache_path else cand_ranks
    query2_ranks = _load_rank_cache(query2_cache_path, "pub2/priv2 題目棋力快取") if query2_cache_path else query_ranks

    splits_config = [
        ("pub_", "player_identification_test_public.csv", "player_identification_candidates_public.csv", False),
        ("priv_", "player_identification_test_private.csv", "player_identification_candidates_private.csv", False),
        ("pub2_", "player_identification_test2_public.csv", "player_identification_candidates2_public.csv", True),
        ("priv2_", "player_identification_test2_private.csv", "player_identification_candidates2_private.csv", True),
    ]

    all_rows = []
    t_start = time.time()

    print(
        f"\n[雙軌融合] 開始執行四組測試題推論 "
        f"(alpha={alpha}, delta_thresh={delta_thresh}, metric={rank_metric}, pattern_beta={pattern_beta})..."
    )

    for prefix, test_file, cand_file, is_v2 in splits_config:
        test_path = os.path.join(tests_dir, test_file)
        cand_path = os.path.join(tests_dir, cand_file)

        cur_cand_ranks = cand2_ranks if is_v2 else cand_ranks
        cur_query_ranks = query2_ranks if is_v2 else query_ranks
        cur_alpha = pub2_alpha if (is_v2 and pub2_alpha is not None) else alpha
        cur_delta = pub2_delta_thresh if (is_v2 and pub2_delta_thresh is not None) else delta_thresh
        cur_metric = pub2_metric if (is_v2 and pub2_metric is not None) else rank_metric
        cur_sigma = pub2_sigma if (is_v2 and pub2_sigma is not None) else sigma
        cur_csls_k = pub2_csls_k if (is_v2 and pub2_csls_k is not None) else csls_k
        cur_emb_gamma = pub2_emb_gamma if (is_v2 and pub2_emb_gamma is not None) else emb_gamma

        print(
            f"\n[{prefix}] 處理中 (alpha={cur_alpha}, delta={cur_delta}, metric={cur_metric}, "
            f"sigma={cur_sigma}, pat_beta={pattern_beta}, csls_k={cur_csls_k}, "
            f"emb_gamma={cur_emb_gamma}, top2={emb_top2_weight}, head_dirs={len(head_dir_set)}, raw_open={raw_open})..."
        )
        t0 = time.time()
        cand_profiles, idf_weights = build_candidate_profiles_and_idf(
            cand_path, max_moves=max_moves, idf_power=idf_power, w_resp=w_resp,
            color_cond=color_cond, raw_coords=raw_open
        )
        print(f"  - 候選庫開局特徵建置完成 ({time.time() - t0:.1f}s)")

        pattern_index = None
        if pattern_beta > 0.0:
            t0 = time.time()
            cand_pat_counters = build_candidate_pattern_counters(cand_path, workers=pattern_workers)
            pattern_index = build_chi2_pattern_index(
                cand_pat_counters, min_df=pattern_min_df, eps=pattern_eps
            )
            print(f"  - 候選庫局部棋形索引建置完成，詞表 {len(pattern_index.vocab_to_idx)} ({time.time() - t0:.1f}s)")

        emb_tracks = None
        if emb_dirs and cur_emb_gamma > 0.0:
            tag = prefix.rstrip("_")
            emb_tracks = []
            for d_path, w_val in zip(emb_dirs, emb_weight_list):
                has_head = os.path.normpath(d_path) in head_dir_set
                c_head = os.path.join(d_path, f"{tag}_cand_head_emb.npz") if has_head else None
                q_head = os.path.join(d_path, f"{tag}_query_head_emb.npz") if has_head else None
                c_e = _load_emb_cache_npz(os.path.join(d_path, f"{tag}_cand_emb.npz"), head_path=c_head)
                q_e = _load_emb_cache_npz(os.path.join(d_path, f"{tag}_query_emb.npz"), head_path=q_head)
                emb_tracks.append((c_e, q_e, w_val))
            print(f"  - 深度風格 Embedding 快取載入完成（{len(emb_tracks)} 組模型，首組候選 {len(emb_tracks[0][0])} 人、題目 {len(emb_tracks[0][1])} 題）")

        t0 = time.time()
        sub_rows = predict_split_fused(
            test_path, cand_profiles, idf_weights,
            cand_ranks=cur_cand_ranks, query_ranks=cur_query_ranks,
            top_n=TOP_N, max_moves=max_moves, w_resp=w_resp,
            alpha=cur_alpha, delta_thresh=cur_delta,
            rank_metric=cur_metric, sigma=cur_sigma,
            color_cond=color_cond,
            pattern_index=pattern_index,
            pattern_beta=pattern_beta,
            pattern_workers=pattern_workers,
            csls_k=cur_csls_k,
            emb_gamma=cur_emb_gamma,
            emb_top2_weight=emb_top2_weight,
            emb_tracks=emb_tracks,
            raw_open=raw_open,
        )
        print(f"  - 題目推論完成 ({time.time() - t0:.1f}s)")

        # 格式檢查
        for r in sub_rows:
            qid = r[0]
            top5 = r[1:]
            assert qid.startswith(prefix), f"qid {qid} 前綴不符 {prefix}"
            assert len(top5) == TOP_N, f"{qid} Top-{TOP_N} 數量不符"
            assert len(set(top5)) == TOP_N, f"{qid} 候選人重複: {top5}"
            expected_p_prefix = f"{prefix.rstrip('_')}_p_"
            for p in top5:
                assert p.startswith(expected_p_prefix), f"{p} 與題目 {qid} 前綴不一致"

        all_rows.extend(sub_rows)

    sub_df = pd.DataFrame(
        all_rows,
        columns=["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]
    )

    assert len(sub_df) == 1600, f"預期 1600 題，實際產生 {len(sub_df)} 題"
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    tmp_csv = out_csv + ".tmp"
    sub_df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, out_csv)
    print(f"\n[雙軌融合] 全部 1,600 題順利完成！耗時 {time.time() - t_start:.1f}s，已寫入 {out_csv}")
    return sub_df


def main():
    parser = argparse.ArgumentParser(description="雙軌融合圍棋選手辨識評分器")
    parser.add_argument("--tests-dir", default="tests", help="測試與候選庫 CSV 目錄")
    parser.add_argument("--cand-cache", default=DEFAULT_CAND_CACHE, help="候選人棋力快取 JSON")
    parser.add_argument("--query-cache", default=DEFAULT_QUERY_CACHE, help="題目棋力快取 JSON")
    parser.add_argument("--out-csv", default="submission_player.csv", help="輸出提交檔路徑")
    parser.add_argument("--alpha", type=float, default=0.2, help="棋力相容度加權係數")
    parser.add_argument("--delta-thresh", type=float, default=3.0, help="硬性階層遮罩閥值（段位差大於此值剔除，0為不啟用）")
    parser.add_argument("--metric", choices=["overlap", "gaussian", "hybrid"], default="hybrid", help="棋力相容度計算指標")
    parser.add_argument("--sigma", type=float, default=1.5, help="高斯相容度標準差參數")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument("--idf-power", type=float, default=IDF_POWER)
    parser.add_argument("--w-resp", type=float, default=W_RESP)
    parser.add_argument("--color-cond", action="store_true", help="啟用黑白執色分離建檔與比對")
    parser.add_argument("--cand2-cache", default=None, help="pub2/priv2 專用候選人棋力快取 JSON")
    parser.add_argument("--query2-cache", default=None, help="pub2/priv2 專用題目棋力快取 JSON")
    parser.add_argument("--pub2-alpha", type=float, default=None, help="pub2/priv2 專用棋力加權係數")
    parser.add_argument("--pub2-delta-thresh", type=float, default=None, help="pub2/priv2 專用硬遮罩閥值，指定 --cand2-cache 時預設 0")
    parser.add_argument("--pub2-metric", choices=["overlap", "gaussian", "hybrid"], default=None, help="pub2/priv2 專用指標，指定 --cand2-cache 時預設 gaussian")
    parser.add_argument("--pub2-sigma", type=float, default=None, help="pub2/priv2 專用高斯標準差")
    parser.add_argument("--pattern-beta", type=float, default=0.0, help="全盤局部棋形殘差相似度加權係數（0 為停用）")
    parser.add_argument("--pattern-eps", type=float, default=DEFAULT_PATTERN_EPS, help="Chi-squared 殘差分母平滑常數")
    parser.add_argument("--pattern-min-df", type=int, default=DEFAULT_PATTERN_MIN_DF, help="局部棋形詞表最小候選人數門檻")
    parser.add_argument("--pattern-workers", type=int, default=1, help="局部棋形平行抽取行程數")
    parser.add_argument("--csls-k", type=int, default=0, help="候選側 CSLS 去樞紐化鄰居數 k（0 為停用）")
    parser.add_argument("--pub2-csls-k", type=int, default=None, help="pub2/priv2 專用 CSLS 鄰居數 k，未指定時沿用 --csls-k")
    parser.add_argument("--emb-cache-dir", nargs="+", default=None, help="深度風格 Embedding 快取目錄（可傳多個目錄做分數平均集成）")
    parser.add_argument("--emb-gamma", type=float, default=0.0, help="深度風格 Embedding 相似度加權係數（0 為停用）")
    parser.add_argument("--pub2-emb-gamma", type=float, default=None, help="pub2/priv2 專用 Embedding 加權係數，未指定時沿用 --emb-gamma")
    parser.add_argument("--emb-top2-weight", type=float, default=0.0, help="局對局 top2 細粒度相似度加權係數（0 為停用）")
    parser.add_argument("--emb-weights", type=float, nargs="+", default=None, help="多組 Embedding 模型的集成權重，預設等權平均")
    parser.add_argument("--head-emb-dirs", nargs="+", default=None, help="需要合併 *_head_emb.npz 的快取目錄子集，預設空")
    parser.add_argument("--raw-open", action="store_true", help="開局指紋改用不正規化絕對座標（含 HD>=2 換色）")
    args = parser.parse_args()

    if bool(args.cand2_cache) != bool(args.query2_cache):
        parser.error("--cand2-cache 與 --query2-cache 必須同時指定")
    if args.emb_gamma < 0.0 or (args.pub2_emb_gamma is not None and args.pub2_emb_gamma < 0.0):
        parser.error("--emb-gamma 與 --pub2-emb-gamma 必須 >= 0")
    if args.emb_top2_weight < 0.0:
        parser.error("--emb-top2-weight 必須 >= 0")
    if (args.emb_gamma > 0.0 or (args.pub2_emb_gamma is not None and args.pub2_emb_gamma > 0.0)) and not args.emb_cache_dir:
        parser.error("--emb-gamma 或 --pub2-emb-gamma > 0 時必須指定 --emb-cache-dir")
    if args.emb_weights is not None:
        if not args.emb_cache_dir or len(args.emb_weights) != len(args.emb_cache_dir):
            parser.error("--emb-weights 長度必須與 --emb-cache-dir 相同")
        if any(w < 0.0 for w in args.emb_weights) or sum(args.emb_weights) <= 0.0:
            parser.error("--emb-weights 必須皆 >= 0 且總和 > 0")
    if args.head_emb_dirs:
        emb_set = {os.path.normpath(d) for d in (args.emb_cache_dir or ())}
        bad = {os.path.normpath(d) for d in args.head_emb_dirs} - emb_set
        if bad:
            parser.error(f"--head-emb-dirs 包含未列於 --emb-cache-dir 的目錄：{sorted(bad)}")
    # HSL 快取搭配高斯相容度，不啟用硬遮罩
    if args.cand2_cache:
        if args.pub2_metric is None:
            args.pub2_metric = "gaussian"
        if args.pub2_delta_thresh is None:
            args.pub2_delta_thresh = 0.0

    run_full_fused_pipeline(
        tests_dir=args.tests_dir,
        cand_cache_path=args.cand_cache,
        query_cache_path=args.query_cache,
        out_csv=args.out_csv,
        max_moves=args.max_moves,
        idf_power=args.idf_power,
        w_resp=args.w_resp,
        alpha=args.alpha,
        delta_thresh=args.delta_thresh,
        rank_metric=args.metric,
        sigma=args.sigma,
        color_cond=args.color_cond,
        cand2_cache_path=args.cand2_cache,
        query2_cache_path=args.query2_cache,
        pub2_alpha=args.pub2_alpha,
        pub2_delta_thresh=args.pub2_delta_thresh,
        pub2_metric=args.pub2_metric,
        pub2_sigma=args.pub2_sigma,
        pattern_beta=args.pattern_beta,
        pattern_eps=args.pattern_eps,
        pattern_min_df=args.pattern_min_df,
        pattern_workers=args.pattern_workers,
        csls_k=args.csls_k,
        pub2_csls_k=args.pub2_csls_k,
        emb_cache_dir=args.emb_cache_dir,
        emb_gamma=args.emb_gamma,
        pub2_emb_gamma=args.pub2_emb_gamma,
        emb_top2_weight=args.emb_top2_weight,
        emb_weights=args.emb_weights,
        head_emb_dirs=args.head_emb_dirs,
        raw_open=args.raw_open,
    )


if __name__ == "__main__":
    main()
