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
    color_cond: bool = False
) -> List[List[str]]:
    """結合開局指紋特徵與棋力相容度對單一測試檔案進行雙軌融合預測。"""
    test_df = pd.read_csv(test_csv_path)
    cand_names = sorted(cand_profiles.keys())
    results = []

    cand_ranks = cand_ranks or {}
    query_ranks = query_ranks or {}

    sgf_cols = [c for c in test_df.columns if c.startswith("sgf_")]

    for _, row in test_df.iterrows():
        qid = str(row["question_id"]).strip()
        num_games = int(row.get("num_games", len(sgf_cols)))

        # 抽取題目開局風格
        query_games = []
        for i in range(1, num_games + 1):
            s_col = f"sgf_{i}"
            c_col = f"color_{i}"
            if s_col in row and pd.notna(row[s_col]) and str(row[s_col]).strip():
                sgf_content = str(row[s_col])
                color = str(row[c_col]).strip().lower() if c_col in row and pd.notna(row[c_col]) else "b"
                if color not in ("b", "w"):
                    color = "b"
                mvs = get_canonical_moves(sgf_content, color, max_moves=max_moves)
                query_games.append((color, mvs))

        q_vecs = build_query_vectors(query_games, idf_weights, w_resp=w_resp, color_cond=color_cond)

        q_rank_info = query_ranks.get(qid)

        # 評估所有候選人
        valid_candidates = []
        pruned_candidates = []

        for c_pid in cand_names:
            style_sim = compute_style_similarity(q_vecs, cand_profiles[c_pid], color_cond=color_cond)

            c_rank_info = cand_ranks.get(c_pid)
            rank_score, passed_mask = compute_rank_compatibility(
                c_rank_info, q_rank_info,
                metric=rank_metric, delta_thresh=delta_thresh, sigma=sigma
            )

            if passed_mask and not math.isinf(rank_score):
                final_score = style_sim + alpha * rank_score
                valid_candidates.append((final_score, style_sim, c_pid))
            else:
                # 被硬性遮罩剔除，記錄下來以供極端兜底
                pruned_candidates.append((style_sim, c_pid))

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

    return results


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
    color_cond: bool = False
) -> pd.DataFrame:
    """執行全量四份測試題雙軌融合推論並輸出提交 CSV。"""
    # 載入棋力快取（若存在）
    cand_ranks = {}
    query_ranks = {}
    if os.path.exists(cand_cache_path):
        print(f"[雙軌融合] 載入候選人棋力快取: {cand_cache_path}")
        with open(cand_cache_path, "r", encoding="utf-8") as f:
            cand_ranks = json.load(f)
        print(f"  - 包含 {len(cand_ranks)} 位候選人棋力資料")
    else:
        print(f"[警告] 找不到候選人棋力快取 {cand_cache_path}，將以純開局風格退行運算！")

    if os.path.exists(query_cache_path):
        print(f"[雙軌融合] 載入題目棋力快取: {query_cache_path}")
        with open(query_cache_path, "r", encoding="utf-8") as f:
            query_ranks = json.load(f)
        print(f"  - 包含 {len(query_ranks)} 道題目棋力資料")
    else:
        print(f"[警告] 找不到題目棋力快取 {query_cache_path}，將以純開局風格退行運算！")

    splits_config = [
        ("pub_", "player_identification_test_public.csv", "player_identification_candidates_public.csv"),
        ("priv_", "player_identification_test_private.csv", "player_identification_candidates_private.csv"),
        ("pub2_", "player_identification_test2_public.csv", "player_identification_candidates2_public.csv"),
        ("priv2_", "player_identification_test2_private.csv", "player_identification_candidates2_private.csv"),
    ]

    all_rows = []
    t_start = time.time()

    print(f"\n[雙軌融合] 開始執行四組測試題推論 (alpha={alpha}, delta_thresh={delta_thresh}, metric={rank_metric})...")

    for prefix, test_file, cand_file in splits_config:
        test_path = os.path.join(tests_dir, test_file)
        cand_path = os.path.join(tests_dir, cand_file)

        print(f"\n[{prefix}] 處理中...")
        t0 = time.time()
        cand_profiles, idf_weights = build_candidate_profiles_and_idf(
            cand_path, max_moves=max_moves, idf_power=idf_power, w_resp=w_resp,
            color_cond=color_cond
        )
        print(f"  - 候選庫特徵建置完成 ({time.time() - t0:.1f}s)")

        t0 = time.time()
        sub_rows = predict_split_fused(
            test_path, cand_profiles, idf_weights,
            cand_ranks=cand_ranks, query_ranks=query_ranks,
            top_n=TOP_N, max_moves=max_moves, w_resp=w_resp,
            alpha=alpha, delta_thresh=delta_thresh,
            rank_metric=rank_metric, sigma=sigma,
            color_cond=color_cond
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
    args = parser.parse_args()

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
        color_cond=args.color_cond
    )


if __name__ == "__main__":
    main()
