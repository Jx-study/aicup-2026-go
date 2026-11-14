"""本地模擬題雙軌融合（開局風格與棋力分佈）網格搜索與評估腳本。

建立基準測試集，使用 GoRankResNet 推論棋力分佈，
網格搜索相容度加權係數、遮罩閥值與算法，輸出評估指標與最佳超參數。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from aicup.eval_player import (
    evaluate_predictions,
    load_benchmark_players,
    load_balanced_benchmark_players,
)
from aicup.fuse_player import compute_rank_compatibility
from aicup.infer_player_rank import (
    aggregate_probs,
    compute_expected_rank,
    get_default_ckpt,
    load_rank_model,
    predict_sgf_batch,
)
from aicup.opening_player import (
    DEFAULT_MAX_MOVES,
    IDF_POWER,
    TOP_N,
    W_RESP,
    extract_raw_counts,
    get_canonical_moves,
)

DEFAULT_CACHE = os.path.join("player-trained-models", "rank-cache", "benchmark_ranks.json")


def prepare_benchmark_features(
    players: Dict[str, List[str]],
    cand_games: int = 20,
    query_games: int = 5,
    max_moves: int = DEFAULT_MAX_MOVES,
    idf_power: float = IDF_POWER,
    w_resp: float = W_RESP
) -> Tuple[Dict[str, Dict[tuple, float]], Dict[str, Dict[tuple, float]], Dict[str, str]]:
    """抽取 400 人開局特徵向量。"""
    cand_raw = {}
    query_raw = {}
    df_counts = Counter()
    n_cands = len(players)

    for pid, sgfs in players.items():
        c_mvs = [get_canonical_moves(s, "b", max_moves=max_moves) for s in sgfs[:cand_games]]
        q_mvs = [get_canonical_moves(s, "b", max_moves=max_moves) for s in sgfs[cand_games:cand_games + query_games]]
        c_cnt = extract_raw_counts(c_mvs, w_resp=w_resp)
        q_cnt = extract_raw_counts(q_mvs, w_resp=w_resp)
        cand_raw[pid] = c_cnt
        query_raw[pid] = q_cnt
        for k in c_cnt.keys():
            df_counts[k] += 1

    # 計算全域特徵 IDF
    idf = {}
    for k, v in df_counts.items():
        if idf_power > 0:
            idf[k] = (math.log((n_cands + 1) / (v + 1)) + 1.0) ** idf_power
        else:
            idf[k] = 1.0

    # 候選人與題目單位特徵向量
    cand_vecs = {}
    for pid, cnt in cand_raw.items():
        w = {k: v * idf.get(k, 1.0) for k, v in cnt.items()}
        norm = math.sqrt(sum(x * x for x in w.values())) + 1e-9
        cand_vecs[pid] = {k: x / norm for k, x in w.items()}

    query_vecs = {}
    for pid, cnt in query_raw.items():
        w = {k: v * idf.get(k, 1.0) for k, v in cnt.items()}
        norm = math.sqrt(sum(x * x for x in w.values())) + 1e-9
        query_vecs[pid] = {k: x / norm for k, x in w.items()}

    gt = {pid: pid for pid in players.keys()}
    return cand_vecs, query_vecs, gt


def compute_or_load_benchmark_ranks(
    players: Dict[str, List[str]],
    cache_path: str = DEFAULT_CACHE,
    ckpt_path: Optional[str] = None,
    cand_games: int = 20,
    query_games: int = 5,
    max_cand_rank_games: int = 5,
    batch_size: int = 64,
    nproc: int = 4
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """推論或載入 400 人的棋力分佈與期望段位。"""
    if os.path.exists(cache_path):
        print(f"[Benchmark] 讀取既有棋力快取: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["cand_ranks"], data["query_ranks"]

    ckpt_path = ckpt_path or get_default_ckpt()
    print(f"[Benchmark] 啟動棋力推論建檔: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_rank_model(ckpt_path, device=device)

    all_cand_sgfs = []
    cand_slice = {}
    all_query_sgfs = []
    query_slice = {}

    for pid, sgfs in players.items():
        # 候選庫取前 max_cand_rank_games 盤以加速推論
        c_sgfs = sgfs[:min(cand_games, max_cand_rank_games)]
        q_sgfs = sgfs[cand_games:cand_games + query_games]

        st_c = len(all_cand_sgfs)
        all_cand_sgfs.extend(c_sgfs)
        cand_slice[pid] = (st_c, len(all_cand_sgfs))

        st_q = len(all_query_sgfs)
        all_query_sgfs.extend(q_sgfs)
        query_slice[pid] = (st_q, len(all_query_sgfs))

    print(f"  - 候選局總數: {len(all_cand_sgfs)}, 題目局總數: {len(all_query_sgfs)}")
    t0 = time.time()
    all_cand_probs = predict_sgf_batch(model, all_cand_sgfs, device, batch_size=batch_size, nproc=nproc)
    all_query_probs = predict_sgf_batch(model, all_query_sgfs, device, batch_size=batch_size, nproc=nproc)
    print(f"  - 棋力推論完畢，耗時: {time.time() - t0:.1f}s")

    cand_ranks = {}
    for pid, (st, ed) in cand_slice.items():
        p_slice = all_cand_probs[st:ed]
        agg_p = aggregate_probs(p_slice)
        exp_r = compute_expected_rank(agg_p)
        cand_ranks[pid] = {
            "probs": [round(float(x), 6) for x in agg_p],
            "expected_rank": round(float(exp_r), 4),
            "num_games": int(ed - st)
        }

    query_ranks = {}
    for pid, (st, ed) in query_slice.items():
        p_slice = all_query_probs[st:ed]
        agg_p = aggregate_probs(p_slice)
        exp_r = compute_expected_rank(agg_p)
        query_ranks[pid] = {
            "probs": [round(float(x), 6) for x in agg_p],
            "expected_rank": round(float(exp_r), 4),
            "num_games": int(ed - st)
        }

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"cand_ranks": cand_ranks, "query_ranks": query_ranks}, f, indent=2)
    print(f"[Benchmark] 棋力快取已存檔至: {cache_path}")

    return cand_ranks, query_ranks


def evaluate_fusion(
    cand_vecs: Dict[str, Dict[tuple, float]],
    query_vecs: Dict[str, Dict[tuple, float]],
    gt: Dict[str, str],
    cand_ranks: Optional[Dict[str, dict]] = None,
    query_ranks: Optional[Dict[str, dict]] = None,
    alpha: float = 0.5,
    delta_thresh: float = 3.0,
    metric: str = "overlap",
    sigma: float = 1.5,
    top_n: int = TOP_N
) -> Dict[str, float]:
    """評估給定超參數組合的得分。"""
    cand_names = sorted(cand_vecs.keys())
    cand_ranks = cand_ranks or {}
    query_ranks = query_ranks or {}

    predictions = {}
    for q_pid, qv in query_vecs.items():
        q_rank = query_ranks.get(q_pid)

        valid_cands = []
        pruned_cands = []

        for c_pid in cand_names:
            cv = cand_vecs[c_pid]
            style_sim = sum(val * cv[k] for k, val in qv.items() if k in cv)

            c_rank = cand_ranks.get(c_pid)
            rank_score, passed = compute_rank_compatibility(
                c_rank, q_rank, metric=metric, delta_thresh=delta_thresh, sigma=sigma
            )

            if passed and not math.isinf(rank_score):
                final_score = style_sim + alpha * rank_score
                valid_cands.append((final_score, style_sim, c_pid))
            else:
                pruned_cands.append((style_sim, c_pid))

        valid_cands.sort(key=lambda x: x[0], reverse=True)
        top_cands = [pid for _, _, pid in valid_cands[:top_n]]

        if len(top_cands) < top_n:
            pruned_cands.sort(key=lambda x: x[0], reverse=True)
            for _, pid in pruned_cands:
                if pid not in top_cands:
                    top_cands.append(pid)
                if len(top_cands) == top_n:
                    break

        predictions[q_pid] = top_cands[:top_n]

    return evaluate_predictions(predictions, gt)


def run_grid_search(
    cand_vecs, query_vecs, gt,
    cand_ranks, query_ranks,
    alpha_list: List[float],
    delta_thresh_list: List[float],
    metrics: List[str]
):
    """執行全網格搜索並輸出格式化表格。"""
    # 基準分數 (alpha=0, 無遮罩)
    base_res = evaluate_fusion(
        cand_vecs, query_vecs, gt, cand_ranks, query_ranks,
        alpha=0.0, delta_thresh=0.0
    )
    print("\n" + "=" * 80)
    print(f"【Baseline 純開局風格】：Score = {base_res['score']:.4f} | Top-1 = {base_res['top1_acc']*100:.1f}% | Top-5 = {base_res['top5_acc']*100:.1f}%")
    print("=" * 80)

    best_score = base_res["score"]
    best_config = {"alpha": 0.0, "delta_thresh": 0.0, "metric": "none"}
    best_metrics = base_res

    results_table = []

    for metric in metrics:
        for delta in delta_thresh_list:
            for alpha in alpha_list:
                res = evaluate_fusion(
                    cand_vecs, query_vecs, gt, cand_ranks, query_ranks,
                    alpha=alpha, delta_thresh=delta, metric=metric
                )
                diff = res["score"] - base_res["score"]
                results_table.append({
                    "metric": metric,
                    "delta": delta,
                    "alpha": alpha,
                    "score": res["score"],
                    "diff": diff,
                    "top1": res["top1_acc"],
                    "top5": res["top5_acc"]
                })
                if res["score"] > best_score:
                    best_score = res["score"]
                    best_config = {"alpha": alpha, "delta_thresh": delta, "metric": metric}
                    best_metrics = res

    # 排序並輸出前 15 名配置
    results_table.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n【網格搜索 Top-15 最優配置】（共 {len(results_table)} 組）：")
    print(f"{'Metric':<10} | {'Delta':<6} | {'Alpha':<6} | {'Score':<8} | {'Delta_Base':<10} | {'Top-1':<7} | {'Top-5':<7}")
    print("-" * 72)
    for row in results_table[:15]:
        print(f"{row['metric']:<10} | {row['delta']:<6.1f} | {row['alpha']:<6.2f} | {row['score']:<8.4f} | {row['diff']:+<10.4f} | {row['top1']*100:<6.1f}% | {row['top5']*100:<6.1f}%")

    print("\n" + "=" * 80)
    print(f"【最佳配置】: metric={best_config['metric']}, delta_thresh={best_config['delta_thresh']}, alpha={best_config['alpha']}")
    print(f"  Score: {best_metrics['score']:.4f} (增益: {best_metrics['score'] - base_res['score']:+.4f})")
    print(f"  Top-1: {best_metrics['top1_acc']*100:.1f}%")
    print(f"  Top-5: {best_metrics['top5_acc']*100:.1f}%")
    print("=" * 80)
    return best_config, best_metrics


def main():
    parser = argparse.ArgumentParser(description="400 人本地模擬題雙軌融合網格搜索")
    parser.add_argument("--train-dir", default="training")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--balanced", action="store_true", help="使用 10 個等級均勻分佈之選手資料")
    parser.add_argument("--n-per-level", type=int, default=20, help="--balanced 時每個等級抽幾位選手")
    parser.add_argument("--n-players", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    t0 = time.time()
    cache_path = args.cache
    if args.balanced:
        if cache_path == DEFAULT_CACHE:
            cache_path = os.path.join("player-trained-models", "rank-cache", "benchmark_balanced_ranks.json")
        print(f"[Benchmark] 載入跨 10 級平衡選手（每級 {args.n_per_level} 人）...")
        players = load_balanced_benchmark_players(train_dir=args.train_dir, n_per_level=args.n_per_level, min_games=25)
    else:
        print(f"[Benchmark] 載入 {args.n_players} 位選手資料...")
        players = load_benchmark_players(train_dir=args.train_dir, n_players=args.n_players, min_games=25)
    print(f"  - 成功載入 {len(players)} 位選手 ({time.time() - t0:.1f}s)")

    print("[Benchmark] 抽取開局特徵...")
    cand_vecs, query_vecs, gt = prepare_benchmark_features(players)

    print("[Benchmark] 準備棋力向量...")
    cand_ranks, query_ranks = compute_or_load_benchmark_ranks(
        players, cache_path=cache_path, ckpt_path=args.ckpt,
        batch_size=args.batch_size, nproc=args.nproc
    )

    alpha_list = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
    delta_list = [0.0, 2.0, 2.5, 3.0, 3.5, 4.0]
    metrics = ["overlap", "gaussian", "hybrid"]

    print(f"[Benchmark] 啟動超參數網格搜索 (共 {len(alpha_list)*len(delta_list)*len(metrics)} 組)...")
    run_grid_search(
        cand_vecs, query_vecs, gt,
        cand_ranks, query_ranks,
        alpha_list=alpha_list,
        delta_thresh_list=delta_list,
        metrics=metrics
    )


if __name__ == "__main__":
    main()
