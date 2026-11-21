"""競賽一真實結構本地模擬評估器。

以留出玩家建構對齊測試集段位密度與局數結構的 B 組模擬題，
並以多 seed 玩家層級配對 bootstrap 比較不同開局指紋與棋力融合方案。
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aicup.fuse_player import compute_rank_compatibility
from aicup.opening_player import (
    DEFAULT_MAX_MOVES,
    TOP_N,
    build_candidate_profiles_from_moves,
    build_query_vectors,
    compute_style_similarity,
    get_canonical_moves,
)
from aicup.scoring import LEVELS

DEFAULT_CONFIGS = (
    # (名稱, 執色分離, 棋力來源, alpha, metric, delta_thresh, sigma)
    ("PV1_flat",        False, None,  0.0, "hybrid",   0.0, 1.5),
    ("PV2_cnn_mask",    False, "cnn", 0.2, "hybrid",   3.0, 1.5),
    ("PV2.1_cc_mask",   True,  "cnn", 0.2, "hybrid",   3.0, 1.5),
    ("PV2.1_cc_gauss",  True,  "cnn", 0.2, "gaussian", 0.0, 1.5),
    ("PV3_hsl_a02_s15", True,  "hsl", 0.2, "gaussian", 0.0, 1.5),
)


def top5_score(true_pid: str, top5: List[str]) -> float:
    """計算單題競賽一 Top-5 得分，命中第 r 名得 exp(-(r-1))。"""
    for idx, pid in enumerate(top5):
        if pid == true_pid:
            return math.exp(-idx)
    return 0.0


def build_hsl_target_features(z) -> np.ndarray:
    """從快取檔組出目標側 7 欄特徵（g_hsl、執黑、HD1 與交互項）。"""
    hsl = z["g_hsl"].astype(np.float64)
    blk = z["d_black"].astype(np.float64)[:, None]
    hd1 = (z["d_hd"] == 1).astype(np.float64)[:, None]
    return np.hstack([hsl, blk, hd1, blk * hd1])


def load_target_rank_weights(cand_cache_path: str) -> np.ndarray:
    """由 pub2/priv2 候選人棋力快取估計 10 級目標分佈權重。"""
    with open(cand_cache_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    vals = [
        v["expected_rank"]
        for k, v in d.items()
        if (k.startswith("pub2_p_") or k.startswith("priv2_p_")) and "expected_rank" in v
    ]
    if not vals:
        raise ValueError(f"{cand_cache_path} 找不到 pub2_p_/priv2_p_ 棋力紀錄")
    hist, _ = np.histogram(vals, bins=np.arange(11))
    weights = hist.astype(np.float64)
    return weights / weights.sum()


def load_val648_dataset(
    train_dir: str,
    val_cache: str,
    pool_cache: str,
    cnn_probs_path: str,
    max_moves: int = DEFAULT_MAX_MOVES,
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """載入留出玩家的對局開局落子、CNN 機率與互斥擬合之 HSL 目標側機率。"""
    tr = np.load(pool_cache, allow_pickle=True)
    va = np.load(val_cache, allow_pickle=True)
    cnn = np.load(cnn_probs_path, allow_pickle=True)

    # 強制驗證擬合池與留出玩家完全互斥
    tr_pids = set(tr["d_player_id"].tolist())
    va_pids = set(va["d_player_id"].tolist())
    if not tr_pids.isdisjoint(va_pids):
        raise ValueError("擬合池與留出玩家重疊")

    tr_m = tr["d_hd"] < 2
    va_m = va["d_hd"] < 2

    # 擬合目標側 HSL 邏輯回歸模型
    X_tr = build_hsl_target_features(tr)[tr_m]
    y_tr = tr["d_rank_idx"][tr_m]
    X_va = build_hsl_target_features(va)[va_m]
    sc = StandardScaler()
    clf = LogisticRegression(C=0.5, max_iter=1000)
    clf.fit(sc.fit_transform(X_tr), y_tr)
    hsl_probs_va = clf.predict_proba(sc.transform(X_va))

    # 對齊 CNN 逐局機率
    cnn_map = {str(k): p.astype(np.float64) for k, p in zip(cnn["key"], cnn["probs"])}

    va_keys = [str(k) for k in va["d_key"][va_m]]
    va_pids_arr = [str(p) for p in va["d_player_id"][va_m]]
    va_ranks_arr = va["d_rank_idx"][va_m].astype(int)
    va_black_arr = va["d_black"][va_m].astype(int)

    key_to_meta = {}
    player_rank: Dict[str, int] = {}
    for i, k in enumerate(va_keys):
        if k not in cnn_map:
            raise ValueError(f"找不到 CNN 機率: {k}")
        pid = va_pids_arr[i]
        rk_idx = int(va_ranks_arr[i])
        player_rank[pid] = rk_idx
        key_to_meta[k] = {
            "key": k,
            "player_id": pid,
            "rank_idx": rk_idx,
            "color": "b" if va_black_arr[i] == 1 else "w",
            "cnn_probs": cnn_map[k],
            "hsl_probs": hsl_probs_va[i],
        }

    # 依等級掃描訓練 CSV 取出對應局的開局落子
    needed_by_rank: Dict[str, set] = {}
    for k in key_to_meta:
        rk, gid = k.split("_", 1)
        needed_by_rank.setdefault(rk, set()).add(gid)

    by_player: Dict[str, List[dict]] = {}
    for rk in LEVELS:
        need = needed_by_rank.get(rk)
        if not need:
            continue
        csv_path = os.path.join(train_dir, f"train_{rk}.csv")
        for chunk in pd.read_csv(csv_path, usecols=["game_id", "color", "sgf_content"], chunksize=50000):
            sub = chunk[chunk["game_id"].isin(need)]
            for _, row in sub.iterrows():
                k = f"{rk}_{row['game_id']}"
                meta = key_to_meta[k]
                col = str(row["color"]).strip().lower()
                if col not in ("b", "w"):
                    col = meta["color"]
                mvs = get_canonical_moves(str(row["sgf_content"]), col, max_moves=max_moves)
                meta["color"] = col
                meta["moves"] = mvs
                by_player.setdefault(meta["player_id"], []).append(meta)

    return by_player, player_rank


def agg_rank_info(games: List[dict], prob_key: str) -> dict:
    """以逐局機率乘積（對數平均）合併多局機率，回傳機率分佈與期望段位。"""
    # 乘積合併會重複計入逐局先驗，比較不同棋力來源時條件相同
    probs = np.array([g[prob_key] for g in games], dtype=np.float64)
    log_p = np.mean(np.log(np.clip(probs, 1e-6, 1.0)), axis=0)
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()
    exp_r = float(np.dot(p, np.arange(10, dtype=np.float64)))
    return {"probs": p.tolist(), "expected_rank": exp_r}


def sample_candidates(
    valid_pids: List[str],
    player_rank: Dict[str, int],
    n_cands: int,
    rng: np.random.Generator,
    rank_weights: Optional[np.ndarray] = None,
) -> List[str]:
    """依指定段位分佈無放回抽樣候選玩家。"""
    if rank_weights is None:
        chosen = rng.choice(valid_pids, size=min(n_cands, len(valid_pids)), replace=False).tolist()
        return sorted(chosen)

    # 依各級玩家數反比調整抽樣機率，逼近目標段位分佈
    rank_counts = np.bincount([player_rank[p] for p in valid_pids], minlength=10).astype(np.float64)
    probs = np.array([
        rank_weights[player_rank[p]] / max(rank_counts[player_rank[p]], 1.0)
        for p in valid_pids
    ], dtype=np.float64)
    # 避免零權重等級在人數不足時失敗，保留極小平滑值
    probs = np.where(probs > 0, probs, 1e-6)
    probs /= probs.sum()
    chosen = rng.choice(valid_pids, size=min(n_cands, len(valid_pids)), replace=False, p=probs).tolist()
    return sorted(chosen)


def evaluate_b_group_seed(
    by_player: Dict[str, List[dict]],
    player_rank: Dict[str, int],
    valid_pids: List[str],
    configs: tuple,
    seed: int,
    n_cands: int = 300,
    n_queries: int = 200,
    rank_weights: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """執行單一 seed 的 B 組模擬，回傳各方案逐題得分、局數與真實 player_id。"""
    rng = np.random.default_rng(seed)
    chosen = sample_candidates(valid_pids, player_rank, n_cands, rng, rank_weights=rank_weights)

    # 切分每人的候選 10 局與保留查詢局，強制互斥
    cand_games_raw: Dict[str, List[dict]] = {}
    query_pool_raw: Dict[str, List[dict]] = {}
    for pid in chosen:
        g = by_player[pid]
        perm = rng.permutation(len(g))
        c_idx = perm[:10]
        q_idx = perm[10:]
        if not set(c_idx.tolist()).isdisjoint(set(q_idx.tolist())):
            raise ValueError(f"玩家 {pid} 候選局與查詢局重疊")
        cand_games_raw[pid] = [g[i] for i in c_idx]
        query_pool_raw[pid] = [g[i] for i in q_idx]

    # 抽取題目（1/2/3/5 局各佔 1/4），其餘候選人為干擾項
    q_pids = rng.choice(chosen, size=min(n_queries, len(chosen)), replace=False).tolist()
    q_sizes = np.array(([1, 2, 3, 5] * (len(q_pids) // 4 + 1))[:len(q_pids)], dtype=int)
    rng.shuffle(q_sizes)

    # 預先建置 flat 與 color_cond 候選檔案
    player_moves = {pid: [(g["color"], g["moves"]) for g in cand_games_raw[pid]] for pid in chosen}
    prof_flat, idf_flat = build_candidate_profiles_from_moves(player_moves, color_cond=False)
    prof_cc, idf_cc = build_candidate_profiles_from_moves(player_moves, color_cond=True)

    cand_ranks_map = {
        "cnn": {pid: agg_rank_info(cand_games_raw[pid], "cnn_probs") for pid in chosen},
        "hsl": {pid: agg_rank_info(cand_games_raw[pid], "hsl_probs") for pid in chosen},
    }

    scores_out = {cfg[0]: np.zeros(len(q_pids), dtype=np.float64) for cfg in configs}

    for qi, (true_pid, qn) in enumerate(zip(q_pids, q_sizes)):
        q_games = query_pool_raw[true_pid][:qn]
        q_moves = [(g["color"], g["moves"]) for g in q_games]
        qv_flat = build_query_vectors(q_moves, idf_flat, color_cond=False)
        qv_cc = build_query_vectors(q_moves, idf_cc, color_cond=True)

        sim_flat = {pid: compute_style_similarity(qv_flat, prof_flat[pid], color_cond=False) for pid in chosen}
        sim_cc = {pid: compute_style_similarity(qv_cc, prof_cc[pid], color_cond=True) for pid in chosen}

        q_rank_map = {
            "cnn": agg_rank_info(q_games, "cnn_probs"),
            "hsl": agg_rank_info(q_games, "hsl_probs"),
        }

        for name, cc, rk_src, alpha, metric, d_thresh, sigma in configs:
            sims = sim_cc if cc else sim_flat
            if rk_src is None or alpha == 0.0:
                ranked = sorted(chosen, key=lambda p: sims[p], reverse=True)[:TOP_N]
            else:
                cr_dict = cand_ranks_map[rk_src]
                qr = q_rank_map[rk_src]
                valid_c = []
                pruned_c = []
                for pid in chosen:
                    r_sc, ok = compute_rank_compatibility(
                        cr_dict[pid], qr,
                        metric=metric, delta_thresh=d_thresh, sigma=sigma,
                    )
                    if ok and not math.isinf(r_sc):
                        valid_c.append((sims[pid] + alpha * r_sc, pid))
                    else:
                        pruned_c.append((sims[pid], pid))
                valid_c.sort(key=lambda x: x[0], reverse=True)
                ranked = [pid for _, pid in valid_c[:TOP_N]]
                if len(ranked) < TOP_N:
                    pruned_c.sort(key=lambda x: x[0], reverse=True)
                    for _, pid in pruned_c:
                        if pid not in ranked:
                            ranked.append(pid)
                        if len(ranked) == TOP_N:
                            break
            scores_out[name][qi] = top5_score(true_pid, ranked)

    return scores_out, q_sizes, np.array(q_pids, dtype=object)


def player_bootstrap_ci(
    diff: np.ndarray,
    pids: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """依真實 player_id 彙整後進行玩家層級配對 bootstrap，回傳平均與 95% 區間。"""
    uniq = sorted(set(pids.tolist()))
    p_idx = {p: i for i, p in enumerate(uniq)}
    g = np.array([p_idx[p] for p in pids], dtype=int)
    p_means = np.bincount(g, weights=diff, minlength=len(uniq)) / np.maximum(np.bincount(g, minlength=len(uniq)), 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot_means = p_means[idx].mean(axis=1)
    return float(p_means.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    parser = argparse.ArgumentParser(description="競賽一 B 組真實結構本地模擬評估")
    parser.add_argument("--train-dir", default="training-v1.1/training")
    parser.add_argument("--val-cache", default="humansl-analysis/cache/val648-6d.npz")
    parser.add_argument("--pool-cache", default="humansl-analysis/cache/trainpool-6d.npz")
    parser.add_argument("--cnn-probs", default="rank-trained-models/9a3b6eb/val_probs.npz")
    parser.add_argument("--cand-cache", default="player-trained-models/rank-cache/cand_ranks.json")
    parser.add_argument("--uniform", action="store_true", help="停用段位加權，改為均勻抽樣")
    parser.add_argument("--n-cands", type=int, default=300, help="每組候選人數")
    parser.add_argument("--n-queries", type=int, default=200, help="每組題數")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    args = parser.parse_args()

    t0 = time.time()
    by_player, player_rank = load_val648_dataset(
        train_dir=args.train_dir,
        val_cache=args.val_cache,
        pool_cache=args.pool_cache,
        cnn_probs_path=args.cnn_probs,
    )
    valid_pids = sorted([pid for pid, gs in by_player.items() if len(gs) >= 15])
    rank_weights = None if args.uniform else load_target_rank_weights(args.cand_cache)
    mode_str = "均勻抽樣" if args.uniform else "依 pub2/priv2 棋力直方圖加權"
    print(f"載入完成 ({time.time() - t0:.1f}s)：可用留出玩家 {len(valid_pids)} 位，模式={mode_str}")

    all_scores = {cfg[0]: [] for cfg in DEFAULT_CONFIGS}
    all_sizes = []
    all_pids = []
    for s in args.seeds:
        t1 = time.time()
        sc, sz, pids = evaluate_b_group_seed(
            by_player, player_rank, valid_pids, DEFAULT_CONFIGS,
            seed=s, n_cands=args.n_cands, n_queries=args.n_queries, rank_weights=rank_weights,
        )
        for k, v in sc.items():
            all_scores[k].append(v)
        all_sizes.append(sz)
        all_pids.append(pids)
        print(f"  seed {s} ({time.time() - t1:.1f}s)：" + "，".join(f"{k}={v.mean():.4f}" for k, v in sc.items()))

    sizes_cat = np.concatenate(all_sizes)
    pids_cat = np.concatenate(all_pids)
    cat = {k: np.concatenate(v) for k, v in all_scores.items()}

    uniq_q = len(set(pids_cat.tolist()))
    print(f"\n=== B 組彙總（{len(args.seeds)} seeds，共 {len(sizes_cat)} 題，{uniq_q} 位獨立查詢玩家，{mode_str}）===")
    print(f"{'方案':<18s} | {'總分':>7s} | {'1 局':>7s} | {'2 局':>7s} | {'3 局':>7s} | {'5 局':>7s}")
    print("-" * 68)
    for cfg in DEFAULT_CONFIGS:
        name = cfg[0]
        v = cat[name]
        parts = [f"{v[sizes_cat == q].mean():.4f}" for q in (1, 2, 3, 5)]
        print(f"{name:<18s} | {v.mean():7.4f} | " + " | ".join(f"{p:>7s}" for p in parts))

    print("\n=== 玩家層級配對 Bootstrap 比較（95% 區間）===")
    pairs = [
        ("PV2.1_cc_mask", "PV2_cnn_mask"),
        ("PV2.1_cc_gauss", "PV2.1_cc_mask"),
        ("PV3_hsl_a02_s15", "PV2.1_cc_mask"),
    ]
    for a, b in pairs:
        diff = cat[a] - cat[b]
        m, lo, hi = player_bootstrap_ci(diff, pids_cat)
        by_q = "，".join(f"{q}局 {diff[sizes_cat == q].mean():+.4f}" for q in (1, 2, 3, 5))
        print(f"{a} 減 {b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] ({by_q})")


if __name__ == "__main__":
    main()
