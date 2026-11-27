"""競賽一 pub2/priv2 目標側 Human SL 棋力推論與快取建置模組。

以競賽二 trainpool、val648、newdev 的目標側 6d 分析擬合 7 欄邏輯回歸模型，讀取 analyze_humansl 輸出之 jsonl，
依對數平均合併多局 10 級機率，輸出供 fuse_player 使用之候選人與題目棋力快取 JSON。
"""

import argparse
import json
import os
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aicup.fit_humansl import load_games
from aicup.scoring import LEVELS

EXPECTED_PROFILES = ("rank_11k", "rank_1d", "rank_6d")
DEFAULT_TRAIN = (
    "humansl-analysis/hist6d-trainpool/trainpool-p3-hist.jsonl",
    "humansl-analysis/hist6d-val648/val648-p3-hist.jsonl",
    "humansl-analysis/hist6d-newdev/newdev-p3-hist.jsonl",
)
DEFAULT_C = 0.5


def build_target_features(hsl: np.ndarray, black: np.ndarray, hd: np.ndarray) -> np.ndarray:
    """組合目標側 7 欄特徵（g_hsl 4 欄、執黑、HD1 與交互項）。"""
    blk = black.astype(np.float64)[:, None]
    hd1 = (hd == 1).astype(np.float64)[:, None]
    return np.hstack([hsl.astype(np.float64), blk, hd1, blk * hd1])


def fit_hsl_lr(X: np.ndarray, y: np.ndarray, c_val: float = DEFAULT_C) -> Tuple[StandardScaler, LogisticRegression]:
    """標準化後擬合目標側邏輯回歸，回傳 (scaler, clf)。"""
    scaler = StandardScaler()
    clf = LogisticRegression(C=c_val, max_iter=1000)
    clf.fit(scaler.fit_transform(X), y)
    return scaler, clf


def combine_log_mean(probs: np.ndarray) -> Tuple[np.ndarray, float]:
    """以逐局機率對數平均合併多局，回傳 (10 級機率分佈, 期望段位)。"""
    log_p = np.mean(np.log(np.clip(probs, 1e-6, 1.0)), axis=0)
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()
    return p, float(np.dot(p, np.arange(10, dtype=np.float64)))


def load_target_jsonl(jsonl_path: str, expected_profiles: Sequence[str] = EXPECTED_PROFILES) -> dict:
    """讀目標側 jsonl，回傳 fit_humansl.load_games 的結果另加 hsl（3 個 profile 平均 log 機率與 log 手數）。

    profiles 不符、缺 color 或 hd 欄位、或 hsl 含非有限數值時丟 ValueError。
    """
    data = load_games(jsonl_path)
    if tuple(data["profiles"] or ()) != tuple(expected_profiles):
        raise ValueError(f"{jsonl_path} 的 profiles 不符：預期 {tuple(expected_profiles)}，實際 {data['profiles']}")
    if "black" not in data or "hd" not in data:
        raise ValueError(f"{jsonl_path} 缺少 color 或 hd 欄位")
    data["hsl"] = data["x"][:, : len(expected_profiles) + 1]
    if not np.isfinite(data["hsl"]).all():
        raise ValueError(f"{jsonl_path} 特徵含非有限數值")
    return data


def load_train_arrays(path: str) -> dict:
    """擬合資料的 hsl、black、hd、rank_idx、player_id；.npz 是 feature_cache 的快取，其他視為目標側 jsonl。"""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: z[src] for k, src in (("hsl", "g_hsl"), ("black", "d_black"), ("hd", "d_hd"),
                                         ("rank_idx", "d_rank_idx"), ("player_id", "d_player_id"))}
    return load_target_jsonl(path)


def fit_target_hsl_model(
    train_paths: Sequence[str] = DEFAULT_TRAIN,
    c_val: float = DEFAULT_C,
) -> Tuple[StandardScaler, LogisticRegression, int]:
    """依序讀取擬合資料（過濾 HD < 2）擬合目標側 7 欄邏輯回歸模型，回傳 (scaler, clf, 玩家數)。"""
    xs, ys = [], []
    seen_players = set()

    for path in train_paths:
        z = load_train_arrays(path)
        mask = z["hd"] < 2
        xs.append(build_target_features(z["hsl"][mask], z["black"][mask], z["hd"][mask]))
        ys.append(z["rank_idx"][mask].astype(int))
        seen_players.update(z["player_id"][mask].tolist())

    scaler, clf = fit_hsl_lr(np.vstack(xs), np.concatenate(ys), c_val)
    return scaler, clf, len(seen_players)


def aggregate_entity_probs(probs: np.ndarray) -> dict:
    """以逐局機率對數平均合併多局機率，回傳機率分佈、期望段位與預測等級。"""
    p, exp_r = combine_log_mean(probs)
    pred_idx = int(np.argmax(p))
    return {
        "probs": [round(float(v), 6) for v in p],
        "expected_rank": round(exp_r, 4),
        "pred_rank": LEVELS[pred_idx],
        "num_games": int(len(probs)),
    }


def infer_from_jsonl(
    jsonl_path: str,
    scaler: StandardScaler,
    clf: LogisticRegression,
    expected_profiles: Sequence[str] = EXPECTED_PROFILES,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """對 pub2/priv2 Human SL jsonl 進行逐局推論並依實體合併，回傳 (cand_ranks, query_ranks)。"""
    data = load_target_jsonl(jsonl_path, expected_profiles)
    X = build_target_features(data["hsl"], data["black"], data["hd"])

    game_probs = clf.predict_proba(scaler.transform(X))
    by_entity: Dict[str, list] = {}
    cand_ids = set()
    for key, pid, p in zip(data["key"], data["player_id"], game_probs):
        by_entity.setdefault(str(pid), []).append(p)
        if str(key).startswith("cand2_"):
            cand_ids.add(str(pid))

    cand_ranks: Dict[str, dict] = {}
    query_ranks: Dict[str, dict] = {}
    for eid, plist in sorted(by_entity.items()):
        rec = aggregate_entity_probs(np.asarray(plist, dtype=np.float64))
        if eid in cand_ids:
            cand_ranks[eid] = rec
        else:
            query_ranks[eid] = rec

    return cand_ranks, query_ranks


def _write_json(path: str, obj: dict) -> None:
    """以暫存檔原子寫入 JSON。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="競賽一 pub2/priv2 目標側 Human SL 棋力推論")
    parser.add_argument("--jsonl", default="player-trained-models/hsl-cache/pub2-p3-hist.jsonl")
    parser.add_argument("--train", nargs="+", default=list(DEFAULT_TRAIN),
                        help="擬合用的目標側 jsonl 或 feature_cache 快取，依序合併")
    parser.add_argument("--c-val", type=float, default=DEFAULT_C)
    parser.add_argument("--out-cand", default="player-trained-models/hsl-cache/cand2_hsl_ranks.json")
    parser.add_argument("--out-query", default="player-trained-models/hsl-cache/query2_hsl_ranks.json")
    parser.add_argument("--expect-cand", type=int, default=1200, help="預期候選人數，不符則拒絕寫檔")
    parser.add_argument("--expect-query", type=int, default=800, help="預期題目數，不符則拒絕寫檔")
    args = parser.parse_args()

    scaler, clf, n_players = fit_target_hsl_model(args.train, c_val=args.c_val)
    print(f"已擬合目標側 7 欄 HSL 模型（共 {n_players} 位競賽二訓練玩家，C={args.c_val}）")

    cand_ranks, query_ranks = infer_from_jsonl(args.jsonl, scaler, clf)
    if len(cand_ranks) != args.expect_cand or len(query_ranks) != args.expect_query:
        raise ValueError(
            f"實體數不符：候選人 {len(cand_ranks)}（預期 {args.expect_cand}），題目 {len(query_ranks)}（預期 {args.expect_query}）"
        )
    _write_json(args.out_cand, cand_ranks)
    _write_json(args.out_query, query_ranks)

    # 列印候選人 HSL 期望段位與預測等級分佈以供核對
    er_vals = np.array([v["expected_rank"] for v in cand_ranks.values()], dtype=np.float64)
    er_hist = np.histogram(er_vals, bins=np.arange(11))[0]
    pred_counts = {rk: sum(1 for v in cand_ranks.values() if v["pred_rank"] == rk) for rk in LEVELS}
    print(f"已寫入候選人快取 {args.out_cand}（{len(cand_ranks)} 人）與題目快取 {args.out_query}（{len(query_ranks)} 題）")
    print("候選人期望段位直方圖：" + " ".join(f"{LEVELS[i]}:{er_hist[i]}" for i in range(10)))
    print("候選人 argmax 分佈：" + " ".join(f"{rk}:{pred_counts[rk]}" for rk in LEVELS))


if __name__ == "__main__":
    main()
