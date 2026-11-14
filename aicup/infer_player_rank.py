"""基於 GoRankResNet 之選手與題目棋力分佈批次推論模組。

利用競賽二已收斂之 ResNet 棋力模型，解析 SGF 棋譜並推論 10 級棋力機率分佈與期望段位。
支援候選庫與題目資料之批次推論、多局聚合與 JSON 快取儲存。
"""

import argparse
import json
from multiprocessing import Pool
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

# 避免 Windows OpenMP 衝突錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

from aicup.predict_rank import load_model, sgf_to_features
from aicup.scoring import LEVELS, pick_level

CANDIDATE_CKPTS = [
    os.path.join("rank-trained-models", "9a3b6eb", "best_model.pth"),
    os.path.join("rank-trained-models", "59844b9", "best_model.pth"),
]


def get_default_ckpt() -> str:
    """自動偵測可用的 CNN 權重，優先選擇 9a3b6eb。"""
    for p in CANDIDATE_CKPTS:
        if os.path.exists(p):
            return p
    return CANDIDATE_CKPTS[0]


DEFAULT_CKPT = get_default_ckpt()
DEFAULT_CACHE_DIR = os.path.join("player-trained-models", "rank-cache")
RANK_CLASSES = len(LEVELS)


def load_rank_model(
    ckpt_path: str = DEFAULT_CKPT,
    device: Optional[torch.device] = None,
    channels: int = 256,
    blocks: int = 20,
    in_channels: int = 500
) -> torch.nn.Module:
    """載入 GoRankResNet 模型並設為 eval 模式。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(ckpt_path, channels=channels, blocks=blocks, device=device, in_channels=in_channels)



def compute_expected_rank(probs: np.ndarray) -> float:
    """計算給定 10 級機率分佈的期望段位索引值（0.0 至 9.0）。

    等級對應：D=0, C=1, B=2, A=3, 1D=4, 2D=5, 3D=6, 4D=7, 5D=8, 6D=9。
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        assert len(probs) == RANK_CLASSES, f"預期長度 {RANK_CLASSES}，實際為 {len(probs)}"
        weights = np.arange(RANK_CLASSES, dtype=np.float64)
        return float(np.sum(probs * weights))
    elif probs.ndim == 2:
        assert probs.shape[1] == RANK_CLASSES, f"預期類別數 {RANK_CLASSES}，實際為 {probs.shape[1]}"
        weights = np.arange(RANK_CLASSES, dtype=np.float64)
        return np.sum(probs * weights, axis=1)
    else:
        raise ValueError(f"不支援的維度: {probs.ndim}")


def predict_sgf_batch(
    model: torch.nn.Module,
    sgf_list: List[str],
    device: torch.device,
    batch_size: int = 64,
    amp: bool = False,
    nproc: int = 1
) -> np.ndarray:
    """批次預測一組 SGF 棋譜的 10 級機率分佈。

    若某局解析失敗，回傳均勻分佈 [0.1]*10 以確保管線穩定。
    回傳形狀為 (len(sgf_list), 10) 的 numpy array。
    """
    n_sgfs = len(sgf_list)
    if n_sgfs == 0:
        return np.zeros((0, RANK_CLASSES), dtype=np.float32)

    probs = np.full((n_sgfs, RANK_CLASSES), 1.0 / RANK_CLASSES, dtype=np.float32)
    valid_indices = []
    features_buf = []

    model.eval()

    def _eval_and_fill(buf: List[np.ndarray], idxs: List[int]):
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device).float()
        with torch.no_grad():
            if amp and device.type == "cuda":
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    out = model(x)
            else:
                out = model(x)
            batch_probs = torch.softmax(out.float(), dim=-1).cpu().numpy()
        probs[idxs] = batch_probs
        buf.clear()
        idxs.clear()

    pool = Pool(nproc) if nproc > 1 else None
    feats = pool.imap(sgf_to_features, sgf_list, chunksize=16) if pool else map(sgf_to_features, sgf_list)
    t_start = time.time()

    try:
        for i, feat in enumerate(feats):
            if feat is not None:
                features_buf.append(feat)
                valid_indices.append(i)
            if len(features_buf) >= batch_size:
                _eval_and_fill(features_buf, valid_indices)
            if (i + 1) % 1000 == 0 or (i + 1) == n_sgfs:
                elapsed = max(time.time() - t_start, 0.1)
                print(f"    推論進度: {i + 1}/{n_sgfs} 局 ({i + 1} / {elapsed:.1f}s = {(i + 1) / elapsed:.1f} 局/s)", flush=True)

        if features_buf:
            _eval_and_fill(features_buf, valid_indices)
    finally:
        if pool:
            pool.close()
            pool.join()

    return probs


def aggregate_probs(probs_list: Sequence[np.ndarray], agg: str = "mean") -> np.ndarray:
    """聚合多局的機率分佈並重新正規化為和為 1.0。"""
    probs_arr = np.asarray(probs_list, dtype=np.float64)
    if len(probs_arr) == 0:
        return np.full(RANK_CLASSES, 1.0 / RANK_CLASSES, dtype=np.float64)

    if agg == "mean":
        agg_p = np.mean(probs_arr, axis=0)
    elif agg == "log":
        log_p = np.mean(np.log(np.clip(probs_arr, 1e-12, 1.0)), axis=0)
        exp_p = np.exp(log_p - np.max(log_p))
        agg_p = exp_p / np.sum(exp_p)
    else:
        raise ValueError(f"不支援的聚合方式: {agg}")

    s = np.sum(agg_p)
    if s > 0:
        agg_p /= s
    else:
        agg_p = np.full(RANK_CLASSES, 1.0 / RANK_CLASSES, dtype=np.float64)
    return agg_p


def infer_candidates_from_df(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    max_games_per_player: Optional[int] = None,
    batch_size: int = 64,
    amp: bool = False,
    agg: str = "mean",
    nproc: int = 1,
    verbose: bool = True
) -> Dict[str, dict]:
    """從候選庫 DataFrame 推論每位候選人的棋力分佈與期望段位。"""
    grouped = df.groupby("player_id")
    pids = sorted(grouped.groups.keys())

    all_sgfs = []
    pid_to_slice = {}

    for pid in pids:
        sub_df = grouped.get_group(pid)
        sgfs = sub_df["sgf_content"].dropna().astype(str).tolist()
        if max_games_per_player and max_games_per_player > 0:
            sgfs = sgfs[:max_games_per_player]
        start_idx = len(all_sgfs)
        all_sgfs.extend(sgfs)
        end_idx = len(all_sgfs)
        pid_to_slice[pid] = (start_idx, end_idx)

    if verbose:
        print(f"  - 候選人數: {len(pids)}, 總棋局數: {len(all_sgfs)}", flush=True)

    t0 = time.time()
    all_probs = predict_sgf_batch(
        model, all_sgfs, device, batch_size=batch_size, amp=amp, nproc=nproc
    )
    if verbose:
        print(f"  - 批次推論完成 ({time.time() - t0:.1f}s)", flush=True)

    results = {}
    for pid in pids:
        st, ed = pid_to_slice[pid]
        p_slice = all_probs[st:ed]
        agg_p = aggregate_probs(p_slice, agg=agg)
        exp_rank = compute_expected_rank(agg_p)
        pred_lv = LEVELS[pick_level(agg_p)]
        results[pid] = {
            "probs": [round(float(x), 6) for x in agg_p],
            "expected_rank": round(float(exp_rank), 4),
            "pred_rank": pred_lv,
            "num_games": int(ed - st)
        }

    return results


def infer_questions_from_df(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    amp: bool = False,
    agg: str = "mean",
    nproc: int = 1,
    verbose: bool = True
) -> Dict[str, dict]:
    """從測試題 DataFrame 推論每道題目的棋力分佈與期望段位。"""
    all_sgfs = []
    qid_to_slice = {}
    sgf_cols = [c for c in df.columns if c.startswith("sgf_")]

    qids = []
    for _, row in df.iterrows():
        qid = str(row["question_id"]).strip()
        qids.append(qid)
        num_games = int(row.get("num_games", len(sgf_cols)))
        q_sgfs = []
        for i in range(1, num_games + 1):
            col = f"sgf_{i}"
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                q_sgfs.append(str(row[col]))
        start_idx = len(all_sgfs)
        all_sgfs.extend(q_sgfs)
        end_idx = len(all_sgfs)
        qid_to_slice[qid] = (start_idx, end_idx)

    if verbose:
        print(f"  - 題目數: {len(qids)}, 總棋局數: {len(all_sgfs)}", flush=True)

    t0 = time.time()
    all_probs = predict_sgf_batch(
        model, all_sgfs, device, batch_size=batch_size, amp=amp, nproc=nproc
    )
    if verbose:
        print(f"  - 批次推論完成 ({time.time() - t0:.1f}s)", flush=True)

    results = {}
    for qid in qids:
        st, ed = qid_to_slice[qid]
        p_slice = all_probs[st:ed]
        agg_p = aggregate_probs(p_slice, agg=agg)
        exp_rank = compute_expected_rank(agg_p)
        pred_lv = LEVELS[pick_level(agg_p)]
        results[qid] = {
            "probs": [round(float(x), 6) for x in agg_p],
            "expected_rank": round(float(exp_rank), 4),
            "pred_rank": pred_lv,
            "num_games": int(ed - st)
        }

    return results


def save_rank_cache(cache: Dict[str, dict], out_json_path: str):
    """將棋力字典儲存為 JSON 檔案。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
    tmp_path = out_json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_json_path)
    print(f"  - 快取已寫入: {out_json_path} (共 {len(cache)} 筆)", flush=True)


def load_rank_cache(json_path: str) -> Dict[str, dict]:
    """載入棋力快取 JSON 檔案。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_full_rank_inference(
    ckpt_path: str = DEFAULT_CKPT,
    tests_dir: str = "tests",
    out_dir: str = DEFAULT_CACHE_DIR,
    max_games_per_player: Optional[int] = None,
    batch_size: int = 64,
    no_amp: bool = False,
    nproc: int = 1
):
    """執行全量四份候選庫與四份測試題棋力推論，產出 cand_ranks.json 與 query_ranks.json。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda" and not no_amp)
    print(f"[棋力推論] 載入 GoRankResNet: {ckpt_path} (裝置: {device}, AMP: {amp}, nproc: {nproc})")

    model = load_model(ckpt_path, channels=256, blocks=20, device=device, in_channels=500)

    # 1. 候選庫推論
    cand_files = [
        "player_identification_candidates_public.csv",
        "player_identification_candidates_private.csv",
        "player_identification_candidates2_public.csv",
        "player_identification_candidates2_private.csv",
    ]
    all_cands = {}
    print("\n[棋力推論] 開始推論候選庫...")
    for cfile in cand_files:
        cpath = os.path.join(tests_dir, cfile)
        if not os.path.exists(cpath):
            print(f"警告：找不到 {cpath}，略過")
            continue
        print(f"處理候選庫: {cfile} (最大每人抽樣: {max_games_per_player or '全部'})...")
        df = pd.read_csv(cpath)
        cands_dict = infer_candidates_from_df(
            model, df, device,
            max_games_per_player=max_games_per_player,
            batch_size=batch_size,
            amp=amp,
            nproc=nproc
        )
        all_cands.update(cands_dict)

    cand_out = os.path.join(out_dir, "cand_ranks.json")
    save_rank_cache(all_cands, cand_out)

    # 2. 測試題推論
    test_files = [
        "player_identification_test_public.csv",
        "player_identification_test_private.csv",
        "player_identification_test2_public.csv",
        "player_identification_test2_private.csv",
    ]
    all_queries = {}
    print("\n[棋力推論] 開始推論測試題目...")
    for tfile in test_files:
        tpath = os.path.join(tests_dir, tfile)
        if not os.path.exists(tpath):
            print(f"警告：找不到 {tpath}，略過")
            continue
        print(f"處理測試題: {tfile}...")
        df = pd.read_csv(tpath)
        queries_dict = infer_questions_from_df(
            model, df, device,
            batch_size=batch_size,
            amp=amp,
            nproc=nproc
        )
        all_queries.update(queries_dict)

    query_out = os.path.join(out_dir, "query_ranks.json")
    save_rank_cache(all_queries, query_out)

    print("\n[棋力推論] 全量推論完成！")
    print(f"  - 候選人快取: {cand_out} ({len(all_cands)} 人)")
    print(f"  - 題目快取: {query_out} ({len(all_queries)} 題)")


def main():
    parser = argparse.ArgumentParser(description="GoRankResNet 棋力分佈批次推論模組")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="GoRankResNet 權重路徑")
    parser.add_argument("--tests-dir", default="tests", help="測試與候選庫 CSV 目錄")
    parser.add_argument("--out-dir", default=DEFAULT_CACHE_DIR, help="輸出快取目錄")
    parser.add_argument("--max-games", type=int, default=0, help="每位候選人最多抽樣幾盤（0 代表全部）")
    parser.add_argument("--batch-size", type=int, default=64, help="批次推論大小")
    parser.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 4) - 1), help="多行程特徵解析數量")
    parser.add_argument("--no-amp", action="store_true", help="關閉自動混合精度")
    args = parser.parse_args()

    max_games = args.max_games if args.max_games > 0 else None
    run_full_rank_inference(
        ckpt_path=args.ckpt,
        tests_dir=args.tests_dir,
        out_dir=args.out_dir,
        max_games_per_player=max_games,
        batch_size=args.batch_size,
        no_amp=args.no_amp,
        nproc=args.nproc
    )


if __name__ == "__main__":
    main()
