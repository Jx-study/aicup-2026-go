"""競賽一 pub2/priv2 Human SL 推論清單產生工具。

掃描 tests/ 下之 candidates2 與 test2 四份 CSV，輸出統一格式之清單 CSV
（欄位為 key、group、entity_id、color、sgf_content），供 analyze_humansl --manifest 使用。
"""

import argparse
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

PUB2_FILES = (
    ("cand2_pub", "player_identification_candidates2_public.csv", "cand"),
    ("cand2_priv", "player_identification_candidates2_private.csv", "cand"),
    ("test2_pub", "player_identification_test2_public.csv", "test"),
    ("test2_priv", "player_identification_test2_private.csv", "test"),
)

HD_RE = re.compile(r"HD\[(\d+)\]")


def extract_hd(sgf_str: str) -> int:
    """從 SGF 字串解析 HD 數值，缺少時回傳 -1。"""
    m = HD_RE.search(sgf_str)
    return int(m.group(1)) if m else -1


def _append_game(
    rows: List[dict],
    hd_counts: Counter,
    key: str,
    group: str,
    entity_id: str,
    color_raw,
    sgf_s: str,
    ctx: str,
) -> None:
    """驗證單局顏色、累計 HD 分佈並寫入清單列。"""
    col = str(color_raw).strip().upper()
    if col not in ("B", "W"):
        raise ValueError(f"{ctx} 出現未知顏色: {col}")
    hd_counts[extract_hd(sgf_s)] += 1
    rows.append({
        "key": key,
        "group": group,
        "entity_id": entity_id,
        "color": col,
        "sgf_content": sgf_s,
    })


def build_pub2_manifest(tests_dir: str = "tests") -> Tuple[pd.DataFrame, Dict[int, int]]:
    """讀取 pub2/priv2 四份檔案，回傳清單 DataFrame 與 HD 分佈統計。"""
    rows: List[dict] = []
    hd_counts: Counter = Counter()

    for group, fname, kind in PUB2_FILES:
        path = os.path.join(tests_dir, fname)
        df = pd.read_csv(path)
        if kind == "cand":
            for _, r in df.iterrows():
                gid = str(r["game_id"]).strip()
                pid = str(r["player_id"]).strip()
                _append_game(rows, hd_counts, f"{group}_{gid}", group, pid, r["color"], str(r["sgf_content"]), fname)
        else:
            sgf_cols = [c for c in df.columns if c.startswith("sgf_")]
            for _, r in df.iterrows():
                qid = str(r["question_id"]).strip()
                num_g = int(r.get("num_games", len(sgf_cols)))
                idx_out = 0
                for i in range(1, num_g + 1):
                    s_col = f"sgf_{i}"
                    c_col = f"color_{i}"
                    if s_col in r and pd.notna(r[s_col]) and str(r[s_col]).strip():
                        idx_out += 1
                        _append_game(
                            rows, hd_counts, f"{qid}_g{idx_out:02d}", group, qid,
                            r[c_col], str(r[s_col]), f"{fname} {qid}",
                        )

    out_df = pd.DataFrame(rows, columns=["key", "group", "entity_id", "color", "sgf_content"])
    if out_df["key"].duplicated().any():
        raise ValueError("清單出現重複 key")
    return out_df, dict(hd_counts)


def main():
    parser = argparse.ArgumentParser(description="產生競賽一 pub2/priv2 Human SL 推論清單")
    parser.add_argument("--tests-dir", default="tests", help="競賽一測試 CSV 目錄")
    parser.add_argument("--out", default="player-trained-models/hsl-cache/player_hsl_pub2_manifest.csv")
    args = parser.parse_args()

    out_df, hd_counts = build_pub2_manifest(args.tests_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    out_df.to_csv(tmp, index=False)
    os.replace(tmp, args.out)

    by_group = out_df["group"].value_counts().to_dict()
    print(f"清單已寫入 {args.out}：共 {len(out_df)} 局，分組 {by_group}，HD 分佈 {hd_counts}")


if __name__ == "__main__":
    main()
