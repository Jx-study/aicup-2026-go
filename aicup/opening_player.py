"""基於開局指紋（Opening Signature）的圍棋選手識別 Baseline。

功能：
1. 從 SGF 提取目標選手前 N 手落子（支援執黑 B 或執白 W）。
2. 依目標選手的第一手落子，計算 8 軸幾何對稱群（D4 dihedral symmetry）的最佳旋轉/鏡射變換，
   將整個開局棋譜正規化至標準頂角區域（Canonical Octant）。
3. 統計選手的開局落點分佈（單步位置、手數序號、相鄰兩手 Bigram），建立稀疏指紋特徵向量。
4. 對測試題的多局查詢進行特徵累計，比對候選庫各選手的餘弦相似度，輸出 Top-5。
5. 支援原測試集（pub_/priv_，執黑，每人 100 盤）與新增測試集（pub2_/priv2_，黑白混合，每人 10 盤）。
"""

import argparse
import math
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from sgfmill import sgf

TOP_N = 5
DEFAULT_MAX_MOVES = 30


def get_canonical_moves(
    sgf_str: str,
    target_color: str,
    max_moves: int = DEFAULT_MAX_MOVES
) -> List[Tuple[str, int, int, int]]:
    """從 SGF 字串解析出目標選手落子的正規化坐標。

    回傳：[(col, target_step_idx, canonical_r, canonical_c), ...]
    """
    target_color = str(target_color).strip().lower()
    if target_color not in ('b', 'w'):
        target_color = 'b'

    try:
        g = sgf.Sgf_game.from_string(sgf_str)
    except Exception:
        return []

    # 挑出目標選手的落子（直接過濾，避免因 Pass 或讓子造成雙層截斷誤差）
    target_moves = []
    for node in g.get_main_sequence()[1:]:
        col, mv = node.get_move()
        if mv is not None and col is not None and col.lower() == target_color:
            target_moves.append((len(target_moves), mv[0], mv[1]))
            if len(target_moves) >= max_moves:
                break

    if not target_moves:
        return []

    # 依目標選手的第一手計算 canonical symmetry k in 0..7
    r1, c1 = target_moves[0][1], target_moves[0][2]
    best_k = 0
    best_coord = (19, 19)

    for k in range(8):
        r, c = r1, c1
        for _ in range(k % 4):
            r, c = 18 - c, r
        if k >= 4:
            c = 18 - c
        if (r, c) < best_coord:
            best_coord = (r, c)
            best_k = k

    # 對所有目標落手套用 best_k
    canonical = []
    for step_idx, r, c in target_moves:
        for _ in range(best_k % 4):
            r, c = 18 - c, r
        if best_k >= 4:
            c = 18 - c
        canonical.append((target_color, step_idx, r, c))

    return canonical


def extract_profile_from_moves(
    canonical_moves_list: List[List[Tuple[str, int, int, int]]]
) -> Dict[tuple, float]:
    """將同一位選手多局的 canonical moves 轉為 L2 正規化的特徵向量。"""
    cnt = Counter()
    for mvs in canonical_moves_list:
        for col, step_idx, r, c in mvs:
            # 1. 帶執色與手數的精確特徵
            cnt[(col, step_idx, r, c)] += 2.0
            # 2. 帶執色的全域位置分佈
            cnt[(col, 'any_step', r, c)] += 1.0
            # 3. 不分執色的全域偏好
            cnt[('any_col', 'any_step', r, c)] += 0.5

        # 4. 相鄰連續兩手 Bigram
        for i in range(len(mvs) - 1):
            color, _, r1, c1 = mvs[i]
            _, _, r2, c2 = mvs[i + 1]
            cnt[('bi', color, r1, c1, r2, c2)] += 2.5

    norm = math.sqrt(sum(v * v for v in cnt.values())) + 1e-9
    return {k: v / norm for k, v in cnt.items()}


def build_candidate_profiles(
    cand_csv_path: str,
    max_moves: int = DEFAULT_MAX_MOVES
) -> Dict[str, Dict[tuple, float]]:
    """讀取候選庫 CSV，建立每位 candidate player 的指紋特徵。"""
    df = pd.read_csv(cand_csv_path)
    player_moves = {}

    for _, row in df.iterrows():
        pid = str(row['player_id']).strip()
        sgf_content = str(row['sgf_content'])
        color = str(row.get('color', 'b'))
        mvs = get_canonical_moves(sgf_content, color, max_moves=max_moves)
        if pid not in player_moves:
            player_moves[pid] = []
        player_moves[pid].append(mvs)

    profiles = {}
    for pid, moves_list in player_moves.items():
        profiles[pid] = extract_profile_from_moves(moves_list)

    return profiles


def predict_split(
    test_csv_path: str,
    cand_profiles: Dict[str, Dict[tuple, float]],
    top_n: int = TOP_N,
    max_moves: int = DEFAULT_MAX_MOVES
) -> List[List[str]]:
    """對單一測試題檔案進行預測，回傳 [[question_id, top_1, ..., top_5], ...]"""
    test_df = pd.read_csv(test_csv_path)
    cand_names = sorted(cand_profiles.keys())
    results = []

    sgf_cols = [c for c in test_df.columns if c.startswith('sgf_')]
    color_cols = [c for c in test_df.columns if c.startswith('color_')]

    for _, row in test_df.iterrows():
        qid = str(row['question_id']).strip()
        num_games = int(row.get('num_games', len(sgf_cols)))

        query_moves = []
        for i in range(1, num_games + 1):
            s_col = f'sgf_{i}'
            c_col = f'color_{i}'
            if s_col in row and pd.notna(row[s_col]) and str(row[s_col]).strip():
                sgf_content = str(row[s_col])
                color = str(row[c_col]) if c_col in row and pd.notna(row[c_col]) else 'b'
                mvs = get_canonical_moves(sgf_content, color, max_moves=max_moves)
                query_moves.append(mvs)

        q_profile = extract_profile_from_moves(query_moves)

        # 計算與每位候選者的餘弦相似度
        scores = []
        for c_pid in cand_names:
            c_prof = cand_profiles[c_pid]
            sim = sum(val * c_prof[k] for k, val in q_profile.items() if k in c_prof)
            scores.append((sim, c_pid))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [pid for _, pid in scores[:top_n]]

        # 防呆：確保填滿 top_n 且不重複
        if len(top_candidates) < top_n:
            for c_pid in cand_names:
                if c_pid not in top_candidates:
                    top_candidates.append(c_pid)
                if len(top_candidates) == top_n:
                    break

        results.append([qid] + top_candidates[:top_n])

    return results


def run_full_pipeline(
    tests_dir: str = 'tests',
    out_csv: str = 'submission_player.csv',
    max_moves: int = DEFAULT_MAX_MOVES
) -> pd.DataFrame:
    """執行全量四份測試題預測並輸出最終提交檔。"""
    splits_config = [
        ('pub_', 'player_identification_test_public.csv', 'player_identification_candidates_public.csv'),
        ('priv_', 'player_identification_test_private.csv', 'player_identification_candidates_private.csv'),
        ('pub2_', 'player_identification_test2_public.csv', 'player_identification_candidates2_public.csv'),
        ('priv2_', 'player_identification_test2_private.csv', 'player_identification_candidates2_private.csv'),
    ]

    all_rows = []
    t_start = time.time()

    for prefix, test_file, cand_file in splits_config:
        test_path = os.path.join(tests_dir, test_file)
        cand_path = os.path.join(tests_dir, cand_file)

        print(f'[{prefix}] 處理中...')
        t0 = time.time()
        cand_profiles = build_candidate_profiles(cand_path, max_moves=max_moves)
        print(f'  - 候選庫 {len(cand_profiles)} 人建置完成 ({time.time() - t0:.1f}s)')

        t0 = time.time()
        sub_rows = predict_split(test_path, cand_profiles, top_n=TOP_N, max_moves=max_moves)
        print(f'  - 測試題 {len(sub_rows)} 題推論完成 ({time.time() - t0:.1f}s)')

        # 檢查 prefix 與相異性
        for r in sub_rows:
            qid = r[0]
            top5 = r[1:]
            assert qid.startswith(prefix), f'qid {qid} 前綴不符 {prefix}'
            assert len(top5) == TOP_N, f'{qid} Top-{TOP_N} 數量不符'
            assert len(set(top5)) == TOP_N, f'{qid} 候選人重複: {top5}'
            expected_p_prefix = f"{prefix.rstrip('_')}_p_"
            for p in top5:
                assert p.startswith(expected_p_prefix), f'{p} 與題目 {qid} 前綴不一致'

        all_rows.extend(sub_rows)

    sub_df = pd.DataFrame(
        all_rows,
        columns=['question_id', 'top_1', 'top_2', 'top_3', 'top_4', 'top_5']
    )

    assert len(sub_df) == 1600, f'預期 1600 題，實際產生 {len(sub_df)} 題'
    tmp_csv = out_csv + '.tmp'
    sub_df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, out_csv)
    print(f'全部 1,600 題順利完成！耗時 {time.time() - t_start:.1f}s，已寫入 {out_csv}')
    return sub_df


def main():
    parser = argparse.ArgumentParser(description='開局指紋選手辨識 Baseline')
    parser.add_argument('--tests-dir', default='tests')
    parser.add_argument('--out-csv', default='submission_player.csv')
    parser.add_argument('--max-moves', type=int, default=DEFAULT_MAX_MOVES)
    args = parser.parse_args()

    run_full_pipeline(tests_dir=args.tests_dir, out_csv=args.out_csv, max_moves=args.max_moves)


if __name__ == '__main__':
    main()
