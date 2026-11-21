"""基於開局指紋與應手風格的圍棋選手識別模組。

提取開局落子與局部應手，經幾何對稱變換正規化後計算特徵分佈與 IDF 加權，比對候選庫餘弦相似度輸出預測名單。
"""

import argparse
import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sgfmill import sgf

TOP_N = 5
DEFAULT_MAX_MOVES = 10
IDF_POWER = 0.75
W_RESP = 2.0

MoveInfo = Tuple[str, int, int, int, Optional[Tuple[int, int]]]
Moves = List[MoveInfo]
ColoredGames = List[Tuple[str, Moves]]


def get_canonical_moves(
    sgf_str: str,
    target_color: str,
    max_moves: int = DEFAULT_MAX_MOVES
) -> List[Tuple[str, int, int, int, Optional[Tuple[int, int]]]]:
    """從 SGF 字串解析出目標選手落子的正規化坐標與對手應手資訊。

    回傳每手落子元組清單，欄位依序為執色、手數、列坐標、行坐標與前手對手落子坐標。
    """
    target_color = str(target_color).strip().lower()
    if target_color not in ('b', 'w'):
        target_color = 'b'

    try:
        g = sgf.Sgf_game.from_string(sgf_str)
    except Exception:
        return []

    raw_moves = []
    for node in g.get_main_sequence()[1:]:
        col, mv = node.get_move()
        if mv is not None and col is not None:
            raw_moves.append((col.lower(), mv[0], mv[1]))
            if len(raw_moves) >= max_moves * 3:
                break

    # 找到目標選手第一手以計算對稱變換
    first_target = None
    for col, r, c in raw_moves:
        if col == target_color:
            first_target = (r, c)
            break

    if first_target is None:
        return []

    r1, c1 = first_target
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

    # 對所有落手套用對稱變換
    transformed_moves = []
    for col, r, c in raw_moves:
        for _ in range(best_k % 4):
            r, c = 18 - c, r
        if best_k >= 4:
            c = 18 - c
        transformed_moves.append((col, r, c))

    # 抽取目標手數與接續對手手數
    target_moves_info = []
    step_idx = 0
    for i in range(len(transformed_moves)):
        col, r, c = transformed_moves[i]
        if col == target_color:
            opp_move = None
            if i > 0 and transformed_moves[i - 1][0] != target_color:
                opp_move = (transformed_moves[i - 1][1], transformed_moves[i - 1][2])
            target_moves_info.append((target_color, step_idx, r, c, opp_move))
            step_idx += 1
            if step_idx >= max_moves:
                break

    return target_moves_info


def extract_raw_counts(
    canonical_moves_list: List[List[Tuple[str, int, int, int, Optional[Tuple[int, int]]]]],
    w_resp: float = W_RESP
) -> Counter:
    """從多局開局落子提取未經 IDF 加權的特徵頻率計數。"""
    cnt = Counter()
    for mvs in canonical_moves_list:
        for col, step_idx, r, c, opp in mvs:
            # 帶執色與手數的精確落點
            cnt[(col, step_idx, r, c)] += 1.5
            # 全盤落點分佈
            cnt[('any_col', r, c)] += 0.5

            # 局部應手與脫先行為
            if opp is not None and w_resp > 0:
                dr = r - opp[0]
                dc = c - opp[1]
                dist = abs(dr) + abs(dc)
                if dist <= 3:
                    cnt[(col, 'local_resp', step_idx, dr, dc)] += w_resp
                else:
                    cnt[(col, 'tenuki', step_idx)] += w_resp

        # 相鄰手 Bigram 連續序列
        for i in range(len(mvs) - 1):
            col1, _, r1, c1, _ = mvs[i]
            col2, _, r2, c2, _ = mvs[i + 1]
            cnt[('bi', col1, r1, c1, r2, c2)] += 2.0

    return cnt


def normalize_counts(
    cnt: Counter,
    idf_weights: Dict[tuple, float]
) -> Dict[tuple, float]:
    """套用 IDF 權重並進行 L2 正規化，回傳特徵向量字典。"""
    weighted = {k: val * idf_weights.get(k, 1.0) for k, val in cnt.items()}
    norm = math.sqrt(sum(v * v for v in weighted.values())) + 1e-9
    return {k: v / norm for k, v in weighted.items()}


def build_query_vectors(
    query_games: ColoredGames,
    idf_weights: Dict[tuple, float],
    w_resp: float = W_RESP,
    color_cond: bool = False
) -> dict:
    """從題目多局落子建構查詢向量，支援分色與全局模式。

    回傳 dict 含下列欄位：
        all: 全部盤面合併之正規化特徵向量
        b: 執黑盤面正規化特徵向量（無黑棋或未啟用分色時為 None）
        w: 執白盤面正規化特徵向量（無白棋或未啟用分色時為 None）
        n_b: 執黑盤數
        n_w: 執白盤數
    """
    all_moves = [mvs for _, mvs in query_games]
    q_all = normalize_counts(extract_raw_counts(all_moves, w_resp=w_resp), idf_weights)
    if not color_cond:
        return {'all': q_all, 'b': None, 'w': None, 'n_b': 0, 'n_w': 0}

    b_moves = [mvs for col, mvs in query_games if col == 'b']
    w_moves = [mvs for col, mvs in query_games if col == 'w']
    q_b = normalize_counts(extract_raw_counts(b_moves, w_resp=w_resp), idf_weights) if b_moves else None
    q_w = normalize_counts(extract_raw_counts(w_moves, w_resp=w_resp), idf_weights) if w_moves else None
    return {'all': q_all, 'b': q_b, 'w': q_w, 'n_b': len(b_moves), 'n_w': len(w_moves)}


def compute_style_similarity(
    q_vecs: dict,
    cand_profile: dict,
    color_cond: bool = False
) -> float:
    """計算題目查詢向量與候選人風格檔案之餘弦相似度。

    啟用 color_cond 時，依執黑與執白之有效盤數比例計算加權平均餘弦相似度。
    """
    if not color_cond:
        q_all = q_vecs['all']
        return sum(val * cand_profile[k] for k, val in q_all.items() if k in cand_profile)

    weighted_sim = 0.0
    active_weight = 0.0
    for col, n_key in (('b', 'n_b'), ('w', 'n_w')):
        n_c = q_vecs.get(n_key, 0)
        q_c = q_vecs.get(col)
        if n_c > 0 and q_c:
            c_c = cand_profile[col]
            weighted_sim += n_c * sum(val * c_c[k] for k, val in q_c.items() if k in c_c)
            active_weight += n_c

    if active_weight == 0.0:
        q_all = q_vecs['all']
        c_all = cand_profile['all']
        return sum(val * c_all[k] for k, val in q_all.items() if k in c_all)

    return weighted_sim / active_weight


def build_candidate_profiles_from_moves(
    player_games: Dict[str, ColoredGames],
    idf_power: float = IDF_POWER,
    w_resp: float = W_RESP,
    color_cond: bool = False
) -> Tuple[Dict[str, dict], Dict[tuple, float]]:
    """由已解析的候選人對局清單建立風格檔案與全域 IDF 權重字典。"""
    cand_raw_counts = {}
    df_counts = Counter()
    n_cands = len(player_games)

    for pid, games_list in player_games.items():
        all_moves = [mvs for _, mvs in games_list]
        cnt = extract_raw_counts(all_moves, w_resp=w_resp)
        cand_raw_counts[pid] = cnt
        for k in cnt.keys():
            df_counts[k] += 1

    # 計算全域特徵 IDF
    idf_weights = {}
    for k, df_val in df_counts.items():
        if idf_power > 0:
            idf_weights[k] = (math.log((n_cands + 1) / (df_val + 1)) + 1.0) ** idf_power
        else:
            idf_weights[k] = 1.0

    # 加權並正規化候選人向量
    profiles = {}
    for pid, cnt in cand_raw_counts.items():
        vec_all = normalize_counts(cnt, idf_weights)
        if not color_cond:
            profiles[pid] = vec_all
        else:
            b_moves = [mvs for col, mvs in player_games[pid] if col == 'b']
            w_moves = [mvs for col, mvs in player_games[pid] if col == 'w']
            vec_b = normalize_counts(extract_raw_counts(b_moves, w_resp=w_resp), idf_weights) if b_moves else vec_all
            vec_w = normalize_counts(extract_raw_counts(w_moves, w_resp=w_resp), idf_weights) if w_moves else vec_all
            profiles[pid] = {'b': vec_b, 'w': vec_w, 'all': vec_all}

    return profiles, idf_weights


def build_candidate_profiles_and_idf(
    cand_csv_path: str,
    max_moves: int = DEFAULT_MAX_MOVES,
    idf_power: float = IDF_POWER,
    w_resp: float = W_RESP,
    color_cond: bool = False
) -> Tuple[Dict[str, dict], Dict[tuple, float]]:
    """讀取候選庫 CSV，建立候選人指紋特徵與全域 IDF 權重字典。"""
    df = pd.read_csv(cand_csv_path)
    player_games = {}

    for _, row in df.iterrows():
        pid = str(row['player_id']).strip()
        sgf_content = str(row['sgf_content'])
        color = str(row.get('color', 'b')).strip().lower()
        if color not in ('b', 'w'):
            color = 'b'
        mvs = get_canonical_moves(sgf_content, color, max_moves=max_moves)
        if pid not in player_games:
            player_games[pid] = []
        player_games[pid].append((color, mvs))

    return build_candidate_profiles_from_moves(
        player_games, idf_power=idf_power, w_resp=w_resp, color_cond=color_cond
    )


def predict_split(
    test_csv_path: str,
    cand_profiles: Dict[str, dict],
    idf_weights: Dict[tuple, float],
    top_n: int = TOP_N,
    max_moves: int = DEFAULT_MAX_MOVES,
    w_resp: float = W_RESP,
    color_cond: bool = False
) -> List[List[str]]:
    """對單一測試題檔案進行預測，回傳各題目預測候選人清單。"""
    test_df = pd.read_csv(test_csv_path)
    cand_names = sorted(cand_profiles.keys())
    results = []

    sgf_cols = [c for c in test_df.columns if c.startswith('sgf_')]

    for _, row in test_df.iterrows():
        qid = str(row['question_id']).strip()
        num_games = int(row.get('num_games', len(sgf_cols)))

        query_games = []
        for i in range(1, num_games + 1):
            s_col = f'sgf_{i}'
            c_col = f'color_{i}'
            if s_col in row and pd.notna(row[s_col]) and str(row[s_col]).strip():
                sgf_content = str(row[s_col])
                color = str(row[c_col]).strip().lower() if c_col in row and pd.notna(row[c_col]) else 'b'
                if color not in ('b', 'w'):
                    color = 'b'
                mvs = get_canonical_moves(sgf_content, color, max_moves=max_moves)
                query_games.append((color, mvs))

        q_vecs = build_query_vectors(query_games, idf_weights, w_resp=w_resp, color_cond=color_cond)

        # 計算餘弦相似度
        scores = []
        for c_pid in cand_names:
            sim = compute_style_similarity(q_vecs, cand_profiles[c_pid], color_cond=color_cond)
            scores.append((sim, c_pid))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [pid for _, pid in scores[:top_n]]

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
    max_moves: int = DEFAULT_MAX_MOVES,
    idf_power: float = IDF_POWER,
    w_resp: float = W_RESP
) -> pd.DataFrame:
    """執行全量四份測試題推論並寫出提交檔。"""
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
        cand_profiles, idf_weights = build_candidate_profiles_and_idf(
            cand_path, max_moves=max_moves, idf_power=idf_power, w_resp=w_resp
        )
        print(f'  - 候選庫 {len(cand_profiles)} 人與 IDF 建置完成 ({time.time() - t0:.1f}s)')

        t0 = time.time()
        sub_rows = predict_split(
            test_path, cand_profiles, idf_weights, top_n=TOP_N, max_moves=max_moves, w_resp=w_resp
        )
        print(f'  - 測試題 {len(sub_rows)} 題推論完成 ({time.time() - t0:.1f}s)')

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
    parser = argparse.ArgumentParser(description='開局指紋選手辨識模組')
    parser.add_argument('--tests-dir', default='tests')
    parser.add_argument('--out-csv', default='submission_player.csv')
    parser.add_argument('--max-moves', type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument('--idf-power', type=float, default=IDF_POWER)
    parser.add_argument('--w-resp', type=float, default=W_RESP)
    args = parser.parse_args()

    run_full_pipeline(
        tests_dir=args.tests_dir,
        out_csv=args.out_csv,
        max_moves=args.max_moves,
        idf_power=args.idf_power,
        w_resp=args.w_resp
    )


if __name__ == '__main__':
    main()
