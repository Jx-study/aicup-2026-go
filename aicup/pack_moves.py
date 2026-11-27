"""把 Human SL 逐手分析的 jsonl 打包成緊湊的 .npz，供 tf_run.py 在 GPU 上訓練。

輸入是目標側與對手側各一份 jsonl，以 key 對應，路徑是 colab_humansl 的輸出：
6d 讀 humansl-analysis/hist6d-<run>/，2d 讀 humansl-analysis/hist2d-<run>/；train_p4all 依序串接 train_p4 與 train_p4b。
輸出每局 (200, 14) 的逐手特徵與長度 200 的 mask，加上 loss 特徵時為 (200, 20)。
用法：python -m aicup.pack_moves test                  # 2d，寫到 packed-moves-2d
      python -m aicup.pack_moves --loss train_p1       # 2d + loss，寫到 packed-moves-2dl
      python -m aicup.pack_moves --6d test             # 6d，寫到 packed-moves
      python -m aicup.pack_moves test --out-dir DIR    # 改寫到 DIR
"""
import os
import sys
import json
import math
import numpy as np

LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']

def parse_move_features(target_row, opp_row, max_turns=200):
    """每手 14 維，最多取 max_turns 手。

    目標側 6 維：log p(11k)、log p(1d)、log p(第三個 profile)、log pmax、entropy、t/T。
    對手側 5 維：log p(11k)、log p(1d)、log p(第三個 profile)、log pmax、entropy。
    其他 3 維：color == B、hd == 1、1 / log(T+2)。
    回傳 (feats, mask)，形狀 (max_turns, 14) 與 (max_turns,)。
    """
    eps = 1e-6
    t_turns = target_row.get('turns', [])
    T = len(t_turns)
    if T == 0:
        return np.zeros((max_turns, 14), dtype=np.float32), np.zeros(max_turns, dtype=bool)
        
    t_probs = target_row['probs'] # 3 個 profile 的機率串列
    t_pmax = target_row['pmax'] # 3 個 profile
    t_entropy = target_row['entropy'] # 3 個 profile
    
    o_probs = opp_row['probs'] if opp_row else [[0.1]*T]*3
    o_pmax = opp_row['pmax'] if opp_row else [[0.1]*T]*3
    o_entropy = opp_row['entropy'] if opp_row else [[1.5]*T]*3
    
    color_b = 1.0 if target_row.get('color') == 'B' else 0.0
    hd1 = 1.0 if target_row.get('hd') == 1 else 0.0
    inv_log_t = 1.0 / math.log(T + 2)
    
    actual_t = min(T, max_turns)
    feats = np.zeros((max_turns, 14), dtype=np.float32)
    mask = np.zeros(max_turns, dtype=bool)
    
    for t in range(actual_t):
        # 目標側 6 維
        lp_11k = math.log(max(t_probs[0][t], eps))
        lp_1d = math.log(max(t_probs[1][t], eps))
        lp_6d = math.log(max(t_probs[2][t], eps))  # 第三個 profile（2d 或 6d）
        lp_max = math.log(max(t_pmax[1][t], eps)) # 取 1d 的 pmax
        ent = t_entropy[1][t]
        ratio = float(t) / float(T)
        
        # 對手側 5 維
        olp_11k = math.log(max(o_probs[0][min(t, len(o_probs[0])-1)], eps)) if o_probs[0] else -2.3
        olp_1d = math.log(max(o_probs[1][min(t, len(o_probs[1])-1)], eps)) if o_probs[1] else -2.3
        olp_6d = math.log(max(o_probs[2][min(t, len(o_probs[2])-1)], eps)) if o_probs[2] else -2.3
        olp_max = math.log(max(o_pmax[1][min(t, len(o_pmax[1])-1)], eps)) if o_pmax[1] else -1.0
        o_ent = o_entropy[1][min(t, len(o_entropy[1])-1)] if o_entropy[1] else 1.5
        
        feats[t] = [
            lp_11k, lp_1d, lp_6d, lp_max, ent, ratio,
            olp_11k, olp_1d, olp_6d, olp_max, o_ent,
            color_b, hd1, inv_log_t
        ]
        mask[t] = True
        
    return feats, mask

def loss_features(target_row, opp_row, max_turns=200):
    """每手 6 維：兩側各為 asinh(score_loss)、winrate_loss（截到 ±1）、log(1 + rank_1d 的 move_rank)，缺值補 0。"""
    out = np.zeros((max_turns, 6), dtype=np.float32)
    T = min(len(target_row.get('turns', [])), max_turns)
    for side, row in ((0, target_row), (3, opp_row)):
        if not row:
            continue
        sl, wl, mr = row['score_loss'], row['winrate_loss'], row['move_rank'][1]
        n = len(sl)
        if n == 0:
            continue
        for t in range(T):
            i = min(t, n - 1)  # 對手側手數可能少一手，和 parse_move_features 的對齊方式相同
            out[t, side] = math.asinh(sl[i]) if sl[i] is not None else 0.0
            out[t, side + 1] = max(-1.0, min(1.0, wl[i])) if wl[i] is not None else 0.0
            out[t, side + 2] = math.log1p(mr[i]) if mr[i] is not None else 0.0
    return out


def read_lines(paths):
    # 多個 jsonl 依序串接，用於同一批玩家分批抽取的局（例如 train_p4 加補局 train_p4b）
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield line


def pack_dataset(t_path, o_path, out_npz, max_turns=200, with_loss=False):
    """t_path、o_path 可以是單一路徑或路徑 tuple；tuple 時依序串接成一個訓練檔。"""
    t_paths = t_path if isinstance(t_path, tuple) else (t_path,)
    o_paths = o_path if isinstance(o_path, tuple) else (o_path,)
    print(f"正在打包: {t_paths} + {o_paths} -> {out_npz}")
    
    # 對手側以 key 建索引
    opp_dict = {}
    for line in read_lines([p for p in o_paths if os.path.exists(p)]):
        r = json.loads(line)
        opp_dict[r['key']] = r
                    
    keys = []
    pids = []
    ranks = []
    colors = []
    hds = []
    feats_list = []
    masks_list = []
    
    for idx, line in enumerate(read_lines(t_paths)):
        r = json.loads(line)
        k = r['key']
        opp = opp_dict.get(k)
        
        f_mat, m_vec = parse_move_features(r, opp, max_turns=max_turns)
        if with_loss:
            for row in (r, opp):
                if row and 'score_loss' not in row:  # --no-eval 的 jsonl 沒有 loss 欄
                    raise ValueError('--loss 需要 score_loss 欄，%s 的 %s 沒有' % (t_paths, k))
            f_mat = np.concatenate([f_mat, loss_features(r, opp, max_turns)], axis=1)  # 原 14 維之後接 6 維 loss 欄，共 20 維
        
        keys.append(k)
        rk = r.get('rank_idx')
        rk = -1 if rk is None else int(rk)
        pids.append('%d|%s' % (rk, r.get('player_id', '')))  # 玩家由等級與 player_id 一起區分
        ranks.append(rk)
        colors.append(1 if r.get('color') == 'B' else 0)
        hd_val = r.get('hd')
        hds.append(0 if hd_val is None else int(hd_val))
        feats_list.append(f_mat)
        masks_list.append(m_vec)
        
        if (idx + 1) % 2000 == 0:
            print(f"  已處理 {idx + 1} 局...")
            
    N = len(keys)
    feats_arr = np.array(feats_list, dtype=np.float32) # (N, max_turns, 特徵維數)
    masks_arr = np.array(masks_list, dtype=bool) # (N, max_turns)
    
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        keys=np.array(keys),
        player_ids=np.array(pids),
        ranks=np.array(ranks, dtype=np.int64),
        colors=np.array(colors, dtype=np.int32),
        hds=np.array(hds, dtype=np.int32),
        moves=feats_arr,
        masks=masks_arr
    )
    print(f"完成! 打包局數: {N}, 檔案大小: {os.path.getsize(out_npz) / 1e6:.1f} MB")

RUNS_6D = ['val648', 'trainpool', 'newdev', 'test']
RUNS_2D = ['val648', 'train_p1', 'train_p2', 'test', 'train_p3', 'train_p4', 'train_p4all']
PARTS = {'train_p4all': ('train_p4', 'train_p4b')}  # 同一批玩家分兩次抽的局，依序串接


def jsonl_pair(h, prefix, run):
    """colab_humansl 寫出的 (目標側, 對手側) jsonl 路徑：<h><prefix><run>/<run>-p3-hist.jsonl 與 -p3-opp-hist.jsonl。"""
    d = '%s%s%s/%s' % (h, prefix, run, run)
    return d + '-p3-hist.jsonl', d + '-p3-opp-hist.jsonl'


def build_jobs(h, out, six_d=False):
    """回傳 (目標側 jsonl, 對手側 jsonl, 輸出 npz) 的清單；jsonl 可以是路徑 tuple。"""
    prefix = 'hist6d-' if six_d else 'hist2d-'
    jobs = []
    for name in (RUNS_6D if six_d else RUNS_2D):
        pairs = [jsonl_pair(h, prefix, r) for r in PARTS.get(name, (name,))]
        t, o = tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)
        jobs.append((t if len(t) > 1 else t[0], o if len(o) > 1 else o[0], out + name + '_moves.npz'))
    return jobs


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='打包逐手特徵')
    ap.add_argument('names', nargs='*', help='要打包的輸出名稱（例如 train_p3）；不給則打包 val648、train_p1、train_p2、test')
    ap.add_argument('--loss', action='store_true', help='每手多 6 維 loss 欄，預設輸出到 packed-moves-2dl；train_p4 系列是 --no-eval 的 jsonl，不能加')
    ap.add_argument('--6d', dest='six_d', action='store_true', help='打包 6d 的 4 個檔，預設輸出到 packed-moves')
    ap.add_argument('--out-dir', help='輸出資料夾，預設依 --loss、--6d 決定')
    args = ap.parse_args()
    H = 'humansl-analysis/'
    OUT = args.out_dir or H + ('packed-moves-2dl' if args.loss else 'packed-moves' if args.six_d else 'packed-moves-2d')
    OUT = OUT.rstrip('/') + '/'
    jobs = build_jobs(H, OUT, args.six_d)
    sel = args.names or (['val648', 'trainpool', 'newdev', 'test'] if args.six_d else ['val648', 'train_p1', 'train_p2', 'test'])
    jobs = [j for j in jobs if os.path.basename(j[2])[:-len('_moves.npz')] in sel]
    assert len(jobs) == len(sel), sel
    for t, o, out in jobs:
        paths = (t if isinstance(t, tuple) else (t,)) + (o if isinstance(o, tuple) else (o,))
        assert all(os.path.exists(p) for p in paths), paths
        pack_dataset(t, o, out, with_loss=args.loss)
