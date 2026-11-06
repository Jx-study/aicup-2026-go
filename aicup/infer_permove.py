"""逐手模型推論：每局目標方與對手方各手 log p_k(實際落子) 的平均，寫成每局一行的 jsonl

讀 extract_permove 的分片（val648 或測試集），用 train_permove 的 model.pt，只用原方向。
每行欄位：key、rank_idx、player_id、hd（沒有 HD 時為 -1）、color（目標玩家執色 B 或 W）、
pm（目標側 10 個等級的平均 log 機率，依 LEVELS 排列）、pmo（對手側，沒有對手側局面時為 null）、
n_pm、n_pmo（各自的手數）、model、dtype。行的順序和分片中局第一次出現的順序相同。

用法（在 repo 根目錄）：
    python -m aicup.infer_permove --model rank-trained-models/permove/<hash>/model.pt --data permove-features/val648/val --out humansl-analysis/val648-pm.jsonl
    python -m aicup.infer_permove --model ... --data ... --compare-dtypes 20   # 只比較 float32 與 bf16，不寫檔
"""
import argparse
import json
import os
import time

import numpy as np

from aicup import katago_torch as kt
from aicup.train_permove import (METADATA_PROFILE, N_LEVELS, Batcher, check_moves_legal, level_log_probs,
                                 load_permove, load_shards)

COLORS = ('B', 'W')


def position_log_probs(model, batcher, dtype, batch):
    """每個局面在 N_LEVELS 份 head 下實際落子的 log 機率，(N, N_LEVELS) float64。"""
    import torch
    meta = torch.from_numpy(kt.profile_row(METADATA_PROFILE)).to(batcher.device)
    out = np.zeros((batcher.n, N_LEVELS))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, batcher.n, batch):
            idx = torch.arange(s, min(s + batch, batcher.n), device=batcher.device)
            spatial, glob, legal, move, _ = batcher.get(idx)
            with torch.autocast(device_type=spatial.device.type, dtype=torch.bfloat16, enabled=dtype == 'bf16'):
                logits = model(spatial, glob, meta.expand(len(idx), -1))
            out[s:s + len(idx)] = level_log_probs(logits, legal, move).double().cpu().numpy()
            if (s // batch) % 200 == 0:
                print('  %d/%d 個局面，每秒 %.0f 個' % (s + len(idx), batcher.n, (s + len(idx)) / (time.time() - t0)),
                      flush=True)
    return out


def game_features(data, logp):
    """依 (key, side) 平均，回傳每局一個 dict，順序同局在分片中第一次出現的順序。"""
    keys, first = np.unique(data['key'], return_index=True)
    order = np.argsort(first)
    gi = np.searchsorted(keys, data['key'])
    sums = np.zeros((len(keys), 2, N_LEVELS))
    counts = np.zeros((len(keys), 2), int)
    side = data['side'].astype(int)
    np.add.at(sums, (gi, side), logp)
    np.add.at(counts, (gi, side), 1)
    rows = []
    for g in order:
        i = first[g]
        mean = [(sums[g, s] / counts[g, s]).tolist() if counts[g, s] else None for s in (0, 1)]
        rows.append({'key': str(keys[g]), 'rank_idx': int(data['rank_idx'][i]),
                     'player_id': str(data['player_id'][i]), 'hd': int(data['hd'][i]),
                     'color': COLORS[int(data['color'][i])],
                     'pm': mean[0], 'pmo': mean[1], 'n_pm': int(counts[g, 0]), 'n_pmo': int(counts[g, 1])})
    return rows


def first_games(data, n_games):
    """只留前 n_games 局的局面。"""
    _, first = np.unique(data['key'], return_index=True)
    keep = set(data['key'][np.sort(first)[:n_games]].tolist())
    mask = np.isin(data['key'], list(keep))
    return {k: v[mask] for k, v in data.items()}


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True, help='train_permove 的 model.pt')
    ap.add_argument('--data', required=True, help='extract_permove 的一個 split 目錄，例如 permove-features/val648/val')
    ap.add_argument('--out', help='輸出 jsonl')
    ap.add_argument('--dtype', choices=('float32', 'bf16'), default='float32')
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--device')
    ap.add_argument('--compare-dtypes', type=int, default=0,
                    help='非 0 時只對前幾局比較 float32 與 bf16 的每局特徵，印出差的最大值，不寫檔')
    ap.add_argument('--katago-python', help='KataGo 原始碼的 python/ 目錄，缺省時讀環境變數 KATAGO_PYTHON')
    args = ap.parse_args()
    if args.katago_python:
        os.environ['KATAGO_PYTHON'] = args.katago_python
    if not args.out and not args.compare_dtypes:
        ap.error('要給 --out 或 --compare-dtypes')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = load_permove(args.model, device)
    data = load_shards(args.data)
    check_moves_legal(data)
    if args.compare_dtypes:
        data = first_games(data, args.compare_dtypes)
        batcher = Batcher(data, device)
        feats = {}
        for dtype in ('float32', 'bf16'):
            rows = game_features(data, position_log_probs(model, batcher, dtype, args.batch))
            feats[dtype] = np.array([r['pm'] + (r['pmo'] or [np.nan] * N_LEVELS) for r in rows])
        diff = np.nanmax(np.abs(feats['float32'] - feats['bf16']))
        print('%d 局，float32 與 bf16 的每局特徵差最大 %.4g' % (len(rows), diff))
        return
    batcher = Batcher(data, device)
    print('%d 局、%d 個局面，dtype %s，device %s' % (len(np.unique(data['key'])), batcher.n, args.dtype, device),
          flush=True)
    logp = position_log_probs(model, batcher, args.dtype, args.batch)
    if not np.isfinite(logp).all():
        raise SystemExit('%d 個局面的 log 機率不是有限值，不寫檔' % (~np.isfinite(logp).all(axis=1)).sum())
    rows = game_features(data, logp)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(dict(r, model=args.model, dtype=args.dtype)) + '\n')
    os.replace(tmp, args.out)
    print('寫出 %s，%d 局' % (args.out, len(rows)))


if __name__ == '__main__':
    main()
