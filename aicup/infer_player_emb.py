"""深度棋風 Embedding 批次推論與快取工具。

支援兩種推論來源：
    packed  從 bit-packed val 分片（搭配 hd.npz 與可選玩家篩選 CSV）抽取 Embedding 快取
    test    從競賽一候選庫與測試題 CSV 解析 SGF 並以多視角 TTA 抽取 Embedding 快取

另提供候選池中心化與多局聚合餘弦相似度計算函式，供離線評估與融合使用。

用法（在 repo 根目錄）：
    python -m aicup.infer_player_emb --mode packed --ckpt player-trained-models/emb/best_model.pth \\
        --feat-dir rank-features-v180000 --players humansl-analysis/val-players-newdev-v180000.csv \\
        --tta 8 --out player-trained-models/emb-cache/newdev_emb.npz
    python -m aicup.infer_player_emb --mode test --ckpt player-trained-models/emb/best_model.pth \\
        --groups b --tta 8 --out-dir player-trained-models/emb-cache
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from aicup import color, packed
from aicup.extract_features import COLOR_TO_ID
from aicup.predict_rank import file_sha256, sgf_to_features
from aicup.train_player_emb import (
    GoStyleEmbedNet,
    extract_split_embeddings,
    forward_embeddings,
    read_player_ids_csv,
)
from aicup.train_rank import use_amp

SPLIT_GROUPS = {
    'a': [
        ('pub', 'player_identification_candidates_public.csv', 'player_identification_test_public.csv'),
        ('priv', 'player_identification_candidates_private.csv', 'player_identification_test_private.csv'),
    ],
    'b': [
        ('pub2', 'player_identification_candidates2_public.csv', 'player_identification_test2_public.csv'),
        ('priv2', 'player_identification_candidates2_private.csv', 'player_identification_test2_private.csv'),
    ],
}


def load_emb_model(ckpt_path, device):
    """依 checkpoint 同目錄之 train_config.json 建立並載入 GoStyleEmbedNet。"""
    cfg_path = os.path.join(os.path.dirname(ckpt_path), 'train_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = {'channels': 256, 'blocks': 20, 'embed_dim': 256, 'dropout': 0.2, 'color': 'swap', 'raw_color': False}

    model = GoStyleEmbedNet(
        in_channels=color.in_channels(cfg['color']),
        num_channels=int(cfg['channels']),
        num_blocks=int(cfg['blocks']),
        embed_dim=int(cfg['embed_dim']),
        dropout=float(cfg.get('dropout', 0.2)),
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def compute_sgf_feature_colors(sgfs, raw_colors, raw_color_flag=False):
    """由原始執色（'B'/'W' 或 'b'/'w'）與 SGF 內 HD 標記推導特徵平面顏色（0 黑 1 白）。"""
    c_arr = np.array([COLOR_TO_ID[str(c).strip().upper()] for c in raw_colors], dtype=np.uint8)
    if raw_color_flag:
        return c_arr
    hds = [color.parse_hd(s) or 0 for s in sgfs]
    return color.feature_color(c_arr, hds).astype(np.uint8)


def infer_sgf_embeddings(
    model,
    sgfs,
    raw_colors,
    device,
    amp,
    color_mode='swap',
    raw_color_flag=False,
    batch_size=64,
    tta=8,
    nproc=1,
):
    """批次解析 SGF 並推論 L2 正規化風格向量，解析失敗的局填零向量。"""
    n = len(sgfs)
    embs = np.zeros((n, model.embed_dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return embs, valid

    feat_colors = compute_sgf_feature_colors(sgfs, raw_colors, raw_color_flag) if color_mode != 'none' else None
    pool = Pool(nproc) if nproc > 1 else None
    feats = pool.imap(sgf_to_features, sgfs, chunksize=8) if pool else map(sgf_to_features, sgfs)
    buf, idxs = [], []
    t0 = time.time()

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(device).float()
        if feat_colors is not None:
            c = torch.from_numpy(feat_colors[idxs].astype(np.int64)).to(device)
            x = color.apply_color(x, c, color_mode)
        with torch.no_grad():
            e = forward_embeddings(model, x, amp, tta=tta).cpu().numpy()
        embs[idxs] = e
        valid[idxs] = True
        buf.clear()
        idxs.clear()

    model.eval()
    try:
        for i, feat in enumerate(feats):
            if feat is not None:
                buf.append(feat)
                idxs.append(i)
            if len(buf) >= batch_size:
                flush()
            if (i + 1) % 1000 == 0 or (i + 1) == n:
                elapsed = max(time.time() - t0, 0.1)
                print('  推論進度 %d/%d 局 (%.1f 局/秒)' % (i + 1, n, (i + 1) / elapsed), flush=True)
        if buf:
            flush()
    finally:
        if pool:
            pool.close()
            pool.join()
    return embs, valid


def save_npz_atomic(path, **arrays):
    """先寫暫存檔再改名，確保快取不留半成品。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def aggregate_and_center_similarity(cand_embs_by_id, query_embs_by_id, cand_ids, query_ids):
    """計算多局平均、候選池中心化並 L2 正規化後之餘弦相似度矩陣 (Q, C)。

    cand_embs_by_id  dict[cand_id -> (n_i, D) array]
    query_embs_by_id dict[query_id -> (m_j, D) array]
    沒有有效向量的候選欄位回傳 -inf，避免排在負相似度候選之前。
    """
    if not cand_ids or not query_ids:
        return np.zeros((len(query_ids), len(cand_ids)), dtype=np.float64)

    dim = None
    for cid in cand_ids:
        arr = cand_embs_by_id.get(cid)
        if arr is not None and len(arr) > 0:
            dim = np.asarray(arr).shape[-1]
            break
    if dim is None:
        return np.full((len(query_ids), len(cand_ids)), float('-inf'), dtype=np.float64)

    cand_mat = np.zeros((len(cand_ids), dim), dtype=np.float64)
    cand_ok = np.zeros(len(cand_ids), dtype=bool)
    for i, cid in enumerate(cand_ids):
        arr = cand_embs_by_id.get(cid)
        if arr is not None and len(arr) > 0:
            cand_mat[i] = np.asarray(arr, dtype=np.float64).mean(axis=0)
            cand_ok[i] = True

    mu_cand = cand_mat[cand_ok].mean(axis=0, keepdims=True)
    cand_centered = cand_mat - mu_cand
    cand_norms = np.linalg.norm(cand_centered, axis=1, keepdims=True)
    cand_unit = np.where(cand_ok[:, None], cand_centered / np.maximum(cand_norms, 1e-12), 0.0)

    q_mat = np.zeros((len(query_ids), dim), dtype=np.float64)
    q_ok = np.zeros(len(query_ids), dtype=bool)
    for j, qid in enumerate(query_ids):
        arr = query_embs_by_id.get(qid)
        if arr is not None and len(arr) > 0:
            q_mat[j] = np.asarray(arr, dtype=np.float64).mean(axis=0)
            q_ok[j] = True

    q_centered = q_mat - mu_cand
    q_norms = np.linalg.norm(q_centered, axis=1, keepdims=True)
    q_unit = np.where(q_ok[:, None], q_centered / np.maximum(q_norms, 1e-12), 0.0)
    sim = q_unit @ cand_unit.T
    if not cand_ok.all():
        sim[:, ~cand_ok] = float('-inf')
    return sim


def run_packed_mode(args, model, cfg, device, amp, ckpt_hash):
    """從 bit-packed val 分片抽取 Embedding 並存成單一 .npz 快取。"""
    split_dir = os.path.join(args.feat_dir, args.split)
    d = packed.load_split(split_dir)
    if args.players:
        keep_pids = read_player_ids_csv(args.players)
        indices = np.array([i for i, pid in enumerate(d['player_id']) if str(pid) in keep_pids], dtype=np.int64)
    else:
        indices = np.arange(len(d['key']), dtype=np.int64)

    if len(indices) == 0:
        raise SystemExit('篩選後沒有任何對局可推論')

    x = torch.from_numpy(d['features'])
    c = None
    if cfg['color'] != 'none':
        col = d['color']
        if not cfg.get('raw_color', False):
            col = color.feature_color(col, color.load_hd(os.path.join(args.feat_dir, 'hd.npz'), d['key']))
        c = torch.from_numpy(col.astype(np.int64))

    print('packed 推論：%s，%d 局，tta=%d' % (split_dir, len(indices), args.tta), flush=True)
    t0 = time.time()
    embs = extract_split_embeddings(
        model, x, indices, args.batch_size, device, amp,
        c=c, color_mode=cfg['color'], tta=args.tta,
    )
    save_npz_atomic(
        args.out,
        emb=embs.astype(np.float16),
        key=d['key'][indices],
        player_id=d['player_id'][indices],
        rank_idx=d['rank_idx'][indices],
        color=d['color'][indices],
        tta=np.int64(args.tta),
        ckpt_sha256=np.array(ckpt_hash),
    )
    print('寫出 %s（%d 局，%.1f 秒）' % (args.out, len(indices), time.time() - t0), flush=True)


def _parse_color_value(val, context):
    """驗證顏色欄位為 b 或 w，缺值或非法值直接報錯。"""
    if pd.isna(val):
        raise ValueError('%s 缺少顏色欄位' % context)
    col = str(val).strip().lower()
    if col not in ('b', 'w'):
        raise ValueError('%s 顏色值無效：%r' % (context, val))
    return col


def _read_cand_csv(path):
    df = pd.read_csv(path)
    entity_ids = df['player_id'].astype(str).tolist()
    sgfs = df['sgf_content'].fillna('').astype(str).tolist()
    colors = [_parse_color_value(c, '%s 第 %d 列' % (path, i)) for i, c in enumerate(df['color'])]
    return entity_ids, sgfs, colors


def _read_test_csv(path):
    df = pd.read_csv(path)
    sgf_cols = [c for c in df.columns if c.startswith('sgf_')]
    entity_ids, sgfs, colors = [], [], []
    for _, row in df.iterrows():
        qid = str(row['question_id']).strip()
        num_games = int(row.get('num_games', len(sgf_cols)))
        for i in range(1, num_games + 1):
            sc = 'sgf_%d' % i
            cc = 'color_%d' % i
            if sc in row and pd.notna(row[sc]) and str(row[sc]).strip():
                if cc not in row:
                    raise ValueError('%s 的 %s 缺少欄位 %s' % (path, qid, cc))
                col = _parse_color_value(row[cc], '%s %s %s' % (path, qid, cc))
                entity_ids.append(qid)
                sgfs.append(str(row[sc]))
                colors.append(col)
    return entity_ids, sgfs, colors


def run_test_mode(args, model, cfg, device, amp, ckpt_hash):
    """從競賽一候選庫與測試題 CSV 抽取 Embedding 並依分組存成 .npz 快取。"""
    groups = ['a', 'b'] if args.groups == 'all' else [args.groups]
    os.makedirs(args.out_dir, exist_ok=True)
    for g in groups:
        for tag, cand_file, test_file in SPLIT_GROUPS[g]:
            for role, fname, reader in (
                ('cand', cand_file, _read_cand_csv),
                ('query', test_file, _read_test_csv),
            ):
                out_path = os.path.join(args.out_dir, '%s_%s_emb.npz' % (tag, role))
                if os.path.exists(out_path) and not args.overwrite:
                    print('已存在 %s，略過（重跑請加 --overwrite）' % out_path, flush=True)
                    continue
                csv_path = os.path.join(args.tests_dir, fname)
                if not os.path.exists(csv_path):
                    print('警告：找不到 %s，略過' % csv_path, flush=True)
                    continue
                entity_ids, sgfs, raw_colors = reader(csv_path)
                print('[%s_%s] %d 局，tta=%d' % (tag, role, len(sgfs), args.tta), flush=True)
                embs, valid = infer_sgf_embeddings(
                    model, sgfs, raw_colors, device, amp,
                    color_mode=cfg['color'],
                    raw_color_flag=cfg.get('raw_color', False),
                    batch_size=args.batch_size,
                    tta=args.tta,
                    nproc=args.nproc,
                )
                save_npz_atomic(
                    out_path,
                    emb=embs.astype(np.float16),
                    entity_id=np.array(entity_ids, dtype=str),
                    color=np.array(raw_colors, dtype=str),
                    valid=valid,
                    tta=np.int64(args.tta),
                    ckpt_sha256=np.array(ckpt_hash),
                )
                failed = int((~valid).sum())
                print('  寫出 %s（失敗 %d 局）' % (out_path, failed), flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mode', choices=('packed', 'test'), required=True)
    ap.add_argument('--ckpt', required=True, help='train_player_emb 產出之 best_model.pth')
    ap.add_argument('--feat-dir', default='rank-features-v180000')
    ap.add_argument('--split', default='val', choices=('train', 'val'))
    ap.add_argument('--players', default='', help='packed 模式可選之玩家 CSV 篩選清單')
    ap.add_argument('--out', default='', help='packed 模式輸出 .npz 路徑')
    ap.add_argument('--tests-dir', default='tests')
    ap.add_argument('--groups', choices=('a', 'b', 'all'), default='b')
    ap.add_argument('--out-dir', default=os.path.join('player-trained-models', 'emb-cache'))
    ap.add_argument('--tta', type=int, default=8, help='D4 對稱視角數（1～8）')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)
    if args.mode == 'packed' and not args.out:
        ap.error('--mode packed 必須指定 --out')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and use_amp(device, args.no_amp)
    model, cfg = load_emb_model(args.ckpt, device)
    ckpt_hash = file_sha256(args.ckpt)
    print('載入 %s（device=%s, amp=%s, cfg=%s）' % (args.ckpt, device, amp, cfg), flush=True)
    if args.mode == 'packed':
        run_packed_mode(args, model, cfg, device, amp, ckpt_hash)
    else:
        run_test_mode(args, model, cfg, device, amp, ckpt_hash)


if __name__ == '__main__':
    main()
