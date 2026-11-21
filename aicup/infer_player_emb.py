"""深度棋風 Embedding 批次推論與快取工具。

支援兩種推論來源：
    packed  從 bit-packed val 分片（搭配 hd.npz 與可選玩家篩選 CSV）抽取 Embedding 快取
    test    從競賽一候選庫與測試題 CSV 解析 SGF 並以多視角 TTA 抽取 Embedding 快取

另提供候選池中心化與多局聚合餘弦相似度計算函式，供離線評估與融合使用。

--head-moves N 產生前段視角：SGF 只保留前 N 手再抽特徵，只對超過 N 手的局推論，
其他局與解析失敗的局填零向量、valid 為 False。packed 模式改由 --sgf-dir 的 CSV 讀 SGF，
因為打包分片只有最後 250 個盤面。

用法（在 repo 根目錄）：
    python -m aicup.infer_player_emb --mode packed --ckpt player-trained-models/emb/best_model.pth \\
        --feat-dir rank-features-v180000 --players humansl-analysis/val-players-newdev-v180000.csv \\
        --tta 8 --out player-trained-models/emb-cache/newdev_emb.npz
    python -m aicup.infer_player_emb --mode test --ckpt player-trained-models/emb/best_model.pth \\
        --groups b --tta 8 --out-dir player-trained-models/emb-cache
    python -m aicup.infer_player_emb --mode packed --ckpt player-trained-models/emb/best_model.pth \\
        --players humansl-analysis/val-players-newdev-v180000.csv --head-moves 249 \\
        --sgf-dir training-v1.1/training --out player-trained-models/emb-cache/newdev_head_emb.npz
"""
import argparse
import functools
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
from aicup.scoring import LEVELS
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

_head_parser = None  # 每個子進程各自建立的 SGF parser


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


def sgf_to_head_features(sgf_content, head_moves):
    """只保留前 head_moves 手抽特徵，頂層函式供 multiprocessing 使用。

    局長不超過 head_moves 時截斷前後特徵相同，回傳 None 表示沿用完整視角；解析失敗也回傳 None。
    """
    global _head_parser
    if sgf_content.count(';') - 1 <= head_moves:  # 節點數上限
        return None
    if _head_parser is None:
        from utils import SGFParseRankPrediction  # 子進程才載入 sgfmill
        _head_parser = SGFParseRankPrediction()
    try:
        from sgfmill import sgf
        states = _head_parser._extract_game_states(sgf.Sgf_game.from_string(sgf_content))
    except Exception:
        return None
    if len(states) - 1 <= head_moves:  # 第 0 個盤面是根節點的空盤
        return None
    return np.array(_head_parser._extract_features(states[:head_moves + 1]), dtype=np.uint8)


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
    head_moves=0,
):
    """批次解析 SGF 並推論 L2 正規化風格向量，沒有推論的局填零向量、valid 為 False。

    head_moves > 0 時只推論超過 head_moves 手的局，特徵取前 head_moves 手。
    """
    n = len(sgfs)
    embs = np.zeros((n, model.embed_dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return embs, valid

    feat_colors = compute_sgf_feature_colors(sgfs, raw_colors, raw_color_flag) if color_mode != 'none' else None
    feat_fn = functools.partial(sgf_to_head_features, head_moves=head_moves) if head_moves > 0 else sgf_to_features
    pool = Pool(nproc) if nproc > 1 else None
    feats = pool.imap(feat_fn, sgfs, chunksize=8) if pool else map(feat_fn, sgfs)
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


def load_split_meta(split_dir):
    """只讀分片的 key、player_id、rank_idx、color，不載入 features。"""
    paths = packed.list_shards(split_dir)
    if not paths:
        raise FileNotFoundError('%s 下沒有 .npz 分片' % split_dir)
    metas = packed.load_meta(paths)
    return {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}


def read_sgfs_by_key(sgf_dir, keys):
    """從 train_<rank>.csv 讀出 keys（<rank>_<game_id>）對應的 SGF 與執色，順序同 keys。"""
    need = {}
    for k in keys:
        rk, gid = str(k).split('_', 1)
        need.setdefault(rk, set()).add(gid)
    found = {}
    for rk in LEVELS:
        if rk not in need:
            continue
        path = os.path.join(sgf_dir, 'train_%s.csv' % rk)
        if not os.path.exists(path):
            raise FileNotFoundError('找不到 %s' % path)
        for chunk in pd.read_csv(path, usecols=['game_id', 'color', 'sgf_content'], chunksize=50000):
            sub = chunk[chunk['game_id'].isin(need[rk])]
            for gid, col, s in zip(sub['game_id'], sub['color'], sub['sgf_content']):
                found['%s_%s' % (rk, gid)] = (str(s), _parse_color_value(col, '%s %s' % (path, gid)))
    missing = [str(k) for k in keys if str(k) not in found]
    if missing:
        raise SystemExit('%s 缺少 %d 局，例如 %s' % (sgf_dir, len(missing), missing[:3]))
    return [found[str(k)][0] for k in keys], [found[str(k)][1] for k in keys]


def run_packed_head_mode(args, model, cfg, device, amp, ckpt_hash):
    """packed 模式的前段視角：key 與玩家篩選取自分片，SGF 取自 --sgf-dir 的 CSV。"""
    split_dir = os.path.join(args.feat_dir, args.split)
    d = load_split_meta(split_dir)
    if args.players:
        keep_pids = read_player_ids_csv(args.players)
        indices = np.array([i for i, pid in enumerate(d['player_id']) if str(pid) in keep_pids], dtype=np.int64)
    else:
        indices = np.arange(len(d['key']), dtype=np.int64)
    if len(indices) == 0:
        raise SystemExit('篩選後沒有任何對局可推論')

    keys = d['key'][indices]
    sgfs, colors = read_sgfs_by_key(args.sgf_dir, keys)
    csv_ids = np.array([COLOR_TO_ID[c.upper()] for c in colors])
    if not np.array_equal(csv_ids, d['color'][indices]):
        raise SystemExit('CSV 執色與分片 color 不一致')

    print('packed 前段視角推論：%s，%d 局，head_moves=%d，tta=%d' % (
        split_dir, len(indices), args.head_moves, args.tta), flush=True)
    t0 = time.time()
    embs, valid = infer_sgf_embeddings(
        model, sgfs, colors, device, amp,
        color_mode=cfg['color'],
        raw_color_flag=cfg.get('raw_color', False),
        batch_size=args.batch_size,
        tta=args.tta,
        nproc=args.nproc,
        head_moves=args.head_moves,
    )
    save_npz_atomic(
        args.out,
        emb=embs.astype(np.float16),
        key=keys,
        player_id=d['player_id'][indices],
        rank_idx=d['rank_idx'][indices],
        color=d['color'][indices],
        valid=valid,
        head_moves=np.int64(args.head_moves),
        tta=np.int64(args.tta),
        ckpt_sha256=np.array(ckpt_hash),
    )
    print('寫出 %s（%d 局，前段視角 %d 局，%.1f 秒）' % (
        args.out, len(indices), int(valid.sum()), time.time() - t0), flush=True)


def run_packed_mode(args, model, cfg, device, amp, ckpt_hash):
    """從 bit-packed val 分片抽取 Embedding 並存成單一 .npz 快取。"""
    if args.head_moves > 0:
        return run_packed_head_mode(args, model, cfg, device, amp, ckpt_hash)
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
                suffix = '_head_emb.npz' if args.head_moves > 0 else '_emb.npz'
                out_path = os.path.join(args.out_dir, '%s_%s%s' % (tag, role, suffix))
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
                    head_moves=args.head_moves,
                )
                extra = {'head_moves': np.int64(args.head_moves)} if args.head_moves > 0 else {}
                save_npz_atomic(
                    out_path,
                    emb=embs.astype(np.float16),
                    entity_id=np.array(entity_ids, dtype=str),
                    color=np.array(raw_colors, dtype=str),
                    valid=valid,
                    tta=np.int64(args.tta),
                    ckpt_sha256=np.array(ckpt_hash),
                    **extra,
                )
                label = '前段視角' if args.head_moves > 0 else '失敗'
                count = int(valid.sum()) if args.head_moves > 0 else int((~valid).sum())
                print('  寫出 %s（%s %d 局）' % (out_path, label, count), flush=True)


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
    ap.add_argument('--head-moves', type=int, default=0,
                    help='前段視角：只取前 N 手，只推論超過 N 手的局；0 為完整視角')
    ap.add_argument('--sgf-dir', default=os.path.join('training-v1.1', 'training'),
                    help='packed 前段視角讀 SGF 的 train_<rank>.csv 目錄')
    args = ap.parse_args(argv)
    if args.mode == 'packed' and not args.out:
        ap.error('--mode packed 必須指定 --out')
    if args.head_moves < 0:
        ap.error('--head-moves 要 >= 0')
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
