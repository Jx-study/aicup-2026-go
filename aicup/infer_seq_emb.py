"""整局落子序列 Transformer 批次推論與快取工具。

支援兩種推論來源，輸出欄位與 infer_player_emb.py 完全一致，供 fuse_player.py 直接讀取：
    packed  從打包序列 .npz（newdev.npz / val648.npz）抽取 Embedding 快取
    test    從競賽一候選庫與測試題 CSV 解析 SGF 並抽取 Embedding 快取

用法（在 repo 根目錄）：
    python -m aicup.infer_seq_emb --mode packed --ckpt player-trained-models/seq/best_model.pth \\
        --seq-npz move-seq-features-v180000/newdev.npz --tta 1 \\
        --out player-trained-models/seq/emb-cache/newdev_emb.npz
    python -m aicup.infer_seq_emb --mode test --ckpt player-trained-models/seq/best_model.pth \\
        --groups b --tta 1 --out-dir player-trained-models/seq/emb-cache
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from aicup.extract_features import COLOR_TO_ID
from aicup.infer_player_emb import (
    SPLIT_GROUPS,
    _read_cand_csv,
    _read_test_csv,
    save_npz_atomic,
)
from aicup.pack_move_patch import load_patch_npz
from aicup.pack_move_seq import D4_POS_TABLE, MAX_MOVES, PAT3_VOCAB_FILE, map_state_fields, parse_sgf_batch
from aicup.predict_rank import file_sha256
from aicup.train_player_emb import read_player_ids_csv
from aicup.train_rank import use_amp
from aicup.train_seq_emb import (
    GoMoveSeqTransformer,
    extract_seq_embeddings,
    forward_seq_embeddings,
    load_seq_npz,
)


def load_seq_emb_model(ckpt_path, device):
    """依 checkpoint 同目錄之 train_config.json 建立並載入 GoMoveSeqTransformer。"""
    cfg_path = os.path.join(os.path.dirname(ckpt_path), 'train_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'embed_dim': 256,
            'max_len': MAX_MOVES,
            'dropout': 0.1,
            'head_dropout': 0.2,
            'pool_mode': 'target',
            'augment_d4': False,
            'raw_color': False,
        }

    model = GoMoveSeqTransformer(
        d_model=int(cfg['d_model']),
        nhead=int(cfg['nhead']),
        num_layers=int(cfg['num_layers']),
        dim_feedforward=int(cfg['dim_feedforward']),
        embed_dim=int(cfg['embed_dim']),
        max_len=int(cfg.get('max_len', MAX_MOVES)),
        dropout=float(cfg.get('dropout', 0.1)),
        head_dropout=float(cfg.get('head_dropout', 0.2)),
        pool_mode=str(cfg.get('pool_mode', 'target')),
        state_feats=bool(cfg.get('state_feats', False)),
        conv_kernel=int(cfg.get('conv_kernel', 0)),
        patch_size=int(cfg.get('patch_size', 0)),
        spatial_bias=bool(cfg.get('spatial_bias', False)),
        stage_pool=bool(cfg.get('stage_pool', False)),
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def infer_sgf_seq_embeddings(
    model,
    sgfs,
    raw_colors,
    device,
    amp,
    max_moves=MAX_MOVES,
    raw_color_flag=False,
    batch_size=256,
    tta=1,
    nproc=1,
    pat3_vocab=None,
):
    """批次解析 SGF 落子序列並推論 L2 正規化風格向量，解析失敗的局填零向量、valid 為 False。

    state_feats 模型需傳入訓練時的 pat3_vocab。
    """
    n = len(sgfs)
    embs = np.zeros((n, model.embed_dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return embs, valid

    color_ids = [COLOR_TO_ID[str(c).strip().upper()] for c in raw_colors]
    use_state = bool(getattr(model, 'state_feats', False))
    patch_size = int(getattr(model, 'patch_size', 0))
    if use_state and pat3_vocab is None:
        raise ValueError('state_feats 模型需要 pat3_vocab')
    res = parse_sgf_batch(
        sgfs, color_ids, max_moves=max_moves, raw_color=raw_color_flag, nproc=nproc, state_feats=use_state,
        patch_size=patch_size,
    )
    pos_np, role_np, ok_np = res.pos, res.is_target, res.ok
    state_np = map_state_fields(res.st4, res.pat3, pat3_vocab) if use_state else None
    patch_np = res.patch
    valid_idxs = np.nonzero(ok_np)[0]
    if len(valid_idxs) == 0:
        return embs, valid

    d4_table = torch.from_numpy(D4_POS_TABLE.astype(np.int64)).to(device) if tta > 1 else None
    model.eval()
    with torch.no_grad():
        for i in range(0, len(valid_idxs), batch_size):
            sl = valid_idxs[i:i + batch_size]
            pb = torch.from_numpy(pos_np[sl].astype(np.int64)).to(device)
            rb = torch.from_numpy(role_np[sl].astype(np.int64)).to(device)
            sb = torch.from_numpy(state_np[sl].astype(np.int64)).to(device) if use_state else None
            xb = torch.from_numpy(patch_np[sl]).to(device) if patch_size else None
            e = forward_seq_embeddings(
                model, pb, rb, amp, tta=tta, d4_table=d4_table, state=sb, patch=xb,
            ).cpu().numpy()
            embs[sl] = e
            valid[sl] = True
    return embs, valid


def run_packed_mode(args, model, device, amp, ckpt_hash):
    """從打包序列 .npz 抽取 Embedding 並存成與 infer_player_emb 相容的 .npz 快取。"""
    seq_path = args.seq_npz or os.path.join(args.seq_dir, '%s.npz' % args.split)
    pos, is_target, meta = load_seq_npz(seq_path, torch.device('cpu'), model.state_feats)
    state = meta.pop('state', None) if model.state_feats else None
    if model.state_feats and state is None:
        raise SystemExit('%s 沒有狀態欄位，state_feats 模型需要 --state-feats 打包的序列' % seq_path)
    patch = None
    if model.patch_size:
        if not args.patch_dir:
            raise SystemExit('局部盤面模型需要 --patch-dir')
        split_name = os.path.splitext(os.path.basename(seq_path))[0]
        patch_np, ps = load_patch_npz(os.path.join(args.patch_dir, '%s_patch.npz' % split_name), meta['key'])
        if ps != model.patch_size:
            raise SystemExit('局部盤面大小 %d 與模型 %d 不同' % (ps, model.patch_size))
        patch = torch.from_numpy(patch_np)
    if args.players:
        keep_pids = read_player_ids_csv(args.players)
        indices = np.array([i for i, pid in enumerate(meta['player_id']) if str(pid) in keep_pids], dtype=np.int64)
    else:
        indices = np.arange(len(meta['key']), dtype=np.int64)

    if len(indices) == 0:
        raise SystemExit('篩選後沒有任何對局可推論')

    print('packed 序列推論：%s，%d 局，tta=%d' % (seq_path, len(indices), args.tta), flush=True)
    t0 = time.time()
    embs = extract_seq_embeddings(
        model, pos, is_target, indices, args.batch_size, device, amp, tta=args.tta, state=state, patch=patch,
    )
    save_npz_atomic(
        args.out,
        emb=embs.astype(np.float16),
        key=meta['key'][indices],
        player_id=meta['player_id'][indices],
        rank_idx=meta['rank_idx'][indices],
        color=meta['color'][indices],
        tta=np.int64(args.tta),
        ckpt_sha256=np.array(ckpt_hash),
    )
    print('寫出 %s（%d 局，%.1f 秒）' % (args.out, len(indices), time.time() - t0), flush=True)


def run_test_mode(args, model, cfg, device, amp, ckpt_hash):
    """從競賽一候選庫與測試題 CSV 抽取序列 Embedding 並依分組存成 .npz 快取。"""
    groups = ['a', 'b'] if args.groups == 'all' else [args.groups]
    max_moves = int(cfg.get('max_len', MAX_MOVES))
    raw_color_flag = bool(cfg.get('raw_color', False))
    pat3_vocab = None
    if model.state_feats:
        vocab_path = os.path.join(os.path.dirname(args.ckpt), PAT3_VOCAB_FILE)
        if not os.path.exists(vocab_path):
            raise SystemExit('找不到 %s，state_feats 模型需要訓練時的 pat3 詞表' % vocab_path)
        pat3_vocab = np.load(vocab_path)
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
                t0 = time.time()
                embs, valid = infer_sgf_seq_embeddings(
                    model, sgfs, raw_colors, device, amp,
                    max_moves=max_moves,
                    raw_color_flag=raw_color_flag,
                    batch_size=args.batch_size,
                    tta=args.tta,
                    nproc=args.nproc,
                    pat3_vocab=pat3_vocab,
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
                print(
                    '  寫出 %s（失敗 %d 局，%.1f 秒）'
                    % (out_path, int((~valid).sum()), time.time() - t0),
                    flush=True,
                )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mode', choices=('packed', 'test'), required=True)
    ap.add_argument('--ckpt', required=True, help='train_seq_emb 產出之 best_model.pth')
    ap.add_argument('--seq-dir', default='move-seq-features-v180000')
    ap.add_argument('--split', default='newdev', choices=('train', 'newdev', 'val648'))
    ap.add_argument('--seq-npz', default='', help='直接指定單一序列 .npz 路徑，優先於 --seq-dir/--split')
    ap.add_argument('--players', default='', help='packed 模式可選之玩家 CSV 篩選清單')
    ap.add_argument('--patch-dir', default='', help='packed 模式局部盤面模型用的 pack_move_patch 目錄')
    ap.add_argument('--out', default='', help='packed 模式輸出 .npz 路徑')
    ap.add_argument('--tests-dir', default='tests')
    ap.add_argument('--groups', choices=('a', 'b', 'all'), default='b')
    ap.add_argument('--out-dir', default=os.path.join('player-trained-models', 'seq-emb-cache'))
    ap.add_argument('--tta', type=int, default=1, help='D4 對稱視角數（預設 1，D4 訓練組可用 8）')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)
    if args.mode == 'packed' and not args.out:
        ap.error('--mode packed 必須指定 --out')
    if args.tta < 1 or args.tta > 8:
        ap.error('--tta 必須在 1 到 8 之間')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and use_amp(device, args.no_amp)
    model, cfg = load_seq_emb_model(args.ckpt, device)
    ckpt_hash = file_sha256(args.ckpt)
    print('載入 %s（device=%s, amp=%s, cfg=%s）' % (args.ckpt, device, amp, cfg), flush=True)
    if args.mode == 'packed':
        run_packed_mode(args, model, device, amp, ckpt_hash)
    else:
        run_test_mode(args, model, cfg, device, amp, ckpt_hash)


if __name__ == '__main__':
    main()
