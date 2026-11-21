"""整局落子序列 Transformer 風格編碼器訓練腳本。

直接讀取緊湊落子序列 .npz（前 320 手座標與目標方標記），於 GPU 即時派生線高與
前手相對位移特徵，經 6 層 Pre-LN Transformer、Attentive Statistics Pooling 與
BNNeck 輸出 L2 正規化單位向量。預設不做 D4 增強以保留絕對角落偏好，--augment-d4
開啟 D4 增強。一個 epoch 定義為完整看過一遍訓練局。

用法（在 repo 根目錄）：
    python -m aicup.train_seq_emb --seq-dir move-seq-features-v180000 \\
        --ckpt-dir player-trained-models/seq-smoke --epochs 5
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from aicup.pack_move_seq import (
    BOARD_SIZE,
    D4_POS_TABLE,
    MAX_MOVES,
    NUM_POS,
    NUM_ROLES,
    PAD_POS,
    PASS_POS,
    ROLE_TARGET,
)
from aicup.train_player_emb import (
    SubCenterArcFace,
    build_player_index,
    evaluate_b_group_embeddings,
    read_player_ids_csv,
    sample_within_rank_pk_batches,
    supcon_loss,
)
from aicup.train_rank import log_run, lr_at, pick_data_device, save_atomic, use_amp

REL_CLAMP = 9
REL_NUM = REL_CLAMP * 2 + 2  # -9..+9 共 19 格，加 1 格無效
DIST_BUCKETS = 9  # 0..8 桶，加 1 格無效
DIST_NUM = DIST_BUCKETS + 1
LINE_NUM = (BOARD_SIZE + 1) // 2 + 1  # 0..9 線共 10 格，加 1 格 pass/pad


def bucket_l1_distance(l1):
    """將曼哈頓距離映射至 0..8 距離桶。"""
    # 距離桶 0..4 各自一桶，其餘依 5-6、7-9、10-14、>=15 分 4 桶
    b = torch.clamp(l1, max=4)
    b = torch.where(l1 >= 5, torch.full_like(b, 5), b)
    b = torch.where(l1 >= 7, torch.full_like(b, 6), b)
    b = torch.where(l1 >= 10, torch.full_like(b, 7), b)
    b = torch.where(l1 >= 15, torch.full_like(b, 8), b)
    return b


def apply_d4_pos(pos, k, table):
    """依 D4 座標置換表對 (B, L) 落子座標套用對稱變換，k 可為純量或 (B,) 張量。"""
    if isinstance(k, int):
        if k == 0:
            return pos
        return table[k][pos]
    return table[k.unsqueeze(1), pos]


class PreLNTransformerBlock(nn.Module):
    """單層 Pre-LN Transformer Encoder，使用 PyTorch 原生 SDPA 加速。"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError('d_model (%d) 必須能被 nhead (%d) 整除' % (d_model, nhead))
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = float(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_valid_mask):
        """x: (B, L, D)，key_valid_mask: (B, L) bool，True 表示可被注意。"""
        bsz, seq_len, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(bsz, seq_len, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # bool attn_mask：True 表示允許注意，形狀 (B, 1, 1, L)
        attn_mask = key_valid_mask[:, None, None, :]
        drop_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.drop1(self.out_proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x


class AttentiveStatsPool(nn.Module):
    """Attentive Statistics Pooling，輸出加權平均與加權標準差串接（2 * d_model）。"""

    def __init__(self, d_model):
        super().__init__()
        hidden = max(32, d_model // 2)
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, mask):
        """h: (B, L, D)，mask: (B, L) bool，回傳 (B, 2 * D) float32。"""
        h_f = h.float()
        scores = self.attn(h).squeeze(-1).float()
        has_any = mask.any(dim=1)
        safe_mask = mask.clone()
        if not has_any.all():
            safe_mask[~has_any, 0] = True
        scores = scores.masked_fill(~safe_mask, -1e4)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (w * h_f).sum(dim=1)
        var = (w * (h_f - mean.unsqueeze(1)).pow(2)).sum(dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        out = torch.cat([mean, std], dim=1)
        if not has_any.all():
            out = out * has_any.unsqueeze(1).to(out.dtype)
        return out


class GoMoveSeqTransformer(nn.Module):
    """整局落子序列 Transformer 風格編碼器，輸出 L2 正規化單位向量。"""

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        embed_dim=256,
        max_len=MAX_MOVES,
        dropout=0.1,
        head_dropout=0.2,
        pool_mode='target',
    ):
        super().__init__()
        if pool_mode not in ('target', 'all'):
            raise ValueError('pool_mode 必須為 target 或 all，收到 %r' % (pool_mode,))
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.embed_dim = int(embed_dim)
        self.max_len = int(max_len)
        self.dropout_p = float(dropout)
        self.head_dropout_p = float(head_dropout)
        self.pool_mode = pool_mode

        self.coord_emb = nn.Embedding(NUM_POS, 128)
        self.role_emb = nn.Embedding(NUM_ROLES, 32)
        self.line_emb = nn.Embedding(LINE_NUM, 16)
        self.rel_dr_emb = nn.Embedding(REL_NUM, 24)
        self.rel_dc_emb = nn.Embedding(REL_NUM, 24)
        self.rel_dist_emb = nn.Embedding(DIST_NUM, 16)
        self.self_dist_emb = nn.Embedding(DIST_NUM, 16)
        self.step_emb = nn.Embedding(self.max_len, 16)

        in_dim = 128 + 32 + 16 + 24 + 24 + 16 + 16 + 16  # 272
        self.in_proj = nn.Linear(in_dim, self.d_model)
        self.in_norm = nn.LayerNorm(self.d_model)
        self.in_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PreLNTransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout)
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        self.pool = AttentiveStatsPool(self.d_model)
        feat_dim = self.d_model * 2
        self.bn_pool = nn.BatchNorm1d(feat_dim)
        self.head_drop = nn.Dropout(head_dropout)
        self.fc_emb = nn.Linear(feat_dim, self.embed_dim, bias=False)
        self.bn_emb = nn.BatchNorm1d(self.embed_dim)

    def _derive_features(self, pos, is_target):
        """由座標與角色張量即時派生線高、前手相對位移與前手己方距離。"""
        bsz, seq_len = pos.shape
        device = pos.device
        is_board = pos < PASS_POS
        r = torch.where(is_board, pos // BOARD_SIZE, torch.zeros_like(pos))
        c = torch.where(is_board, pos % BOARD_SIZE, torch.zeros_like(pos))

        # 線高：0..9，pass/pad 對應 LINE_NUM - 1
        line_r = torch.minimum(r, (BOARD_SIZE - 1) - r)
        line_c = torch.minimum(c, (BOARD_SIZE - 1) - c)
        line = torch.where(is_board, torch.minimum(line_r, line_c), torch.full_like(pos, LINE_NUM - 1))

        # 距前一手對手落子之相對位移與曼哈頓距離桶
        prev_r = F.pad(r[:, :-1], (1, 0), value=0)
        prev_c = F.pad(c[:, :-1], (1, 0), value=0)
        prev_board = F.pad(is_board[:, :-1], (1, 0), value=False)
        prev_role = F.pad(is_target[:, :-1], (1, 0), value=NUM_ROLES - 1)
        rel_valid = is_board & prev_board & (is_target != prev_role)
        dr_raw = torch.clamp(r - prev_r, -REL_CLAMP, REL_CLAMP) + REL_CLAMP
        dc_raw = torch.clamp(c - prev_c, -REL_CLAMP, REL_CLAMP) + REL_CLAMP
        l1_prev = (r - prev_r).abs() + (c - prev_c).abs()
        dr_idx = torch.where(rel_valid, dr_raw, torch.full_like(pos, REL_NUM - 1))
        dc_idx = torch.where(rel_valid, dc_raw, torch.full_like(pos, REL_NUM - 1))
        dist_idx = torch.where(rel_valid, bucket_l1_distance(l1_prev), torch.full_like(pos, DIST_NUM - 1))

        # 距前手己方落子（標準交替落子為 t - 2）之距離桶
        if seq_len >= 2:
            prev2_r = F.pad(r[:, :-2], (2, 0), value=0)
            prev2_c = F.pad(c[:, :-2], (2, 0), value=0)
            prev2_board = F.pad(is_board[:, :-2], (2, 0), value=False)
            prev2_role = F.pad(is_target[:, :-2], (2, 0), value=NUM_ROLES - 1)
            self_valid = is_board & prev2_board & (is_target == prev2_role)
            l1_self = (r - prev2_r).abs() + (c - prev2_c).abs()
            self_dist_idx = torch.where(self_valid, bucket_l1_distance(l1_self), torch.full_like(pos, DIST_NUM - 1))
        else:
            self_dist_idx = torch.full_like(pos, DIST_NUM - 1)

        steps = torch.arange(seq_len, device=device, dtype=torch.long).clamp_max(self.max_len - 1)
        steps = steps.unsqueeze(0).expand(bsz, seq_len)
        return line, dr_idx, dc_idx, dist_idx, self_dist_idx, steps

    def forward(self, pos, is_target):
        """輸入 pos (B, L) 與 is_target (B, L)，回傳 (B, embed_dim) L2 正規化向量。"""
        pos_l = pos.long()
        role_l = is_target.long()
        valid_mask = pos_l != PAD_POS
        has_valid = valid_mask.any(dim=1)

        # 空局確保第 0 格開啟，避免 SDPA 全遮罩產生 NaN
        attn_valid = valid_mask.clone()
        if not has_valid.all():
            attn_valid[~has_valid, 0] = True

        line, dr_idx, dc_idx, dist_idx, self_dist_idx, steps = self._derive_features(pos_l, role_l)
        tok = torch.cat([
            self.coord_emb(pos_l),
            self.role_emb(role_l),
            self.line_emb(line),
            self.rel_dr_emb(dr_idx),
            self.rel_dc_emb(dc_idx),
            self.rel_dist_emb(dist_idx),
            self.self_dist_emb(self_dist_idx),
            self.step_emb(steps),
        ], dim=-1)
        h = self.in_drop(self.in_norm(self.in_proj(tok)))

        for blk in self.blocks:
            h = blk(h, attn_valid)
        h = self.final_norm(h)

        if self.pool_mode == 'target':
            target_mask = valid_mask & (role_l == ROLE_TARGET)
            has_target = target_mask.any(dim=1, keepdim=True)
            pool_mask = torch.where(has_target, target_mask, valid_mask)
        else:
            pool_mask = valid_mask

        pooled = self.pool(h, pool_mask)
        pooled = self.head_drop(self.bn_pool(pooled))
        emb = self.bn_emb(self.fc_emb(pooled))
        emb = F.normalize(emb.float(), p=2, dim=1)
        if not has_valid.all():
            emb = emb * has_valid.unsqueeze(1).to(emb.dtype)
        return emb


def load_seq_npz(path, device):
    """載入單檔落子序列 .npz，將 pos（int16）與 is_target（uint8）移至指定裝置。"""
    if not os.path.exists(path):
        raise SystemExit('找不到序列特徵檔：%s' % path)
    with np.load(path, allow_pickle=False) as z:
        pos = torch.from_numpy(z['pos'].astype(np.int16)).to(device)
        is_target = torch.from_numpy(z['is_target'].astype(np.uint8)).to(device)
        meta = {
            'key': z['key'],
            'player_id': z['player_id'],
            'rank_idx': z['rank_idx'],
            'color': z['color'],
            'length': z['length'],
        }
        if 'hd' in z.files:
            meta['hd'] = z['hd']
        meta['raw_color'] = bool(z['raw_color']) if 'raw_color' in z.files else False
    return pos, is_target, meta


def forward_seq_embeddings(model, pos, is_target, amp, tta=1, d4_table=None):
    """計算一個 batch 的序列風格向量，tta 支援 1 至 8 種 D4 對稱平均。"""
    n_views = max(1, min(int(tta), 8))
    if n_views > 1 and d4_table is None:
        d4_table = torch.from_numpy(D4_POS_TABLE.astype(np.int64)).to(pos.device)
    total = None
    pos_l = pos.long()
    role_l = is_target.long()
    for k in range(n_views):
        pk = apply_d4_pos(pos_l, k, d4_table) if k else pos_l
        with torch.autocast(pos.device.type, dtype=torch.bfloat16, enabled=amp):
            e = model(pk, role_l)
        e_f = e.float()
        total = e_f if total is None else (total + e_f)
    return F.normalize(total / n_views, p=2, dim=1)


def recompute_seq_bn_stats(model, pos, is_target, batch_size, device, amp, n_batches=100, seed=42):
    """回傳 model 副本，以隨機訓練 batch 重算 BNNeck 累積統計量。"""
    bn = copy.deepcopy(model)
    norms = [m for m in bn.modules() if isinstance(m, nn.BatchNorm1d)]
    orig_mom = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None

    n_total = min(len(pos), n_batches * batch_size)
    gen = torch.Generator(device=pos.device).manual_seed(seed)
    order = torch.randperm(len(pos), generator=gen, device=pos.device)[:n_total]
    bn.train()
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            sl = order[i:i + batch_size]
            if len(sl) < 2:
                continue
            pb = pos[sl].to(device, non_blocking=True)
            rb = is_target[sl].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                bn(pb, rb)

    for m, mom in zip(norms, orig_mom):
        m.momentum = mom
    bn.eval()
    return bn


def extract_seq_embeddings(model, pos, is_target, indices, batch_size, device, amp, tta=1):
    """對指定局索引批次抽取 L2 正規化風格向量，回傳 (len(indices), embed_dim) float32。"""
    was_training = model.training
    model.eval()
    idx_t = torch.as_tensor(indices, dtype=torch.long, device=pos.device)
    d4_table = torch.from_numpy(D4_POS_TABLE.astype(np.int64)).to(device) if tta > 1 else None
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(idx_t), batch_size):
                sl = idx_t[i:i + batch_size]
                pb = pos[sl].to(device, non_blocking=True)
                rb = is_target[sl].to(device, non_blocking=True)
                emb = forward_seq_embeddings(model, pb, rb, amp, tta=tta, d4_table=d4_table)
                out.append(emb.cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(out, axis=0).astype(np.float32) if out else np.zeros((0, model.embed_dim), dtype=np.float32)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seq-dir', default='move-seq-features-v180000')
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--exclude-players', nargs='*', default=[],
                    help='額外從 train 排除的玩家 CSV 清單')
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--steps-per-epoch', type=int, default=0,
                    help='每個 epoch 的 PK 批次數，0 為依訓練集總局數自動計算（完整看過一遍）')
    ap.add_argument('--p-players', type=int, default=64, help='每個 batch 採樣的玩家數 P')
    ap.add_argument('--k-games', type=int, default=4, help='每位玩家採樣的局數 K')
    ap.add_argument('--min-games', type=int, default=5, help='納入訓練的最少玩家局數')
    ap.add_argument('--adjacent-prob', type=float, default=0.5, help='批次混合相鄰 2 個段位的機率')
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--num-layers', type=int, default=6)
    ap.add_argument('--dim-feedforward', type=int, default=1024)
    ap.add_argument('--embed-dim', type=int, default=256)
    ap.add_argument('--max-len', type=int, default=MAX_MOVES)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--head-dropout', type=float, default=0.2)
    ap.add_argument('--pool-mode', choices=('target', 'all'), default='target')
    ap.add_argument('--sub-centers', type=int, default=2)
    ap.add_argument('--arc-scale', type=float, default=30.0)
    ap.add_argument('--arc-margin', type=float, default=0.25)
    ap.add_argument('--supcon-weight', type=float, default=0.3)
    ap.add_argument('--supcon-temp', type=float, default=0.07)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--warmup-steps', type=int, default=500)
    ap.add_argument('--grad-clip', type=float, default=5.0)
    ap.add_argument('--augment-d4', action='store_true', default=False,
                    help='開啟 D4 8 對稱增強（預設關閉以保留絕對角落偏好）')
    ap.add_argument('--val-batch-size', type=int, default=256)
    ap.add_argument('--val-tta', type=int, default=1, help='dev 驗證的 TTA 視角數（1～8）')
    ap.add_argument('--bn-batches', type=int, default=100,
                    help='每個 epoch 評估前重算 BNNeck 統計量的批次數，0 為不重算')
    ap.add_argument('--eval-seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--fresh', action='store_true', help='忽略既有 last.pt')
    ap.add_argument('--max-steps', type=int, default=0, help='每個 epoch 最多步數，0 為不限，用於冒煙測試')
    ap.add_argument('--log-every', type=int, default=400)
    args = ap.parse_args(argv)
    if args.p_players < 2 or args.k_games < 1 or args.sub_centers < 1:
        ap.error('--p-players 要 >= 2，--k-games 與 --sub-centers 要 >= 1')
    if args.min_games < 1 or args.supcon_weight < 0 or args.bn_batches < 0:
        ap.error('--min-games 要 >= 1，--supcon-weight 與 --bn-batches 要 >= 0')
    return args


def _seq_files_bytes(seq_dir):
    """估算 train 與 newdev 解壓後 pos、is_target 的記憶體占用（byte）。"""
    total = 0
    for name in ('train.npz', 'newdev.npz'):
        p = os.path.join(seq_dir, name)
        if os.path.exists(p):
            total += os.path.getsize(p) * 3
    return total


def main(argv=None):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = use_amp(device, args.no_amp)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_device = pick_data_device(args, device, _seq_files_bytes(args.seq_dir))
    print('載入序列資料到 %s' % data_device, flush=True)
    dev_pos, dev_role, dev_meta = load_seq_npz(os.path.join(args.seq_dir, 'newdev.npz'), data_device)
    tr_pos, tr_role, tr_meta = load_seq_npz(os.path.join(args.seq_dir, 'train.npz'), data_device)

    dev_pids = dev_meta['player_id']
    dev_pid_set = {str(p) for p in dev_pids.tolist()}
    exclude_pids = set(dev_pid_set)
    for p in args.exclude_players:
        exclude_pids |= read_player_ids_csv(p)

    # 再次檢查 train 與 newdev 玩家互斥
    tr_pid_set = {str(p) for p in tr_meta['player_id'].tolist()}
    if tr_pid_set & dev_pid_set:
        raise SystemExit('train.npz 與 newdev.npz 玩家有重疊：%d 位' % len(tr_pid_set & dev_pid_set))

    player_games, rank_players, class_keys = build_player_index(
        tr_meta, min_games=args.min_games, exclude_pids=exclude_pids,
    )
    num_classes = len(class_keys)
    total_train_games = sum(len(g) for g in player_games.values())
    batch_size = args.p_players * args.k_games
    steps_per_epoch = (
        args.steps_per_epoch if args.steps_per_epoch > 0
        else max(1, math.ceil(total_train_games / batch_size))
    )
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)

    seq_len = int(tr_pos.shape[1])
    max_len = max(args.max_len, seq_len)

    model = GoMoveSeqTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        embed_dim=args.embed_dim,
        max_len=max_len,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        pool_mode=args.pool_mode,
    ).to(device)
    arcface = SubCenterArcFace(
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_sub_centers=args.sub_centers,
        scale=args.arc_scale,
        margin=args.arc_margin,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, 'last.pt')
    best_path = os.path.join(args.ckpt_dir, 'best_model.pth')
    config = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'embed_dim': args.embed_dim,
        'max_len': max_len,
        'dropout': args.dropout,
        'head_dropout': args.head_dropout,
        'pool_mode': args.pool_mode,
        'p_players': args.p_players,
        'k_games': args.k_games,
        'min_games': args.min_games,
        'sub_centers': args.sub_centers,
        'arc_scale': args.arc_scale,
        'arc_margin': args.arc_margin,
        'supcon_weight': args.supcon_weight,
        'supcon_temp': args.supcon_temp,
        'lr': args.lr,
        'epochs': args.epochs,
        'steps_per_epoch': steps_per_epoch,
        'augment_d4': args.augment_d4,
        'seed': args.seed,
        'num_classes': num_classes,
        'amp': amp,
    }

    start_epoch, history, best_score = 0, [], -1.0
    if os.path.exists(last_path) and not args.fresh:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt['config'] != config:
            raise SystemExit(
                'last.pt 的設定與這次不同：\n  舊 %s\n  新 %s\n'
                '要用新設定請加 --fresh 或換 --ckpt-dir' % (ckpt['config'], config)
            )
        model.load_state_dict(ckpt['model'])
        arcface.load_state_dict(ckpt['arcface'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, history, best_score = ckpt['epoch'], ckpt['history'], ckpt['best_score']
        print('從 epoch %d 接續（目前最佳 dev_top5_csls %.4f）' % (start_epoch, best_score), flush=True)

    infer_cfg = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'embed_dim': args.embed_dim,
        'max_len': max_len,
        'dropout': args.dropout,
        'head_dropout': args.head_dropout,
        'pool_mode': args.pool_mode,
        'augment_d4': args.augment_d4,
        'raw_color': bool(tr_meta.get('raw_color', False)),
    }
    with open(os.path.join(args.ckpt_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump(infer_cfg, f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        'device %s | data %s | amp %s | augment_d4 %s | pool %s | train 玩家 %d (%s 局, %d steps/ep) | dev %s 局 | 參數量 %s'
        % (
            device, data_device, amp, args.augment_d4, args.pool_mode,
            num_classes, format(total_train_games, ','), steps_per_epoch,
            format(len(dev_pos), ','), format(n_params, ','),
        ),
        flush=True,
    )
    if start_epoch >= args.epochs:
        print('已完成 %d 個 epoch，要改設定重訓請換 --ckpt-dir 或加 --fresh' % start_epoch)
        return

    log_run(args.ckpt_dir, start_epoch, device)
    total_steps = steps_per_epoch * args.epochs
    d4_table = torch.from_numpy(D4_POS_TABLE.astype(np.int64)).to(device)
    dev_indices = np.arange(len(dev_pos), dtype=np.int64)

    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)
        pk_rng = np.random.default_rng(args.seed + epoch)
        aug_gen = torch.Generator(device=device).manual_seed(args.seed + epoch)

        epoch_batches = sample_within_rank_pk_batches(
            player_games,
            rank_players,
            n_batches=steps_per_epoch,
            p_players=args.p_players,
            k_games=args.k_games,
            rng=pk_rng,
            adjacent_prob=args.adjacent_prob,
        )

        model.train()
        arcface.train()
        t0 = time.time()
        loss_sum, arc_sum, sup_sum, correct_sum, total_sum = 0.0, 0.0, 0.0, 0, 0

        for step, (g_idx, y_np) in enumerate(epoch_batches):
            global_step = epoch * steps_per_epoch + step
            cur_lr = lr_at(global_step, total_steps, args.lr, 'cosine', args.warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = cur_lr

            idx_t = torch.as_tensor(g_idx, dtype=torch.long, device=tr_pos.device)
            pb = tr_pos[idx_t].to(device, non_blocking=True).long()
            rb = tr_role[idx_t].to(device, non_blocking=True).long()
            if args.augment_d4:
                ks = torch.randint(0, 8, (pb.size(0),), generator=aug_gen, device=device)
                pb = apply_d4_pos(pb, ks, d4_table)
            yb = torch.as_tensor(y_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                emb = model(pb, rb)
            l_arc, corr = arcface(emb, yb)
            l_sup = (
                supcon_loss(emb, yb, temperature=args.supcon_temp)
                if args.supcon_weight > 0 else l_arc.new_tensor(0.0)
            )
            loss = l_arc + args.supcon_weight * l_sup
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface.parameters()),
                    args.grad_clip,
                )
            optimizer.step()

            loss_sum += float(loss.item())
            arc_sum += float(l_arc.item())
            sup_sum += float(l_sup.item())
            correct_sum += corr
            total_sum += int(yb.size(0))

            if args.log_every and (step + 1) % args.log_every == 0:
                print(
                    '  ep%d step %d/%d  seen %s  loss %.4f (arc %.4f sup %.4f)  cos_acc %.2f%%  %.1f 分'
                    % (
                        epoch + 1, step + 1, steps_per_epoch, format(total_sum, ','),
                        loss_sum / (step + 1), arc_sum / (step + 1), sup_sum / (step + 1),
                        100.0 * correct_sum / max(total_sum, 1),
                        (time.time() - t0) / 60,
                    ),
                    flush=True,
                )

        train_min = (time.time() - t0) / 60
        raw_dev_emb = extract_seq_embeddings(
            model, dev_pos, dev_role, dev_indices, args.val_batch_size, device, amp, tta=args.val_tta,
        )
        raw_dev_metrics = evaluate_b_group_embeddings(raw_dev_emb, dev_pids, seeds=args.eval_seeds)

        eval_model = model
        dev_metrics = raw_dev_metrics
        if args.bn_batches > 0:
            eval_model = recompute_seq_bn_stats(
                model, tr_pos, tr_role, args.val_batch_size, device, amp,
                n_batches=args.bn_batches, seed=args.seed + epoch,
            )
            bn_dev_emb = extract_seq_embeddings(
                eval_model, dev_pos, dev_role, dev_indices, args.val_batch_size, device, amp, tta=args.val_tta,
            )
            dev_metrics = evaluate_b_group_embeddings(bn_dev_emb, dev_pids, seeds=args.eval_seeds)

        end_lr = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch + 1,
            'seen_games': total_sum,
            'train_loss': loss_sum / max(steps_per_epoch, 1),
            'arc_loss': arc_sum / max(steps_per_epoch, 1),
            'supcon_loss': sup_sum / max(steps_per_epoch, 1),
            'train_cos_acc': 100.0 * correct_sum / max(total_sum, 1),
            'lr': end_lr,
            'raw_dev_top5_csls': raw_dev_metrics['dev_top5_csls'],
            **dev_metrics,
            'train_minutes': train_min,
            'total_minutes': (time.time() - t0) / 60,
        }
        history.append(row)
        print(
            'Epoch %d/%d  seen %s  loss %.4f  cos_acc %.2f%% | dev_top5 %.4f  csls %.4f (raw %.4f; q1 %.4f q2 %.4f q3 %.4f q5 %.4f) | %.1f 分'
            % (
                epoch + 1, args.epochs, format(total_sum, ','),
                row['train_loss'], row['train_cos_acc'],
                row['dev_top5'], row['dev_top5_csls'], row['raw_dev_top5_csls'],
                row['dev_csls_q1'], row['dev_csls_q2'], row['dev_csls_q3'], row['dev_csls_q5'],
                row['total_minutes'],
            ),
            flush=True,
        )

        torch.save(eval_model.state_dict(), os.path.join(args.ckpt_dir, 'epoch_%d.pth' % (epoch + 1)))
        if row['dev_top5_csls'] > best_score:
            best_score = row['dev_top5_csls']
            save_atomic(eval_model.state_dict(), best_path)
            print('  更新 best_model.pth (dev_top5_csls %.4f)' % best_score, flush=True)
        if eval_model is not model:
            del eval_model

        pd.DataFrame(history).to_csv(os.path.join(args.ckpt_dir, 'history.csv'), index=False)
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'arcface': arcface.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_score': best_score,
            'history': history,
            'config': config,
        }
        save_atomic(state, last_path)


if __name__ == '__main__':
    main()
