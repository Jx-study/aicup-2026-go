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
import shutil
import time
import zipfile

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
    PAT3_NUM,
    PAT3_VOCAB_FILE,
    ROLE_PAD,
    ROLE_TARGET,
    STATE_FIELDS,
)
from aicup.pack_move_patch import load_patch_npz
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
PATCH_EMB_DIM = 64
SPATIAL_NUM = 9  # 距離桶 0..7，加 1 格 pass/pad
STAGE_BOUNDS = (50, 150)  # 開局、中盤、官子的分段手數


def amp_autocast(device_type, amp):
    """amp 為 False、True（bf16）或 'fp16'，回傳對應的 autocast context。"""
    dtype = torch.float16 if amp == 'fp16' else torch.bfloat16
    return torch.autocast(device_type, dtype=dtype, enabled=bool(amp))


def bucket_l1_distance(l1):
    """將曼哈頓距離映射至 0..8 距離桶。"""
    # 距離桶 0..4 各自一桶，其餘依 5-6、7-9、10-14、>=15 分 4 桶
    b = torch.clamp(l1, max=4)
    b = torch.where(l1 >= 5, torch.full_like(b, 5), b)
    b = torch.where(l1 >= 7, torch.full_like(b, 6), b)
    b = torch.where(l1 >= 10, torch.full_like(b, 7), b)
    b = torch.where(l1 >= 15, torch.full_like(b, 8), b)
    return b


def spatial_pair_buckets(pos):
    """回傳 (B, L, L) int32 兩手落子點曼哈頓距離桶，任一手為 pass/pad 時為 SPATIAL_NUM - 1。"""
    is_board = pos < PASS_POS
    # (B, L, L) 暫存用 int16 降低 VRAM
    r = torch.where(is_board, pos // BOARD_SIZE, torch.zeros_like(pos)).to(torch.int16)
    c = torch.where(is_board, pos % BOARD_SIZE, torch.zeros_like(pos)).to(torch.int16)
    l1 = (r[:, :, None] - r[:, None, :]).abs() + (c[:, :, None] - c[:, None, :]).abs()
    # 0..4 各一桶，5-6、7-10、>=11 各一桶
    b = torch.clamp(l1, max=4)
    b = torch.where(l1 >= 5, torch.full_like(b, 5), b)
    b = torch.where(l1 >= 7, torch.full_like(b, 6), b)
    b = torch.where(l1 >= 11, torch.full_like(b, 7), b)
    both = is_board[:, :, None] & is_board[:, None, :]
    return torch.where(both, b, torch.full_like(b, SPATIAL_NUM - 1)).to(torch.int32)


def apply_prefix_crop(pos, role, prob, min_keep, max_keep, gen):
    """以機率 prob 把每局截成只剩前 U[min_keep, max_keep] 手，其餘改為 pad；回傳新的 (pos, role)。"""
    bsz, seq_len = pos.shape
    hit = torch.rand(bsz, generator=gen, device=pos.device) < prob
    keep = torch.randint(min_keep, max_keep + 1, (bsz,), generator=gen, device=pos.device)
    keep = torch.where(hit, keep, torch.full_like(keep, seq_len))
    cut = torch.arange(seq_len, device=pos.device).unsqueeze(0) >= keep.unsqueeze(1)
    return pos.masked_fill(cut, PAD_POS), role.masked_fill(cut, ROLE_PAD)


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

    def forward(self, x, key_valid_mask, attn_bias=None):
        """x: (B, L, D)，key_valid_mask: (B, L) bool，True 表示可被注意。

        attn_bias 為 (B, H, L, L) 浮點 bias，已含 pad 遮罩；給定時取代 key_valid_mask。
        """
        bsz, seq_len, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(bsz, seq_len, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # bool attn_mask：True 表示允許注意，形狀 (B, 1, 1, L)
        attn_mask = key_valid_mask[:, None, None, :] if attn_bias is None else attn_bias.to(q.dtype)
        drop_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.drop1(self.out_proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x


class ConvModule(nn.Module):
    """Conformer 卷積模組：依序為 pointwise GLU、depthwise conv、LayerNorm、SiLU、pointwise，外加殘差。"""

    def __init__(self, d_model, kernel_size, dropout=0.1):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('conv kernel_size 必須為奇數，收到 %d' % kernel_size)
        self.norm = nn.LayerNorm(d_model)
        self.pw_in = nn.Linear(d_model, d_model * 2)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.dw_norm = nn.LayerNorm(d_model)
        self.pw_out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid_mask):
        """x: (B, L, D)，valid_mask: (B, L) bool；pad 位置在卷積前歸零。"""
        y = F.glu(self.pw_in(self.norm(x)), dim=-1)
        y = y * valid_mask.unsqueeze(-1).to(y.dtype)
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        y = self.pw_out(F.silu(self.dw_norm(y)))
        return x + self.drop(y)


class PatchEncoder(nn.Module):
    """每手共用的局部盤面 CNN：以無 padding 的 3x3 valid conv 收斂至中心 1x1 保留幾何位置。"""

    def __init__(self, patch_size, out_dim=PATCH_EMB_DIM):
        super().__init__()
        self.patch_size = int(patch_size)
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError('patch_size 必須為 >= 3 的奇數，收到 %d' % self.patch_size)
        self.out_dim = int(out_dim)
        self.n_cells = self.patch_size * self.patch_size
        n_layers = (self.patch_size - 1) // 2
        # 逐層 valid conv 縮至 1x1，保留相對落子點的位置
        if n_layers == 1:
            ch_pairs = [(4, 64)]
        else:
            ch_pairs = [(4, 32), (32, 64)] + [(64, 64)] * (n_layers - 2)
        layers = []
        for cin, cout in ch_pairs:
            layers.append(nn.Conv2d(cin, cout, 3, padding=0))
            layers.append(nn.GELU())
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(64, self.out_dim)
        self.register_buffer('shifts', torch.tensor([6, 4, 2, 0], dtype=torch.long), persistent=False)

    def forward(self, patch, valid_mask=None):
        """patch: (B, L, patch_nbytes) 整數張量，valid_mask 為 (B, L) bool，回傳 (B, L, out_dim)。"""
        bsz, seq_len, nb = patch.shape
        if valid_mask is not None:
            flat_mask = valid_mask.reshape(-1)
            if not flat_mask.any():
                return patch.new_zeros((bsz, seq_len, self.out_dim), dtype=self.proj.weight.dtype)
            p_valid = patch.reshape(-1, nb)[flat_mask]
        else:
            p_valid = patch.reshape(-1, nb)

        cells = ((p_valid.long().unsqueeze(-1) >> self.shifts) & 3).reshape(-1, nb * 4)[:, :self.n_cells]
        x = F.one_hot(cells, 4).float()
        x = x.transpose(1, 2).reshape(-1, 4, self.patch_size, self.patch_size)
        y = self.proj(self.conv(x).flatten(1))
        if valid_mask is not None:
            out = y.new_zeros((bsz * seq_len, self.out_dim))
            out[flat_mask] = y
            return out.view(bsz, seq_len, self.out_dim)
        return y.view(bsz, seq_len, self.out_dim)


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
        state_feats=False,
        conv_kernel=0,
        patch_size=0,
        spatial_bias=False,
        stage_pool=False,
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
        self.state_feats = bool(state_feats)
        self.conv_kernel = int(conv_kernel)
        self.patch_size = int(patch_size)
        self.spatial_bias = bool(spatial_bias)
        self.stage_pool = bool(stage_pool)

        self.coord_emb = nn.Embedding(NUM_POS, 128)
        self.role_emb = nn.Embedding(NUM_ROLES, 32)
        self.line_emb = nn.Embedding(LINE_NUM, 16)
        self.rel_dr_emb = nn.Embedding(REL_NUM, 24)
        self.rel_dc_emb = nn.Embedding(REL_NUM, 24)
        self.rel_dist_emb = nn.Embedding(DIST_NUM, 16)
        self.self_dist_emb = nn.Embedding(DIST_NUM, 16)
        self.step_emb = nn.Embedding(self.max_len, 16)

        in_dim = 128 + 32 + 16 + 24 + 24 + 16 + 16 + 16  # 272
        if self.state_feats:
            # 氣數、提子、叫吃、打劫、3x3 棋形，各欄最後一格為 pass/pad
            self.libs_emb = nn.Embedding(6, 16)
            self.cap_emb = nn.Embedding(5, 8)
            self.atari_emb = nn.Embedding(3, 4)
            self.ko_emb = nn.Embedding(3, 4)
            self.pat3_emb = nn.Embedding(PAT3_NUM, 32)
            in_dim += 16 + 8 + 4 + 4 + 32  # 336
        self.patch_enc = None
        if self.patch_size > 0:
            self.patch_enc = PatchEncoder(self.patch_size)
            in_dim += PATCH_EMB_DIM
        self.in_proj = nn.Linear(in_dim, self.d_model)
        self.in_norm = nn.LayerNorm(self.d_model)
        self.in_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PreLNTransformerBlock(self.d_model, self.nhead, self.dim_feedforward, dropout=dropout)
            for _ in range(self.num_layers)
        ])
        self.convs = None
        if self.conv_kernel > 0:
            self.convs = nn.ModuleList([
                ConvModule(self.d_model, self.conv_kernel, dropout=dropout) for _ in range(self.num_layers)
            ])
        self.spatial_table = None
        if self.spatial_bias:
            # 各層共用，初始為 0，訓練起點與無 bias 相同
            self.spatial_table = nn.Parameter(torch.zeros(SPATIAL_NUM, self.nhead))
        self.final_norm = nn.LayerNorm(self.d_model)

        self.pool = AttentiveStatsPool(self.d_model)
        feat_dim = self.d_model * 2
        self.stage_pools = None
        if self.stage_pool:
            # 全局池化外另接各分段池化，短局缺的分段輸出 0
            self.stage_pools = nn.ModuleList([
                AttentiveStatsPool(self.d_model) for _ in range(len(STAGE_BOUNDS) + 1)
            ])
            feat_dim += self.d_model * 2 * len(self.stage_pools)
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

    def forward(self, pos, is_target, state=None, patch=None):
        """輸入 pos (B, L) 與 is_target (B, L)，回傳 (B, embed_dim) L2 正規化向量。

        state_feats 模型另需 state (B, L, 5)，欄位順序同 STATE_FIELDS；
        patch_size 模型另需 patch (B, L, patch_nbytes)。
        """
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
        if self.state_feats:
            if state is None:
                raise ValueError('state_feats 模型需要 state 輸入')
            st = state.long()
            tok = torch.cat([
                tok,
                self.libs_emb(st[..., 0]),
                self.cap_emb(st[..., 1]),
                self.atari_emb(st[..., 2]),
                self.ko_emb(st[..., 3]),
                self.pat3_emb(st[..., 4]),
            ], dim=-1)
        if self.patch_enc is not None:
            if patch is None:
                raise ValueError('patch_size 模型需要 patch 輸入')
            tok = torch.cat([tok, self.patch_enc(patch, valid_mask)], dim=-1)
        h = self.in_drop(self.in_norm(self.in_proj(tok)))

        attn_bias = None
        if self.spatial_table is not None:
            # 轉成 attention 實際的計算 dtype 並連續化一次，各層共用同一份張量
            dev = h.device.type
            attn_dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else h.dtype
            table = self.spatial_table.to(attn_dtype)
            attn_bias = F.embedding(spatial_pair_buckets(pos_l), table).permute(0, 3, 1, 2)
            attn_bias = attn_bias.masked_fill(~attn_valid[:, None, None, :], float('-inf')).contiguous()
        for i, blk in enumerate(self.blocks):
            h = blk(h, attn_valid, attn_bias)
            if self.convs is not None:
                h = self.convs[i](h, valid_mask)
        h = self.final_norm(h)

        if self.pool_mode == 'target':
            target_mask = valid_mask & (role_l == ROLE_TARGET)
            has_target = target_mask.any(dim=1, keepdim=True)
            pool_mask = torch.where(has_target, target_mask, valid_mask)
        else:
            pool_mask = valid_mask

        pooled = self.pool(h, pool_mask)
        if self.stage_pools is not None:
            step = torch.arange(h.shape[1], device=h.device)
            edges = (0,) + STAGE_BOUNDS + (h.shape[1],)
            parts = [pooled]
            for j, sp in enumerate(self.stage_pools):
                in_stage = (step >= edges[j]) & (step < edges[j + 1])
                parts.append(sp(h, pool_mask & in_stage.unsqueeze(0)))
            pooled = torch.cat(parts, dim=1)
        pooled = self.head_drop(self.bn_pool(pooled))
        emb = self.bn_emb(self.fc_emb(pooled))
        emb = F.normalize(emb.float(), p=2, dim=1)
        if not has_valid.all():
            emb = emb * has_valid.unsqueeze(1).to(emb.dtype)
        return emb


def load_seq_npz(path, device, load_state=True):
    """載入單檔落子序列 .npz，將 pos（int16）與 is_target（uint8）移至指定裝置。

    load_state 為真且檔內有狀態欄位時，狀態放在 meta['state']。
    """
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
        # 有規則引擎狀態時合成 (N, L, 5) int16 放在 meta['state']
        if load_state and all(f in z.files for f in STATE_FIELDS):
            st = np.empty(z['pos'].shape + (len(STATE_FIELDS),), dtype=np.int16)
            for j, f in enumerate(STATE_FIELDS):
                st[..., j] = z[f]
            meta['state'] = torch.from_numpy(st).to(device)
    return pos, is_target, meta


def forward_seq_embeddings(model, pos, is_target, amp, tta=1, d4_table=None, state=None, patch=None):
    """計算一個 batch 的序列風格向量，tta 支援 1 至 8 種 D4 對稱平均；狀態欄位 D4 不變，各視角共用。

    局部盤面不隨 D4 變換，patch 模型只接受 tta=1。
    """
    n_views = max(1, min(int(tta), 8))
    if n_views > 1 and patch is not None:
        raise ValueError('局部盤面模型不支援 D4 TTA')
    if n_views > 1 and d4_table is None:
        d4_table = torch.from_numpy(D4_POS_TABLE.astype(np.int64)).to(pos.device)
    total = None
    pos_l = pos.long()
    role_l = is_target.long()
    for k in range(n_views):
        pk = apply_d4_pos(pos_l, k, d4_table) if k else pos_l
        with amp_autocast(pos.device.type, amp):
            e = model(pk, role_l, state, patch)
        e_f = e.float()
        total = e_f if total is None else (total + e_f)
    return F.normalize(total / n_views, p=2, dim=1)


def recompute_seq_bn_stats(model, pos, is_target, batch_size, device, amp, n_batches=100, seed=42, state=None,
                           patch=None):
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
            sb = state[sl].to(device, non_blocking=True) if state is not None else None
            xb = patch[sl].to(device, non_blocking=True) if patch is not None else None
            with amp_autocast(device.type, amp):
                bn(pb, rb, sb, xb)

    for m, mom in zip(norms, orig_mom):
        m.momentum = mom
    bn.eval()
    return bn


def extract_seq_embeddings(model, pos, is_target, indices, batch_size, device, amp, tta=1, state=None, patch=None):
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
                sb = state[sl].to(device, non_blocking=True) if state is not None else None
                xb = patch[sl].to(device, non_blocking=True) if patch is not None else None
                emb = forward_seq_embeddings(model, pb, rb, amp, tta=tta, d4_table=d4_table, state=sb, patch=xb)
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
    ap.add_argument('--state-feats', action='store_true', help='token 加入規則引擎狀態欄位（需 --state-feats 打包的序列）')
    ap.add_argument('--conv-kernel', type=int, default=0, help='每層後接 Conformer 卷積模組的 kernel，0 為不加')
    ap.add_argument('--patch-dir', default='', help='pack_move_patch 產出的局部盤面目錄，空字串為不用')
    ap.add_argument('--spatial-bias', action='store_true', help='attention 加兩手落子點距離桶的 per-head bias')
    ap.add_argument('--stage-pool', action='store_true', help='全局池化外另接開局、中盤、官子分段池化')
    ap.add_argument('--prefix-crop-prob', type=float, default=0.0, help='訓練時每局截成只剩前段手數的機率，0 為不截')
    ap.add_argument('--prefix-crop-range', type=int, nargs=2, default=[80, 150], metavar=('MIN', 'MAX'),
                    help='截斷後保留的手數範圍')
    ap.add_argument('--color-balance', action='store_true', help='同一玩家的 K 局盡量黑白各半')
    ap.add_argument('--fp16', action='store_true', help='GPU 用 fp16 autocast 與 GradScaler（給不支援 bf16 的 T4）')
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
    if args.conv_kernel < 0 or (args.conv_kernel and args.conv_kernel % 2 == 0):
        ap.error('--conv-kernel 要是 0 或正奇數')
    if not 0.0 <= args.prefix_crop_prob <= 1.0:
        ap.error('--prefix-crop-prob 要在 0 到 1 之間')
    if not 1 <= args.prefix_crop_range[0] <= args.prefix_crop_range[1]:
        ap.error('--prefix-crop-range 要 1 <= MIN <= MAX')
    if args.color_balance and args.k_games < 2:
        ap.error('--color-balance 需要 --k-games >= 2')
    if args.patch_dir and args.augment_d4:
        ap.error('局部盤面不隨 D4 變換，--patch-dir 不能與 --augment-d4 同時使用')
    return args


def _npz_shape(path, name):
    """不解壓，從 .npz 內陣列的 header 讀出形狀。"""
    with zipfile.ZipFile(path) as zf, zf.open(name + '.npy') as f:
        major, _ = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        return read(f)[0]


def _seq_files_bytes(seq_dir, state_feats=False, patch_dir=''):
    """估算 train 與 newdev 載入後 pos、is_target（與狀態欄位、局部盤面）的記憶體占用（byte）。"""
    per_move = 2 + 1 + (2 * len(STATE_FIELDS) if state_feats else 0)  # pos int16、role uint8、狀態 int16
    total = 0
    for name in ('train', 'newdev'):
        p = os.path.join(seq_dir, name + '.npz')
        if os.path.exists(p):
            n, length = _npz_shape(p, 'pos')
            total += n * length * per_move
        pp = os.path.join(patch_dir, name + '_patch.npz') if patch_dir else ''
        if pp and os.path.exists(pp):
            total += int(np.prod(_npz_shape(pp, 'patch')))
    return total


def main(argv=None):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = use_amp(device, args.no_amp)
    if args.fp16 and device.type == 'cuda' and not args.no_amp:
        amp = 'fp16'
    scaler = torch.amp.GradScaler('cuda', enabled=(amp == 'fp16'))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_device = pick_data_device(args, device, _seq_files_bytes(args.seq_dir, args.state_feats, args.patch_dir))
    print('載入序列資料到 %s' % data_device, flush=True)
    dev_pos, dev_role, dev_meta = load_seq_npz(os.path.join(args.seq_dir, 'newdev.npz'), data_device, args.state_feats)
    tr_pos, tr_role, tr_meta = load_seq_npz(os.path.join(args.seq_dir, 'train.npz'), data_device, args.state_feats)
    tr_state = dev_state = None
    if args.state_feats:
        if 'state' not in tr_meta or 'state' not in dev_meta:
            raise SystemExit('%s 沒有狀態欄位，請用 pack_move_seq --state-feats 打包' % args.seq_dir)
        tr_state, dev_state = tr_meta.pop('state'), dev_meta.pop('state')
    tr_patch = dev_patch = None
    patch_size = 0
    if args.patch_dir:
        tr_np, patch_size = load_patch_npz(os.path.join(args.patch_dir, 'train_patch.npz'), tr_meta['key'])
        dev_np, dev_ps = load_patch_npz(os.path.join(args.patch_dir, 'newdev_patch.npz'), dev_meta['key'])
        if dev_ps != patch_size:
            raise SystemExit('train 與 newdev 局部盤面大小不同：%d vs %d' % (patch_size, dev_ps))
        tr_patch = torch.from_numpy(tr_np).to(data_device)
        dev_patch = torch.from_numpy(dev_np).to(data_device)
        del tr_np, dev_np

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
        state_feats=args.state_feats,
        conv_kernel=args.conv_kernel,
        patch_size=patch_size,
        spatial_bias=args.spatial_bias,
        stage_pool=args.stage_pool,
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
    if args.state_feats:
        config['state_feats'] = True
    if args.conv_kernel:
        config['conv_kernel'] = args.conv_kernel
    if patch_size:
        config['patch_size'] = patch_size
    if args.spatial_bias:
        config['spatial_bias'] = True
    if args.stage_pool:
        config['stage_pool'] = True
    if args.prefix_crop_prob > 0:
        config['prefix_crop'] = [args.prefix_crop_prob] + list(args.prefix_crop_range)
    if args.color_balance:
        config['color_balance'] = True

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
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
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
        'state_feats': bool(args.state_feats),
        'conv_kernel': int(args.conv_kernel),
        'patch_size': int(patch_size),
        'spatial_bias': bool(args.spatial_bias),
        'stage_pool': bool(args.stage_pool),
    }
    with open(os.path.join(args.ckpt_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump(infer_cfg, f, indent=2)
    if args.state_feats:
        # test 模式推論從 checkpoint 目錄讀 pat3 詞表
        shutil.copyfile(os.path.join(args.seq_dir, PAT3_VOCAB_FILE), os.path.join(args.ckpt_dir, PAT3_VOCAB_FILE))

    n_params = sum(p.numel() for p in model.parameters())
    print(
        'device %s | data %s | amp %s | augment_d4 %s | pool %s | state %s | conv %d | patch %d | spatial %s | stage %s | crop %s | color_bal %s | train 玩家 %d (%s 局, %d steps/ep) | dev %s 局 | 參數量 %s'
        % (
            device, data_device, amp, args.augment_d4, args.pool_mode, args.state_feats, args.conv_kernel, patch_size,
            args.spatial_bias, args.stage_pool,
            (args.prefix_crop_prob, *args.prefix_crop_range) if args.prefix_crop_prob > 0 else 'off', args.color_balance,
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
            game_color=np.asarray(tr_meta['color']) if args.color_balance else None,
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
            sb = tr_state[idx_t].to(device, non_blocking=True) if tr_state is not None else None
            xb = tr_patch[idx_t].to(device, non_blocking=True) if tr_patch is not None else None
            if args.prefix_crop_prob > 0:
                pb, rb = apply_prefix_crop(pb, rb, args.prefix_crop_prob, args.prefix_crop_range[0],
                                           args.prefix_crop_range[1], aug_gen)
            if args.augment_d4:
                ks = torch.randint(0, 8, (pb.size(0),), generator=aug_gen, device=device)
                pb = apply_d4_pos(pb, ks, d4_table)
            yb = torch.as_tensor(y_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device.type, amp):
                emb = model(pb, rb, sb, xb)
            l_arc, corr = arcface(emb, yb)
            l_sup = (
                supcon_loss(emb, yb, temperature=args.supcon_temp)
                if args.supcon_weight > 0 else l_arc.new_tensor(0.0)
            )
            loss = l_arc + args.supcon_weight * l_sup
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface.parameters()),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

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
            state=dev_state, patch=dev_patch,
        )
        raw_dev_metrics = evaluate_b_group_embeddings(raw_dev_emb, dev_pids, seeds=args.eval_seeds)

        eval_model = model
        dev_metrics = raw_dev_metrics
        if args.bn_batches > 0:
            eval_model = recompute_seq_bn_stats(
                model, tr_pos, tr_role, args.val_batch_size, device, amp,
                n_batches=args.bn_batches, seed=args.seed + epoch, state=tr_state, patch=tr_patch,
            )
            bn_dev_emb = extract_seq_embeddings(
                eval_model, dev_pos, dev_role, dev_indices, args.val_batch_size, device, amp, tta=args.val_tta,
                state=dev_state, patch=dev_patch,
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
        if device.type == 'cuda':
            print('  峰值 VRAM %.2f GB' % (torch.cuda.max_memory_allocated(device) / 1024 ** 3), flush=True)

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
        if amp == 'fp16':
            state['scaler'] = scaler.state_dict()
        save_atomic(state, last_path)


if __name__ == '__main__':
    main()
