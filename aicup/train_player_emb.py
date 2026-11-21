"""深度棋風 Embedding 訓練：以 500-ch 整局特徵訓練同段位玩家辨識表徵。

複用 500-ch bit-packed 棋譜特徵，於 GPU 即時解包並依目標玩家執色對調黑白平面，
主幹可由預訓練 GoRankResNet 初始化，接 Avg+Max 雙池化與 BNNeck 輸出單位球面向量。
訓練目標結合 Sub-Center ArcFace 與同段位批次強化之 Within-Rank PK SupCon，
每個 epoch 結束於獨立 dev 玩家集合執行 B 組模擬並追蹤 Top-5 指標。

用法（在 repo 根目錄）：
    python -m aicup.train_player_emb --feat-dir rank-features-v180000 \\
        --init rank-trained-models/59844b9/best_model.pth \\
        --ckpt-dir player-trained-models/emb-smoke --epochs 2 --steps-per-epoch 1500
"""
import argparse
import copy
import csv
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from aicup import color, packed, symmetry
from aicup.fuse_player import apply_csls_column_penalty
from aicup.train_rank import (
    load_colors,
    load_to,
    log_run,
    lr_at,
    pick_data_device,
    save_atomic,
    split_bytes,
    use_amp,
)
from network import ResidualBlock

DEFAULT_DEV_PLAYERS = os.path.join('humansl-analysis', 'val-players-newdev-v180000.csv')
TOP_N = 5


class GoStyleEmbedNet(nn.Module):
    """整局 500-ch 2D ResNet 深度風格編碼器，輸出 L2 正規化單位向量。"""

    def __init__(
        self,
        in_channels=500,
        num_channels=256,
        num_blocks=20,
        embed_dim=256,
        dropout=0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.dropout_p = dropout

        # 命名與 GoRankResNet 一致，可直接載入預訓練主幹權重
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        self.layers = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        feat_dim = num_channels * 2
        self.bn_pool = nn.BatchNorm1d(feat_dim)
        self.drop = nn.Dropout(p=dropout)
        self.fc_emb = nn.Linear(feat_dim, embed_dim, bias=False)
        self.bn_emb = nn.BatchNorm1d(embed_dim)

    def backbone_parameters(self):
        """回傳主幹層參數，供分層學習率設定。"""
        for module in (self.conv_in, self.bn_in, self.layers):
            yield from module.parameters()

    def head_parameters(self):
        """回傳 Embedding Head 層參數，供分層學習率設定。"""
        for module in (self.bn_pool, self.fc_emb, self.bn_emb):
            yield from module.parameters()

    def forward(self, x):
        """輸入 (B, C, 19, 19)，回傳 (B, embed_dim) L2 正規化向量。"""
        h = F.relu(self.bn_in(self.conv_in(x)))
        h = self.layers(h)
        pooled = torch.cat([self.avg_pool(h), self.max_pool(h)], dim=1).flatten(1)
        pooled = self.drop(self.bn_pool(pooled))
        emb = self.bn_emb(self.fc_emb(pooled))
        return F.normalize(emb.float(), p=2, dim=1)


def load_rank_backbone(model, ckpt_path):
    """從 GoRankResNet checkpoint 載入相容主幹權重，回傳載入的鍵數。"""
    src = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(src, dict) and 'model' in src and isinstance(src['model'], dict):
        src = src['model']
    dst = model.state_dict()
    loaded = 0
    for k, v in src.items():
        if not (k.startswith('conv_in.') or k.startswith('bn_in.') or k.startswith('layers.')):
            continue
        if k not in dst:
            continue
        if dst[k].shape == v.shape:
            dst[k].copy_(v)
            loaded += 1
        elif k == 'conv_in.weight' and v.ndim == 4 and dst[k].shape[0] == v.shape[0] and dst[k].shape[1] > v.shape[1]:
            # 輸入含額外 color plane 時保留前 500 通道預訓練權重並將新增通道歸零
            dst[k].zero_()
            dst[k][:, :v.shape[1]].copy_(v)
            loaded += 1
    if loaded == 0:
        raise ValueError('在 %s 中找不到任何與主幹相容的權重' % ckpt_path)
    model.load_state_dict(dst)
    return loaded


class SubCenterArcFace(nn.Module):
    """Sub-Center ArcFace 分類損失，每位玩家配置多個子中心以容納不同棋風。"""

    def __init__(self, embed_dim, num_classes, num_sub_centers=2, scale=30.0, margin=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_sub_centers = num_sub_centers
        self.scale = float(scale)
        self.margin = float(margin)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

        self.weight = nn.Parameter(torch.empty(num_classes * num_sub_centers, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        """計算 Sub-Center ArcFace 交叉熵損失與預測正確數。"""
        emb_f = F.normalize(emb.float(), p=2, dim=1)
        w_f = F.normalize(self.weight.float(), p=2, dim=1)
        cos_all = F.linear(emb_f, w_f).view(-1, self.num_classes, self.num_sub_centers)
        cos = cos_all.max(dim=2).values.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        sin = torch.sqrt((1.0 - cos * cos).clamp_min(1e-12))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)

        logits = cos.clone()
        row_idx = torch.arange(emb_f.size(0), device=emb_f.device)
        logits[row_idx, labels] = phi[row_idx, labels]
        logits = logits * self.scale
        loss = F.cross_entropy(logits, labels)
        correct = int((cos.argmax(dim=1) == labels).sum().item())
        return loss, correct


def supcon_loss(emb, labels, temperature=0.07):
    """批次內監督式對比損失，搭配同段位 PK 採樣強化同級區分力。"""
    emb_f = F.normalize(emb.float(), p=2, dim=1)
    bsz = emb_f.size(0)
    if bsz <= 1:
        return emb_f.new_tensor(0.0)

    sim = torch.matmul(emb_f, emb_f.t()) / float(temperature)
    self_mask = torch.eye(bsz, device=emb_f.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, float('-inf'))

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not valid.any():
        return emb_f.new_tensor(0.0)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # 僅累加正樣本位置，避免負樣本對應到 -inf 產生 NaN
    pos_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    return -(pos_log_prob[valid] / pos_counts[valid].to(emb_f.dtype)).mean()


def apply_tail_crop(x, max_crop, gen):
    """隨機截斷尾盤 0 至 max_crop 手，以空盤補齊前端維持 500 通道。"""
    if max_crop <= 0:
        return x
    crop = int(torch.randint(0, max_crop + 1, (1,), generator=gen).item())
    if crop <= 0:
        return x
    shift = crop * 2
    n_board_ch = 500
    if shift >= n_board_ch:
        return x
    out = x.clone()
    out[:, :shift] = 0
    out[:, shift:n_board_ch] = x[:, :n_board_ch - shift]
    return out


def read_player_ids_csv(path):
    """讀取含 player_id 欄位的 CSV，回傳玩家集合。"""
    pids = set()
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pid = str(row['player_id']).strip()
            if pid:
                pids.add(pid)
    return pids


def build_player_index(meta, min_games=5, exclude_pids=None):
    """建立訓練集玩家索引，排除指定玩家與局數不足者。

    以 (rank_idx, player_id) 識別獨立玩家，回傳：
        player_games  dict[class_id -> np.ndarray of game indices]
        rank_players  dict[rank_idx -> np.ndarray of class_ids]
        class_keys    list of (rank_idx, player_id)
    """
    exclude = set(exclude_pids) if exclude_pids else set()
    groups = {}
    ranks = meta['rank_idx']
    pids = meta['player_id']
    for idx, (rk, pid) in enumerate(zip(ranks.tolist(), pids.tolist())):
        pid_s = str(pid)
        if pid_s in exclude:
            continue
        groups.setdefault((int(rk), pid_s), []).append(idx)

    class_keys = sorted(k for k, idxs in groups.items() if len(idxs) >= min_games)
    if not class_keys:
        raise ValueError('過濾後沒有任何局數 >= %d 的訓練玩家' % min_games)

    player_games = {}
    rank_players_list = {}
    for cid, key in enumerate(class_keys):
        player_games[cid] = np.asarray(groups[key], dtype=np.int64)
        rank_players_list.setdefault(key[0], []).append(cid)
    rank_players = {rk: np.asarray(cids, dtype=np.int64) for rk, cids in rank_players_list.items()}
    return player_games, rank_players, class_keys


def sample_within_rank_pk_batches(
    player_games,
    rank_players,
    n_batches,
    p_players,
    k_games,
    rng,
    adjacent_prob=0.5,
):
    """產生同段位或相鄰段位集中之 PK 採樣批次。

    每個批次回傳 (game_indices, class_labels)，長度皆為 p_players * k_games。
    """
    avail_ranks = sorted(rank_players.keys())
    rank_weights = np.array([len(rank_players[r]) for r in avail_ranks], dtype=np.float64)
    rank_weights /= rank_weights.sum()
    all_cids = np.concatenate([rank_players[r] for r in avail_ranks])
    eff_p = min(p_players, len(all_cids))

    batches_out = []
    for _ in range(n_batches):
        center_r = avail_ranks[int(rng.choice(len(avail_ranks), p=rank_weights))]
        use_adj = (len(avail_ranks) > 1) and (rng.random() < adjacent_prob)
        if use_adj:
            # 選取與 center_r 最接近的鄰近段位組成雙段位池
            neighbors = [r for r in avail_ranks if r != center_r]
            neighbors.sort(key=lambda r: (abs(r - center_r), rng.random()))
            chosen_ranks = [center_r, neighbors[0]]
        else:
            chosen_ranks = [center_r]

        pool = np.concatenate([rank_players[r] for r in chosen_ranks])
        if len(pool) < eff_p:
            # 單段位人數不足 eff_p 時由近至遠擴充鄰近段位
            rest = sorted((r for r in avail_ranks if r not in chosen_ranks), key=lambda r: abs(r - center_r))
            for r in rest:
                pool = np.concatenate([pool, rank_players[r]])
                if len(pool) >= eff_p:
                    break

        picked_cids = rng.choice(pool, size=eff_p, replace=False)
        game_idx_list = []
        label_list = []
        for cid in picked_cids.tolist():
            g_arr = player_games[cid]
            replace = len(g_arr) < k_games
            chosen_g = rng.choice(g_arr, size=k_games, replace=replace)
            game_idx_list.append(chosen_g)
            label_list.append(np.full(k_games, cid, dtype=np.int64))
        batches_out.append((np.concatenate(game_idx_list), np.concatenate(label_list)))
    return batches_out


def forward_embeddings(model, x, amp, tta=1):
    """計算一個 batch 的單位球面風格向量，tta 支援 1 至 8 種對稱平均。"""
    n_views = max(1, min(int(tta), symmetry.N_SYMMETRIES))
    total = None
    for k in range(n_views):
        xk = symmetry.apply(x, k).contiguous() if k else x
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            e = model(xk)
        e_f = e.float()
        total = e_f if total is None else (total + e_f)
    return F.normalize(total / n_views, p=2, dim=1)


def recompute_bn_stats(model, x, batch_size, device, amp, c=None, color_mode='none', n_batches=100, seed=42):
    """回傳 model 的副本，以混合段位隨機 batch 重算所有 BatchNorm 的累積統計量。"""
    bn = copy.deepcopy(model)
    norms = [m for m in bn.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    orig_mom = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # None 為累積平均，每個 batch 權重相同

    n_total = min(len(x), n_batches * batch_size)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    order = torch.randperm(len(x), generator=gen, device=x.device)[:n_total]
    bn.train()
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            sl = order[i:i + batch_size]
            if len(sl) < 2:
                continue  # BatchNorm1d 在 batch 為 1 時無法計算變異數
            xb = packed.unpack_torch(x[sl].to(device, non_blocking=True))
            if c is not None and color_mode != 'none':
                xb = color.apply_color(xb, c[sl].to(device, non_blocking=True), color_mode)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                bn(xb)

    for m, mom in zip(norms, orig_mom):
        m.momentum = mom
    bn.eval()
    return bn


def extract_split_embeddings(model, x, indices, batch_size, device, amp, c=None, color_mode='none', tta=1):
    """對指定局索引批次抽取 L2 正規化風格向量，回傳 (len(indices), embed_dim) float32。"""
    was_training = model.training
    model.eval()
    idx_t = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(idx_t), batch_size):
                sl = idx_t[i:i + batch_size]
                xb = packed.unpack_torch(x[sl].to(device, non_blocking=True))
                if c is not None and color_mode != 'none':
                    xb = color.apply_color(xb, c[sl].to(device, non_blocking=True), color_mode)
                emb = forward_embeddings(model, xb, amp, tta=tta)
                out.append(emb.cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(out, axis=0).astype(np.float32) if out else np.zeros((0, model.embed_dim), dtype=np.float32)


def _l2_normalize_rows(mat, eps=1e-12):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def evaluate_b_group_embeddings(
    embeddings,
    player_ids,
    seeds=(42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    n_cands=300,
    n_queries=200,
    cand_games=10,
    q_size_choices=(1, 2, 3, 5),
    csls_k=10,
):
    """以純風格向量執行 B 組模擬評估，含候選池中心化與可選 CSLS。

    回傳含 dev_top5、dev_top5_csls 與各查詢局數細項的字典。
    """
    by_player = {}
    for idx, pid in enumerate(player_ids):
        by_player.setdefault(str(pid), []).append(idx)

    max_q = max(q_size_choices)
    min_games = cand_games + max_q
    valid_pids = sorted(pid for pid, idxs in by_player.items() if len(idxs) >= min_games)
    if len(valid_pids) < 2:
        return {
            'dev_top5': 0.0,
            'dev_top5_csls': 0.0,
            'dev_csls_q1': 0.0,
            'dev_csls_q2': 0.0,
            'dev_csls_q3': 0.0,
            'dev_csls_q5': 0.0,
            'dev_players': len(valid_pids),
        }

    eff_cands = min(n_cands, len(valid_pids))
    eff_queries = min(n_queries, eff_cands)
    raw_scores_all = []
    csls_scores_all = []
    sizes_all = []
    emb_f64 = np.asarray(embeddings, dtype=np.float64)

    for s in seeds:
        rng = np.random.default_rng(s)
        chosen = sorted(rng.choice(valid_pids, size=eff_cands, replace=False).tolist())
        pid_to_col = {p: i for i, p in enumerate(chosen)}

        cand_vecs = np.zeros((eff_cands, emb_f64.shape[1]), dtype=np.float64)
        query_pools = {}
        for ci, pid in enumerate(chosen):
            g_arr = np.asarray(by_player[pid], dtype=np.int64)
            perm = rng.permutation(g_arr)
            cand_vecs[ci] = emb_f64[perm[:cand_games]].mean(axis=0)
            query_pools[pid] = perm[cand_games:]

        # 僅使用候選池計算平均向量以去共線性，不觸碰查詢局
        mu_cand = cand_vecs.mean(axis=0, keepdims=True)
        cand_norm = _l2_normalize_rows(cand_vecs - mu_cand)

        q_pids = rng.choice(chosen, size=eff_queries, replace=False).tolist()
        q_sizes = np.array(
            (list(q_size_choices) * (eff_queries // len(q_size_choices) + 1))[:eff_queries],
            dtype=np.int64,
        )
        rng.shuffle(q_sizes)

        q_vecs = np.zeros((eff_queries, emb_f64.shape[1]), dtype=np.float64)
        true_cols = np.zeros(eff_queries, dtype=np.int64)
        for qi, (pid, qn) in enumerate(zip(q_pids, q_sizes.tolist())):
            q_vecs[qi] = emb_f64[query_pools[pid][:qn]].mean(axis=0)
            true_cols[qi] = pid_to_col[pid]

        q_norm = _l2_normalize_rows(q_vecs - mu_cand)
        sim_mat = q_norm @ cand_norm.T

        def _score_matrix(mat):
            sc = np.zeros(eff_queries, dtype=np.float64)
            k_top = min(TOP_N, eff_cands)
            for qi in range(eff_queries):
                order = np.argsort(mat[qi])[::-1][:k_top]
                hits = np.where(order == true_cols[qi])[0]
                if len(hits):
                    sc[qi] = math.exp(-int(hits[0]))
            return sc

        raw_scores_all.append(_score_matrix(sim_mat))
        if csls_k > 0:
            adj_mat = apply_csls_column_penalty(sim_mat, csls_k)
            csls_scores_all.append(_score_matrix(adj_mat))
        else:
            csls_scores_all.append(raw_scores_all[-1])
        sizes_all.append(q_sizes)

    raw_cat = np.concatenate(raw_scores_all)
    csls_cat = np.concatenate(csls_scores_all)
    sz_cat = np.concatenate(sizes_all)
    out = {
        'dev_top5': float(raw_cat.mean()),
        'dev_top5_csls': float(csls_cat.mean()),
        'dev_players': len(valid_pids),
    }
    for q in (1, 2, 3, 5):
        mask = sz_cat == q
        out['dev_csls_q%d' % q] = float(csls_cat[mask].mean()) if mask.any() else 0.0
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--feat-dir', default='rank-features-v180000')
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--init', default='', help='預訓練 GoRankResNet 權重路徑，空字串為從零訓練')
    ap.add_argument('--dev-players', default=DEFAULT_DEV_PLAYERS,
                    help='val split 中用於早停評估的 dev 玩家 CSV，找不到且未顯式指定時使用全部 val')
    ap.add_argument('--exclude-players', nargs='*', default=[],
                    help='額外從 train 排除的玩家 CSV 清單')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--steps-per-epoch', type=int, default=1500,
                    help='每個 epoch 的 PK 批次數，0 為依訓練集總局數自動計算')
    ap.add_argument('--p-players', type=int, default=64, help='每個 batch 採樣的玩家數 P')
    ap.add_argument('--k-games', type=int, default=4, help='每位玩家採樣的局數 K')
    ap.add_argument('--min-games', type=int, default=5, help='納入訓練的最少玩家局數')
    ap.add_argument('--adjacent-prob', type=float, default=0.5, help='批次混合相鄰 2 個段位的機率')
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--embed-dim', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--sub-centers', type=int, default=2, help='Sub-Center ArcFace 每位玩家子中心數')
    ap.add_argument('--arc-scale', type=float, default=30.0)
    ap.add_argument('--arc-margin', type=float, default=0.25)
    ap.add_argument('--supcon-weight', type=float, default=0.3)
    ap.add_argument('--supcon-temp', type=float, default=0.07)
    ap.add_argument('--lr', type=float, default=1e-3, help='Head 與 ArcFace 權重學習率')
    ap.add_argument('--backbone-lr', type=float, default=3e-4, help='預訓練主幹學習率（未載入 --init 時同 --lr）')
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--warmup-steps', type=int, default=500)
    ap.add_argument('--grad-clip', type=float, default=5.0)
    ap.add_argument('--color', choices=color.MODES, default='swap')
    ap.add_argument('--raw-color', action='store_true', help='不讀 hd.npz，直接用 color 欄位')
    ap.add_argument('--augment', action='store_true', default=True, help='隨機套用 D4 盤面對稱')
    ap.add_argument('--no-augment', dest='augment', action='store_false', help='關閉 D4 盤面對稱')
    ap.add_argument('--tail-crop', type=int, default=0, help='隨機截斷尾盤最多幾手，0 為停用')
    ap.add_argument('--val-batch-size', type=int, default=256)
    ap.add_argument('--val-tta', type=int, default=1, help='dev 驗證的 TTA 視角數（1～8）')
    ap.add_argument('--bn-batches', type=int, default=100,
                    help='每個 epoch 評估前以混合段位 train batch 重算 BN 統計量的批次數，0 為不重算')
    ap.add_argument('--eval-seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--fresh', action='store_true', help='忽略既有 last.pt')
    ap.add_argument('--max-steps', type=int, default=0, help='每個 epoch 最多步數，0 為不限，用於冒煙測試')
    ap.add_argument('--log-every', type=int, default=200)
    args = ap.parse_args(argv)
    if args.p_players < 2 or args.k_games < 1 or args.sub_centers < 1:
        ap.error('--p-players 要 >= 2，--k-games 與 --sub-centers 要 >= 1')
    if args.min_games < 1 or args.supcon_weight < 0 or args.tail_crop < 0 or args.bn_batches < 0:
        ap.error('--min-games 要 >= 1，--supcon-weight、--tail-crop 與 --bn-batches 要 >= 0')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = use_amp(device, args.no_amp)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dir = os.path.join(args.feat_dir, 'train')
    val_dir = os.path.join(args.feat_dir, 'val')
    data_device = pick_data_device(args, device, split_bytes(train_dir) + split_bytes(val_dir))
    print('載入資料到 %s' % data_device, flush=True)
    x_val, _, val_meta = load_to(val_dir, data_device)
    c_val = load_colors(val_meta, args, data_device)
    x_train, _, train_meta = load_to(train_dir, data_device)
    c_train = load_colors(train_meta, args, data_device)

    # 讀取排除清單與 dev 玩家清單，缺檔直接中止以防誤用含留出玩家的整個 val
    exclude_pids = set()
    for p in args.exclude_players:
        exclude_pids |= read_player_ids_csv(p)
    if args.dev_players:
        if not os.path.exists(args.dev_players):
            raise SystemExit('找不到 --dev-players 檔案：%s' % args.dev_players)
        dev_pid_set = read_player_ids_csv(args.dev_players)
        exclude_pids |= dev_pid_set
        dev_indices = np.array([i for i, pid in enumerate(val_meta['player_id']) if str(pid) in dev_pid_set], dtype=np.int64)
        if len(dev_indices) == 0:
            raise SystemExit('在 val split 中找不到任何 %s 的玩家' % args.dev_players)
    else:
        dev_indices = np.arange(len(val_meta['player_id']), dtype=np.int64)
    dev_pids = val_meta['player_id'][dev_indices]

    player_games, rank_players, class_keys = build_player_index(
        train_meta, min_games=args.min_games, exclude_pids=exclude_pids
    )
    num_classes = len(class_keys)
    total_train_games = sum(len(g) for g in player_games.values())
    batch_size = args.p_players * args.k_games
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else max(1, total_train_games // batch_size)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)

    in_ch = color.in_channels(args.color)
    model = GoStyleEmbedNet(
        in_channels=in_ch,
        num_channels=args.channels,
        num_blocks=args.blocks,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)
    arcface = SubCenterArcFace(
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_sub_centers=args.sub_centers,
        scale=args.arc_scale,
        margin=args.arc_margin,
    ).to(device)

    if args.init:
        loaded = load_rank_backbone(model, args.init)
        print('已由 %s 載入主幹權重（%d 個張量）' % (args.init, loaded), flush=True)

    eff_backbone_lr = args.backbone_lr if args.init else args.lr
    optimizer = torch.optim.AdamW(
        [
            {'params': list(model.backbone_parameters()), 'lr': eff_backbone_lr, 'base_lr': eff_backbone_lr},
            {'params': list(model.head_parameters()) + list(arcface.parameters()), 'lr': args.lr, 'base_lr': args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, 'last.pt')
    best_path = os.path.join(args.ckpt_dir, 'best_model.pth')
    config = {
        'channels': args.channels,
        'blocks': args.blocks,
        'embed_dim': args.embed_dim,
        'dropout': args.dropout,
        'color': args.color,
        'raw_color': args.raw_color,
        'p_players': args.p_players,
        'k_games': args.k_games,
        'min_games': args.min_games,
        'sub_centers': args.sub_centers,
        'arc_scale': args.arc_scale,
        'arc_margin': args.arc_margin,
        'supcon_weight': args.supcon_weight,
        'supcon_temp': args.supcon_temp,
        'lr': args.lr,
        'backbone_lr': eff_backbone_lr,
        'epochs': args.epochs,
        'steps_per_epoch': steps_per_epoch,
        'augment': args.augment,
        'tail_crop': args.tail_crop,
        'seed': args.seed,
        'num_classes': num_classes,
        'amp': amp,
    }

    start_epoch, history, best_score = 0, [], -1.0
    if os.path.exists(last_path) and not args.fresh:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt['config'] != config:
            raise SystemExit('last.pt 的設定與這次不同：\n  舊 %s\n  新 %s\n'
                             '要用新設定請加 --fresh 或換 --ckpt-dir' % (ckpt['config'], config))
        model.load_state_dict(ckpt['model'])
        arcface.load_state_dict(ckpt['arcface'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, history, best_score = ckpt['epoch'], ckpt['history'], ckpt['best_score']
        print('從 epoch %d 接續（目前最佳 dev_top5_csls %.4f）' % (start_epoch, best_score), flush=True)

    infer_cfg = {
        'channels': args.channels,
        'blocks': args.blocks,
        'embed_dim': args.embed_dim,
        'dropout': args.dropout,
        'color': args.color,
        'raw_color': args.raw_color,
    }
    with open(os.path.join(args.ckpt_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump(infer_cfg, f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        'device %s | data %s | amp %s | train 玩家 %d (%s 局) | dev 局數 %s | 參數量 %s'
        % (
            device, data_device, amp, num_classes, format(total_train_games, ','),
            format(len(dev_indices), ','), format(n_params, ','),
        ),
        flush=True,
    )
    if start_epoch >= args.epochs:
        print('已完成 %d 個 epoch，要改設定重訓請換 --ckpt-dir 或加 --fresh' % start_epoch)
        return

    log_run(args.ckpt_dir, start_epoch, device)
    total_steps = steps_per_epoch * args.epochs

    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)
        pk_rng = np.random.default_rng(args.seed + epoch)
        aug_gen = torch.Generator().manual_seed(args.seed + epoch)
        crop_gen = torch.Generator().manual_seed(args.seed + 10000 + epoch)

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
            for group in optimizer.param_groups:
                group['lr'] = lr_at(global_step, total_steps, group['base_lr'], 'cosine', args.warmup_steps)

            idx_t = torch.as_tensor(g_idx, dtype=torch.long, device=x_train.device)
            xb = packed.unpack_torch(x_train[idx_t].to(device, non_blocking=True))
            if c_train is not None and args.color != 'none':
                xb = color.apply_color(xb, c_train[idx_t].to(device, non_blocking=True), args.color)
            if args.tail_crop > 0:
                xb = apply_tail_crop(xb, args.tail_crop, crop_gen)
            if args.augment:
                k = int(torch.randint(symmetry.N_SYMMETRIES, (1,), generator=aug_gen).item())
                if k:
                    xb = symmetry.apply(xb, k).contiguous()
            yb = torch.as_tensor(y_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                emb = model(xb)
            l_arc, corr = arcface(emb, yb)
            l_sup = supcon_loss(emb, yb, temperature=args.supcon_temp) if args.supcon_weight > 0 else l_arc.new_tensor(0.0)
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
                    '  ep%d step %d/%d  loss %.4f (arc %.4f sup %.4f)  cos_acc %.2f%%  %.1f 分'
                    % (
                        epoch + 1, step + 1, steps_per_epoch,
                        loss_sum / (step + 1), arc_sum / (step + 1), sup_sum / (step + 1),
                        100.0 * correct_sum / max(total_sum, 1),
                        (time.time() - t0) / 60,
                    ),
                    flush=True,
                )

        train_min = (time.time() - t0) / 60
        raw_dev_emb = extract_split_embeddings(
            model, x_val, dev_indices, args.val_batch_size, device, amp,
            c=c_val, color_mode=args.color, tta=args.val_tta,
        )
        raw_dev_metrics = evaluate_b_group_embeddings(raw_dev_emb, dev_pids, seeds=args.eval_seeds)

        eval_model = model
        dev_metrics = raw_dev_metrics
        if args.bn_batches > 0:
            eval_model = recompute_bn_stats(
                model, x_train, args.val_batch_size, device, amp,
                c=c_train, color_mode=args.color,
                n_batches=args.bn_batches, seed=args.seed + epoch,
            )
            bn_dev_emb = extract_split_embeddings(
                eval_model, x_val, dev_indices, args.val_batch_size, device, amp,
                c=c_val, color_mode=args.color, tta=args.val_tta,
            )
            dev_metrics = evaluate_b_group_embeddings(bn_dev_emb, dev_pids, seeds=args.eval_seeds)

        cur_lr = optimizer.param_groups[1]['lr']
        row = {
            'epoch': epoch + 1,
            'train_loss': loss_sum / max(steps_per_epoch, 1),
            'arc_loss': arc_sum / max(steps_per_epoch, 1),
            'supcon_loss': sup_sum / max(steps_per_epoch, 1),
            'train_cos_acc': 100.0 * correct_sum / max(total_sum, 1),
            'lr': cur_lr,
            'raw_dev_top5_csls': raw_dev_metrics['dev_top5_csls'],
            **dev_metrics,
            'train_minutes': train_min,
            'total_minutes': (time.time() - t0) / 60,
        }
        history.append(row)
        print(
            'Epoch %d/%d  loss %.4f  cos_acc %.2f%% | dev_top5 %.4f  csls %.4f (raw %.4f; q1 %.4f q2 %.4f q3 %.4f q5 %.4f) | %.1f 分'
            % (
                epoch + 1, args.epochs, row['train_loss'], row['train_cos_acc'],
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
            'model': model.state_dict(),
            'arcface': arcface.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'history': history,
            'best_score': best_score,
            'config': config,
        }
        save_atomic(state, last_path)

    print('完成。最佳 dev_top5_csls %.4f' % best_score, flush=True)


if __name__ == '__main__':
    main()
