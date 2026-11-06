"""逐手模型微調：humanv0 主幹加 10 份 policy head，第 k 份是 LEVELS[k] 的落子策略

每個局面的 loss 是 CE(真實等級那份 head 的 policy, 實際落子) + λ × CE(rank_logits, 等級)，
rank_logits[k] 是第 k 份 head 給實際落子的 log 機率。softmax 只在合法點上做，和推論相同。
metadata 固定為 METADATA_PROFILE。主幹最後一層 norm 固定用 humanv0 的 running 統計量，訓練與推論的 forward 相同。

資料是 extract_permove 的分片，整份載入 GPU（VRAM 不足時放 CPU），每個 batch 在 GPU 上解包並套用一種隨機對稱。
排程是 WSD：warmup、constant、最後 --decay-frac 的步數線性降到 0。
每 10% 的步數存一次 last.pt 並在 holdout 上算 loss，寫進 history.jsonl；重跑時從 last.pt 接續，RESUME_KEYS 的參數不同時拒絕執行。
--amp auto 在支援 bf16 硬體的 GPU 上用 bf16，其他情況用 float32。
訓練結束寫出 model.pt（Permove 的 state_dict 與 humanv0 的 config）。

用法（在 repo 根目錄）：
    python -m aicup.train_permove --data permove-features/spike --ckpt-dir rank-trained-models/permove/<commit hash>
    python -m aicup.train_permove --data permove-features/spike --max-steps 20 --batch 8 --device cpu --amp none --ckpt-dir ...
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np

from aicup import katago_torch as kt
from aicup import symmetry
from aicup.extract_permove import LEGAL_BYTES, SPATIAL_BITS, SPATIAL_BYTES
from aicup.scoring import LEVELS

METADATA_PROFILE = 'rank_1d'
N_LEVELS = len(LEVELS)
# 接續時必須相同的參數，不同時資料順序或 loss 會對不上；max_steps 可改，lr 排程跟著新的總步數
RESUME_KEYS = ('data', 'limit', 'batch', 'seed', 'epochs', 'lam', 'lr_trunk', 'lr_head', 'weight_decay',
               'warmup_frac', 'decay_frac')


def resume_diff(saved, current):
    """RESUME_KEYS 中 saved 與 current 不同的參數，回傳 {名稱: (舊, 新)}。"""
    return {k: (saved.get(k), current.get(k)) for k in RESUME_KEYS if saved.get(k) != current.get(k)}


def resolve_amp(amp, device):
    """--amp auto 解析成 bf16 或 none：只有 GPU 硬體支援 bf16 時用 bf16，不算模擬。"""
    import torch
    if amp != 'auto':
        return amp
    if device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False):
        return 'bf16'
    return 'none'


def _model_class():
    """匯入 torch 與 KataGo 後才定義模型類別，這個模組在沒有 torch 時也能匯入。"""
    import torch

    class Permove(torch.nn.Module):
        """humanv0 的主幹加 N_LEVELS 份 policy head，forward 回傳 (N, N_LEVELS, 362) 的 policy logits。"""

        def __init__(self, base):
            super().__init__()
            self.config = base.config
            self.conv_spatial = base.conv_spatial
            self.linear_global = base.linear_global
            self.metadata_encoder = base.metadata_encoder
            self.blocks = base.blocks
            self.norm_trunkfinal = base.norm_trunkfinal
            self.act_trunkfinal = base.act_trunkfinal
            self.heads = torch.nn.ModuleList(copy.deepcopy(base.policy_head) for _ in range(N_LEVELS))

        def train(self, mode=True):
            super().train(mode)
            self.norm_trunkfinal.eval()  # 用 humanv0 的 running 統計量，訓練與推論一致
            return self

        def forward(self, spatial, glob, meta):
            mask = spatial[:, 0:1, :, :].contiguous()
            mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)
            mask_sum = torch.sum(mask)
            out = self.conv_spatial(spatial) + self.linear_global(glob).unsqueeze(-1).unsqueeze(-1)
            out = out + self.metadata_encoder.forward(meta, None).unsqueeze(-1).unsqueeze(-1)
            for block in self.blocks:
                out = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)
            out = self.act_trunkfinal(self.norm_trunkfinal(out, mask=mask, mask_sum=mask_sum))
            # 每份 head 輸出 6 個 policy channel，只用第 0 個（主要 policy）
            return torch.stack([h(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum,
                                  extra_outputs=None)[:, 0, :] for h in self.heads], dim=1)

    return Permove


def from_humanv0(path=kt.CHECKPOINT):
    """用 humanv0（SWA 權重）初始化，10 份 head 的初值都等於原本的 policy head。"""
    return _model_class()(kt.load_humanv0(path)).train()


def load_permove(path, device='cpu'):
    """讀 train_permove 存的 model.pt 或 last.pt，回傳 eval 模式的模型。"""
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    kg = kt.katago()
    base = kg['model_pytorch'].Model(sd['config'], kt.BOARD)
    base.initialize()  # fixscale 的 scale 是一般屬性，不在 state_dict 內
    model = _model_class()(base)
    model.load_state_dict(sd['model'])
    return model.to(device).eval()


def symmetry_tables():
    """(8, 362) int64：第 k 列是 symmetry.apply 第 k 種變換後，每個 policy index 的新位置。"""
    eye = np.eye(kt.N_POLICY, dtype=np.float32)
    return np.stack([kt.policy_symmetry(eye, k).argmax(axis=1) for k in range(symmetry.N_SYMMETRIES)])


def unpack_bits(packed, n_bits):
    """GPU 上解包：(B, n_bytes) uint8 tensor 還原成 (B, n_bits) uint8，np.packbits 的 big bit order。"""
    import torch
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :n_bits]


def load_shards(split_dir):
    """一個 split 的分片串接成 numpy 陣列，欄位是 extract_permove 的 spatial、global、legal、move、rank_idx、side、
    key、turn、player_id、hd、color。"""
    paths = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
    if not paths:
        raise FileNotFoundError('%s 下沒有 .npz 分片' % split_dir)
    keys = ('spatial', 'global', 'legal', 'move', 'rank_idx', 'side', 'key', 'turn', 'player_id', 'hd', 'color')
    parts = {k: [] for k in keys}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            for k in keys:
                parts[k].append(z[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    if data['spatial'].shape[1] != SPATIAL_BYTES or data['legal'].shape[1] != LEGAL_BYTES:
        raise ValueError('%s 的分片格式不符' % split_dir)
    return data


def check_moves_legal(data):
    """實際落子都要在合法遮罩內，否則 log 機率是 -inf。"""
    legal = np.unpackbits(data['legal'], axis=1, count=kt.N_POLICY).astype(bool)
    bad = ~legal[np.arange(len(legal)), data['move'].astype(np.int64)]
    if bad.any():
        raise ValueError('%d 個局面的實際落子不在合法遮罩內，例如 %s turn %d'
                         % (bad.sum(), data['key'][bad][0], data['turn'][bad][0]))


class Batcher:
    """整份資料放在 device 上，依索引取 batch 並解包，可套用一種對稱。"""

    def __init__(self, data, device):
        import torch
        self.device = device
        self.spatial = torch.from_numpy(data['spatial']).to(device)
        self.glob = torch.from_numpy(data['global']).to(device)
        self.legal = torch.from_numpy(data['legal']).to(device)
        self.move = torch.from_numpy(data['move'].astype(np.int64)).to(device)
        self.rank = torch.from_numpy(data['rank_idx'].astype(np.int64)).to(device)
        self.sym_table = torch.from_numpy(symmetry_tables()).to(device)
        self.n = len(data['move'])

    def get(self, idx, sym=0):
        """回傳 (spatial float, global, legal bool, move, rank)，idx 是 device 上的 long tensor。"""
        b = len(idx)
        spatial = unpack_bits(self.spatial[idx], SPATIAL_BITS).reshape(b, kt.N_SPATIAL, kt.BOARD, kt.BOARD).float()
        legal = unpack_bits(self.legal[idx], kt.N_POLICY).bool()
        move = self.move[idx]
        if sym:
            spatial = symmetry.apply(spatial, sym)
            legal = kt.policy_symmetry(legal, sym)
            move = self.sym_table[sym][move]
        return spatial, self.glob[idx], legal, move, self.rank[idx]


def level_log_probs(logits, legal, move):
    """logits (B, L, 362) 在合法點上做 log_softmax，取實際落子：回傳 (B, L) 的 log 機率。"""
    import torch
    masked = logits.float().masked_fill(~legal.unsqueeze(1), float('-inf'))
    logp = torch.log_softmax(masked, dim=-1)
    return logp.gather(-1, move.view(-1, 1, 1).expand(-1, logits.shape[1], 1)).squeeze(-1)


def losses(model, meta_row, batch, lam, amp):
    """回傳 (總 loss, policy loss, rank loss, rank 正確數)。"""
    import torch
    spatial, glob, legal, move, rank = batch
    meta = meta_row.expand(len(move), -1)
    with torch.autocast(device_type=spatial.device.type, dtype=torch.bfloat16, enabled=amp == 'bf16'):
        logits = model(spatial, glob, meta)
    rank_logits = level_log_probs(logits, legal, move)
    policy = -rank_logits.gather(1, rank.view(-1, 1)).mean()
    rank_loss = torch.nn.functional.cross_entropy(rank_logits, rank)
    correct = (rank_logits.argmax(1) == rank).sum()
    return policy + lam * rank_loss, policy, rank_loss, correct


def wsd_factor(step, total, warmup_frac, decay_frac):
    """WSD 排程的 lr 倍率：前 warmup_frac 線性升到 1，最後 decay_frac 線性降到 0，中間維持 1。"""
    warmup = max(1, int(total * warmup_frac))
    decay = int(total * decay_frac)
    if step < warmup:
        return (step + 1) / warmup
    if step >= total - decay:
        return max(0.0, (total - step) / max(1, decay))
    return 1.0


def evaluate(model, meta_row, batcher, lam, amp, batch):
    """整份 holdout（原方向）的平均 policy loss、rank loss 與 rank 準確率。"""
    import torch
    model.eval()
    sums = np.zeros(3)
    with torch.no_grad():
        for s in range(0, batcher.n, batch):
            idx = torch.arange(s, min(s + batch, batcher.n), device=batcher.device)
            b = tuple(t.to(meta_row.device) for t in batcher.get(idx))  # 資料可能放在 CPU
            _, policy, rank_loss, correct = losses(model, meta_row, b, lam, amp)
            sums += [policy.item() * len(idx), rank_loss.item() * len(idx), correct.item()]
    model.train()
    return dict(zip(('policy_loss', 'rank_loss', 'rank_acc'), (sums / batcher.n).tolist()))


def save_checkpoint(path, model, opt, step, args):
    """先寫暫存檔再改名，存 state_dict、humanv0 的 config、optimizer（可為 None）、步數與參數。"""
    import torch
    tmp = path + '.tmp'
    torch.save({'model': model.state_dict(), 'config': model.config, 'optimizer': opt.state_dict() if opt else None,
                'step': step, 'args': vars(args)}, tmp)
    os.replace(tmp, path)


def train(args):
    import torch
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    args.amp = resolve_amp(args.amp, device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    data = load_shards(os.path.join(args.data, 'train'))
    holdout = load_shards(os.path.join(args.data, 'holdout'))
    check_moves_legal(data)
    check_moves_legal(holdout)
    if args.limit:
        # 分片依等級排序，取固定 seed 的隨機子集才涵蓋各等級
        rng = np.random.default_rng(args.seed)
        pick = np.sort(rng.permutation(len(data['move']))[:args.limit])
        data = {k: v[pick] for k, v in data.items()}
        pick = np.sort(rng.permutation(len(holdout['move']))[:args.limit])
        holdout = {k: v[pick] for k, v in holdout.items()}
    try:
        train_b, hold_b = Batcher(data, device), Batcher(holdout, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print('VRAM 不足，資料改放 CPU', flush=True)
        train_b, hold_b = Batcher(data, torch.device('cpu')), Batcher(holdout, torch.device('cpu'))
    steps_per_epoch = train_b.n // args.batch
    if steps_per_epoch == 0:
        raise SystemExit('train 只有 %d 個局面，少於 batch %d' % (train_b.n, args.batch))
    total = args.max_steps or steps_per_epoch * args.epochs
    eval_steps = sorted({max(1, round(total * f / 10)) for f in range(1, 11)})
    print('train %d 個局面、holdout %d 個；batch %d，共 %d 步，device %s，amp %s'
          % (train_b.n, hold_b.n, args.batch, total, device, args.amp), flush=True)

    model = from_humanv0(args.humanv0).to(device)
    heads = [p for n, p in model.named_parameters() if n.startswith('heads.')]
    trunk = [p for n, p in model.named_parameters() if not n.startswith('heads.')]
    opt = torch.optim.AdamW([{'params': trunk, 'lr': args.lr_trunk}, {'params': heads, 'lr': args.lr_head}],
                            weight_decay=args.weight_decay)
    base_lrs = [args.lr_trunk, args.lr_head]
    meta_row = torch.from_numpy(kt.profile_row(METADATA_PROFILE)).to(device)
    last = os.path.join(args.ckpt_dir, 'last.pt')
    history = os.path.join(args.ckpt_dir, 'history.jsonl')
    step = 0
    if os.path.exists(last):
        sd = torch.load(last, map_location='cpu', weights_only=False)
        diff = resume_diff(sd['args'], vars(args))
        if diff:
            raise SystemExit('%s 的參數和這次不同（舊, 新）：%s\n要用新參數請換 --ckpt-dir' % (last, diff))
        model.load_state_dict(sd['model'])
        opt.load_state_dict(sd['optimizer'])
        step = sd['step']
        print('從 %s 接續，第 %d 步' % (last, step), flush=True)
        if sd['args'].get('max_steps') != args.max_steps:
            print('max_steps 從 %s 改成 %s，lr 排程依新的總步數 %d' % (sd['args'].get('max_steps'), args.max_steps, total),
                  flush=True)
    elif os.path.exists(history):
        raise SystemExit('%s 有 history.jsonl 但沒有 last.pt，請換 --ckpt-dir 或刪掉舊檔' % args.ckpt_dir)
    else:
        with open(history, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'step': 0, **evaluate(model, meta_row, hold_b, args.lam, args.amp, args.eval_batch)}) + '\n')

    model.train()
    train_sec, seen, run, n_run = 0.0, 0, np.zeros(4), 0  # train_sec 只計訓練步，不含 eval 與存檔
    t0 = time.time()
    perm_epoch, perm = None, None
    while step < total:
        t_step = time.time()
        epoch, offset = divmod(step, steps_per_epoch)
        if epoch != perm_epoch:
            # 每個 epoch 的順序與每步的對稱由 seed 決定，接續時結果相同
            perm = torch.randperm(train_b.n, generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
            perm_epoch = epoch
        idx = perm[offset * args.batch:(offset + 1) * args.batch].to(train_b.device)
        sym = int(np.random.default_rng((args.seed, step)).integers(symmetry.N_SYMMETRIES))
        batch = tuple(t.to(device) for t in train_b.get(idx, sym))
        factor = wsd_factor(step, total, args.warmup_frac, args.decay_frac)
        for g, lr in zip(opt.param_groups, base_lrs):
            g['lr'] = lr * factor
        loss, policy, rank_loss, correct = losses(model, meta_row, batch, args.lam, args.amp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        seen += len(idx)
        run += [loss.item(), policy.item(), rank_loss.item(), correct.item() / len(idx)]  # item() 會等 GPU 算完
        n_run += 1
        train_sec += time.time() - t_step
        if step % args.log_every == 0 or step == total:
            print('step %d/%d loss %.4f policy %.4f rank %.4f acc %.3f lr×%.2f %.1f 局面/秒'
                  % (step, total, *(run / n_run), factor, seen / train_sec), flush=True)
            run[:], n_run = 0, 0
        if step in eval_steps:
            save_checkpoint(last, model, opt, step, args)
            row = {'step': step, 'elapsed_min': (time.time() - t0) / 60,
                   **evaluate(model, meta_row, hold_b, args.lam, args.amp, args.eval_batch)}
            with open(history, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            print('holdout %s' % row, flush=True)
    save_checkpoint(os.path.join(args.ckpt_dir, 'model.pt'), model, None, step, args)
    print('完成，模型在 %s' % os.path.join(args.ckpt_dir, 'model.pt'), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default='permove-features/spike', help='含 train/、holdout/ 分片的目錄')
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--humanv0', default=kt.CHECKPOINT)
    ap.add_argument('--katago-python', help='KataGo 原始碼的 python/ 目錄，缺省時讀環境變數 KATAGO_PYTHON')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-steps', type=int, default=0, help='非 0 時只跑這麼多步，用於冒煙測試')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--eval-batch', type=int, default=512)
    ap.add_argument('--lr-trunk', type=float, default=2e-5)
    ap.add_argument('--lr-head', type=float, default=2e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--lam', type=float, default=1.0, help='rank loss 的權重 λ')
    ap.add_argument('--warmup-frac', type=float, default=0.02)
    ap.add_argument('--decay-frac', type=float, default=0.2)
    ap.add_argument('--amp', choices=('auto', 'bf16', 'none'), default='auto',
                    help='auto 在支援 bf16 硬體的 GPU 上用 bf16，其他情況用 float32')
    ap.add_argument('--device')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='train、holdout 各取這麼多個局面的固定 seed 隨機子集，用於冒煙測試')
    ap.add_argument('--log-every', type=int, default=100)
    args = ap.parse_args()
    if args.katago_python:
        os.environ['KATAGO_PYTHON'] = args.katago_python
    train(args)


if __name__ == '__main__':
    main()
