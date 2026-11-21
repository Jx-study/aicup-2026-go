"""兩層 transformer 的訓練與同題預測，不依賴 repo 其他程式，可放進 Colab bundle 單獨執行。

資料目錄（--data）需要（pack_dataset.py 產生）：
  --pool 列出的訓練檔（預設 train_p1_moves.npz、train_p2_moves.npz）、val648_moves.npz、test_moves.npz
  questions-s{seed}.npz（export_questions.py 產生）
輸出目錄（--out）：
  tune 模式：用 --tune-train 訓練，在 --tune-dev 上比較 epoch 數，寫 tune-E{E}-ts{ts}.json 與 epochs.txt
  pool 模式：pool-ts{ts}-s{seed}.npz（SC/MX 每題機率）、pool-ts{ts}-test.npy
  cv 模式：cv-ts{ts}-s{seed}-f{fold}.npz，五折齊了再合併成 cv-ts{ts}-s{seed}.npz
  final 模式：--pool 加全部 val648 訓練，寫 final-ts{ts}.pt（state_dict）、final-ts{ts}-test.npy，
    有 --predict-data 時另寫 final-ts{ts}-{tag}.npz
  curve 模式：用 --tune-train 訓練，每個 epoch 把 dev 得分、dev NLL、train NLL、步數寫進 curve-ts{ts}.json
  ablate 模式：pool 模式的訓練，val648 SC/MX 題在原輸入與 4 種消融輸入下的機率，寫 ablate-ts{ts}-s{seed}.npz
--sched 選學習率排程：cosine（預設）或 const（固定 3e-4）。
已存在的輸出會跳過，中斷後重跑同一指令會接續。
--max-players 只取部分訓練玩家，--max-games-per-player 每人只留多數色的 k 局（pool 模式），
--opp-aug 把 HD0 局的目標與對手互換後加入訓練。
--joint（cv、final 模式）：--data 是 6d 打包資料夾、--data2 是 2d 打包資料夾，--pool、--pool2 分列兩邊的訓練檔。
  每手 18 維，第三個 profile 分成 6d、2d 兩個槽加有無旗標，每局只有一個槽有值。
  val648 與 test 每局有兩種單槽版本：訓練時每次抽題各局隨機選一種，預測時兩種各預測一次取平均。
訓練 loss 由 --loss 選擇：ce 是 cross-entropy，exp 是負的期望得分；預測時都用 pick_level 選期望得分最高的等級。
用法：python tf_run.py --mode tune --data DATA --out OUT
      python tf_run.py --mode pool --data DATA --out OUT
      python tf_run.py --mode cv --data DATA --out OUT
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

NEAR_SCORE = math.exp(-1.0)
SIZES = (5, 10, 15, 20)
MAX_GAMES, MAX_TURNS = 20, 200
IN_DIM = 14  # 讀入打包檔時改成檔案的每手維度，加 loss 欄的打包檔是 20


class MoveLevelEncoder(nn.Module):
    """局內逐手 self-attention，再用 2 個 query 做 attention pooling，輸出 64 維單局表示。"""

    def __init__(self, in_dim=None, d_model=32, n_heads=4, n_query_heads=2, dropout=0.3, max_len=250):
        super().__init__()
        self.in_proj = nn.Linear(in_dim or IN_DIM, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(n_query_heads, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model * n_query_heads)

    def forward(self, x, mask):
        h = self.in_proj(x) + self.pos_emb[:, :x.shape[1], :]
        pad = ~mask
        h = self.transformer(h, src_key_padding_mask=pad)
        scores = torch.einsum('qd,btd->bqt', self.queries, h) / math.sqrt(h.shape[-1])
        attn = F.softmax(scores.masked_fill(pad.unsqueeze(1), -1e9), dim=-1)
        return self.norm(torch.einsum('bqt,btd->bqd', attn, h).reshape(x.shape[0], -1))


class GameLevelSetTransformer(nn.Module):
    """題內跨局 self-attention 加 attention pooling，輸出 64 維題目表示。"""

    def __init__(self, d_model=64, n_heads=4, dropout=0.3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.set_transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, g, mask):
        pad = ~mask
        h = self.set_transformer(g, src_key_padding_mask=pad)
        scores = torch.einsum('qd,bnd->bqn', self.query, h).squeeze(1) / math.sqrt(h.shape[-1])
        attn = F.softmax(scores.masked_fill(pad, -1e4), dim=-1).unsqueeze(1)
        return self.norm(torch.einsum('bqn,bnd->bqd', attn, h).squeeze(1))


class HierarchicalRankTransformer(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.move_encoder = MoveLevelEncoder(dropout=dropout)
        self.set_transformer = GameLevelSetTransformer(dropout=dropout)
        self.classifier = nn.Linear(64, 10)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, move_mask, game_mask):
        B, N, T, D = x.shape
        valid = game_mask.view(-1)  # 只編碼真實存在的局，padding 局全遮罩會產生 NaN
        g = torch.zeros(B * N, 64, device=x.device, dtype=torch.float32)
        g[valid] = self.move_encoder(x.view(B * N, T, D)[valid], move_mask.view(B * N, T)[valid])
        return self.classifier(self.set_transformer(g.view(B, N, -1), game_mask))


def score_matrix(n=10):
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return np.where(d == 0, 1.0, np.where(d == 1, NEAR_SCORE, 0.0)).astype(np.float32)


class ExpectationLoss(nn.Module):
    """負的期望得分。"""

    def __init__(self):
        super().__init__()
        self.register_buffer('W', torch.from_numpy(score_matrix()))

    def forward(self, logits, y):
        return -(F.softmax(logits, -1) * self.W[y]).sum(-1).mean()


def pick_level(p):
    return (p @ score_matrix()).argmax(1)


class Games:
    """一份打包檔的逐局陣列。"""

    def __init__(self, path):
        d = np.load(path)
        self.keys = d['keys'].astype(str)
        self.moves, self.masks = d['moves'], d['masks']
        self.ranks, self.pids = d['ranks'], d['player_ids'].astype(str)
        self.colors, self.hds = d['colors'], d['hds']
        self.pos = {k: i for i, k in enumerate(self.keys)}
        global IN_DIM
        IN_DIM = self.moves.shape[2]


def concat(parts):
    out = Games.__new__(Games)
    for f in ('keys', 'moves', 'masks', 'ranks', 'pids', 'colors', 'hds'):
        setattr(out, f, np.concatenate([getattr(p, f) for p in parts]))
    out.pos = {k: i for i, k in enumerate(out.keys)}
    return out


def subset(g, keep):
    out = Games.__new__(Games)
    for f in ('keys', 'moves', 'masks', 'ranks', 'pids', 'colors', 'hds'):
        setattr(out, f, getattr(g, f)[keep])
    out.pos = {k: i for i, k in enumerate(out.keys)}
    return out


def swap_sides(g):
    """HD0 局互換目標與對手，標籤沿用原等級（互先局的對手和目標同等級）。"""
    h = subset(g, g.hds == 0)
    m = h.moves.copy()
    m[:, :, 0:5], m[:, :, 6:11] = h.moves[:, :, 6:11], h.moves[:, :, 0:5]  # 兩側的 3 個 log p、log pmax、entropy
    m[:, :, 11] = (1 - h.moves[:, :, 11]) * h.masks  # 執黑旗標翻轉，padding 維持 0
    if m.shape[2] == 20:
        m[:, :, 14:17], m[:, :, 17:20] = h.moves[:, :, 17:20], h.moves[:, :, 14:17]  # 兩側的 loss 欄
    h.moves, h.colors = m, 1 - h.colors
    h.keys = np.char.add(h.keys, '|opp')
    h.pids = np.char.add(h.pids, '|opp')  # 每位原玩家的對手局另成一組，不和原玩家混題
    h.pos = {k: i for i, k in enumerate(h.keys)}
    return h


def widen(g, slot):
    """14 維轉成 18 維：第三個 profile 放進 6d 或 2d 槽（14、15 是 2d 槽的兩側），16、17 是兩個槽的有無旗標。"""
    global IN_DIM
    assert g.moves.shape[2] == 14
    m = np.zeros(g.moves.shape[:2] + (18,), np.float32)
    m[:, :, :14] = g.moves
    if slot == '2d':
        m[:, :, 14:16] = g.moves[:, :, [2, 8]]
        m[:, :, [2, 8]] = 0.0
    m[:, :, 16 if slot == '6d' else 17] = g.masks  # padding 手維持 0
    g.moves, IN_DIM = m, 18
    return g


def joint_train_set(pool, v6, v2):
    """pool 加 val648 的兩種版本；v2 的列只當 v6 列的替代版本，不單獨出題。"""
    assert np.array_equal(v6.keys, v2.keys) and np.array_equal(v6.masks, v2.masks)
    g = concat([pool, v6, v2])
    n0, nv = len(pool.keys), len(v6.keys)
    g.alt = np.full(len(g.keys), -1, np.int64)
    g.alt[n0:n0 + nv] = np.arange(n0 + nv, n0 + 2 * nv)
    g.primary = np.ones(len(g.keys), bool)
    g.primary[n0 + nv:] = False
    return g


def load_sets(args, D):
    """回傳 (訓練池, val648 版本清單)；--joint 時版本清單是 [6d, 2d]，兩者列順序相同。"""
    if not args.joint:
        return concat([Games(D + f) for f in args.pool]), [Games(D + 'val648_moves.npz')]
    D2 = os.path.join(args.data2, '')
    parts = [widen(Games(D + f), '6d') for f in args.pool] + [widen(Games(D2 + f), '2d') for f in args.pool2]
    for slot in (parts[:len(args.pool)], parts[len(args.pool):]):  # 6d、2d 各自互斥，兩槽之間允許同一玩家
        assert sum(len(set(p.pids)) for p in slot) == len(set().union(*(set(p.pids) for p in slot))), '同一個槽的訓練檔之間有重複玩家'
    pool = concat(parts)
    views = [widen(Games(D + 'val648_moves.npz'), '6d'), widen(Games(D2 + 'val648_moves.npz'), '2d')]
    assert np.array_equal(views[0].keys, views[1].keys)
    return pool, views


def load_views(args, D, name):
    """同一份打包檔的版本清單，--joint 時是 [6d, 2d]。"""
    if not args.joint:
        return [Games(D + name)]
    v = [widen(Games(D + name), '6d'), widen(Games(os.path.join(args.data2, '') + name), '2d')]
    assert np.array_equal(v[0].keys, v[1].keys) and np.array_equal(v[0].masks, v[1].masks)
    return v


def train_set(pool, views, keep, args):
    """訓練池加 keep 選中的 val648 局。"""
    if len(views) == 1:
        return prepare(concat([pool, subset(views[0], keep)]), args)
    return joint_train_set(pool, subset(views[0], keep), subset(views[1], keep))


def cap_games(g, k, seed):
    """每位玩家只留 HD < 2 局中局數較多那一色的 k 局（不足 k 局全留），模擬每人只抽 k 局同色局。"""
    rng = np.random.default_rng(seed)
    keep = []
    ok = g.hds < 2
    for p in np.unique(g.pids):
        own = np.flatnonzero((g.pids == p) & ok)
        b, w = own[g.colors[own] == 1], own[g.colors[own] == 0]
        side = b if len(b) >= len(w) else w
        keep.extend(rng.permutation(side)[:k].tolist())
    return subset(g, np.sort(np.array(keep, np.int64)))


def prepare(g, args):
    """依 --max-players 抽玩家子集，依 --max-games-per-player 限制每人局數，依 --opp-aug 加入互換局。"""
    if args.max_players:
        players = np.unique(g.pids)
        keep = np.random.default_rng(args.subset_seed).permutation(players)[:args.max_players]
        g = subset(g, np.isin(g.pids, keep))
    if args.max_games_per_player:
        g = cap_games(g, args.max_games_per_player, args.subset_seed)
    if args.opp_aug:
        g = concat([g, swap_sides(g)])
    return g


class TrainQuestions(Dataset):
    """每位玩家每 epoch 出 repeats 題：同色、HD < 2，局數在 SIZES 中隨機。"""

    def __init__(self, g, repeats=10):
        self.g, self.repeats, self.by_player = g, repeats, {}
        self.alt = getattr(g, 'alt', None)
        sel = g.hds < 2
        if self.alt is not None:
            sel &= g.primary
        for i in np.flatnonzero(sel):
            self.by_player.setdefault(g.pids[i], ([], []))[int(g.colors[i])].append(i)
        self.players = sorted(p for p, c in self.by_player.items() if max(len(c[0]), len(c[1])) >= SIZES[0])

    def __len__(self):
        return len(self.players) * self.repeats

    def __getitem__(self, idx):
        c = self.by_player[self.players[idx % len(self.players)]]
        avail = random.choice([x for x in c if len(x) >= SIZES[0]])
        n = random.choice([s for s in SIZES if s <= len(avail)])
        idx = random.sample(avail, n)
        if self.alt is not None:
            idx = [int(self.alt[i]) if self.alt[i] >= 0 and random.random() < 0.5 else i for i in idx]  # 各局隨機選 6d 或 2d 版本
        return pack(self.g, idx, self.g.ranks[avail[0]])


def pack(g, idx, y):
    x = np.zeros((MAX_GAMES, MAX_TURNS, IN_DIM), np.float32)
    m = np.zeros((MAX_GAMES, MAX_TURNS), bool)
    gm = np.zeros(MAX_GAMES, bool)
    x[:len(idx)], m[:len(idx)], gm[:len(idx)] = g.moves[idx], g.masks[idx], True
    return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(gm), torch.tensor(int(y))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train(g, ts, args, device, epochs=None, on_epoch=None):
    """回傳最後一個 epoch 的模型；on_epoch(ep, model) 在每個 epoch 結束時呼叫。"""
    epochs = epochs or args.epochs
    seed_all(ts)
    ds = TrainQuestions(g)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=args.workers, drop_last=True,
                        worker_init_fn=lambda w: random.seed(torch.initial_seed() % 2 ** 32),  # 每個 epoch 不同
                        generator=torch.Generator().manual_seed(ts))
    model = HierarchicalRankTransformer().to(device)
    loss_fn = ExpectationLoss().to(device) if args.loss == 'exp' else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5) if args.sched == 'cosine' else None
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b, (x, m, gm, y) in enumerate(loader):
            if args.limit_batches and b >= args.limit_batches:
                break
            x, m, gm, y = x.to(device), m.to(device), gm.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, m, gm), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        if sched:
            sched.step()
        model.steps = getattr(model, 'steps', 0) + n  # 累計 optimizer 步數
        print('  ts %d epoch %2d/%d  %.1fs  train loss %.4f  players %d' % (ts, ep + 1, epochs, time.time() - t0, tot / n, len(ds.players)), flush=True)
        if on_epoch:
            on_epoch(ep + 1, model)
    return model


@torch.no_grad()
def predict(model, g, groups, device, bs=128):
    """groups 是每題的局 index 清單，回傳 (題數, 10) 機率。"""
    model.eval()
    out = []
    for s in range(0, len(groups), bs):
        batch = [pack(g, idx, 0) for idx in groups[s:s + bs]]
        x, m, gm = (torch.stack([b[i] for b in batch]).to(device) for i in range(3))
        out.append(F.softmax(model(x, m, gm).float(), -1).cpu().numpy())
    return np.concatenate(out)


def predict_views(model, views, groups, device):
    """每個版本各預測一次取平均；版本間列順序相同，groups 共用。"""
    if len(views) == 1:
        return predict(model, views[0], groups, device)
    return np.mean([predict(model, v, groups, device) for v in views], 0).astype(np.float32)


def question_groups(q, t, val, sel=None):
    keys, n = q[t + '_keys'], q[t + '_n']
    rows = range(len(n)) if sel is None else np.flatnonzero(sel)
    return [[val.pos[k] for k in keys[i, :n[i]]] for i in rows]


def test_groups(test):
    by_q = {}
    for i, k in enumerate(test.keys):
        by_q.setdefault(k.rsplit('_g', 1)[0], []).append(i)
    order = ['pub_q_%04d' % i for i in range(1, 401)] + ['priv_q_%04d' % i for i in range(1, 401)]
    assert all(q in by_q for q in order), '測試題不足 800 題'
    return [by_q[q][:MAX_GAMES] for q in order]


def save_npz(path, **kw):
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, path)


def fixed_questions(g, repeats=10, seed=7):
    """固定的評估題：同色、HD < 2，局數輪流取 5、10、15、20（不足時取最大可行），回傳 (局 index 清單, 標籤)。"""
    rng = random.Random(seed)
    ds = TrainQuestions(g)
    groups, ys = [], []
    for r in range(repeats):
        for i, p in enumerate(ds.players):
            c = ds.by_player[p]
            avail = rng.choice([x for x in c if len(x) >= SIZES[0]])
            ok = [s for s in SIZES if s <= len(avail)]
            n = SIZES[(r + i) % 4] if SIZES[(r + i) % 4] in ok else ok[-1]
            groups.append(rng.sample(avail, n)); ys.append(g.ranks[avail[0]])
    return groups, np.array(ys)


def run_tune(args, device, D, qs):
    """用 --tune-train 訓練，在 --tune-dev 的固定題上比較 epoch 數，結果寫入 epochs.txt。"""
    pool, dev = Games(D + args.tune_train), Games(D + args.tune_dev)
    assert not set(pool.pids) & set(dev.pids)
    groups, ys = fixed_questions(dev)
    W = score_matrix()
    res = {}
    for E in args.tune_epochs:
        for ts in args.train_seeds[:2]:
            path = args.out + 'tune-E%d-ts%d.json' % (E, ts)
            if not os.path.exists(path):
                curve = []

                def ev(ep, model):
                    sc = float(W[ys, pick_level(predict(model, dev, groups, device))].mean())
                    curve.append(sc)
                    print('    tune-dev 得分 %.4f' % sc, flush=True)
                print('tune E %d ts %d：訓練 %d 局，評分 %d 題' % (E, ts, len(pool.keys), len(ys)), flush=True)
                train(pool, 1000 + ts, args, device, epochs=E, on_epoch=ev)
                with open(path + '.tmp', 'w') as f:
                    json.dump({'final': curve[-1], 'curve': curve}, f)
                os.replace(path + '.tmp', path)
            with open(path) as f:
                res.setdefault(E, []).append(json.load(f)['final'])
    mean = {E: float(np.mean(v)) for E, v in res.items()}
    for E in sorted(mean):
        print('E %d：tune-dev 最後一個 epoch 得分 %s，平均 %.4f' % (E, ' '.join('%.4f' % x for x in res[E]), mean[E]))
    best = max(mean.values())
    pick = min(E for E in mean if mean[E] >= best - 0.003)  # 差距 0.003 以內取較小的 E
    with open(args.out + 'epochs.txt', 'w') as f:
        f.write(str(pick))
    print('選定 epochs = %d，已寫入 %sepochs.txt' % (pick, args.out))


def run_pool(args, device, D, qs):
    pool = concat([Games(D + f) for f in args.pool])
    val = Games(D + 'val648_moves.npz')
    assert not set(pool.pids) & set(val.pids)
    pool = prepare(pool, args)
    test = Games(D + 'test_moves.npz') if os.path.exists(D + 'test_moves.npz') else None
    for ts in args.train_seeds:
        outs = [args.out + 'pool-ts%d-s%d.npz' % (ts, s) for s in args.q_seeds]
        tout = args.out + 'pool-ts%d-test.npy' % ts
        if all(os.path.exists(o) for o in outs) and (test is None or os.path.exists(tout)):
            print('跳過 pool ts %d（已完成）' % ts); continue
        print('pool ts %d：訓練 %d 局' % (ts, len(pool.keys)), flush=True)
        model = train(pool, ts, args, device)
        for s, o in zip(args.q_seeds, outs):
            q = qs[s]
            save_npz(o, **{t + '_probs': predict(model, val, question_groups(q, t, val), device) for t in ('SC', 'MX')},
                     **{t + '_y': q[t + '_y'] for t in ('SC', 'MX')})
        if test is not None:
            np.save(tout + '.tmp.npy', predict(model, test, test_groups(test), device))
            os.replace(tout + '.tmp.npy', tout)
        print('pool ts %d 完成' % ts, flush=True)


def run_cv(args, device, D, qs):
    pool, views = load_sets(args, D)
    val = views[0]
    assert not set(pool.pids) & set(val.pids)
    for ts in args.train_seeds:
        for s in args.q_seeds:
            final = args.out + 'cv-ts%d-s%d.npz' % (ts, s)
            if os.path.exists(final):
                print('跳過 cv ts %d s %d（已完成）' % (ts, s)); continue
            q = qs[s]
            for f in range(5):
                part = args.out + 'cv-ts%d-s%d-f%d.npz' % (ts, s, f)
                if os.path.exists(part):
                    continue
                held = set(q['SC_p'][q['SC_fold'] == f]) | set(q['MX_p'][q['MX_fold'] == f])
                g = train_set(pool, views, ~np.isin(val.pids, sorted(held)), args)
                print('cv ts %d s %d f %d：訓練 %d 局，排除 %d 位' % (ts, s, f, len(g.keys), len(held)), flush=True)
                model = train(g, ts * 100 + s * 10 + f, args, device)
                save_npz(part, **{t + '_probs': predict_views(model, views, question_groups(q, t, val, q[t + '_fold'] == f), device)
                                  for t in ('SC', 'MX')})
            out = {}
            for t in ('SC', 'MX'):
                p = np.zeros((len(q[t + '_y']), 10), np.float32)
                for f in range(5):
                    p[q[t + '_fold'] == f] = np.load(args.out + 'cv-ts%d-s%d-f%d.npz' % (ts, s, f))[t + '_probs']
                out[t + '_probs'], out[t + '_y'] = p, q[t + '_y']
            save_npz(final, **out)
            print('cv ts %d s %d 完成' % (ts, s), flush=True)


def nll(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def run_curve(args, device, D):
    """用 --tune-train 訓練，每個 epoch 記 --tune-dev 固定題的得分與 NLL、訓練玩家固定題的 NLL、累計步數。"""
    pool, dev = Games(D + args.tune_train), Games(D + args.tune_dev)
    assert not set(pool.pids) & set(dev.pids)
    groups, ys = fixed_questions(dev)
    tgroups, tys = fixed_questions(pool, repeats=2)
    W = score_matrix()
    for ts in args.train_seeds:
        path = args.out + 'curve-ts%d.json' % ts
        if os.path.exists(path):
            with open(path) as f:
                if len(json.load(f)['curve']) == args.epochs:
                    print('跳過 curve ts %d（已完成）' % ts); continue
        rec = {'sched': args.sched, 'epochs': args.epochs, 'curve': []}

        def ev(ep, model):
            p, tp = predict(model, dev, groups, device), predict(model, pool, tgroups, device)
            row = {'epoch': ep, 'dev_score': float(W[ys, pick_level(p)].mean()), 'dev_nll': nll(p, ys),
                   'train_nll': nll(tp, tys), 'steps': int(model.steps)}
            rec['curve'].append(row)
            with open(path + '.tmp', 'w') as f:
                json.dump(rec, f)
            os.replace(path + '.tmp', path)  # 每個 epoch 存檔，中斷後保留已完成的部分
            print('    dev 得分 %.4f  dev NLL %.4f  train NLL %.4f  步數 %d' % (
                row['dev_score'], row['dev_nll'], row['train_nll'], row['steps']), flush=True)
        print('curve ts %d：訓練 %d 局，dev %d 題，train 固定題 %d 題' % (ts, len(pool.keys), len(ys), len(tys)), flush=True)
        train(pool, 1000 + ts, args, device, on_epoch=ev)


def col_mean(g):
    """訓練資料有效手的每欄平均。"""
    s, n = np.zeros(g.moves.shape[2]), 0
    for a in range(0, len(g.keys), 2000):
        m = g.masks[a:a + 2000]
        s += g.moves[a:a + 2000][m].sum(0, dtype=np.float64)
        n += m.sum()
    return (s / n).astype(np.float32)


def ablated(g, kind, mu):
    """回傳只換掉 moves 的副本；padding 位置維持 0。"""
    m = g.moves.copy()
    if kind == 'SHUF':
        rng = np.random.default_rng(0)
        for i in range(len(m)):
            idx = np.flatnonzero(g.masks[i])
            m[i, idx] = g.moves[i, rng.permutation(idx)]  # 整手一起移動
    else:
        cols = ABLATE_COLS[kind]
        m[:, :, cols] = np.where(g.masks[:, :, None], mu[cols], 0.0)
    out = Games.__new__(Games)
    out.__dict__.update(g.__dict__)
    out.moves = m
    return out


ABLATE_COLS = {'NOOPP': [6, 7, 8, 9, 10], 'NOP3': [2, 8], 'NOMETA': [11, 12, 13]}
ABLATIONS = ('BASE', 'SHUF', 'NOOPP', 'NOP3', 'NOMETA')


def run_ablate(args, device, D, qs):
    """pool 模式的訓練，對 val648 SC、MX 題用原輸入與 4 種消融輸入各預測一次。"""
    pool = concat([Games(D + f) for f in args.pool])
    val = Games(D + 'val648_moves.npz')
    assert not set(pool.pids) & set(val.pids)
    pool = prepare(pool, args)
    mu = col_mean(pool)
    print('訓練資料欄平均 %s' % np.round(mu, 4), flush=True)
    for ts in args.train_seeds:
        outs = {s: args.out + 'ablate-ts%d-s%d.npz' % (ts, s) for s in args.q_seeds}
        if all(os.path.exists(o) for o in outs.values()):
            print('跳過 ablate ts %d（已完成）' % ts); continue
        print('ablate ts %d：訓練 %d 局' % (ts, len(pool.keys)), flush=True)
        model = train(pool, ts, args, device)
        res = {s: {t + '_y': qs[s][t + '_y'] for t in ('SC', 'MX')} for s in args.q_seeds}
        for a in ABLATIONS:
            v = val if a == 'BASE' else ablated(val, a, mu)
            for s in args.q_seeds:
                for t in ('SC', 'MX'):
                    res[s]['%s_probs_%s' % (t, a)] = predict(model, v, question_groups(qs[s], t, v), device)
            del v
        for s, o in outs.items():
            save_npz(o, **res[s])
        print('ablate ts %d 完成' % ts, flush=True)


def run_final(args, device, D):
    """提交用：--pool 列出的檔案加全部 val648 玩家訓練，存權重與 test 機率；可另外預測 --predict-data 的固定題。"""
    base, views = load_sets(args, D)
    pool = train_set(base, views, np.ones(len(views[0].keys), bool), args)
    test = load_views(args, D, 'test_moves.npz')
    extra = None
    if args.predict_data:
        extra = (load_views(args, D, args.predict_data), dict(np.load(D + args.predict_questions)))
        assert not set(extra[0][0].pids) & set(pool.pids), '預測組的玩家出現在訓練資料'
    for ts in args.train_seeds:
        wpath, tout = args.out + 'final-ts%d.pt' % ts, args.out + 'final-ts%d-test.npy' % ts
        if os.path.exists(wpath) and os.path.exists(tout):
            print('跳過 final ts %d（已完成）' % ts); continue
        print('final ts %d：訓練 %d 局' % (ts, len(pool.keys)), flush=True)
        model = train(pool, ts, args, device)
        np.save(tout + '.tmp.npy', predict_views(model, test, test_groups(test[0]), device))
        os.replace(tout + '.tmp.npy', tout)
        if extra is not None:
            g, q = extra
            save_npz(args.out + 'final-ts%d-%s.npz' % (ts, args.predict_tag),
                     **{t + '_probs': predict_views(model, g, question_groups(q, t, g[0]), device) for t in ('SC', 'MX') if t + '_keys' in q})
        torch.save(model.state_dict(), wpath + '.tmp')
        os.replace(wpath + '.tmp', wpath)  # 權重最後寫，存在即代表這個 seed 全部完成
        print('final ts %d 完成' % ts, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('tune', 'pool', 'cv', 'final', 'curve', 'ablate'), required=True)
    ap.add_argument('--sched', choices=('cosine', 'const'), default='cosine')  # const 是學習率固定 3e-4
    ap.add_argument('--predict-data', default='')  # final 模式另外預測的打包檔，例如封存組
    ap.add_argument('--predict-questions', default='')  # 對應的固定題，格式同 export_questions.py
    ap.add_argument('--predict-tag', default='extra')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=0)  # 0 表示讀 tune 寫的 OUT/epochs.txt
    ap.add_argument('--max-players', type=int, default=0)  # 0 表示用全部訓練玩家
    ap.add_argument('--max-games-per-player', type=int, default=0)  # 0 表示不限，只用於 pool 模式
    ap.add_argument('--subset-seed', type=int, default=0)
    ap.add_argument('--opp-aug', action='store_true')
    ap.add_argument('--loss', choices=('ce', 'exp'), default='ce')  # exp 是負期望得分，會讓夾在中間的等級塌成 0 機率
    ap.add_argument('--pool', nargs='+', default=['train_p1_moves.npz', 'train_p2_moves.npz'])
    ap.add_argument('--joint', action='store_true')  # 6d、2d 兩池合訓，只用於 cv、final
    ap.add_argument('--data2', default='')  # --joint 的 2d 打包資料夾
    ap.add_argument('--pool2', nargs='+', default=['train_p1_moves.npz', 'train_p2_moves.npz'])
    ap.add_argument('--tune-train', default='train_p1_moves.npz')
    ap.add_argument('--tune-dev', default='train_p2_moves.npz')
    ap.add_argument('--tune-epochs', type=int, nargs='+', default=[15, 30, 45])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--train-seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--q-seeds', type=int, nargs='+', default=[47, 48, 49, 50, 51])
    ap.add_argument('--limit-batches', type=int, default=0)  # 冒煙測試用，0 表示不限
    args = ap.parse_args()
    assert not args.joint or (args.mode in ('cv', 'final') and args.data2 and not args.opp_aug and not args.max_players)
    assert not args.max_games_per_player or args.mode == 'pool'
    D = os.path.join(args.data, '')
    args.out = os.path.join(args.out, '')
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('裝置 %s，參數量 %d' % (device, sum(p.numel() for p in HierarchicalRankTransformer().parameters())))
    if args.mode == 'tune':
        return run_tune(args, device, D, None)
    if not args.epochs:
        with open(args.out + 'epochs.txt') as f:  # 先跑 --mode tune
            args.epochs = int(f.read())
    print('epochs = %d，loss = %s，sched = %s' % (args.epochs, args.loss, args.sched))
    if args.mode == 'final':
        return run_final(args, device, D)
    if args.mode == 'curve':
        return run_curve(args, device, D)
    qs = {s: dict(np.load(D + 'questions-s%d.npz' % s)) for s in args.q_seeds}
    {'pool': run_pool, 'cv': run_cv, 'ablate': run_ablate}[args.mode](args, device, D, qs)


if __name__ == '__main__':
    main()
