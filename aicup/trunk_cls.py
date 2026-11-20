"""humanv0 主幹判別式微調：單局 10 級分類，各局 log 機率可聚合成題目機率

模型：humanv0 的 stem 與前段 block 凍結，最後 --train-blocks 個 block 與 norm_trunkfinal 參與訓練，
主幹輸出（act_trunkfinal 之後）取每通道平均與最大值共 768 維，接 LayerNorm 與線性層輸出 10 級 logit。
一局的 logit 是該局各手 logit 的平均。模型固定在 eval 模式，norm 不更新統計量。
輸入有 history，metadata 固定 rank_1d，只用原方向。

訓練資料是 split_by_player(df, 180_000) 的 train 玩家。每個 epoch 每位玩家最多抽 --cap 局，每局目標方隨機 --train-moves 手，
loss 用該 epoch 各級局數的反比加權，所以輸出對應均勻先驗。每個 epoch 結束在 newdev 子集上算各級等權的單局 log loss。

三個模式：
    train   訓練，輸出目錄有 last.pt 時接續，每個 epoch 存 epoch-<n>.pt 與 newdev 的單局 log 機率
    infer   對 --questions（題目檔 SC_keys、MX_keys 的全部局）或 --test（測試集全部局）輸出單局 log 機率
    dtype   比較 fp32 與 fp16、bf16 的 humanv0 policy logit 與分類 logit 的差距

用法（在 repo 根目錄，需要 KataGo python/ 原始碼）：
    python -m aicup.trunk_cls train --out rank-trained-models/<commit> --train-dtype bf16
    python -m aicup.trunk_cls infer --ckpt rank-trained-models/<commit>/epoch-1.pt --questions <questions.npz> --out rank-trained-models/<commit>/val-games.npz
    python -m aicup.trunk_cls infer --ckpt rank-trained-models/<commit>/epoch-1.pt --test tests --out rank-trained-models/<commit>/test-games.npz
    python -m aicup.trunk_cls dtype --games 8
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd

from aicup.extract_features import SEED, load_dataframe, split_by_player
from aicup.scoring import LEVELS

SPLIT_VAL_SIZE = 180_000
PROFILE = 'rank_1d'
N_CLASSES = len(LEVELS)
NEWDEV_CSV = 'humansl-analysis/val-players-newdev-v180000.csv'
NEWDEV_GAMES = 20  # newdev 每位玩家最多取的局數，rng 為 SEED
COLOR_CODE = {'B': 0, 'W': 1}
DTYPES = ('fp32', 'fp16', 'bf16')


class GameStore:
    """全部局的 SGF 串成一個 uint8 陣列加位移，DataLoader fork 出的 worker 讀取時不會因 refcount 複製記憶體。

    欄位：blob、offsets（n + 1）、rank_idx、player（玩家編號，(rank, player_id) 唯一）、color（0 黑 1 白）、key。
    """

    def __init__(self, df):
        data = [s.encode('utf-8') for s in df['sgf_content']]
        lens = np.fromiter((len(b) for b in data), np.int64, len(data))
        self.offsets = np.concatenate([[0], np.cumsum(lens)])
        self.blob = np.frombuffer(b''.join(data), np.uint8)
        self.rank_idx = df['rank'].map(LEVELS.index).to_numpy(np.int64)
        pk = df['rank'].astype(str) + '/' + df['player_id'].astype(str)
        self.player = pd.factorize(pk)[0].astype(np.int64)
        self.color = df['color'].map(COLOR_CODE).to_numpy(np.int64)
        self.key = (df['rank'].astype(str) + '_' + df['game_id'].astype(str)).to_numpy()

    def __len__(self):
        return len(self.rank_idx)

    def sgf(self, i):
        return self.blob[self.offsets[i]:self.offsets[i + 1]].tobytes().decode('utf-8')


def epoch_games(player, cap, seed, epoch):
    """每位玩家最多 cap 局的索引，順序打亂。rng 由 (seed, epoch) 決定，所以接續時同一個 epoch 會得到同一份清單。"""
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(len(player))
    # 打亂後每位玩家取前 cap 個出現的局，等於每人不放回抽 cap 局
    seen = np.zeros(player.max() + 1, np.int64)
    keep = np.zeros(len(order), bool)
    for j, i in enumerate(order):
        p = player[i]
        if seen[p] < cap:
            seen[p] += 1
            keep[j] = True
    return order[keep]


def class_weights(labels, n=N_CLASSES):
    """各級局數的反比，平均為 1，(n,) float32。某級沒有局時丟 ValueError。"""
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    if (counts == 0).any():
        raise ValueError('有等級沒有局：%s' % counts.tolist())
    w = counts.sum() / (n * counts)
    return (w / w.mean()).astype(np.float32)


def spaced(turns, n):
    """turns 中等距取 n 個，不足全取。"""
    if len(turns) <= n:
        return list(turns)
    return [turns[i] for i in np.round(np.linspace(0, len(turns) - 1, n)).astype(int)]


def encode_sample(sgf, color, n_moves, rng=None):
    """一局目標方的輸入，(spatial (k, 22, 19, 19) uint8, global (k, 19) float32)。

    rng 有值時隨機不放回取 n_moves 手，否則等距取 n_moves 手。沒有可用的手或棋譜不合法時回傳 None。
    """
    from aicup.analyze_humansl import parse_game, target_turns
    from aicup.katago_torch import encode_game
    try:
        parsed = parse_game(sgf)
        turns = target_turns(parsed['moves'], color)
        if not turns:
            return None
        if rng is not None and len(turns) > n_moves:
            turns = sorted(rng.choice(turns, n_moves, replace=False).tolist())
        else:
            turns = spaced(turns, n_moves)
        enc = encode_game(parsed, turns, history=True)
    except ValueError:  # 壞 SGF、非 19 路、encore 等，該局略過
        return None
    return enc['spatial'], enc['global']


class GameDataset:
    """torch Dataset：第 j 個樣本是 store 中第 idx[j] 局的編碼、標籤與 j，失敗的局回傳 None。

    seed 不是 None 時隨機取手，rng 由 (seed, 局索引) 決定，可重現；否則等距取手。
    """

    def __init__(self, store, idx, n_moves, seed=None):
        self.store, self.idx, self.n_moves, self.seed = store, np.asarray(idx), n_moves, seed

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = int(self.idx[j])
        rng = None if self.seed is None else np.random.default_rng([self.seed, i])
        color = 'BW'[self.store.color[i]]
        out = encode_sample(self.store.sgf(i), color, self.n_moves, rng)
        if out is None:
            return None
        return out[0], out[1], int(self.store.rank_idx[i]), j


def collate(items):
    """把一批局的局面串起來，回傳 dict：spatial、global、game（每個局面屬於第幾局）、label、index（在 dataset 中的位置）。"""
    import torch
    items = [x for x in items if x is not None]
    if not items:
        return None
    game = np.concatenate([np.full(len(x[0]), g, np.int64) for g, x in enumerate(items)])
    return {'spatial': torch.from_numpy(np.concatenate([x[0] for x in items])),
            'global': torch.from_numpy(np.concatenate([x[1] for x in items])),
            'game': torch.from_numpy(game),
            'label': torch.tensor([x[2] for x in items]),
            'index': torch.tensor([x[3] for x in items])}


def game_mean(logits, game, n_games):
    """逐局面 logit (P, C) 依 game (P,) 取平均，(n_games, C)。"""
    import torch
    out = torch.zeros(n_games, logits.shape[1], dtype=logits.dtype, device=logits.device)
    out.index_add_(0, game, logits)
    cnt = torch.bincount(game, minlength=n_games).clamp_min(1).to(logits.dtype)
    return out / cnt[:, None]


def level_equal_nll(logp, y, n=N_CLASSES):
    """各級等權的平均負 log 機率：每級先平均再取 n 級平均。logp (N, n)，y (N,)。"""
    nll = -logp[np.arange(len(y)), y]
    return float(np.mean([nll[y == k].mean() for k in range(n) if (y == k).any()]))


def build_model(checkpoint, train_blocks):
    """回傳分類模型（torch.nn.Module）：humanv0 主幹加 768 維池化與線性頭。"""
    import torch
    from aicup import katago_torch as kt

    class TrunkClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = kt.load_humanv0(checkpoint)
            self.train_blocks = train_blocks
            for p in self.net.parameters():
                p.requires_grad_(False)
            for m in list(self.net.blocks[-train_blocks:]) + [self.net.norm_trunkfinal]:
                for p in m.parameters():
                    p.requires_grad_(True)
            c = self.net.c_trunk
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(2 * c), torch.nn.Linear(2 * c, N_CLASSES))
            self.register_buffer('meta', torch.from_numpy(kt.profile_row(PROFILE)))

        def train(self, mode=True):
            # 主幹固定 eval 模式，norm 不更新統計量；頭沒有 dropout，模式不影響結果
            super().train(mode)
            self.net.eval()
            return self

        def trunk_params(self):
            return [p for p in self.net.parameters() if p.requires_grad]

        def forward(self, spatial, glob):
            """逐局面 10 級 logit，(P, 10) float32。spatial (P, 22, 19, 19)，glob (P, 19)。"""
            net = self.net
            x = spatial.float()
            mask = x[:, 0:1].contiguous()
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
            mask_sum = mask.sum()
            k = len(net.blocks) - self.train_blocks
            with torch.no_grad():
                out = net.conv_spatial(x) + net.linear_global(glob).unsqueeze(-1).unsqueeze(-1)
                meta = net.metadata_encoder(self.meta.expand(len(glob), -1), None)
                out = out + meta.unsqueeze(-1).unsqueeze(-1)
                for block in net.blocks[:k]:
                    out = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)
            for block in net.blocks[k:]:
                out = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)
            out = net.act_trunkfinal(net.norm_trunkfinal(out, mask=mask, mask_sum=mask_sum))
            pooled = torch.cat([out.mean(dim=(2, 3)), out.amax(dim=(2, 3))], dim=1)
            return self.head(pooled.float()).float()

    return TrunkClassifier()


def autocast(device, name):
    import torch
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[name]
    return torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda' and name != 'fp32')


def predict(model, loader, device, dtype, n_items):
    """loader 的每局 log 機率 (n_items, 10) float32 與局面數，失敗的局 log 機率為 nan、局面數 0。"""
    import torch
    logp = np.full((n_items, N_CLASSES), np.nan, np.float32)
    npos = np.zeros(n_items, np.int64)
    model.eval()
    with torch.no_grad():
        for b in loader:
            if b is None:
                continue
            with autocast(device, dtype):
                z = model(b['spatial'].to(device, non_blocking=True), b['global'].to(device, non_blocking=True))
            g = game_mean(z.float(), b['game'].to(device), len(b['index']))
            idx = b['index'].numpy()
            logp[idx] = torch.log_softmax(g, dim=1).cpu().numpy()
            npos[idx] = np.bincount(b['game'].numpy(), minlength=len(idx))
    return logp, npos


def loader(dataset, games_per_batch, workers):
    import torch
    return torch.utils.data.DataLoader(dataset, batch_size=games_per_batch, num_workers=workers,
                                       collate_fn=collate, pin_memory=True, persistent_workers=False,
                                       prefetch_factor=4 if workers else None)


def load_split(train_dir):
    """回傳 (val 切分 df, train 切分 df)，都已打亂，player_id 為字串。"""
    df = load_dataframe(train_dir)
    df['player_id'] = df['player_id'].astype(str)
    if not df['sgf_content'].str.contains('HD[', regex=False).any():
        raise SystemExit('%s 不是官方訓練集 v1.1（沒有 HD[n]）' % train_dir)
    val, tr = split_by_player(df, SPLIT_VAL_SIZE)
    return val.reset_index(drop=True), tr.reset_index(drop=True)


def newdev_subset(val_df, newdev_csv=NEWDEV_CSV, per_player=NEWDEV_GAMES):
    """newdev 玩家每人最多 per_player 局（rng SEED），依 key 排序。"""
    nd = pd.read_csv(newdev_csv, dtype={'player_id': str})
    pk = set(nd['rank'].astype(str) + '/' + nd['player_id'])
    sub = val_df[(val_df['rank'].astype(str) + '/' + val_df['player_id']).isin(pk)]
    sub = sub.sample(frac=1.0, random_state=SEED).groupby(['rank', 'player_id'], group_keys=False).head(per_player)
    key = sub['rank'].astype(str) + '_' + sub['game_id'].astype(str)
    return sub.assign(_key=key).sort_values('_key').drop(columns='_key').reset_index(drop=True)


def train(args):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    config = {k: getattr(args, k) for k in ('cap', 'train_moves', 'eval_moves', 'train_blocks', 'epochs', 'games_per_batch',
                                             'lr', 'trunk_lr_ratio', 'weight_decay', 'warmup', 'seed', 'max_train_players')}
    config['train_csv'] = os.path.basename(args.train_csv) if args.train_csv else None
    cfg_path = os.path.join(args.out, 'train_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            old = json.load(f)
        if old != config:
            raise SystemExit('%s 的設定和這次不同：%s，這次 %s' % (args.out, old, config))
    else:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=1)

    if args.train_csv:
        # 打包檔模式：切分、交集檢查與每人局池已在打包時完成
        tr_df = pd.read_csv(args.train_csv, dtype={'player_id': str})
        nd_df = pd.read_csv(args.newdev_games_csv, dtype={'player_id': str})
        val_df = nd_df
    else:
        val_df, tr_df = load_split(args.train_dir)
        nd_df = newdev_subset(val_df, args.newdev_csv, args.newdev_games)
    if args.max_train_players:
        groups = tr_df.groupby(['rank', 'player_id'])
        step = groups.ngroups // args.max_train_players + 1
        tr_df = tr_df[groups.ngroup() % step == 0].reset_index(drop=True)
    tr_keys = set(tr_df['rank'].astype(str) + '/' + tr_df['player_id'])
    ev_keys = set(val_df['rank'].astype(str) + '/' + val_df['player_id'])
    assert not tr_keys & ev_keys, '訓練玩家和 val 切分重疊'
    store, nd_store = GameStore(tr_df), GameStore(nd_df)
    del tr_df, val_df
    print('train %d 局、%d 位；newdev 子集 %d 局' % (len(store), store.player.max() + 1, len(nd_store)), flush=True)

    model = build_model(args.checkpoint, args.train_blocks).to(device)
    groups = [{'params': model.trunk_params(), 'lr': args.lr * args.trunk_lr_ratio},
              {'params': list(model.head.parameters()), 'lr': args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    for g in opt.param_groups:
        g['base_lr'] = g['lr']
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.train_dtype == 'fp16')
    lists = [epoch_games(store.player, args.cap, args.seed, e) for e in range(args.epochs)]
    steps_per_epoch = [math.ceil(len(x) / args.games_per_batch) for x in lists]
    total = sum(steps_per_epoch)
    state = {'epoch': 0, 'step': 0, 'global_step': 0}
    last = os.path.join(args.out, 'last.pt')
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        state = ck['state']
        print('接續 epoch %d step %d' % (state['epoch'], state['step']), flush=True)

    def save_last():
        tmp = last + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'state': state}, tmp)
        os.replace(tmp, last)

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))

    log_path = os.path.join(args.out, 'history.jsonl')
    t_start = time.time()
    while state['epoch'] < args.epochs:
        e = state['epoch']
        idx = lists[e]
        weight = torch.from_numpy(class_weights(store.rank_idx[idx])).to(device)
        ds = GameDataset(store, idx[state['step'] * args.games_per_batch:], args.train_moves, seed=args.seed * 1000 + e)
        model.train()
        t0, run_loss, run_n = time.time(), 0.0, 0
        for b in loader(ds, args.games_per_batch, args.workers):
            if b is None:
                state['step'] += 1
                continue
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * lr_scale(state['global_step'])
            with autocast(device, args.train_dtype):
                z = model(b['spatial'].to(device, non_blocking=True), b['global'].to(device, non_blocking=True))
            logits = game_mean(z.float(), b['game'].to(device), len(b['label']))
            loss = torch.nn.functional.cross_entropy(logits, b['label'].to(device), weight=weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if not torch.isfinite(loss):
                raise SystemExit('loss 不是有限值（epoch %d step %d）' % (e, state['step']))
            state['step'] += 1
            state['global_step'] += 1
            run_loss += loss.item()
            run_n += 1
            if state['step'] % args.log_every == 0:
                el = time.time() - t0
                print('epoch %d step %d/%d loss %.4f %.1f 局/秒' % (e, state['step'], steps_per_epoch[e], run_loss / run_n,
                                                               run_n * args.games_per_batch / el), flush=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'epoch': e, 'step': state['step'], 'loss': run_loss / run_n}) + '\n')
                run_loss, run_n, t0 = 0.0, 0, time.time()
            if state['step'] % args.save_every == 0:
                save_last()
            if args.max_steps and state['global_step'] >= args.max_steps:
                save_last()
                print('到達 --max-steps，停止', flush=True)
                return
            if args.time_limit and time.time() - t_start > args.time_limit * 3600:
                save_last()
                print('到達 --time-limit，存 last.pt 後停止，重跑同一指令接續', flush=True)
                return
        # epoch 結束：newdev 評估與存檔
        nd_ds = GameDataset(nd_store, np.arange(len(nd_store)), args.eval_moves)
        logp, npos = predict(model, loader(nd_ds, args.games_per_batch, args.workers), device, args.infer_dtype, len(nd_store))
        ok = npos > 0
        nll = level_equal_nll(logp[ok], nd_store.rank_idx[ok])
        acc = float((logp[ok].argmax(1) == nd_store.rank_idx[ok]).mean())
        print('epoch %d newdev 各級等權 log loss %.4f、正確率 %.4f（%d 局，%d 局失敗）' % (e, nll, acc, ok.sum(), (~ok).sum()), flush=True)
        np.savez_compressed(os.path.join(args.out, 'newdev-epoch-%d.npz' % e), key=nd_store.key, logp=logp, npos=npos,
                            rank_idx=nd_store.rank_idx)
        torch.save(model.state_dict(), os.path.join(args.out, 'epoch-%d.pt' % e))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'epoch': e, 'newdev_nll': nll, 'newdev_acc': acc}) + '\n')
        state = {'epoch': e + 1, 'step': 0, 'global_step': state['global_step']}
        save_last()


def question_store(path, csv_glob):
    """題目檔全部題目用到的局（去重、排序），GameStore。棋譜從 csv_glob 的 CSV 取。"""
    q = np.load(path)
    keys = sorted({k for t in ('SC', 'MX') for k in q[t + '_keys'].ravel() if k})
    paths = sorted(glob.glob(csv_glob))
    df = pd.concat([pd.read_csv(p, dtype={'player_id': str}) for p in paths], ignore_index=True)
    df.index = df['rank'].astype(str) + '_' + df['game_id'].astype(str)
    missing = [k for k in keys if k not in df.index]
    if missing:
        raise SystemExit('找不到 %d 局，例如 %s' % (len(missing), missing[:3]))
    return GameStore(df.loc[keys].reset_index(drop=True))


def test_store(test_dir):
    """測試集全部局，GameStore；key 是 analyze_humansl.load_test_games 的 key，rank_idx 為 0（未知）。"""
    from aicup.analyze_humansl import load_test_games
    games = load_test_games(test_dir)
    df = pd.DataFrame({'sgf_content': [g['sgf'] for g in games], 'rank': 'D',
                       'player_id': [g['player_id'] for g in games], 'color': [g['color'] for g in games],
                       'game_id': [g['key'] for g in games]})
    store = GameStore(df)
    store.key = np.array([g['key'] for g in games])
    return store


def infer(args):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.test:
        store = test_store(args.test)
    else:
        store = question_store(args.questions, args.games_csv or os.path.join(args.train_dir, 'train_*.csv'))
    model = build_model(args.checkpoint, args.train_blocks).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    ds = GameDataset(store, np.arange(len(store)), args.eval_moves)
    t0 = time.time()
    logp, npos = predict(model, loader(ds, args.games_per_batch, args.workers), device, args.infer_dtype, len(store))
    np.savez_compressed(args.out, key=store.key, logp=logp, npos=npos, rank_idx=store.rank_idx)
    print('%d 局，%d 局失敗，%d 個局面，%.0f 秒' % (len(store), (npos == 0).sum(), npos.sum(), time.time() - t0))


def dtype_check(args):
    """同一批局面上比較 fp32 與 fp16、bf16：humanv0 policy logit（所有合法點）與分類 logit 的最大絕對差、argmax 一致率。"""
    import torch
    from aicup import katago_torch as kt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.newdev_games_csv:
        sub = pd.read_csv(args.newdev_games_csv, dtype={'player_id': str}).head(args.games)
    else:
        sub = newdev_subset(load_split(args.train_dir)[0], args.newdev_csv, 1).head(args.games)
    sp, gl = [], []
    for _, r in sub.iterrows():
        out = encode_sample(r['sgf_content'], r['color'], 8)
        if out is not None:
            sp.append(out[0])
            gl.append(out[1])
    sp, gl = torch.from_numpy(np.concatenate(sp)[:64]).to(device), torch.from_numpy(np.concatenate(gl)[:64]).to(device)
    model = build_model(args.checkpoint, args.train_blocks).to(device).eval()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    meta = model.meta.expand(len(gl), -1)
    res = {}
    with torch.no_grad():
        for name in DTYPES:
            if name != 'fp32' and device.type != 'cuda':
                continue
            if name == 'bf16' and not torch.cuda.is_bf16_supported():
                continue
            with autocast(device, name):
                pol = kt.policy_logits(model.net, sp.float(), gl, meta).float()
                cls = model(sp, gl).float()
            res[name] = (pol, cls)
    base = res['fp32']
    for name, (pol, cls) in res.items():
        if name == 'fp32':
            continue
        for label, a, b in (('policy', pol, base[0]), ('分類', cls, base[1])):
            diff = (a - b).abs().max().item()
            agree = (a.argmax(1) == b.argmax(1)).float().mean().item()
            print('%s %s：最大絕對差 %.4f，argmax 一致率 %.4f，%s' % (name, label, diff, agree,
                  '通過' if diff <= 0.05 and agree >= 0.99 else '不通過'))
    if len(res) == 1:
        print('沒有 CUDA，只跑了 fp32：%d 個局面，分類 logit 範圍 %.3f～%.3f' % (len(sp), base[1].min(), base[1].max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=['train', 'infer', 'dtype'])
    ap.add_argument('--train-dir', default='training-v1.1/training')
    ap.add_argument('--newdev-csv', default=NEWDEV_CSV)
    ap.add_argument('--newdev-games', type=int, default=NEWDEV_GAMES)
    ap.add_argument('--train-csv', help='打包檔的訓練局 CSV，給了就不讀 --train-dir')
    ap.add_argument('--newdev-games-csv', help='打包檔的 newdev 子集 CSV，和 --train-csv 一起用')
    ap.add_argument('--games-csv', help='infer --questions 的棋譜 CSV（可用萬用字元），預設 --train-dir 的 train_*.csv')
    ap.add_argument('--katago-python', help='KataGo 原始碼的 python/ 目錄，缺省時讀環境變數 KATAGO_PYTHON')
    ap.add_argument('--checkpoint', help='humanv0 checkpoint')
    ap.add_argument('--out')
    ap.add_argument('--ckpt', help='infer、dtype 用的分類模型權重（epoch-<n>.pt）')
    ap.add_argument('--questions', help='infer：含 SC_keys、MX_keys 的題目 npz')
    ap.add_argument('--test', help='infer：測試集目錄')
    ap.add_argument('--cap', type=int, default=20, help='每個 epoch 每位玩家最多幾局')
    ap.add_argument('--train-moves', type=int, default=8)
    ap.add_argument('--eval-moves', type=int, default=48)
    ap.add_argument('--train-blocks', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--games-per-batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-4, help='頭的學習率，主幹乘 --trunk-lr-ratio')
    ap.add_argument('--trunk-lr-ratio', type=float, default=0.1)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--train-dtype', choices=DTYPES, default='fp16')
    ap.add_argument('--infer-dtype', choices=DTYPES, default='fp32')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--save-every', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=0, help='冒煙測試用，到達後存 last.pt 停止')
    ap.add_argument('--max-train-players', type=int, default=0, help='冒煙測試用，約只留這麼多位訓練玩家')
    ap.add_argument('--time-limit', type=float, default=0, help='小時，到達後存 last.pt 停止')
    ap.add_argument('--games', type=int, default=8, help='dtype 用的局數')
    args = ap.parse_args()
    from aicup import katago_torch as kt
    if args.katago_python:
        os.environ['KATAGO_PYTHON'] = args.katago_python
    kt.katago()
    args.checkpoint = kt.ensure_humanv0(args.checkpoint or kt.CHECKPOINT)
    if args.mode == 'train':
        if not args.out:
            ap.error('train 要給 --out')
        if bool(args.train_csv) != bool(args.newdev_games_csv):
            ap.error('--train-csv 與 --newdev-games-csv 要一起給')
        train(args)
    elif args.mode == 'infer':
        if not args.ckpt or not args.out or bool(args.questions) == bool(args.test):
            ap.error('infer 要給 --ckpt、--out，以及 --questions 或 --test 其中一個')
        infer(args)
    else:
        dtype_check(args)


if __name__ == '__main__':
    main()
