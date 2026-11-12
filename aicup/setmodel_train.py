"""set 模型：凍結 humanv0 主幹向量上的兩層 attention pooling，輸出每題的 10 級 log 機率

第一層：每局 30 個 768 維向量，標準化後過 Linear + GELU，attention pooling 成一個局向量。
第二層：每題的各局向量 attention pooling，接 Linear 輸出 10 級 logits。
兩層都只對有向量的局與手做 softmax，所以題內局的順序與題外的局都不影響輸出。

訓練只用 setmodel_trunk 抽出的 train 玩家：
    依玩家切出 --val-frac 當內部驗證，其餘玩家訓練，不碰開發組與封存組
    每個 epoch 每位訓練玩家重新出 --questions-per-player 題，題目局數從 SIZES 均勻抽，局不重複
    學習率與 epoch 數由驗證玩家的 pick 得分選，模型保留得分最高的那個 (學習率, epoch)
    驗證題目固定，每位驗證玩家 --questions-per-player 題

評估端：embed_games 把每局向量變成局向量，make_q_fn 回傳 stack_humansl.question_features 的 q_fn。

用法（在 repo 根目錄）：
    python -m aicup.setmodel_train --train-dir humansl-analysis/setmodel/trunk-train-350 --out humansl-analysis/setmodel/set-model.pt
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from aicup.eval_posthoc import SIZES, pick_question
from aicup.fit_humansl import question_points
from aicup.scoring import LEVELS, pick_level
from aicup.setmodel_trunk import DIM, PER_GAME

N_LEVELS = len(LEVELS)
STD_SAMPLE = 200_000  # 估計標準化的平均與標準差用的向量數
LRS = (1e-3, 3e-4)


def load_games(out_dir):
    """讀 setmodel_trunk 的輸出，回傳 dict：key、player_id、rank_idx（各 (n,)）、vec (n, PER_GAME, DIM) float16、mask (n, PER_GAME) bool。

    每局向量依序放在 vec 前 count 個位置，其餘補 0。沒有 part 檔或 count 超過 PER_GAME 時以 SystemExit 拒絕。
    """
    paths = sorted(glob.glob(os.path.join(out_dir, 'part-*.npz')))
    if not paths:
        raise SystemExit('%s 沒有 part 檔' % out_dir)
    keys, players, ranks, counts, padded = [], [], [], [], []
    for path in paths:
        with np.load(path) as z:
            count, vec = z['count'], z['vec']
            if (count > PER_GAME).any():
                raise SystemExit('%s 有局的向量數超過 %d' % (path, PER_GAME))
            keys.append(z['key'].astype(str))
            players.append(z['player_id'].astype(str))
            ranks.append(z['rank_idx'])
            counts.append(count)
            out = np.zeros((len(count), PER_GAME, DIM), np.float16)  # 逐 part 補齊，不保留串接後的原始陣列
            for i, (s, c) in enumerate(zip(np.cumsum(count) - count, count)):
                out[i, :c] = vec[s:s + c]
            padded.append(out)
    count = np.concatenate(counts)
    return {'key': np.concatenate(keys), 'player_id': np.concatenate(players), 'rank_idx': np.concatenate(ranks),
            'vec': np.concatenate(padded), 'mask': np.arange(PER_GAME)[None, :] < count[:, None]}


def align(games, keys):
    """依 keys 的順序取出 games 的 vec 與 mask，回傳 (vec, mask)。有 key 不在 games 內時丟 ValueError。"""
    pos = {k: i for i, k in enumerate(games['key'])}
    missing = [k for k in keys if str(k) not in pos]
    if missing:
        raise ValueError('%d 個 key 沒有抽取結果，例如 %s' % (len(missing), missing[0]))
    idx = np.array([pos[str(k)] for k in keys], np.int64)
    return games['vec'][idx], games['mask'][idx]


def fit_standardizer(vec, mask, rng, n=STD_SAMPLE):
    """從 mask 為真的向量中抽 n 個估計每維平均與標準差，回傳 (mean, std) float32。"""
    valid = np.argwhere(mask)
    pick = valid[rng.choice(len(valid), min(n, len(valid)), replace=False)]
    x = vec[pick[:, 0], pick[:, 1]].astype(np.float32)
    return x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)


def build_model(mean, std, d=128, dropout=0.2):
    import torch
    from torch import nn

    class SetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(np.asarray(mean, np.float32)))
            self.register_buffer('std', torch.from_numpy(np.asarray(std, np.float32)))
            self.enc = nn.Sequential(nn.Linear(DIM, d), nn.GELU())
            self.move_score = nn.Linear(d, 1)
            self.game_score = nn.Linear(d, 1)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d, N_LEVELS)

        @staticmethod
        def pool(h, score, mask):
            """(n, m, d) 以 score (n, m, 1) 在 mask (n, m) 為真的位置做 softmax 加權平均；整列都是 False 時回傳 0。"""
            s = score.squeeze(-1).masked_fill(~mask, float('-inf'))
            w = torch.softmax(s, dim=1).nan_to_num(0.0)
            return (w.unsqueeze(-1) * h).sum(dim=1)

        def embed(self, vec, mask):
            """vec (n, PER_GAME, DIM) float 與 mask (n, PER_GAME) bool，回傳局向量 (n, d)。"""
            h = self.drop(self.enc((vec - self.mean) / self.std))
            return self.pool(h, self.move_score(h), mask)

        def classify(self, games, mask):
            """games (q, G, d) 與 mask (q, G) bool，回傳 10 級 logits (q, 10)。"""
            return self.head(self.drop(self.pool(games, self.game_score(games), mask)))

    return SetModel()


def pad_questions(g, lens):
    """g 是各題局向量依序串接的 (sum(lens), d) tensor，回傳 (padded (q, max(lens), d), mask (q, max(lens)))。"""
    import torch
    lens = np.asarray(lens)
    rows = torch.from_numpy(np.repeat(np.arange(len(lens)), lens))
    cols = torch.from_numpy(np.concatenate([np.arange(n) for n in lens]))
    padded = torch.zeros(len(lens), int(lens.max()), g.shape[1], dtype=g.dtype)
    mask = torch.zeros(len(lens), int(lens.max()), dtype=torch.bool)
    padded[rows, cols] = g
    mask[rows, cols] = True
    return padded, mask


def question_logits(model, vec, mask, questions):
    """questions 是局 index 陣列的 list，回傳 (len(questions), 10) 的 logits tensor。每題至少要有一局。"""
    import torch
    flat = np.concatenate(questions)
    g = model.embed(torch.from_numpy(vec[flat]).float(), torch.from_numpy(mask[flat]))
    return model.classify(*pad_questions(g, [len(q) for q in questions]))


def player_questions(player_games, rng, per_player, sizes=SIZES):
    """player_games 是每位玩家的局 index 陣列 list，每位出 per_player 題，回傳 (questions, owner)，owner 是題所屬玩家的位置。"""
    questions, owner = [], []
    for p, idx in enumerate(player_games):
        for _ in range(per_player):
            q = pick_question(idx, sizes, rng)
            if q is not None:
                questions.append(q)
                owner.append(p)
    return questions, np.array(owner, np.int64)


def split_players(player_id, rank_idx, val_frac, rng):
    """依玩家切分，回傳 (train 玩家的局 index list, val 玩家的局 index list)，每位玩家一個遞增的 index 陣列。

    題目標籤取玩家第一局的等級，所以同一位玩家的局跨多個等級時丟 ValueError。
    """
    order, rank_of = {}, {}
    for i, (p, r) in enumerate(zip(player_id, rank_idx)):
        if rank_of.setdefault(p, r) != r:
            raise ValueError('玩家 %s 的局跨多個等級' % p)
        order.setdefault(p, []).append(i)
    ids = sorted(order)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = {ids[i] for i in perm[:n_val]}
    train = [np.array(order[p]) for p in ids if p not in val_ids]
    val = [np.array(order[p]) for p in ids if p in val_ids]
    return train, val


def evaluate(model, vec, mask, rank_idx, val_games, val_questions, owner, batch=256):
    """驗證題的 (pick 得分, log loss)。"""
    import torch
    model.eval()
    labels = np.array([rank_idx[val_games[o][0]] for o in owner])
    logits = []
    with torch.no_grad():
        for s in range(0, len(val_questions), batch):
            logits.append(question_logits(model, vec, mask, val_questions[s:s + batch]))
    logp = torch.log_softmax(torch.cat(logits), dim=1).numpy()
    loss = float(-logp[np.arange(len(labels)), labels].mean())
    score = float(question_points(labels, pick_level(np.exp(logp))).mean())
    return score, loss


def train_one(vec, mask, rank_idx, train_games, val_games, val_q, val_owner, lr, epochs, args, mean, std, rng, log):
    """用固定學習率訓練 epochs 個 epoch，回傳 (最佳 epoch 的 state_dict, 最佳得分, 最佳 epoch, 每個 epoch 的紀錄)。"""
    import torch
    torch.manual_seed(args.seed)
    model = build_model(mean, std, args.dim, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    best, history = (None, -1.0, 0), []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        questions, owner = player_questions(train_games, rng, args.questions_per_player)
        labels = np.array([rank_idx[train_games[o][0]] for o in owner])
        order = rng.permutation(len(questions))
        total = 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            logits = question_logits(model, vec, mask, [questions[i] for i in b])
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        score, val_loss = evaluate(model, vec, mask, rank_idx, val_games, val_q, val_owner)
        row = {'lr': lr, 'epoch': epoch, 'train_loss': round(total / len(order), 4), 'val_loss': round(val_loss, 4),
               'val_score': round(score, 4), 'seconds': round(time.time() - t0, 1)}
        history.append(row)
        log(row)
        if score > best[1]:
            best = ({k: v.clone() for k, v in model.state_dict().items()}, score, epoch)
    return best[0], best[1], best[2], history


def embed_games(model, vec, mask, batch=512):
    """每局的手向量 pooling 成局向量，回傳 (n, d) float32。沒有任何有效向量的局為 0。"""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(vec), batch):
            out.append(model.embed(torch.from_numpy(vec[s:s + batch]).float(), torch.from_numpy(mask[s:s + batch])).numpy())
    return np.concatenate(out)


def make_q_fn(model, game_vecs, has_vec, prior_logp):
    """stack_humansl.question_features 的 q_fn：每題用有向量的局算 10 級 log 機率，沒有任何局有向量時用 prior_logp。

    game_vecs 是 embed_games 的輸出，has_vec 是每局是否有向量。每題只由自己的局決定。
    """
    import torch
    model.eval()

    def q_fn(questions, batch=256):
        out = np.tile(np.asarray(prior_logp, np.float64), (len(questions), 1))
        kept = [(i, np.asarray(q)[has_vec[np.asarray(q)]]) for i, q in enumerate(questions)]
        kept = [(i, q) for i, q in kept if len(q)]
        with torch.no_grad():
            for s in range(0, len(kept), batch):
                part = kept[s:s + batch]
                g = torch.from_numpy(np.asarray(game_vecs[np.concatenate([q for _, q in part])], np.float32))
                logp = torch.log_softmax(model.classify(*pad_questions(g, [len(q) for _, q in part])), dim=1).numpy()
                for r, (i, _) in enumerate(part):
                    out[i] = logp[r]
        return out

    return q_fn


def save_model(path, state, mean, std, prior_logp, args, summary):
    import torch
    torch.save({'state': state, 'mean': mean, 'std': std, 'prior_logp': prior_logp,
                'config': {'dim': args.dim, 'dropout': args.dropout}, 'summary': summary}, path)


def load_model(path):
    """回傳 (model, prior_logp)，model 為 eval 模式。"""
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt['mean'], ckpt['std'], ckpt['config']['dim'], ckpt['config']['dropout'])
    model.load_state_dict(ckpt['state'])
    return model.eval(), ckpt['prior_logp']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train-dir', required=True, help='setmodel_trunk extract --set train 的輸出目錄')
    ap.add_argument('--out', required=True, help='輸出的 checkpoint 路徑，旁邊另存同名 .json 記錄驗證曲線')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lrs', type=lambda t: tuple(float(v) for v in t.split(',')), default=LRS)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--batch', type=int, default=64, help='每個 batch 的題數')
    ap.add_argument('--questions-per-player', type=int, default=20)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-games', type=int, help='冒煙測試：只用前 N 局')
    args = ap.parse_args()

    games = load_games(args.train_dir)
    if args.max_games:
        games = {k: v[:args.max_games] for k, v in games.items()}
    has_vec = games['mask'].any(axis=1)
    print('%d 局，%d 局有向量，%d 位玩家' % (len(has_vec), has_vec.sum(), len(set(games['player_id']))), flush=True)
    rng = np.random.default_rng(args.seed)
    ids = np.where(has_vec)[0]
    train_p, val_p = split_players(games['player_id'][ids], games['rank_idx'][ids], args.val_frac, rng)
    train_games = [ids[p] for p in train_p]
    val_games = [ids[p] for p in val_p]
    print('train %d 位，val %d 位' % (len(train_games), len(val_games)), flush=True)
    vec, mask, rank_idx = games['vec'], games['mask'], games['rank_idx']
    train_idx = np.concatenate(train_games)
    mean, std = fit_standardizer(vec[train_idx], mask[train_idx], rng)  # 只用訓練玩家，驗證玩家不影響標準化
    val_q, val_owner = player_questions(val_games, np.random.default_rng([args.seed, 1]), args.questions_per_player)
    train_labels = np.array([rank_idx[g[0]] for g in train_games])
    prior = np.bincount(train_labels, minlength=N_LEVELS).astype(float) + 1.0
    prior_logp = np.log(prior / prior.sum())

    best, runs = None, []
    for lr in args.lrs:
        state, score, epoch, history = train_one(vec, mask, rank_idx, train_games, val_games, val_q, val_owner, lr,
                                                 args.epochs, args, mean, std, np.random.default_rng([args.seed, 2]),
                                                 lambda row: print(json.dumps(row), flush=True))
        runs += history
        if best is None or score > best[1]:
            best = (state, score, lr, epoch)
    summary = {'selected_val_score': best[1], 'lr': best[2], 'epoch': best[3], 'n_train_players': len(train_games),
               'n_val_players': len(val_games), 'args': vars(args)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_model(args.out, best[0], mean, std, prior_logp, args, summary)
    with open(os.path.splitext(args.out)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'history': runs}, f, indent=1)
    print('選定 lr %g、epoch %d，驗證 pick 得分 %.4f' % (best[2], best[3], best[1]))


if __name__ == '__main__':
    main()
