"""端對端玩家模型：用玩家層級的 loss 微調 GoRankResNet。

一題由同一位玩家的 5、10、15 或 20 局組成，題目的 logit 是各局 logit 的平均，loss 是每題 CrossEntropy 的平均。
模型參數與 GoRankResNet 完全相同，best_model.pth 只存 state_dict，推論與評估程式不必修改。
BN 固定 running mean/var（eval 模式），affine 參數照常訓練；dropout 維持 train 模式。
每個 epoch 每局最多出現一次，訓練固定跑完 --epochs 個 epoch，最終模型是最後一個 epoch，不依 val 分數挑選。
每個 epoch 的 val 指標寫進 history.csv，只當紀錄。
累積 --accum 個 batch 才更新一次，loss 是視窗內每題 CrossEntropy 的總和除以視窗內的總題數；lr、warmup、--max-steps 都以更新步數計。
從 train 玩家固定切出 monitor 玩家，不參與更新。monitor.csv 每個更新步一列，記錄 lr、train loss、梯度 L2 norm、視窗內不重複玩家數；
訓練前（step 0）、每 --monitor-every 步與每個 epoch 結束的那一列另有 monitor loss 與預測最多那一級的佔比，只當紀錄。
每個 epoch 存 last.pt，重跑自動接續；設定不同時拒絕執行。

用法（在 repo 根目錄）：
    python -m aicup.train_player --feat-dir rank-features-v180000 --init rank-trained-models/<hash>/best_model.pth \\
        --ckpt-dir rank-trained-models/<hash>-e2e --epochs 5 --lr 1e-4 --warmup-steps 50 --accum 4 --schedule cosine --augment
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from aicup import packed, symmetry
from aicup.scoring import LEVELS
from aicup.train_rank import (load_to, log_run, lr_at, pick_data_device, save_atomic, save_val_probs,
                              split_bytes, use_amp, validate)
from network import GoRankResNet

NUM_CLASSES = len(LEVELS)
SIZES = (5, 10, 15, 20)  # 每題局數


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--feat-dir', default='rank-features-v180000')
    ap.add_argument('--init', required=True, help='初始化用的 GoRankResNet state_dict')
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256, help='一個 batch 的局數上限，也是 val 的 batch 大小')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--schedule', choices=('constant', 'cosine'), default='cosine',
                    help='cosine 在 --warmup-steps 內線性升到 --lr，之後依總步數降到 0')
    ap.add_argument('--warmup-steps', type=int, default=50, help='以更新步數計')
    ap.add_argument('--accum', type=int, default=4, help='累積幾個 batch 才更新一次')
    ap.add_argument('--monitor-per-rank', type=int, default=20, help='每級切出幾位 monitor 玩家，0 為不切')
    ap.add_argument('--monitor-min-games', type=int, default=20, help='monitor 玩家至少要有幾局')
    ap.add_argument('--monitor-every', type=int, default=100, help='每幾個更新步算一次 monitor loss，訓練前與每個 epoch 結束也會算')
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto',
                    help='打包資料放哪裡；auto 在 VRAM 足夠時放 GPU')
    ap.add_argument('--no-amp', action='store_true', help='關閉 bf16，改用純 fp32')
    ap.add_argument('--augment', action='store_true', help='每個 batch 隨機套 8 種盤面對稱之一')
    ap.add_argument('--fresh', action='store_true', help='忽略既有 last.pt')
    ap.add_argument('--max-steps', type=int, default=0, help='每個 epoch 最多幾個更新步，0 為不限，用於冒煙測試')
    ap.add_argument('--log-every', type=int, default=200, help='每幾個更新步印一次訓練狀態')
    ap.set_defaults(color='none')  # train_rank.validate 讀取，端對端模型不用顏色資訊
    args = ap.parse_args()
    if args.accum < 1 or args.monitor_every < 1 or args.log_every < 1:
        ap.error('--accum、--monitor-every、--log-every 要 >= 1')
    return args


def game_embed(model, x):
    """GoRankResNet.forward 算到 dropout 之後、fc_2 之前，回傳 (B, 722)。"""
    out = F.relu(model.bn_in(model.conv_in(x)))  # 第一層
    out = model.layers(out)  # 殘差 block
    out = F.relu(model.bn_in2(model.conv_in2(out)))
    out = out.view(out.size(0), -1)  # 攤平
    out = F.relu(model.fc(out))
    return model.dropout(out)


def question_logits(model, x, qid, n_questions):
    """各局 embedding 依 qid 取平均後過 fc_2，回傳 (n_questions, 10)。

    qid 是長度 B 的 long tensor，值在 [0, n_questions)，每題至少一局。fc_2 是線性層，
    結果等於各局 logit 的平均。
    """
    emb = game_embed(model, x).float()
    pooled = torch.zeros(n_questions, emb.size(1), device=emb.device, dtype=emb.dtype).index_add_(0, qid, emb)
    pooled = pooled / torch.bincount(qid, minlength=n_questions).unsqueeze(1).to(emb.dtype)
    return model.fc_2(pooled)


def set_train_mode(model):
    """BN 用 eval 模式固定 running mean/var，其餘層 train 模式。validate 結束會把整個模型設回 train，之後要重設。"""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def make_epoch_questions(player_id, rng, sizes=SIZES):
    """一個 epoch 的題目，回傳局 index 陣列的 list，每局最多出現一次。

    每位玩家的局打亂後依序切題，每題局數從 sizes 中放得下的值均勻抽，剩下不到 min(sizes) 局就丟掉。
    局數少於 min(sizes) 的玩家沒有題目。題目的順序打亂，同一個 rng 狀態得到相同結果。
    """
    _, inverse = np.unique(np.asarray(player_id), return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    ends = np.cumsum(np.bincount(inverse))
    starts = ends - np.bincount(inverse)
    questions = []
    for s, e in zip(starts, ends):
        games = rng.permutation(order[s:e])
        pos = 0
        while True:
            fit = [n for n in sizes if n <= len(games) - pos]
            if not fit:
                break
            n = fit[int(rng.integers(len(fit)))]
            questions.append(games[pos:pos + n])
            pos += n
    return [questions[i] for i in rng.permutation(len(questions))]


def pack_batches(questions, max_games=256):
    """依序把整題放進 batch，局數總和不超過 max_games，一題不拆開。回傳 batch 的 list，每個 batch 是題目的 list。"""
    batches, current, used = [], [], 0
    for q in questions:
        if len(q) > max_games:
            raise ValueError('一題有 %d 局，超過 max_games %d' % (len(q), max_games))
        if used + len(q) > max_games:
            batches.append(current)
            current, used = [], 0
        current.append(q)
        used += len(q)
    if current:
        batches.append(current)
    return batches


def games_by_player(group_id):
    """{玩家: 該玩家的局 index 陣列}，玩家依排序。"""
    players, inverse = np.unique(np.asarray(group_id), return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    ends = np.cumsum(np.bincount(inverse, minlength=len(players)))
    starts = ends - np.bincount(inverse, minlength=len(players))
    return {p: order[a:b] for p, a, b in zip(players, starts, ends)}


def select_monitor_players(rank_idx, group_id, per_rank=20, min_games=20, seed=42):
    """每級抽 per_rank 位局數至少 min_games 的玩家，回傳排序後的 group id 陣列。

    rank_idx 與 group_id 都是每局一筆。某一級符合條件的玩家不足 per_rank 位時丟 ValueError。
    抽樣只由 seed 與等級決定。
    """
    by_player = games_by_player(group_id)
    rank_of = {p: int(rank_idx[idx[0]]) for p, idx in by_player.items()}
    chosen = []
    for rank in sorted(set(rank_of.values())):
        pool = [p for p, idx in by_player.items() if rank_of[p] == rank and len(idx) >= min_games]
        if len(pool) < per_rank:
            raise ValueError('等級 %d 只有 %d 位玩家有 %d 局以上，不足 %d 位' % (rank, len(pool), min_games, per_rank))
        pick = np.random.default_rng([seed, rank]).choice(len(pool), per_rank, replace=False)
        chosen += [pool[i] for i in pick]
    return np.array(sorted(chosen), dtype=str)


def monitor_questions(group_id, monitor, seed=42):
    """每位 monitor 玩家 4 題：同一個打亂順序的前 5、10、15、20 局。回傳 (題目的 list, 每題局數陣列)。"""
    by_player = games_by_player(group_id)
    rng = np.random.default_rng([seed, 10_000])  # 和每個 epoch 抽題的 rng key 錯開
    questions, sizes = [], []
    for p in monitor:
        order = rng.permutation(by_player[p])
        for n in SIZES:
            questions.append(order[:n])
            sizes.append(n)
    return questions, np.array(sizes)


def epoch_plans(group_id, args, games=None):
    """每個 epoch 的更新視窗計畫，只由 seed 與 epoch 決定，接續訓練時結果相同。

    視窗是 args.accum 個 batch，最後一個視窗可以不足。group_id 是參與訓練的局的玩家，games 是這些局在整份資料的 index
    （None 時就是 0..n-1）。--max-steps 時每個 epoch 只留前幾個視窗。
    """
    plans = []
    for epoch in range(args.epochs):
        rng = np.random.default_rng([args.seed, epoch])
        questions = make_epoch_questions(group_id, rng)
        if games is not None:
            questions = [games[q] for q in questions]
        batches = pack_batches(questions, args.batch_size)
        windows = [batches[i:i + args.accum] for i in range(0, len(batches), args.accum)]
        plans.append(windows[:args.max_steps] if args.max_steps else windows)
    return plans


def load_batch(x, y, batch, device):
    """解包一個 batch 的局，回傳 (xb, qid, 每題標籤)。"""
    games = np.concatenate(batch)
    qid = np.repeat(np.arange(len(batch)), [len(q) for q in batch])
    xb = packed.unpack_torch(x[torch.as_tensor(games, device=x.device)].to(device, non_blocking=True))
    first = torch.as_tensor([q[0] for q in batch], device=y.device)
    return xb, torch.as_tensor(qid, device=device), y[first].to(device, non_blocking=True)


def window_backward(model, x, y, window, device, amp, symmetries=None):
    """對一個視窗的每個 batch 反向傳播，梯度累積在 model 上，不更新也不清梯度。

    loss 是視窗內每題 CrossEntropy 的總和除以視窗內的總題數。symmetries 是每個 batch 的盤面對稱編號，None 為不轉。
    回傳 (每題 CE 總和, 答對題數, 題數)。
    """
    n_total = sum(len(b) for b in window)
    loss_sum, correct = 0.0, 0
    for i, batch in enumerate(window):
        xb, qid, yb = load_batch(x, y, batch, device)
        if symmetries is not None:
            xb = symmetry.apply(xb, symmetries[i]).contiguous()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = question_logits(model, xb, qid, len(batch))
        loss = F.cross_entropy(out.float(), yb, reduction='sum')
        (loss / n_total).backward()
        loss_sum += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum, correct, n_total


def monitor_losses(model, x, y, questions, sizes, args, device, amp):
    """eval 模式下算 monitor 題目的指標。

    回傳 {'monitor_loss': 每題 CrossEntropy 平均, 'monitor_loss_<局數>': 該局數的平均,
    'monitor_top_frac': 預測最多的那一級佔全部題目的比例}。結束時把模型設回 set_train_mode 的狀態。
    """
    model.eval()
    losses, preds = [], []
    with torch.no_grad():
        for batch in pack_batches(questions, args.batch_size):
            xb, qid, yb = load_batch(x, y, batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = question_logits(model, xb, qid, len(batch))
            losses.append(F.cross_entropy(out.float(), yb, reduction='none').cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
    set_train_mode(model)
    losses, preds = np.concatenate(losses), np.concatenate(preds)
    res = {'monitor_loss': float(losses.mean())}
    res.update({'monitor_loss_%d' % n: float(losses[sizes == n].mean()) for n in SIZES})
    res['monitor_top_frac'] = float(np.bincount(preds, minlength=NUM_CLASSES).max() / len(preds))
    return res


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = use_amp(device, args.no_amp)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dir = os.path.join(args.feat_dir, 'train')
    val_dir = os.path.join(args.feat_dir, 'val')
    data_device = pick_data_device(args, device, split_bytes(train_dir) + split_bytes(val_dir))
    print('載入資料到 %s' % data_device, flush=True)
    x_val, y_val, val_meta = load_to(val_dir, data_device)
    x_train, y_train, train_meta = load_to(train_dir, data_device)
    group_id = np.array(['%d|%s' % (r, p) for r, p in zip(train_meta['rank_idx'], train_meta['player_id'])])
    monitor = select_monitor_players(train_meta['rank_idx'], group_id, args.monitor_per_rank,
                                     args.monitor_min_games, args.seed)
    games = np.flatnonzero(~np.isin(group_id, monitor))
    plans = epoch_plans(group_id[games], args, games)
    total_steps = sum(len(p) for p in plans)
    mon_q, mon_sizes = monitor_questions(group_id, monitor, args.seed) if len(monitor) else ([], None)

    model = GoRankResNet(in_channels=packed.FEATURE_SHAPE[0], num_channels=args.channels,
                         num_blocks=args.blocks, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, 'last.pt')
    config = {k: getattr(args, k) for k in ('batch_size', 'lr', 'channels', 'blocks', 'seed', 'schedule',
                                            'warmup_steps', 'epochs', 'augment', 'init', 'accum', 'feat_dir',
                                            'max_steps', 'monitor_per_rank', 'monitor_min_games')}
    config['amp'] = amp
    start_epoch, history, monitor_rows = 0, [], []
    if os.path.exists(last_path) and not args.fresh:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt['config'] != config:
            raise SystemExit('last.pt 的設定與這次不同：\n  舊 %s\n  新 %s\n'
                             '要用新設定請加 --fresh 或換 --ckpt-dir' % (ckpt['config'], config))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, history, monitor_rows = ckpt['epoch'], ckpt['history'], ckpt['monitor']
        print('從 epoch %d 接續' % start_epoch)
    else:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))

    # 通過設定比對才寫，predict_rank 依此設定輸入
    with open(os.path.join(args.ckpt_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'color': 'none', 'raw_color': False}, f)

    n_games = sum(len(q) for w in plans[0] for b in w for q in b)
    print('device %s | data %s | amp %s | train %s / val %s | monitor %d 位 | 每個 epoch %s 局 %s 個更新步'
          % (device, data_device, amp, format(len(y_train), ','), format(len(y_val), ','), len(monitor),
             format(n_games, ','), format(len(plans[0]), ',')))
    print('設定', json.dumps(config, ensure_ascii=False))
    if start_epoch >= args.epochs:
        print('已完成 %d 個 epoch，要繼續訓練請加大 --epochs' % start_epoch)
        return

    log_run(args.ckpt_dir, start_epoch, device)
    done_steps = sum(len(p) for p in plans[:start_epoch])
    params = list(model.parameters())
    monitor_path = os.path.join(args.ckpt_dir, 'monitor.csv')
    if mon_q and start_epoch == 0:  # 訓練前的基準
        monitor_rows.append({'step': 0, 'epoch': 0,
                             **monitor_losses(model, x_train, y_train, mon_q, mon_sizes, args, device, amp)})
        print('  訓練前 monitor loss %.4f  預測最多一級佔 %.2f'
              % (monitor_rows[-1]['monitor_loss'], monitor_rows[-1]['monitor_top_frac']), flush=True)
    for epoch in range(start_epoch, args.epochs):
        set_train_mode(model)
        torch.manual_seed(args.seed + epoch)  # dropout 的亂數也隨 epoch 固定
        aug_gen = torch.Generator().manual_seed(args.seed + epoch)  # 對稱的選擇只由 seed 與 epoch 決定
        t0 = time.time()
        loss_sum, correct, total, steps = 0.0, 0, 0, 0
        for window in plans[epoch]:
            lr = lr_at(done_steps + steps, total_steps, args.lr, args.schedule, args.warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            ks = None
            if args.augment:
                ks = [int(torch.randint(symmetry.N_SYMMETRIES, (1,), generator=aug_gen)) for _ in window]
            optimizer.zero_grad(set_to_none=True)
            l, c, n = window_backward(model, x_train, y_train, window, device, amp, ks)
            norm = float(torch.nn.utils.clip_grad_norm_(params, float('inf')))  # 只量測，不裁切
            optimizer.step()

            loss_sum += l
            correct += c
            total += n
            steps += 1
            step = done_steps + steps
            row = {'step': step, 'epoch': epoch + 1, 'lr': lr, 'train_loss': l / n, 'grad_norm': norm,
                   'players': len({group_id[q[0]] for b in window for q in b})}
            if mon_q and (step % args.monitor_every == 0 or steps == len(plans[epoch])):
                row.update(monitor_losses(model, x_train, y_train, mon_q, mon_sizes, args, device, amp))
            monitor_rows.append(row)
            if steps % args.log_every == 0:
                recent = monitor_rows[-args.log_every:]
                norms = [r['grad_norm'] for r in recent]
                print('  ep%d step %d  loss %.4f  acc %.2f%%  grad norm 平均 %.3f 最大 %.3f  視窗玩家 %.1f  %.2f 秒/步'
                      % (epoch + 1, steps, loss_sum / total, 100 * correct / total, np.mean(norms), np.max(norms),
                         np.mean([r['players'] for r in recent]), (time.time() - t0) / steps), flush=True)
        done_steps += steps
        if mon_q and steps:
            print('  monitor loss %.4f  預測最多一級佔 %.2f'
                  % (monitor_rows[-1]['monitor_loss'], monitor_rows[-1]['monitor_top_frac']), flush=True)
        pd.DataFrame(monitor_rows).to_csv(monitor_path, index=False)

        train_min = (time.time() - t0) / 60
        metrics, val_probs = validate(model, x_val, y_val, args, device, amp, criterion)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / max(total, 1),
               'train_acc': 100 * correct / max(total, 1), 'lr': lr, **metrics, 'steps': steps,
               'train_minutes': train_min, 'total_minutes': (time.time() - t0) / 60}
        history.append(row)
        print('Epoch %d/%d  train loss %.4f acc %.2f%% | val acc %.2f%%  score argmax %.4f  pick %.4f | %.1f 分'
              % (epoch + 1, args.epochs, row['train_loss'], row['train_acc'], row['val_acc'],
                 row['val_score_argmax'], row['val_score_pick'], row['total_minutes']), flush=True)

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'epoch_%d.pth' % (epoch + 1)))
        if epoch + 1 == args.epochs:  # 最終模型是最後一個 epoch
            save_val_probs(os.path.join(args.ckpt_dir, 'val_probs.npz'), val_probs, val_meta, epoch + 1)
            save_atomic(model.state_dict(), os.path.join(args.ckpt_dir, 'best_model.pth'))
            print('  寫出 best_model.pth 與 val_probs.npz')
        pd.DataFrame(history).to_csv(os.path.join(args.ckpt_dir, 'history.csv'), index=False)
        # last.pt 最後寫，確保它記錄的 epoch 之前的檔案都已存好
        save_atomic({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                     'history': history, 'monitor': monitor_rows, 'config': config}, last_path)

    print('完成。')


if __name__ == '__main__':
    main()
