"""棋力預測推論：讀測試 CSV，逐局預測等級機率，合併同一題的各局後選等級，輸出 submission_rank.csv

每局的機率存成 .npz 快取，換合併方式或選法時加 --probs 指向同一個檔案即可，不必重新推論。
快取記錄 checkpoint 的 SHA-256，傳入不同的 --ckpt 時拒絕執行。
public 與 private 兩份題目都要預測，串接成同一份提交檔，question_id 原樣保留。

用法（在 repo 根目錄）：
    !python -m aicup.predict_rank --ckpt rank-trained-models/<commit hash>/best_model.pth
    !python -m aicup.predict_rank --ckpt ... --agg mean --select argmax --temp 1 --probs game_probs.npz
"""
import argparse
import hashlib
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from aicup.scoring import LEVELS, aggregate_games, apply_temperature, pick_level

NUM_CLASSES = len(LEVELS)
SPLITS = ('public', 'private')

_parser = None


def sgf_to_features(sgf_content):
    """單局棋譜轉特徵，解析失敗回傳 None。頂層函式，multiprocessing 可直接 pickle。"""
    global _parser
    if _parser is None:
        from utils import SGFParseRankPrediction  # 子進程才載入 sgfmill
        _parser = SGFParseRankPrediction()
    feat, err = _parser.features_from_sgf_content(sgf_content)
    return None if err is not None else feat


def read_questions(test_dir):
    """讀 public 與 private 兩份測試 CSV，回傳 (question_id 清單, 逐局的 (題目 index, sgf))。

    空的 sgf_ 欄位（NaN 或空字串）略過。
    """
    qids, games = [], []
    for split in SPLITS:
        test = pd.read_csv(os.path.join(test_dir, 'rank_prediction_test_%s.csv' % split))
        sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
        for _, row in test.iterrows():
            q = len(qids)
            qids.append(str(row['question_id']))
            for c in sgf_cols:
                if pd.notna(row[c]) and str(row[c]).strip():
                    games.append((q, str(row[c])))
    return qids, games


def predict_games(model, sgfs, device, amp, batch_size=64, nproc=1):
    """逐局預測等級機率，形狀 (G, 10)；解析失敗的局整列填 NaN。"""
    import torch

    probs = np.full((len(sgfs), NUM_CLASSES), np.nan, np.float32)
    pool = Pool(nproc) if nproc > 1 else None
    feats = pool.imap(sgf_to_features, sgfs, chunksize=4) if pool else map(sgf_to_features, sgfs)
    t0 = time.time()
    buf, idx = [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(device).float()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        probs[idx] = torch.softmax(out.float(), 1).cpu().numpy()
        buf.clear()
        idx.clear()

    try:
        for i, feat in enumerate(feats):
            if feat is not None:
                buf.append(feat)
                idx.append(i)
            if len(buf) == batch_size:
                flush()
            if (i + 1) % 500 == 0:
                print('  %d/%d 局，%.0f 秒' % (i + 1, len(sgfs), time.time() - t0), flush=True)
        if buf:
            flush()
    finally:
        if pool:
            pool.close()
            pool.join()
    return probs


def decide(n_questions, game_q, game_probs, agg='log', select='pick', temp=1.0):
    """各局機率先以 temp 縮放，合併同一題後選等級，回傳每題的等級 index。

    整題的局都解析失敗時，回傳最弱等級。
    """
    game_q = np.asarray(game_q)
    preds = np.zeros(n_questions, np.int64)
    for q in range(n_questions):
        p = game_probs[game_q == q]
        p = p[~np.isnan(p).any(axis=1)]
        if len(p) == 0:
            continue
        dist = aggregate_games(apply_temperature(p, temp), agg)
        preds[q] = int(dist.argmax()) if select == 'argmax' else int(pick_level(dist))
    return preds


def load_model(ckpt, channels, blocks, device):
    import torch
    from network import GoRankResNet

    model = GoRankResNet(num_channels=channels, num_blocks=blocks, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    return model.to(device).eval()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', help='train_rank.py 存的 best_model.pth；有 --probs 且檔案存在時可省略')
    ap.add_argument('--test-dir', default='tests')
    ap.add_argument('--out', default='submission_rank.csv')
    ap.add_argument('--probs', default='game_probs.npz', help='每局機率的快取檔，存在就直接讀')
    ap.add_argument('--agg', choices=('mean', 'log'), default='log')
    ap.add_argument('--select', choices=('argmax', 'pick'), default='pick')
    ap.add_argument('--temp', type=float, default=3.0,
                    help='合併前的機率溫度；換 checkpoint 時用 eval_posthoc --calibrate 重新挑')
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--no-amp', action='store_true', help='關閉 bf16，改用純 fp32')
    args = ap.parse_args()

    qids, games = read_questions(args.test_dir)
    game_q = np.array([q for q, _ in games], np.int64)
    print('%d 題，%d 局' % (len(qids), len(games)), flush=True)

    if os.path.exists(args.probs):
        z = np.load(args.probs, allow_pickle=False)
        if 'ckpt_sha256' not in z.files:
            raise SystemExit('%s 沒有 checkpoint 紀錄，請刪除後重跑' % args.probs)
        if args.ckpt and file_sha256(args.ckpt) != str(z['ckpt_sha256']):
            raise SystemExit('%s 不是 %s 產生的，請刪除或改用其他 --probs' % (args.probs, args.ckpt))
        if len(z['probs']) != len(games) or not np.array_equal(z['game_q'], game_q):
            raise SystemExit('%s 與目前的測試資料對不上，請刪除後重跑' % args.probs)
        probs = z['probs']
        print('讀取快取 %s' % args.probs)
    else:
        if not args.ckpt:
            raise SystemExit('沒有快取檔 %s，需要 --ckpt' % args.probs)
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        amp = device.type == 'cuda' and not args.no_amp and torch.cuda.is_bf16_supported()
        model = load_model(args.ckpt, args.channels, args.blocks, device)
        probs = predict_games(model, [s for _, s in games], device, amp, args.batch_size, args.nproc)
        np.savez(args.probs, probs=probs, game_q=game_q, ckpt_sha256=file_sha256(args.ckpt))

    failed = int(np.isnan(probs).any(axis=1).sum())
    if failed:
        print('解析失敗 %d 局（%.2f%%）' % (failed, 100 * failed / max(len(games), 1)))
    preds = decide(len(qids), game_q, probs, args.agg, args.select, args.temp)
    sub = pd.DataFrame({'question_id': qids, 'pred_rank': [LEVELS[i] for i in preds]})
    sub.to_csv(args.out, index=False)
    print('寫出 %s（%d 列），合併 %s、選法 %s、溫度 %g' % (args.out, len(sub), args.agg, args.select, args.temp))
    print('預測分布:', sub['pred_rank'].value_counts().to_dict())


if __name__ == '__main__':
    main()
