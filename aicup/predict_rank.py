"""棋力預測推論：讀測試 CSV，逐局預測等級機率，合併同一題的各局後選等級，輸出 submission_rank.csv

每局的機率存成 .npz 快取，換合併方式或選法時加 --probs 指向同一個檔案即可，不必重新推論。
快取記錄 checkpoint 的 SHA-256，傳入不同的 --ckpt 時拒絕執行。
顏色設定讀 checkpoint 目錄的 train_config.json，沒有這個檔案時視為沒有顏色資訊。
public 與 private 兩份題目都要預測，串接成同一份提交檔，question_id 原樣保留。

用法（在 repo 根目錄）：
    !python -m aicup.predict_rank --ckpt rank-trained-models/<commit hash>/best_model.pth
    !python -m aicup.predict_rank --ckpt ... --agg mean --select argmax --temp 1 --probs game_probs.npz
    !python -m aicup.predict_rank --ckpt ... --tta --probs game_probs_tta.npz
"""
import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from aicup import color
from aicup.extract_features import COLOR_TO_ID
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
    """讀 public 與 private 兩份測試 CSV，回傳 (question_id 清單, 逐局的 (題目 index, sgf, color))。

    空的 sgf_ 欄位（NaN 或空字串）略過，沒有對應的 color_ 欄位時 color 為 None。
    """
    qids, games = [], []
    for split in SPLITS:
        test = pd.read_csv(os.path.join(test_dir, 'rank_prediction_test_%s.csv' % split))
        sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
        for _, row in test.iterrows():
            q = len(qids)
            qids.append(str(row['question_id']))
            for sgf_col in sgf_cols:
                if pd.notna(row[sgf_col]) and str(row[sgf_col]).strip():
                    game_color = row.get('color_' + sgf_col[len('sgf_'):])
                    games.append((q, str(row[sgf_col]), str(game_color) if pd.notna(game_color) else None))
    return qids, games


def game_colors(games, raw_color):
    """回傳逐局的特徵顏色；raw_color 時直接用 color_ 欄位，缺 HD 的局無法修正。"""
    if any(c is None for _, _, c in games):
        raise SystemExit('模型需要顏色資訊，但測試資料缺 color_ 欄位')
    colors = np.array([COLOR_TO_ID[c] for _, _, c in games], np.uint8)
    if raw_color:
        return colors
    hd = [color.parse_hd(s) for _, s, _ in games]
    missing = sum(h is None for h in hd)
    if missing:
        print('%d/%d 局沒有 HD，無法修正讓子局' % (missing, len(games)), flush=True)
    return color.feature_color(colors, [h or 0 for h in hd])


def read_train_config(ckpt):
    """回傳 checkpoint 目錄 train_config.json 的內容，沒有這個檔案時回傳無顏色設定。"""
    path = os.path.join(os.path.dirname(ckpt), 'train_config.json')
    if not os.path.exists(path):
        print('警告：找不到 %s，視為沒有顏色資訊；--color swap 的模型必須連同這個檔案複製' % path, flush=True)
        return {'color': 'none', 'raw_color': False}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def predict_games(model, sgfs, device, amp, batch_size=64, nproc=1, colors=None, color_mode='none', tta=False):
    """逐局預測等級機率，形狀 (G, 10)；解析失敗的局整列填 NaN。

    colors 是逐局的特徵顏色，color_mode 不是 none 時必須提供；tta 時平均 8 種對稱方向。
    """
    import torch

    from aicup.train_rank import forward_probs

    probs = np.full((len(sgfs), NUM_CLASSES), np.nan, np.float32)
    pool = Pool(nproc) if nproc > 1 else None
    feats = pool.imap(sgf_to_features, sgfs, chunksize=4) if pool else map(sgf_to_features, sgfs)
    t0 = time.time()
    buf, idx = [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(device).float()
        if color_mode != 'none':
            c = torch.from_numpy(colors[idx].astype(np.int64)).to(device)
            x = color.apply_color(x, c, color_mode)
        with torch.no_grad():
            probs[idx] = forward_probs(model, x, amp, tta).cpu().numpy()
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


def load_model(ckpt, channels, blocks, device, in_channels=500):
    import torch
    from network import GoRankResNet

    model = GoRankResNet(in_channels=in_channels, num_channels=channels, num_blocks=blocks,
                         num_classes=NUM_CLASSES)
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
    ap.add_argument('--tta', action='store_true', help='每局平均 8 種對稱方向的機率，推論時間約 8 倍')
    args = ap.parse_args()

    qids, games = read_questions(args.test_dir)
    game_q = np.array([q for q, _, _ in games], np.int64)
    print('%d 題，%d 局' % (len(qids), len(games)), flush=True)

    if os.path.exists(args.probs):
        z = np.load(args.probs, allow_pickle=False)
        if 'ckpt_sha256' not in z.files:
            raise SystemExit('%s 沒有 checkpoint 紀錄，請刪除後重跑' % args.probs)
        if args.ckpt and file_sha256(args.ckpt) != str(z['ckpt_sha256']):
            raise SystemExit('%s 不是 %s 產生的，請刪除或改用其他 --probs' % (args.probs, args.ckpt))
        if len(z['probs']) != len(games) or not np.array_equal(z['game_q'], game_q):
            raise SystemExit('%s 與目前的測試資料對不上，請刪除後重跑' % args.probs)
        cached_tta = bool(z['tta']) if 'tta' in z.files else False  # 舊快取沒有這個欄位，都是未開 TTA
        if cached_tta != args.tta:
            raise SystemExit('%s 的 TTA 設定是 %s，和這次不同，請刪除或改用其他 --probs' % (args.probs, cached_tta))
        probs = z['probs']
        print('讀取快取 %s' % args.probs)
    else:
        if not args.ckpt:
            raise SystemExit('沒有快取檔 %s，需要 --ckpt' % args.probs)
        import torch
        from aicup.train_rank import use_amp
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        amp = device.type == 'cuda' and use_amp(device, args.no_amp)
        cfg = read_train_config(args.ckpt)
        print('顏色設定', cfg, flush=True)
        colors = None if cfg['color'] == 'none' else game_colors(games, cfg['raw_color'])
        model = load_model(args.ckpt, args.channels, args.blocks, device, color.in_channels(cfg['color']))
        probs = predict_games(model, [s for _, s, _ in games], device, amp, args.batch_size, args.nproc,
                              colors, cfg['color'], tta=args.tta)
        np.savez(args.probs, probs=probs, game_q=game_q, ckpt_sha256=file_sha256(args.ckpt), tta=args.tta)

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
