"""中研院 strength estimator 推論：目標玩家每一手的 SE∞ score 與 SL 11 級棋力機率

模型是 rlglab/strength-estimator 的 TorchScript 權重（GPL-3.0），由 MODEL_URL 下載 go.tar.gz，
只解出 se_infty_go.pt 與 sl_go.pt。輸入編碼對應 MiniZero 的 GoEnv::getFeatures。
每局一行寫入 jsonl：目標玩家各手 score 的平均與標準差、SL 機率的平均，重跑會跳過已完成的局。

SE∞ score 會隨盤面方向偏移，所以對前 --symmetries 種對稱（aicup.symmetry 的 k = 0 起）各算一次，
score_mean、score_std 是依 k 排列的清單，取前 m 個就等於用 m 種對稱跑的結果。SL 只用原方向。
--side opponent 時改評目標玩家的對手那一方的手，color 欄位仍是目標玩家的顏色。

輸入二選一：
    --keys-from 讀另一份 jsonl 的 key（例如 analyze_humansl 的輸出），從 --csv 找出同樣的局
    --test-dir 讀競賽二的測試題，每題拆成單局，player_id 記為 question_id，rank_idx 記為 -1

用法（在 repo 根目錄）：
    python -m aicup.strength_se --keys-from humansl-analysis/val648-p3.jsonl --out humansl-analysis/val648-se.jsonl
    python -m aicup.strength_se --test-dir /content/tests --out humansl-analysis/test-se.jsonl
    python -m aicup.strength_se --keys-from humansl-analysis/val648-p3.jsonl --side opponent --symmetries 4 --out humansl-analysis/val648-se-opp.jsonl
"""
import argparse
import glob
import json
import os
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from aicup import symmetry
from aicup.analyze_humansl import load_test_games, parse_game
from aicup.color import parse_hd
from aicup.scoring import LEVELS

MODEL_URL = 'https://rlg.iis.sinica.edu.tw/papers/strength-estimator/assets/models/go.tar.gz'
ARCHIVE = 'go.tar.gz'
ARCHIVE_BYTES = 448_813_641
SE_MODEL = 'se_infty_go.pt'
SL_MODEL = 'sl_go.pt'
MODELS = [SE_MODEL, SL_MODEL]
MODEL_BYTES = {SE_MODEL: 97_033_576, SL_MODEL: 96_303_391}
BOARD = 19
HISTORY = 8  # 最近 8 個盤面
PLANES = 2 * HISTORY + 2
PASS_INDEX = BOARD * BOARD
SL_RANKS = 11  # 3-5k、1-2k、1d～9d
SIDES = ('target', 'opponent')


def ensure_models(model_dir, url=MODEL_URL):
    """確認 model_dir 內有大小正確的 MODELS，缺的話下載 go.tar.gz 並只解出 MODELS，回傳 {檔名: 路徑}。"""
    paths = {m: os.path.join(model_dir, m) for m in MODELS}
    if all(os.path.exists(p) and os.path.getsize(p) == MODEL_BYTES[m] for m, p in paths.items()):
        return paths
    os.makedirs(model_dir, exist_ok=True)
    archive = os.path.join(model_dir, ARCHIVE)
    if not os.path.exists(archive) or os.path.getsize(archive) != ARCHIVE_BYTES:
        print('下載 %s' % url, flush=True)
        part = archive + '.part'
        urllib.request.urlretrieve(url, part)
        if os.path.getsize(part) != ARCHIVE_BYTES:
            raise SystemExit('%s 大小 %d，預期 %d' % (url, os.path.getsize(part), ARCHIVE_BYTES))
        os.replace(part, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(model_dir, members=[tar.getmember(m) for m in MODELS], filter='data')
    for m, p in paths.items():
        if os.path.getsize(p) != MODEL_BYTES[m]:
            raise SystemExit('%s 大小 %d，預期 %d' % (p, os.path.getsize(p), MODEL_BYTES[m]))
    return paths


def board_states(moves):
    """逐手重播，回傳 (len(moves) + 1, 2, 19, 19) 的 uint8，第 p 個是下完前 p 手的盤面，平面 0 黑 1 白。

    盤面座標同 sgfmill 的 (row, col)。pass 與落在已有子的點不改變盤面，但仍佔一個狀態。
    """
    from sgfmill import boards
    board = boards.Board(BOARD)
    out = np.zeros((len(moves) + 1, 2, BOARD, BOARD), np.uint8)
    for i, (c, point) in enumerate(moves):
        if point is not None:
            try:
                board.play(point[0], point[1], c.lower())
            except ValueError:
                pass
        for color, (row, col) in board.list_occupied_points():
            out[i + 1, 0 if color == 'b' else 1, row, col] = 1
    return out


def encode(states, moves, p):
    """評第 p 手（1 起算）用的 18 平面：下完前 p 手的盤面，由輪到的一方（第 p 手的對手）看。

    平面 2k、2k+1 是 states[p - k] 中輪到的一方、對手的棋子，p - k < 0 時全 0；
    平面 16 在輪到黑方時全 1，平面 17 在輪到白方時全 1。p = 0 時輪到黑方。
    """
    turn = 1 if p > 0 and moves[p - 1][0] == 'B' else 0  # 0 = 輪到黑方
    f = np.zeros((PLANES, BOARD, BOARD), np.uint8)
    for k in range(min(HISTORY, p + 1)):
        f[2 * k] = states[p - k, turn]
        f[2 * k + 1] = states[p - k, 1 - turn]
    f[2 * HISTORY + turn] = 1
    return f


def target_positions(moves, color):
    """目標玩家每一手對應的 p：第 t 手（0 起算）由 p = t + 1 評。"""
    return [t + 1 for t, (c, _) in enumerate(moves) if c == color]


def policy_index(point):
    """落子在 policy 的 index，和 encode 的平面同為 row * 19 + col，最後一格是 pass。"""
    return PASS_INDEX if point is None else point[0] * BOARD + point[1]


def game_features(game):
    """一局目標玩家各手的編碼 (n, 18, 19, 19) uint8；ProcessPoolExecutor 的工作函數。"""
    moves = game['moves']
    states = board_states(moves)
    ps = target_positions(moves, game['color'])
    if not ps:
        return np.zeros((0, PLANES, BOARD, BOARD), np.uint8)
    return np.stack([encode(states, moves, p) for p in ps])


def summarize(scores, ranks):
    """一局的輸出欄位。scores 是 (n, 對稱數)，ranks 是 (n, 11)。

    score_mean、score_std 是每種對稱各手 score 的平均與母體標準差，sl_rank 是 SL 機率的平均；
    沒有目標手時三者皆為 None。
    """
    if len(scores) == 0:
        return {'n': 0, 'score_mean': None, 'score_std': None, 'sl_rank': None}
    return {'n': int(len(scores)), 'score_mean': [float(v) for v in np.mean(scores, axis=0)],
            'score_std': [float(v) for v in np.std(scores, axis=0)],
            'sl_rank': [round(float(v), 6) for v in np.mean(ranks, axis=0)]}


class Estimator:
    """SE∞ 與 SL 兩個 TorchScript 模型，run() 對編碼好的局面批次推論。"""

    def __init__(self, paths, device, dtype, symmetries=symmetry.N_SYMMETRIES):
        import torch
        if not 1 <= symmetries <= symmetry.N_SYMMETRIES:
            raise ValueError('symmetries 必須在 1 到 %d 之間' % symmetry.N_SYMMETRIES)
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.symmetries = symmetries
        self.se = torch.jit.load(paths[SE_MODEL], map_location=self.device).eval().to(self.dtype)
        self.sl = torch.jit.load(paths[SL_MODEL], map_location=self.device).eval().to(self.dtype)

    def run(self, x):
        """x 是 (n, 18, 19, 19) uint8，回傳 (score (n, 對稱數), SL 機率 (n, 11))，皆為 float32 numpy。

        score 的第 k 欄是 symmetry.apply(x, k) 的 SE∞ score，對稱在 device 上做。
        """
        torch = self.torch
        with torch.inference_mode():
            t = torch.from_numpy(x).to(self.device).to(self.dtype)
            score = torch.stack([self.se(symmetry.apply(t, k).contiguous())['score'].float().flatten()
                                 for k in range(self.symmetries)], dim=1).cpu().numpy()
            rank = self.sl(t)['rank'].float().cpu().numpy()
        return score, rank


def load_keyed_games(keys_path, csv_glob, limit=0):
    """keys_path 每行的 key 對應到 csv_glob 中的局，順序同 keys_path，欄位同 analyze_humansl.load_games。

    key 是 <rank>_<game_id>。有 key 找不到，或找到的 player_id、rank_idx、hd 和 keys_path 不一致時拒絕。
    hd 不一致通常是 CSV 不是 v1.1：沒有 HD 時 parse_game 不會對調讓子局的黑白。
    """
    with open(keys_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        rows = rows[:limit]
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise SystemExit('找不到 %s' % csv_glob)
    df = pd.concat([pd.read_csv(p, dtype={'player_id': str}) for p in paths], ignore_index=True)
    df.index = df['rank'].astype(str) + '_' + df['game_id'].astype(str)
    missing = [r['key'] for r in rows if r['key'] not in df.index]
    if missing:
        raise SystemExit('%s 找不到 %d 局，例如 %s' % (csv_glob, len(missing), ', '.join(missing[:3])))
    games = []
    for r in rows:
        g = df.loc[r['key']]
        game = {'key': r['key'], 'rank_idx': LEVELS.index(g['rank']), 'player_id': str(g['player_id']),
                'color': g['color'], 'sgf': g['sgf_content']}
        if ('player_id' in r and r['player_id'] != game['player_id']) or \
                ('rank_idx' in r and r['rank_idx'] != game['rank_idx']):
            raise SystemExit('%s 的 player_id 或 rank_idx 和 %s 不一致' % (r['key'], csv_glob))
        if 'hd' in r and r['hd'] != parse_hd(game['sgf']):
            raise SystemExit('%s 的 hd 是 %r，%s 是 %r，CSV 可能不是 v1.1'
                             % (r['key'], r['hd'], csv_glob, parse_hd(game['sgf'])))
        games.append(game)
    return games


def run_config(symmetries, dtype, side='target'):
    """每列都記錄的推論設定，接續時必須相同。"""
    return {'models': MODELS, 'symmetries': symmetries, 'dtype': dtype, 'side': side}


def side_color(color, side):
    """要評的那一方的顏色：target 是目標玩家，opponent 是對手。"""
    return color if side == 'target' else {'B': 'W', 'W': 'B'}[color]


def encode_job(game, side):
    """game_features 的輸入：解析後的手順，加上要評的那一方的顏色。"""
    return {'moves': game['parsed']['moves'], 'color': side_color(game['color'], side)}


def done_keys(path, config):
    """已完成的 key。檔案裡有任一列的設定和 config 不同時拒絕接續。

    最後一行沒有換行時是寫到一半中斷的列，從檔案截掉後再讀，其他壞掉的行照常丟錯。
    """
    if not os.path.exists(path):
        return set()
    with open(path, 'rb') as f:
        data = f.read()
    end = data.rfind(b'\n') + 1
    if end < len(data):
        print('%s 最後一行不完整，截掉 %d byte' % (path, len(data) - end), flush=True)
        with open(path, 'r+b') as f:
            f.truncate(end)
        data = data[:end]
    rows = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    for k, v in config.items():
        # 加入 side 之前的檔案沒有這個欄位，都是 target
        if any(r.get(k, 'target' if k == 'side' else None) != v for r in rows):
            raise SystemExit('%s 的 %s 不是 %s，請換一個 --out' % (path, k, v))
    return {r['key'] for r in rows}


def base_row(game, hd, config):
    return {'key': game['key'], 'player_id': game['player_id'], 'rank_idx': game['rank_idx'],
            'color': game['color'], 'hd': hd, **config}


def failed_row(game, config):
    """解析失敗的局也寫一列，n 為 0、parse_error 為 True，讓題目缺的局看得出來。"""
    row = base_row(game, parse_hd(game['sgf']), config)
    row.update(summarize(np.zeros((0, config['symmetries'])), np.zeros((0, SL_RANKS))))
    row['parse_error'] = True
    return row


def run_chunk(estimator, games, feats, batch, config):
    """一組局的特徵合併後分批推論，回傳每局的輸出列。"""
    sizes = [len(f) for f in feats]
    x = np.concatenate(feats) if sum(sizes) else np.zeros((0, PLANES, BOARD, BOARD), np.uint8)
    scores, ranks = [], []
    for i in range(0, len(x), batch):
        s, r = estimator.run(x[i:i + batch])
        scores.append(s)
        ranks.append(r)
    scores = np.concatenate(scores) if scores else np.zeros((0, estimator.symmetries), np.float32)
    ranks = np.concatenate(ranks) if ranks else np.zeros((0, SL_RANKS), np.float32)
    rows, start = [], 0
    for g, n in zip(games, sizes):
        row = base_row(g, g['parsed']['hd'], config)
        row.update(summarize(scores[start:start + n], ranks[start:start + n]))
        rows.append(row)
        start += n
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--keys-from', help='jsonl，每行有 key，只推論這些局，順序照檔案')
    src.add_argument('--test-dir', help='競賽二測試題所在目錄')
    ap.add_argument('--csv', default='training-v1.1/training/train_*.csv',
                    help='--keys-from 時查詢棋譜的 CSV，要用有 HD 的 v1.1')
    ap.add_argument('--model-dir', default='se-models', help='放 go.tar.gz 與解出的 .pt，缺檔時自動下載')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='最多推論幾局，用於冒煙測試')
    ap.add_argument('--batch', type=int, default=256, help='每次送進模型的局面數')
    ap.add_argument('--chunk', type=int, default=64, help='每次編碼並寫出的局數')
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 2), help='編碼用的 process 數')
    ap.add_argument('--device', default=None, help='預設有 CUDA 時用 cuda，否則 cpu')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
    ap.add_argument('--side', default='target', choices=SIDES, help='評目標玩家或對手的手')
    ap.add_argument('--symmetries', type=int, default=symmetry.N_SYMMETRIES, choices=range(1, 9),
                    help='SE∞ 用前幾種對稱，SL 只用原方向')
    args = ap.parse_args()

    if args.test_dir:
        games = load_test_games(args.test_dir, args.limit)
    else:
        games = load_keyed_games(args.keys_from, args.csv, args.limit)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    config = run_config(args.symmetries, args.dtype, args.side)
    done = done_keys(args.out, config)
    todo, bad = [], []
    for g in games:
        if g['key'] in done:
            continue
        try:
            g['parsed'] = parse_game(g['sgf'])
        except Exception:
            bad.append(failed_row(g, config))
            continue
        todo.append(g)
    print('待推論 %d 局，已完成 %d，解析失敗 %d' % (len(todo), len(done), len(bad)), flush=True)
    if bad:
        with open(args.out, 'a', encoding='utf-8') as f:
            for r in bad:
                f.write(json.dumps(r) + '\n')
    if not todo:
        return

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        torch.set_num_threads(os.cpu_count() or 1)
    estimator = Estimator(ensure_models(args.model_dir), device, args.dtype, args.symmetries)
    t0, n_pos = time.time(), 0
    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    with open(args.out, 'a', encoding='utf-8') as f, ProcessPoolExecutor(args.nproc) as pool:
        def submit(chunk):
            return [pool.submit(game_features, encode_job(g, args.side)) for g in chunk]
        # 推論目前這組時先編碼下一組，記憶體只放兩組
        pending = submit(chunks[0])
        for j, chunk in enumerate(chunks):
            feats = [fut.result() for fut in pending]
            if j + 1 < len(chunks):
                pending = submit(chunks[j + 1])
            rows = run_chunk(estimator, chunk, feats, args.batch, config)
            for r in rows:
                f.write(json.dumps(r) + '\n')
            f.flush()
            n_pos += sum(r['n'] for r in rows)
            dt = time.time() - t0
            print('  %d/%d 局，%d 個局面，每秒 %.0f 個局面' % (j * args.chunk + len(chunk), len(todo), n_pos, n_pos / dt),
                  flush=True)


if __name__ == '__main__':
    main()
