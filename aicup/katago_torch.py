# Third-party notice
#
# This module imports and runs code from KataGo (https://github.com/lightvector/KataGo),
# tag v1.15.0, commit c560d38585b446bc93b0a62c77364bc8e7d266a4: python/board.py,
# python/features.py, python/sgfmetadata.py and python/model_pytorch.py. That code is not
# copied into this repository; it is loaded from a separate checkout (--katago-python).
# profile_row() and inverse_rank() are ports of makeBasicRankProfile() and getProfile()
# in cpp/neuralnet/sgfmetadata.cpp, and the policy masking in move_log_probs() follows
# NNEvaluator in cpp/neuralnet/nneval.cpp.
#
# The model weights are b18c384nbt-humanv0.ckpt, distributed by the KataGo project at
# https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.ckpt
#
# KataGo is licensed under the MIT License:
#
# Copyright 2019 David J Wu ("lightvector") and/or other authors of the content in this
# repository. (See 'CONTRIBUTORS' file for a list of authors as well as other indirect
# contributors).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""KataGo humanv0 的 PyTorch 載入、輸入編碼與實際落子機率

模型定義、輸入特徵與 metadata 編碼都用 KataGo python/ 目錄（tag v1.15.0，MIT）的程式，
位置依序取 --katago-python、環境變數 KATAGO_PYTHON、DEFAULT_KATAGO_PYTHON。checkpoint 是 b18c384nbt-humanv0.ckpt。
棋譜解析、規則與貼目和 analyze_humansl 的 Human SL 分析相同（japanese，KM 缺省 6.5，讓子局 0），
policy 只在合法點上做 softmax，對應 KataGo C++ 的 humanPolicy。

用法（在 repo 根目錄）：
    python -m aicup.katago_torch --verify humansl-analysis/val648-p3.jsonl --games 20
    python -m aicup.katago_torch --bench 50
"""
import argparse
import datetime
import importlib
import json
import os
import sys
import tempfile
import time

import numpy as np

from aicup import symmetry
from aicup.analyze_humansl import BOARD, PASS_INDEX, parse_game, policy_index, target_turns

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KATAGO_PYTHON = os.path.join(_ROOT, 'Note', 'katago', 'src', 'python')
CHECKPOINT = os.path.join(_ROOT, 'Note', 'katago', 'models', 'b18c384nbt-humanv0.ckpt')
INPUT_VERSION = 14  # humanv0 的 config version，決定 Features 的編碼版本
N_SPATIAL = 22
N_GLOBAL = 19
N_META = 192
N_POLICY = PASS_INDEX + 1
# KataGo GameState.RULES_JAPANESE，whiteKomi 由每局的 komi 取代
RULES_JAPANESE = {
    'koRule': 'KO_SIMPLE', 'scoringRule': 'SCORING_TERRITORY', 'taxRule': 'TAX_SEKI',
    'multiStoneSuicideLegal': False, 'hasButton': False, 'encorePhase': 0,
    'passWouldEndPhase': False, 'whiteKomi': 6.5, 'asymPowersOfTwo': 0.0,
}


def katago(path=None):
    """匯入 KataGo python/ 的模組，回傳 dict：board、features、sgfmetadata、model_pytorch。

    path 缺省時讀環境變數 KATAGO_PYTHON，spawn 的子 process 也會繼承，再缺省才用 DEFAULT_KATAGO_PYTHON。
    """
    path = os.path.abspath(path or os.environ.get('KATAGO_PYTHON') or DEFAULT_KATAGO_PYTHON)
    if not os.path.exists(os.path.join(path, 'features.py')):
        raise FileNotFoundError('%s 沒有 KataGo 的 features.py' % path)
    if path not in sys.path:
        sys.path.insert(0, path)
    return {name: importlib.import_module(name)
            for name in ('board', 'features', 'sgfmetadata', 'model_pytorch')}


def inverse_rank(profile):
    """rank_<n>k、rank_<n>d 轉成 KataGo 的 inverse rank（9d = 1、1d = 9、1k = 10），對應 C++ getInverseRank。"""
    rank = profile[len('rank_'):] if profile.startswith('rank_') else ''
    n, unit = rank[:-1], rank[-1:]
    if n.isdigit() and unit == 'd' and 1 <= int(n) <= 9:
        return 10 - int(n)
    if n.isdigit() and unit == 'k' and 1 <= int(n) <= 20:
        return 9 + int(n)
    raise ValueError('不支援的 profile：%s' % profile)


def profile_row(profile, pla='B'):
    """profile 的 192 維 metadata，移植 C++ makeBasicRankProfile（KGS、讀秒 1200 + 5×30、2020-03-01）。

    雙方都設成同一個等級，所以 pla 是黑或白結果相同；要讓雙方不同等級時這裡要改。
    Python 的 SGFMetadata 預設 tcIsUnknown 為真，C++ 為假，這裡明確關掉。
    """
    kg = katago()
    meta = kg['sgfmetadata'].SGFMetadata
    r = inverse_rank(profile)
    m = meta(inverseBRank=r, inverseWRank=r, bIsHuman=True, wIsHuman=True,
             gameRatednessIsUnknown=True, tcIsUnknown=False, tcIsByoYomi=True,
             mainTimeSeconds=1200, periodTimeSeconds=30, byoYomiPeriods=5,
             gameDate=datetime.date(2020, 3, 1), source=meta.SOURCE_KGS)
    return m.get_metadata_row(nextPlayer=pla, boardArea=BOARD * BOARD)


def to_loc(board, point):
    """sgfmill 的 (row, col)（row 從下往上）轉成 KataGo Board 的 loc，None 是 pass。"""
    if point is None:
        return board.PASS_LOC
    row, col = point
    return board.loc(col, BOARD - 1 - row)


def encode_game(parsed, turns, history=False):
    """parse_game 的結果在 turns 各回合落子前的 KataGo 輸入。

    history 為假時不給網路看前幾手（特徵 9～13、全域 0～4 為 0，前一、前二盤面的征子改用當下盤面），
    對應 analysis engine 預設的 ignorePreRootHistory，analyze_humansl 的 probs 就是這樣算的。
    history 為真的路徑沒有和 C++ 比對過。

    回傳 dict：
        spatial  (n, 22, 19, 19) uint8
        global   (n, 19) float32
        legal    (n, 362) bool，最後一格是 pass
        move     (n,) int64，實際落子的 policy index
    turns 要遞增且都在 moves 範圍內。最後一個 turn 之前的著手不是黑白交替、或回合前已連續兩次 pass
    （日本規則進入 encore）時丟 ValueError；最後一個 turn 之後的著手不檢查。
    """
    kg = katago()
    Board = kg['board'].Board
    feats = kg['features'].Features({'version': INPUT_VERSION}, BOARD)
    color = {'B': Board.BLACK, 'W': Board.WHITE}
    moves = parsed['moves']
    if any(t < 0 or t >= len(moves) for t in turns) or list(turns) != sorted(set(turns)):
        raise ValueError('turns 要遞增且在 0..%d 之間' % (len(moves) - 1))
    last = max(turns) if turns else -1
    if any(moves[i][0] == moves[i + 1][0] for i in range(last)):
        raise ValueError('回合 %d 之前的著手不是黑白交替' % last)
    rules = dict(RULES_JAPANESE, whiteKomi=parsed['komi'])

    board = Board(size=BOARD)
    for c, p in parsed['initial_stones']:
        board.set_stone(color[c], to_loc(board, p))  # set_stone 會計入 num_non_pass_moves_made，貼目換算同 C++
    if moves:
        board.set_pla(color[moves[0][0]])
    boards, played = [board.copy()], []
    n = len(turns)
    out = {'spatial': np.zeros((n, N_SPATIAL, BOARD, BOARD), np.uint8),
           'global': np.zeros((n, N_GLOBAL), np.float32),
           'legal': np.zeros((n, N_POLICY), bool),
           'move': np.zeros(n, np.int64)}
    want = {t: i for i, t in enumerate(turns)}
    passes = 0
    for t, (c, p) in enumerate(moves[:last + 1]):
        pla = color[c]
        if t in want:
            if passes >= 2:
                raise ValueError('回合 %d 前已連續兩次 pass，不支援 encore' % t)
            i = want[t]
            bin_row = np.zeros((1, BOARD * BOARD, N_SPATIAL), np.float32)
            glob_row = np.zeros((1, N_GLOBAL), np.float32)
            if history:
                feats.fill_row_features(board, pla, Board.get_opp(pla), boards, played, t, rules,
                                        bin_row, glob_row, idx=0)
            else:
                feats.fill_row_features(board, pla, Board.get_opp(pla), [board], [], 0, rules,
                                        bin_row, glob_row, idx=0)
            # 日本規則在 encore 之前，對手剛 pass 時再 pass 就結束這個階段；不受 history 影響
            glob_row[0, 14] = 1.0 if played and played[-1][1] == Board.PASS_LOC else 0.0
            out['spatial'][i] = bin_row[0].T.reshape(N_SPATIAL, BOARD, BOARD)
            out['global'][i] = glob_row[0]
            out['legal'][i] = legal_mask(board, pla)
            out['move'][i] = policy_index(p)
        loc = to_loc(board, p)
        board.play(pla, loc)
        passes = passes + 1 if p is None else 0
        if history:
            boards.append(board.copy())
        played.append((pla, loc))
    return out


def legal_mask(board, pla):
    """(362,) 合法點，日本規則：禁單子與多子自殺、禁 simple ko 回提，pass 一律合法。"""
    mask = np.zeros(N_POLICY, bool)
    mask[PASS_INDEX] = True
    for y in range(BOARD):
        for x in range(BOARD):
            loc = board.loc(x, y)
            mask[y * BOARD + x] = board.would_be_legal(pla, loc) and not board.would_be_suicide(pla, loc)
    return mask


def policy_symmetry(policy, k, inverse=False):
    """對 (..., 362) 的 policy 套用 symmetry.apply 的第 k 種對稱，inverse 時改用 symmetry.invert，pass 不動。"""
    import torch
    lead = policy.shape[:-1]
    transform = symmetry.invert if inverse else symmetry.apply
    grid = transform(policy[..., :PASS_INDEX].reshape(*lead, BOARD, BOARD), k)
    flat = grid.reshape(*lead, PASS_INDEX)
    if isinstance(policy, torch.Tensor):
        return torch.cat([flat, policy[..., PASS_INDEX:]], dim=-1)
    return np.concatenate([flat, policy[..., PASS_INDEX:]], axis=-1)


def load_humanv0(path=CHECKPOINT, device='cpu', use_swa=True):
    """載入 humanv0，回傳 eval 模式的 KataGo Model。use_swa 時用 checkpoint 內的 SWA 權重。

    checkpoint 含 optimizer 等非 tensor 物件，torch.load 要 weights_only=False，只能載入信任的檔案。
    """
    import torch
    kg = katago()
    sd = torch.load(path, map_location='cpu', weights_only=False)
    config = sd['config']
    if config['version'] != INPUT_VERSION:
        raise ValueError('checkpoint 的 version 是 %s，編碼假設 %d' % (config['version'], INPUT_VERSION))
    model = kg['model_pytorch'].Model(config, BOARD)
    model.initialize()
    if use_swa:
        # AveragedModel 的 state_dict 另有不帶 module. 的 n_averaged
        weights = {k.removeprefix('module.'): v for k, v in sd['swa_model'].items() if k.startswith('module.')}
    else:
        weights = {k.removeprefix('module.'): v for k, v in sd['model'].items()}
    model.load_state_dict(weights)
    return model.to(device).eval()


def policy_logits(model, spatial, glob, meta):
    """主要 policy head 的第 0 個 channel，(N, 362)，最後一格是 pass。"""
    outputs = model(spatial, glob, input_meta=meta)
    return outputs[0][0][:, 0, :]


def move_log_probs(model, enc, profile, batch=64, sym=0):
    """enc（encode_game 的輸出）各手實際落子的 log 機率，(n,) float64。sym 是輸入與 policy 的對稱。"""
    import torch
    device = next(model.parameters()).device
    meta_row = torch.from_numpy(profile_row(profile)).to(device)  # 雙方同等級，黑白的 metadata 相同
    out = []
    with torch.no_grad():
        for s in range(0, len(enc['move']), batch):
            sp = torch.from_numpy(symmetry.apply(enc['spatial'][s:s + batch], sym)).to(device).float()
            gl = torch.from_numpy(enc['global'][s:s + batch]).to(device)
            logits = policy_symmetry(policy_logits(model, sp, gl, meta_row.expand(len(gl), -1)), sym, inverse=True)
            legal = torch.from_numpy(enc['legal'][s:s + batch]).to(device)
            logp = torch.log_softmax(logits.float().masked_fill(~legal, float('-inf')), dim=-1)
            mv = torch.from_numpy(enc['move'][s:s + batch]).to(device)
            out.append(logp.gather(1, mv[:, None])[:, 0].double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def pick_verify_games(rows, n_games, n_handicap, seed):
    """從 jsonl 的列中抽 n_games 局，其中 n_handicap 局是 hd >= 2。"""
    rng = np.random.default_rng(seed)
    hd = [i for i, r in enumerate(rows) if (r.get('hd') or 0) >= 2]
    rest = [i for i, r in enumerate(rows) if (r.get('hd') or 0) < 2]
    idx = list(rng.choice(hd, n_handicap, replace=False)) + list(rng.choice(rest, n_games - n_handicap, replace=False))
    return [rows[i] for i in sorted(idx)]


def verify(args):
    """和 analyze_humansl 的 probs 逐手比較，印出每個 profile 的相關係數與每局平均 log 機率的差。"""
    from aicup.strength_se import load_keyed_games
    with open(args.verify, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = pick_verify_games(rows, args.games, args.handicap_games, args.seed)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = os.path.join(tmp_dir, 'keys.jsonl')
        with open(tmp, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ('key', 'player_id', 'rank_idx', 'hd')}) + '\n')
        games = {g['key']: g for g in load_keyed_games(tmp, args.csv)}
    model = load_humanv0(args.checkpoint, use_swa=not args.no_swa)
    profiles = args.profiles or rows[0]['profiles']
    ours = {p: [] for p in profiles}
    theirs = {p: [] for p in profiles}
    worst = []
    for r in rows:
        g = games[r['key']]
        parsed = parse_game(g['sgf'])
        turns = target_turns(parsed['moves'], g['color'] if r.get('side', 'target') == 'target'
                             else {'B': 'W', 'W': 'B'}[g['color']])
        if turns != r['turns']:
            raise SystemExit('%s 的 turns 和 jsonl 不同' % r['key'])
        enc = encode_game(parsed, turns)
        for p in profiles:
            j = r['profiles'].index(p)
            if args.symmetries > 1:
                # 機率對 8 種對稱平均後再取 log，同 C++ 的 rootNumSymmetriesToSample
                per = [move_log_probs(model, enc, p, sym=k) for k in range(args.symmetries)]
                a = np.log(np.mean(np.exp(per), axis=0))
            else:
                a = move_log_probs(model, enc, p, sym=args.sym)
            # analyze_humansl.analyze_group 把 probs 四捨五入到 1e-6
            b = np.log(np.maximum(np.array(r['probs'][j], float), 1e-6))
            ours[p].append(a)
            theirs[p].append(b)
            worst.append((abs(a.mean() - b.mean()), r['key'], r.get('hd'), p))
    passed = True
    for p in profiles:
        a, b = np.concatenate(ours[p]), np.concatenate(theirs[p])
        corr = np.corrcoef(a, b)[0, 1]
        diff = np.median([abs(x.mean() - y.mean()) for x, y in zip(ours[p], theirs[p])])
        ok = corr >= 0.98 and diff <= 0.02
        passed &= ok
        print('%-9s 手數 %d  逐手相關 %.4f  每局平均差中位數 %.4f  %s'
              % (p, len(a), corr, diff, '通過' if ok else '不通過'))
    print('每局平均差最大的 5 局：')
    for d, key, hd, p in sorted(worst, reverse=True)[:5]:
        print('  %s hd=%s %s %.4f' % (key, hd, p, d))
    print('判定：%s' % ('通過' if passed else '不通過'))


def bench(args):
    """量 encode_game 每秒產生的局面數（單一 process，不跑網路）。"""
    from aicup.strength_se import load_keyed_games
    games = load_keyed_games(args.keys_from, args.csv, args.bench)
    parsed = [(parse_game(g['sgf']), g['color']) for g in games]
    t0 = time.time()
    n = 0
    for pg, c in parsed:
        n += len(encode_game(pg, target_turns(pg['moves'], c))['move'])
    dt = time.time() - t0
    print('%d 局 %d 個局面，%.1f 秒，每秒 %.1f 個局面' % (len(parsed), n, dt, n / dt))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--katago-python', help='KataGo 原始碼的 python/ 目錄，缺省時讀環境變數 KATAGO_PYTHON')
    ap.add_argument('--checkpoint', default=CHECKPOINT)
    ap.add_argument('--csv', default='training-v1.1/training/train_*.csv')
    ap.add_argument('--verify', help='analyze_humansl 的 jsonl，和其中的 probs 逐手比較')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--handicap-games', type=int, default=2, help='--games 中 hd >= 2 的局數')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sym', type=int, default=0, help='推論時的對稱')
    ap.add_argument('--symmetries', type=int, default=1, help='大於 1 時改用前幾種對稱的平均機率，忽略 --sym')
    ap.add_argument('--profiles', nargs='+', help='只比較這些 profile，預設是 jsonl 的全部 profile')
    ap.add_argument('--no-swa', action='store_true', help='用 checkpoint 的 model 權重，不用 SWA')
    ap.add_argument('--bench', type=int, default=0, help='量前幾局的編碼速度')
    ap.add_argument('--keys-from', default='humansl-analysis/val648-p3.jsonl', help='--bench 用的局')
    args = ap.parse_args()
    if not args.verify and not args.bench:
        ap.error('要給 --verify 或 --bench')
    if args.katago_python:
        os.environ['KATAGO_PYTHON'] = args.katago_python
    if args.verify:
        verify(args)
    if args.bench:
        bench(args)


if __name__ == '__main__':
    main()
