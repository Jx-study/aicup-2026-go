"""KataGo Human SL 分析：目標玩家每一手在各棋力 profile 下的落子機率

每局對 PROFILES 的每個 profile 送一個 analysis query，analyzeTurns 列出目標玩家要落子的回合，
記錄實際落子在 humanPolicy 中的機率。結果追加寫入 jsonl，每局一行，重跑會跳過已完成的局。

用法（Colab，KataGo 執行檔在 --katago-dir，模型在 --model-dir）：
    !python -m aicup.analyze_humansl --per-rank 10 --out humansl-analysis/pilot.jsonl
"""
import argparse
import glob
import json
import os
import queue
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from aicup.scoring import LEVELS

# 各等級對應的 profile，取等級範圍的中間值
PROFILES = ['rank_11k', 'rank_8k', 'rank_5k', 'rank_2k',
            'rank_1d', 'rank_2d', 'rank_3d', 'rank_4d', 'rank_5d', 'rank_6d']
BOARD = 19
GTP_COLS = 'ABCDEFGHJKLMNOPQRST'  # GTP 座標沒有 I
PASS_INDEX = BOARD * BOARD


def to_gtp(point):
    """sgfmill 的 (row, col)，row 從下往上，轉成 GTP 座標；None 是 pass。"""
    if point is None:
        return 'pass'
    row, col = point
    return '%s%d' % (GTP_COLS[col], row + 1)


def policy_index(point):
    """落子在 humanPolicy 的 index：由左上角起 row-major，最後一格是 pass。"""
    if point is None:
        return PASS_INDEX
    row, col = point
    return (BOARD - 1 - row) * BOARD + col


def parse_game(sgf_content, default_komi=7.5):
    """解析一局，回傳 dict：moves、initial_stones、komi。非 19 路棋盤丟 ValueError。"""
    from sgfmill import sgf  # 只有解析時才需要 sgfmill
    game = sgf.Sgf_game.from_string(sgf_content)
    if game.get_size() != BOARD:
        raise ValueError('棋盤大小 %d' % game.get_size())
    root = game.get_root()
    stones = []
    for prop, color in (('AB', 'B'), ('AW', 'W')):
        if root.has_property(prop):
            stones += [(color, p) for p in sorted(root.get(prop))]
    moves = []
    for node in game.get_main_sequence():
        color, point = node.get_move()
        if color is not None:
            moves.append((color.upper(), point))
    # get_komi() 在沒有 KM 時回傳 0，要先看屬性是否存在
    komi = float(game.get_komi()) if root.has_property('KM') else default_komi
    return {'moves': moves, 'initial_stones': stones, 'komi': komi}


def target_turns(moves, color, max_turns=0):
    """目標玩家落子前的回合：turn t 是 moves[:t] 下完後的局面。"""
    turns = [t for t, (c, _) in enumerate(moves) if c == color]
    return turns[:max_turns] if max_turns else turns


def build_query(qid, game, turns, profile, rules):
    return {
        'id': qid,
        'moves': [[c, to_gtp(p)] for c, p in game['moves']],
        'initialStones': [[c, to_gtp(p)] for c, p in game['initial_stones']],
        'rules': rules, 'komi': game['komi'],
        'boardXSize': BOARD, 'boardYSize': BOARD,
        'analyzeTurns': turns, 'maxVisits': 1, 'includePolicy': True,
        'overrideSettings': {'humanSLProfile': profile},
    }


class Engine:
    """KataGo analysis engine 子進程。stdout 由另一條 thread 讀，避免回應塞滿 pipe 時雙方互等。"""

    def __init__(self, katago_dir, model, human_model, config=None, override=''):
        env = dict(os.environ)
        # Colab 系統只有 CUDA 13 的函式庫，KataGo 的 CUDA 12 版要從 pip 的 nvidia 套件找
        libs = [d for d in glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/*/lib')
                if '/cu13/' not in d]
        if libs:
            env['LD_LIBRARY_PATH'] = ':'.join(libs + [env.get('LD_LIBRARY_PATH', '')])
        self.log = open(os.path.join(katago_dir, 'katago.log'), 'w')
        cmd = [os.path.join(katago_dir, 'katago'), 'analysis',
               '-config', config or os.path.join(katago_dir, 'analysis_example.cfg'),
               '-model', model, '-human-model', human_model]
        if override:
            cmd += ['-override-config', override]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, env=env)
        self.out = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.out.put(json.loads(line))
        self.out.put(None)  # 進程結束

    def run(self, queries, timeout=600):
        """送出所有 query，回傳 {id: {turnNumber: humanPolicy}}；出錯的 id 值為 None。"""
        want = {q['id']: len(q['analyzeTurns']) for q in queries}
        for q in queries:
            self.proc.stdin.write(json.dumps(q) + '\n')
        self.proc.stdin.flush()
        got = {qid: {} for qid in want}
        while any(qid in got and len(got[qid]) < n for qid, n in want.items()):
            try:
                r = self.out.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeError('KataGo %d 秒沒有回應' % timeout)
            if r is None:
                raise RuntimeError('KataGo 已結束，結束碼 %s，log 見 %s'
                                   % (self.proc.poll(), self.log.name))
            qid = r.get('id')
            if 'error' in r:
                print('  query 錯誤 %s：%s' % (qid, r['error']), flush=True)
                if qid in got:
                    got[qid] = None
                continue
            if got.get(qid) is not None:
                got[qid][r['turnNumber']] = r['humanPolicy']
        return got

    def close(self):
        self.proc.terminate()


def analyze_group(engine, games, rules, max_turns):
    """同時送出一組局的所有 query，回傳可寫入 jsonl 的 dict 清單，失敗的局略過。"""
    queries, meta = [], {}
    for g in games:
        turns = target_turns(g['parsed']['moves'], g['color'], max_turns)
        if not turns:
            continue
        meta[g['key']] = (g, turns)
        queries += [build_query('%s|%d' % (g['key'], i), g['parsed'], turns, prof, rules)
                    for i, prof in enumerate(PROFILES)]
    resp = engine.run(queries)
    rows = []
    for key, (g, turns) in meta.items():
        per_profile = [resp['%s|%d' % (key, i)] for i in range(len(PROFILES))]
        if any(r is None for r in per_profile):
            continue
        probs = np.array([[r[t][policy_index(g['parsed']['moves'][t][1])] for t in turns]
                          for r in per_profile])
        probs[probs < 0] = np.nan  # -1 代表該點在這組規則下不合法
        rows.append({'key': key, 'rank_idx': g['rank_idx'], 'color': g['color'],
                     'turns': turns, 'probs': np.round(probs, 6).tolist()})
    return rows


def load_games(csv_glob, per_rank, seed, limit):
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise SystemExit('找不到 %s' % csv_glob)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if per_rank:
        df = df.groupby('rank', group_keys=False).sample(n=per_rank, random_state=seed)
    if limit:
        df = df.iloc[:limit]
    # game_id 只在單一等級內唯一，key 加等級前綴
    return [{'key': '%s_%s' % (r['rank'], r['game_id']), 'rank_idx': LEVELS.index(r['rank']),
             'color': r['color'], 'sgf': r['sgf_content']} for _, r in df.iterrows()]


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        return {json.loads(line)['key'] for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', default='training/train_*.csv')
    ap.add_argument('--per-rank', type=int, default=0, help='每個等級抽幾局，0 為全部')
    ap.add_argument('--limit', type=int, default=0, help='最多分析幾局，用於冒煙測試')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='humansl-analysis/humansl.jsonl')
    ap.add_argument('--katago-dir', default='/content/katago')
    ap.add_argument('--model-dir', default='/content/drive/MyDrive/aicup-2026/katago')
    ap.add_argument('--model', default='kata1-b28c512nbt-s13255194368-d5935380940.bin.gz')
    ap.add_argument('--human-model', default='b18c384nbt-humanv0.bin.gz')
    ap.add_argument('--rules', default='chinese')
    ap.add_argument('--default-komi', type=float, default=7.5, help='SGF 沒有 KM 時使用')
    ap.add_argument('--max-turns', type=int, default=0, help='每局最多分析目標玩家的前幾手，0 為全部')
    ap.add_argument('--group', type=int, default=8, help='同時送出幾局')
    # maxVisits 為 1 時每個 query 只有一次網路評估，搜尋執行緒用不到，改成多個 query 並行來填滿 GPU 批次
    ap.add_argument('--override-config', default='numAnalysisThreads=32,numSearchThreadsPerAnalysisThread=1',
                    help='passed to katago as -override-config; empty string keeps the config file values')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    done = done_keys(args.out)
    todo, bad = [], 0
    for g in load_games(args.csv, args.per_rank, args.seed, args.limit):
        if g['key'] in done:
            continue
        try:
            g['parsed'] = parse_game(g.pop('sgf'), args.default_komi)
        except Exception:
            bad += 1
            continue
        todo.append(g)
    print('待分析 %d 局，已完成 %d，解析失敗 %d' % (len(todo), len(done), bad), flush=True)
    if not todo:
        return

    engine = Engine(args.katago_dir, os.path.join(args.model_dir, args.model),
                    os.path.join(args.model_dir, args.human_model), override=args.override_config)
    t0, n_done = time.time(), 0
    try:
        with open(args.out, 'a', encoding='utf-8') as f:
            for i in range(0, len(todo), args.group):
                rows = analyze_group(engine, todo[i:i + args.group], args.rules, args.max_turns)
                for r in rows:
                    f.write(json.dumps(r) + '\n')
                f.flush()
                n_done += len(rows)
                print('  %d/%d 局，平均每局 %.1f 秒' % (i + len(todo[i:i + args.group]), len(todo),
                                                  (time.time() - t0) / max(n_done, 1)), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
