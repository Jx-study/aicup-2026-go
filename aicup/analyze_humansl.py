"""KataGo Human SL 分析：目標玩家每一手在各棋力 profile 下的落子機率

每局對選定的每個 profile（預設是全部 PROFILES）送一個 analysis query，analyzeTurns 列出目標玩家要落子的回合，
記錄實際落子在 humanPolicy 中的機率。每局另送一個不帶 profile 的 query，評估目標玩家落子前後的局面，
記錄每手的目數損失與勝率損失，--no-eval 時不送。結果追加寫入 jsonl，每局一行，重跑會跳過已完成的局。

--side opponent 時改評目標玩家的對手那一方的手，turns、probs 與損失都屬於對手，color 欄位仍是目標玩家的顏色。

--test-dir 時改讀競賽二的測試題，每題拆成單局，player_id 記為 question_id，rank_idx 記為 -1。

用法（Colab，KataGo 執行檔在 --katago-dir，模型在 --model-dir）：
    !python -m aicup.analyze_humansl --players-per-rank 1 --games-per-player 10 --out humansl-analysis/pilot.jsonl
    !python -m aicup.analyze_humansl --test-dir /content/tests --profiles rank_11k rank_1d rank_6d --no-eval --out humansl-analysis/test-p3.jsonl
    !python -m aicup.analyze_humansl --players-file /content/val-players.csv --profiles rank_11k rank_1d rank_6d --no-eval --side opponent --out humansl-analysis/val648-p3-opp.jsonl
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
SIDES = ('target', 'opponent')
# 讓子數對應的標準星位，SGF 座標
HANDICAP_STONES = {2: 'pd dp', 3: 'pd dp dd', 4: 'pd dp dd pp', 5: 'pd dp dd pp jj',
                   6: 'pd dp dd pp dj pj', 7: 'pd dp dd pp dj pj jj',
                   8: 'pd dp dd pp dj pj jd jp', 9: 'pd dp dd pp dj pj jd jp jj'}


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


def sgf_point(s):
    """SGF 座標（aa 在左上角）轉成 sgfmill 的 (row, col)。"""
    return BOARD - 1 - (ord(s[1]) - ord('a')), ord(s[0]) - ord('a')


def parse_game(sgf_content, default_komi=6.5):
    """解析一局，回傳 dict：moves、initial_stones、komi、hd。非 19 路棋盤丟 ValueError。

    HD[n] 是讓子資訊，沒有這個屬性時 hd 為 None。n >= 1 不貼目。
    n >= 2 的棋譜沒記錄讓子石，著手黑白標記與實際相反，這裡補上讓子石並對調，
    moves 一律是實際顏色。color 欄位已是實際顏色，不用對調。
    """
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
    hd = int(root.get_raw('HD')) if root.has_property('HD') else None
    if hd is not None and hd >= 2:
        stones += [('B', sgf_point(s)) for s in HANDICAP_STONES[hd].split()]
        moves = [('W' if c == 'B' else 'B', p) for c, p in moves]
    # get_komi() 在沒有 KM 時回傳 0，要先看屬性是否存在
    if root.has_property('KM'):
        komi = float(game.get_komi())
    else:
        komi = 0.0 if hd else default_komi
    return {'moves': moves, 'initial_stones': stones, 'komi': komi, 'hd': hd}


def side_color(color, side):
    """要評的那一方的顏色：target 是目標玩家，opponent 是對手。"""
    return color if side == 'target' else {'B': 'W', 'W': 'B'}[color]


def target_turns(moves, color, max_turns=0):
    """color 方落子前的回合：turn t 是 moves[:t] 下完後的局面。"""
    turns = [t for t, (c, _) in enumerate(moves) if c == color]
    return turns[:max_turns] if max_turns else turns


def eval_turns(turns):
    """算損失要評估的回合：每手的落子前（t）與落子後（t + 1）。"""
    return sorted({t for turn in turns for t in (turn, turn + 1)})


def move_losses(root_infos, turns, color):
    """目標玩家每手的 (目數損失, 勝率損失)，正值代表變差。

    root_infos 是 {turn: rootInfo}，數值以黑方視角回報。
    """
    sign = 1 if color == 'B' else -1
    score = [sign * (root_infos[t]['scoreLead'] - root_infos[t + 1]['scoreLead']) for t in turns]
    winrate = [sign * (root_infos[t]['winrate'] - root_infos[t + 1]['winrate']) for t in turns]
    return score, winrate


def base_query(qid, game, turns, rules):
    return {
        'id': qid,
        'moves': [[c, to_gtp(p)] for c, p in game['moves']],
        'initialStones': [[c, to_gtp(p)] for c, p in game['initial_stones']],
        'rules': rules, 'komi': game['komi'],
        'boardXSize': BOARD, 'boardYSize': BOARD,
        'analyzeTurns': turns,
    }


def build_query(qid, game, turns, profile, rules):
    return dict(base_query(qid, game, turns, rules), maxVisits=1, includePolicy=True,
                overrideSettings={'humanSLProfile': profile})


def build_eval_query(qid, game, turns, rules, max_visits):
    """不帶 profile，用一般模型評估局面。"""
    return dict(base_query(qid, game, eval_turns(turns), rules), maxVisits=max_visits)


class Engine:
    """KataGo analysis engine 子進程。stdout 由另一條 thread 讀，避免回應塞滿 pipe 時雙方互等。"""

    def __init__(self, katago_dir, model, human_model, config=None, override=''):
        env = dict(os.environ)
        # KataGo 的 CUDA 12 版要用 pip nvidia 套件的函式庫
        libs = [d for d in glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/*/lib')
                if '/cu13/' not in d]
        if libs:
            env['LD_LIBRARY_PATH'] = ':'.join(libs + [env.get('LD_LIBRARY_PATH', '')])
        self.log = open(os.path.join(katago_dir, 'katago.log'), 'w')
        # 損失的正負號依賴黑方視角，不跟著 config 檔
        override = ','.join(filter(None, [override, 'reportAnalysisWinratesAs=BLACK']))
        cmd = [os.path.join(katago_dir, 'katago'), 'analysis',
               '-config', config or os.path.join(katago_dir, 'analysis_example.cfg'),
               '-model', model, '-human-model', human_model, '-override-config', override]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, env=env)
        self.out = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.out.put(json.loads(line))
        self.out.put(None)  # 進程結束

    def run(self, queries, timeout=600):
        """送出所有 query，回傳 {id: {turnNumber: 回應}}，回應只留 humanPolicy、rootInfo；出錯的 id 值為 None。"""
        want = {q['id']: len(q['analyzeTurns']) for q in queries}
        for q in queries:
            self.proc.stdin.write(json.dumps(q) + '\n')
        self.proc.stdin.flush()
        got = {qid: {} for qid in want}
        while any(got[qid] is not None and len(got[qid]) < n for qid, n in want.items()):
            try:
                r = self.out.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeError('KataGo %d 秒沒有回應' % timeout)
            if r is None:
                raise RuntimeError('KataGo 已結束，結束碼 %s，log 見 %s'
                                   % (self.proc.poll(), self.log.name))
            qid = r.get('id')
            if 'warning' in r:
                print('  query 警告 %s：%s' % (qid, r['warning']), flush=True)
                continue
            if 'error' in r:
                print('  query 錯誤 %s：%s' % (qid, r['error']), flush=True)
                if qid in got:
                    got[qid] = None
                continue
            if got.get(qid) is not None:
                got[qid][r['turnNumber']] = {k: r[k] for k in ('humanPolicy', 'rootInfo') if k in r}
        return got

    def close(self):
        self.proc.terminate()


def analyze_group(engine, games, rules, max_turns, eval_visits=1, profiles=PROFILES, side='target'):
    """同時送出一組局的所有 query，回傳可寫入 jsonl 的 dict 清單，失敗的局略過。

    probs 的列順序和 profiles 相同。eval_visits 為 0 時不送損失 query，輸出沒有損失欄位。
    side 決定評哪一方的手，見 side_color；輸出的 color 一律是目標玩家的顏色。
    """
    queries, meta = [], {}
    for g in games:
        turns = target_turns(g['parsed']['moves'], side_color(g['color'], side), max_turns)
        if not turns:
            continue
        meta[g['key']] = (g, turns)
        queries += [build_query('%s|%d' % (g['key'], i), g['parsed'], turns, prof, rules)
                    for i, prof in enumerate(profiles)]
        if eval_visits:
            queries.append(build_eval_query('%s|eval' % g['key'], g['parsed'], turns, rules, eval_visits))
    resp = engine.run(queries)
    rows = []
    for key, (g, turns) in meta.items():
        per_profile = [resp['%s|%d' % (key, i)] for i in range(len(profiles))]
        evals = resp['%s|eval' % key] if eval_visits else {}
        if evals is None or any(r is None for r in per_profile):
            continue
        probs = np.array([[r[t]['humanPolicy'][policy_index(g['parsed']['moves'][t][1])] for t in turns]
                          for r in per_profile])
        probs[probs < 0] = np.nan  # -1 代表該點在這組規則下不合法
        row = {'key': key, 'rank_idx': g['rank_idx'], 'player_id': g['player_id'],
               'hd': g['parsed']['hd'],
               'color': g['color'], 'side': side,
               'turns': turns, 'eval_visits': eval_visits, 'profiles': list(profiles),
               'probs': np.round(probs, 6).tolist()}
        if eval_visits:
            score_loss, winrate_loss = move_losses({t: r['rootInfo'] for t, r in evals.items()},
                                                   turns, side_color(g['color'], side))
            row['score_loss'] = np.round(score_loss, 3).tolist()
            row['winrate_loss'] = np.round(winrate_loss, 5).tolist()
        rows.append(row)
    return rows


def sample_players(df, players_per_rank, games_per_player, seed):
    """每個等級抽 players_per_rank 位局數足夠的玩家，每人抽 games_per_player 局，模擬題才湊得出同一玩家的多局。"""
    enough = df.groupby('player_id')['player_id'].transform('size') >= games_per_player
    players = df.loc[enough, ['rank', 'player_id']].drop_duplicates()
    counts = players.groupby('rank').size().reindex(df['rank'].unique(), fill_value=0)
    short = counts[counts < players_per_rank]
    if len(short):
        raise SystemExit('局數 %d 以上的玩家不足 %d 位：%s'
                         % (games_per_player, players_per_rank, short.to_dict()))
    picked = players.groupby('rank')['player_id'].sample(n=players_per_rank, random_state=seed)
    return (df[df['player_id'].isin(picked)].groupby('player_id', group_keys=False)
            .sample(n=games_per_player, random_state=seed))


def sample_listed_players(df, players, games_per_player, seed):
    """只留 players 內 (player_id, rank) 的局，每人抽 games_per_player 局，seed 用法同 sample_players()。

    同一個 player_id 可能出現在不同等級，所以連同 rank 比對。局數不足的玩家以 SystemExit 拒絕。
    回傳順序只由清單內容與 seed 決定，和清單的列順序無關。
    """
    listed = pd.MultiIndex.from_frame(players[['player_id', 'rank']].astype(str))
    key = [df['player_id'].astype(str), df['rank'].astype(str)]
    df = df[pd.MultiIndex.from_arrays(key).isin(listed)]
    groups = df.groupby([df['player_id'].astype(str), df['rank'].astype(str)], group_keys=False)
    counts = groups.size().reindex(listed, fill_value=0)
    short = counts[counts < games_per_player]
    if len(short):
        raise SystemExit('局數不足 %d 的玩家：%s' % (games_per_player, short.to_dict()))
    return groups.sample(n=games_per_player, random_state=seed)


def load_games(csv_glob, per_rank, seed, limit, players_per_rank=0, games_per_player=20,
               players_file=None):
    """讀 CSV 並抽樣，回傳每局一個 dict。

    players_file 是含 player_id、rank 欄位的 CSV，給定時只抽清單內的玩家，不能和 per_rank、players_per_rank 同時使用。
    players_per_rank 非 0 時依玩家抽樣，忽略 per_rank。
    """
    if players_file and (per_rank or players_per_rank):
        raise SystemExit('--players-file 不能和 --per-rank、--players-per-rank 同時使用')
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise SystemExit('找不到 %s' % csv_glob)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if players_file:
        # player_id 全是數字時預設會被讀成整數
        df = sample_listed_players(df, pd.read_csv(players_file, dtype=str), games_per_player, seed)
    elif players_per_rank:
        df = sample_players(df, players_per_rank, games_per_player, seed)
    elif per_rank:
        df = df.groupby('rank', group_keys=False).sample(n=per_rank, random_state=seed)
    if limit:
        df = df.iloc[:limit]
    # game_id 只在單一等級內唯一，key 加等級前綴
    return [{'key': '%s_%s' % (r['rank'], r['game_id']), 'rank_idx': LEVELS.index(r['rank']),
             'player_id': str(r['player_id']), 'color': r['color'], 'sgf': r['sgf_content']}
            for _, r in df.iterrows()]


def test_game_keys(qids, games):
    """測試題逐局的 key：question_id 加題內序號，順序同 predict_rank.read_questions 的 games。"""
    seen = {}
    keys = []
    for q, _, _ in games:
        seen[q] = seen.get(q, 0) + 1
        keys.append('%s_g%02d' % (qids[q], seen[q]))
    return keys


def load_test_games(test_dir, limit=0):
    """讀競賽二的 public 與 private 測試題，回傳每局一個 dict，欄位同 load_games。

    color 是題目給的目標玩家顏色，沒有 color_ 欄位時拒絕。
    """
    from aicup.predict_rank import read_questions
    qids, games = read_questions(test_dir)
    if any(c is None for _, _, c in games):
        raise SystemExit('%s 的測試題缺 color_ 欄位' % test_dir)
    out = [{'key': k, 'rank_idx': -1, 'player_id': qids[q], 'color': c, 'sgf': s}
           for k, (q, s, c) in zip(test_game_keys(qids, games), games)]
    return out[:limit] if limit else out


def done_keys(path, eval_visits, profiles=PROFILES, side='target'):
    """已完成的 key。檔案裡的 eval_visits、profiles 或 side 和這次不同時拒絕接續，避免同一檔混用兩種設定。

    沒有 eval_visits 欄位的列視為不同，沒有 profiles 欄位的列視為全部 PROFILES，沒有 side 欄位的列視為 target。
    """
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if any(r.get('eval_visits') != eval_visits for r in rows):
        raise SystemExit('%s 的 eval_visits 不是 %d，請換一個 --out' % (path, eval_visits))
    if any(r.get('profiles', PROFILES) != list(profiles) for r in rows):
        raise SystemExit('%s 的 profiles 不是 %s，請換一個 --out' % (path, ' '.join(profiles)))
    if any(r.get('side', 'target') != side for r in rows):
        raise SystemExit('%s 的 side 不是 %s，請換一個 --out' % (path, side))
    return {r['key'] for r in rows}


def check_profiles(profiles):
    unknown = [p for p in profiles if p not in PROFILES]
    if unknown:
        raise SystemExit('未知的 profile：%s，可用的有 %s' % (' '.join(unknown), ' '.join(PROFILES)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', default='training/train_*.csv')
    ap.add_argument('--per-rank', type=int, default=0, help='每個等級抽幾局，0 為全部')
    ap.add_argument('--players-per-rank', type=int, default=0, help='每個等級抽幾位玩家，非 0 時忽略 --per-rank')
    ap.add_argument('--games-per-player', type=int, default=20, help='每位玩家抽幾局，局數不足的玩家不抽')
    ap.add_argument('--players-file', help='含 player_id、rank 欄位的 CSV，只抽清單內的玩家，有人局數不足時拒絕')
    ap.add_argument('--test-dir', help='競賽二測試題所在目錄；給定時不讀 --csv，不能和抽樣參數同時使用')
    ap.add_argument('--limit', type=int, default=0, help='最多分析幾局，用於冒煙測試')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='humansl-analysis/humansl.jsonl')
    ap.add_argument('--katago-dir', default='/content/katago')
    ap.add_argument('--model-dir', default='/content/drive/MyDrive/aicup-2026/katago')
    ap.add_argument('--model', default='kata1-b28c512nbt-s13255194368-d5935380940.bin.gz')
    ap.add_argument('--human-model', default='b18c384nbt-humanv0.bin.gz')
    ap.add_argument('--rules', default='japanese')
    ap.add_argument('--default-komi', type=float, default=6.5, help='SGF 沒有 KM 且不是讓先、讓子時使用')
    ap.add_argument('--max-turns', type=int, default=0, help='每局最多分析目標玩家的前幾手，0 為全部')
    ap.add_argument('--eval-visits', type=int, default=1, help='算每手損失的 query 用的 maxVisits')
    ap.add_argument('--no-eval', action='store_true', help='不送算損失的 query，eval_visits 記為 0')
    ap.add_argument('--profiles', nargs='+', default=PROFILES, help='要送 query 的 profile，預設全部')
    ap.add_argument('--side', default='target', choices=SIDES, help='評目標玩家或對手的手')
    ap.add_argument('--group', type=int, default=8, help='同時送出幾局')
    # 1 visit 用不到搜尋執行緒，改用多個 query 並行填滿 GPU 批次
    ap.add_argument('--override-config', default='numAnalysisThreads=32,numSearchThreadsPerAnalysisThread=1',
                    help='passed to katago as -override-config; empty string keeps the config file values')
    args = ap.parse_args()
    check_profiles(args.profiles)
    eval_visits = 0 if args.no_eval else args.eval_visits

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    done = done_keys(args.out, eval_visits, args.profiles, args.side)
    todo, bad = [], 0
    if args.test_dir:
        if args.per_rank or args.players_per_rank or args.players_file:
            raise SystemExit('--test-dir 不能和 --per-rank、--players-per-rank、--players-file 同時使用')
        games = load_test_games(args.test_dir, args.limit)
    else:
        games = load_games(args.csv, args.per_rank, args.seed, args.limit,
                           args.players_per_rank, args.games_per_player, args.players_file)
    for g in games:
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
                rows = analyze_group(engine, todo[i:i + args.group], args.rules, args.max_turns,
                                     eval_visits, args.profiles, args.side)
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
