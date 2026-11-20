"""b28 逐手軌跡特徵：ownership、scoreStdev 與損失，串流轉成每手 17 維存 .npz

每局在目標方落子的回合中等距取 k 個 t（t + 1 要有對手的落子），查詢 (t, t + 1, t + 2) 三個回合，
得到目標方第 t 手與對手第 t + 1 手的特徵，兩側各最多 k 手。只載入 b28，不載入 humanv0，網路看得到前幾手。
回應的 361 維 ownership 與 362 維 policy 不落地，每批局寫一個 part-*.npz，重跑會跳過已完成的局。

輸出欄位：key、rank_idx、player_id、color（目標玩家顏色）、hd、turns（局, 2, k，-1 補齊）、
feats（局, 2, k, 17，NaN 補齊），第 2 維 0 是目標方、1 是對手。17 維的順序見 FEATURES。

用法（Colab，KataGo 執行檔在 --katago-dir，模型在 --model-dir）：
    python -m aicup.b28_traj --csv /content/b28/games.csv --limit 50 --out-dir /content/b28/step0
    python -m aicup.b28_traj --csv /content/b28/games.csv --out-dir /content/b28/newdev
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from aicup.analyze_humansl import PASS_INDEX, Engine, base_query, parse_game, policy_index
from aicup.scoring import LEVELS

GROUP_A = ['score_loss', 'winrate_loss', 'log_prior', 'is_top']
GROUP_B = ['own_at_move', 'dead_zone', 'own_gain_signed', 'own_flip_l1', 'unsettled', 'unsettled_delta',
           'score_stdev', 'norm_score_loss', 'stdev_delta']
CONTEXT = ['turn', 'wr_mover', 'lead_mover', 'policy_entropy']
FEATURES = GROUP_A + GROUP_B + CONTEXT
K = 16  # 每側手數
DEAD = 0.9  # |own| 超過視為已定型
UNSETTLED = 0.5  # |own| 低於視為未定
SIGN_MIN = 0.3  # 只確認正負，不要求接近 1


def triplet_starts(moves, color, k=K):
    """color 方落子、下一手是對手落子的回合，等距取 k 個；不足 k 個全取。"""
    ok = [t for t in range(len(moves) - 1) if moves[t][0] == color and moves[t + 1][0] != color]
    if len(ok) <= k:
        return ok
    return [ok[i] for i in np.round(np.linspace(0, len(ok) - 1, k)).astype(int)]


def query_turns(starts):
    return sorted({t + d for t in starts for d in (0, 1, 2)})


def build_traj_query(qid, game, turns, rules):
    """1 visit 的軌跡 query；關掉 ignorePreRootHistory，網路看得到前幾手。"""
    return dict(base_query(qid, game, turns, rules), maxVisits=1, includeOwnership=True, includePolicy=True,
                overrideSettings={'ignorePreRootHistory': False})


def move_features(before, after, point, color, turn):
    """一手的 17 維特徵，順序同 FEATURES。

    before、after 是落子前後回合的回應（rootInfo、policy、ownership），數值與 ownership 都是黑方視角；
    ownership 與 policy 的 index 相同，由左上角 row-major。pass 的 own_at_move、dead_zone、log_prior、is_top 為 NaN；
    落點不合法（policy 為 -1）時 log_prior、is_top 為 NaN。
    """
    s = 1.0 if color == 'B' else -1.0
    rb, ra = before['rootInfo'], after['rootInfo']
    pol = np.asarray(before['policy'], dtype=float)
    ob = np.asarray(before['ownership'], dtype=float)
    oa = np.asarray(after['ownership'], dtype=float)
    idx = policy_index(point)
    score_loss = s * (rb['scoreLead'] - ra['scoreLead'])
    legal = pol[pol >= 0]
    pos = legal[legal > 0]
    if idx == PASS_INDEX:
        own_at, dead = np.nan, np.nan
    else:
        own_at = s * ob[idx]
        dead = float(abs(ob[idx]) > DEAD)
    if idx == PASS_INDEX or pol[idx] < 0:  # -1 代表該點在這組規則下不合法
        log_prior, is_top = np.nan, np.nan
    else:
        log_prior = float(np.log(max(pol[idx], 1e-6)))
        is_top = float(pol[idx] >= legal.max())
    unsettled_b = float((np.abs(ob) < UNSETTLED).mean())
    unsettled_a = float((np.abs(oa) < UNSETTLED).mean())
    return np.array([
        score_loss,
        s * (rb['winrate'] - ra['winrate']),
        log_prior,
        is_top,
        own_at,
        dead,
        s * float((oa - ob).sum()),
        float(np.abs(oa - ob).sum()),
        unsettled_b,
        unsettled_a - unsettled_b,
        rb['scoreStdev'],
        score_loss / (rb['scoreStdev'] + 1.0),
        ra['scoreStdev'] - rb['scoreStdev'],
        float(turn),
        s * (rb['winrate'] - 0.5) + 0.5,
        s * rb['scoreLead'],
        float(-(pos * np.log(pos)).sum()),
    ])


def game_features(resp, game, starts, k=K):
    """一局的 (turns (2, k), feats (2, k, 17))；resp 是 {turn: 回應}。第 0 列是目標方的第 t 手，第 1 列是對手的第 t + 1 手。"""
    moves = game['parsed']['moves']
    turns = np.full((2, k), -1, dtype=np.int32)
    feats = np.full((2, k, len(FEATURES)), np.nan, dtype=np.float32)
    for j, t in enumerate(starts):
        for side, m in enumerate((t, t + 1)):
            color, point = moves[m]
            turns[side, j] = m
            feats[side, j] = move_features(resp[m], resp[m + 1], point, color, m)
    return turns, feats


def analyze_group(engine, games, rules, k=K):
    """同時送出一組局的 query，回傳 dict 清單（每局一個），失敗或沒有可評手的局略過。"""
    queries, meta = [], {}
    for g in games:
        starts = triplet_starts(g['parsed']['moves'], g['color'], k)
        if not starts:
            continue
        meta[g['key']] = (g, starts)
        queries.append(build_traj_query(g['key'], g['parsed'], query_turns(starts), rules))
    resp = engine.run(queries, keep=('rootInfo', 'policy', 'ownership'))
    rows = []
    for key, (g, starts) in meta.items():
        r = resp[key]
        if r is None:
            continue
        turns, feats = game_features(r, g, starts, k)
        rows.append({'key': key, 'rank_idx': g['rank_idx'], 'player_id': g['player_id'], 'color': g['color'],
                     'hd': -1 if g['parsed']['hd'] is None else g['parsed']['hd'], 'turns': turns, 'feats': feats})
    return rows


def corner_points(n=6):
    """左下角 n × n 的空點與圍住它們的黑牆（sgfmill 的 (row, col)，row 從下往上）。"""
    inside = [(r, c) for r in range(n) for c in range(n)]
    wall = [(n, c) for c in range(n + 1)] + [(r, n) for r in range(n)]
    return inside, wall


def sign_check_queries(rules):
    """黑牆圍住左下角的盤面，黑先與白先各一個 query，用來確認 ownership 是黑方為正、不隨輪到誰而翻轉。"""
    _, wall = corner_points()
    game = {'moves': [], 'initial_stones': [('B', p) for p in wall], 'komi': 6.5}
    return [dict(build_traj_query('sign|%s' % pla, game, [0], rules), initialPlayer=pla) for pla in ('B', 'W')]


def check_sign(resp):
    """左下角空點的 ownership 平均，兩個 query 都要 > SIGN_MIN（黑方為正），否則丟 RuntimeError。回傳兩個平均。"""
    inside, _ = corner_points()
    idx = [policy_index(p) for p in inside]
    means = {}
    for pla in ('B', 'W'):
        r = resp.get('sign|%s' % pla)
        if not r:
            raise RuntimeError('ownership 方向檢查的 query 沒有回應')
        means[pla] = float(np.mean(np.asarray(r[0]['ownership'], dtype=float)[idx]))
    if min(means.values()) <= SIGN_MIN:
        raise RuntimeError('ownership 不是黑方為正：黑先 %.3f，白先 %.3f' % (means['B'], means['W']))
    return means


def save_part(out_dir, rows):
    """一批局寫成下一個 part-*.npz。"""
    nums = [int(os.path.basename(p)[5:-4]) for p in glob.glob(os.path.join(out_dir, 'part-*.npz'))]
    path = os.path.join(out_dir, 'part-%05d.npz' % (max(nums) + 1 if nums else 0))  # 取最大編號，刪過檔也不覆寫
    tmp = path + '.tmp'  # 先寫暫存檔再改名，斷線時不留下半個 part
    with open(tmp, 'wb') as f:
        _savez(f, rows)
    os.replace(tmp, path)
    return path


def _savez(f, rows):
    np.savez_compressed(f, key=np.array([r['key'] for r in rows]),
                        rank_idx=np.array([r['rank_idx'] for r in rows], dtype=np.int64),
                        player_id=np.array([r['player_id'] for r in rows]),
                        color=np.array([r['color'] for r in rows]),
                        hd=np.array([r['hd'] for r in rows], dtype=np.int64),
                        turns=np.stack([r['turns'] for r in rows]), feats=np.stack([r['feats'] for r in rows]))


def load_parts(out_dir):
    """合併 out_dir 的全部 part，回傳欄位同 save_part 的 dict；沒有 part 時回傳 None。"""
    paths = sorted(glob.glob(os.path.join(out_dir, 'part-*.npz')))
    if not paths:
        return None
    parts = []
    for p in paths:
        with np.load(p) as z:
            parts.append({f: z[f] for f in z.files})
    return {f: np.concatenate([z[f] for z in parts]) for f in parts[0]}


def load_csv_games(path, limit=0):
    """讀單一 CSV 的全部局（player_id 保留字串），順序同檔案。"""
    df = pd.read_csv(path, dtype={'player_id': str, 'game_id': str})
    if limit:
        df = df.iloc[:limit]
    return [{'key': '%s_%s' % (r['rank'], r['game_id']), 'rank_idx': LEVELS.index(r['rank']),
             'player_id': r['player_id'], 'color': r['color'], 'sgf': r['sgf_content']} for _, r in df.iterrows()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', required=True)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 局，用於步驟 0 的吞吐量實測')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--katago-dir', default='/content/katago')
    ap.add_argument('--model-dir', default='/content/models')
    ap.add_argument('--model', default='kata1-b28c512nbt-s13255194368-d5935380940.bin.gz')
    ap.add_argument('--rules', default='japanese')
    ap.add_argument('--default-komi', type=float, default=6.5)
    ap.add_argument('--group', type=int, default=64, help='同時送出幾局，也是每個 part 的局數')
    ap.add_argument('--override-config', default='numAnalysisThreads=32,numSearchThreadsPerAnalysisThread=1',
                    help='passed to katago as -override-config; empty string keeps the config file values')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    done = load_parts(args.out_dir)
    done = set(done['key']) if done is not None else set()
    todo, bad = [], 0
    for g in load_csv_games(args.csv, args.limit):
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

    engine = Engine(args.katago_dir, os.path.join(args.model_dir, args.model), None, override=args.override_config)
    try:
        means = check_sign(engine.run(sign_check_queries(args.rules), keep=('ownership',)))
        print('ownership 方向檢查通過：黑牆內平均 黑先 %.3f，白先 %.3f' % (means['B'], means['W']), flush=True)
        t0, n_games, n_turns = time.time(), 0, 0
        for i in range(0, len(todo), args.group):
            rows = analyze_group(engine, todo[i:i + args.group], args.rules)
            if rows:
                save_part(args.out_dir, rows)
            n_games += len(rows)
            n_turns += sum(len(query_turns([t for t in r['turns'][0] if t >= 0])) for r in rows)
            dt = time.time() - t0
            print('  %d/%d 局，%.0f 秒，每秒 %.1f 個查詢回合' % (min(i + args.group, len(todo)), len(todo), dt,
                                                       n_turns / max(dt, 1e-9)), flush=True)
    finally:
        engine.close()
    print('THROUGHPUT turns_per_sec=%.3f games=%d turns=%d seconds=%.1f'
          % (n_turns / max(time.time() - t0, 1e-9), n_games, n_turns, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
