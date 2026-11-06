"""逐手模型的局面抽取：用 katago_torch 的編碼把局面打包成 .npz 分片

兩種模式：
    spike   從 CNN train 切分的玩家抽樣，每個等級 --per-rank 局、每位玩家最多 --max-games-per-player 局，
            每局從目標方的回合均勻抽 --moves-per-game 手。依玩家切出 --holdout-frac 當 holdout。
    val648  --keys-from 的每一局抽全部目標手，--opp-from 給定時再抽全部對手手，回合和兩份 jsonl 的 turns 相同。
            給定 --test-dir 時局改讀競賽二測試題（rank_idx 為 -1），--keys-from 與 --opp-from 用 test-p3 的兩份 jsonl。

分片 {split}-{n:05d}.npz（np.savez_compressed），每列是一個局面：
    spatial    (N, SPATIAL_BYTES) uint8，(22, 19, 19) 的 np.packbits
    global     (N, 19) float32
    legal      (N, LEGAL_BYTES) uint8，(362,) 合法遮罩的 np.packbits
    move       (N,) uint16，實際落子的 policy index，361 是 pass
    turn       (N,) int16，落子前的回合
    side       (N,) int8，0 目標方、1 對手
    rank_idx   (N,) int8
    hd         (N,) int8，SGF 沒有 HD 時為 -1
    color      (N,) int8，目標玩家執色，0 黑 1 白
    key        (N,) str
    player_id  (N,) str
每個 split 目錄另有 summary.json，記錄局面數、局數、玩家數與丟棄的局數。中斷後重跑同一指令會跳過已存在的分片。

用法（在 repo 根目錄）：
    python -m aicup.extract_permove spike --out permove-features/spike
    python -m aicup.extract_permove val648 --out permove-features/val648
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from aicup import katago_torch as kt
from aicup.analyze_humansl import parse_game, target_turns
from aicup.extract_features import SEED, VAL_SIZE, check_config, load_dataframe, split_by_player
from aicup.scoring import LEVELS

FORMAT = 'permove-v1'
SPATIAL_BITS = kt.N_SPATIAL * kt.BOARD * kt.BOARD
SPATIAL_BYTES = (SPATIAL_BITS + 7) // 8
LEGAL_BYTES = (kt.N_POLICY + 7) // 8
OPP = {'B': 'W', 'W': 'B'}
FIELDS = ('spatial', 'global', 'legal', 'move', 'turn', 'side', 'rank_idx', 'hd', 'color', 'key', 'player_id')


def empty_arrays():
    """沒有任何局面的分片欄位，dtype 與形狀的第二維和有資料時相同。"""
    return {'spatial': np.zeros((0, SPATIAL_BYTES), np.uint8), 'global': np.zeros((0, kt.N_GLOBAL), np.float32),
            'legal': np.zeros((0, LEGAL_BYTES), np.uint8), 'move': np.zeros(0, np.uint16),
            'turn': np.zeros(0, np.int16), 'side': np.zeros(0, np.int8), 'rank_idx': np.zeros(0, np.int8),
            'hd': np.zeros(0, np.int8), 'color': np.zeros(0, np.int8),
            'key': np.zeros(0, dtype='<U1'), 'player_id': np.zeros(0, dtype='<U1')}


def pack_positions(enc):
    """encode_game 的輸出打包成分片的 spatial、global、legal、move 欄位。"""
    n = len(enc['move'])
    return {'spatial': np.packbits(enc['spatial'].reshape(n, -1).astype(bool), axis=1),
            'global': enc['global'].astype(np.float32),
            'legal': np.packbits(enc['legal'], axis=1),
            'move': enc['move'].astype(np.uint16)}


def unpack_spatial(packed):
    """(N, SPATIAL_BYTES) 還原成 (N, 22, 19, 19) uint8。"""
    bits = np.unpackbits(packed, axis=-1, count=SPATIAL_BITS)
    return bits.reshape(-1, kt.N_SPATIAL, kt.BOARD, kt.BOARD)


def unpack_legal(packed):
    """(N, LEGAL_BYTES) 還原成 (N, 362) bool。"""
    return np.unpackbits(packed, axis=-1, count=kt.N_POLICY).astype(bool)


def sample_spike_games(df, per_rank, max_games, seed=SEED):
    """每個等級抽 per_rank 局，每位玩家最多 max_games 局。

    湊不到 per_rank 局的等級把上限放寬為 2 倍再抽一次，仍不足時丟 SystemExit。
    回傳 (抽到的 DataFrame, {等級: 用到的上限})。玩家以 (rank, player_id) 識別。
    """
    shuffled = df.sample(frac=1.0, random_state=seed)
    player = shuffled['rank'].astype(str) + '/' + shuffled['player_id'].astype(str)
    order = shuffled.groupby(player, sort=False).cumcount()
    parts, caps = [], {}
    for rank in LEVELS:
        in_rank = (shuffled['rank'] == rank).to_numpy()
        for cap in (max_games, 2 * max_games):
            picked = shuffled[in_rank & (order < cap).to_numpy()].head(per_rank)
            if len(picked) == per_rank:
                break
        else:
            raise SystemExit('%s 每位玩家最多 %d 局仍湊不到 %d 局' % (rank, 2 * max_games, per_rank))
        parts.append(picked)
        caps[rank] = cap
    return pd.concat(parts), caps


def split_holdout(df, frac, seed=SEED):
    """依玩家切出 frac 的玩家當 holdout，回傳 (train, holdout)。"""
    player = df['rank'].astype(str) + '/' + df['player_id'].astype(str)
    players = np.array(sorted(player.unique()))
    np.random.default_rng(seed).shuffle(players)
    held = set(players[:int(np.ceil(len(players) * frac))])
    is_held = player.isin(held).to_numpy()
    return df[~is_held], df[is_held]


def player_set(df):
    """DataFrame 中 (rank_idx, player_id) 的集合，和分片的 rank_idx、player_id 欄位同一種表示。"""
    return set(zip(df['rank'].map(LEVELS.index), df['player_id'].astype(str)))


def shard_players(shard_dir):
    """rank-features 分片中 (rank_idx, player_id) 的集合。"""
    out = set()
    for p in sorted(glob.glob(os.path.join(shard_dir, '*.npz'))):
        with np.load(p, allow_pickle=False) as z:
            out |= set(zip(z['rank_idx'].astype(int).tolist(), z['player_id'].tolist()))
    return out


def game_job(row, sides):
    """一局的抽取設定。sides 是 {side: turns 清單或 ('sample', k, seed)}。"""
    return {'key': row['key'], 'rank_idx': int(row['rank_idx']), 'player_id': str(row['player_id']),
            'color': row['color'], 'sgf': row['sgf'], 'sides': sides}


def choose_turns(parsed, color, spec):
    """spec 是回合清單時原樣回傳；是 ('sample', k, seed) 時從 color 方的回合均勻抽 k 手（不足 k 手時全取）。"""
    if not (isinstance(spec, tuple) and spec[0] == 'sample'):
        return list(spec)
    _, k, seed = spec
    turns = target_turns(parsed['moves'], color)
    if len(turns) <= k:
        return turns
    return sorted(np.random.default_rng(seed).choice(turns, k, replace=False).tolist())


def encode_job(game):
    """抽一局的局面，回傳 (分片欄位的 dict 或 None, 丟棄原因或 None)。"""
    try:
        parsed = parse_game(game['sgf'])
    except Exception as e:  # sgfmill 的錯誤種類很多，一律當解析失敗
        return None, 'parse: %s' % type(e).__name__
    per_side = {}
    for s, spec in game['sides'].items():
        color = game['color'] if s == 0 else OPP[game['color']]
        per_side[s] = choose_turns(parsed, color, spec)
    side_of = {t: s for s, turns in per_side.items() for t in turns}
    turns = sorted(side_of)
    if not turns:
        return None, 'no turns'
    try:
        enc = kt.encode_game(parsed, turns)
    except ValueError as e:
        return None, 'encode: %s' % e
    n = len(turns)
    out = pack_positions(enc)
    out.update({'turn': np.array(turns, np.int16),
                'side': np.array([side_of[t] for t in turns], np.int8),
                'rank_idx': np.full(n, game['rank_idx'], np.int8),
                'hd': np.full(n, -1 if parsed['hd'] is None else parsed['hd'], np.int8),
                'color': np.full(n, 0 if game['color'] == 'B' else 1, np.int8),
                'key': np.array([game['key']] * n, dtype=str),
                'player_id': np.array([game['player_id']] * n, dtype=str)})
    return out, None


def encode_shard(args):
    """抽一個分片並寫檔，回傳 (局數, 局面數, 丟棄原因清單)。

    丟棄紀錄先寫，.npz 最後由暫存檔改名產生：.npz 存在就代表這個分片完整，重跑時可以跳過。
    """
    path, games = args
    parts, dropped = [], []
    for g in games:
        out, why = encode_job(g)
        if out is None:
            dropped.append((g['key'], why))
        else:
            parts.append(out)
    if parts:
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    else:
        arrays = empty_arrays()
    with open(path + '.dropped.json', 'w', encoding='utf-8') as f:
        json.dump(dropped, f, ensure_ascii=False)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:  # 傳檔案物件，避免 numpy 自動補 .npz 副檔名
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return len(parts), len(arrays['move']), dropped


def extract_split(games, split_dir, split, nproc, per_shard):
    """games 依序切成分片，用 nproc 個 process 抽取，已存在的分片跳過。"""
    os.makedirs(split_dir, exist_ok=True)
    jobs = [(os.path.join(split_dir, '%s-%05d.npz' % (split, i // per_shard)), games[i:i + per_shard])
            for i in range(0, len(games), per_shard)]
    todo = [j for j in jobs if not os.path.exists(j[0])]
    print('%s：%d 局、%d 個分片，待抽 %d 個' % (split, len(games), len(jobs), len(todo)), flush=True)
    t0 = time.time()
    n_pos = 0
    with ProcessPoolExecutor(nproc) as ex:
        for i, (_, n, _) in enumerate(ex.map(encode_shard, todo), 1):
            n_pos += n
            rate = n_pos / (time.time() - t0)
            print('  %s 分片 %d/%d，每秒 %.0f 個局面，剩約 %.0f 分'
                  % (split, i, len(todo), rate, (len(todo) - i) * (time.time() - t0) / i / 60), flush=True)
    return summarize(split_dir)


def summarize(split_dir):
    """讀整個 split 目錄的分片，寫出並回傳 summary.json 的內容。"""
    paths = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
    sides, keys, players, dropped = np.zeros(2, int), set(), set(), []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            sides += np.bincount(z['side'], minlength=2)[:2]
            keys |= set(z['key'].tolist())
            players |= set(zip(z['rank_idx'].astype(int).tolist(), z['player_id'].tolist()))
        with open(p + '.dropped.json', encoding='utf-8') as f:
            dropped += json.load(f)
    summary = {'shards': len(paths), 'positions': int(sides.sum()),
               'positions_target': int(sides[0]), 'positions_opponent': int(sides[1]),
               'games': len(keys), 'players': len(players), 'dropped_games': len(dropped),
               'dropped_examples': dropped[:20],
               'bytes': int(sum(os.path.getsize(p) for p in paths))}
    with open(os.path.join(split_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('%s：%s' % (split_dir, {k: v for k, v in summary.items() if k != 'dropped_examples'}), flush=True)
    return summary


def df_games(df):
    """DataFrame 轉成 game_job 需要的欄位。"""
    return [{'key': '%s_%s' % (r['rank'], r['game_id']), 'rank_idx': LEVELS.index(r['rank']),
             'player_id': str(r['player_id']), 'color': r['color'], 'sgf': r['sgf_content']}
            for _, r in df.iterrows()]


def run_spike(args):
    df = load_dataframe(args.train_dir)
    df_val, df_train = split_by_player(df, VAL_SIZE)
    if args.check_features:
        expected = shard_players(args.check_features)
        got = player_set(df_train)
        if got != expected:
            raise SystemExit('train 玩家和 %s 不同：多 %d 位、少 %d 位'
                             % (args.check_features, len(got - expected), len(expected - got)))
        print('train 玩家 %d 位，和 %s 相同' % (len(got), args.check_features), flush=True)
    picked, caps = sample_spike_games(df_train, args.per_rank, args.max_games_per_player)
    train, holdout = split_holdout(picked, args.holdout_frac)
    if player_set(train) & player_set(holdout) or player_set(picked) & player_set(df_val):
        raise SystemExit('spike 的 train、holdout、CNN val 玩家有重疊')
    with open(os.path.join(args.out, 'sampling.json'), 'w', encoding='utf-8') as f:
        json.dump({'caps': caps, 'games': len(picked), 'players': len(player_set(picked)),
                   'train_players': len(player_set(train)), 'holdout_players': len(player_set(holdout))},
                  f, ensure_ascii=False, indent=2)
    print('每位玩家上限：%s' % caps, flush=True)
    for split, part in (('holdout', holdout), ('train', train)):
        games = df_games(part)
        # 抽樣 seed 依局在抽樣結果中的位置決定，重跑結果相同
        jobs = [game_job(g, {0: ('sample', args.moves_per_game, (SEED, i))}) for i, g in enumerate(games)]
        extract_split(jobs, os.path.join(args.out, split), split, args.nproc, args.games_per_shard)


def read_turns(path):
    """analyze_humansl 的 jsonl 讀成 {key: 列}，順序同檔案，略過空行。"""
    with open(path, encoding='utf-8') as f:
        return {r['key']: r for r in (json.loads(line) for line in f if line.strip())}


def run_val648(args):
    target = read_turns(args.keys_from)
    opp = read_turns(args.opp_from) if args.opp_from else {}
    if opp and set(opp) != set(target):
        raise SystemExit('%s 和 %s 的 key 不同' % (args.keys_from, args.opp_from))
    if args.test_dir:
        from aicup.analyze_humansl import load_test_games
        by_key = {g['key']: g for g in load_test_games(args.test_dir)}
        missing = [k for k in target if k not in by_key]
        if missing:
            raise SystemExit('%s 找不到 %d 局，例如 %s' % (args.test_dir, len(missing), ', '.join(missing[:3])))
        games = [by_key[k] for k in target]
    else:
        from aicup.strength_se import load_keyed_games
        games = load_keyed_games(args.keys_from, args.csv)
    jobs = []
    for g in games:
        sides = {0: target[g['key']]['turns']}
        if opp:
            sides[1] = opp[g['key']]['turns']
        jobs.append(game_job(g, sides))
    summary = extract_split(jobs, os.path.join(args.out, 'val'), 'val', args.nproc, args.games_per_shard)
    want = (sum(len(r['turns']) for r in target.values()), sum(len(r['turns']) for r in opp.values()))
    got = (summary['positions_target'], summary['positions_opponent'])
    print('局面數：目標側 %d（預期 %d）、對手側 %d（預期 %d）' % (got[0], want[0], got[1], want[1]))
    if got != want:
        raise SystemExit('局面數和 jsonl 的 turns 不同')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=('spike', 'val648'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--katago-python', help='KataGo 原始碼的 python/ 目錄，缺省時讀環境變數 KATAGO_PYTHON')
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--games-per-shard', type=int, default=1000)
    # spike
    ap.add_argument('--train-dir', default='training-v1.1/training', help='要有 HD，所以用 v1.1')
    ap.add_argument('--check-features', default='rank-features/train',
                    help='確認 train 玩家和這個目錄的分片相同；空字串不檢查')
    ap.add_argument('--per-rank', type=int, default=10_000)
    ap.add_argument('--max-games-per-player', type=int, default=30)
    ap.add_argument('--moves-per-game', type=int, default=10)
    ap.add_argument('--holdout-frac', type=float, default=0.05)
    # val648
    ap.add_argument('--keys-from', default='humansl-analysis/val648-p3.jsonl')
    ap.add_argument('--opp-from', default='humansl-analysis/val648-p3-opp.jsonl', help='空字串時不抽對手側')
    ap.add_argument('--csv', default='training-v1.1/training/train_*.csv')
    ap.add_argument('--test-dir', help='競賽二測試題所在目錄；給定時不讀 --csv')
    args = ap.parse_args()
    if args.katago_python:
        os.environ['KATAGO_PYTHON'] = args.katago_python  # spawn 的子 process 會繼承
    kt.katago()  # 主 process 先確認找得到 KataGo 原始碼

    if args.mode == 'spike':
        config = {'format': FORMAT, 'mode': 'spike', 'seed': SEED, 'val_size': VAL_SIZE,
                  'per_rank': args.per_rank, 'max_games_per_player': args.max_games_per_player,
                  'moves_per_game': args.moves_per_game, 'holdout_frac': args.holdout_frac,
                  'games_per_shard': args.games_per_shard, 'train_dir': args.train_dir}
    else:
        config = {'format': FORMAT, 'mode': 'val648', 'keys_from': args.keys_from,
                  'opp_from': args.opp_from, 'games_per_shard': args.games_per_shard,
                  'source': args.test_dir or args.csv}  # 棋譜來源：測試題目錄或訓練 CSV
    check_config(args.out, config)
    t0 = time.time()
    (run_spike if args.mode == 'spike' else run_val648)(args)
    print('完成，總耗時 %.1f 分' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
