"""訓練資料檢查：欄位、等級與執色分布、玩家局數、SGF 標頭、手數、解析失敗率。

輸出 markdown 格式的報告，印在 stdout。只讀資料，不寫任何檔案。

用法（在 repo 根目錄）：
    python -m aicup.data_check
    python -m aicup.data_check --per-rank 200   # SGF 相關統計每個等級只抽 200 局
"""
import argparse
import collections
import re

import numpy as np
import pandas as pd

from aicup.extract_features import load_dataframe
from aicup.scoring import LEVELS

REQUIRED_COLS = ('player_id', 'game_id', 'rank', 'color', 'sgf_content')
HEADER_PROPS = ('SZ', 'HA', 'KM', 'RU', 'RE', 'PB', 'PW', 'BR', 'WR')


def missing_columns(df):
    return [c for c in REQUIRED_COLS if c not in df.columns]


def player_stats(df):
    """每個等級的玩家數與每位玩家的局數（最小、中位、最大）。"""
    games = df.groupby(['rank', 'player_id']).size()
    out = games.groupby(level='rank').agg(['count', 'min', 'median', 'max'])
    out.columns = ['players', 'games_min', 'games_median', 'games_max']
    return out.reindex([lv for lv in LEVELS if lv in out.index])


def cross_rank_players(df):
    """出現在多個等級的 player_id，回傳 (跨等級人數, 總人數)。"""
    n_ranks = df.groupby('player_id')['rank'].nunique()
    return int((n_ranks > 1).sum()), len(n_ranks)


def game_info(sgf_content):
    """解析一局，回傳根節點 property 的值（字串）、手數、pass 數。解析失敗丟出例外。"""
    from sgfmill import sgf  # 只有解析 SGF 時才需要 sgfmill
    game = sgf.Sgf_game.from_string(sgf_content)
    root = game.get_root()
    # sgfmill 解析後會自動補上 CA，只算原文裡真的有的 property
    props = {p: root.get_raw(p).decode('utf-8', 'replace') for p in root.properties()
             if re.search(r'(?<![A-Za-z])%s\s*\[' % p, sgf_content)}
    moves = [node.get_move() for node in game.get_main_sequence()]
    moves = [point for color, point in moves if color is not None]
    return props, len(moves), sum(p is None for p in moves)


def header_and_moves(sample):
    """統計標頭 property 的出現比例與常見值，以及每個等級的手數與 pass 比例。

    回傳 (n_ok, present, values, moves, failed)：
        n_ok     成功解析的局數
        present  Counter，鍵為 property，值為出現的局數
        values   dict，鍵為 property，值為各值的 Counter
        moves    dict，鍵為等級，值為 (手數中位數, pass 佔手數的比例)
        failed   Counter，鍵為例外類型名稱，值為解析失敗的局數
    """
    present, values = collections.Counter(), collections.defaultdict(collections.Counter)
    per_rank, n_ok, failed = collections.defaultdict(list), 0, collections.Counter()
    for rank, sgf_content in zip(sample['rank'], sample['sgf_content']):
        try:
            props, n_moves, n_pass = game_info(sgf_content)
        except Exception as e:
            failed[type(e).__name__] += 1
            continue
        n_ok += 1
        for p, v in props.items():
            present[p] += 1
            values[p][v] += 1
        per_rank[rank].append((n_moves, n_pass))
    moves = {}
    for lv in LEVELS:
        if lv in per_rank:
            a = np.array(per_rank[lv])
            moves[lv] = (int(np.median(a[:, 0])), a[:, 1].sum() / max(a[:, 0].sum(), 1))
    return n_ok, present, values, moves, failed


def parse_failures(sgfs):
    """用教學版的 SGFParseRankPrediction 解析，失敗分成兩類計數。

    回傳 (returned, raised)，兩者都是 Counter，鍵為「例外類型: 訊息前 80 字」：
        returned  features_from_sgf_content 回傳 err，抽特徵時會跳過這一局
        raised    落子階段直接丟出例外，抽特徵時整個 Pool 會中斷
    """
    from utils import SGFParseRankPrediction  # 只有解析 SGF 時才需要 sgfmill
    parser = SGFParseRankPrediction()
    returned, raised = collections.Counter(), collections.Counter()
    for s in sgfs:
        try:
            _, err = parser.features_from_sgf_content(s)
        except Exception as e:
            raised['%s: %s' % (type(e).__name__, str(e)[:80])] += 1
            continue
        if err is not None:
            returned['%s: %s' % (type(err).__name__, str(err)[:80])] += 1
    return returned, raised


def report(df, per_rank, seed):
    lines = ['## 欄位與筆數', '']
    missing = missing_columns(df)
    lines.append('- 欄位：%s' % ', '.join(df.columns))
    if missing:
        lines.append('- 缺少欄位：%s，停止檢查' % ', '.join(missing))
        return '\n'.join(lines)
    lines.append('- 總筆數：%d' % len(df))

    lines += ['', '## rank 與 color', '']
    for col in ('rank', 'color'):
        counts = df[col].value_counts()
        lines.append('- `%s`：%s' % (col, ', '.join('%s=%d' % kv for kv in counts.items())))
        lines.append('- `%s` 缺值：%d' % (col, df[col].isna().sum()))
    unknown = sorted(set(df['rank'].astype(str)) - set(LEVELS))
    if unknown:
        lines.append('- 不在 LEVELS 的 rank：%s' % unknown)
    bad_color = sorted(set(df['color'].astype(str)) - {'B', 'W'})
    if bad_color:
        lines.append('- 不是 B/W 的 color：%s' % bad_color)

    lines += ['', '## 玩家與局數', '', '| rank | 玩家數 | 局數最小 | 局數中位 | 局數最大 |',
              '|---|---|---|---|---|']
    for lv, r in player_stats(df).iterrows():
        lines.append('| %s | %d | %d | %g | %d |' % (lv, r['players'], r['games_min'],
                                                   r['games_median'], r['games_max']))
    lines.append('')
    n_cross, n_players = cross_rank_players(df)
    lines.append('- 跨等級的 player_id：%d / %d（%.2f%%）' % (n_cross, n_players, 100 * n_cross / max(n_players, 1)))

    n_sample = min(per_rank, int(df['rank'].value_counts().min()))
    sample = df.groupby('rank').sample(n=n_sample, random_state=seed)
    n_ok, present, values, moves, failed = header_and_moves(sample)
    lines += ['', '## SGF 標頭（每個等級抽 %d 局，成功解析 %d 局）' % (n_sample, n_ok), '',
              '- sgfmill 解析失敗：%d 局%s' % (sum(failed.values()), ''.join(
                  '，%s×%d' % kv for kv in failed.most_common())), '',
              '| property | 出現比例 | 常見值 |', '|---|---|---|']
    for p in sorted(present, key=lambda p: (p not in HEADER_PROPS, p)):
        top = ', '.join('`%s`×%d' % (v[:20], c) for v, c in values[p].most_common(5))
        lines.append('| %s | %.1f%% | %s |' % (p, 100 * present[p] / max(n_ok, 1), top))
    absent = [p for p in HEADER_PROPS if p not in present]
    if absent:
        lines.append('')
        lines.append('- 沒出現的 property：%s' % ', '.join(absent))
    if 'BR' in present or 'WR' in present:
        lines.append('- 警告：有 BR 或 WR，可能直接寫出棋力')

    lines += ['', '## 手數與 pass', '', '| rank | 手數中位數 | pass 比例 |', '|---|---|---|']
    for lv, (med, pass_ratio) in moves.items():
        lines.append('| %s | %d | %.3f%% |' % (lv, med, 100 * pass_ratio))

    returned, raised = parse_failures(sample['sgf_content'])
    lines += ['', '## SGFParseRankPrediction 解析', '']
    for label, errors in (('回傳 err（抽特徵時跳過該局）', returned),
                          ('丟出例外（抽特徵時整個 Pool 中斷）', raised)):
        n = sum(errors.values())
        lines.append('- %s：%d / %d（%.2f%%）' % (label, n, len(sample), 100 * n / max(len(sample), 1)))
        lines += ['  - `%s` ×%d' % kv for kv in errors.most_common(10)]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train-dir', default='training')
    ap.add_argument('--per-rank', type=int, default=1000, help='SGF 相關統計每個等級抽幾局')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    print(report(load_dataframe(args.train_dir), args.per_rank, args.seed))


if __name__ == '__main__':
    main()
