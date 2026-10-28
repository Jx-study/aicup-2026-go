import numpy as np
import pandas as pd

from aicup import data_check

SGF = '(;GM[1]SZ[19]KM[6.5]RU[Chinese];B[pd];W[dp];B[];W[pp])'
BAD_SGF = '(;GM[1]SZ[19];B[pd'
RAISING_SGF = '(;SZ[19];B[pd];W[pd])'  # 下在已有子的點，落子階段才丟出例外


def _df():
    rows = []
    for lv, players in (('D', ('p1', 'p2')), ('3D', ('p2', 'p3'))):
        for p in players:
            for g in range(3):
                rows.append({'player_id': p, 'game_id': '%s_%d' % (p, g), 'rank': lv,
                             'color': 'B', 'sgf_content': SGF})
    return pd.DataFrame(rows)


def test_missing_columns():
    assert data_check.missing_columns(_df()) == []
    assert data_check.missing_columns(_df().drop(columns='color')) == ['color']


def test_player_stats_orders_by_level():
    stats = data_check.player_stats(_df())
    assert stats.index.tolist() == ['D', '3D']
    assert stats.loc['D', 'players'] == 2 and stats.loc['D', 'games_max'] == 3


def test_cross_rank_players():
    assert data_check.cross_rank_players(_df()) == (1, 3)


def test_game_info_counts_moves_and_passes():
    props, n_moves, n_pass = data_check.game_info(SGF)
    assert props['KM'] == '6.5' and props['RU'] == 'Chinese'
    assert (n_moves, n_pass) == (4, 1)


def test_game_info_allows_space_before_bracket():
    props, _, _ = data_check.game_info('(;GM[1]SZ[19]KM [6.5];B[pd])')
    assert props['KM'] == '6.5'


def test_parse_failures_counts_bad_sgf():
    returned, raised = data_check.parse_failures([SGF, BAD_SGF, SGF])
    assert sum(returned.values()) == 1 and sum(raised.values()) == 0


def test_parse_failures_separates_raised():
    returned, raised = data_check.parse_failures([SGF, RAISING_SGF])
    assert sum(raised.values()) == 1 and sum(returned.values()) == 0


def test_report_stops_on_missing_column():
    text = data_check.report(_df().drop(columns='rank'), 2, 0)
    assert '缺少欄位：rank' in text and '玩家' not in text


def test_report_full():
    df = _df()
    df.loc[0, 'sgf_content'] = BAD_SGF
    per_rank = int(df['rank'].value_counts().min())  # 抽樣涵蓋全部列
    text = data_check.report(df, per_rank, 0)
    for expected in ('## rank 與 color', '## 玩家與局數', '## SGF 標頭', '## 手數與 pass',
                     '跨等級的 player_id：1 / 3', 'sgfmill 解析失敗：1 局',
                     '回傳 err（抽特徵時跳過該局）：1 / 12', '丟出例外（抽特徵時整個 Pool 中斷）：0 / 12'):
        assert expected in text


def test_report_handles_nan_color():
    df = _df()
    df.loc[0, 'color'] = np.nan
    text = data_check.report(df, 2, 0)
    assert '`color` 缺值：1' in text
