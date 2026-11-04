import io
import json
import pathlib
import tarfile

import numpy as np
import pandas as pd
import pytest

from aicup import strength_se as se
from aicup import symmetry
from aicup.analyze_humansl import parse_game


def moves_of(sgf):
    return parse_game(sgf)['moves']


def test_encode_p0_is_empty_board_black_to_move():
    moves = moves_of('(;GM[1]SZ[19];B[dd];W[pp])')
    f = se.encode(se.board_states(moves), moves, 0)
    assert f.shape == (18, 19, 19)
    assert f[:16].sum() == 0
    assert f[16].all() and not f[17].any()


def test_encode_turn_planes_and_own_stones():
    moves = moves_of('(;GM[1]SZ[19];B[dd];W[pp])')
    states = se.board_states(moves)
    bd, wp = (15, 3), (3, 15)  # dd、pp 的 sgfmill (row, col)
    f1 = se.encode(states, moves, 1)  # 黑下完，輪到白
    assert f1[17].all() and not f1[16].any()
    assert f1[0].sum() == 0 and f1[1][bd] == 1 and f1[1].sum() == 1
    assert f1[2:4].sum() == 0  # states[0] 是空盤
    f2 = se.encode(states, moves, 2)  # 白下完，輪到黑
    assert f2[16].all()
    assert f2[0][bd] == 1 and f2[1][wp] == 1 and f2[0].sum() == 1 and f2[1].sum() == 1
    assert f2[2][bd] == 1 and f2[3].sum() == 0  # 前一個盤面只有黑子
    assert f2[4:16].sum() == 0


def test_encode_keeps_eight_boards_of_history():
    sgf = '(;GM[1]SZ[19]' + ''.join(';%s[%s]' % ('BW'[i % 2], 'a' + 'abcdefghijk'[i]) for i in range(10)) + ')'
    moves = moves_of(sgf)
    states = se.board_states(moves)
    f = se.encode(states, moves, 10)
    for k in range(8):
        stones = 10 - k
        assert f[2 * k].sum() + f[2 * k + 1].sum() == stones


def test_pass_keeps_board_but_counts_as_state():
    moves = moves_of('(;GM[1]SZ[19];B[dd];W[];B[pp])')
    states = se.board_states(moves)
    assert len(states) == 4
    assert (states[2] == states[1]).all()
    f2 = se.encode(states, moves, 2)  # 白 pass 後輪到黑
    assert f2[16].all()
    assert (f2[0] == f2[2]).all()
    assert se.policy_index(None) == se.PASS_INDEX
    assert se.policy_index((15, 3)) == 15 * 19 + 3


def test_capture_removes_stone():
    # 黑 ca、bb、aa 圍住白 ba 後提掉
    moves = moves_of('(;GM[1]SZ[19];B[ca];W[ba];B[bb];W[pp];B[aa])')
    states = se.board_states(moves)
    ba = (18, 1)
    assert states[4][1][ba] == 1
    assert states[5][1][ba] == 0
    assert states[5][1].sum() == 1  # 只剩 pp


def test_occupied_point_is_skipped():
    moves = moves_of('(;GM[1]SZ[19];B[dd];W[dd])')
    states = se.board_states(moves)
    assert (states[2] == states[1]).all()


def test_handicap_game_uses_actual_colors_without_stones():
    sgf = '(;GM[1]SZ[19]HD[2];B[qd];W[dc];B[pq])'  # HD>=2 的黑白標記相反
    moves = moves_of(sgf)
    assert [c for c, _ in moves] == ['W', 'B', 'W']
    states = se.board_states(moves)
    assert states[0].sum() == 0  # 不放讓子石
    assert se.target_positions(moves, 'W') == [1, 3]
    f1 = se.encode(states, moves, 1)  # 白先下，輪到黑
    assert f1[16].all()
    assert f1[1][(15, 16)] == 1


def test_game_features_shapes():
    moves = moves_of('(;GM[1]SZ[19];B[dd];W[pp];B[dp])')
    assert se.game_features({'moves': moves, 'color': 'B'}).shape == (2, 18, 19, 19)
    assert se.game_features({'moves': moves, 'color': 'W'}).shape == (1, 18, 19, 19)
    assert se.game_features({'moves': [], 'color': 'B'}).shape == (0, 18, 19, 19)


def test_summarize_per_symmetry():
    out = se.summarize(np.array([[1.0, 0.0], [3.0, 4.0]]), np.full((2, 11), 1 / 11))
    assert out['n'] == 2
    assert out['score_mean'] == [2.0, 2.0] and out['score_std'] == [1.0, 2.0]
    assert len(out['sl_rank']) == 11
    assert se.summarize(np.zeros((0, 2)), np.zeros((0, 11)))['score_mean'] is None


class PositionModel:
    """假的 TorchScript 模型：score 依棋子位置加權，方向不同就不同。"""

    def __init__(self, torch):
        self.torch = torch
        self.w = torch.arange(19 * 19, dtype=torch.float32).reshape(19, 19)

    def __call__(self, t):
        score = (t[:, :16].sum(dim=1) * self.w).sum(dim=(1, 2)).unsqueeze(1)
        return {'score': score, 'rank': self.torch.full((len(t), 11), 1 / 11)}


def test_estimator_scores_each_symmetry_and_sl_once():
    torch = pytest.importorskip('torch')
    est = se.Estimator.__new__(se.Estimator)
    est.torch, est.device, est.dtype, est.symmetries = torch, torch.device('cpu'), torch.float32, 3
    est.se = PositionModel(torch)
    calls = []
    est.sl = lambda t: calls.append(t.clone()) or PositionModel(torch)(t)
    moves = moves_of('(;GM[1]SZ[19];B[ab];W[pp];B[dc])')
    x = se.game_features({'moves': moves, 'color': 'B'})
    score, rank = est.run(x)
    assert score.shape == (2, 3) and rank.shape == (2, 11)
    for k in range(3):
        want = est.se(torch.from_numpy(symmetry.apply(x, k)).float())['score'].flatten().numpy()
        np.testing.assert_allclose(score[:, k], want)
    assert score[0, 0] != score[0, 1]
    assert len(calls) == 1 and torch.equal(calls[0], torch.from_numpy(x).float())  # SL 只跑原方向


def test_estimator_rejects_bad_symmetries():
    with pytest.raises(ValueError):
        se.Estimator({}, 'cpu', 'float32', symmetries=9)


class FakeEstimator:
    symmetries = 2

    def __init__(self):
        self.calls = []

    def run(self, x):
        self.calls.append(len(x))
        score = x[:, :16].reshape(len(x), -1).sum(axis=1).astype(np.float32)  # 8 個歷史盤面的棋子數總和
        return np.stack([score, -score], axis=1), np.full((len(x), 11), 1 / 11, np.float32)


def test_run_chunk_splits_batches_back_to_games():
    games, feats = [], []
    for key, sgf, color in [('a', '(;GM[1]SZ[19];B[dd];W[pp];B[dp])', 'B'),
                            ('b', '(;GM[1]SZ[19];B[dd])', 'W'),
                            ('c', '(;GM[1]SZ[19];B[dd];W[pp])', 'W')]:
        g = {'key': key, 'player_id': 'p', 'rank_idx': 0, 'color': color, 'parsed': parse_game(sgf)}
        games.append(g)
        feats.append(se.game_features({'moves': g['parsed']['moves'], 'color': color}))
    est = FakeEstimator()
    rows = se.run_chunk(est, games, feats, batch=2, config=se.run_config(2, 'float32'))
    assert est.calls == [2, 1]
    assert [r['n'] for r in rows] == [2, 0, 1]
    assert rows[1]['score_mean'] is None
    assert rows[2]['models'] == se.MODELS and rows[2]['symmetries'] == 2 and rows[2]['dtype'] == 'float32'
    assert rows[2]['score_mean'][0] == -rows[2]['score_mean'][1]


def test_failed_row_keeps_game_with_no_positions():
    game = {'key': 'q_g01', 'player_id': 'q', 'rank_idx': -1, 'color': 'B', 'sgf': '(;GM[1]SZ[9]HD[3])'}
    row = se.failed_row(game, se.run_config(8, 'float32'))
    assert row['n'] == 0 and row['score_mean'] is None and row['parse_error'] is True
    assert row['hd'] == 3 and row['symmetries'] == 8


def write_csv(tmp_path):
    df = pd.DataFrame({'game_id': ['g_1', 'g_1'], 'rank': ['D', '6D'], 'player_id': ['0012', 'x'],
                       'color': ['B', 'W'], 'sgf_content': ['(;GM[1]SZ[19];B[dd])', '(;GM[1]SZ[19];B[pp])']})
    df.iloc[:1].to_csv(tmp_path / 'train_D.csv', index=False)
    df.iloc[1:].to_csv(tmp_path / 'train_6D.csv', index=False)
    return str(tmp_path / 'train_*.csv')


def write_keys(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return str(path)


def test_load_keyed_games_follows_key_order(tmp_path):
    csv = write_csv(tmp_path)
    keys = write_keys(tmp_path / 'k.jsonl', [{'key': '6D_g_1', 'player_id': 'x', 'rank_idx': 9},
                                             {'key': 'D_g_1'}])
    games = se.load_keyed_games(keys, csv)
    assert [g['key'] for g in games] == ['6D_g_1', 'D_g_1']
    assert games[1]['player_id'] == '0012'  # 前導 0 不能被讀成整數
    assert games[1]['rank_idx'] == 0 and games[1]['sgf'].endswith('B[dd])')
    assert len(se.load_keyed_games(keys, csv, limit=1)) == 1


def test_load_keyed_games_rejects_missing_or_mismatch(tmp_path):
    csv = write_csv(tmp_path)
    with pytest.raises(SystemExit, match='找不到'):
        se.load_keyed_games(write_keys(tmp_path / 'a.jsonl', [{'key': 'C_g_1'}]), csv)
    with pytest.raises(SystemExit, match='不一致'):
        se.load_keyed_games(write_keys(tmp_path / 'b.jsonl', [{'key': 'D_g_1', 'rank_idx': 3}]), csv)


def test_load_keyed_games_rejects_csv_without_hd(tmp_path):
    csv = write_csv(tmp_path)  # 棋譜沒有 HD，等同 v1.0
    keys = write_keys(tmp_path / 'k.jsonl', [{'key': 'D_g_1', 'hd': 0}])
    with pytest.raises(SystemExit, match='v1.1'):
        se.load_keyed_games(keys, csv)
    assert len(se.load_keyed_games(write_keys(tmp_path / 'n.jsonl', [{'key': 'D_g_1', 'hd': None}]), csv)) == 1


CONFIG = se.run_config(8, 'float32')


def test_done_keys_requires_same_config(tmp_path):
    out = tmp_path / 'o.jsonl'
    assert se.done_keys(str(out), CONFIG) == set()
    write_keys(out, [{'key': 'a', **CONFIG}])
    assert se.done_keys(str(out), CONFIG) == {'a'}
    for k, v in [('symmetries', 4), ('dtype', 'bfloat16'), ('models', ['other.pt'])]:
        with pytest.raises(SystemExit, match=k):
            se.done_keys(str(out), {**CONFIG, k: v})


def test_done_keys_drops_partial_last_line(tmp_path):
    out = tmp_path / 'o.jsonl'
    write_keys(out, [{'key': 'a', **CONFIG}])
    with open(out, 'a', encoding='utf-8') as f:
        f.write('{"key": "b", "mod')
    assert se.done_keys(str(out), CONFIG) == {'a'}
    assert out.read_text(encoding='utf-8') == json.dumps({'key': 'a', **CONFIG}) + '\n'


def test_done_keys_rejects_broken_middle_line(tmp_path):
    out = tmp_path / 'o.jsonl'
    out.write_text('{"key": \n' + json.dumps({'key': 'a', **CONFIG}) + '\n', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        se.done_keys(str(out), CONFIG)


def test_ensure_models_downloads_and_extracts_only_needed(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in se.MODELS + ['se_go.pt']:
            data = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    src = tmp_path / 'src.tar.gz'
    src.write_bytes(buf.getvalue())
    monkeypatch.setattr(se, 'ARCHIVE_BYTES', len(buf.getvalue()))
    monkeypatch.setattr(se, 'MODEL_BYTES', {m: len(m.encode()) for m in se.MODELS})
    model_dir = tmp_path / 'models'
    paths = se.ensure_models(str(model_dir), url=src.as_uri())
    assert sorted(p.name for p in model_dir.iterdir()) == sorted(se.MODELS + [se.ARCHIVE])
    sl = pathlib.Path(paths[se.SL_MODEL])
    assert sl.read_bytes() == se.SL_MODEL.encode()
    src.unlink()
    se.ensure_models(str(model_dir), url=src.as_uri())  # 已解出時不再下載
    sl.write_bytes(b'x')  # 解壓到一半的檔案
    se.ensure_models(str(model_dir), url=src.as_uri())  # 壓縮檔還在，重新解壓不用下載
    assert sl.read_bytes() == se.SL_MODEL.encode()


def test_ensure_models_rejects_wrong_size(tmp_path):
    src = tmp_path / 'src.tar.gz'
    src.write_bytes(b'abc')
    with pytest.raises(SystemExit, match='大小'):
        se.ensure_models(str(tmp_path / 'models'), url=src.as_uri())
