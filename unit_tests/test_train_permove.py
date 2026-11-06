import argparse
import json
import os

import numpy as np
import pytest

from aicup import extract_permove as ep
from aicup import katago_torch as kt
from aicup import symmetry
from aicup.analyze_humansl import parse_game, target_turns

torch = pytest.importorskip('torch')

HAS_KATAGO = os.path.exists(os.path.join(os.environ.get('KATAGO_PYTHON') or kt.DEFAULT_KATAGO_PYTHON, 'features.py'))
pytestmark = pytest.mark.skipif(not HAS_KATAGO, reason='沒有 KataGo python/ 原始碼')

from aicup import infer_permove as ip  # noqa: E402
from aicup import stack_humansl as sh  # noqa: E402
from aicup import train_permove as tp  # noqa: E402

SGF = '(;SZ[19]HD[0];B[pd];W[dp];B[pp];W[dd];B[fc];W[cf];B[qj];W[jq];B[nq];W[qn];B[ba];W[aa];B[ab])'


def toy_data(n_games=3):
    """幾局的分片欄位（兩側都有），rank_idx 依局遞增。"""
    moves = parse_game(SGF)['moves']
    parts = []
    for i in range(n_games):
        g = {'key': 'D_g_%d' % i, 'rank_idx': i, 'player_id': 'p%d' % i, 'color': 'BW'[i % 2], 'sgf': SGF}
        sides = {0: target_turns(moves, g['color']), 1: target_turns(moves, ep.OPP[g['color']])}
        out, why = ep.encode_job(ep.game_job(g, sides))
        assert why is None
        parts.append(out)
    return {k: np.concatenate([p[k] for p in parts]) for k in ep.FIELDS}


def test_unpack_bits_matches_numpy():
    data = toy_data(1)
    got = tp.unpack_bits(torch.from_numpy(data['spatial']), tp.SPATIAL_BITS).numpy()
    np.testing.assert_array_equal(got, np.unpackbits(data['spatial'], axis=1, count=tp.SPATIAL_BITS))


def test_level_log_probs_masked():
    logits = torch.randn(2, 3, 362)
    legal = torch.ones(2, 362, dtype=torch.bool)
    legal[:, 5] = False
    move = torch.tensor([7, 361])
    got = tp.level_log_probs(logits, legal, move)
    masked = logits.masked_fill(~legal[:, None], float('-inf'))
    want = torch.log_softmax(masked, -1)[torch.arange(2)[:, None], torch.arange(3)[None], move[:, None]]
    torch.testing.assert_close(got, want)
    assert got.shape == (2, 3)


def test_wsd_factor():
    total = 100
    f = [tp.wsd_factor(s, total, 0.02, 0.2) for s in range(total)]
    assert f[0] == 0.5 and f[1] == 1.0  # warmup 2 步
    assert f[50] == 1.0 and f[79] == 1.0
    assert f[80] == pytest.approx(1.0) and f[99] == pytest.approx(0.05)
    assert all(a >= b for a, b in zip(f[80:], f[81:]))


def test_symmetry_tables_consistent():
    tables = tp.symmetry_tables()
    assert tables.shape == (8, 362)
    for k in range(8):
        assert sorted(tables[k]) == list(range(362)) and tables[k][361] == 361
        grid = np.zeros((19, 19), np.float32)
        grid.reshape(-1)[60] = 1
        assert tables[k][60] == symmetry.apply(grid, k).argmax()


class Equivariant(torch.nn.Module):
    """每格 logit 只看該格的平面（1×1 conv），pass 是全域常數：對盤面對稱是等變的。"""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(kt.N_SPATIAL, tp.N_LEVELS, 1)
        self.pass_logit = torch.nn.Parameter(torch.randn(tp.N_LEVELS))

    def forward(self, spatial, glob, meta):
        board = self.conv(spatial).flatten(2)
        return torch.cat([board, self.pass_logit.view(1, -1, 1).expand(len(spatial), -1, 1)], dim=2)


@pytest.mark.parametrize('sym', range(8))
def test_loss_invariant_under_symmetry(sym):
    torch.manual_seed(0)
    batcher = tp.Batcher(toy_data(), torch.device('cpu'))
    idx = torch.arange(batcher.n)
    model = Equivariant()
    meta = torch.zeros(tp.kt.N_META)
    base = tp.losses(model, meta, batcher.get(idx, 0), 1.0, 'none')
    moved = tp.losses(model, meta, batcher.get(idx, sym), 1.0, 'none')
    for a, b in zip(base, moved):
        torch.testing.assert_close(a, b)
    # 對稱後實際落子仍在合法點上
    _, _, legal, move, _ = batcher.get(idx, sym)
    assert legal[torch.arange(len(move)), move].all()


def test_check_moves_legal():
    data = toy_data(1)
    tp.check_moves_legal(data)
    data['legal'] = np.zeros_like(data['legal'])
    with pytest.raises(ValueError):
        tp.check_moves_legal(data)


def test_game_features_split_by_side():
    data = toy_data(2)
    logp = np.zeros((len(data['move']), tp.N_LEVELS))
    logp[data['side'] == 0] = -1.0
    logp[data['side'] == 1] = -2.0
    logp[(data['side'] == 0) & (data['key'] == 'D_g_1'), 3] = -5.0
    rows = ip.game_features(data, logp)
    assert [r['key'] for r in rows] == ['D_g_0', 'D_g_1']
    assert rows[0]['pm'] == [-1.0] * 10 and rows[0]['pmo'] == [-2.0] * 10
    assert rows[1]['pm'][3] == -5.0 and rows[1]['pm'][0] == -1.0  # pm 只平均目標側
    assert rows[1]['color'] == 'W' and rows[1]['rank_idx'] == 1 and rows[1]['hd'] == 0
    assert rows[0]['n_pm'] == (data['side'][data['key'] == 'D_g_0'] == 0).sum()


def write_pm(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def test_load_pm_blocks_aligns_by_key(tmp_path):
    p = tmp_path / 'pm.jsonl'
    write_pm(p, [{'key': 'b', 'rank_idx': 1, 'pm': [1.0] * 10, 'pmo': [2.0] * 10},
                 {'key': 'a', 'rank_idx': 0, 'pm': [3.0] * 10, 'pmo': [4.0] * 10}])
    blocks = sh.load_pm_blocks(str(p), np.array(['a', 'b']), np.array([0, 1]))
    assert blocks['pm'][:, 0].tolist() == [3.0, 1.0] and blocks['pmo'][:, 0].tolist() == [4.0, 2.0]
    with pytest.raises(ValueError):  # 等級不一致
        sh.load_pm_blocks(str(p), np.array(['a', 'b']), np.array([0, 2]))
    with pytest.raises(ValueError):  # 缺局
        sh.load_pm_blocks(str(p), np.array(['a', 'c']), np.array([0, 1]))
    write_pm(p, [{'key': 'a', 'rank_idx': 0, 'pm': [3.0] * 10, 'pmo': None}])
    assert set(sh.load_pm_blocks(str(p), np.array(['a']), np.array([0]))) == {'pm'}


def test_load_pm_blocks_rejects_partial_pmo(tmp_path):
    p = tmp_path / 'pm.jsonl'
    write_pm(p, [{'key': 'a', 'rank_idx': 0, 'pm': [1.0] * 10, 'pmo': [2.0] * 10},
                 {'key': 'b', 'rank_idx': 1, 'pm': [1.0] * 10, 'pmo': None}])
    with pytest.raises(ValueError, match='對手側'):
        sh.load_pm_blocks(str(p), np.array(['a', 'b']), np.array([0, 1]))


def test_parse_cs():
    assert sh.parse_cs('0.001,0.003,0.01') == (0.001, 0.003, 0.01)
    for bad in ('', '0', '-1,1'):
        with pytest.raises(argparse.ArgumentTypeError):
            sh.parse_cs(bad)


def test_resume_diff():
    saved = {k: 1 for k in tp.RESUME_KEYS}
    assert tp.resume_diff(saved, dict(saved, max_steps=5, eval_batch=9)) == {}
    assert tp.resume_diff(saved, dict(saved, batch=2)) == {'batch': (1, 2)}


def test_resolve_amp_cpu():
    cpu = torch.device('cpu')
    assert tp.resolve_amp('auto', cpu) == 'none'
    assert tp.resolve_amp('bf16', cpu) == 'bf16'


@pytest.mark.skipif(not os.path.exists(kt.CHECKPOINT), reason='沒有 humanv0 checkpoint')
def test_new_heads_equal_humanv0_and_train_eval_forward():
    batcher = tp.Batcher(toy_data(1), torch.device('cpu'))
    spatial, glob, _, _, _ = batcher.get(torch.arange(4))
    meta = torch.from_numpy(kt.profile_row(tp.METADATA_PROFILE)).expand(4, -1)
    with torch.no_grad():
        ref = kt.policy_logits(kt.load_humanv0(), spatial, glob, meta)
        model = tp.from_humanv0()
        assert not model.norm_trunkfinal.training
        out_train = model(spatial, glob, meta)
        out_eval = model.eval()(spatial, glob, meta)
    assert out_train.shape == (4, tp.N_LEVELS, 362)
    torch.testing.assert_close(out_eval, ref[:, None].expand_as(out_eval))
    torch.testing.assert_close(out_train, out_eval)


@pytest.mark.skipif(not os.path.exists(kt.CHECKPOINT), reason='沒有 humanv0 checkpoint')
def test_load_permove_roundtrip(tmp_path):
    batcher = tp.Batcher(toy_data(1), torch.device('cpu'))
    spatial, glob, _, _, _ = batcher.get(torch.arange(4))
    meta = torch.from_numpy(kt.profile_row(tp.METADATA_PROFILE)).expand(4, -1)
    model = tp.from_humanv0().eval()
    path = str(tmp_path / 'model.pt')
    tp.save_checkpoint(path, model, None, 0, argparse.Namespace())
    with torch.no_grad():
        torch.testing.assert_close(tp.load_permove(path)(spatial, glob, meta), model(spatial, glob, meta))
