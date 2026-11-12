import numpy as np
import pandas as pd
import pytest

from aicup import setmodel_trunk as st
from aicup.scoring import LEVELS


def test_pick_turns_range_and_count():
    turns = list(range(0, 400, 2))  # 目標玩家的回合
    picked = st.pick_turns(turns)
    assert len(picked) == st.PER_GAME
    assert all(st.FIRST_MOVE <= t + 1 <= st.LAST_MOVE for t in picked)
    assert picked == sorted(set(picked))
    assert picked[0] == 30 and picked[-1] == 178  # 範圍內第一手與最後一手


def test_pick_turns_short_and_empty():
    assert st.pick_turns([0, 2, 4]) == []
    few = [30, 32, 34]
    assert st.pick_turns(few) == few
    assert st.pick_turns([]) == []


def make_df(n_per_level, games):
    rows = []
    for rank in LEVELS:
        for p in range(n_per_level):
            rows += [{'rank': rank, 'player_id': '%s%03d' % (rank, p)} for _ in range(games + p % 3 - 2)]
    return pd.DataFrame(rows)


def test_select_train_players_prefix_and_counts():
    df = make_df(30, 20)
    full = st.select_train_players(df, 10)
    small = st.select_train_players(df, 5)
    assert len(full) == 10 * len(LEVELS)
    assert set(full['rank']) == set(LEVELS)
    for rank in LEVELS:  # 縮減時取同一份名單的前綴
        assert list(small[small['rank'] == rank]['player_id']) == list(full[full['rank'] == rank]['player_id'][:5])
    counts = df.groupby(['rank', 'player_id']).size()
    assert all(counts[(r, p)] >= st.MIN_GAMES for p, r in zip(full['player_id'], full['rank']))


def test_select_train_players_rejects_short_level():
    with pytest.raises(SystemExit):
        st.select_train_players(make_df(5, 20), 10)


def test_pool_trunk():
    torch = pytest.importorskip('torch')
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    v = st.pool_trunk(x)
    assert v.shape == (2, 6)
    assert torch.allclose(v[:, :3], x.mean(dim=(2, 3)))
    assert torch.allclose(v[:, 3:], x.amax(dim=(2, 3)))


def test_write_and_load_vectors(tmp_path):
    games = [{'key': 'a', 'player_id': 'p', 'rank_idx': 1}, {'key': 'b', 'player_id': 'p', 'rank_idx': 1},
             {'key': 'c', 'player_id': 'q', 'rank_idx': 2}]
    results = [('a', np.zeros((2, 1)), None), ('b', None, None), ('c', np.zeros((1, 1)), None)]
    vecs = np.arange(3 * st.DIM, dtype=np.float16).reshape(3, st.DIM)
    st.write_part(st.part_path(str(tmp_path), 0), games, results, vecs)
    got = st.load_vectors(str(tmp_path))
    assert set(got) == {'a', 'c'}
    assert np.array_equal(got['a'], vecs[:2]) and np.array_equal(got['c'], vecs[2:])


def test_split_val_size_covers_dev_and_holdout():
    # 開發組與封存組取自 180000 的切分，train 池要用同一個大小
    assert st.SPLIT_VAL_SIZE == 180_000


def test_check_config_rejects_change(tmp_path):
    import argparse
    args = argparse.Namespace(set='train', players_per_rank=200, chunk=200, keys_from=None, out=str(tmp_path))
    st.check_config(args)
    st.check_config(args)
    args.players_per_rank = 100
    with pytest.raises(SystemExit):
        st.check_config(args)


def test_load_vectors_ignores_tmp(tmp_path):
    (tmp_path / 'tmp-part-000001.npz').write_bytes(b'broken')
    assert st.load_vectors(str(tmp_path)) == {}
