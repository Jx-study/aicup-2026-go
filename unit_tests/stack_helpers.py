"""stacking 相關測試共用的玩具資料與測試題檔案。"""
import numpy as np
import pandas as pd


def toy_data(n_players=50, games=20, seed=0):
    """預設每級 5 位玩家，Human SL 特徵與 CNN 機率都帶有等級訊號。"""
    rng = np.random.default_rng(seed)
    rank = np.repeat(np.arange(n_players) % 10, games)
    player = np.repeat(['p%d' % i for i in range(n_players)], games)
    x = rank[:, None] + rng.normal(0, 2, (len(rank), 4))
    logits = -0.5 * (np.arange(10)[None, :] - rank[:, None] - rng.normal(0, 2, (len(rank), 1))) ** 2
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    keys = np.array(['k%d' % i for i in range(len(rank))])
    data = {'x': x, 'rank_idx': rank.astype(np.int64), 'player_id': player, 'key': keys,
            'n_profiles': 3, 'has_loss': False}
    return data, probs


def write_npz(path, keys, probs, rank_idx):
    np.savez(path, key=keys, probs=probs, rank_idx=rank_idx)


def write_rank_tests(test_dir, questions):
    """questions 為 {split: [(sgf 清單, color), ...]}，寫出 rank_prediction_test_{split}.csv。"""
    test_dir.mkdir(exist_ok=True)
    for split, qs in questions.items():
        rows = []
        for i, (sgfs, c) in enumerate(qs):
            row = {'question_id': '%s_q_%04d' % ('pub' if split == 'public' else 'priv', i + 1),
                   'num_games': len(sgfs)}
            for k in range(1, 21):
                row['sgf_%d' % k] = sgfs[k - 1] if k <= len(sgfs) else ''
            for k in range(1, 21):
                row['color_%d' % k] = c if k <= len(sgfs) else ''
            rows.append(row)
        pd.DataFrame(rows).to_csv(test_dir / ('rank_prediction_test_%s.csv' % split), index=False)

