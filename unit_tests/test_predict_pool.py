import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aicup import predict_pool as pp
from aicup.scoring import LEVELS
from stack_helpers import write_rank_tests
from test_stack_hgb import toy_files


def test_align_test_hsl_aligns_by_keys_and_pads_nan(tmp_path):
    target, opp = toy_files(tmp_path)
    data = pp.load_games(target)
    keys = list(data['key'][:2]) + ['nonexistent_key']
    out = pp.align_test_hsl(data, opp, keys)
    assert out.shape == (3, 8)
    assert not np.isnan(out[0]).any()
    assert not np.isnan(out[1]).any()
    assert np.isnan(out[2]).all()


def test_predict_pool_smoke_and_empty_question_fallback(tmp_path, monkeypatch):
    # 建立 toy 開發組、train 池與測試集資料
    dev_target, dev_opp = toy_files(tmp_path, name='dev', n_players=10)
    pool_target, pool_opp = toy_files(tmp_path, name='pool', n_players=10, prefix='q', seed=7)
    test_target, test_opp = toy_files(tmp_path, name='test', n_players=6, prefix='t', seed=8)

    dev_data = pp.load_games(dev_target)
    np.savez(tmp_path / 'val_probs.npz', key=dev_data['key'], probs=np.full((len(dev_data['key']), 10), 0.1),
             rank_idx=dev_data['rank_idx'])

    test_data = pp.load_games(test_target)
    test_dir = tmp_path / 'tests'
    # 2 道題：public 1 題（2 局），private 1 題（0 局）
    write_rank_tests(test_dir, {
        'public': [([test_data['key'][0], test_data['key'][1]], 'B')],
        'private': [([], 'B')],
    })

    np.savez(tmp_path / 'game_probs_test.npz',
             game_q=np.array([0, 0], np.int64),
             probs=np.full((2, 10), 0.1),
             ckpt_sha256='')

    exclude_csv = tmp_path / 'exclude.csv'
    exclude_csv.write_text('player_id\nnone\n', encoding='utf-8')
    out_dir = tmp_path / 'out'

    monkeypatch.setattr('sys.argv', [
        'predict_pool',
        '--val-jsonl', dev_target, '--val-opp', dev_opp,
        '--val-probs', str(tmp_path / 'val_probs.npz'),
        '--pool-jsonl', pool_target, '--pool-opp', pool_opp,
        '--exclude-players', str(exclude_csv),
        '--test-jsonl', test_target, '--test-opp', test_opp,
        '--test-probs', str(tmp_path / 'game_probs_test.npz'),
        '--ckpt', '',
        '--test-dir', str(test_dir),
        '--out-dir', str(out_dir),
        '--sizes', '2', '--train-repeats', '1', '--inner-folds', '2',
        '--ref-sub', '',
    ])
    pp.main()

    sub_pool = pd.read_csv(out_dir / 'submission_rank_pool.csv')
    sub_s1 = pd.read_csv(out_dir / 'submission_rank_pool_s1.csv')
    sub_def = pd.read_csv(out_dir / 'submission_rank.csv')

    assert len(sub_pool) == 2
    assert len(sub_s1) == 2
    assert sub_pool['pred_rank'].iloc[0] in LEVELS
    assert sub_pool.equals(sub_def)
