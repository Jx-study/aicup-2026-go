"""兩個第一段模型的每題 10 級機率取算術平均，用 pick_level() 選等級，寫出提交檔。

輸入是 predict_pool --save-probs 的 .npz（question_id、probs），權重固定各 0.5。
兩份的 question_id 順序或內容不同、任一題機率有 NaN 時拒絕執行。

用法（在 repo 根目錄）：
    python -m aicup.average_probs --probs a.npz b.npz --out-dir rank-trained-models/<commit>-3.7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.scoring import LEVELS, pick_level


def load_probs(path):
    """回傳 (question_id 陣列, (N, 10) 機率)。"""
    z = np.load(path, allow_pickle=False)
    return z['question_id'].astype(str), z['probs'].astype(np.float64)


def average_probs(qids_a, probs_a, qids_b, probs_b):
    """兩份逐題機率的算術平均，回傳 (N, 10)。

    question_id 不完全相同（含順序）、形狀不是 (N, 10) 或有 NaN 時丟 ValueError。
    """
    if len(qids_a) != len(qids_b) or (np.asarray(qids_a) != np.asarray(qids_b)).any():
        raise ValueError('兩份機率檔的 question_id 不同')
    for p in (probs_a, probs_b):
        if p.shape != (len(qids_a), len(LEVELS)):
            raise ValueError('機率形狀是 %s，應為 (%d, %d)' % (p.shape, len(qids_a), len(LEVELS)))
        if np.isnan(p).any():
            raise ValueError('機率有 NaN（%d 題）' % int(np.isnan(p).any(axis=1).sum()))
    return 0.5 * probs_a + 0.5 * probs_b


def submission(qids, probs):
    return pd.DataFrame({'question_id': qids, 'pred_rank': [LEVELS[i] for i in pick_level(probs)]})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--probs', nargs=2, required=True, help='兩份 predict_pool --save-probs 的 .npz')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ref-subs', nargs='*', default=[], help='比對預測一致率的提交檔')
    args = ap.parse_args()

    (qa, pa), (qb, pb) = load_probs(args.probs[0]), load_probs(args.probs[1])
    avg = average_probs(qa, pa, qb, pb)
    sub = submission(qa, avg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'submission_rank_avg.csv'
    sub.to_csv(path, index=False)
    np.savez(out_dir / 'probs_avg.npz', question_id=qa, probs=avg)
    print('寫出 %s（%d 列）' % (path, len(sub)))
    print('預測分布:', {lv: int((sub['pred_rank'] == lv).sum()) for lv in LEVELS})
    for name, p in zip(args.probs, (pa, pb)):
        print('和 %s 單獨選等級的一致率 %.1f%%' % (name, 100 * (submission(qa, p)['pred_rank'] == sub['pred_rank']).mean()))
    for ref_path in args.ref_subs:
        ref = pd.read_csv(ref_path)
        if list(ref['question_id']) != list(sub['question_id']):
            raise ValueError('%s 的 question_id 和輸出不同' % ref_path)
        print('和 %s 的一致率 %.1f%%' % (ref_path, 100 * (ref['pred_rank'] == sub['pred_rank']).mean()))


if __name__ == '__main__':
    main()
