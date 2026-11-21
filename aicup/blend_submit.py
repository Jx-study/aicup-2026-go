"""提交檔：(1 - w) × (A + B) / 2 + w × TF，用 pick_level() 選等級；w 預設 1/3。

輸入：
  --a、--b      模型 A（6d）、模型 B（2d loss）的 test 機率（predict_pool --save-probs 的 .npz：question_id、probs）
  --tf-dir      final 模式的輸出，讀 final-ts{0,1,2}-test.npy 取機率平均；給多個目錄時再彼此平均
                列順序是 pub_q_0001～0400、priv_q_0001～0400
  --tf-weight   TF 的權重 w
  --ref         比較用的提交檔，可給多個
輸出 --out-dir 下的 probs_avg.npz（question_id、probs）與 submission_rank.csv。
question_id 順序不符、形狀不對、有 NaN、少了任一個訓練 seed 時拒絕執行。
用法：python -m aicup.blend_submit --tf-dir humansl-analysis/tf-final-6d/out humansl-analysis/tf-final-2d/out --out-dir rank-trained-models/fusion
"""
import argparse
import os

import numpy as np
import pandas as pd

from aicup.scoring import LEVELS, pick_level

BASE_DIR = 'rank-trained-models/9ba3725-3.8/'
ORDER = ['pub_q_%04d' % i for i in range(1, 401)] + ['priv_q_%04d' % i for i in range(1, 401)]


def load(path):
    z = np.load(path, allow_pickle=False)
    q, p = z['question_id'].astype(str), z['probs'].astype(np.float64)
    if list(q) != ORDER:
        raise ValueError('%s 的 question_id 順序不符' % path)
    return p


def check(p, name):
    if p.shape != (len(ORDER), len(LEVELS)):
        raise ValueError('%s 的形狀是 %s' % (name, p.shape))
    if np.isnan(p).any():
        raise ValueError('%s 有 NaN' % name)
    if not np.allclose(p.sum(1), 1, atol=1e-3):
        raise ValueError('%s 的每列機率和不是 1' % name)
    return p


def blend(a, b, tf, w):
    """(1 - w) × (A + B) / 2 + w × TF。"""
    return (1 - w) * (a + b) / 2 + w * tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default=BASE_DIR + 'model-a/probs.npz')
    ap.add_argument('--b', default=BASE_DIR + 'model-b/probs.npz')
    ap.add_argument('--tf-dir', required=True, nargs='+', help='給多個時先各自平均 seed，再彼此平均')
    ap.add_argument('--train-seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--tf-weight', type=float, default=1 / 3)
    ap.add_argument('--ref', nargs='*', default=[BASE_DIR + 'submission_rank.csv', 'rank-trained-models/fusion-3.8-tf05/submission_rank.csv'])
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    a, b = check(load(args.a), 'A'), check(load(args.b), 'B')
    tfs = []
    for d in args.tf_dir:
        paths = [os.path.join(d, 'final-ts%d-test.npy' % ts) for ts in args.train_seeds]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError('缺少 %s' % missing)
        tfs.append(check(np.mean([np.load(p).astype(np.float64) for p in paths], 0), 'TF ' + d))
    tf = np.mean(tfs, 0)  # 各目錄先平均 3 個訓練 seed，再彼此平均
    w = args.tf_weight
    probs = blend(a, b, tf, w)
    sub = pd.DataFrame({'question_id': ORDER, 'pred_rank': [LEVELS[i] for i in pick_level(probs)]})
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, 'probs_avg.npz'), question_id=np.array(ORDER), probs=probs)
    sub.to_csv(os.path.join(args.out_dir, 'submission_rank.csv'), index=False)
    print('TF 權重 %.4f；寫出 %s（%d 列）' % (w, os.path.join(args.out_dir, 'submission_rank.csv'), len(sub)))
    print('預測分布：' + '、'.join('%s %d' % (k, v) for k, v in sub['pred_rank'].value_counts().items()))
    print('TF 單獨每級被選比例：' + ' '.join('%.3f' % x for x in np.bincount(pick_level(tf), minlength=10) / len(ORDER)))
    for r in args.ref:
        if os.path.exists(r):
            ref = pd.read_csv(r).set_index('question_id').loc[ORDER, 'pred_rank'].values
            print('和 %s 一致率 %.1f%%（改答 %d 題）' % (r, 100 * (ref == sub['pred_rank'].values).mean(), (ref != sub['pred_rank'].values).sum()))


if __name__ == '__main__':
    main()
