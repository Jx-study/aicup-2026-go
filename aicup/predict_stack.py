"""stacking 推論：用 val 玩家擬合 CNN 與 Human SL 的 stacking，預測競賽二的測試題，輸出提交檔。

擬合：
    val jsonl 的全部玩家都用來擬合，出題方式同 eval_posthoc，C 依玩家切內層折選，擬合函式同 stack_humansl。
    val_probs.npz 要和 val jsonl 出自同一個 CNN 沒訓練過的 val 玩家。
預測：
    每題的特徵是 CNN 各局機率以溫度縮放、log 合併後的 log 機率，接各局 Human SL 特徵組 A 的平均。
    CNN 各局機率讀 predict_rank 的快取，Human SL 讀 analyze_humansl --test-dir 的輸出。
    缺 Human SL 結果或 CNN 機率為 NaN 的局不參與合併；整題都無效時回傳最弱等級，同 predict_rank。

用法（在 repo 根目錄）：
    python -m aicup.predict_stack --val-jsonl humansl-analysis/val648-p3.jsonl \\
        --val-probs rank-trained-models/<commit hash>/val_probs.npz \\
        --test-jsonl humansl-analysis/test-p3.jsonl \\
        --test-probs rank-trained-models/<commit hash>/game_probs_test.npz \\
        --ckpt rank-trained-models/<commit hash>/best_model.pth \\
        --out rank-trained-models/<stacking 的 commit hash>/submission_rank.csv
"""
import argparse

import numpy as np
import pandas as pd

from aicup.analyze_humansl import test_game_keys
from aicup.eval_posthoc import SIZES, parse_sizes
from aicup.fit_humansl import level_probs, load_games, questions_of
from aicup.predict_rank import file_sha256, read_questions
from aicup.scoring import LEVELS, pick_level
from aicup.stack_humansl import fit_model, hsl_columns, load_cnn_probs, prior_class_weight, question_features


def fit_stack(data, cnn_probs, sizes, train_repeats, temp, seed, inner_folds=5, prior='keep'):
    """用 data 的全部玩家出題並擬合，回傳 (model, 選到的 C)。

    prior 為 balanced 時用 class_weight 去掉 val 玩家的等級先驗，同 stack_humansl 的 --prior balanced。
    """
    rng = np.random.default_rng(seed)
    q, y, players = questions_of(data, np.ones(len(data['x']), bool), train_repeats, rng, sizes)
    return fit_model(data['x'][:, hsl_columns(data)], cnn_probs, q, y, players, temp, inner_folds=inner_folds,
                     class_weight=prior_class_weight(prior))


def check_profiles(fit_data, test_data):
    """擬合與測試的 profile 要相同；兩邊都有名稱時比對名稱與順序，否則只比對數量。不同時丟 ValueError。"""
    a, b = fit_data.get('profiles'), test_data.get('profiles')
    if a is not None and b is not None:
        if a != b:
            raise ValueError('測試 jsonl 的 profile 是 %s，擬合資料是 %s' % (list(b), list(a)))
    elif fit_data['n_profiles'] != test_data['n_profiles']:
        raise ValueError('測試 jsonl 有 %d 個 profile，擬合資料有 %d 個'
                         % (test_data['n_profiles'], fit_data['n_profiles']))


def align_test(test_data, keys, game_probs):
    """回傳 (Human SL 特徵組 A, 每局是否可用)，特徵依 keys 排列。

    缺 Human SL 結果或 CNN 機率有 NaN 的局不可用，特徵列填 NaN。
    """
    index = {k: i for i, k in enumerate(test_data['key'])}
    cols = hsl_columns(test_data)
    x = np.full((len(keys), len(cols)), np.nan)
    has_hsl = np.array([k in index for k in keys])
    x[has_hsl] = test_data['x'][[index[k] for k in np.asarray(keys)[has_hsl]]][:, cols]
    return x, has_hsl & ~np.isnan(game_probs).any(axis=1)


def read_test_probs(path, game_q, ckpt=None):
    """讀 predict_rank 的快取，回傳 ((N, 10) 機率, checkpoint 的 SHA-256)。

    局的順序和 game_q 不同、開了 TTA，或給定 ckpt 而雜湊不符時丟 ValueError。
    """
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z['game_q'], game_q):
            raise ValueError('%s 與目前的測試資料對不上' % path)
        if 'tta' in z.files and bool(z['tta']):
            raise ValueError('%s 開了 TTA，和 val_probs.npz 的條件不同' % path)
        sha = str(z['ckpt_sha256']) if 'ckpt_sha256' in z.files else ''
        probs = z['probs'].astype(np.float64)
    if ckpt and file_sha256(ckpt) != sha:
        raise ValueError('%s 不是 %s 產生的' % (path, ckpt))
    return probs, sha


def predict(model, x_test, test_probs, game_q, valid, n_questions, temp):
    """回傳 (每題的等級 index, 沒有可用局的題數)。只用 valid 的局合併，沒有可用局的題目回傳 0。"""
    questions = [np.flatnonzero((game_q == q) & valid) for q in range(n_questions)]
    has = np.array([len(q) > 0 for q in questions])
    preds = np.zeros(n_questions, np.int64)
    if has.any():
        kept = [q for q, h in zip(questions, has) if h]
        preds[has] = pick_level(level_probs(model, question_features(x_test, test_probs, kept, temp)))
    return preds, int((~has).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--val-jsonl', required=True, help='analyze_humansl 對 CNN val 玩家的輸出')
    ap.add_argument('--val-probs', required=True, help='CNN 的 val_probs.npz')
    ap.add_argument('--test-jsonl', required=True, help='analyze_humansl --test-dir 的輸出')
    ap.add_argument('--test-probs', required=True, help='predict_rank 的每局機率快取，不開 TTA')
    ap.add_argument('--ckpt', help='產生 val_probs.npz 的 best_model.pth；給定時檢查 --test-probs 出自同一個 checkpoint')
    ap.add_argument('--test-dir', default='tests', help='含 rank_prediction_test_{public,private}.csv 的目錄')
    ap.add_argument('--out', default='submission_rank.csv')
    ap.add_argument('--cnn-temp', type=float, default=3.0, help='CNN 每局機率的溫度，預設同 predict_rank')
    ap.add_argument('--sizes', type=parse_sizes, default=SIZES, help='擬合用的每題局數，逗號分隔')
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--inner-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prior', default='keep', choices=('keep', 'balanced'),
                    help='balanced 時不帶 val 玩家的等級先驗')
    args = ap.parse_args()

    qids, games = read_questions(args.test_dir)
    game_q = np.array([q for q, _, _ in games], np.int64)
    test_probs, sha = read_test_probs(args.test_probs, game_q, args.ckpt)
    print('測試局機率的 checkpoint SHA-256：%s%s' % (sha[:12] or '未記錄', '（已比對 --ckpt）' if args.ckpt else ''))
    data = load_games(args.val_jsonl)
    test_data = load_games(args.test_jsonl)
    check_profiles(data, test_data)
    x_test, valid = align_test(test_data, test_game_keys(qids, games), test_probs)
    print('測試：%d 題，%d 局，可用 %d 局' % (len(qids), len(games), valid.sum()))

    cnn = load_cnn_probs(args.val_probs, data['key'], data['rank_idx'])
    model, c = fit_stack(data, cnn, args.sizes, args.train_repeats, args.cnn_temp, args.seed, args.inner_folds,
                         args.prior)
    print('擬合：%d 局，%d 位玩家，選到 C=%g，prior %s' % (len(data['x']), len(set(data['player_id'])), c, args.prior))

    preds, empty = predict(model, x_test, test_probs, game_q, valid, len(qids), args.cnn_temp)
    if empty:
        print('警告：%d 題沒有可用的局，預測為 %s' % (empty, LEVELS[0]))
    sub = pd.DataFrame({'question_id': qids, 'pred_rank': [LEVELS[i] for i in preds]})
    sub.to_csv(args.out, index=False)
    print('寫出 %s（%d 列）' % (args.out, len(sub)))
    print('預測分布:', sub['pred_rank'].value_counts().to_dict())


if __name__ == '__main__':
    main()
