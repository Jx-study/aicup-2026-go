"""兩段式第二層加 train 池（及第一段單獨）推論：輸出競賽二測試集提交檔。

擬合：
    第一段：Human SL 特徵（hsl+hslo，8 欄），擬合玩家為開發組加 train 池全體玩家。
    第二段：CNN 塊（10 欄，溫度 3.0）加第一段 out-of-fold log 機率（10 欄），只在開發組上擬合。
預測：
    pool_s1：第一段模型直接對測試題預測，不使用 CNN。
    pool：第二段模型結合測試題的 CNN 與第一段 log 機率進行預測。
輸出：
    同時輸出 submission_rank_pool.csv 與 submission_rank_pool_s1.csv。
    --s1-only 時只擬合第一段、不讀任何 CNN 檔案，只輸出 submission_rank_pool_s1.csv。
    --loss（需 --s1-only）時第一段再加兩側 KataGo 失誤特徵（loss + losso），NaN 用擬合題的欄平均填補。
    --save-probs（需 --s1-only）時另存每題第一段的 10 級機率（.npz：question_id、probs），沒有可用局的題機率為 NaN。
    --profiles 指定兩側 jsonl 的 3 個 profile，預設 rank_11k,rank_1d,rank_2d。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from aicup.analyze_humansl import test_game_keys
from aicup.eval_posthoc import SIZES, parse_sizes
from aicup.fit_humansl import (CS, level_probs, load_games, new_model, player_keys,
                               questions_of)
from aicup.loss_features import N_LOSS, game_loss_features
from aicup.predict_rank import read_questions
from aicup.predict_stack import check_profiles, read_test_probs
from aicup.scoring import LEVELS, aggregate_games, apply_temperature, pick_level
from aicup.stack_hgb import (EPS, POOL_RNG, PROFILES, check_disjoint, column_means, fill_with, load_cnn_probs,
                             load_game_features, load_pool, nanmean, parse_profiles, question_matrix,
                             read_player_ids, select_c, stage1_log_probs, stack_groups)
from aicup.stack_humansl import hsl_columns, load_hsl_opp


def align_test_hsl(test_data, opp_path, keys):
    """讀測試集兩側 jsonl，依 keys 排列成 (N, 8) 的 (hsl + hslo) 特徵。"""
    x_tgt = test_data['x'][:, hsl_columns(test_data)]
    x_opp = load_hsl_opp(opp_path, test_data, allow_missing=True)
    x_both = np.hstack([x_tgt, x_opp])
    index = {k: i for i, k in enumerate(test_data['key'])}
    out = np.full((len(keys), x_both.shape[1]), np.nan)
    in_test = np.array([k in index for k in keys])
    out[in_test] = x_both[[index[k] for k in np.asarray(keys)[in_test]]]
    return out


def loss_by_keys(path, keys):
    """jsonl 每局的 game_loss_features 依 keys 排列成 (len(keys), N_LOSS)，不在檔案內的局為 NaN。"""
    with open(path, encoding='utf-8') as f:
        feats = {r['key']: game_loss_features(r) for r in map(json.loads, (line for line in f if line.strip()))}
    return np.array([feats.get(k, np.full(N_LOSS, np.nan)) for k in keys])


def predict_s1_only(args, model, dev_data, out_dir, fill_mean=None):
    """第一段模型直接預測測試題，寫出 submission_rank_pool_s1.csv。

    每題特徵是可用局的 (hsl + hslo) 平均，--loss 時再接兩側失誤特徵的題內 nanmean，剩下的 NaN 用 fill_mean 填補。
    可用局是 hsl + hslo 都有值的局；整題沒有可用局時預測最弱等級。
    """
    qids, games = read_questions(args.test_dir)
    game_q = np.array([q for q, _, _ in games], np.int64)
    test_data = load_games(args.test_jsonl)
    check_profiles(dev_data, test_data)
    keys = test_game_keys(qids, games)
    x = align_test_hsl(test_data, args.test_opp, keys)
    valid = ~np.isnan(x).any(axis=1)
    if args.loss:
        x = np.hstack([x, loss_by_keys(args.test_jsonl, keys), loss_by_keys(args.test_opp, keys)])
    print('測試集：%d 題，%d 局，可用 %d 局' % (len(qids), len(games), valid.sum()))
    q_idx = [np.flatnonzero((game_q == q) & valid) for q in range(len(qids))]
    has = np.array([len(i) > 0 for i in q_idx])
    preds = np.zeros(len(qids), np.int64)
    probs = np.full((len(qids), len(LEVELS)), np.nan)
    if has.any():
        f = np.array([nanmean(x[i], axis=0) for i, h in zip(q_idx, has) if h])
        if fill_mean is not None:
            f = fill_with(f, fill_mean)
        probs[has] = np.exp(stage1_log_probs(model, f))
        preds[has] = pick_level(probs[has])
    if args.save_probs:
        np.savez(args.save_probs, question_id=np.asarray(qids), probs=probs)
        print('每題機率 ->', args.save_probs)
    sub = pd.DataFrame({'question_id': qids, 'pred_rank': [LEVELS[i] for i in preds]})
    path = out_dir / 'submission_rank_pool_s1.csv'
    sub.to_csv(path, index=False)
    print('寫出 %s（%d 列），沒有可用局的題 %d 題' % (path, len(sub), int((~has).sum())))
    print('pool_s1 預測分布:', sub['pred_rank'].value_counts().to_dict())
    ref_path = Path(args.ref_sub) if args.ref_sub else None
    if ref_path is not None and ref_path.exists():
        ref = pd.read_csv(ref_path)
        print('和參考提交檔（%s）預測一致率 %.1f%%' % (ref_path.name, 100 * (sub['pred_rank'] == ref['pred_rank']).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--val-jsonl', required=True, help='開發組目標側 history 版 jsonl，profiles 要和 --profiles 一致')
    ap.add_argument('--val-opp', required=True)
    ap.add_argument('--val-probs', default='rank-trained-models/59844b9/val_probs_dev.npz')
    ap.add_argument('--pool-jsonl', required=True)
    ap.add_argument('--pool-opp', required=True)
    ap.add_argument('--exclude-players', nargs='*',
                    default=['humansl-analysis/val-players-dev-v180000.csv', 'humansl-analysis/holdout-v180000.csv'])
    ap.add_argument('--test-jsonl', required=True)
    ap.add_argument('--test-opp', required=True)
    ap.add_argument('--test-probs', default='rank-trained-models/59844b9/game_probs_test.npz')
    ap.add_argument('--ckpt', default='rank-trained-models/59844b9/best_model.pth')
    ap.add_argument('--test-dir', default='tests')
    ap.add_argument('--out-dir', default='rank-trained-models/59844b9-pool')
    ap.add_argument('--ref-sub', default='rank-trained-models/59844b9-hist/submission_rank.csv',
                    help='比對預測一致率的參考提交檔')
    ap.add_argument('--cnn-temp', type=float, default=3.0)
    ap.add_argument('--sizes', type=parse_sizes, default=SIZES)
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--inner-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--s1-only', action='store_true', help='只用第一段，不讀 CNN 的 val、test 機率')
    ap.add_argument('--loss', action='store_true', help='第一段加兩側 KataGo 失誤特徵，需 --s1-only')
    ap.add_argument('--save-probs', help='另存每題第一段 10 級機率的 .npz 路徑，需 --s1-only')
    ap.add_argument('--profiles', type=parse_profiles, default=PROFILES,
                    help='兩側 jsonl 的 3 個 profile，逗號分隔，第三個是強 profile')
    args = ap.parse_args()
    if args.loss and not args.s1_only:
        ap.error('--loss 需要 --s1-only')
    if args.save_probs and not args.s1_only:
        ap.error('--save-probs 需要 --s1-only')
    s1_groups = ('hsl', 'hslo', 'loss', 'losso') if args.loss else ('hsl', 'hslo')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 讀取並驗證開發組與 train 池資料
    print('載入開發組與 train 池特徵...')
    dev_data, dev_games = load_game_features(args.val_jsonl, args.val_opp, args.profiles)
    pool_data, pool_games = load_game_features(args.pool_jsonl, args.pool_opp, args.profiles)
    check_profiles(dev_data, pool_data)
    others = {Path(p).name: read_player_ids(p) for p in args.exclude_players}
    check_disjoint(set(pool_data['player_id']), others)
    check_disjoint(set(pool_data['player_id']), {'開發組': set(dev_data['player_id'])})
    cnn_val = None if args.s1_only else load_cnn_probs(args.val_probs, dev_data['key'], dev_data['rank_idx'])

    # 2. 準備出題特徵
    print('產生擬合用題目...')
    dev_q, dev_y, dev_g = questions_of(dev_data, np.ones(len(dev_data['key']), bool),
                                       args.train_repeats, np.random.default_rng(args.seed), args.sizes)
    m_dev = question_matrix(dev_games, cnn_val, dev_q, args.cnn_temp)
    f_dev_s1 = stack_groups(m_dev, s1_groups)

    pool_q, pool_y, pool_g = questions_of(pool_data, np.ones(len(pool_data['key']), bool),
                                         args.train_repeats, np.random.default_rng([args.seed, POOL_RNG]), args.sizes)
    m_pool = question_matrix(pool_games, None, pool_q, args.cnn_temp)
    f_pool_s1 = stack_groups(m_pool, s1_groups)

    # 第一段整體擬合資料（開發組 + train 池）
    f_all_s1 = np.vstack([f_dev_s1, f_pool_s1])
    y_all = np.concatenate([dev_y, pool_y])
    g_all = np.concatenate([dev_g, pool_g])
    fill_mean = None
    if args.loss:
        fill_mean = column_means(f_all_s1)
        f_all_s1 = fill_with(f_all_s1, fill_mean)

    # 3. 擬合第一段
    print('選取第一段 C 並產生 out-of-fold 預測...')
    c1 = select_c(f_all_s1, y_all, g_all, CS, args.inner_folds)
    print('第一段選到 C=%g（在 %d 位玩家、%d 題上）' % (c1, len(set(g_all)), len(y_all)))

    final_s1_model = new_model(c1).fit(f_all_s1, y_all)
    if args.s1_only:
        predict_s1_only(args, final_s1_model, dev_data, out_dir, fill_mean)
        return

    oof_dev = np.full((len(dev_y), len(LEVELS)), np.nan)
    for fit_i, pred_i in GroupKFold(n_splits=args.inner_folds).split(f_dev_s1, dev_y, dev_g):
        f_tr = np.vstack([f_dev_s1[fit_i], f_pool_s1])
        y_tr = np.concatenate([dev_y[fit_i], pool_y])
        oof_dev[pred_i] = stage1_log_probs(new_model(c1).fit(f_tr, y_tr), f_dev_s1[pred_i])
    assert not np.isnan(oof_dev).any(), '開發組第一段 out-of-fold 有 NaN'

    # 4. 擬合第二段（CNN + 第一段 OOF，僅在開發組）
    print('選取第二段 C 並擬合最終第二層...')
    f_dev_s2 = np.hstack([m_dev['cnn'], oof_dev])
    c2 = select_c(f_dev_s2, dev_y, dev_g, CS, args.inner_folds)
    print('第二段選到 C=%g（在 %d 位開發組玩家、%d 題上）' % (c2, len(set(dev_g)), len(dev_y)))
    final_s2_model = new_model(c2).fit(f_dev_s2, dev_y)

    # 5. 測試集預測
    print('讀取測試集資料與 CNN 機率...')
    qids, games = read_questions(args.test_dir)
    game_q = np.array([q for q, _, _ in games], np.int64)
    test_probs, sha = read_test_probs(args.test_probs, game_q, args.ckpt)
    test_data = load_games(args.test_jsonl)
    check_profiles(dev_data, test_data)

    test_keys = test_game_keys(qids, games)
    x_test_games = align_test_hsl(test_data, args.test_opp, test_keys)
    valid = ~np.isnan(x_test_games).any(axis=1) & ~np.isnan(test_probs).any(axis=1)
    print('測試集：%d 題，%d 局，可用 %d 局' % (len(qids), len(games), valid.sum()))

    # 題層級特徵
    n_questions = len(qids)
    test_q_indices = [np.flatnonzero((game_q == q) & valid) for q in range(n_questions)]
    has_games = np.array([len(idx) > 0 for idx in test_q_indices], dtype=bool)

    # 每題 CNN log 機率
    scaled_cnn = apply_temperature(test_probs, args.cnn_temp)
    cnn_test_q = np.array([aggregate_games(scaled_cnn[idx], 'log') if len(idx) else np.full(10, 0.1)
                           for idx in test_q_indices])
    cnn_test_logp = np.log(np.maximum(cnn_test_q, EPS))

    # 每題 Human SL (hsl + hslo) 平均特徵 (8 欄)
    hsl_test_q = np.array([x_test_games[idx].mean(axis=0) if len(idx) else np.zeros(8)
                           for idx in test_q_indices])

    # 執行第一段與第二段預測
    preds_s1 = np.zeros(n_questions, dtype=np.int64)
    preds_pool = np.zeros(n_questions, dtype=np.int64)
    if has_games.any():
        s1_test_logp = stage1_log_probs(final_s1_model, hsl_test_q[has_games])
        s1_test_probs = np.exp(s1_test_logp)
        preds_s1[has_games] = pick_level(s1_test_probs)

        f_test_s2 = np.hstack([cnn_test_logp[has_games], s1_test_logp])
        s2_test_probs = level_probs(final_s2_model, f_test_s2)
        preds_pool[has_games] = pick_level(s2_test_probs)

    # 6. 寫出提交檔案
    # 輸出 pool_s1
    sub_s1 = pd.DataFrame({'question_id': qids, 'pred_rank': [LEVELS[i] for i in preds_s1]})
    out_s1_path = out_dir / 'submission_rank_pool_s1.csv'
    sub_s1.to_csv(out_s1_path, index=False)
    print('寫出 %s（%d 列）' % (out_s1_path, len(sub_s1)))
    print('pool_s1 預測分布:', sub_s1['pred_rank'].value_counts().to_dict())

    # 輸出 pool
    sub_pool = pd.DataFrame({'question_id': qids, 'pred_rank': [LEVELS[i] for i in preds_pool]})
    out_pool_path = out_dir / 'submission_rank_pool.csv'
    sub_pool.to_csv(out_pool_path, index=False)
    print('寫出 %s（%d 列）' % (out_pool_path, len(sub_pool)))
    print('pool 預測分布:', sub_pool['pred_rank'].value_counts().to_dict())

    # 預設 submission_rank.csv 指向 pool（主方案）
    default_sub_path = out_dir / 'submission_rank.csv'
    sub_pool.to_csv(default_sub_path, index=False)
    print('寫出預設主方案 %s（%d 列）' % (default_sub_path, len(sub_pool)))

    # 和先前版本比較一致性
    ref_path = Path(args.ref_sub) if args.ref_sub else None
    if ref_path is not None and ref_path.exists():
        sub_ref = pd.read_csv(ref_path)
        match_pool = (sub_pool['pred_rank'] == sub_ref['pred_rank']).mean()
        match_s1 = (sub_s1['pred_rank'] == sub_ref['pred_rank']).mean()
        print('和參考提交檔（%s）預測一致率：pool %.1f%%, pool_s1 %.1f%%' % (ref_path.name, match_pool * 100, match_s1 * 100))

    match_between = (sub_pool['pred_rank'] == sub_s1['pred_rank']).mean()
    print('pool 與 pool_s1 之間一致率：%.1f%%' % (match_between * 100))


if __name__ == '__main__':
    main()
