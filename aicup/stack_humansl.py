"""CNN 與 Human SL 的 stacking：每題的 CNN log 機率加 Human SL 基本特徵組 base，用 logistic regression 輸出 10 類機率。

只用 CNN 沒訓練過的 val 玩家擬合。出題方式與依玩家的 5 折沿用 fit_humansl，
CNN 單獨、Human SL 單獨、stacking 在同一批題目上比較，差用以玩家重抽的配對 bootstrap。
正則化強度用巢狀交叉驗證選：只在外層折的訓練玩家內再依玩家切內層折。

--combos 給定時改為比較多種特徵組合，組合由特徵塊以 + 連接：
    cnn     CNN 合併後的 10 個 log 機率
    hsl     Human SL 基本特徵組 base
    se1、se4、se8  SE∞ 前 1、4、8 種對稱的 score 平均與標準差（--se 給定時才有，檔案的對稱數要足夠）
    sl      SL 11 級平均機率的 log（--se 給定時才有）
    hd      讓子或讓先局的方向：目標玩家執黑（被讓、對手較強）為 +1，執白為 -1，HD 0 為 0（--se 給定時才有）
    se1o、se4o、se8o、slo  同上，評的是對手的手（--se-opp 給定時才有）
    hslo    Human SL 基本特徵組 base，評的是對手的手（--hsl-opp 給定時才有）
    pm      逐手模型 10 個等級的平均 log 機率，評的是目標玩家的手（--pm 給定時才有）
    pmo     同上，評的是對手的手（--pm 的每局都有對手側時才有）
    set     set 模型對每題輸出的 10 級 log 機率（--set-model 與 --set-dir 給定時才有）
            是題層級特徵，不是各局特徵
--prior 決定是否去掉訓練題的等級先驗：keep 不處理，balanced 用 class_weight，divide 把機率除以訓練題的等級比例。
--cs 是巢狀交叉驗證的候選 C，不給時用 fit_humansl.CS。
--combos 時分數同時印出不加權與每級等權兩種，每級等權是先算各等級玩家的平均，再對等級平均。
--se、--se-opp、--hsl-opp、--pm、--set-model、--set-dir、--out、--prior、--cs 只用於 --combos。
--out 給定時，把每個組合的逐題分數存成 score_<組合> 的 npz，欄位同 stack_hgb run，可交給 stack_hgb compare 合併多個 seed。
--eval-players 給定時，只有清單內的玩家切折、出測試題，jsonl 中其他玩家每一折都只當訓練玩家。

用法（在 repo 根目錄）：
    python -m aicup.stack_humansl humansl-analysis/val648-p3.jsonl --cnn-probs rank-trained-models/<commit hash>/val_probs.npz
    python -m aicup.stack_humansl humansl-analysis/val648-p3.jsonl --cnn-probs ... --se se-models/val648-se.jsonl --combos cnn+hsl,cnn+hsl+se8
    python -m aicup.stack_humansl humansl-analysis/val648-p3.jsonl --cnn-probs ... --se se-models/val648-se.jsonl --hsl-opp humansl-analysis/val648-p3-opp.jsonl --combos cnn+hsl+hd,cnn+hsl+hd+hslo
    python -m aicup.stack_humansl humansl-analysis/val-v180000-p3.jsonl --cnn-probs ... --eval-players humansl-analysis/val-players.csv --combos cnn+hsl
    python -m aicup.stack_humansl humansl-analysis/val-v180000-dev-p3-hist.jsonl --hsl-opp ... --cnn-probs ... --combos hsl+hslo,hsl+hslo+set --set-model humansl-analysis/setmodel/set-model.pt --set-dir humansl-analysis/setmodel/trunk-eval --out runs/s42.npz
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from aicup.eval_posthoc import SIZES, parse_sizes
from aicup.fit_humansl import (CS, bootstrap_mean, feature_sets, level_probs, load_games, new_model,
                               player_folds, player_keys, player_means, predict_avg, question_points,
                               question_x, questions_of, read_val_probs)
from aicup.scoring import LEVELS, aggregate_games, apply_temperature, pick_level

EPS = 1e-12  # log 前的下限，與 scoring.aggregate_games 一致
HSL_C = 0.1  # Human SL 單獨比較用的固定組合 (avg base, C=0.1)
SE_SYMMETRIES = (1, 4, 8)
PRIORS = ('keep', 'balanced', 'divide')


def load_cnn_probs(path, keys, rank_idx):
    """val_probs.npz 中對應 keys 的每局機率，形狀 (len(keys), 10)。

    有 key 不在檔案內，或檔案有 rank_idx 且和 rank_idx 不一致時丟 ValueError。
    """
    index, probs, val_rank = read_val_probs(path)
    missing = [k for k in keys if k not in index]
    if missing:
        raise ValueError('%s 缺少 %d 局，例如 %s' % (path, len(missing), ', '.join(missing[:3])))
    rows = np.array([index[k] for k in keys])
    bad = np.flatnonzero(np.isnan(probs[rows]).any(axis=1))
    if len(bad):
        raise ValueError('%s 有 %d 局機率是 NaN，例如 %s' % (path, len(bad), keys[bad[0]]))
    if val_rank is not None:
        bad = np.flatnonzero(val_rank[rows] != rank_idx)
        if len(bad):
            raise ValueError('%s 的等級和 jsonl 不一致，例如 %s' % (path, keys[bad[0]]))
    return probs[rows].astype(np.float64)


def load_se_blocks(path, keys, rank_idx, side='target'):
    """strength_se 的 jsonl 依 keys 對齊，回傳 {特徵塊名稱: (len(keys), D)}。

    se<m> 是前 m 種對稱的 score_mean 平均與 score_std 平均，只產生 m 不超過檔案對稱數的塊；
    sl 是 sl_rank 的 log；hd 見 hd_direction。side 為 opponent 時塊名稱加上後綴 o，hd 不重複產生。
    有 key 不在檔案內、等級不一致、沒有可評的手、各局對稱數不同，或檔案的 side 和參數不同時丟 ValueError。
    """
    with open(path, encoding='utf-8') as f:
        rows = {r['key']: r for r in map(json.loads, (line for line in f if line.strip()))}
    missing = [k for k in keys if k not in rows]
    if missing:
        raise ValueError('%s 缺少 %d 局，例如 %s' % (path, len(missing), ', '.join(missing[:3])))
    picked = [rows[k] for k in keys]
    bad = [r['key'] for r, y in zip(picked, rank_idx) if r['rank_idx'] != y]
    if bad:
        raise ValueError('%s 的等級和 jsonl 不一致，例如 %s' % (path, bad[0]))
    bad = [r['key'] for r in picked if r.get('side', 'target') != side]
    if bad:
        raise ValueError('%s 的 side 不是 %s，例如 %s' % (path, side, bad[0]))
    bad = [r['key'] for r in picked if not r['n']]
    if bad:
        raise ValueError('%s 有 %d 局沒有 %s 方可評的手，例如 %s' % (path, len(bad), side, bad[0]))
    n_sym = {len(r['score_mean']) for r in picked}
    if len(n_sym) > 1:
        raise ValueError('%s 各局的對稱數不同：%s' % (path, sorted(n_sym)))
    n_sym = n_sym.pop()
    mean = np.array([r['score_mean'] for r in picked], np.float64)
    std = np.array([r['score_std'] for r in picked], np.float64)
    suffix = '' if side == 'target' else 'o'
    blocks = {'se%d%s' % (m, suffix): np.column_stack([mean[:, :m].mean(1), std[:, :m].mean(1)])
              for m in SE_SYMMETRIES if m <= n_sym}
    blocks['sl' + suffix] = np.log(np.maximum(np.array([r['sl_rank'] for r in picked], np.float64), EPS))
    if side == 'target':
        blocks['hd'] = np.array([[hd_direction(r['hd'], r['color'])] for r in picked])
    return blocks


def load_hsl_opp(path, data, allow_missing=False):
    """analyze_humansl --side opponent 的 jsonl 依 data['key'] 對齊，回傳基本特徵組 base 的欄位 (len(keys), D)。

    profiles 與是否有損失欄位要和 data 相同。等級不一致或 side 不是 opponent 時丟 ValueError。
    有 key 不在檔案內時，allow_missing 為 False 丟 ValueError，為 True 則該列填 NaN。
    """
    opp = load_games(path)
    if (opp['profiles'], opp['has_loss']) != (data['profiles'], data['has_loss']):
        raise ValueError('%s 的 profiles 或損失欄位和目標玩家的 jsonl 不同' % path)
    with open(path, encoding='utf-8') as f:
        sides = [json.loads(line).get('side', 'target') for line in f if line.strip()]
    bad = [k for k, s in zip(opp['key'], sides) if s != 'opponent']
    if bad:
        raise ValueError('%s 的 side 不是 opponent，例如 %s' % (path, bad[0]))
    row_of = {k: i for i, k in enumerate(opp['key'])}
    has = np.array([k in row_of for k in data['key']], bool)
    if not allow_missing and not has.all():
        missing = data['key'][~has]
        raise ValueError('%s 缺少 %d 局，例如 %s' % (path, len(missing), ', '.join(missing[:3])))
    rows = np.array([row_of[k] for k in data['key'][has]], np.int64)
    wrong = opp['rank_idx'][rows] != data['rank_idx'][has]
    if wrong.any():
        raise ValueError('%s 的等級和目標玩家的 jsonl 不一致，例如 %s' % (path, data['key'][has][wrong][0]))
    cols = hsl_columns(opp)
    x = np.full((len(has), len(cols)), np.nan)
    x[has] = opp['x'][rows][:, cols]
    return x


def load_pm_blocks(path, keys, rank_idx):
    """infer_permove 的 jsonl 依 keys 對齊，回傳 {'pm': (len(keys), 10)}，每局都有 pmo 時另有 'pmo'。

    有 key 不在檔案內、等級不一致、有局沒有目標側的手，或只有部分局有對手側的手時丟 ValueError。
    """
    with open(path, encoding='utf-8') as f:
        rows = {r['key']: r for r in map(json.loads, (line for line in f if line.strip()))}
    missing = [k for k in keys if k not in rows]
    if missing:
        raise ValueError('%s 缺少 %d 局，例如 %s' % (path, len(missing), ', '.join(missing[:3])))
    picked = [rows[k] for k in keys]
    bad = [r['key'] for r, y in zip(picked, rank_idx) if r['rank_idx'] != y]
    if bad:
        raise ValueError('%s 的等級和 jsonl 不一致，例如 %s' % (path, bad[0]))
    bad = [r['key'] for r in picked if r['pm'] is None]
    if bad:
        raise ValueError('%s 有 %d 局沒有目標側的手，例如 %s' % (path, len(bad), bad[0]))
    blocks = {'pm': np.array([r['pm'] for r in picked], np.float64)}
    bad = [r['key'] for r in picked if r.get('pmo') is None]
    if bad and len(bad) < len(picked):
        raise ValueError('%s 有 %d 局沒有對手側的手，例如 %s' % (path, len(bad), bad[0]))
    if not bad:
        blocks['pmo'] = np.array([r['pmo'] for r in picked], np.float64)
    return blocks


def load_eval_players(path):
    """含 player_id、rank 欄位的 CSV，回傳 fit_humansl.player_keys 格式的 set。"""
    with open(path, encoding='utf-8', newline='') as f:
        return {'%d|%s' % (LEVELS.index(r['rank']), r['player_id']) for r in csv.DictReader(f)}


def make_eval_splits(data, sizes, repeats, eval_players=None, folds=5, seed=42):
    """同 fit_humansl.make_splits，但只有 eval_players 的玩家切折、出測試題，其他玩家每一折都只當訓練。

    eval_players 是 player_keys 格式的 set，None 時所有玩家都參與，結果和 make_splits 相同。
    eval_players 的局在 data 中的相對順序和另一份只含這些玩家的 data 相同時，兩者的測試題選到相同的局。
    有清單內的玩家不在 data 時丟 ValueError。回傳 [(train 遮罩, questions, labels, players)]。
    """
    keys = player_keys(data)
    if eval_players is None:
        member = np.ones(len(keys), bool)
    else:
        missing = sorted(eval_players - set(keys))
        if missing:
            raise ValueError('有 %d 位評估玩家不在資料內，例如 %s' % (len(missing), missing[0]))
        member = np.isin(keys, sorted(eval_players))
    fold = np.full(len(keys), -1)
    fold[member] = player_folds({'player_id': data['player_id'][member], 'rank_idx': data['rank_idx'][member]},
                                folds, seed)
    splits = []
    for f in range(folds):
        q, y, p = questions_of(data, fold == f, repeats, np.random.default_rng([seed, f]), sizes)
        splits.append((fold != f, q, y, p))
    return splits


def parse_cs(text):
    """'0.001,0.01' 轉成正則化強度的 tuple，順序照輸入（同分時 select_c 取較前者）。"""
    cs = tuple(float(v) for v in text.split(',') if v.strip())
    if not cs or any(c <= 0 for c in cs):
        raise argparse.ArgumentTypeError('--cs 要是逗號分隔的正數')
    return cs


def hd_direction(hd, color):
    """讓子或讓先局的方向：目標玩家執黑為 +1（被讓的一方，對手較強），執白為 -1，HD 0 或沒有 HD 為 0。"""
    if not hd:
        return 0.0
    return 1.0 if color == 'B' else -1.0


def cnn_question_probs(probs, questions, temp):
    """每題的 CNN 機率，先 apply_temperature 再 aggregate_games 以 log 合併，同 predict_rank。"""
    scaled = apply_temperature(probs, temp)
    return np.array([aggregate_games(scaled[q], 'log') for q in questions])


def question_features(x, cnn_probs, questions, temp, q_fn=None):
    """每題的特徵：CNN 合併後的 10 個 log 機率，接各局特徵 x 的平均，最後接 q_fn 的輸出。

    x、cnn_probs、q_fn 為 None 的部分不放，至少要有一個。
    q_fn 接收 questions，回傳 (題數, k)；第 i 列只能由 questions[i] 的局決定，且不受局的順序影響。
    """
    parts = []
    if cnn_probs is not None:
        parts.append(np.log(np.maximum(cnn_question_probs(cnn_probs, questions, temp), EPS)))
    if x is not None:
        parts.append(question_x(x, questions))
    if q_fn is not None:
        out = np.asarray(q_fn(questions), dtype=float)
        if out.ndim != 2 or len(out) != len(questions):
            raise ValueError('q_fn 要回傳 (題數, k)，得到 %s' % (out.shape,))
        parts.append(out)
    return np.hstack(parts)


def hsl_columns(data):
    """Human SL 基本特徵組 base 在 data['x'] 的欄位。"""
    return feature_sets(data['n_profiles'], data['has_loss'])['base']


def parse_combo(combo, available):
    """'cnn+hsl+se8' 拆成特徵塊名稱的 list，有未知或重複的塊時丟 ValueError。"""
    names = combo.split('+')
    unknown = [n for n in names if n not in available]
    if unknown or len(set(names)) != len(names):
        raise ValueError('組合 %r 有未知或重複的特徵塊，可用的有 %s' % (combo, ', '.join(sorted(available))))
    return names


def combo_inputs(names, blocks, cnn_probs):
    """組合的 (各局特徵 x 或 None, CNN 機率或 None)。blocks 是 {名稱: 各局特徵}，不含 cnn 與 set。"""
    cols = [blocks[n] for n in names if n not in ('cnn', 'set')]
    return (np.hstack(cols) if cols else None), (cnn_probs if 'cnn' in names else None)


def train_rng(seed, fold):
    """第 fold 折抽訓練題的 rng，三種方法各自取一個，抽到的訓練題相同。"""
    return np.random.default_rng([seed, fold])


def select_c(x, y, groups, cs=CS, inner_folds=5, class_weight=None):
    """依 groups 切內層折，回傳平均 pick 得分最高的 C，同分取 cs 中較前者。"""
    splitter = GroupKFold(n_splits=inner_folds)
    scores = []
    for c in cs:
        points = []
        for tr, te in splitter.split(x, y, groups):
            model = new_model(c, class_weight).fit(x[tr], y[tr])
            points.append(question_points(y[te], pick_level(level_probs(model, x[te]))))
        scores.append(np.concatenate(points).mean())
    return cs[int(np.argmax(scores))]


def fit_model(x, cnn_probs, questions, labels, players, temp, cs=CS, inner_folds=5, class_weight=None, q_fn=None):
    """在 questions 上依 players 選 C 後擬合，回傳 (model, 選到的 C)；特徵由 question_features 組成，q_fn 見該函數。"""
    f = question_features(x, cnn_probs, questions, temp, q_fn)
    c_best = select_c(f, labels, players, cs, inner_folds, class_weight)
    return new_model(c_best, class_weight).fit(f, labels), c_best


def divide_prior(probs, labels):
    """機率除以 labels 的等級比例後重新正規化；labels 沒有的等級維持 0。"""
    prior = np.bincount(labels, minlength=len(LEVELS)) / len(labels)
    out = np.where(prior > 0, probs / np.maximum(prior, EPS), 0.0)
    return out / out.sum(axis=1, keepdims=True)


def prior_class_weight(prior):
    """prior 對應的 class_weight：只有 balanced 在擬合時處理，keep 與 divide 都是 None。"""
    return 'balanced' if prior == 'balanced' else None


def predict_inputs(x, cnn_probs, data, train, questions, rng, sizes, train_repeats, temp, cs=CS, inner_folds=5,
                   prior='keep', q_fn=None):
    """用 train 遮罩的玩家切訓練題、選 C、擬合，回傳 (questions 的 10 類機率, 選到的 C)。

    x 是各局特徵或 None，cnn_probs 是每局 CNN 機率或 None，q_fn 是題層級特徵函數或 None，三者不能都是 None。
    sizes 為每題可用的局數，temp 為 CNN 每局機率的溫度，cs 為候選 C，inner_folds 為內層折數。
    prior 為 keep 時不處理等級先驗；balanced 擬合時用 class_weight='balanced'；
    divide 照常擬合，預測機率再除以訓練題的等級比例後正規化。
    """
    tr_q, tr_y, tr_players = questions_of(data, train, train_repeats, rng, sizes)
    model, c_best = fit_model(x, cnn_probs, tr_q, tr_y, tr_players, temp, cs, inner_folds, prior_class_weight(prior),
                              q_fn)
    probs = level_probs(model, question_features(x, cnn_probs, questions, temp, q_fn))
    if prior == 'divide':
        probs = divide_prior(probs, tr_y)
    return probs, c_best


def predict_stack(data, cnn_probs, train, questions, rng, sizes, train_repeats, temp, cs=CS, inner_folds=5,
                  use_hsl=True):
    """CNN 加 Human SL 基本特徵組 base 的 stacking，參數見 predict_inputs。

    use_hsl 為 False 時只用 CNN 特徵，用來分開「在 val 上重新擬合」與「加入 Human SL」的貢獻。
    """
    x = data['x'][:, hsl_columns(data)] if use_hsl else None
    return predict_inputs(x, cnn_probs, data, train, questions, rng, sizes, train_repeats, temp, cs, inner_folds)


def level_weighted(means, player_labels):
    """每級等權的分數：先算各等級玩家的平均，再對等級平均。"""
    levels = np.unique(player_labels)
    return float(np.mean([means[player_labels == y].mean() for y in levels]))


def print_interval(label, values, seed):
    """印出平均與 95% bootstrap 區間，回傳 (平均, 下限, 上限)。"""
    interval = bootstrap_mean(values, seed=seed)
    print('  %s：%.4f，95%% 區間 %.4f～%.4f' % ((label,) + interval))
    return interval


def load_set_q_fn(model_path, set_dir, keys):
    """set 塊的 q_fn：載入 set 模型與 setmodel_trunk 的輸出，依 keys 對齊每局向量。

    有 key 不在 set_dir 的輸出內時丟 ValueError。沒有向量的局不參與，題內全部局都沒有向量時用訓練玩家的等級先驗。
    """
    from aicup.setmodel_train import align, embed_games, load_games as load_trunk, load_model, make_q_fn
    vec, mask = align(load_trunk(set_dir), keys)
    model, prior_logp = load_model(model_path)
    return make_q_fn(model, embed_games(model, vec, mask), mask.any(axis=1), prior_logp)


def run_combos(args, data, cnn, splits, labels, players):
    """--combos：同一批題目、同一組折比較各組合，以第一個組合為基準算配對差。"""
    blocks = {'hsl': data['x'][:, hsl_columns(data)]}
    if args.se:
        blocks.update(load_se_blocks(args.se, data['key'], data['rank_idx']))
    if args.se_opp:
        blocks.update(load_se_blocks(args.se_opp, data['key'], data['rank_idx'], side='opponent'))
    if args.hsl_opp:
        blocks['hslo'] = load_hsl_opp(args.hsl_opp, data)
    if args.pm:
        blocks.update(load_pm_blocks(args.pm, data['key'], data['rank_idx']))
    combos = [c.strip() for c in args.combos.split(',') if c.strip()]
    parsed = {c: parse_combo(c, set(blocks) | {'cnn', 'set'}) for c in combos}
    need_set = any('set' in names for names in parsed.values())
    if need_set and not args.set_model:
        raise ValueError('組合含 set 塊，要給 --set-model 與 --set-dir')
    set_q_fn = load_set_q_fn(args.set_model, args.set_dir, data['key']) if need_set else None
    keys = sorted(set(players))
    player_label = dict(zip(players, labels))
    player_labels = np.array([player_label[k] for k in keys])
    means, chosen, scores = {}, {}, {}
    for combo in combos:
        x, cnn_c = combo_inputs(parsed[combo], blocks, cnn)
        probs, fold_c = [], []
        for f, (train, q, _, _) in enumerate(splits):
            p, c_best = predict_inputs(x, cnn_c, data, train, q, train_rng(args.seed, f), args.sizes,
                                       args.train_repeats, args.cnn_temp, cs=args.cs, inner_folds=args.inner_folds,
                                       prior=args.prior, q_fn=set_q_fn if 'set' in parsed[combo] else None)
            probs.append(p)
            fold_c.append(c_best)
        scores[combo] = question_points(labels, pick_level(np.concatenate(probs)))
        m = player_means(scores[combo], players)
        means[combo] = np.array([m[k] for k in keys])
        chosen[combo] = fold_c
    if args.out:
        questions = [q for s in splits for q in s[1]]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, player_id=players, keys=np.array([','.join(data['key'][q]) for q in questions]),
                 rank_idx=labels, n_games=np.array([len(q) for q in questions]),
                 **{'score_' + c: v for c, v in scores.items()})
        print('逐題分數 ->', out)
    base = combos[0]
    print('prior %s，基準 %s' % (args.prior, base))
    print('| 組合 | 不加權 | 95% 區間 | 每級等權 | 減基準 | 95% 區間 | 各折 C |')
    print('|---|---|---|---|---|---|---|')
    for combo in combos:
        mean, lo, hi = bootstrap_mean(means[combo], seed=args.seed)
        d, dlo, dhi = bootstrap_mean(means[combo] - means[base], seed=args.seed)
        print('| %s | %.4f | %.4f～%.4f | %.4f | %+.4f | %+.4f～%+.4f | %s |'
              % (combo, mean, lo, hi, level_weighted(means[combo], player_labels), d, dlo, dhi,
                 ', '.join('%g' % v for v in chosen[combo])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('jsonl', help='analyze_humansl 的輸出，玩家要是 CNN 沒訓練過的 val 玩家')
    ap.add_argument('--cnn-probs', required=True, help='CNN 的 val_probs.npz')
    ap.add_argument('--cnn-temp', type=float, default=3.0, help='CNN 每局機率的溫度，預設同 predict_rank')
    ap.add_argument('--sizes', type=parse_sizes, default=SIZES, help='每題局數，逗號分隔，預設 5,10,15,20')
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--test-repeats', type=int, default=10)
    ap.add_argument('--inner-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42, help='切折、出題與 bootstrap 的 seed')
    ap.add_argument('--se', help='strength_se 的 jsonl，提供 se1、se4、se8、sl、hd 特徵塊')
    ap.add_argument('--se-opp', help='strength_se --side opponent 的 jsonl，提供 se1o、se4o、se8o、slo 特徵塊')
    ap.add_argument('--hsl-opp', help='analyze_humansl --side opponent 的 jsonl，提供 hslo 特徵塊')
    ap.add_argument('--pm', help='infer_permove 的 jsonl，提供 pm 特徵塊，每局都有對手側時另有 pmo')
    ap.add_argument('--set-model',
                    help='setmodel_train 輸出的 checkpoint，提供 set 塊，只用於 --combos，要搭配 --set-dir')
    ap.add_argument('--set-dir', help='setmodel_trunk extract --set eval 的輸出目錄，要搭配 --set-model')
    ap.add_argument('--out', help='只用於 --combos：逐題分數的 npz 路徑')
    ap.add_argument('--combos', help='逗號分隔的組合，例如 cnn+hsl,cnn+hsl+se8；第一個是配對差的基準')
    ap.add_argument('--prior', default='keep', choices=PRIORS, help='只用於 --combos')
    ap.add_argument('--cs', type=parse_cs, default=CS,
                    help='逗號分隔的候選 C，只用於 --combos，預設 %s' % ','.join('%g' % c for c in CS))
    ap.add_argument('--eval-players', help='含 player_id、rank 欄位的 CSV，只有這些玩家出測試題，其他玩家只當訓練')
    args = ap.parse_args()
    if not args.combos and (args.se or args.se_opp or args.hsl_opp or args.pm or args.set_model or args.set_dir
                            or args.out or args.prior != 'keep' or args.cs != CS):
        ap.error('--se、--se-opp、--hsl-opp、--pm、--set-model、--set-dir、--out、--prior、--cs'
                 ' 只用於 --combos')
    if bool(args.set_model) != bool(args.set_dir):
        ap.error('--set-model 與 --set-dir 要一起給')
    if args.combos and any('set' in c.split('+') for c in args.combos.split(',')) and not args.set_model:
        ap.error('組合含 set 塊，要給 --set-model 與 --set-dir')

    data = load_games(args.jsonl)
    cnn = load_cnn_probs(args.cnn_probs, data['key'], data['rank_idx'])
    eval_players = load_eval_players(args.eval_players) if args.eval_players else None
    splits = make_eval_splits(data, args.sizes, args.test_repeats, eval_players, seed=args.seed)
    questions = [q for s in splits for q in s[1]]
    labels = np.concatenate([s[2] for s in splits])
    players = np.concatenate([s[3] for s in splits])
    n_train_only = 0 if eval_players is None else len(set(player_keys(data)) - eval_players)
    print('%d 局，%d 位玩家出測試題，%d 位只當訓練，%d 題，seed %d'
          % (len(data['x']), len(set(players)), n_train_only, len(questions), args.seed))
    if args.combos:
        run_combos(args, data, cnn, splits, labels, players)
        return

    hsl_cols = hsl_columns(data)
    stack, refit, hsl, chosen = [], [], [], []
    for f, (train, q, _, _) in enumerate(splits):
        # 每種方法各自從同一個 seed 起算，訓練題相同，不受執行順序影響
        p_stack, c_stack = predict_stack(data, cnn, train, q, train_rng(args.seed, f), args.sizes,
                                         args.train_repeats, args.cnn_temp, inner_folds=args.inner_folds)
        p_refit, c_refit = predict_stack(data, cnn, train, q, train_rng(args.seed, f), args.sizes,
                                         args.train_repeats, args.cnn_temp, inner_folds=args.inner_folds,
                                         use_hsl=False)
        stack.append(p_stack)
        refit.append(p_refit)
        chosen.append((c_stack, c_refit))
        hsl.append(predict_avg(data, hsl_cols, HSL_C, train, q, train_rng(args.seed, f), args.sizes,
                               args.train_repeats))
    probs = {'stacking': np.concatenate(stack), 'CNN 重新擬合': np.concatenate(refit),
             'Human SL avg base C=%g' % HSL_C: np.concatenate(hsl),
             'CNN': cnn_question_probs(cnn, questions, args.cnn_temp)}
    print('各折選到的 C（stacking / CNN 重新擬合）：%s' % ', '.join('%g/%g' % c for c in chosen))
    means = {}
    for name, p in probs.items():
        m = player_means(question_points(labels, pick_level(p)), players)
        means[name] = np.array([m[k] for k in sorted(m)])
        print_interval(name, means[name], args.seed)
    diffs = {other: print_interval('stacking - %s' % other, means['stacking'] - means[other], args.seed)
             for other in list(probs)[1:]}
    lo = diffs['CNN'][1]
    print('暫定門檻（stacking - CNN 的區間下限 > 0）：%s' % ('通過' if lo > 0 else '未通過'))


if __name__ == '__main__':
    main()
