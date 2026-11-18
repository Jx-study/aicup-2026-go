"""stack_hgb.load_game_features 的結果存成 .npz，之後直接讀，不必重新解析兩側與額外對手側的 jsonl。

快取記錄來源 jsonl 的路徑、大小與修改時間，以及 dist、stats、s、r 用的 profiles，讀取時來源有變就拒絕。
--opp-extra 給同一批局的另一份對手側 jsonl，主對手側沒有的 profile 存成特徵群 --extra-group（以 hslx 開頭），
欄位是每個新 profile 的每手平均 log 機率；meta 的 extra 記錄特徵群名稱與 profile，concat 時要相同。

用法（在 repo 根目錄）：
    python -m aicup.feature_cache --target humansl-analysis/hist-val648/val648-p3-hist.jsonl --opp humansl-analysis/hist-val648_opp/val648-p3-opp-hist.jsonl --out humansl-analysis/cache/val648.npz [--profiles rank_11k,rank_1d,rank_6d]
    python -m aicup.feature_cache --target humansl-analysis/val648-p3-hist.jsonl --opp humansl-analysis/hist-val648_opp/val648-p3-opp-hist-v1.jsonl --profiles rank_11k,rank_1d,rank_6d --opp-extra humansl-analysis/hist-val648_opp/val648-p3-opp-hist.jsonl --extra-group hslx --out humansl-analysis/cache/val648-6dx.npz
"""
import argparse
import json
import os

import numpy as np

from aicup.fit_humansl import game_features
from aicup.stack_hgb import EXTRA_PREFIX, PROFILES, load_game_features, parse_profiles

DATA_ARRAYS = ('x', 'rank_idx', 'player_id', 'key')
OPTIONAL_ARRAYS = ('black', 'hd')  # 目標側 jsonl 有 color、hd 時才有


def source_info(*paths):
    return [{'path': os.path.abspath(p), 'size': os.path.getsize(p), 'mtime': os.path.getmtime(p)} for p in paths]


def present(data):
    """data 中有的選用陣列名稱；兩個要一起有。"""
    return OPTIONAL_ARRAYS if all(k in data for k in OPTIONAL_ARRAYS) else ()


def read_rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def check_extra_args(opp_extra, extra_group):
    """opp_extra 與 extra_group 要一起給，extra_group 要以 EXTRA_PREFIX 開頭，否則丟 ValueError。"""
    if (opp_extra is None) != (extra_group is None):
        raise ValueError('opp_extra 與 extra_group 要一起給')
    if extra_group is not None and not extra_group.startswith(EXTRA_PREFIX):
        raise ValueError('extra_group 要以 %s 開頭，收到 %s' % (EXTRA_PREFIX, extra_group))


def load_opp_extra(path, data, main_opp):
    """額外的對手側 jsonl 依 data['key'] 對齊，回傳 (欄位 (N, P), profile 名稱 list)。

    只取 data['profiles'] 沒有的 profile，欄位是該 profile 每手平均 log 機率，定義同 hslo。
    key 集合和 data 不同、等級不一致、side 不是 opponent、各局 profiles 不同，
    或逐局的 pre_root_history、turns 和主對手側 jsonl main_opp 不同，或沒有新 profile 時丟 ValueError。
    """
    rows = read_rows(path)
    bad = [r['key'] for r in rows if r.get('side', 'target') != 'opponent']
    if bad:
        raise ValueError('%s 的 side 不是 opponent，例如 %s' % (path, bad[0]))
    names = {tuple(r.get('profiles', ())) for r in rows}
    if len(names) != 1 or not next(iter(names)):
        raise ValueError('%s 的 profiles 欄位缺少或各局不同' % path)
    profiles = list(names.pop())
    keys = [r['key'] for r in rows]
    if len(keys) != len(data['key']) or set(keys) != set(data['key']):
        raise ValueError('%s 的局和主資料不同' % path)
    new = [p for p in profiles if p not in data['profiles']]
    if not new:
        raise ValueError('%s 沒有主資料以外的 profile' % path)
    main = {r['key']: r for r in read_rows(main_opp)}
    bad = [r['key'] for r in rows if bool(r.get('pre_root_history', False)) != bool(main[r['key']].get('pre_root_history', False))
           or r['turns'] != main[r['key']]['turns']]
    if bad:
        raise ValueError('%s 的 pre_root_history 或 turns 和主對手側不同，例如 %s' % (path, bad[0]))
    row_of = dict(zip(keys, rows))
    ordered = [row_of[k] for k in data['key']]
    wrong = np.array([r['rank_idx'] for r in ordered]) != data['rank_idx']
    if wrong.any():
        raise ValueError('%s 的等級和主資料不一致，例如 %s' % (path, data['key'][wrong][0]))
    cols = [profiles.index(p) for p in new]  # game_features 的前幾欄依 profile 順序
    return np.array([game_features(r)[cols] for r in ordered]), new


def build(target, opp, out, profiles=PROFILES, opp_extra=None, extra_group=None):
    """讀兩側 jsonl 並寫出快取；dist、stats、s、r 用 profiles 計算。

    opp_extra 是額外的對手側 jsonl，新 profile 的欄位存成特徵群 extra_group，名稱必須以 EXTRA_PREFIX 開頭。
    """
    check_extra_args(opp_extra, extra_group)
    data, games = load_game_features(target, opp, profiles)
    paths = [target, opp]
    meta = {}
    if opp_extra is not None:
        games[extra_group], new = load_opp_extra(opp_extra, data, opp)
        paths.append(opp_extra)
        meta['extra'] = {'group': extra_group, 'profiles': new}
    meta.update({'n_profiles': int(data['n_profiles']), 'profiles': list(data['profiles']),
                 'feature_profiles': list(profiles),
                 'has_loss': bool(data['has_loss']), 'sources': source_info(*paths)})
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez(out, meta=json.dumps(meta), **{'d_' + k: data[k] for k in DATA_ARRAYS + present(data)},
             **{'g_' + k: v for k, v in games.items()})


def load(path, check_sources=True):
    """回傳和 load_game_features 相同格式的 (data, games)。

    check_sources 時來源 jsonl 的大小或修改時間和建立快取時不同就丟 ValueError。
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    if check_sources:
        for s in meta['sources']:
            if not os.path.exists(s['path']) or source_info(s['path'])[0] != s:
                raise ValueError('%s 的來源 %s 已變更，請重建快取' % (path, s['path']))
    data = {k: z['d_' + k] for k in DATA_ARRAYS}
    data.update({k: z['d_' + k] for k in OPTIONAL_ARRAYS if 'd_' + k in z.files})
    data.update(n_profiles=meta['n_profiles'], profiles=tuple(meta['profiles']), has_loss=meta['has_loss'],
                feature_profiles=tuple(meta.get('feature_profiles', meta['profiles'])), extra=meta.get('extra'))
    games = {k[2:]: z[k] for k in z.files if k.startswith('g_')}
    return data, games


def concat(parts):
    """多份 (data, games) 依序接起來；profiles、損失欄位、額外 profile 或特徵群不同時丟 ValueError。"""
    data0, games0 = parts[0]
    for data, games in parts[1:]:
        if (data['profiles'], data['has_loss'], data['feature_profiles']) != (
                data0['profiles'], data0['has_loss'], data0['feature_profiles']):
            raise ValueError('快取的 profiles 或損失欄位不同')
        if data.get('extra') != data0.get('extra'):
            raise ValueError('快取的額外 profile 不同：%s、%s' % (data0.get('extra'), data.get('extra')))
        if set(games) != set(games0):
            raise ValueError('快取的特徵群不同')
    if len({present(d) for d, _ in parts}) > 1:
        raise ValueError('快取有些有 color、hd，有些沒有')
    data = {k: np.concatenate([d[k] for d, _ in parts]) for k in DATA_ARRAYS + present(data0)}
    data.update(n_profiles=data0['n_profiles'], profiles=data0['profiles'], has_loss=data0['has_loss'],
                feature_profiles=data0['feature_profiles'], extra=data0.get('extra'))
    if len(set(data['key'])) != len(data['key']):
        raise ValueError('快取之間有重複的局')
    return data, {k: np.concatenate([g[k] for _, g in parts]) for k in games0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--target', required=True, help='目標側 history 版 jsonl')
    ap.add_argument('--opp', required=True, help='對手側 history 版 jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--profiles', type=parse_profiles, default=PROFILES,
                    help='dist、stats、s、r 用的 3 個 profile，逗號分隔，第三個是強 profile')
    ap.add_argument('--opp-extra', help='額外的對手側 jsonl，只取主對手側沒有的 profile')
    ap.add_argument('--extra-group', help='額外 profile 的特徵群名稱，以 %s 開頭' % EXTRA_PREFIX)
    args = ap.parse_args()
    try:
        check_extra_args(args.opp_extra, args.extra_group)
    except ValueError as e:
        ap.error(str(e))
    build(args.target, args.opp, args.out, args.profiles, args.opp_extra, args.extra_group)
    print('快取 ->', args.out)


if __name__ == '__main__':
    main()
