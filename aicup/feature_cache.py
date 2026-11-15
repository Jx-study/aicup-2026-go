"""stack_hgb.load_game_features 的結果存成 .npz，之後直接讀，不必重新解析兩側 jsonl。

快取記錄來源 jsonl 的路徑、大小與修改時間，以及 dist、stats、s、r 用的 profiles，讀取時來源有變就拒絕。

用法（在 repo 根目錄）：
    python -m aicup.feature_cache --target humansl-analysis/hist-val648/val648-p3-hist.jsonl --opp humansl-analysis/hist-val648_opp/val648-p3-opp-hist.jsonl --out humansl-analysis/cache/val648.npz [--profiles rank_11k,rank_1d,rank_6d]
"""
import argparse
import json
import os

import numpy as np

from aicup.stack_hgb import PROFILES, load_game_features, parse_profiles

DATA_ARRAYS = ('x', 'rank_idx', 'player_id', 'key')


def source_info(*paths):
    return [{'path': os.path.abspath(p), 'size': os.path.getsize(p), 'mtime': os.path.getmtime(p)} for p in paths]


def build(target, opp, out, profiles=PROFILES):
    """讀兩側 jsonl 並寫出快取；dist、stats、s、r 用 profiles 計算。"""
    data, games = load_game_features(target, opp, profiles)
    meta = {'n_profiles': int(data['n_profiles']), 'profiles': list(data['profiles']),
            'feature_profiles': list(profiles),
            'has_loss': bool(data['has_loss']), 'sources': source_info(target, opp)}
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez(out, meta=json.dumps(meta), **{'d_' + k: data[k] for k in DATA_ARRAYS},
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
    data.update(n_profiles=meta['n_profiles'], profiles=tuple(meta['profiles']), has_loss=meta['has_loss'],
                feature_profiles=tuple(meta.get('feature_profiles', meta['profiles'])))
    games = {k[2:]: z[k] for k in z.files if k.startswith('g_')}
    return data, games


def concat(parts):
    """多份 (data, games) 依序接起來；profiles、損失欄位或特徵群不同時丟 ValueError。"""
    data0, games0 = parts[0]
    for data, games in parts[1:]:
        if (data['profiles'], data['has_loss'], data['feature_profiles']) != (
                data0['profiles'], data0['has_loss'], data0['feature_profiles']):
            raise ValueError('快取的 profiles 或損失欄位不同')
        if set(games) != set(games0):
            raise ValueError('快取的特徵群不同')
    data = {k: np.concatenate([d[k] for d, _ in parts]) for k in DATA_ARRAYS}
    data.update(n_profiles=data0['n_profiles'], profiles=data0['profiles'], has_loss=data0['has_loss'],
                feature_profiles=data0['feature_profiles'])
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
    args = ap.parse_args()
    build(args.target, args.opp, args.out, args.profiles)
    print('快取 ->', args.out)


if __name__ == '__main__':
    main()
