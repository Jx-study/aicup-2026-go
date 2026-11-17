"""產生 Human SL 分析用的打包檔與 notebook，輸出到 humansl-analysis/hist-<run>/

在 Google Colab T4 GPU 上執行：zip 內含程式與棋譜，可直接上傳或放 Google Drive。
從 repo 根目錄執行：
    python utils/hist_bundle.py --run test --move-stats
    python utils/hist_bundle.py --run val648 --move-stats
    python utils/hist_bundle.py --run train_p1 --move-stats
    python utils/hist_bundle.py --run train_p2 --move-stats
    python utils/hist_bundle.py --run train_p3 --move-stats
    python utils/hist_bundle.py --run holdout --move-stats
    python utils/hist_bundle.py --run train_p4 --move-stats
--move-stats 時主分析多輸出 move_rank、pmax、entropy。
train_p4 第一次執行時寫出玩家清單 train-p4.csv：還沒抽過、HD < 2 的局中多數色 ≥ 10 局的玩家，
每人抽多數色、HD < 2 的 10 局，主分析不送損失 query。
train_p4b 是補局：train_p4 玩家中同色未抽局 ≥ 10 局的玩家，同色再抽 10 局，玩家清單 train-p4b.csv。
"""
import argparse
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

HA = 'humansl-analysis'
TRAIN_GLOB = 'training-v1.1/training/train_*.csv'
# Human SL 抽取用的 3 個 profile
PROFILES3 = ['rank_11k', 'rank_1d', 'rank_2d']
N_CTRL = 40
SEED, PER_PLAYER = 42, 20

NEW498 = 'val-players-new-v180000.csv'
RUNS = {
    'val648': dict(base='val648', old='val648-p3.jsonl', players='val-players.csv', n=12960),
    'test': dict(base='test', test=True, n=10000),
    'train_p1': dict(base='train-p1', players='train-p1-v180000.csv', n=20000),
    'train_p2': dict(base='train-p2', players='train-p2-v180000.csv', n=20000),
    'train_p3': dict(base='train-p3', players='train-p3-v180000.csv', n=20000),
    'holdout': dict(base='holdout', players='holdout-v180000.csv', sample=NEW498, n=3200),
    'train_p4': dict(base='train-p4', players='train-p4.csv', same_color=10, no_eval=True, n=19950),
    'train_p4b': dict(base='train-p4b', players='train-p4b.csv', topup='train_p4', same_color=10, no_eval=True, n=11980),
}
for _name in list(RUNS):
    RUNS[_name + '_opp'] = dict(RUNS[_name], side='opponent')
for _cfg in RUNS.values():
    _cfg.setdefault('side', 'target')

GAME_COLS = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']
BASE_FILES = ['aicup/__init__.py', 'aicup/analyze_humansl.py', 'aicup/scoring.py']
TEST_FILES = ['aicup/predict_rank.py', 'aicup/color.py', 'aicup/extract_features.py', 'aicup/packed.py']
TEST_CSVS = ['tests/rank_prediction_test_public.csv', 'tests/rank_prediction_test_private.csv']


def analyzer():
    """從 repo 根目錄載入 aicup.analyze_humansl；只在需要棋譜解析時才載入。"""
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    from aicup import analyze_humansl
    return analyze_humansl


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_logp(probs):
    p = np.array(probs, dtype=float)
    return np.nanmean(np.log(np.clip(p, 1e-6, None)), axis=1)


def load_train():
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(TRAIN_GLOB))], ignore_index=True)
    df['key'] = df['rank'].astype(str) + '_' + df['game_id'].astype(str)
    assert not df['key'].duplicated().any()
    return df


def listed_pairs(players_file):
    listed = pd.read_csv(os.path.join(HA, players_file), dtype=str)
    return set(zip(listed['player_id'], listed['rank']))


def hd_of(sgf):
    m = re.search(r'HD\[(\d+)\]', sgf)
    return int(m.group(1)) if m else 0  # 沒有 HD 標記視為互先


def make_p4_players(df, k, path):
    """寫出 train_p4 的玩家清單：不在 humansl-analysis/*.csv 任何清單內，HD < 2 局中多數色 ≥ k 局。"""
    used = set()
    for f in glob.glob(os.path.join(HA, '*.csv')):
        if os.path.abspath(f) == os.path.abspath(path):
            continue
        if 'player_id' in pd.read_csv(f, nrows=0).columns:
            used |= set(pd.read_csv(f, usecols=['player_id'], dtype=str)['player_id'])
    ok = df[df['sgf_content'].map(hd_of) < 2]
    cnt = ok.groupby(['player_id', 'rank', 'color']).size().unstack(fill_value=0)
    cnt = cnt[cnt.max(axis=1) >= k].reset_index()
    cnt = cnt[~cnt['player_id'].astype(str).isin(used)]
    cnt[['player_id', 'rank']].sort_values(['rank', 'player_id']).to_csv(path, index=False)
    print('train_p4 玩家清單 %d 位，排除已抽過 %d 位 -> %s' % (len(cnt), len(used), path))


def sample_same_color(df, listed, k, seed):
    """清單內每位玩家取 HD < 2 局中多數色（同數取黑）的 k 局，seed 固定時結果只由清單決定。"""
    key = set(zip(listed['player_id'], listed['rank']))
    d = df[[(str(p), str(r)) in key for p, r in zip(df['player_id'], df['rank'])]]
    d = d[d['sgf_content'].map(hd_of) < 2].sort_values('key')
    parts = []
    for _, g in d.groupby(['player_id', 'rank'], sort=True):
        nb, nw = (g['color'] == 'B').sum(), (g['color'] == 'W').sum()
        side = g[g['color'] == ('B' if nb >= nw else 'W')]
        assert len(side) >= k, (g['player_id'].iloc[0], len(side))
        parts.append(side.sample(n=k, random_state=seed))
    assert len(parts) == len(key), (len(parts), len(key))
    return pd.concat(parts, ignore_index=True)


def sample_topup(df, prev_run, k, seed, path):
    """補局：prev_run 的每位玩家取和 prev_run 相同的顏色、HD < 2、未抽過的局，剩餘 ≥ k 局的玩家抽 k 局。

    第一次執行時寫出玩家清單 path。
    """
    prev = pd.read_csv(os.path.join(HA, 'hist-' + prev_run, 'games.csv'), dtype=str)
    prev['key'] = prev['rank'] + '_' + prev['game_id']
    color = prev.groupby(['player_id', 'rank'])['color'].agg(lambda c: c.iloc[0])
    assert (prev.groupby(['player_id', 'rank'])['color'].nunique() == 1).all()
    d = df[~df['key'].isin(set(prev['key'])) & (df['sgf_content'].map(hd_of) < 2)].copy()
    d['pk'] = list(zip(d['player_id'].astype(str), d['rank'].astype(str)))
    want = {pk: c for pk, c in color.items()}
    d = d[[want.get(pk) == c for pk, c in zip(d['pk'], d['color'])]].sort_values('key')
    cnt = d.groupby('pk').size()
    ok = sorted(pk for pk, n in cnt.items() if n >= k)
    if not os.path.exists(path):
        pd.DataFrame(ok, columns=['player_id', 'rank']).to_csv(path, index=False)
        print('%s 補局玩家 %d 位 -> %s' % (prev_run, len(ok), path))
    listed = pd.read_csv(path, dtype=str)
    keep = set(zip(listed['player_id'], listed['rank']))
    assert keep <= set(ok)
    d = d[d['pk'].isin(keep)]
    return pd.concat([g.sample(n=k, random_state=seed) for _, g in d.groupby('pk', sort=True)], ignore_index=True)


def control_rows(df):
    """對照組：val648-p3.jsonl 前 40 局，回傳 (棋譜列, ref)。"""
    old = []
    with open(os.path.join(HA, 'val648-p3.jsonl'), encoding='utf-8') as f:
        for line in f:
            old.append(json.loads(line))
            if len(old) == N_CTRL:
                break
    ref = {r['key']: mean_logp(r['probs'])[:2].tolist() for r in old}
    keys = [r['key'] for r in old]
    return df.set_index('key').loc[keys].reset_index()[GAME_COLS], ref


def run_games(run, cfg, df):
    """回傳這個執行的棋譜列，順序是分析順序。測試集回傳 None。"""
    if cfg.get('test'):
        return None
    ah = analyzer()

    if 'old' in cfg and not run.startswith('trainpool'):
        old = read_jsonl(os.path.join(HA, cfg['old']))
        pid_rank = listed_pairs(cfg['players'])
        old = [r for r in old if (r['player_id'], r['key'].split('_', 1)[0]) in pid_rank]
        keys = [r['key'] for r in old]
        rows = df.set_index('key').loc[keys].reset_index()
        assert (rows['player_id'].astype(str).values == np.array([r['player_id'] for r in old])).all()
        return rows[GAME_COLS]
    if cfg.get('topup'):
        rows = sample_topup(df, cfg['topup'], cfg['same_color'], SEED, os.path.join(HA, cfg['players']))
        assert len(rows) == cfg['n'], (len(rows), cfg['n'])
        return rows[GAME_COLS]
    if cfg.get('same_color'):
        path = os.path.join(HA, cfg['players'])
        if not os.path.exists(path):
            make_p4_players(df, cfg['same_color'], path)
        listed = pd.read_csv(path, dtype=str)
        assert len(listed) * cfg['same_color'] == cfg['n'], (len(listed), cfg['n'])
        return sample_same_color(df, listed, cfg['same_color'], SEED)[GAME_COLS]
    listed = pd.read_csv(os.path.join(HA, cfg.get('sample', cfg['players'])), dtype=str)
    rows = ah.sample_listed_players(df, listed, PER_PLAYER, SEED)
    keep = listed_pairs(cfg['players'])
    mask = [(str(p), str(r)) in keep for p, r in zip(rows['player_id'], rows['rank'])]
    return rows[mask][GAME_COLS]


def out_name(cfg):
    return '%s-p3%s-hist.jsonl' % (cfg['base'], '-opp' if cfg['side'] == 'opponent' else '')


def check_run(run, cfg, out):
    """檢查局數，以及局可以從官方 CSV 重新抽樣產生。"""
    ah = analyzer()

    if cfg.get('test'):
        keys = [g['key'] for g in ah.load_test_games(os.path.join(out, 'tests'))]
        assert len(keys) == cfg['n'] == len(set(keys)), len(keys)
        return
    back = ah.load_games(os.path.join(out, 'games.csv'), 0, SEED, 0)
    keys = [g['key'] for g in back]
    assert len(keys) == cfg['n'] == len(set(keys)), (len(keys), cfg['n'])


def make_notebook(run, cfg, move_stats):
    total, out = cfg['n'], out_name(cfg)
    src = '--test-dir /content/hist/tests' if cfg.get('test') else '--csv /content/hist/games.csv'
    flags = '--pre-root-history' + (' --side opponent' if cfg['side'] == 'opponent' else '')
    flags += ' --move-stats' if move_stats else ''
    eval_flag = '--no-eval' if cfg.get('no_eval') else '--eval-visits 1'
    prof_str = ' '.join(PROFILES3)
    override_str = 'numAnalysisThreads=32,numSearchThreadsPerAnalysisThread=1'
    
    md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s}
    code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
    
    cells = [
        md('# Human SL 抽取：%s（Google Colab T4 GPU 版）\n\n'
           '- 任務：%s（%s，共 %d 局）\n'
           '- Profiles：%s，損失 query：`%s`\n'
           '- 平行設定：`%s`\n'
           '- 旗標：`%s`\n'
           '- 進度每批備份至 Google Drive，斷線後重跑會接續\n'
           % (run, run, '對手側' if cfg['side'] == 'opponent' else '目標側', total, prof_str, eval_flag, override_str, flags)),
        
        md('### 1. 掛載 Google Drive 與建立工作目錄'),
        code("BATCH = 400\n"
             "RUN = %r\n"
             "TOTAL = %d\n"
             "OUT_NAME = %r\n"
             "DRIVE_BACKUP = '/content/drive/MyDrive/aicup-2026/hist'\n\n"
             "from google.colab import drive\n"
             "import os, shutil, time, zipfile\n\n"
             "# 1. 掛載 Google Drive\n"
             "drive.mount('/content/drive')\n"
             "os.makedirs(DRIVE_BACKUP, exist_ok=True)\n"
             "os.makedirs('/content/hist', exist_ok=True)\n\n"
             "# 2. 導入資料集與程式碼（優先從 Drive 或上傳讀取）\n"
             "if os.path.exists(f'{DRIVE_BACKUP}/{RUN}.zip'):\n"
             "    print(f'從 Google Drive 解壓 {RUN}.zip')\n"
             "    zipfile.ZipFile(f'{DRIVE_BACKUP}/{RUN}.zip').extractall('/content/hist')\n"
             "elif os.path.exists(f'/content/{RUN}.zip'):\n"
             "    print(f'從本地上傳解壓 {RUN}.zip')\n"
             "    zipfile.ZipFile(f'/content/{RUN}.zip').extractall('/content/hist')\n"
             "elif os.path.exists('/content/games.csv') or os.path.exists('/content/tests'):\n"
             "    os.system('cp -rf /content/games.csv /content/ctrl.csv /content/ref.json /content/aicup /content/tests /content/hist/ 2>/dev/null || true')\n"
             "else:\n"
             "    print(f'請將 {RUN}.zip 上傳到 Colab 根目錄或放至 Google Drive：{DRIVE_BACKUP}')\n\n"
             "if not os.path.exists('/content/hist/ctrl.csv'):\n"
             "    raise FileNotFoundError(f'未找到 {RUN}.zip 或解壓檔案，請確認檔案位置！')\n\n"
             "# 3. 切換至工作目錄\n"
             "os.chdir('/content/hist')\n\n"
             "# 4. 安裝套件與驗證硬體\n"
             "!pip install -q sgfmill\n"
             "!nvidia-smi -L\n\n"
             "LOCAL = f'/content/{OUT_NAME}'\n"
             "REMOTE = f'{DRIVE_BACKUP}/{OUT_NAME}'\n"
             "CTRL = f'/content/{RUN}-ctrl.jsonl'\n\n"
             "# 5. 自動從 Google Drive 還原已完成的進度\n"
             "if os.path.exists(REMOTE) and not os.path.exists(LOCAL):\n"
             "    shutil.copy2(REMOTE, LOCAL)\n"
             "if os.path.exists(LOCAL):\n"
             "    !wc -l {LOCAL}" % (run, total, out)),
        
        md('### 2. 下載 KataGo 引擎與雙模型權重（自動快取至 Google Drive）'),
        code('%%bash\n'
             'DRIVE_KATAGO=/content/drive/MyDrive/aicup-2026/katago\n'
             'mkdir -p /content/katago /content/models "$DRIVE_KATAGO"\n\n'
             '# 1. 取得 KataGo 執行檔\n'
             'ZIP="katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip"\n'
             'if [ ! -f /content/katago/katago-bin ]; then\n'
             '    if [ -f "$DRIVE_KATAGO/$ZIP" ]; then\n'
             '        cp -u "$DRIVE_KATAGO/$ZIP" /content/katago/\n'
             '    else\n'
             '        echo "Downloading KataGo binary..."\n'
             '        wget -q -O "/content/katago/$ZIP" "https://github.com/lightvector/KataGo/releases/download/v1.18.2/$ZIP"\n'
             '        cp -u "/content/katago/$ZIP" "$DRIVE_KATAGO/" 2>/dev/null || true\n'
             '    fi\n'
             '    unzip -qjo "/content/katago/$ZIP" -d /content/katago/\n'
             '    mv /content/katago/katago /content/katago/katago-bin\n'
             '    cat << \'EOF\' > /content/katago/katago\n'
             '#!/bin/bash\n'
             'CUDA_PATHS=$(python3 -c "import glob; print(\':\'.join(glob.glob(\'/usr/local/cuda*/lib64\') + glob.glob(\'/usr/local/lib/python3*/dist-packages/nvidia/*/lib\')))")\n'
             'export LD_LIBRARY_PATH=$CUDA_PATHS:/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n'
             'exec /content/katago/katago-bin "$@"\n'
             'EOF\n'
             '    chmod +x /content/katago/katago\n'
             'fi\n'
             'mkdir -p /content/katago/gtp_logs\n\n'
             '# 2. 取得雙模型（優先從 Drive 複製，缺失則下載並自動備份回 Drive）\n'
             'if ls "$DRIVE_KATAGO"/*.bin.gz 1> /dev/null 2>&1; then\n'
             '    cp -u "$DRIVE_KATAGO"/*.bin.gz /content/models/\n'
             'fi\n'
             'if [ ! -f /content/models/b18c384nbt-humanv0.bin.gz ]; then\n'
             '    echo "Downloading human model..."\n'
             '    wget -q -O /content/models/b18c384nbt-humanv0.bin.gz https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz\n'
             '    cp -u /content/models/b18c384nbt-humanv0.bin.gz "$DRIVE_KATAGO/" 2>/dev/null || true\n'
             'fi\n'
             'if [ ! -f /content/models/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz ]; then\n'
             '    echo "Downloading main 28b model..."\n'
             '    wget -q -O /content/models/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz\n'
             '    cp -u /content/models/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz "$DRIVE_KATAGO/" 2>/dev/null || true\n'
             'fi\n\n'
             '# 3. 驗證\n'
             '/content/katago/katago version\n'
             'ls -lh /content/models'),
        
        md('### 3. 對照組檢驗（確認 11k/1d 完全一致）'),
        code("import json\n"
             "import numpy as np\n\n"
             "if os.path.exists(CTRL):\n"
             "    os.remove(CTRL)\n"
             "!python -m aicup.analyze_humansl --csv /content/hist/ctrl.csv "
             "--model-dir /content/models --katago-dir /content/katago "
             "--model kata1-b28c512nbt-s13255194368-d5935380940.bin.gz --human-model b18c384nbt-humanv0.bin.gz "
             "--profiles %s --eval-visits 1 "
             '--override-config "%s" --out {CTRL}\n\n'
             "def mean_logp(probs):\n"
             "    p = np.array(probs, dtype=float)\n"
             "    return np.nanmean(np.log(np.clip(p, 1e-6, None)), axis=1)\n\n"
             "ref = json.load(open('/content/hist/ref.json'))\n"
             "rows = [json.loads(l) for l in open(CTRL)]\n"
             "diffs = np.concatenate([np.abs(mean_logp(r['probs'])[:2] - np.array(ref[r['key']])[:2]) for r in rows])\n"
             "print('對照組 %%d 局，11k/1d 中位數絕對差 %%.4g，最大差 %%.4g' %% (len(rows), np.median(diffs), diffs.max()))\n"
             "if len(rows) < %d or np.median(diffs) > 0.03:\n"
             "    print('環境和原本的執行不一致，停止')\n"
             "    raise SystemExit('對照組不一致')"
             % (prof_str, override_str, N_CTRL)),
        
        md('### 4. 主分析執行（分批寫入 Checkpoint，自動備份至 Google Drive）'),
        code("t0 = time.time()\n"
             "if os.path.exists(REMOTE) and not os.path.exists(LOCAL):\n"
             "    shutil.copy2(REMOTE, LOCAL)\n"
             "n0 = sum(1 for _ in open(LOCAL)) if os.path.exists(LOCAL) else 0\n"
             "print('目前已完成：%%d / %d 局' %% n0)\n"
             "for n in sorted({*range(BATCH, TOTAL, BATCH), TOTAL}):\n"
             "    if n <= n0:\n"
             "        continue\n"
             "    !python -m aicup.analyze_humansl %s "
             "--model-dir /content/models --katago-dir /content/katago "
             "--model kata1-b28c512nbt-s13255194368-d5935380940.bin.gz --human-model b18c384nbt-humanv0.bin.gz "
             "--profiles %s %s %s "
             '--override-config "%s" '
             "--limit {n} --out {LOCAL}\n"
             "    # 每批自動同步回 Google Drive 備份，斷線不掉局\n"
             "    !cp -f {LOCAL} {REMOTE} && wc -l {REMOTE}\n"
             "n1 = sum(1 for _ in open(LOCAL))\n"
             "print('本 session 完成 %%d 局，%%.0f 秒，每局 %%.2f 秒' %% (n1 - n0, time.time() - t0, (time.time() - t0) / max(n1 - n0, 1)))"
             % (total, src, prof_str, eval_flag, flags, override_str)),
        
        md('### 5. 輸出完整性驗證'),
        code("rows = [json.loads(l) for l in open(LOCAL)]\n"
             "keys = [r['key'] for r in rows]\n"
             "print('已完成 %d / %d 局，重複 key %d' % (len(rows), TOTAL, len(keys) - len(set(keys))))\n"
             "print('pre_root_history 全為真：', all(r.get('pre_root_history') for r in rows))\n"
             "print('profiles：', rows[0].get('profiles'))\n"
             "print('score_loss 存在：', 'score_loss' in rows[0])\n"
             "print('side：', sorted({r['side'] for r in rows}))\n"
             "print('欄位：', sorted(rows[0]))\n"
             "print('檔案已成功產出並備份至：', REMOTE)"),
        
        md('### 6. 釋放 Runtime（可選）'),
        code("# 全部完成後可取消註解執行，避免閒置扣除 Colab 運算額度\n"
             "# from google.colab import drive, runtime\n"
             "# drive.flush_and_unmount()\n"
             "# runtime.unassign()")
    ]
    for c in cells:
        lines = c['source'].split('\n')
        c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {'cells': cells, 'metadata': {'accelerator': 'GPU',
                                         'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 0}


def build(run, df, ctrl, ref, move_stats):
    cfg = RUNS[run]
    out = os.path.join(HA, 'hist-' + run)
    os.makedirs(out, exist_ok=True)
    rows = run_games(run, cfg, df)
    if rows is not None:
        rows.to_csv(os.path.join(out, 'games.csv'), index=False)
    else:
        os.makedirs(os.path.join(out, 'tests'), exist_ok=True)
        for p in TEST_CSVS:
            with open(p, 'rb') as fi, open(os.path.join(out, 'tests', os.path.basename(p)), 'wb') as fo:
                fo.write(fi.read())
    ctrl.to_csv(os.path.join(out, 'ctrl.csv'), index=False)
    with open(os.path.join(out, 'ref.json'), 'w') as f:
        json.dump(ref, f)
    check_run(run, cfg, out)

    zpath = os.path.join(out, run + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for src in BASE_FILES + (TEST_FILES if cfg.get('test') else []):
            z.write(src, src)
        names = ['ctrl.csv', 'ref.json'] + (['games.csv'] if rows is not None
                                            else ['tests/' + os.path.basename(p) for p in TEST_CSVS])
        for name in names:
            z.write(os.path.join(out, name), name)
    with open(os.path.join(out, run + '.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(make_notebook(run, cfg, move_stats), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, 'bundle.json'), 'w') as f:
        json.dump({'run': run, 'n': cfg['n'], 'side': cfg['side'], 'move_stats': move_stats,
                   'out': out_name(cfg)}, f, indent=1)
    print('OK %-14s games=%d zip=%.1f MB move_stats=%s' % (run, cfg['n'], os.path.getsize(zpath) / 1e6, move_stats))


def build_notebook(run, move_stats):
    cfg = RUNS[run]
    out = os.path.join(HA, 'hist-' + run)
    os.makedirs(out, exist_ok=True)
    nb_path = os.path.join(out, run + '.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(make_notebook(run, cfg, move_stats), f, ensure_ascii=False, indent=1)
    print('OK notebook %-14s -> %s (move_stats=%s)' % (run, nb_path, move_stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--run', choices=sorted(RUNS))
    g.add_argument('--all', action='store_true')
    ap.add_argument('--move-stats', action='store_true', help='主分析多輸出 move_rank、pmax、entropy')
    ap.add_argument('--notebook-only', action='store_true', help='只重新產生 notebook，不重新抽樣與打包')
    args = ap.parse_args()
    runs = sorted(RUNS) if args.all else [args.run]
    if args.notebook_only:
        for run in runs:
            build_notebook(run, args.move_stats)
        return
    df = load_train()
    ctrl, ref = control_rows(df)
    for run in runs:
        build(run, df, ctrl, ref, args.move_stats)


if __name__ == '__main__':
    main()
