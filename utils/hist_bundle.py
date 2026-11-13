"""產生 Human SL 分析用的打包檔與 notebook，輸出到 humansl-analysis/hist-<run>/

專為 Kaggle T4 x2 雙 GPU 最佳化：zip 內含程式與棋譜，可直接作為 Kaggle Dataset 或上傳執行。
從 repo 根目錄執行：
    python utils/hist_bundle.py --run test --move-stats
    python utils/hist_bundle.py --run val648 --move-stats
    python utils/hist_bundle.py --run trainpool2_p1 --move-stats
    python utils/hist_bundle.py --run trainpool2_p2 --move-stats
    python utils/hist_bundle.py --run holdout --move-stats
--move-stats 時主分析多輸出 move_rank、pmax、entropy。
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from aicup import analyze_humansl as ah

HA = 'humansl-analysis'
TRAIN_GLOB = 'training-v1.1/training/train_*.csv'
# 方案乙：最優 3 Profiles 組合
PROFILES3 = ['rank_11k', 'rank_1d', 'rank_2d']
N_CTRL = 40
SEED, PER_PLAYER = 42, 20

NEW498 = 'val-players-new-v180000.csv'
RUNS = {
    'val648': dict(base='val648', old='val648-p3.jsonl', players='val-players.csv', n=12960),
    'test': dict(base='test', test=True, n=10000),
    'train_p1': dict(base='train-p1', players='train-p1-v180000.csv', n=20000),
    'train_p2': dict(base='train-p2', players='train-p2-v180000.csv', n=20000),
    'holdout': dict(base='holdout', players='holdout-v180000.csv', sample=NEW498, n=3200),
}
for _name in list(RUNS):
    RUNS[_name + '_opp'] = dict(RUNS[_name], side='opponent')
for _cfg in RUNS.values():
    _cfg.setdefault('side', 'target')

GAME_COLS = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']
BASE_FILES = ['aicup/__init__.py', 'aicup/analyze_humansl.py', 'aicup/scoring.py']
TEST_FILES = ['aicup/predict_rank.py', 'aicup/color.py', 'aicup/extract_features.py', 'aicup/packed.py']
TEST_CSVS = ['tests/rank_prediction_test_public.csv', 'tests/rank_prediction_test_private.csv']


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
    if 'old' in cfg and not run.startswith('trainpool'):
        old = read_jsonl(os.path.join(HA, cfg['old']))
        pid_rank = listed_pairs(cfg['players'])
        old = [r for r in old if (r['player_id'], r['key'].split('_', 1)[0]) in pid_rank]
        keys = [r['key'] for r in old]
        rows = df.set_index('key').loc[keys].reset_index()
        assert (rows['player_id'].astype(str).values == np.array([r['player_id'] for r in old])).all()
        return rows[GAME_COLS]
    listed = pd.read_csv(os.path.join(HA, cfg.get('sample', cfg['players'])), dtype=str)
    rows = ah.sample_listed_players(df, listed, PER_PLAYER, SEED)
    keep = listed_pairs(cfg['players'])
    mask = [(str(p), str(r)) in keep for p, r in zip(rows['player_id'], rows['rank'])]
    return rows[mask][GAME_COLS]


def out_name(cfg):
    return '%s-p3%s-hist.jsonl' % (cfg['base'], '-opp' if cfg['side'] == 'opponent' else '')


def check_run(run, cfg, out):
    """檢查局數，以及局可以從官方 CSV 重新抽樣產生。"""
    if cfg.get('test'):
        keys = [g['key'] for g in ah.load_test_games(os.path.join(out, 'tests'))]
        assert len(keys) == cfg['n'] == len(set(keys)), len(keys)
        return
    back = ah.load_games(os.path.join(out, 'games.csv'), 0, SEED, 0)
    keys = [g['key'] for g in back]
    assert len(keys) == cfg['n'] == len(set(keys)), (len(keys), cfg['n'])


def make_notebook(run, cfg, move_stats):
    total, out = cfg['n'], out_name(cfg)
    src = '--test-dir /kaggle/working/hist/tests' if cfg.get('test') else '--csv /kaggle/working/hist/games.csv'
    flags = '--pre-root-history' + (' --side opponent' if cfg['side'] == 'opponent' else '')
    flags += ' --move-stats' if move_stats else ''
    prof_str = ' '.join(PROFILES3)
    override_str = 'numAnalysisThreads=32,numSearchThreadsPerAnalysisThread=1,numNNServerThreadsPerModel=2,cudaDeviceToUseThread0=0,cudaDeviceToUseThread1=1'
    
    md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s}
    code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
    
    cells = [
        md('# Human SL 抽取：%s（Kaggle T4 x2 雙 GPU 版）\n\n'
           '- 任務：%s（%s，共 %d 局）\n'
           '- Profiles：%s + eval_visits=1（包含目數損失）\n'
           '- 雙 GPU 平行設定：`%s`\n'
           '- 旗標：`%s`\n'
           % (run, run, '對手側' if cfg['side'] == 'opponent' else '目標側', total, prof_str, override_str, flags)),
        
        md('### 1. 建立工作目錄與導入資料集'),
        code("BATCH = 400\n"
             "RUN = %r\n"
             "TOTAL = %d\n"
             "OUT_NAME = %r\n\n"
             "import os, glob, shutil, time, zipfile\n\n"
             "# 1. 建立工作目錄\n"
             "os.makedirs('/kaggle/working/hist', exist_ok=True)\n"
             "# 自動偵測 Kaggle Dataset 路徑（支援掛載的 Dataset 或直接上傳）\n"
             "dataset_candidates = [\n"
             "    f'/kaggle/input/{RUN}-dataset',\n"
             "    f'/kaggle/input/{RUN}',\n"
             "] + glob.glob(f'/kaggle/input/**/{RUN}*', recursive=True)\n"
             "DATASET_DIR = None\n"
             "for d in dataset_candidates:\n"
             "    if os.path.exists(d):\n"
             "        DATASET_DIR = d if os.path.isdir(d) else os.path.dirname(d)\n"
             "        break\n"
             "if DATASET_DIR is None:\n"
             "    found = [p for p in glob.glob('/kaggle/input/**/aicup', recursive=True) if os.path.isdir(p)]\n"
             "    DATASET_DIR = os.path.dirname(found[0]) if found else f'/kaggle/input/{RUN}-dataset'\n"
             "print('使用資料集路徑：', DATASET_DIR)\n\n"
             "# 2. 導入資料\n"
             "if os.path.exists(f'{DATASET_DIR}/{RUN}.zip'):\n"
             "    zipfile.ZipFile(f'{DATASET_DIR}/{RUN}.zip').extractall('/kaggle/working/hist')\n"
             "elif os.path.exists(f'{RUN}.zip'):\n"
             "    zipfile.ZipFile(f'{RUN}.zip').extractall('/kaggle/working/hist')\n"
             "else:\n"
             "    os.system(f'cp -rf {DATASET_DIR}/* /kaggle/working/hist/')\n\n"
             "# 3. 切換至工作目錄\n"
             "os.chdir('/kaggle/working/hist')\n\n"
             "# 4. 安裝套件與驗證硬體\n"
             "!pip install -q sgfmill\n"
             "!nvidia-smi -L\n"
             "!ls -la /kaggle/working/hist\n\n"
             "LOCAL = '/kaggle/working/' + OUT_NAME\n"
             "CTRL = '/kaggle/working/' + RUN + '-ctrl.jsonl'\n"
             "if os.path.exists(LOCAL):\n"
             "    !wc -l {LOCAL}" % (run, total, out)),
        
        md('### 2. 下載 KataGo 引擎與雙模型權重（一般主模型 + Human SL 模型）'),
        code('%%bash\n'
             'mkdir -p /kaggle/working/models /kaggle/working/katago\n'
             'cd /kaggle/working/katago\n'
             'ZIP=katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip\n'
             '[ -f katago ] || { wget -q https://github.com/lightvector/KataGo/releases/download/v1.18.2/$ZIP && unzip -qjo $ZIP; }\n'
             'chmod +x katago\n'
             '[ -f /kaggle/working/models/b18c384nbt-humanv0.bin.gz ] || wget -q -O /kaggle/working/models/b18c384nbt-humanv0.bin.gz '
             'https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz\n'
             '[ -f /kaggle/working/models/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz ] || wget -q -O /kaggle/working/models/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz '
             'https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz\n'
             'ls -lh /kaggle/working/models /kaggle/working/katago/katago analysis_example.cfg'),
        
        md('### 3. 對照組檢驗（確認 KataGo 在此環境計算 11k/1d 完全一致）'),
        code("import json\n"
             "import numpy as np\n\n"
             "os.environ['APPIMAGE_EXTRACT_AND_RUN'] = '1'\n\n"
             "if os.path.exists(CTRL):\n"
             "    os.remove(CTRL)\n"
             "!python -m aicup.analyze_humansl --csv /kaggle/working/hist/ctrl.csv "
             "--model-dir /kaggle/working/models --katago-dir /kaggle/working/katago "
             "--model kata1-b28c512nbt-s13255194368-d5935380940.bin.gz --human-model b18c384nbt-humanv0.bin.gz "
             "--profiles %s --eval-visits 1 "
             '--override-config "%s" --out {CTRL} 2>&1 | tail -3\n\n'
             "def mean_logp(probs):\n"
             "    p = np.array(probs, dtype=float)\n"
             "    return np.nanmean(np.log(np.clip(p, 1e-6, None)), axis=1)\n\n"
             "ref = json.load(open('/kaggle/working/hist/ref.json'))\n"
             "rows = [json.loads(l) for l in open(CTRL)]\n"
             "diffs = np.concatenate([np.abs(mean_logp(r['probs'])[:2] - np.array(ref[r['key']])[:2]) for r in rows])\n"
             "print('對照組 %%d 局，11k/1d 中位數絕對差 %%.4g，最大差 %%.4g' %% (len(rows), np.median(diffs), diffs.max()))\n"
             "if len(rows) < %d or np.median(diffs) > 0.03:\n"
             "    print('環境和原本的執行不一致，停止')\n"
             "    raise SystemExit('對照組不一致')"
             % (prof_str, override_str, N_CTRL)),
        
        md('### 4. 主分析執行（分批寫入 Checkpoint，中途斷線自動接續）'),
        code("t0 = time.time()\n"
             "n0 = sum(1 for _ in open(LOCAL)) if os.path.exists(LOCAL) else 0\n"
             "print('目前已完成：%%d / %d 局' %% n0)\n"
             "for n in sorted({*range(BATCH, TOTAL, BATCH), TOTAL}):\n"
             "    if n <= n0:\n"
             "        continue\n"
             "    !python -m aicup.analyze_humansl %s "
             "--model-dir /kaggle/working/models --katago-dir /kaggle/working/katago "
             "--model kata1-b28c512nbt-s13255194368-d5935380940.bin.gz --human-model b18c384nbt-humanv0.bin.gz "
             "--profiles %s --eval-visits 1 %s "
             '--override-config "%s" '
             "--limit {n} --out {LOCAL} 2>&1 | tail -3\n"
             "    !wc -l {LOCAL}\n"
             "n1 = sum(1 for _ in open(LOCAL))\n"
             "print('本 session 完成 %%d 局，%%.0f 秒，每局 %%.2f 秒' %% (n1 - n0, time.time() - t0, (time.time() - t0) / max(n1 - n0, 1)))"
             % (total, src, prof_str, flags, override_str)),
        
        md('### 5. 輸出完整性驗證'),
        code("rows = [json.loads(l) for l in open(LOCAL)]\n"
             "keys = [r['key'] for r in rows]\n"
             "print('已完成 %d / %d 局，重複 key %d' % (len(rows), TOTAL, len(keys) - len(set(keys))))\n"
             "print('pre_root_history 全為真：', all(r.get('pre_root_history') for r in rows))\n"
             "print('profiles：', rows[0].get('profiles'))\n"
             "print('score_loss 存在：', 'score_loss' in rows[0])\n"
             "print('side：', sorted({r['side'] for r in rows}))\n"
             "print('欄位：', sorted(rows[0]))\n"
             "print('檔案已成功產出至：', LOCAL)")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--run', choices=sorted(RUNS))
    g.add_argument('--all', action='store_true')
    ap.add_argument('--move-stats', action='store_true', help='主分析多輸出 move_rank、pmax、entropy')
    args = ap.parse_args()
    df = load_train()
    ctrl, ref = control_rows(df)
    for run in (sorted(RUNS) if args.all else [args.run]):
        build(run, df, ctrl, ref, args.move_stats)


if __name__ == '__main__':
    main()
