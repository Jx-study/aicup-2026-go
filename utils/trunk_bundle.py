"""產生主幹判別式微調（aicup.trunk_cls）的 Colab 打包檔與 notebook，輸出到 humansl-analysis/trunk-<commit>/

zip 內含程式、KataGo python/ 的 5 個模組與棋譜，Colab 不需要完整訓練集。
棋譜：
    train-games.csv     train 切分每位玩家最多 cap × epochs 局（訓練集已依 SEED 打亂，取前面的局），每 epoch 從中抽 cap 局
    newdev-games.csv    newdev 子集（trunk_cls.newdev_subset）
    question-games.csv  題目檔全部題目用到的局
    tests/              競賽二測試題
程式有未 commit 的改動時拒絕，checkpoint 目錄以 commit hash 命名。

用法（在 repo 根目錄）：
    python utils/trunk_bundle.py --questions <questions.npz> --gate1 <newdev log loss 門檻>
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aicup import trunk_cls as tc  # noqa: E402

CODE = ['aicup/__init__.py', 'aicup/trunk_cls.py', 'aicup/extract_features.py', 'aicup/packed.py', 'aicup/scoring.py',
        'aicup/analyze_humansl.py', 'aicup/katago_torch.py', 'aicup/symmetry.py', 'aicup/predict_rank.py',
        'aicup/color.py']
KATAGO = ['board.py', 'features.py', 'sgfmetadata.py', 'model_pytorch.py', 'modelconfigs.py']
KATAGO_DIR = 'Note/katago/src/python'
TESTS = ['tests/rank_prediction_test_public.csv', 'tests/rank_prediction_test_private.csv']
COLS = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']


def commit_hash():
    """HEAD 的短 hash；CODE 有未 commit 的改動時 SystemExit。"""
    dirty = subprocess.run(['git', 'status', '--porcelain', '--'] + CODE, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit('以下檔案有未 commit 的改動，先 commit 並 push：\n' + dirty)
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()


def question_games(path, df):
    q = np.load(path)
    keys = sorted({k for t in ('SC', 'MX') for k in q[t + '_keys'].ravel() if k})
    idx = df['rank'].astype(str) + '_' + df['game_id'].astype(str)
    sub = df[idx.isin(keys)]
    if len(sub) != len(keys):
        raise SystemExit('題目檔有 %d 局，訓練集只找到 %d 局' % (len(keys), len(sub)))
    return sub


def notebook(commit, zip_name, gate, cap, epochs):
    md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s}
    code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
    cells = [
        md('# 主幹判別式微調（commit %s）\n\n'
           '- 第 3 格訓練，到 `TIME_LIMIT` 小時存 `last.pt` 停止，重跑第 1～3 格接續\n'
           '- 每個 epoch 結束印 newdev 各級等權 log loss，第 1 個 epoch > %.4f（門檻）就停止\n'
           '- 第 5 格推論：設定 `EPOCH` 為 newdev log loss 最低的 epoch' % (commit, gate)),
        code("COMMIT = %r\n"
             "ZIP = %r\n"
             "DRIVE = '/content/drive/MyDrive/aicup-2026'\n"
             "OUT = f'{DRIVE}/rank-trained-models/{COMMIT}'\n"
             "TIME_LIMIT = 4.5\n\n"
             "from google.colab import drive\n"
             "import os, shutil, subprocess, zipfile\n"
             "drive.mount('/content/drive')\n"
             "os.makedirs(OUT, exist_ok=True)\n"
             "src = f'{DRIVE}/trunk/{ZIP}' if os.path.exists(f'{DRIVE}/trunk/{ZIP}') else f'/content/{ZIP}'\n"
             "zipfile.ZipFile(src).extractall('/content/trunk')\n"
             "os.chdir('/content/trunk')\n"
             "os.environ['KATAGO_PYTHON'] = '/content/trunk/katago_python'\n"
             "CKPT = 'Note/katago/models/b18c384nbt-humanv0.ckpt'\n"
             "os.makedirs(os.path.dirname(CKPT), exist_ok=True)\n"
             "if os.path.exists(f'{DRIVE}/katago/b18c384nbt-humanv0.ckpt') and not os.path.exists(CKPT):\n"
             "    shutil.copy2(f'{DRIVE}/katago/b18c384nbt-humanv0.ckpt', CKPT)\n"
             "!pip install -q sgfmill\n"
             "gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout\n"
             "print(gpu)\n"
             "TRAIN_DTYPE = 'bf16' if ('L4' in gpu or 'A100' in gpu) else 'fp16'\n"
             "INFER_DTYPE = 'fp32'\n"
             "print('train', TRAIN_DTYPE, 'infer', INFER_DTYPE)\n"
             "DATA = '--train-csv train-games.csv --newdev-games-csv newdev-games.csv'\n"
             "\n"
             "def run(cmd):\n"
             "    print(cmd)\n"
             "    if os.system(cmd) != 0:\n"
             "        raise SystemExit('指令失敗')" % (commit, zip_name)),
        md('### 2. 精度檢查（訓練用 fp16 時，policy 那一行要通過：最大絕對差 ≤ 0.05、argmax 一致率 ≥ 99%）'),
        code("run(f'python -m aicup.trunk_cls dtype {DATA} --games 8')\n"
             "os.makedirs(f'{DRIVE}/katago', exist_ok=True)\n"
             "if not os.path.exists(f'{DRIVE}/katago/b18c384nbt-humanv0.ckpt'):\n"
             "    shutil.copy2(CKPT, f'{DRIVE}/katago/b18c384nbt-humanv0.ckpt')"),
        md('### 3. 訓練（重跑接續）'),
        code("run(f'python -u -m aicup.trunk_cls train {DATA} --out {OUT} --train-dtype {TRAIN_DTYPE} "
             "--infer-dtype {INFER_DTYPE} --cap %d --epochs %d --time-limit {TIME_LIMIT} --workers {os.cpu_count()}')"
             % (cap, epochs)),
        md('### 4. 各 epoch 的 newdev 結果與門檻'),
        code("import json\n"
             "rows = [json.loads(l) for l in open(f'{OUT}/history.jsonl') if 'newdev_nll' in l]\n"
             "for r in rows:\n"
             "    print('epoch %%d newdev log loss %%.4f 正確率 %%.4f %%s' %% (r['epoch'], r['newdev_nll'], r['newdev_acc'],\n"
             "          '通過' if r['newdev_nll'] <= %.4f else '未通過'))" % gate),
        md('### 5. 推論 val648 題目與測試集'),
        code("EPOCH = None  # 填 newdev log loss 最低的 epoch\n"
             "assert EPOCH is not None\n"
             "for flag, name in (('--questions questions.npz --games-csv question-games.csv', 'val-games'), ('--test tests', 'test-games')):\n"
             "    run(f'python -u -m aicup.trunk_cls infer --ckpt {OUT}/epoch-{EPOCH}.pt {flag} '\n"
             "        f'--out {OUT}/{name}-epoch-{EPOCH}.npz --infer-dtype {INFER_DTYPE} --workers {os.cpu_count()}')"),
    ]
    for c in cells:
        lines = c['source'].split('\n')
        c['source'] = [ln + '\n' for ln in lines[:-1]] + [lines[-1]]
    return {'cells': cells, 'metadata': {'accelerator': 'GPU', 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--questions', required=True, help='含 SC_keys、MX_keys 的題目 npz')
    ap.add_argument('--train-dir', default='training-v1.1/training')
    ap.add_argument('--gate1', type=float, required=True, help='newdev 各級等權 log loss 的門檻，印在 notebook')
    ap.add_argument('--cap', type=int, default=20, help='每個 epoch 每位玩家最多幾局，傳給 trunk_cls train')
    ap.add_argument('--epochs', type=int, default=3, help='傳給 trunk_cls train；每人局池 = cap × epochs')
    ap.add_argument('--allow-dirty', action='store_true', help='冒煙測試用，不檢查 commit，目錄名用 dirty')
    args = ap.parse_args()
    commit = 'dirty' if args.allow_dirty else commit_hash()
    out = os.path.join('humansl-analysis', 'trunk-' + commit)
    os.makedirs(out, exist_ok=True)

    val_df, tr_df = tc.load_split(args.train_dir)
    vk = set(val_df['rank'].astype(str) + '/' + val_df['player_id'])
    tk = set(tr_df['rank'].astype(str) + '/' + tr_df['player_id'])
    assert not vk & tk
    pool = tr_df.groupby(['rank', 'player_id'], sort=False).head(args.cap * args.epochs)
    nd = tc.newdev_subset(val_df)
    qg = question_games(args.questions, pd.concat([val_df, tr_df]))
    qk = set(qg['rank'].astype(str) + '/' + qg['player_id'])
    nk = set(nd['rank'].astype(str) + '/' + nd['player_id'])
    assert qk <= vk and nk <= vk, '評估玩家不在 val 切分'

    zip_name = 'trunk-%s.zip' % commit
    zpath = os.path.join(out, zip_name)
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in CODE:
            z.write(f, f)
        for f in KATAGO:
            z.write(os.path.join(KATAGO_DIR, f), 'katago_python/' + f)
        for f in TESTS:
            z.write(f, f)
        z.write(args.questions, 'questions.npz')
        for name, d in (('train-games.csv', pool), ('newdev-games.csv', nd), ('question-games.csv', qg)):
            z.writestr(name, d[COLS].to_csv(index=False))
    with open(os.path.join(out, 'trunk-%s.ipynb' % commit), 'w', encoding='utf-8') as f:
        json.dump(notebook(commit, zip_name, args.gate1, args.cap, args.epochs), f, ensure_ascii=False, indent=1)
    info = {'commit': commit, 'train_games': len(pool), 'train_players': len(tk), 'newdev_games': len(nd),
            'question_games': len(qg), 'zip_mb': round(os.path.getsize(zpath) / 1e6, 1)}
    with open(os.path.join(out, 'bundle.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    print(info)


if __name__ == '__main__':
    main()
