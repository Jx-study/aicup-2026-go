"""產生 b28 軌跡特徵 Pilot 的打包檔與 notebook，輸出到 humansl-analysis/b28-traj/

在 Google Colab GPU 上執行：zip 內含程式與 newdev 6,760 局的棋譜 CSV，可直接上傳或放 Google Drive。
從 repo 根目錄執行：
    python utils/b28traj_bundle.py
notebook 先跑步驟 0（50 局吞吐量，外推 240 萬查詢回合），通過才跑 newdev 全部局，part 直接寫在 Drive，斷線重跑會接續。
"""
import json
import os
import zipfile

import numpy as np
import pandas as pd

import hist_bundle as hb

OUT = os.path.join(hb.HA, 'b28-traj')
FILES = ['aicup/__init__.py', 'aicup/analyze_humansl.py', 'aicup/scoring.py', 'aicup/b28_traj.py']
OVERRIDE = 'numAnalysisThreads=32,numSearchThreadsPerAnalysisThread=1'
FULL_TURNS = 2_400_000  # 全量 50,000 局 × 48 個查詢回合
MAX_HOURS, MAX_UNITS = 8, 80
RATES = {'A100': 6.75, 'L4': 1.94, 'T4': 1.14}  # Colab 每小時運算單元


def katago_cell():
    """沿用 hist_bundle notebook 的 KataGo 安裝 cell。"""
    nb = hb.make_notebook('val648', hb.RUNS['val648'], False)
    cell = next(c for c in nb['cells'] if c['cell_type'] == 'code' and 'DRIVE_KATAGO=' in ''.join(c['source']))
    return ''.join(cell['source'])


def make_notebook(n_games):
    md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s}
    code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
    common = '--model-dir /content/models --katago-dir /content/katago --override-config "{OVERRIDE}"'
    cells = [
        md('# b28 軌跡特徵 Pilot（Colab GPU）\n\n'
           '- 步驟 0：前 50 局量吞吐量，外推全量 %d 個查詢回合；超過 %d 小時或 %d 單元就停止\n'
           '- Pilot：newdev %d 局，每局最多 48 個查詢回合，part 直接寫進 Google Drive，斷線重跑會接續\n'
           % (FULL_TURNS, MAX_HOURS, MAX_UNITS, n_games)),
        md('### 1. 掛載 Google Drive 並解壓'),
        code("OVERRIDE = '%s'  # 步驟 0 未通過時可調整後重跑步驟 0\n"
             "DRIVE_OUT = '/content/drive/MyDrive/aicup-2026/b28-traj'\n\n"
             "from google.colab import drive\n"
             "import json, os, re, shutil, subprocess, zipfile\n"
             "drive.mount('/content/drive')\n"
             "os.makedirs(DRIVE_OUT, exist_ok=True)\n"
             "os.makedirs('/content/b28', exist_ok=True)\n"
             "for src in (f'{DRIVE_OUT}/b28traj.zip', '/content/b28traj.zip'):\n"
             "    if os.path.exists(src):\n"
             "        zipfile.ZipFile(src).extractall('/content/b28')\n"
             "        break\n"
             "else:\n"
             "    raise FileNotFoundError('請把 b28traj.zip 上傳到 /content 或放到 ' + DRIVE_OUT)\n"
             "os.chdir('/content/b28')\n"
             "!pip install -q sgfmill\n"
             "gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout\n"
             "print(gpu)\n"
             "RATES = %s\n"
             "RATE = next((v for k, v in RATES.items() if k in gpu), None)\n"
             "assert RATE is not None, '不認得這個 GPU，請手動設定 RATE（每小時運算單元）'\n"
             "print('每小時運算單元', RATE)" % (OVERRIDE, json.dumps(RATES))),
        md('### 2. 下載 KataGo 引擎與模型'),
        code(katago_cell()),
        md('### 3. 步驟 0：吞吐量防呆'),
        code("STEP0 = f'{DRIVE_OUT}/step0.json'\n"
             "if os.path.exists(STEP0):\n"
             "    s0 = json.load(open(STEP0))\n"
             "    print('沿用已記錄的步驟 0：', s0)\n"
             "else:\n"
             "    shutil.rmtree('/content/step0', ignore_errors=True)\n"
             "    log = !python -m aicup.b28_traj --csv /content/b28/games.csv --limit 50 --out-dir /content/step0 " + common + "\n"
             "    print('\\n'.join(log))\n"
             "    m = [re.search(r'turns_per_sec=([0-9.]+) games=(\\d+) turns=(\\d+) seconds=([0-9.]+)', l) for l in log]\n"
             "    m = [x for x in m if x]\n"
             "    assert m, '沒有吞吐量輸出，看上面的錯誤訊息'\n"
             "    tps, games, turns, sec = float(m[-1][1]), int(m[-1][2]), int(m[-1][3]), float(m[-1][4])\n"
             "    hours = %d / tps / 3600\n"
             "    s0 = {'gpu': gpu.strip(), 'override': OVERRIDE, 'turns_per_sec': tps, 'games': games, 'turns': turns,\n"
             "          'seconds': sec, 'full_hours': hours, 'full_units': hours * RATE,\n"
             "          'pass': hours <= %d and hours * RATE <= %d}\n"
             "    json.dump(s0, open(STEP0, 'w'), indent=1)\n"
             "    print(s0)\n"
             "print('步驟 0：每秒 %%.1f 個查詢回合，全量外推 %%.2f 小時、%%.1f 單元 -> %%s'\n"
             "      %% (s0['turns_per_sec'], s0['full_hours'], s0['full_units'], '通過' if s0['pass'] else '未通過'))\n"
             "if not s0['pass']:\n"
             "    raise SystemExit('步驟 0 未通過，不跑 Pilot；調整 OVERRIDE 後刪除 step0.json 重跑')"
             % (FULL_TURNS, MAX_HOURS, MAX_UNITS)),
        md('### 4. Pilot：newdev 全部局（part 直接寫在 Drive）'),
        code("!python -m aicup.b28_traj --csv /content/b28/games.csv --out-dir {DRIVE_OUT}/newdev " + common),
        md('### 5. 驗證'),
        code("import numpy as np\n"
             "import pandas as pd\n"
             "from aicup.b28_traj import FEATURES, load_parts\n"
             "z = load_parts(f'{DRIVE_OUT}/newdev')\n"
             "assert z is not None, '沒有任何 part，看上一個 cell 的錯誤訊息'\n"
             "csv = pd.read_csv('/content/b28/games.csv', dtype={'player_id': str, 'game_id': str})\n"
             "keys = set(csv['rank'] + '_' + csv['game_id'])\n"
             "print('局數', len(z['key']), '/', len(keys), '，重複 key', len(z['key']) - len(set(z['key'])),\n"
             "      '，缺少', len(keys - set(z['key'])))\n"
             "f = z['feats']\n"
             "n = (z['turns'] >= 0).sum(2)\n"
             "print('每側手數中位數', np.median(n, 0), '，最少', n.min(0))\n"
             "for i, name in enumerate(FEATURES):\n"
             "    print('%-16s 目標 %9.4f  對手 %9.4f' % (name, np.nanmean(f[:, 0, :, i]), np.nanmean(f[:, 1, :, i])))"),
    ]
    for c in cells:
        lines = c['source'].split('\n')
        c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {'cells': cells, 'metadata': {'accelerator': 'GPU',
                                         'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 0}


def main():
    os.makedirs(OUT, exist_ok=True)
    games = pd.read_csv(os.path.join(hb.HA, 'hist-newdev', 'games.csv'), dtype={'player_id': str, 'game_id': str})
    keys = games['rank'] + '_' + games['game_id']
    cache = np.load(os.path.join(hb.HA, 'cache', 'newdev-6d.npz'))
    assert set(keys) == set(cache['d_key']) and len(keys) == keys.nunique(), '棋譜 CSV 和 newdev-6d 快取不是同一批局'
    games = games[hb.GAME_COLS]
    games.to_csv(os.path.join(OUT, 'games.csv'), index=False)
    with zipfile.ZipFile(os.path.join(OUT, 'b28traj.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for src in FILES:
            z.write(src, src)
        z.write(os.path.join(OUT, 'games.csv'), 'games.csv')
    with open(os.path.join(OUT, 'b28traj.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(make_notebook(len(games)), f, ensure_ascii=False, indent=1)
    print('newdev %d 局、%d 位 -> %s' % (len(games), games['player_id'].nunique(), OUT))


if __name__ == '__main__':
    main()
