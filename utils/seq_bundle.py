"""產生免費 Colab T4 帳號用的序列模型 bundle：seq_bundle.zip 與 seq_t4.ipynb。

zip 內含 aicup/、network.py、utils.py、狀態序列檔、局部盤面檔與 A、B 組題目 CSV，
免費帳號不需專案 Drive 與 GitHub token。訓練的 checkpoint 寫在該帳號的
MyDrive/<folder>/runs/<ARM>/，中斷或換帳號後重跑同一個 cell 會從 last.pt 接續。
用法：python utils/seq_bundle.py
      python utils/seq_bundle.py --out-dir player-trained-models/seq-bundle --folder seqt4
"""
import argparse
import glob
import json
import os
import subprocess
import zipfile

ROOT_FILES = ('network.py', 'utils.py')
TEST_FILES = (
    'player_identification_candidates_public.csv', 'player_identification_test_public.csv',
    'player_identification_candidates_private.csv', 'player_identification_test_private.csv',
    'player_identification_candidates2_public.csv', 'player_identification_test2_public.csv',
    'player_identification_candidates2_private.csv', 'player_identification_test2_private.csv',
)
SEQ_FILES = ('train.npz', 'newdev.npz', 'val648.npz', 'hold.npz', 'pat3_vocab.npy')
PATCH_FILES = ('train_patch.npz', 'newdev_patch.npz', 'val648_patch.npz', 'hold_patch.npz')

# 各組訓練參數，pooling 在 notebook 的 POOL 填寫
ARMS = {
    'pv6': ['--state-feats', '--patch-dir', 'data/patch'],
    'state': ['--state-feats'],
    'stateconf': ['--state-feats', '--conv-kernel', '9'],
}


def git_info():
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'],
                           capture_output=True, text=True).stdout
    return {'head': head, 'dirty_files': [line[3:] for line in dirty.splitlines() if line.strip()]}


def cells(folder):
    arms = ', '.join("'%s'" % a for a in ARMS)
    return [
        ('markdown', '# 序列模型訓練與推論（免費 T4）\n\n'
                     '1. 把 `seq_bundle.zip` 上傳到這個帳號的 `MyDrive/%s/`。\n'
                     '2. 執行階段選 T4 GPU，改第一格的 `ARM`、`POOL`、`STAGE`，再執行全部 cell。\n'
                     '3. 輸出寫在 `MyDrive/%s/runs/<ARM>/`。中斷或換帳號時，把 `runs/` 複製到新帳號的同一路徑，重跑即可接續。\n'
                     '4. 推論完成後下載 `runs/<ARM>/` 的 `history.csv` 與 `emb-cache/`。' % (folder, folder)),
        ('code', "ARM = 'pv6'  # %s\n"
                 "POOL = 'all'  # 'target' 或 'all'，依 newdev 結果選擇\n"
                 "STAGE = 'train'  # 'train' 或 'infer'\n"
                 "EPOCHS = 12\n"
                 "ARMS = %s" % (arms, json.dumps(ARMS))),
        ('code', "from google.colab import drive\n"
                 "drive.mount('/content/drive')\n"
                 "import os, subprocess, sys\n"
                 "B = '/content/drive/MyDrive/%s'\n"
                 "RUN = os.path.join(B, 'runs', ARM)\n"
                 "os.makedirs(RUN, exist_ok=True)\n"
                 "if not os.path.isdir('/content/seq/aicup'):\n"
                 "    subprocess.run(['unzip', '-oq', B + '/seq_bundle.zip', '-d', '/content/seq'], check=True)\n"
                 "os.chdir('/content/seq')\n"
                 "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sgfmill'], check=True)\n"
                 "print(open('BUNDLE.json').read())\n"
                 "!nvidia-smi --query-gpu=name,memory.total --format=csv" % folder),
        ('code', "def run(cmd, log):\n"
                 "    print(' '.join(cmd), flush=True)\n"
                 "    with open(log, 'a') as f:\n"
                 "        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
                 "        for line in p.stdout:\n"
                 "            print(line, end='', flush=True)\n"
                 "            f.write(line)\n"
                 "    if p.wait():\n"
                 "        raise SystemExit('指令失敗，exit code %d' % p.returncode)\n"
                 "\n"
                 "extra = ARMS[ARM] + (['--pool-mode', 'all'] if POOL == 'all' else [])\n"
                 "ckpt = os.path.join(RUN, 'best_model.pth')\n"
                 "if STAGE == 'train':\n"
                 "    run([sys.executable, '-u', '-m', 'aicup.train_seq_emb', '--seq-dir', 'data/seq',\n"
                 "         '--ckpt-dir', RUN, '--epochs', str(EPOCHS), '--fp16', *extra], RUN + '/train.log')\n"
                 "else:\n"
                 "    pd_args = ['--patch-dir', 'data/patch'] if '--patch-dir' in extra else []\n"
                 "    for split in ('newdev', 'val648'):\n"
                 "        run([sys.executable, '-u', '-m', 'aicup.infer_seq_emb', '--mode', 'packed', '--ckpt', ckpt,\n"
                 "             '--seq-dir', 'data/seq', '--split', split, *pd_args,\n"
                 "             '--out', os.path.join(RUN, 'emb-cache', split + '_emb.npz')], RUN + '/infer.log')\n"
                 "    run([sys.executable, '-u', '-m', 'aicup.infer_seq_emb', '--mode', 'packed', '--ckpt', ckpt,\n"
                 "         '--seq-npz', 'data/seq/hold.npz', *pd_args,\n"
                 "         '--out', os.path.join(RUN, 'emb-cache', 'hold_emb.npz')], RUN + '/infer.log')\n"
                 "    run([sys.executable, '-u', '-m', 'aicup.infer_seq_emb', '--mode', 'test', '--ckpt', ckpt,\n"
                 "         '--groups', 'all', '--tests-dir', 'tests', '--out-dir', os.path.join(RUN, 'emb-cache')],\n"
                 "        RUN + '/infer.log')"),
        ('code', "from google.colab import runtime\n\ndrive.flush_and_unmount()\nruntime.unassign()"),
    ]


def notebook(cs):
    out = []
    for kind, src in cs:
        c = {'cell_type': kind, 'metadata': {}, 'source': src.splitlines(True)}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {'cells': out, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'T4'},
                                       'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 0}


def write_zip(path, seq_dir, patch_dir, tests_dir, info):
    """寫 bundle zip；npz 已壓縮，用 ZIP_STORED。"""
    missing = [os.path.join(seq_dir, f) for f in SEQ_FILES if not os.path.exists(os.path.join(seq_dir, f))]
    missing += [os.path.join(patch_dir, f) for f in PATCH_FILES if not os.path.exists(os.path.join(patch_dir, f))]
    missing += [os.path.join(tests_dir, f) for f in TEST_FILES if not os.path.exists(os.path.join(tests_dir, f))]
    if missing:
        raise SystemExit('缺少檔案：%s' % ', '.join(missing))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
        for f in sorted(glob.glob('aicup/*.py')):
            z.write(f, f.replace(os.sep, '/'))
        for f in ROOT_FILES:
            z.write(f, f)
        for f in SEQ_FILES:
            z.write(os.path.join(seq_dir, f), 'data/seq/' + f)
        for f in PATCH_FILES:
            z.write(os.path.join(patch_dir, f), 'data/patch/' + f)
        for f in TEST_FILES:
            z.write(os.path.join(tests_dir, f), 'tests/' + f)
        z.writestr('BUNDLE.json', json.dumps(info, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seq-dir', default='move-seq-state-v180000')
    ap.add_argument('--patch-dir', default='move-seq-patch-v180000')
    ap.add_argument('--tests-dir', default='tests')
    ap.add_argument('--out-dir', default=os.path.join('player-trained-models', 'seq-bundle'))
    ap.add_argument('--folder', default='seqt4', help='免費帳號 MyDrive 下的資料夾名')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    info = git_info()
    write_zip(os.path.join(args.out_dir, 'seq_bundle.zip'), args.seq_dir, args.patch_dir, args.tests_dir, info)
    with open(os.path.join(args.out_dir, 'seq_t4.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(notebook(cells(args.folder)), f, ensure_ascii=False, indent=1)
    size = os.path.getsize(os.path.join(args.out_dir, 'seq_bundle.zip')) / 1024 ** 2
    print('寫出 %s（%.0f MB）與 seq_t4.ipynb；git %s，未 commit 檔 %d 個'
          % (os.path.join(args.out_dir, 'seq_bundle.zip'), size, info['head'], len(info['dirty_files'])))


if __name__ == '__main__':
    main()
