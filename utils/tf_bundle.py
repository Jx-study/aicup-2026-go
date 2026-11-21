"""產生提交用 TF 的 Colab bundle：tf_final_bundle.zip 與 tf_final.ipynb，放在 humansl-analysis/tf-final-{name}/。

用最終配方的訓練檔加全部 val648 玩家訓練 3 個訓練 seed，寫 final-ts{ts}.pt 與 final-ts{ts}-test.npy。
封存組的預測在封存組確認時另外加 --predict-data、--predict-questions，本 bundle 不含封存組資料。
用法：python utils/tf_bundle.py --name 6d
      python utils/tf_bundle.py --name 2dl --data packed-moves-2dl --pool train_p1_moves.npz train_p2_moves.npz
      python utils/tf_bundle.py --name j20-p34 --joint --epochs 20 --pool2 train_p1_moves.npz train_p2_moves.npz train_p3_moves.npz train_p4_moves.npz
--joint 時 6d 打包資料放 data/、2d 放 data2/，兩邊都帶 val648_moves.npz 與 test_moves.npz。
"""
import argparse
import json
import os
import zipfile

TF_RUN, H = 'aicup/tf_run.py', 'humansl-analysis/'  # tf_run.py 在 zip 內改名為根目錄的 tf_run.py


def cells(name, pool, extra, epochs):
    return [
        ('markdown', '# 提交用 Transformer（配方 %s）\n\n'
                     '1. 把 `tf_final_bundle.zip` 上傳到這個帳號的 `MyDrive/tffinal-%s/`。\n'
                     '2. 執行階段選 T4 GPU，依序執行全部 cell，3 個訓練 seed 依序跑。\n'
                     '3. 輸出寫在 `MyDrive/tffinal-%s/out/`，中斷後重跑同一個 cell 會接續。\n'
                     '4. 跑完把 `out/` 下載回來，放到 `humansl-analysis/tf-final-%s/out/`。' % (name, name, name, name)),
        ('code', "from google.colab import drive\n"
                 "drive.mount('/content/drive')\n"
                 "import os\n"
                 "B = '/content/drive/MyDrive/tffinal-%s'\n"
                 "os.makedirs(B + '/out', exist_ok=True)  # tee 開檔前目錄要先存在\n"
                 "!unzip -oq {B}/tf_final_bundle.zip -d /content/tf\n"
                 "!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
                 "!ls -la /content/tf/data" % name),
        ('code', "%%cd /content/tf\n"
                 "!python tf_run.py --mode final --loss ce --epochs %d --train-seeds 0 1 2 --data data --pool %s%s "
                 "--out {B}/out 2>&1 | tee -a {B}/out/final.log" % (epochs, ' '.join(pool), extra)),
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


def write_zip(path, src, pool, joint, src2, pool2):
    """寫 bundle zip：tf_run.py、data/（6d 或單池）、joint 時再加 data2/。"""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:  # npz 已壓縮
        z.write(TF_RUN, 'tf_run.py')
        for f in pool + ['val648_moves.npz', 'test_moves.npz']:
            z.write(src + f, 'data/' + f)
        if joint:
            for f in pool2 + ['val648_moves.npz', 'test_moves.npz']:
                z.write(src2 + f, 'data2/' + f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', default='packed-moves')  # humansl-analysis/ 下的打包資料夾
    ap.add_argument('--pool', nargs='+', default=['trainpool_moves.npz', 'newdev_moves.npz'])
    ap.add_argument('--opp-aug', action='store_true')
    ap.add_argument('--joint', action='store_true')  # 6d、2d 兩池合訓，--data 是 6d、--data2 是 2d
    ap.add_argument('--data2', default='packed-moves-2d')
    ap.add_argument('--pool2', nargs='+', default=['train_p1_moves.npz', 'train_p2_moves.npz'])
    ap.add_argument('--epochs', type=int, default=30)
    args = ap.parse_args()
    extra = ' --opp-aug' if args.opp_aug else ''
    if args.joint:
        extra += ' --joint --data2 data2 --pool2 ' + ' '.join(args.pool2)
    out, src = H + 'tf-final-%s/' % args.name, H + args.data + '/'
    os.makedirs(out, exist_ok=True)
    write_zip(out + 'tf_final_bundle.zip', src, args.pool, args.joint, H + args.data2 + '/', args.pool2)
    with open(out + 'tf_final.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook(cells(args.name, args.pool, extra, args.epochs)), f, ensure_ascii=False, indent=1)
    print('%s：%.0f MB' % (out + 'tf_final_bundle.zip', os.path.getsize(out + 'tf_final_bundle.zip') / 1e6))


if __name__ == '__main__':
    main()
