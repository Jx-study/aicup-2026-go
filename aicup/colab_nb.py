"""colab_*.ipynb 共用的函數：把 Drive 目錄接到 repo、檢查需要的檔案、執行指令並寫 log、檢查接續訓練的 commit。

notebook 的「取得程式」格 clone repo 後以 from aicup.colab_nb import check_commit, link_drive, need, run 載入。
"""
import os
import subprocess
import sys

DRIVE_ROOT = '/content/drive/MyDrive/aicup-2026'


def link_drive(names, drive_root=DRIVE_ROOT, repo='.'):
    """把 repo/<name> 連到 <drive_root>/<name>，Drive 上沒有的目錄先建立；repo 內已有實體目錄時丟 RuntimeError。"""
    for name in names:
        target = os.path.join(drive_root, name)
        path = os.path.join(repo, name)
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.exists(path):
            raise RuntimeError('%s 是實體目錄，確認後刪除再重跑' % path)
        os.makedirs(target, exist_ok=True)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.symlink(target, path)
        print('%-36s -> %s' % (name, target))


def missing(items):
    """items 是 (路徑, 產生方式) 的 list，回傳不存在的項目，順序同 items。"""
    return [(path, how) for path, how in items if not os.path.exists(path)]


def need(items):
    """檢查 items 的路徑都存在；缺少時逐項印出路徑與產生方式，再丟 SystemExit。"""
    lack = missing(items)
    for path, how in lack:
        print('缺少 %s\n    產生方式：%s' % (path, how))
    if lack:
        raise SystemExit('缺少 %d 個檔案，見上方清單' % len(lack))
    print('需要的 %d 個檔案都在' % len(items))


def run(cmd, log=None):
    """執行 cmd（argv list），逐行印出輸出並附加到 log；結束碼不為 0 時丟 SystemExit。

    cmd 開頭是 'python' 時換成目前的直譯器，並加 -u 讓輸出不緩衝。
    """
    if cmd[0] == 'python':
        cmd = [sys.executable, '-u', *cmd[1:]]
    print('$ ' + ' '.join(cmd), flush=True)
    if log:
        os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
    f = open(log, 'a', encoding='utf-8') if log else None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout:
            print(line, end='', flush=True)
            if f:
                f.write(line)
                f.flush()
        code = p.wait()
    except BaseException:
        p.kill()  # 中斷 cell 時一併停掉子程序，不讓它繼續占用 GPU
        p.wait()
        raise
    finally:
        if f:
            f.close()
    if code:
        raise SystemExit('指令失敗，結束碼 %d' % code)


def check_commit(run_dir, head):
    """run_dir 已有 COMMIT 檔且不是 head 時丟 SystemExit，否則寫入 head；接續訓練要用同一版程式。"""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, 'COMMIT')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            old = fh.read().strip()
        if old != head:
            raise SystemExit('%s 已用 %s 訓練，這次是 %s；COMMIT 要與接續前相同，或換一個目錄' % (run_dir, old, head))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(head + '\n')
