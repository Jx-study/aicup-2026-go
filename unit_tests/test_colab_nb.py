import os
import subprocess
import sys

import pytest

from aicup import colab_nb
from aicup.colab_nb import check_commit, link_drive, missing, need, run


def test_missing_and_need(tmp_path, capsys):
    a = tmp_path / 'a.txt'
    a.write_text('x')
    items = [(str(a), '已有'), (str(tmp_path / 'b.txt'), '用 X 產生')]
    assert missing(items) == [items[1]]
    with pytest.raises(SystemExit):
        need(items)
    out = capsys.readouterr().out
    assert 'b.txt' in out and '用 X 產生' in out
    need(items[:1])


def test_link_drive_calls(tmp_path, monkeypatch):
    # 用假的 symlink 記錄呼叫，Windows 沒有建立 symlink 的權限也能測
    calls = []
    monkeypatch.setattr(colab_nb.os, 'symlink', lambda src, dst: calls.append((src, dst)))
    drive, repo = tmp_path / 'drive', tmp_path / 'repo'
    repo.mkdir()
    link_drive(['tests', 'humansl-analysis/hist-x'], str(drive), str(repo))
    assert calls == [(os.path.join(str(drive), 'tests'), os.path.join(str(repo), 'tests')),
                     (os.path.join(str(drive), 'humansl-analysis/hist-x'),
                      os.path.join(str(repo), 'humansl-analysis/hist-x'))]
    assert (drive / 'tests').is_dir() and (drive / 'humansl-analysis' / 'hist-x').is_dir()
    assert (repo / 'humansl-analysis').is_dir()
    (repo / 'real').mkdir()
    with pytest.raises(RuntimeError):
        link_drive(['real'], str(drive), str(repo))


@pytest.mark.skipif(os.name == 'nt', reason='Windows 建立 symlink 需要權限')
def test_link_drive_real_symlink(tmp_path):
    drive, repo = tmp_path / 'drive', tmp_path / 'repo'
    repo.mkdir()
    link_drive(['tests'], str(drive), str(repo))
    assert os.path.islink(repo / 'tests')
    link_drive(['tests'], str(drive), str(repo))
    assert os.path.realpath(repo / 'tests') == os.path.realpath(drive / 'tests')


def test_run_writes_log_and_raises(tmp_path):
    log = tmp_path / 'sub' / 'run.log'
    run(['python', '-c', 'print("hello")'], str(log))
    assert log.read_text(encoding='utf-8').strip() == 'hello'
    with pytest.raises(SystemExit):
        run([sys.executable, '-c', 'import sys; sys.exit(3)'])


def test_run_kills_child_on_interrupt(monkeypatch):
    procs = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        procs.append(real_popen(*args, **kwargs))
        return procs[-1]

    def interrupt(*args, **kwargs):
        if kwargs.get('end') == '':  # 只在轉印子程序的輸出時中斷，模擬使用者中斷 cell
            raise KeyboardInterrupt

    monkeypatch.setattr(colab_nb.subprocess, 'Popen', popen)
    monkeypatch.setattr('builtins.print', interrupt)
    script = 'import time\nprint("start", flush=True)\ntime.sleep(60)'
    with pytest.raises(KeyboardInterrupt):
        run([sys.executable, '-c', script])
    assert procs[0].poll() is not None


def test_check_commit(tmp_path):
    d = tmp_path / 'ckpt'
    check_commit(str(d), 'abc1234')
    assert (d / 'COMMIT').read_text(encoding='utf-8').strip() == 'abc1234'
    check_commit(str(d), 'abc1234')
    with pytest.raises(SystemExit):
        check_commit(str(d), 'def5678')
