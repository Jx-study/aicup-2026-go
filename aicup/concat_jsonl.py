"""把多份 jsonl 依序串接成一份，供只讀單一 jsonl 的程式（例如 predict_pool）使用。

各輸入的局 key 不能重複；每份輸入最後一行沒有換行時補上。輸出已存在時不覆寫。

用法（在 repo 根目錄）：
    python -m aicup.concat_jsonl --out OUT.jsonl IN1.jsonl IN2.jsonl
"""
import argparse
import json
import os


def concat(paths, out):
    """依序把 paths 寫到 out，回傳局數；key 重複時丟 ValueError，out 不留檔。"""
    seen = set()
    n = 0
    tmp = out + '.part'
    try:
        with open(tmp, 'wb') as fo:
            for p in paths:
                with open(p, 'rb') as fi:
                    for line in fi:
                        if not line.strip():
                            continue
                        key = json.loads(line)['key']
                        if key in seen:
                            raise ValueError('%s 的 %s 已在前面的檔案出現' % (p, key))
                        seen.add(key)
                        fo.write(line if line.endswith(b'\n') else line + b'\n')
                        n += 1
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit('%s 已存在，不覆寫' % args.out)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    print('%d 局 -> %s' % (concat(args.inputs, args.out), args.out))


if __name__ == '__main__':
    main()
