"""OCR 跑到哪了 —— 输出**全 ASCII**, 不怕控制台编码。

为什么要这个脚本
----------------
服务器控制台按 GBK 显示, 而日志是 UTF-8 写的, 直接 Get-Content 出来是乱码:

    鈽呪槄 涓嬩竴姝?*蹇呴』**浜哄伐鏍稿嚑鍗佹潯

贴回来根本没法读, 关键数字全糊了。所以这里**一个中文都不打**,
只报数, 怎么贴都不会坏。

用法
----
    python ocr_status.py --pairs D:\\alipay-ai-data\\pinyin-pairs
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

# kind 列里是中文, 这里翻成 ASCII 再打
KIND_EN = {
    "可用": "usable",
    "混着拼音": "pinyin-mixed",
    "纯拉丁(拼音漏进来了)": "all-latin",
    "空": "empty",
    "太短": "too-short",
    "图不在": "missing-file",
    "读不了": "unreadable",
}


def en(kind: str) -> str:
    if kind in KIND_EN:
        return KIND_EN[kind]
    if kind.startswith("出错"):
        return "error" + kind[2:]
    return "".join(c if c.isascii() else "?" for c in kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--tail", type=int, default=6,
                    help="日志末尾打几行(只挑出数字那部分)")
    a = ap.parse_args()

    print("=" * 56)
    print("  OCR STATUS  (ASCII only - safe to paste)")
    print("=" * 56)
    print(f"  dir        {a.pairs}")

    man = a.pairs / "_pairs.csv"
    lab = a.pairs / "_pairs_labeled.csv"
    log = a.pairs / "_ocr.log"

    n_man = 0
    if man.exists():
        with man.open(encoding="utf-8-sig") as f:
            n_man = sum(1 for _ in f) - 1
        print(f"  _pairs.csv        {n_man:>9,} rows   "
              f"{man.stat().st_size/2**20:7.1f} MB")
    else:
        print("  _pairs.csv        MISSING  <-- crop step never finished")

    for d in ("label", "input"):
        p = a.pairs / d
        if p.exists():
            n = sum(1 for _ in p.iterdir())
            flag = "" if n == n_man else "   <-- MISMATCH"
            print(f"  {d+'/':<18}{n:>9,} files{flag}")
        else:
            print(f"  {d+'/':<18}MISSING")

    if not lab.exists():
        print("  _pairs_labeled.csv  MISSING  <-- OCR produced nothing")
    else:
        rows = list(csv.DictReader(lab.open(encoding="utf-8-sig")))
        done = [r for r in rows if (r.get("kind") or "").strip()]
        st = lab.stat()
        print(f"  _pairs_labeled    {len(rows):>9,} rows   "
              f"{st.st_size/2**20:7.1f} MB")
        print(f"  filled (kind set) {len(done):>9,}   "
              f"{len(done)/max(1,n_man):6.1%} of _pairs.csv")
        age = (time.time() - st.st_mtime) / 60
        print(f"  last written      {age:>9.1f} min ago   "
              f"{'(STALE - not running?)' if age > 15 else '(fresh)'}")
        print()
        for k, v in Counter(r.get("kind", "") for r in done).most_common():
            print(f"      {en(k):<22}{v:>9,}  ({v/max(1,len(done)):5.1%})")

    if log.exists():
        print()
        print(f"  _ocr.log          {log.stat().st_size:>9,} bytes")
        txt = log.read_text(encoding="utf-8", errors="replace")
        # 只挑带数字的进度行, 中文全滤掉
        prog = [l for l in txt.splitlines() if "/" in l and any(c.isdigit() for c in l)]
        print(f"  progress lines    {len(prog):>9,}")
        for l in prog[:2]:
            print(f"      first  {''.join(c if c.isascii() else '' for c in l).strip()}")
        for l in prog[-a.tail:]:
            print(f"      last   {''.join(c if c.isascii() else '' for c in l).strip()}")
    else:
        print("  _ocr.log          MISSING")


if __name__ == "__main__":
    main()
