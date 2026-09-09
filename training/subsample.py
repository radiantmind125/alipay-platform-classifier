r"""从大 CSV 里随机抽一份小的, 保留表头。

为什么必须随机抽而不是取前 N 行
--------------------------------
dot_scan.py 不带 --limit 时是按 `rglob` 的目录序扫的(第 557 行), **不打乱**。
文件名形如 s3_voucher_GWCZ<递增ID>_<时间戳>, 目录序基本等于时间序,
所以取前 N 行 = 只拿到最早那一批图。

这个偏差是实测过的, 不是理论担心:
    2026-07  商家账单页占 0.24%, 负号报出里只有 3% 是商家页
    2026-08  商家账单页占 0.52%, 负号报出里 67% 是商家页
拿七月那一段当全量, 会把"这个页型很少见"这个结论坐实, 而它是错的。

内存
----
逐行流式处理, 不把整个文件读进来。180 MB / 77 万行的文件也只占几 MB 内存。

用法
----
  python training/subsample.py probe\inv1.csv probe\inv1_small.csv --n 250000
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="从大 CSV 随机抽样, 保留表头")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--n", type=int, default=250000, help="想要多少行")
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    # 先数一遍总行数, 才能算保留概率。只读不解析, 很快。
    with io.open(args.src, encoding="utf-8-sig", errors="replace") as f:
        total = sum(1 for _ in f) - 1
    if total <= 0:
        print("源文件是空的"); return
    print(f"源文件 {total:,} 行")

    if args.n >= total:
        print("要的行数不小于总行数, 直接整份复制就行, 不用抽样")
        return

    keep = args.n / total
    rnd = random.Random(args.seed)
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with io.open(args.src, encoding="utf-8-sig", errors="replace") as fin, \
         io.open(args.dst, "w", encoding="utf-8-sig", newline="") as fout:
        header = fin.readline()
        fout.write(header)
        for line in fin:
            # 逐行独立以 keep 的概率保留 —— 等价于均匀随机抽样,
            # 而且不需要把全部行读进内存
            if rnd.random() < keep:
                fout.write(line)
                written += 1

    mb = args.dst.stat().st_size / 1024 / 1024
    print(f"抽出 {written:,} 行 -> {args.dst}   {mb:.1f} MB")
    print("★ 是全语料的随机横截面, 不是前面一段, 所以各月份、各机型的比例和全量一致")


if __name__ == "__main__":
    main()
