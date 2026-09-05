r"""把两条线同时报出的可疑图挑出来拷到一个目录, 供人工看。

背景
----
2026-09-05 交叉比对 `dot_scan.py` 和 `chevron_scan.py` 两份测量, 发现**第二个造假工具**:

  箭头 **44x44 方形**(真图是 56x33, 高是宽的 1.7 倍, iOS 不会画方的)
  负号宽比 **1.0238**(真图 0.7045)
  小数点面积比 **0.0234**(真图 0.0301)

**三个不同的字形同时不对, 而且是同一批图。** 基准率上富集 395 倍:
方形箭头全体只占 0.222%, 但在"负号异常"那批里占 87.5%;
反过来 58 张方形箭头的图里有金额测量的 7 张, **7/7 负号也异常**(全体才 0.33%)。
**58 张全部在八月, 七月一张都没有。**

这个脚本干的事很简单: 读那两份 CSV, 把命中的文件名找出来, 从图库拷到一个目录。
**拷出来一定要人工打开看** —— 光看数字不够, 之前有过"量的根本不是那个东西"的教训
(有条判据实测是"¥ 探测器", 报出的 8 张全部有 ¥)。

用法
----
  python training/collect_suspects.py --dot D:\probe\dot.csv --chev D:\probe\chev.csv ^
      --src D:\download2\OtherImages --out D:\probe\suspect

只要其中一份 CSV 也能跑, 缺的那条自动跳过。
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 判据常量, 与 dot_scan.py / MinusCheck.cs 对齐
BAR_HIGH = 0.78
MIN_DIGIT_HEIGHT = 60
SQUARE_SIZES = {"44x44", "45x44", "44x45", "45x45"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="把可疑图拷出来供人工看")
    ap.add_argument("--dot", type=Path, default=None, help="dot_scan.py 的 --out CSV")
    ap.add_argument("--chev", type=Path, default=None, help="chevron_scan.py 的 --out CSV")
    ap.add_argument("--src", type=Path, required=True, help="图库根目录")
    ap.add_argument("--out", type=Path, required=True, help="拷到哪里")
    ap.add_argument("--limit", type=int, default=0, help="最多拷这么多张, 0 = 不限")
    args = ap.parse_args()

    want = {}   # 文件名 -> 命中原因

    if args.dot and args.dot.exists():
        n = 0
        with open(args.dot, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                n += 1
                try:
                    if float(r["mh"]) < MIN_DIGIT_HEIGHT:
                        continue
                    if float(r["bar_ratio"]) >= BAR_HIGH:
                        want.setdefault(r["name"], []).append(f"负号{float(r['bar_ratio']):.4f}")
                except (KeyError, ValueError):
                    continue
        print(f"{args.dot.name}: 读 {n:,} 行, 负号异常 {sum(1 for v in want.values() if any('负号' in x for x in v))} 张")

    if args.chev and args.chev.exists():
        n = sq = 0
        with open(args.chev, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                n += 1
                try:
                    size = f"{r['chev_h']}x{r['chev_w']}"
                except KeyError:
                    continue
                if size in SQUARE_SIZES:
                    sq += 1
                    want.setdefault(r["file"], []).append(f"方形箭头{size}")
                elif r.get("flagged") == "1":
                    want.setdefault(r["file"], []).append(f"箭头{size}")
        print(f"{args.chev.name}: 读 {n:,} 行, 方形箭头 {sq} 张")

    if not want:
        print("两份 CSV 都没读到, 或者一张都没命中"); return

    both = [k for k, v in want.items() if len(v) >= 2]
    print(f"\n合计 {len(want)} 张, 其中**两条线都命中**的 {len(both)} 张 <- 这些置信度最高")

    print("\n建索引(图库大的话要等一会)...", flush=True)
    idx = {}
    for p in args.src.rglob("*"):
        if p.suffix.lower() in _EXTS:
            idx.setdefault(p.name, p)
    print(f"图库 {len(idx):,} 个文件")

    args.out.mkdir(parents=True, exist_ok=True)
    names = both + [k for k in want if k not in set(both)]     # 双命中的排前面
    if args.limit:
        names = names[: args.limit]

    got = miss = 0
    for name in names:
        src = idx.get(name)
        if src is None:
            miss += 1
            continue
        tag = "BOTH_" if len(want[name]) >= 2 else ""
        shutil.copy2(src, args.out / (tag + name))
        got += 1

    print(f"\n拷了 {got} 张 -> {args.out}   图库里找不到的 {miss} 张")
    print("★ 两条线都命中的文件名前面加了 BOTH_ 前缀, 先看这些。")
    print("★ 一定要打开看 —— 数字对不代表量的就是那个东西。")

    reasons = Counter(tuple(sorted(v)) for v in want.values())
    print("\n命中原因分布(前 8):")
    for k, c in reasons.most_common(8):
        print(f"   {c:>4} 张  {' + '.join(k)}")


if __name__ == "__main__":
    main()
