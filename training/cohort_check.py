r"""**同群比对**: 报出来的图, 跟**同机型同版本**的真图比, 还离不离群?

要解决的问题
------------
2026-09-01 在八月的图上跑负号判据, 报出率 **14.2/万 -> 33.5/万**(2.4 倍)。
但**不能直接报成检出**, 因为有三条证据指向"新版本把负号画长了"而不是欺诈:

1. **八月整个分布都上移了**: 5 位金额中位 **0.7021 -> 0.7143**, p99.5 **0.7381 -> 0.7500**
2. **报出的图在日期上是散开的**(0803~0810), 而欺诈批次会在时间上聚集
   (那个 149 张的重复组是 72 分钟内跑完的)
3. 报出的金额**压倒性是整数**(100/200/300/1000/2000), 那正是真实转账的样子

**拿七月标的阈值去判八月的图, 分不清"负号被拉长了"和"新版本把负号画长了"。**

试过但**失败**的办法
-------------------
`minus_vs_date.py` 想在**同一张图内部**比(金额负号 vs 日期连字符)。
**不成立**: 日期连字符只有 3 px 高, 二值化差一个像素宽度就差约 10%,
1,061 张真图上"两者之比"的 p5~p95 跨 **0.61~2.09**, 根本不是常数。

这个脚本怎么绕开
----------------
**不跟"所有真图"比, 只跟"同一群"比。**

一群 = **同分辨率 + 同编码器指纹**。同分辨率基本锁定机型, 同编码器指纹基本锁定
系统/app 的出图管线 —— 两个一起, 就把"机型不同"和"版本不同"这两个混淆变量都控住了。

  · 同群里绝大多数在 0.71, 少数在 1.10  -> **那少数是异常** -> 篡改
  · 整群都在 1.10                        -> **这一版就这么画** -> 渲染变体

★ 这条不需要第二个字形(避开了 3px 连字符那个坑),
  也不需要跨时间比(避开了"七月标的阈值判八月的图"那个坑)。

★ 边界: 同群里**如果假图占了多数**, 这条会把假的当成"正常"。
  按已知到达率(AI 假图 6.7/万、负号离群 14~34/万)这不太可能, 但**群太小时要当心**,
  所以下面会强制要求每群至少有 `--min-cohort` 张对照图, 不够就报"样本不足"而不是硬给结论。

用法
----
  python training/cohort_check.py --flagged D:\probe\minus_aug.csv ^
      --pool D:\download2\OtherImages --since 20260801 --out D:\probe\cohort.csv

**只读**: 只读图片, 只往 --out 写。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from encoder_fingerprint import fingerprint    # noqa: E402
from minus_outlier import measure as measure_minus  # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{8})\d{6}")


def _cohort_key(p: str):
    """群 = (分辨率, 编码器指纹)。取不到就返回 None。"""
    try:
        with Image.open(p) as im:
            sz = im.size
    except Exception:
        return None
    fp = fingerprint(p)
    if not fp:
        return None
    return (sz, fp)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="同群比对: 报出的图在同机型同版本里还离不离群")
    ap.add_argument("--flagged", type=Path, required=True,
                    help="minus_outlier 出的 csv(要有 image_name 和 bar_width 两列)")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--since", type=str, default=None, help="对照图也限定在这个日期之后")
    ap.add_argument("--min-cohort", type=int, default=25,
                    help="每群至少要有这么多张对照图才下结论, 不够就报'样本不足'")
    ap.add_argument("--max-cohort", type=int, default=60, help="每群最多量这么多张对照图")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.flagged.read_text(encoding="utf-8-sig").splitlines()))
    if not rows or "bar_width" not in rows[0]:
        raise SystemExit("!! --flagged 要是 minus_outlier 出的 csv(需要 image_name 和 bar_width)")
    print(f"报出名单 {len(rows):,} 张  <- {args.flagged}")

    print(f"清点 {args.pool} ...", flush=True)
    idx = {}
    for dp, _, fns in os.walk(args.pool):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                idx[fn] = os.path.join(dp, fn)
    print(f"池里 {len(idx):,} 张", flush=True)

    flagged_names = {r["image_name"] for r in rows}

    # ---- 1. 报出图的群 ----
    print("\n算报出图属于哪一群 ...", flush=True)
    fl = []
    for r in rows:
        p = idx.get(r["image_name"])
        if not p:
            continue
        k = _cohort_key(p)
        if k:
            fl.append((r["image_name"], float(r["bar_width"]), k))
    by_cohort = collections.defaultdict(list)
    for nm, bw, k in fl:
        by_cohort[k].append((nm, bw))
    print(f"  {len(fl):,} 张落在 **{len(by_cohort)}** 个群里")

    # ---- 2. 每群找对照图, 量它们的 bar_width ----
    print("\n给每群找同群对照图并量 ...", flush=True)
    pool_names = [n for n in idx if n not in flagged_names]
    if args.since:
        pool_names = [n for n in pool_names
                      if (_TS.search(n) and _TS.search(n).group(1) >= args.since)]
    random.Random(args.seed).shuffle(pool_names)

    # ★ 先把整池按分辨率建一次索引, **不要每个群各扫一遍池子**。
    #   原来的写法是每群从头扫 pool_names —— 87 万张池子 x 几十个群 = 几千万次开图,
    #   本机 7 个群就跑了 2 分钟, 上服务器会变成几小时。
    #   现在只读一次文件头建索引(约 0.3 ms/张), 之后每群查表即可。
    want_sizes = {k[0] for k in by_cohort}
    print(f"给整池按分辨率建索引(只认这 {len(want_sizes)} 种尺寸) ...", flush=True)
    by_size = collections.defaultdict(list)
    t_idx = time.time()
    for i, n in enumerate(pool_names, 1):
        try:
            with Image.open(idx[n]) as im:
                s = im.size
        except Exception:
            continue
        if s in want_sizes:
            by_size[s].append(n)
        if i % 100000 == 0:
            print(f"    {i:,}/{len(pool_names):,}  用时 {(time.time()-t_idx)/60:.1f} 分", flush=True)
    print(f"  索引好了, 用时 {(time.time()-t_idx)/60:.1f} 分; "
          f"命中的尺寸各有 {sorted((len(v) for v in by_size.values()), reverse=True)[:6]} … 张", flush=True)

    out_rows = []
    t0 = time.time()
    for ci, (key, members) in enumerate(
            sorted(by_cohort.items(), key=lambda kv: -len(kv[1])), 1):
        sz, fp = key
        # 同尺寸的候选已经在索引里了; 再逐张确认指纹(便宜), 最后量 bar_width(最贵)
        ctrl = []
        for n in by_size.get(sz, []):
            if len(ctrl) >= args.max_cohort:
                break
            p = idx[n]
            if fingerprint(p) != fp:
                continue
            m = measure_minus(Path(p))
            if m and m["bar_width"]:
                ctrl.append(float(m["bar_width"]))
        n_ctrl = len(ctrl)
        if n_ctrl >= args.min_cohort:
            arr = np.array(ctrl)
            med, p95, mx = float(np.median(arr)), float(np.percentile(arr, 95)), float(arr.max())
        else:
            med = p95 = mx = float("nan")
        for nm, bw in members:
            if n_ctrl < args.min_cohort:
                verdict = f"样本不足(同群只找到 {n_ctrl} 张)"
            elif bw > mx:
                verdict = "★★ 同群里最离群(超过同群所有对照图)"
            elif bw > p95:
                verdict = "★ 高于同群 p95"
            else:
                verdict = "同群内正常 -> 多半是渲染变体"
            out_rows.append({
                "image_name": nm, "bar_width": round(bw, 4),
                "分辨率": f"{sz[0]}x{sz[1]}", "指纹": fp,
                "同群对照数": n_ctrl,
                "同群中位": round(med, 4) if n_ctrl >= args.min_cohort else "",
                "同群p95": round(p95, 4) if n_ctrl >= args.min_cohort else "",
                "同群最大": round(mx, 4) if n_ctrl >= args.min_cohort else "",
                "判读": verdict,
            })
        print(f"  群 {ci}/{len(by_cohort)}  {sz[0]}x{sz[1]} {fp[:26]}  "
              f"报出 {len(members)} 张, 对照 {n_ctrl} 张  用时 {(time.time()-t0)/60:.1f} 分", flush=True)

    tally = collections.Counter(r["判读"].split("(")[0] for r in out_rows)
    print(f"\n{'='*66}")
    for k, v in tally.most_common():
        print(f"  {k:<44} {v:>5} 张")
    print(f"{'='*66}")
    real = sum(v for k, v in tally.items() if k.startswith("★"))
    n_normal = tally.get("同群内正常 -> 多半是渲染变体", 0)
    if real or n_normal:
        print(f"\n★ **同群内仍然离群的 {real} 张** = 真正可疑(机型和版本已经控住了)")
        print(f"  **同群内正常的 {n_normal} 张** = 那一版就这么画, **不是篡改**")
    print("\n★ 边界: 同群里假图若占多数, 这条会把假的当成正常。群太小时结论不可信,")
    print(f"  所以对照不足 {args.min_cohort} 张的群直接报'样本不足', 不硬给结论。")

    if args.out and out_rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
