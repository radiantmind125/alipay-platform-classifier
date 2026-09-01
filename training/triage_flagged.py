r"""把"编码器指纹报出来的那一堆"按指纹分类 —— 哪些是生成器, 哪些是别的。

为什么需要
----------
`encoder_fingerprint --input` 只给一个总数和指纹分布, **但不告诉你哪些指纹是 AI 生成器**。
2026-09-01 服务器实跑(87.3 万张)报出 **4,904 张 = 56/万**, 指纹前几名是:

```
1056  JPEG:137d848413          <- 编辑后重存(不是生成器)
 483  PNG:IHDR,IDAT            <- 桌面截图(不是生成器)
 379  PNG:IHDR,sRGB,sBIT,IDAT  <- ?
 259  JPEG:f8f3df98e8          <- ?
 157  JPEG:20c5a00b7b          <- **豆包 JPEG**
 152  PNG:IHDR,iTXt,IDAT       <- **豆包 PNG**
```

**"?"那几个是没定性的** —— 可能是没见过的生成器, 也可能只是小众设备/app。
分不清就没法说"报出来的 4,904 张里有多少是真假图", 也就没法给精度。

怎么定性
--------
拿**独立证据**去对: 对每张报出来的图跑一遍 `aigc_metadata()`(AIGC 元数据, 不看模型分数),
然后**按指纹汇总"带铁证的比例"**:

- 某个指纹**几乎全部带 AIGC 铁证** -> **它就是生成器指纹**, 可以直接当判据
- 某个指纹**一张铁证都没有** -> 多半是小众设备或某个 app 的重编码, **不是生成器**
- 中间的 -> 需要人工看几张

★ 这条证据链是**独立**的: 编码器指纹看文件头怎么编码, AIGC 元数据看生成器自己写的标记,
两者互不依赖, 所以拿一个去验另一个**不构成循环论证**。

★ 但要记住**下界性质**: 元数据被洗掉的假图, 这里会显示成"没有铁证",
所以"铁证比例低"**不等于**"不是假图"。

用法
----
  python training/triage_flagged.py --flagged D:\probe\enc_odd.txt ^
      --pool D:\download2\OtherImages --out D:\probe\triage.csv

**只读**: 只读图片文件头, 只往 --out 写。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from watermark_scan import aigc_metadata  # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="把报出来的图按编码器指纹分类, 用 AIGC 元数据定性")
    ap.add_argument("--flagged", type=Path, required=True,
                    help="encoder_fingerprint --out 生成的名单(每行: 文件名 TAB 指纹)")
    ap.add_argument("--pool", type=Path, required=True, help="图在哪个目录")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-n", type=int, default=20,
                    help="少于这么多张的指纹合并成'其它', 免得比例没有意义")
    args = ap.parse_args()

    rows = []
    for ln in args.flagged.read_text(encoding="utf-8-sig").splitlines():
        if not ln.strip():
            continue
        parts = ln.rstrip("\n").split("\t")
        rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    print(f"报出名单 {len(rows):,} 行  <- {args.flagged}", flush=True)

    print(f"清点 {args.pool} ...", flush=True)
    idx = {}
    for dp, _, fns in os.walk(args.pool):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                idx[fn] = os.path.join(dp, fn)
    print(f"池里 {len(idx):,} 张", flush=True)

    per: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "hard": 0, "soft": 0, "marks": collections.Counter()})
    detail = []
    miss = 0
    t0 = time.time()
    for i, (name, fp) in enumerate(rows, 1):
        p = idx.get(name)
        if not p:
            miss += 1
            continue
        hard, soft, _d = aigc_metadata(Path(p))
        d = per[fp]
        d["n"] += 1
        if hard:
            d["hard"] += 1
            for m in hard:
                d["marks"][m] += 1
        elif soft:
            d["soft"] += 1
        detail.append({"image_name": name, "指纹": fp,
                       "铁证": "+".join(sorted(hard)) if hard else "",
                       "软标记": "+".join(sorted(soft)) if soft else ""})
        if i % 2000 == 0:
            el = time.time() - t0
            print(f"  {i:,}/{len(rows):,}  已用 {el/60:.1f} 分  "
                  f"剩约 {(len(rows)-i)/(i/el)/60:.1f} 分", flush=True)

    if miss:
        print(f"!! {miss} 张在池里找不到(可能已被移动/删除)")

    order = sorted(per.items(), key=lambda kv: -kv[1]["n"])
    big = [(k, v) for k, v in order if v["n"] >= args.min_n]
    small = [(k, v) for k, v in order if v["n"] < args.min_n]

    print(f"\n{'='*74}")
    print(f"{'指纹':<40}{'张数':>7}{'带铁证':>8}{'占比':>8}   定性")
    print(f"{'-'*74}")
    tot_n = tot_h = 0
    for k, v in big:
        pct = 100.0 * v["hard"] / v["n"]
        tot_n += v["n"]; tot_h += v["hard"]
        if pct >= 50:
            verdict = "★★ 生成器指纹"
        elif pct >= 10:
            verdict = "★ 混杂, 要人工看"
        elif v["hard"] > 0:
            verdict = "多半不是生成器"
        else:
            verdict = "无铁证(小众设备/app?)"
        print(f"{k[:39]:<40}{v['n']:>7}{v['hard']:>8}{pct:>7.1f}%   {verdict}")
        if v["marks"]:
            print(f"{'':<40}{'':>7}   标记: {dict(v['marks'].most_common(3))}")
    if small:
        sn = sum(v["n"] for _, v in small); sh = sum(v["hard"] for _, v in small)
        tot_n += sn; tot_h += sh
        print(f"{'(其它 %d 种, 每种 <%d 张)' % (len(small), args.min_n):<40}{sn:>7}{sh:>8}"
              f"{100.0*sh/sn if sn else 0:>7.1f}%")
    print(f"{'-'*74}")
    print(f"{'合计':<40}{tot_n:>7}{tot_h:>8}{100.0*tot_h/tot_n if tot_n else 0:>7.1f}%")

    print(f"\n★ **带铁证的 {tot_h} 张是确凿的 AI 假图**(生成器自己写的标记, 不依赖模型分数)。")
    print("★ 但这是**下界** —— 元数据被洗掉的假图在这里会显示成'没有铁证',")
    print("  所以某个指纹'铁证比例低'**不等于**'那些不是假图'。")
    print("★ 定性成'生成器指纹'的, 可以直接当判据用; 定性成'无铁证'的, 先人工看几张再说。")

    if args.out and detail:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
            w.writeheader()
            w.writerows(detail)
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
