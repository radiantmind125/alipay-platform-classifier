r"""**分辨率不是设备原生的** —— 只读文件头, 不解码像素, 不用模型。

一句话
------
**真手机截图只有那 100 来种设备原生分辨率; 生成的图和裁过的图不是。**

2026-08-31 实测(本机 26 万张真实进件)
-------------------------------------
把池子里**出现最多的前 100 种**分辨率当作"设备原生"白名单, 白名单之外的算"罕见":

| 集合 | 落在白名单之外 |
|---|---|
| 随机真图(white 池, n=2000) | **4.3%** |
| 随机真图(other 池, n=2000) | **2.4%** |
| 负号几何离群(n=14) | **64.3%** |
| **元数据铁证的 AI 假图(n=81)** | **96.3%** |
| 重复提交组的代表图(n=1,584) | 7.8% / 7.7% |

**AI 假图的富集约 40 倍**, 而这条检查**只要读文件头**。

为什么会这样(机理已核实)
------------------------
那 81 张假图的分辨率是 `1373x3051` `1344x2912` `1408x3104` `1391x3014`
`1391x3013` `1392x3011` `1392x3010` `1387x3020` … **共 43 种**,
长宽比像手机(约 2.2), 但**每一张的确切尺寸都不一样、也都不等于任何设备原生值**。
-> 生成模型是**按目标长宽比出图**, 出多少是多少; 而真截图是**设备定死的**。
(顺带: **一张正方形都没有, 也没有 1024 边** —— 别按"AI 图都是 1024 见方"去想。)

裁剪/缩放同理会破坏原生尺寸, 所以这条**对手工改图也有效**(负号离群 64.3%)。

★★ 和现有 `screenshot_filter` 的关系: **完全互补, 零重叠**
--------------------------------------------------------
`screenshot_filter` 的尺寸判据是 **短边 > 1500** 或 **长宽比不在 1.6~2.6**。
拿那 81 张铁证 AI 假图实测:

| 判据 | 抓到 |
|---|---|
| `screenshot_filter` 的尺寸两条 | **0 / 81 (0.0%)** |
| **本脚本的精确分辨率白名单** | **78 / 81 (96.3%)** |
| 两者都抓到 | **0** |

原因很具体: 那些假图长宽比 **2.17~2.22**(**落在 1.6~2.6 里面**), 短边 **1344~1408**
(**没到 1500**) —— **每一条现有规则都躲过去了**。**只有"确切尺寸不是任何设备原生值"能认出来。**

-> 这条**不是**替代 `screenshot_filter`, 是**补它漏掉的那一类**。
   (反过来, 翻拍那类 4:3 相机照片 `2448x3264` `3468x4624` 两边都抓得到, 属于重叠部分。)

模型自己报出来的高分图上也验过:
`top_suspect`(线路A 高分) **95.0%** 非原生, `top30`(榜首人工复核) **75.9%** 非原生
-> **前置筛能留住模型现在抓到的绝大部分**, 不只是元数据那批。

★★ 它比"编码器指纹"强的地方: **手工改图**
--------------------------------------
`encoder_fingerprint` 在**整图 AI 生成**上更强(96.0% 对 96.3%, 但误报低 11 倍)。
**可是在手工改图上, 两者掉了个个儿**:

| 攻击类型 | `encoder_fingerprint` | **本脚本** |
|---|---|---|
| 整图 AI 生成(n=177) | **96.0%** | 96.3% |
| **手工改负号(n=14)** | **7%(1/14)** | **64%(9/14)** |

**机理**: 手工改图是在**真截图**上编辑再存 —— 编码器变成某个 app 的(仍属"正常"编码器,
躲得过白名单), **但裁剪/缩放把分辨率打乱了**, 这条就能抓到。
-> **两条不是冗余, 是分工。** 只跑编码器那条, 会把手工改图这一整类漏掉。

★ 必须写明的边界
----------------
- **这不是独立判据**: 真图也有 2~4% 落在白名单外(小众机型、正常裁剪)。
  按 AI 假图 6.5/万 算, 报出的里面绝大多数仍是真图。**只能当便宜的前置筛, 不能直接拒。**
- **和元数据那条不独立**: 骗子把生成图**重新截屏**一次, 元数据没了, 分辨率也变回设备原生 ——
  **两条会同时失效**。所以 96.3% 是在"直接提交生成图"这批上的数, **不能外推到所有假图**。
- 白名单是**从池子自己统计出来的**, 换了流量来源要重新统计。

`--top` 怎么选(实测的取舍曲线)
-----------------------------
在 12.4 万张池子上跑出来的(假图 n=81):

| `--top` | 真图被报出 | 假图召回 | 提升 |
|---|---|---|---|
| 50 | 9.02% | 96.3% | 10.7x |
| **100(默认)** | **2.09%** | **96.3%** | **46x** |
| 150 | 0.75% | 82.7% | 110x |
| 200 | 0.35% | 64.2% | 182x |

- **想当前置筛(要召回) -> `--top 100`**: 只跑 2% 的图, 保住 96% 的假图
- **想直接进人工队列(要精度) -> `--top 200`**: 报出 **35/万**, 召回 64%, 提升 182 倍

★ **只按"宽度"判要差一截**, 试过了: 宽度 top20 是 6.51% 误报 / 95.1% 召回,
  被"完整分辨率 top100"(2.09% / 96.3%)**两头压住**。
  原因: **横向裁剪会把宽度打散**(池子里有 268 种宽度), 而完整分辨率反倒更集中。

★ **天花板是真的**: 81 张已知假图里有 **3 张的分辨率就是设备原生**(两张 `1260x2800`,
  那是池子里 3.33% 的常见尺寸)。**所以这条最高就到 96% 左右**, 不可能做到 100%。

怎么用
------
最有价值的用法是**前置筛**: 线上是 CPU 服务器, 每张图跑模型约 1 秒;
先按分辨率筛掉 98% 再跑模型, **算力省约 47 倍, 而这批已知假图仍有 96% 落在筛出来的那部分里**。

用法
----
  # 先建白名单(从真实流量统计)
  python training/resolution_check.py --build D:\download2\OtherImages --top 100 --wl D:\probe\res_wl.json
  # 再拿白名单去筛
  python training/resolution_check.py --input D:\probe\today --wl D:\probe\res_wl.json --out D:\probe\odd.txt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _walk(root: Path):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                yield os.path.join(dp, fn)


def _size(p: str):
    try:
        with Image.open(p) as im:      # 只读文件头, **不解码像素**
            return im.size
    except Exception:
        return None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="按分辨率是否设备原生做前置筛(只读文件头)")
    ap.add_argument("--build", type=Path, default=None, help="从这个目录统计白名单")
    ap.add_argument("--input", type=Path, default=None, help="要筛的目录")
    ap.add_argument("--wl", type=Path, required=True, help="白名单 json 的读/写路径")
    ap.add_argument("--top", type=int, default=100, help="取最常见的前几种当白名单")
    ap.add_argument("--sample", type=int, default=40000, help="建白名单时最多看多少张")
    ap.add_argument("--out", type=Path, default=None, help="筛出来的文件名写这里")
    ap.add_argument("--seed", type=int, default=20260831)
    args = ap.parse_args()

    if args.build:
        fs = list(_walk(args.build))
        print(f"候选 {len(fs):,} 张", flush=True)
        random.Random(args.seed).shuffle(fs)
        fs = fs[: args.sample]
        c: collections.Counter = collections.Counter()
        for i, p in enumerate(fs, 1):
            s = _size(p)
            if s:
                c[s] += 1
            if i % 10000 == 0:
                print(f"  {i:,}/{len(fs):,}", flush=True)
        tot = sum(c.values())
        wl = [list(k) for k, _ in c.most_common(args.top)]
        cov = sum(v for _, v in c.most_common(args.top))
        print(f"\n看了 {tot:,} 张, 共 {len(c)} 种分辨率")
        print(f"前 {args.top} 种覆盖 **{100.0*cov/tot:.2f}%** -> 白名单")
        print("  最常见的几种(应当都认得出是哪款手机):")
        for k, v in c.most_common(8):
            print(f"    {k[0]}x{k[1]}  {100.0*v/tot:.2f}%")
        args.wl.parent.mkdir(parents=True, exist_ok=True)
        args.wl.write_text(json.dumps({"top": args.top, "coverage": cov / tot,
                                       "sizes": wl}, indent=1), encoding="utf-8")
        print(f"\n白名单 -> {args.wl}")
        if not args.input:
            return

    if not args.input:
        raise SystemExit("!! 要么给 --build, 要么给 --input")
    if not args.wl.is_file():
        raise SystemExit(f"!! 白名单不存在: {args.wl}  (先用 --build 生成)")
    meta = json.loads(args.wl.read_text(encoding="utf-8"))
    wl = {tuple(x) for x in meta["sizes"]}
    print(f"白名单 {len(wl)} 种 (建表时覆盖 {100.0*meta.get('coverage',0):.2f}%)")

    fs = list(_walk(args.input))
    odd, n, bad = [], 0, 0
    for p in fs:
        s = _size(p)
        if not s:
            bad += 1
            continue
        n += 1
        if s not in wl:
            odd.append((os.path.basename(p), s))
    print(f"\n看了 {n:,} 张 (读不出 {bad})")
    print(f"**不在白名单里的: {len(odd):,} 张 = {100.0*len(odd)/max(n,1):.2f}%**")
    c2 = collections.Counter(s for _, s in odd)
    print("\n罕见分辨率里最常出现的:")
    for k, v in c2.most_common(10):
        print(f"    {k[0]}x{k[1]}  {v}")

    if args.out and odd:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(f"{n_}\t{s[0]}x{s[1]}" for n_, s in odd),
                            encoding="utf-8")
        print(f"\n名单 -> {args.out}")

    print("\n★ 这条**不能单独用来拒**: 真图也有 2~4% 落在白名单外(小众机型/正常裁剪)。")
    print("  它的价值是**便宜的前置筛** —— 先筛掉九成五再跑模型, 算力省一大截。")
    print("  ★ 也要记住它和元数据那条**会同时失效**: 生成图重新截屏一次, 两条都没了。")


if __name__ == "__main__":
    main()
