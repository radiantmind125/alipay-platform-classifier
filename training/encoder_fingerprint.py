r"""**编码器指纹** —— 这张图是被什么编码出来的。只读文件头, 不解码像素, 不用模型。

一句话
------
**手机截图是手机编码的; 生成服务的图是它自己的管线编码的。两者的编码器痕迹几乎不重叠。**

指纹是什么
----------
- **JPEG** -> **量化表**(Quantization Table)。不同编码器/不同质量档的表不一样,
  这是图像取证里的经典手法。下载器**保留原始字节**, 所以这张表反映的是**最初是谁编码的**。
- **PNG** -> **chunk 序列**(IHDR 之后到 IDAT 之前出现了哪些块) + IHDR 的位深/颜色类型。

2026-08-31 实测(本机 12.4 万张真实进件, 已查实假图 81 张)
---------------------------------------------------------
**JPEG 量化表**(假图里 33 张是 JPEG):

| 签名 | 假图 | 真图 |
|---|---|---|
| `20c5a00b7b` | **32/33 = 97.0%** | **1/2700 = 0.04%** |
| `2a8b9baf52` | 0 | **2591/2700 = 95.96%** |

**PNG chunk 序列**(假图里 48 张是 PNG):

| 序列 | 假图 | 真图 |
|---|---|---|
| `IHDR,iTXt,IDAT` | **32 (66.7%)** | **0 (0.00%)** |
| `IHDR,eXIf,iTXt,IDAT` | **8 (16.7%)** | **0** |
| `IHDR,tEXt,IDAT` | **4 (8.3%)** | **0** |
| `IHDR,sRGB,eXIf,pHYs,iTXt,IDAT` | 1 | **2162 (65.6%)** |

-> **真手机截图带 `sRGB` 和 `pHYs`(色彩空间 / 像素密度)块, 生成的 PNG 没有。**

合起来做成"白名单之外就报"的判据, **留出验证**的结果:

| `--top` | 误报(池A) | 误报(池B) | 假图召回 |
|---|---|---|---|
| **25(默认)** | **0.29%** | **0.48%** | **95.1%** |
| 30 | 0.18% | 0.33% | 55.6% |
| 40 | 0.10% | 0.18% | 16.0% |

**用剔干净的白名单(建表时排除了 81 张已查实假图)最终实测**:

| | 报出率 | 召回 |
|---|---|---|
| **同人群**(建表池, 已剔假图) | **0.180% = 18/万** | **95.1%(77/81)** |
| 跨人群(另一个池) | 1.715% = 172/万 | — |

★ **两个池子各自建表, 结论复现**(每个池都用**自己**的白名单, 且建表时剔掉了自己那批已查实假图):

| 池子 | 张数 | 误报 | 召回 |
|---|---|---|---|
| 池A | 124,151 | **0.180%(18/万)** | **95.1%(77/81)** |
| 池B | 138,614 | **0.315%(32/万)** | **96.9%(93/96)** |
| 合计 | 262,765 | 约 0.25% | **96.0%(170/177)** |

**两个独立人群、各自标定, 结果一致 —— 这不是单个池子上的巧合。**

**同人群工作点: 报出 18~32/万, 召回 95~97%, 提升约 300~530 倍。**
对比"分辨率非原生"那条最好的(2.09% / 96.3% / 46 倍) -> **误报低 11 倍, 召回持平。**
★ 跨人群还是差 9.5 倍 —— **这就是下面坑 1 说的, 白名单必须在目标流量上建。**

为什么是白名单而不是黑名单
--------------------------
**白名单列的是"真机编码器"**, 所以**换一家生成服务照样报得出来** ——
不需要事先见过那家的指纹。黑名单做不到这一点。

★★ 两个会把这条判据毁掉的坑(都实测踩过)
----------------------------------------
**坑 1: 白名单不能跨人群搬。**
拿 A 池建的 top20 去筛 B 池, 误报从 **0.28% 涨到 2.455%(9 倍)** ——
B 池有 A 池没见过的**合法**指纹。**必须在目标流量上建表, 并定期重建。**

**坑 2(更隐蔽): 建表数据里混了假图, 白名单会把攻击者的指纹一起收进去。**
实测 `--top` 从 25 调到 30, **召回从 95.1% 崩到 55.6%**, 正好少掉 **32 张** ——
就是 `PNG:IHDR,iTXt,IDAT` 那一组。因为建表样本没剔假图,
2 万张里按 6.5/万 约 13 张假图, **它们的共同指纹足以挤进 top30**。
-> 本脚本建表时**强制要求给 `--exclude`(已知假图名单), 或显式加 `--dirty-ok` 承认风险**;
   并且会**把每个入选指纹的占比打出来** —— 占比异常低的那几个要人工看一眼。

边界
----
★★ **对"手工改图"几乎全瞎 —— 这条只管"整图生成"。**
  实测 14 张负号几何离群(手工拉长负号)里, 这条**只报出 1 张(7%)**,
  而"分辨率非原生"报出 **9 张(64%)**。
  原因: 手工改图是**在真截图上编辑再存**, 编码器变成某个 app 的(仍属"正常"编码器),
  **只有裁剪缩放把分辨率打乱了**。**所以两条判据是分工, 不是互为冗余。**

  | 攻击类型 | 该用哪条 | 召回 |
  |---|---|---|
  | 整图 AI 生成 | **本脚本** | **96.0%** |
  | 手工改金额/负号 | **`resolution_check`** | **64%** |

★★ **转发来的图测不出来。** 经理给的那两张样本指纹**完全相同**(都被聊天软件重编码过),
  而且都落在白名单里。**评估这条判据时不能用转发来的样本**, 只能用直接进平台的原件。

- **不能单独拒。** 报出的绝大多数仍是真图(按 6.7/万 的到达率, 精度约两成)。
- **重新截屏就失效**: 骗子把生成图在手机上再截一次, 编码器就变成真机的了 ——
  **和元数据、分辨率两条会同时失效**, 三条不是独立的。
- 只覆盖 JPEG / PNG。

用法
----
  # 建表(**务必剔掉已知假图**)
  python training/encoder_fingerprint.py --build D:\download2\OtherImages ^
      --exclude D:\probe\known_fakes.txt --top 25 --wl D:\probe\enc_wl.json
  # 筛
  python training/encoder_fingerprint.py --input D:\probe\today --wl D:\probe\enc_wl.json ^
      --out D:\probe\enc_odd.txt
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import struct
import sys
from pathlib import Path

from PIL import Image

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def fingerprint(p: str) -> str | None:
    """返回这张图的编码器指纹。**只读文件头, 不解码像素。**"""
    try:
        with open(p, "rb") as f:
            head = f.read(4096)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            chunks, i = [], 8
            while i + 8 <= len(head) and len(chunks) < 9:
                ln = struct.unpack(">I", head[i:i + 4])[0]
                typ = head[i + 4:i + 8].decode("latin1", "replace")
                chunks.append(typ)
                if typ == "IDAT":
                    break
                i += 12 + ln
                if ln > len(head):
                    break
            return "PNG:" + ",".join(chunks)
        if head[:2] == b"\xff\xd8":
            with Image.open(p) as im:
                q = getattr(im, "quantization", None)
                if not q:
                    return "JPEG:noq"
                sig = repr(sorted((k, tuple(v)) for k, v in q.items())).encode()
                return "JPEG:" + hashlib.md5(sig).hexdigest()[:10]
        return "OTHER"
    except Exception:
        return None


def _walk(root: Path):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                yield os.path.join(dp, fn)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="按编码器指纹筛图(只读文件头)")
    ap.add_argument("--build", type=Path, default=None, help="从这个目录建白名单")
    ap.add_argument("--input", type=Path, default=None, help="要筛的目录")
    ap.add_argument("--wl", type=Path, required=True)
    ap.add_argument("--top", type=int, default=25,
                    help="取最常见的前几种当白名单。**默认 25**, 调大会掉召回, 见 docstring")
    ap.add_argument("--sample", type=int, default=40000)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="**已知假图名单**(一行一个文件名)。建表时强烈建议给")
    ap.add_argument("--dirty-ok", action="store_true",
                    help="没有已知假图名单也要建表 —— **明确承认白名单可能吃进攻击者指纹**")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", type=Path, default=None,
                    help="★ 建完表**必须跑一次**: 拿一份已知假图名单量召回, 看白名单是不是被污染了")
    ap.add_argument("--pool", type=Path, default=None, help="--verify 时到哪个目录找那些图")
    ap.add_argument("--seed", type=int, default=20260831)
    args = ap.parse_args()

    # ---- 自检: 白名单有没有把攻击者的指纹收进去 ----
    # ★ 为什么必须单独验: 建表时给的 --exclude 名单**很可能只覆盖了池子的一小段**
    #   (比如 inspect_meta 用 --limit 扫的是遍历顺序的前 N 张 = 基本只有某一天),
    #   而建表是**全池随机抽样**。两者人群不同 -> 名单外的假图照样进了建表数据。
    #   本机实测: 这种污染会让召回**从 95.1% 陡降到 55.6%**, 是断崖不是渐变, 一测就看得出来。
    if args.verify:
        if not args.wl.is_file():
            raise SystemExit(f"!! 白名单不存在: {args.wl}")
        wl = set(json.loads(args.wl.read_text(encoding="utf-8"))["fps"])
        names = [os.path.basename(x.strip()) for x in
                 args.verify.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        root = args.pool or args.input or args.build
        if not root:
            raise SystemExit("!! --verify 还要给 --pool(到哪找这些图)")
        idx = {os.path.basename(p): p for p in _walk(Path(root))}
        hit = miss = gone = 0
        missed_fps: collections.Counter = collections.Counter()
        for n in names:
            p = idx.get(n)
            if not p:
                gone += 1
                continue
            f = fingerprint(p)
            if f is None:
                gone += 1
            elif f not in wl:
                hit += 1
            else:
                miss += 1
                missed_fps[f] += 1
        n_ok = hit + miss
        if not n_ok:
            raise SystemExit("!! 一张都没找到, 检查 --pool 路径")
        rec = 100.0 * hit / n_ok
        print(f"自检: 已知假图 {len(names)} 张, 找到 {n_ok} 张(缺 {gone})")
        print(f"  **召回 {hit}/{n_ok} = {rec:.1f}%**")
        if rec >= 90:
            print("  -> **白名单干净**, 可以用。")
        else:
            print(f"  -> ★★ **白名单被污染了** —— 有 {miss} 张假图的指纹落在白名单里面:")
            for k, v in missed_fps.most_common(6):
                print(f"       漏掉 {v:>4} 张   {k[:56]}")
            print("     处理: 把上面这些指纹从白名单里删掉, 或者把 --top 调小(比如 20)重建,")
            print("     然后**再验一次**。别拿被污染的白名单去筛 —— 它会报出奇少, 看着像流量很干净。")
        return

    if args.build:
        bad: set[str] = set()
        for p in args.exclude:
            if p.is_file():
                bad |= {os.path.basename(x.strip()) for x in
                        p.read_text(encoding="utf-8-sig").splitlines() if x.strip()}
        if not bad and not args.dirty_ok:
            raise SystemExit(
                "!! 建表没给 --exclude(已知假图名单)。\n"
                "   实测教训: 建表数据里混了假图, **白名单会把攻击者的指纹一起收进去** ——\n"
                "   top 从 25 调到 30, 召回就从 95.1% 崩到 55.6%(正好少掉一整组假图)。\n"
                "   先跑 inspect_meta / watermark_scan 出一份已知假图名单再来; \n"
                "   确实没有就加 --dirty-ok 明确承认这个风险。")
        print(f"已知假图名单 {len(bad)} 个(建表时剔除)")

        fs = [p for p in _walk(args.build) if os.path.basename(p) not in bad]
        print(f"候选 {len(fs):,} 张", flush=True)
        random.Random(args.seed).shuffle(fs)
        fs = fs[: args.sample]
        c: collections.Counter = collections.Counter()
        for i, p in enumerate(fs, 1):
            f = fingerprint(p)
            if f:
                c[f] += 1
            if i % 10000 == 0:
                print(f"  {i:,}/{len(fs):,}", flush=True)
        tot = sum(c.values())
        wl = [k for k, _ in c.most_common(args.top)]
        cov = sum(v for _, v in c.most_common(args.top))
        print(f"\n看了 {tot:,} 张, 共 **{len(c)}** 种指纹")
        print(f"前 {args.top} 种覆盖 **{100.0*cov/tot:.3f}%** -> 误报 **{100.0*(tot-cov)/tot:.3f}%**")
        print("\n入选白名单的指纹(★ **占比很低的那几个要人工看一眼, 可能是混进来的假图**):")
        for k, v in c.most_common(args.top):
            flag = "  <- ★占比很低, 存疑" if v / tot < 0.001 else ""
            print(f"   {100.0*v/tot:7.3f}%  {k[:56]}{flag}")
        args.wl.parent.mkdir(parents=True, exist_ok=True)
        args.wl.write_text(json.dumps(
            {"top": args.top, "coverage": cov / tot, "n_build": tot,
             "excluded_known_fakes": len(bad), "fps": wl}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n白名单 -> {args.wl}")
        if not args.input:
            return

    if not args.input:
        raise SystemExit("!! 要么给 --build, 要么给 --input")
    meta = json.loads(args.wl.read_text(encoding="utf-8"))
    wl = set(meta["fps"])
    if not meta.get("excluded_known_fakes"):
        print("!!!! 这份白名单建表时**没有剔除已知假图** —— 召回可能被悄悄拉低, 见 docstring 坑 2")
    print(f"白名单 {len(wl)} 种 (建表覆盖 {100.0*meta.get('coverage',0):.3f}%, "
          f"建表样本 {meta.get('n_build',0):,} 张)")

    odd, n = [], 0
    seen: collections.Counter = collections.Counter()
    for p in _walk(args.input):
        f = fingerprint(p)
        if not f:
            continue
        n += 1
        if f not in wl:
            odd.append((os.path.basename(p), f))
            seen[f] += 1
    print(f"\n看了 {n:,} 张")
    print(f"**白名单之外: {len(odd):,} 张 = {100.0*len(odd)/max(n,1):.3f}% "
          f"= {10000.0*len(odd)/max(n,1):.0f}/万**")
    print("\n报出来的指纹分布:")
    for k, v in seen.most_common(12):
        print(f"   {v:>5}  {k[:58]}")

    if args.out and odd:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(f"{a}\t{b}" for a, b in odd), encoding="utf-8")
        print(f"\n名单 -> {args.out}")

    print("\n★ 不能单独拒 —— 按 6.5/万 的假图到达率, 报出来的里面精度约两成。")
    print("★ 也别把它和元数据、分辨率当成三条独立证据: **重新截屏一次, 三条一起失效。**")


if __name__ == "__main__":
    main()
