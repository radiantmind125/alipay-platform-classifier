r"""扫 AIGC 水印(豆包/千问/…) —— **不看模型分数**的假图判据。

为什么必须有一条"不看模型分数"的判据
------------------------------------
清训练集的时候有个陷阱: 如果用**模型自己打的高分**去挑要删的图, 那删掉的里面既有真假图,
也有**本来就难的真图**(困难负样本)。**困难负样本正是模型最需要的。**
删了它们 -> 模型对难的真图变得更自信 -> **线下每个指标都变好, 线上误杀反而变差**。
这跟"把转发压缩过的真图当假图剔掉"是同一个坑, 只是更深 —— 它直接烙进权重里, 而不只是一个阈值。

**水印是干净的判据**: 它是生成器自己盖上去的, 跟我们的模型完全无关, 不存在循环论证。
所以清训练集**只该按水印这类独立证据删**, 不该按模型分数删。

已编目的三种(实测位置, 相对坐标)
--------------------------------
  豆包AI生成   右下角  x 0.825-0.985  y 0.972-0.993
  千问AI生成   右下角  x 0.794-0.988  y 0.968-0.984
  AI生成       左上角  (诈骗工具那种, 位置不固定, 用大搜索窗)
**踩过的坑: 只扫右下角会漏掉左上角那种。两个角都要扫。**

实测标定(2026-08-07, 20 张已查实假图 + 3000 张随机真图)
-------------------------------------------------------
阈值定在 **0.30**:
- **带水印的 11 张全部抓到(11/11)**, 而且**正好就是那 11 张 PNG** ——
  与人工查证的结论完全一致(「11 张右下角印着豆包水印, 全是 .png」)。
- 3000 张"真图"里命中 **2 张(0.067%)**, **两张都人工确认确实带豆包水印** -> **零误报**。
  按这个比例推, 12.4 万张池子里大约还有 **80 张**这种漏网的。

**为什么不是更高的阈值**: 0.55 只剩 8/11。**为什么不是更低**: 0.25 以下开始碰到压缩噪声。
**已查实假图里的 .jpeg 全部只有 0.05 左右** —— 不是漏检, 是**它们本来就没水印**
(那几张是靠诈骗站URL/订单号日期矛盾/「不是真实交易」字样查实的)。

★ 诚实的边界: 这条判据只抓**还带着水印**的图。**裁掉或洗掉水印的它一张也抓不到。**
  所以它保证的是"**删对**", 不是"**删干净**"。清训练集正需要前者。

用法
----
  # 先自检(必跑): 在已查实假图上应该大面积命中, 在随机真图上应该几乎不命中
  python training/watermark_scan.py --selftest --fakes D:\probe\top_suspect --genuine D:\probe\genuine_20k

  # 扫训练集的 nature 类, 挑出带水印的(= 混进来的假图)
  python training/watermark_scan.py --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\train\nature \
      --out D:\probe\trainnat_watermarked.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 每个模板: (名字, 参照图, 相对框 x0,y0,x1,y1)
# 参照图要挑**压缩噪声小的 PNG**, 否则模板本身就带噪声, 相关值会被拉低。
_TEMPLATES = [
    ("豆包AI生成", "0.9812_s3_voucher_GWCZ2073347314310844416_20260704180639.png",
     (0.820, 0.970, 0.990, 0.995)),
    ("千问AI生成", None, (0.790, 0.965, 0.992, 0.987)),      # 参照图由 --qwen-ref 给
]

# 相对偏移搜索: 不同生成器画水印的位置会差一点点, 死扣一个框会漏
_OFFSETS = [(dx, dy) for dx in (-0.012, 0.0, 0.012) for dy in (-0.010, 0.0, 0.010)]


def _norm(a: np.ndarray) -> tuple[np.ndarray, float]:
    a = a - a.mean()
    return a, float(np.sqrt((a ** 2).sum()))


def build_template(ref: Path, box: tuple[float, float, float, float]):
    im = Image.open(ref).convert("L")
    w, h = im.size
    px = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
    t = np.asarray(im.crop(px)).astype(float)
    tn, tnorm = _norm(t)
    if tnorm < 1e-6:
        raise SystemExit(f"模板区域是纯色, 取不到水印: {ref}")
    return tn, tnorm, t.shape


def score(path: Path, tpl, box) -> float:
    """返回这张图在该模板下的最高归一化相关(-1..1)。"""
    tn, tnorm, (TH, TW) = tpl
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return float("nan")
    w, h = im.size
    best = -1.0
    for dx, dy in _OFFSETS:
        x0, y0 = int((box[0] + dx) * w), int((box[1] + dy) * h)
        x1, y1 = int((box[2] + dx) * w), int((box[3] + dy) * h)
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 20 or y1 - y0 < 8:
            continue
        c = np.asarray(im.crop((x0, y0, x1, y1)).resize((TW, TH), Image.BILINEAR)).astype(float)
        cn, cnorm = _norm(c)
        if cnorm < 1e-6:
            continue
        best = max(best, float((cn * tn).sum() / (cnorm * tnorm)))
    return best


def _imgs(d: Path, limit: int = 0):
    out = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            out.append(p)
            if limit and len(out) >= limit:
                break
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="扫 AIGC 水印(不看模型分数的假图判据)")
    ap.add_argument("--input", type=Path, default=None, help="要扫的目录")
    ap.add_argument("--out", type=Path, default=None, help="命中的文件名写这里")
    ap.add_argument("--ref-dir", type=Path, default=Path(r"c:\projects\China\top_suspect"),
                    help="放模板参照图的目录(默认本地 top_suspect)")
    ap.add_argument("--qwen-ref", type=Path, default=None, help="千问水印的参照图(可选)")
    ap.add_argument("--thr", type=float, default=0.30,
                    help="判命中的相关阈值。实测标定: 0.30 -> 带水印的 11/11 全抓到, 3000 张真图里 2 张(均确认是假图)")
    ap.add_argument("--selftest", action="store_true", help="在已查实假图和真图上各跑一遍验证阈值")
    ap.add_argument("--fakes", type=Path, default=None, help="自检用: 已查实假图目录")
    ap.add_argument("--genuine", type=Path, default=None, help="自检用: 真图目录")
    ap.add_argument("--limit", type=int, default=3000, help="自检时真图最多取多少张")
    ap.add_argument("--show", type=int, default=30)
    args = ap.parse_args()

    tpls = []
    for name, refname, box in _TEMPLATES:
        ref = args.qwen_ref if (name.startswith("千问")) else (
            args.ref_dir / refname if refname else None)
        if not ref or not Path(ref).exists():
            print(f"(跳过模板 {name}: 没有参照图 {ref})")
            continue
        tpls.append((name, build_template(Path(ref), box), box))
        print(f"模板 {name}: 参照 {Path(ref).name}  尺寸 {tpls[-1][1][2][1]}x{tpls[-1][1][2][0]}")
    if not tpls:
        raise SystemExit("一个模板都没建起来")

    def scan(files):
        rows = []
        for p in files:
            best_n, best_s = "", -1.0
            for name, tpl, box in tpls:
                s = score(p, tpl, box)
                if s == s and s > best_s:
                    best_s, best_n = s, name
            rows.append((best_s, best_n, p))
        return rows

    if args.selftest:
        if not args.fakes or not args.genuine:
            raise SystemExit("--selftest 需要同时给 --fakes 和 --genuine")
        fr = scan(_imgs(args.fakes))
        gr = scan(_imgs(args.genuine, args.limit))
        print(f"\n自检: 已查实假图 {len(fr)} 张, 真图 {len(gr)} 张")
        print(f"{'阈值':>6} {'假图命中率':>12} {'真图命中率(=误报)':>18}")
        for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80):
            a = sum(1 for s, _, _ in fr if s >= t) / max(1, len(fr))
            b = sum(1 for s, _, _ in gr if s >= t) / max(1, len(gr))
            print(f"{t:6.2f} {a*100:11.1f}% {b*100:17.3f}%")
        print("\n怎么读: 假图命中率就是这条判据的**召回**(带水印的才可能被抓, 去了水印的抓不到);")
        print("        真图命中率就是**误报**, 必须接近 0 —— 这条判据是要拿去删训练数据的, 宁可漏不可错。")
        print(f"\n假图逐张(前 {args.show}):")
        for s, n, p in sorted(fr, reverse=True)[:args.show]:
            print(f"  {s:6.3f}  {n:10s} {p.name[:60]}")
        if gr:
            print("\n真图里分数最高的 10 张(**人工看一眼, 它们要么是误报要么本来就是混进来的假图**):")
            for s, n, p in sorted(gr, reverse=True)[:10]:
                print(f"  {s:6.3f}  {n:10s} {p.name[:60]}")
        return

    if not args.input:
        raise SystemExit("要么 --selftest, 要么给 --input")
    rows = scan(_imgs(args.input))
    hit = [r for r in rows if r[0] >= args.thr]
    print(f"\n扫了 {len(rows):,} 张, 相关 >= {args.thr} 的有 **{len(hit)} 张** "
          f"({len(hit)/max(1,len(rows))*100:.3f}%)")
    from collections import Counter
    for n, c in Counter(n for _, n, _ in hit).most_common():
        print(f"  {n}: {c} 张")
    print(f"\n前 {min(args.show, len(hit))} 张:")
    for s, n, p in sorted(hit, reverse=True)[:args.show]:
        print(f"  {s:6.3f}  {n:10s} {p.name[:60]}")
    if args.out:
        args.out.write_text("\n".join(sorted(p.name for _, _, p in hit)) + "\n", encoding="utf-8")
        print(f"\n已写出 {len(hit)} 个文件名 -> {args.out}")
    print("\n★ 这些是**带水印的假图**, 证据独立于我们的模型, 可以放心从训练集里挪走(挪不是删, 留证)。")
    print("  但**去了水印的假图这条判据抓不到** —— 所以这是'保证删对', 不是'删干净'。")


if __name__ == "__main__":
    main()
