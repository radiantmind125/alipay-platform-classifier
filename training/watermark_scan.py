r"""扫 AIGC 水印(豆包/千问/…) —— **不看模型分数**的假图判据。

为什么必须有一条"不看模型分数"的判据
------------------------------------
清训练集的时候有个陷阱: 如果用**模型自己打的高分**去挑要删的图, 那删掉的里面既有真假图,
也有**本来就难的真图**(困难负样本)。**困难负样本正是模型最需要的。**
删了它们 -> 模型对难的真图变得更自信 -> **线下每个指标都变好, 线上误杀反而变差**。
这跟"把转发压缩过的真图当假图剔掉"是同一个坑, 只是更深 —— 它直接烙进权重里, 而不只是一个阈值。

**水印是干净的判据**: 它是生成器自己盖上去的, 跟我们的模型完全无关, 不存在循环论证。
所以清训练集**只该按水印这类独立证据删**, 不该按模型分数删。

模板**内嵌在本文件里**, 不依赖任何外部图片
------------------------------------------
第一版让 `--ref-dir` 指向某台机器上的图片, 结果**换台机器直接跑不起来**("一个模板都没建起来"),
而且**阈值不可复现** —— 这份名单是要拿去删训练数据、标误杀阈值的, 不可复现不能接受。
所以模板直接内嵌(zlib+base64, 约 10KB)。**内嵌的两块只含水印笔画和纯色背景,
不含任何金额/姓名/订单号**, 已逐张放大肉眼确认过才提交。
厂商换了新版水印时, 用 `--doubao-ref` / `--qwen-ref` 指一张新样本即可重建。

已编目(实测相对坐标)
--------------------
  豆包AI生成   右下角  x 0.820-0.990  y 0.970-0.995   已内嵌
  千问AI生成   右下角  x 0.790-0.992  y 0.965-0.987   已内嵌
  AI生成       左上角  (诈骗工具那种)                  **还没做模板**
**踩过的坑: 只扫右下角会漏掉左上角那种。左上角这种目前还得靠人工。**

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
import base64
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# 内嵌模板: 名字 -> (灰度数组形状, 相对框 x0,y0,x1,y1, zlib+base64 的字节…)
# 抠模板要挑**压缩噪声小的 PNG**做源, 否则模板自己就带噪声, 相关值会被整体拉低。
# 只含水印笔画和纯色背景, **不含任何金额/姓名/订单号**, 已逐张肉眼确认。
_TEMPLATES = {
    "豆包AI生成": (
        (75, 237),
        (0.82, 0.97, 0.99, 0.995),
        "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"
    ),
    "千问AI生成": (
        (61, 254),
        (0.79, 0.965, 0.992, 0.987),
        "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"
    ),
}

# 相对偏移搜索: 不同生成器画水印的位置会差一点点, 死扣一个框会漏
_OFFSETS = [(dx, dy) for dx in (-0.012, 0.0, 0.012) for dy in (-0.010, 0.0, 0.010)]


def _norm(a: np.ndarray) -> tuple[np.ndarray, float]:
    a = a - a.mean()
    return a, float(np.sqrt((a ** 2).sum()))


def _load_embedded(name: str):
    """从本文件内嵌的数据建模板 —— 换台机器也一样, 阈值可复现。"""
    shape, box, *chunks = _TEMPLATES[name]
    raw = zlib.decompress(base64.b64decode("".join(chunks)))
    t = np.frombuffer(raw, np.uint8).reshape(shape).astype(float)
    tn, tnorm = _norm(t)
    return (tn, tnorm, shape), box


def _load_from_image(ref: Path, box):
    """想用自己的图重建模板时走这条(比如厂商出了新版水印)。"""
    im = Image.open(ref).convert("L")
    w, h = im.size
    t = np.asarray(im.crop((int(box[0] * w), int(box[1] * h),
                            int(box[2] * w), int(box[3] * h)))).astype(float)
    tn, tnorm = _norm(t)
    if tnorm < 1e-6:
        raise SystemExit(f"模板区域是纯色, 取不到水印: {ref}")
    return (tn, tnorm, t.shape), box


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
    ap.add_argument("--doubao-ref", type=Path, default=None,
                    help="不用内嵌模板, 改从这张图重建豆包模板(厂商出了新版水印时用)")
    ap.add_argument("--qwen-ref", type=Path, default=None, help="同上, 千问的")
    ap.add_argument("--thr", type=float, default=0.30,
                    help="判命中的相关阈值。实测标定: 0.30 -> 带水印的 11/11 全抓到, 3000 张真图里 2 张(均确认是假图)")
    ap.add_argument("--selftest", action="store_true", help="在已查实假图和真图上各跑一遍验证阈值")
    ap.add_argument("--fakes", type=Path, default=None, help="自检用: 已查实假图目录")
    ap.add_argument("--genuine", type=Path, default=None, help="自检用: 真图目录")
    ap.add_argument("--limit", type=int, default=3000, help="自检时真图最多取多少张")
    ap.add_argument("--show", type=int, default=30)
    args = ap.parse_args()

    override = {"豆包AI生成": args.doubao_ref, "千问AI生成": args.qwen_ref}
    tpls = []
    for name in _TEMPLATES:
        ref = override.get(name)
        if ref:
            tpl, box = _load_from_image(Path(ref), _TEMPLATES[name][1])
            src = f"来自 {Path(ref).name}"
        else:
            tpl, box = _load_embedded(name)
            src = "内嵌"
        tpls.append((name, tpl, box))
        print(f"模板 {name}: {src}  {tpl[2][1]}x{tpl[2][0]}")

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
