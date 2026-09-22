"""把"一行"再切成"一段", 出真正能喂给识别器的样本。

为什么必须切
------------
识别器吃的是**一条紧贴着的文本行**, 标准输入 [3, 48, 320], 宽高比 6.7:1。

我们的行图是 1080x43, **26:1**, 而且实测 69% 的行里有两段以上 ——
左边字段名, 右边取值, 中间一大片空白:

    付款方式                                    账户余额

整行压进 320 宽要挤掉 75% 的宽度, 字就糊成一团。**行不是识别器的单位, 段才是。**

★ 顺带还解决一件: 切成段之后每段的 x 区间是知道的, 左边哪段是字段名、
  右边哪段是取值, 位置信息留着, 后面做字段抽取用得上。

怎么切 —— 不用重跑那 17 小时的 OCR
----------------------------------
★★★★★ 填标签的时候, 一行里 OCR 读出的几段是**按 x 排好、用空格拼起来的**:

        parts.sort()                      # 按 x
        text = " ".join(s for _, s in parts)

   所以 `text.split(" ")` 就是**从左到右排好的每一段文字**。
   这边只要从图上按空白把段的 x 区间切出来, **两边段数一样就能一一对上**。

★★ 段数对不上的行**直接丢**(比如某一段自己带空格)。丢得起 —— 有 12.7 万行,
   实测对得上的约 81%。宁可少一点, 不要错配 —— 错配的样本比没有更糟。

切在哪张图上, 用在哪张图上
--------------------------
★ x 区间在 **label/**(不含拼音)上算 —— 拼音比汉字宽, 在 input/ 上算会把
  相邻两段的拼音连起来, 该断的地方断不开。
★ 算出来的 x 区间**同时**用在 label/ 和 input/ 上 —— 两张图等宽, x 直接通用。

用法
----
    python split_segments.py --pairs D:\\alipay-ai-data\\pinyin-pairs
    python split_segments.py --pairs ... --sheet 16      # 出一张核对图
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import local_background, text_mask  # noqa: E402

# 比这么宽的空白才算"段与段之间"(按字高的倍数)。
# 实测 1.6~2.4 倍之间对得上的比例都在 81%, 不敏感, 取中间。
HAN = re.compile(r"[一-鿿]")

GAP_RATIO = 1.8

PAD = 4          # 每段左右各留几像素
MIN_SEG_W = 8    # 比这还窄的段不要


def segments(label_img: np.ndarray) -> list[tuple[int, int]]:
    """按列方向的空白把一行切成几段, 返回 [(x0, x1)]。"""
    g = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY) if label_img.ndim == 3 else label_img
    m = text_mask(g, local_background(g))
    col = m.sum(axis=0) > 0
    h = label_img.shape[0]
    gap = max(6, int(h * GAP_RATIO))
    segs: list[tuple[int, int]] = []
    run = None
    blank = 0
    for x, on in enumerate(col):
        if on:
            run = [x, x] if run is None else [run[0], x]
            blank = 0
        elif run is not None:
            blank += 1
            if blank >= gap:
                segs.append((run[0], run[1]))
                run = None
    if run is not None:
        segs.append((run[0], run[1]))

    # 太窄的碎块(标点、图标残留)并进前一段, 不然段数对不上
    out: list[tuple[int, int]] = []
    for a, b in segs:
        if out and (a - out[-1][1]) < gap and (b - a) < h * 0.4:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))

    # ★★★★★ 再把**又窄又孤立**的碎片扔掉。
    #
    #   典型就是蓝色按钮的两个**圆角**: 按钮内部底色是蓝的, 算不出墨;
    #   只有圆角和白底交界处出一条细弧, 被当成了"一段文字"。实测:
    #       再转一笔 那一行  ->  切成 3 段  (35,50) (518,740) (1210,1225)
    #   两头 15 像素宽的就是圆角, 送进识别器读成 `支` 和 `K`。
    #
    #   更要命的是**训练时**的影响: 段数对不上的行会被整行丢掉,
    #   所以这类蓝按钮行根本没进训练集 —— 模型没见过, 上线自然读不好。
    #   这一条修的是两头: 上线不再出碎片, 重切之后训练集也能把这些行收回来。
    #
    # ★ 阈值 0.4 是量出来的: 707 个真文字段里**没有一个**宽高比低于 0.4
    #   (5% 分位是 2.44), 而疑似碎片里有 14.6% 在这条线以下。
    #   也试过用墨密度区分, **不行** —— 真文字和圆角碎片都是 0.31 上下, 分不开。
    return [(a, b) for a, b in out if (b - a) >= h * 0.4]


def _units(t: str) -> float:
    """这段文字大概占几个字宽。汉字算 1, 半角算 0.5。"""
    return sum(1.0 if HAN.match(c) else 0.5 for c in t)


def calibrate_px_per_unit(rows, pairs: Path, sample: int = 400) -> float:
    """量"一个字宽大概多少像素" —— 只拿**段数本来就对得上**的行来量。

    ★★★★★ 不能用行高当一个字宽。试过, 差了一倍:
       `计入收支` 那一行行高约 40, 但四个字只占 88 像素, 每字 22。
       因为行的上下边界是**成员中位**, 而右边那个开关又高又大, 把行高撑起来了。
       拿撑起来的行高当字宽, 算出来的"预期宽度"是实际的两倍,
       于是两个候选都显得"太窄", 谁也选不出来。

    ★ 段数本来就对得上的行是可信的 —— 它们的 宽度/字数 就是真实的字宽。
    """
    vals = []
    for r in rows[: sample * 4]:
        if len(vals) >= sample:
            break
        try:
            lab = cv2.imdecode(np.fromfile(str(pairs / "label" / r["file"]),
                                           np.uint8), cv2.IMREAD_COLOR)
            if lab is None:
                continue
            parts = r["text"].split(" ")
            segs = segments(lab)
            if len(segs) != len(parts):
                continue
            for (x0, x1), t in zip(segs, parts):
                u = _units(t)
                if u > 0:
                    vals.append((x1 - x0) / u)
        except Exception:                    # noqa: BLE001
            continue
    return float(np.median(vals)) if vals else 40.0


def drop_one_extra(segs, parts, px_per_unit):
    """几何段比文字段**正好多一个**时, 试着丢掉一个再配。丢不准就返回 None。

    ★★★★★ 为什么非做不可: 这一类占了丢弃行的大头, 而且是**成类丢**的。
       实测 `计入收支` 那一行 —— 左边四个字, 右边一个开关 ——
       4,286 行里 **400/400 全部对不上**, 无一例外:

           文字 1 段 / 图上 2 段   宽度 [163, 96]

       右边那个 96 像素的是**开关**, 有墨但不是字。于是整行被丢,
       最后 17.9 万段里含"计入收支"的只剩 **1 条**。
       模型等于从没学过这个词, 整页验收里它的命中率在 37%~57% 之间乱跳 ——
       那不是退步, 是**从来就没有过信号**。

    怎么挑丢哪个
    ------------
    一段文字有几个字, 它的图就该有多宽。逐个试着丢掉某一段,
    算剩下的配对有多贴合这个预期, 取最贴合的那种。

    ★ "一个字宽"必须**从数据里量**(见 calibrate_px_per_unit), 不能拿行高凑 ——
      行高是被行里最高的元件撑起来的, 开关、图标一掺进去就偏一倍。

    ★★ 但**丢得不干脆就不丢**: 最好的那种要明显好过次好的(差一倍以上),
       否则宁可整行扔掉。错配的样本比没有更糟 —— 这条已经吃过亏。
    """
    if len(segs) != len(parts) + 1 or px_per_unit <= 0:
        return None
    want = [_units(t) for t in parts]
    if any(u <= 0 for u in want):
        return None

    def cost(sel):
        c = 0.0
        for (x0, x1), u in zip(sel, want):
            ratio = (x1 - x0) / (u * px_per_unit)
            c += abs(math.log(max(ratio, 1e-3)))
        return c

    scored = []
    for i in range(len(segs)):
        sel = segs[:i] + segs[i + 1:]
        scored.append((cost(sel), i, sel))
    scored.sort(key=lambda t: t[0])
    best, second = scored[0], scored[1]
    # 最好的要明显好过次好的, 而且本身得够贴合
    if best[0] > 0.55 * len(parts):
        return None
    if second[0] < best[0] * 2.0:
        return None
    return best[2]


def process(args) -> list[dict]:
    rec, pairs, out_dir, px_per_unit = args
    try:
        lab = cv2.imdecode(np.fromfile(str(Path(pairs) / "label" / rec["file"]),
                                       np.uint8), cv2.IMREAD_COLOR)
        if lab is None:
            return []
        parts = rec["text"].split(" ")
        segs = segments(lab)
        if len(segs) == len(parts) + 1:
            fixed = drop_one_extra(segs, parts, px_per_unit)
            if fixed is not None:
                segs = fixed
        if len(segs) != len(parts):
            return []                       # ★ 还对不上就整行丢掉, 不猜

        inp = cv2.imdecode(np.fromfile(str(Path(pairs) / "input" / rec["file"]),
                                       np.uint8), cv2.IMREAD_COLOR)
        if inp is None or inp.shape[1] != lab.shape[1]:
            return []

        di = Path(out_dir) / "seg_input"
        dl = Path(out_dir) / "seg_label"
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)

        rows = []
        stem = Path(rec["file"]).stem
        W = lab.shape[1]
        for i, ((x0, x1), txt) in enumerate(zip(segs, parts)):
            a = max(0, x0 - PAD)
            b = min(W, x1 + PAD + 1)
            if b - a < MIN_SEG_W or not txt.strip():
                continue
            name = f"{stem}_s{i:02d}.png"
            cv2.imencode(".png", inp[:, a:b])[1].tofile(str(di / name))
            cv2.imencode(".png", lab[:, a:b])[1].tofile(str(dl / name))
            rows.append({
                "file": name, "row_file": rec["file"], "source": rec["source"],
                "seg": i, "x0": a, "x1": b,
                "has_pinyin": rec.get("has_pinyin", "0"), "text": txt,
            })
        return rows
    except Exception:                        # noqa: BLE001
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--sheet", type=int, default=0,
                    help="切完再出一张核对图, 抽这么多段")
    a = ap.parse_args()

    man = a.pairs / "_pairs_labeled.csv"
    if not man.exists():
        print(f"清单不在: {man}")
        return
    rows = [r for r in csv.DictReader(man.open(encoding="utf-8-sig"))
            if r.get("usable") == "1" and (r.get("text") or "").strip()]
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        print("没有可用的行")
        return

    out_dir = a.out or a.pairs
    out_dir.mkdir(parents=True, exist_ok=True)
    ppu = calibrate_px_per_unit(rows, a.pairs)
    print(f"要切 {len(rows):,} 行,  {a.workers} 个进程,  量出来一个字宽 {ppu:.1f} 像素")

    allseg: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, got in enumerate(ex.map(process,
                                       [(r, str(a.pairs), str(out_dir), ppu) for r in rows],
                                       chunksize=64), 1):
            allseg.extend(got)
            if i % 20000 == 0:
                print(f"  {i:,}/{len(rows):,}   出了 {len(allseg):,} 段", flush=True)

    man2 = out_dir / "_segments.csv"
    with man2.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "row_file", "source", "seg",
                                          "x0", "x1", "has_pinyin", "text"])
        w.writeheader()
        w.writerows(allseg)

    ok_rows = len({r["row_file"] for r in allseg})
    print()
    print("=" * 60)
    print(f"  {len(rows):,} 行  ->  段数对得上的 {ok_rows:,} 行 ({ok_rows/len(rows):.0%})")
    print(f"  共出 {len(allseg):,} 段样本")
    if allseg:
        py = sum(1 for r in allseg if r["has_pinyin"] == "1")
        print(f"  其中上面有拼音的 {py:,} ({py/len(allseg):.0%})")
    print("=" * 60)
    print(f"  {out_dir / 'seg_input'}")
    print(f"  {out_dir / 'seg_label'}")
    print(f"  {man2}")

    if a.sheet and allseg:
        from make_check_sheet import _cjk_font
        font = _cjk_font(22)
        random.seed(7)
        pick = random.sample(allseg, min(a.sheet, len(allseg)))
        ims, W = [], 0
        for r in pick:
            im = cv2.imdecode(np.fromfile(str(out_dir / "seg_input" / r["file"]),
                                          np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.copyMakeBorder(im, 3, 3, 8, 8, cv2.BORDER_CONSTANT,
                                    value=(215, 215, 215))
            ims.append(im)
            W = max(W, im.shape[1])
        TW = 520 if font else 0
        ims = [cv2.copyMakeBorder(m, 0, 0, 0, W - m.shape[1] + TW,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
               for m in ims]
        out = np.vstack(ims)
        if font:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            dr = ImageDraw.Draw(pil)
            y = 0
            for m, r in zip(ims, pick):
                h = m.shape[0]
                dr.line([(W, y), (W, y + h)], fill=(190, 190, 190))
                dr.text((W + 10, y + max(0, (h - 26) // 2)), r["text"][:26],
                        font=font, fill=(150, 0, 0))
                y += h
            out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        dst = out_dir / "_check_segments.png"
        cv2.imencode(".png", out)[1].tofile(str(dst))
        print()
        print(f"★ 核对图 -> {dst}")
        print("   左边是切出来的段, 右边红字是配给它的文字。**必须一一对得上。**")


if __name__ == "__main__":
    main()
