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
    return out


def process(args) -> list[dict]:
    rec, pairs, out_dir = args
    try:
        lab = cv2.imdecode(np.fromfile(str(Path(pairs) / "label" / rec["file"]),
                                       np.uint8), cv2.IMREAD_COLOR)
        if lab is None:
            return []
        parts = rec["text"].split(" ")
        segs = segments(lab)
        if len(segs) != len(parts):
            return []                       # ★ 对不上就整行丢掉, 不猜

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
    print(f"要切 {len(rows):,} 行,  {a.workers} 个进程")

    allseg: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, got in enumerate(ex.map(process,
                                       [(r, str(a.pairs), str(out_dir)) for r in rows],
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
