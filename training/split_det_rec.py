"""拼音把 OCR 搞坏, 到底坏在**检测**还是**识别**。

为什么要问这个
--------------
之前量到: 拼音页整页送 OCR 只读出 2.0 个固定字段标签, 不带拼音的读出 9.0 个。
但"读不出"有两种完全不同的原因:

    检测坏   检测器根本没在那行画框  ->  识别器压根没见到这行
    识别坏   框画对了但字认错了      ->  芦荼额 (应当是 账户余额)

★★★★★ 我们要训的是**识别器**。
   坏在识别   -> 训完能回一大截, 这活值得做
   坏在检测   -> 训识别器**回不了多少**, 框都没画, 识别器再准也没用

★ 这个数决定验收时能跟经理承诺到什么程度, 也决定要不要连检测一起弄。

怎么量
------
对同一张图:

    A  整页送 OCR, 拿到所有检测框和文字
    B  用裁行那套找出真实文本行(这套已经人工核过, 是准的)
    C  每一行单独裁出来送 OCR       <- 识别的**上限**, 因为拼音在框外

然后:

    检测覆盖率 = B 里有多少行, 能在 A 里找到压住它的框
    识别一致率 = 那些**被框住的**行里, A 读出来的字和 C 读出来的有多像

★★ 用的是**同一张图的内部对照**, 不是"拼音图 vs 别的不带拼音的图" ——
   后者比的是两拨不一样的图, 差异里混着版面、清晰度一堆别的东西。

用法
----
    python split_det_rec.py --src D:\\download2\\pinyin_hits --n 25

★ 25 张约 3 分钟(整页 25 次 + 按行约 480 次 OCR)。
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import EXTS, annotation_labels, local_background, text_mask   # noqa: E402
from make_pinyin_pairs import LABEL_PAD, MIN_ROW_H, _pinyin_bands, _rows        # noqa: E402

HAN = re.compile(r"[\u4e00-\u9fff]")
LATIN_RUN = re.compile(r"[a-zA-ZÀ-ɏ]{4,}")


def han(s: str) -> str:
    return "".join(HAN.findall(s or ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    if not a.src.exists():
        print(f"目录不在: {a.src}")
        return
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("没装 rapidocr_onnxruntime:  pip install rapidocr_onnxruntime")
        return
    ocr = RapidOCR()

    def run(img):
        """返回 [(y0, y1, x0, 文字)]。"""
        res, _ = ocr(img)
        out = []
        for box, t, _c in (res or []):
            t = (t or "").strip()
            if not t:
                continue
            out.append((min(p[1] for p in box), max(p[1] for p in box),
                        min(p[0] for p in box), t))
        return out

    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print("没找到图")
        return
    random.seed(a.seed)
    files = random.sample(files, min(a.n, len(files)))

    tot_rows = tot_det = 0
    sims: list[float] = []
    full_han = crop_han = full_lat = crop_lat = 0

    for k, p in enumerate(files, 1):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = text_mask(gray, local_background(gray))
        _, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        _, kept, _, big = annotation_labels(mask)
        bands = _pinyin_bands(kept, st, big)
        rows = _rows(big, bands)
        if not rows:
            continue

        full = run(img)                                  # A 整页
        for _, _, _, t in full:
            full_han += len(han(t))
            full_lat += len(LATIN_RUN.findall(t))

        for y0, y1 in rows:
            tot_rows += 1
            # 这一行在整页结果里有没有被框住(纵向压住一半以上算)
            cover = [(bx, t) for by0, by1, bx, t in full
                     if (min(by1, y1) - max(by0, y0)) > 0
                     and (min(by1, y1) - max(by0, y0)) >= 0.5 * min(by1 - by0, y1 - y0)]

            # 这一行单裁出来读(拼音在框外) —— 和裁标签用的是同一套边界
            above = [(x, y) for x, y in bands if y <= y0 + 2 and (y0 - y) < (y1 - y0)]
            la = max(0, y0 - LABEL_PAD)
            if above:
                la = min(y0, max(la, max(y for _, y in above) + 1))
            lb = min(H, y1 + LABEL_PAD)
            if lb - la < MIN_ROW_H:
                continue
            ctext = " ".join(t for _, t in sorted((x, t) for _, _, x, t in run(img[la:lb, :])))
            crop_han += len(han(ctext))
            crop_lat += len(LATIN_RUN.findall(ctext))

            if cover:
                tot_det += 1
                ftext = " ".join(t for _, t in sorted(cover))
                x, y = han(ftext), han(ctext)
                if x or y:
                    sims.append(SequenceMatcher(None, x, y).ratio())
        print(f"  {k}/{len(files)}", flush=True)

    if not tot_rows:
        print("一行都没裁出来, 看看 --src 是不是拼音图目录")
        return

    print()
    print("=" * 62)
    print(f"  {len(files)} 张拼音图")
    print("=" * 62)
    print(f"  裁行找到的文本行          {tot_rows:,}")
    print(f"  整页 OCR 框住了的          {tot_det:,}   ({tot_det/tot_rows:.0%})   <- 检测覆盖率")
    if sims:
        print(f"  框住的行里字认得一样       {np.mean(sims):.0%}   "
              f"(中位 {np.median(sims):.0%})   <- 识别一致率")
    print()
    print(f"  整页读出汉字    {full_han:6,}    拼音段 {full_lat:5,}")
    print(f"  按行裁读出汉字  {crop_han:6,}    拼音段 {crop_lat:5,}")
    print(f"  按行裁多读出    {crop_han - full_han:+,} 个汉字  "
          f"({crop_han/max(1,full_han):.2f} 倍)")
    print()
    print("  怎么读这几个数:")
    print("    检测覆盖率高 + 识别一致率低  ->  坏在识别  ->  训识别器**值得**")
    print("    检测覆盖率低                ->  坏在检测  ->  训识别器只解决一部分")


if __name__ == "__main__":
    main()
