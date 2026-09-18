"""把"不带拼音能读出几个字段"这个天花板量准 —— 用**同一张图的内部对照**。

为什么要重做这个数
------------------
最早报的是"带拼音 2.0 / 不带拼音 9.0"。两个问题:

    1. 2.0 复现不出来(2026-09-18 实测 4.0), 见 PREMISE-CORRECTED
    2. 9.0 是在**另一批图**上量的 —— 那批是干净回单, 和拼音图不是同一批。
       两批图的版面、字段多少、清晰度都不一样, 这么比, 差异里混着一堆别的东西。

★★★★★ 这里换成**同一张图的内部对照**:

    带拼音     整页直接送现成 OCR
    不带拼音   这张图每一行的 label 裁图(拼音在框外)读出来的字, 拼起来

   label 裁图的文字在做训练集的时候**已经读过了**, 存在 _pairs_labeled.csv 里,
   所以这一半不用重跑 OCR, 一秒就能算完 7205 张。

★★ 这个天花板对"不带拼音"是**偏高**的: 按行裁本身就帮了 OCR 一把
   (每行单独送, 版面干扰小)。所以真实的干净页天花板只会比这个低。
   宁可把天花板估高 —— 这样我们自己的成绩只会被压着, 不会被抬着。

用法
----
    python ceiling_bench.py --pairs D:\\alipay-ai-data\\pinyin-pairs
    python ceiling_bench.py --pairs ... --with-ocr --n 200   # 顺便量带拼音那一半
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

# 和 e2e_bench 同一份 15 个词, 也是最早那次用的
FIELDS_15 = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号", "转账备注",
             "收款方", "账户余额", "计入收支", "转账成功", "交易成功", "对方账户",
             "商家名称", "支付时间", "交易号"]


def found(text: str) -> int:
    return sum(1 for f in FIELDS_15 if f in text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--with-ocr", action="store_true",
                    help="同时把整页(带拼音)送现成 OCR 量一遍。慢, 约 5 张/分钟")
    ap.add_argument("--n", type=int, default=200, help="--with-ocr 时量几张")
    ap.add_argument("--img-dir", type=Path, default=None,
                    help="★ 清单里存的是**出清单那台机器上的**路径。换一台机器跑时"
                         "用这个指到本地的图目录, 按文件名去找")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    man = a.pairs / "_pairs_labeled.csv"
    if not man.exists():
        print(f"清单不在: {man}")
        return

    by_img: dict[str, list[str]] = defaultdict(list)
    for r in csv.DictReader(man.open(encoding="utf-8-sig")):
        if (r.get("text") or "").strip():
            by_img[r["source"]].append(r["text"])

    ceil = {src: found(" ".join(ts)) for src, ts in by_img.items()}
    c = np.array(list(ceil.values()))
    print("=" * 58)
    print("  CEILING BENCH  (ASCII only - safe to paste)")
    print("=" * 58)
    print(f"  images {len(c):,}   fields tracked {len(FIELDS_15)}")
    print()
    print("  不带拼音的天花板 (同一张图, 每行 label 裁图读出来的字拼起来)")
    print(f"    median {np.median(c):.1f}   mean {c.mean():.2f}   "
          f"p10 {np.percentile(c,10):.0f}   p90 {np.percentile(c,90):.0f}   "
          f"zero {(c==0).sum()}")
    print()
    print("  ★ 最早报的 9.0 就是这个位置的数, 但那是在**另一批干净图**上量的。")
    print("    这里是同一张图的内部对照, 可比性强得多。")

    if not a.with_ocr:
        print()
        print("  (要量带拼音那一半就加 --with-ocr, 慢)")
        return

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("\n  没装 rapidocr, 跳过")
        return
    import cv2
    ocr = RapidOCR()

    srcs = sorted(by_img)
    random.seed(a.seed)
    srcs = random.sample(srcs, min(a.n, len(srcs)))
    with_py, ceil_sub = [], []
    missing = 0
    for i, s in enumerate(srcs, 1):
        p = Path(s)
        if a.img_dir:
            p = a.img_dir / p.name
        if not p.exists():
            missing += 1
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        res, _ = ocr(img)
        with_py.append(found(" ".join((t or "") for _b, t, _c in (res or []))))
        ceil_sub.append(ceil[s])
        if i % 25 == 0:
            print(f"    {i}/{len(srcs)}", flush=True)

    if missing:
        # ★ 之前这里是**一声不吭**跳过, 结果整批图都找不到还照样往下算,
        #   最后打出来一堆 nan。找不到图是大事, 必须吼出来。
        print()
        print("  ★★ {}/{} 张图找不到 —— 清单里存的是出清单那台机器的路径, "
              "用 --img-dir 指到本地图目录".format(missing, len(srcs)))
    if not with_py:
        print("  一张都没读到, 停")
        return
    w = np.array(with_py)
    cs = np.array(ceil_sub)
    print()
    print(f"  同一批 {len(w)} 张图上:")
    print(f"    {'':<22}{'median':>8}{'mean':>8}{'zero':>7}")
    print(f"    {'带拼音 整页送 OCR':<22}{np.median(w):>8.1f}{w.mean():>8.2f}"
          f"{(w==0).sum():>7}")
    print(f"    {'不带拼音 天花板':<22}{np.median(cs):>8.1f}{cs.mean():>8.2f}"
          f"{(cs==0).sum():>7}")
    print()
    print(f"  ★ 拼音让现成 OCR 掉了 {1 - w.mean()/max(1e-9, cs.mean()):.0%} "
          f"({cs.mean():.2f} -> {w.mean():.2f})")


if __name__ == "__main__":
    main()
