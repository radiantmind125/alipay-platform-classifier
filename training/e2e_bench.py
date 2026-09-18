"""端到端验收: 整张拼音截图, 我们这条链 vs 现成 OCR, 谁把字段读出来得多。

为什么用"数字段名"这个判据
--------------------------
★★★★★ 因为整件事**最早就是用这个数量出来的**:

    带拼音的图    整页送现成 OCR, 每张读出固定字段标签 中位 2.0 个
    不带拼音的图  中位 9.0 个

   经理那句"拼音的不行 ocr 识别出来很多都是不全的"指的就是这个。
   现在拿**同一个判据**回来量一遍, 才算真正回答了他的问题 ——
   换个好看的判据来报, 等于没回答。

★ 字段名是固定词(创建时间/付款方式/订单号...), 出现即说明这一行读对了,
  不用人工标注也不会有歧义。

判据对**现成 OCR 是宽松**的
---------------------------
★★ 它整页读出来的东西里只要**任何位置**出现了字段名就算它读到,
   不要求位置对、不要求别的字读对。这样算它只会高不会低。
   我们这边同样算法。**宁可对自己严一点。**

用法
----
    python e2e_bench.py --src D:\\download2\\pinyin_hits --model rec_best.pt --n 60
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import EXTS  # noqa: E402
from pinyin_ocr import Recognizer, read_image  # noqa: E402

# 回单上的固定字段名。和最早那次量 2.0 / 9.0 时用的是同一批词。
# ★★★★★ 用**最早那次的同一份 15 个词**, 这样才和当初那个数可比。
#   后来我一度加长到 22 个, 数自然变大 —— 换判据再报数等于偷换题目。
FIELDS_15 = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号", "转账备注",
             "收款方", "账户余额", "计入收支", "转账成功", "交易成功", "对方账户",
             "商家名称", "支付时间", "交易号"]
FIELDS_EXTRA = ["备注", "标签", "账单分类", "收款方全称", "账单管理",
                "支付成功", "支付宝"]
FIELDS = FIELDS_15


def found(text: str, fields=None) -> int:
    return sum(1 for f in (fields or FIELDS_15) if f in text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    if not a.model and not a.onnx:
        print("要给 --model 或 --onnx")
        return

    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS)
    random.seed(a.seed)
    files = random.sample(files, min(a.n, len(files)))

    rec = Recognizer(a.model, a.onnx)
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR(det_limit_type="min", det_limit_side_len=64)
    except ImportError:
        ocr = None

    ours, base, t_ours, t_base = [], [], 0.0, 0.0
    ours22, base22 = [], []
    for i, p in enumerate(files, 1):
        t0 = time.time()
        segs = read_image(p, rec)
        t_ours += time.time() - t0
        txt_o = " ".join(s["text"] for s in segs)
        ours.append(found(txt_o))
        ours22.append(found(txt_o, FIELDS_15 + FIELDS_EXTRA))

        if ocr is not None:
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            t0 = time.time()
            res, _ = ocr(img, use_cls=False)
            t_base += time.time() - t0
            txt_b = " ".join((t or "") for _b, t, _c in (res or []))
            base.append(found(txt_b))
            base22.append(found(txt_b, FIELDS_15 + FIELDS_EXTRA))
        if i % 20 == 0:
            print(f"    {i}/{len(files)}", flush=True)

    o = np.array(ours)
    print()
    print("=" * 58)
    print("  E2E BENCH  (ASCII only - safe to paste)")
    print("=" * 58)
    print(f"  images {len(files)}   fields tracked {len(FIELDS_15)} "
          f"(最早那次用的同一份)")
    print()
    print(f"  {'':<12}{'median':>8}{'mean':>8}{'min':>6}{'max':>6}{'zero':>7}")
    print(f"  {'ours':<12}{np.median(o):>8.1f}{o.mean():>8.2f}"
          f"{o.min():>6}{o.max():>6}{(o == 0).sum():>7}")
    if base:
        b = np.array(base)
        print(f"  {'rapidocr':<12}{np.median(b):>8.1f}{b.mean():>8.2f}"
              f"{b.min():>6}{b.max():>6}{(b == 0).sum():>7}")
        print()
        print(f"  ★ ours / rapidocr  =  {o.mean()/max(1e-9,b.mean()):.2f}x")
        win = int((o > b).sum())
        tie = int((o == b).sum())
        print(f"    ours better on {win}/{len(files)} images, "
              f"tie {tie}, worse {len(files)-win-tie}")
    print()
    if ours22:
        print(f"  (22 字段表: ours {np.median(ours22):.1f}  "
              f"rapidocr {np.median(base22) if base22 else float('nan'):.1f})")
    print()
    print("  ---- 参照 ----")
    print("    最早报过: 带拼音 2.0 / 不带拼音 9.0 个字段")
    print("    ★★ 但 2.0 这个数**复现不出来** —— 2026-09-18 拿 25 张拼音图")
    print("       用同一份 15 字段表、原厂默认配置重量, 得到的是 4.0。")
    print("       所以页面级别的差距是 4 比 9 左右, 不是 2 比 9。报数要按 4 报。")
    print()
    print(f"  speed   ours {len(files)/max(t_ours,1e-9):.2f} img/s"
          f"   (生产是 CPU, 底线 1 张/秒)")
    if base:
        print(f"          rapidocr {len(files)/max(t_base,1e-9):.2f} img/s")


if __name__ == "__main__":
    main()
