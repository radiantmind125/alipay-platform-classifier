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

import re

import cv2
import numpy as np

HAN = re.compile(r"[一-鿿]")

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
    ap.add_argument("--pairs", type=Path, default=None,
                    help="给了就同时算**天花板**: 同一张图每行 label 裁图"
                         "(拼音在框外)读出来的字拼起来能找到几个字段")
    ap.add_argument("--dump", type=Path, default=None,
                    help="把每张图三边的文字都存下来, 事后好做失败归类")
    ap.add_argument("--no-baseline", action="store_true",
                    help="★ 跳过现成 OCR 那一半。它整页只有 0.18 张/秒, 400 张要 37 分钟; "
                         "而判'我们对天花板'根本用不着它。想把某个字段量准就用这个跑大样本")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    if not a.model and not a.onnx:
        print("要给 --model 或 --onnx")
        return

    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS)
    random.seed(a.seed)
    files = random.sample(files, min(a.n, len(files)))

    # 天花板: 同一张图, 拼音在框外时现成 OCR 能读出几个字段
    ceil_by_src = {}
    if a.pairs:
        import csv
        from collections import defaultdict
        agg = defaultdict(list)
        man = a.pairs / "_pairs_labeled.csv"
        if man.exists():
            for r in csv.DictReader(man.open(encoding="utf-8-sig")):
                if (r.get("text") or "").strip():
                    agg[Path(r["source"]).name].append(r["text"])
            ceil_by_src = {k: " ".join(v) for k, v in agg.items()}

    rec = Recognizer(a.model, a.onnx)
    ocr = None
    if not a.no_baseline:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR(det_limit_type="min", det_limit_side_len=64)
        except ImportError:
            ocr = None

    import json
    dump_fh = a.dump.open("w", encoding="utf-8") if a.dump else None
    ours, base, t_ours, t_base = [], [], 0.0, 0.0
    ours22, base22 = [], []
    ceil = []
    # ★★★★★ 清单里没有这张图的时候**不能**把它当 0 算进天花板 ——
    #   那张图天花板记 0 字段 0 字, 而 ours 照常读照常记分, 等于白送我们分,
    #   缺得越多我们看着越好看。记下哪些图有清单, 比值只在这些图上算。
    #   (白图实测 400 张里缺 2 张, 天花板被拽低 0.6 字/张; 蓝图一张不缺。)
    cov = []
    txt_all_o, txt_all_b, txt_all_c = [], [], []
    # 每个字段各自被谁找到了 —— 经理那边是按字段抽的, 这个比总数有用
    hit_o = {f: 0 for f in FIELDS_15}
    hit_b = {f: 0 for f in FIELDS_15}
    hit_c = {f: 0 for f in FIELDS_15}
    n_bad = 0
    for i, p in enumerate(files, 1):
        # ★★ 图库里有坏文件, cv2.imdecode 解出来是 None。
        #   之前没防, 直接把 None 递给 rapidocr, 整个跑崩在第 74 张
        #   (白图那次 150 张只跑完 73 张)。
        #   ★ 要**整张跳过**, 不能只跳过 baseline —— 否则三边比的不是同一批图。
        probe = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if probe is None:
            n_bad += 1
            print(f"    坏图跳过: {p.name}", flush=True)
            continue
        t0 = time.time()
        segs = read_image(p, rec)
        t_ours += time.time() - t0
        txt_o = " ".join(s["text"] for s in segs)
        ours.append(found(txt_o))
        ours22.append(found(txt_o, FIELDS_15 + FIELDS_EXTRA))
        txt_all_o.append(txt_o)
        for f in FIELDS_15:
            hit_o[f] += f in txt_o
        if ceil_by_src:
            tc = ceil_by_src.get(p.name, "")
            cov.append(p.name in ceil_by_src)
            ceil.append(found(tc))
            txt_all_c.append(tc)
            for f in FIELDS_15:
                hit_c[f] += f in tc

        if ocr is not None:
            img = probe          # ★ 上面已经解过了, 不用再解一遍
            t0 = time.time()
            res, _ = ocr(img, use_cls=False)
            t_base += time.time() - t0
            txt_b = " ".join((t or "") for _b, t, _c in (res or []))
            base.append(found(txt_b))
            base22.append(found(txt_b, FIELDS_15 + FIELDS_EXTRA))
            txt_all_b.append(txt_b)
            for f in FIELDS_15:
                hit_b[f] += f in txt_b
        if dump_fh:
            dump_fh.write(json.dumps({
                "file": p.name, "ours": txt_o,
                "base": txt_b if ocr is not None else "",
                "ceiling": ceil_by_src.get(p.name, ""),
                "segs": [{"row": s0["row"], "seg": s0["seg"], "x0": s0["x0"],
                          "x1": s0["x1"], "y0": s0["y0"], "y1": s0["y1"],
                          "pin": s0["has_pinyin"], "t": s0["text"]}
                         for s0 in segs],
            }, ensure_ascii=False))
            dump_fh.write("\n")
        if i % 20 == 0:
            print(f"    {i}/{len(files)}", flush=True)

    if dump_fh:
        dump_fh.close()
    o = np.array(ours)
    print()
    print("=" * 58)
    print("  E2E BENCH  (ASCII only - safe to paste)")
    print("=" * 58)
    # ★ 跳过坏图之后, 分母得用**真正跑过的张数**, 不能再用 len(files)
    n_ok = len(ours)
    print(f"  images {n_ok}   fields tracked {len(FIELDS_15)} "
          f"(最早那次用的同一份)")
    if n_bad:
        print(f"  ★ 另有 {n_bad} 张坏图(解不出来), 三边都跳过了")
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
        print(f"    ours better on {win}/{n_ok} images, "
              f"tie {tie}, worse {n_ok-win-tie}")
    cov_m = np.array(cov, dtype=bool) if cov else None
    n_miss = int((~cov_m).sum()) if cov_m is not None else 0
    if ceil:
        c = np.array(ceil)
        print(f"  {'ceiling':<12}{np.median(c):>8.1f}{c.mean():>8.2f}"
              f"{c.min():>6}{c.max():>6}{(c == 0).sum():>7}"
              f"   <- 同图不带拼音时的上限")
        if n_miss:
            print(f"    ★★ {n_miss} 张不在配对清单里 —— 天花板给它们记 0 而我们照常记分,")
            print(f"       白送我们分, 所以下面的比值**只在另外 "
                  f"{int(cov_m.sum())} 张上算**")
        print(f"    ours / ceiling = "
              f"{o[cov_m].mean()/max(1e-9,c[cov_m].mean()):.2f}"
              f"   (1.00 就是拼音这道坎填平了)"
              f"{'  [已剔除缺清单的]' if n_miss else ''}")
    # ★★★★★ 再报一个**按汉字数**的。
    #   数字段名这个判据是给白图回单定的 —— 蓝图页面上本来就没几个那些字段
    #   (实测可用行里只有 16% 含回单字段名, 剩下是促销、徽章、姓名、金额)。
    #   在蓝图上光看字段数会得出"两边都是 1.0, 差不多"的错觉,
    #   而实际差距在**读出多少字**上。两个判据都报, 别只看一个。
    han_o = sum(len(HAN.findall(t)) for t in txt_all_o)
    print()
    print(f"  ---- 按读出的汉字数(蓝图用这个看) ----")
    print(f"    ours      {han_o/max(1,n_ok):>7.1f} 字/张")
    if txt_all_c:
        # ★ 和字段比一样, 汉字比也只在有清单的图上算
        n_cov = max(1, int(cov_m.sum()))
        han_c = sum(len(HAN.findall(t)) for t, k in zip(txt_all_c, cov) if k)
        han_o_c = sum(len(HAN.findall(t)) for t, k in zip(txt_all_o, cov) if k)
        print(f"    ceiling   {han_c/n_cov:>7.1f} 字/张"
              f"   ours/ceiling = {han_o_c/max(1,han_c):.2f}"
              f"{'  [已剔除缺清单的]' if n_miss else ''}")
    if txt_all_b:
        han_b = sum(len(HAN.findall(t)) for t in txt_all_b)
        print(f"    rapidocr  {han_b/max(1,n_ok):>7.1f} 字/张"
              f"   ours/rapidocr = {han_o/max(1,han_b):.2f}")
    print()
    if ceil:
        print("  ---- 每个字段各自被谁读到 (占这批图的比例) ----")
        # ★ 没跑现成 OCR 的时候**不要打那一列**。打个 0% 会被读成"它一个都没读到",
        #   而实际是"根本没量"。没量的东西打成 0 是误导, 比不打更糟。
        head = f"    {'field':<10}{'ceiling':>9}{'ours':>8}"
        print(head + (f"{'rapidocr':>10}" if base else "      (未跑现成 OCR)"))
        n_img = n_ok
        for f in FIELDS_15:
            line = f"    {f:<10}{hit_c[f]/n_img:>8.0%}{hit_o[f]/n_img:>8.0%}"
            print(line + (f"{hit_b[f]/n_img:>9.0%}" if base else ""))
        print()
    if ours22:
        tail = (f"  rapidocr {np.median(base22):.1f}" if base22 else "")
        print(f"  (22 字段表: ours {np.median(ours22):.1f}{tail})")
    print()
    print("  ---- 参照 ----")
    print("    最早报过: 带拼音 2.0 / 不带拼音 9.0 个字段")
    print("    ★★ 但 2.0 这个数**复现不出来** —— 2026-09-18 拿 25 张拼音图")
    print("       用同一份 15 字段表、原厂默认配置重量, 得到的是 4.0。")
    print("       所以页面级别的差距是 4 比 9 左右, 不是 2 比 9。报数要按 4 报。")
    print()
    print(f"  speed   ours {n_ok/max(t_ours,1e-9):.2f} img/s"
          f"   (生产是 CPU, 底线 1 张/秒)")
    if base:
        print(f"          rapidocr {n_ok/max(t_base,1e-9):.2f} img/s")


if __name__ == "__main__":
    main()
