"""量一下训出来的识别器到底行不行 —— 输出**全 ASCII**, 不怕控制台编码。

报三件
------
1. 字准确率 / 整条全对率        用编辑距离算, 不是逐位置比
2. **带拼音的 vs 不带拼音的分开报**
   ★★★★★ 这是最要紧的一个数。整个活就是为了"上面压着拼音也能读对",
      要是带拼音那档明显低, 说明拼音还是在干扰, 训练没达到目的。
3. 按标签长短分档                短的准长的不准, 说明时间步或者感受野不够

还出一张核对图: 左边段图, 中间真值, 右边模型读出来的, 错的标红。
文字画进图里, 不走控制台, 免得 GBK 把中文糊成乱码。

用法
----
    python eval_rec.py --data D:\\alipay-ai-data\\pinyin-pairs
    python eval_rec.py --data ... --onnx D:\\alipay-ai-data\\rec_pinyin.onnx
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_rec import (IMG_H, MIN_W, MAX_W, Charset, edit_distance,  # noqa: E402
                       load_list, prep)


def han_only(s: str) -> str:
    return "".join(c for c in s if "\u4e00" <= c <= "\u9fff")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None, help="默认 <data>/rec_best.pt")
    ap.add_argument("--onnx", type=Path, default=None, help="给了就量 onnx 那份")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--sheet", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--baseline", action="store_true",
                    help="★★★★★ 同一批图再用现成 rapidocr 读一遍做对照。"
                         "这才是要交出去的那个数: 同样带拼音的图, 它读成什么, 我们读成什么")
    a = ap.parse_args()

    va = load_list(a.data / "val.txt")
    if not va:
        print("val.txt 是空的")
        return
    random.seed(11)
    if a.limit and len(va) > a.limit:
        va = random.sample(va, a.limit)

    # 哪些段上面有拼音 —— 从段清单里查
    pin = {}
    seg = a.data / "_segments.csv"
    if seg.exists():
        for r in csv.DictReader(seg.open(encoding="utf-8-sig")):
            pin[r["file"]] = r.get("has_pinyin") == "1"

    # ---------- 载模型 ----------
    if a.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
        chars = [c for c in (a.onnx.parent / "charset.txt").read_text(
            encoding="utf-8").split("\n") if c]
        cs = Charset(chars)

        def run(x):
            return sess.run(None, {"x": x})[0]
        tag = f"onnx {a.onnx.name}"
    else:
        import torch
        from train_rec import CRNN
        mp = a.model or (a.data / "rec_best.pt")
        if not mp.exists():
            print(f"MODEL NOT FOUND: {mp}")
            return
        ck = torch.load(mp, map_location="cpu")
        chars = ck["chars"]
        cs = Charset(chars)
        model = CRNN(len(cs))
        model.load_state_dict(ck["model"])
        model.eval()

        def run(x):
            with torch.no_grad():
                return model(torch.from_numpy(x)).numpy()
        tag = f"torch {mp.name}"

    # ---------- 跑 ----------
    recs = []
    for i in range(0, len(va), a.batch):
        chunk = va[i:i + a.batch]
        imgs, keep = [], []
        for path, txt in chunk:
            im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            imgs.append(prep(im))
            keep.append((path, txt))
        if not imgs:
            continue
        W = max(x.shape[2] for x in imgs)
        x = np.empty((len(imgs), 3, IMG_H, W), np.float32)
        widths = []
        for k, im in enumerate(imgs):
            w = im.shape[2]
            x[k, :, :, :w] = im
            if w < W:
                x[k, :, :, w:] = im[:, :, -1:]
            widths.append(w)
        logits = run(x)
        ids = logits.argmax(2)
        for k, (path, gt) in enumerate(keep):
            hyp = cs.decode(ids[k][: max(1, widths[k] // 4)])
            recs.append({"file": Path(path).name, "path": path, "gt": gt, "hyp": hyp,
                         "pin": pin.get(Path(path).name)})
        if (i // a.batch) % 20 == 0:
            print(f"    {min(i+a.batch,len(va)):,}/{len(va):,}", flush=True)

    # ---------- 报数 ----------
    def score(rs):
        if not rs:
            return 0.0, 0.0, 0
        ed = sum(edit_distance(r["hyp"], r["gt"]) for r in rs)
        tot = sum(max(1, len(r["gt"])) for r in rs)
        ex = sum(1 for r in rs if r["hyp"] == r["gt"])
        return 1 - ed / tot, ex / len(rs), len(rs)

    print()
    print("=" * 58)
    print("  REC EVAL  (ASCII only - safe to paste)")
    print("=" * 58)
    print(f"  model        {tag}")
    print(f"  charset      {len(cs)} (incl blank)")
    acc, ex, n = score(recs)
    print(f"  samples      {n:,}")
    print(f"  char acc     {acc:7.2%}     <- 1 - 编辑距离/总字数")
    print(f"  exact match  {ex:7.2%}")
    print()
    hr = [r for r in recs if han_only(r["gt"])]
    hacc = 1 - (sum(edit_distance(han_only(r["hyp"]), han_only(r["gt"])) for r in hr)
                / max(1, sum(len(han_only(r["gt"])) for r in hr)))
    print(f"  chinese-only char acc  {hacc:7.2%}   ({len(hr):,} samples)")
    print()
    if any(r["pin"] is not None for r in recs):
        print("  ---- 带拼音 vs 不带拼音 (★ 最要紧的一档) ----")
        for name, sub in (("with pinyin", [r for r in recs if r["pin"] is True]),
                          ("no pinyin", [r for r in recs if r["pin"] is False])):
            acc2, ex2, n2 = score(sub)
            print(f"    {name:<14}{n2:>7,}   char {acc2:7.2%}   exact {ex2:7.2%}")
        print()
    print("  ---- 按标签长短 ----")
    for lo, hi, nm in ((1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"),
                       (11, 20, "11-20"), (21, 999, "21+")):
        sub = [r for r in recs if lo <= len(r["gt"]) <= hi]
        acc3, ex3, n3 = score(sub)
        print(f"    len {nm:<8}{n3:>7,}   char {acc3:7.2%}   exact {ex3:7.2%}")

    # ---------- 和现成 OCR 比 ----------
    #
    # ★★★★★ 这才是要交出去的那个数。上面那些是"我们的模型学得像不像老师",
    #    这一段是"**同样一张带拼音的图, 现成 OCR 读成什么, 我们读成什么**"。
    #
    # ★★ 对照对**它**是偏松的: 真值本来就是 rapidocr 在不带拼音的图上读出来的,
    #    和它自己的错法同源。所以这个比较**对我们不利**, 真实差距只会比报出来的更大,
    #    不会更小。宁可这样 —— 报出去的数要经得起对方复现。
    if a.baseline:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            print("\n  (没装 rapidocr, 跳过对照)")
            return
        ocr = RapidOCR(det_limit_type="min", det_limit_side_len=64)
        print()
        print("  ---- 对照: 同样的图给现成 rapidocr 读 ----")
        for r in recs:
            im = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                r["base"] = ""
                continue
            res, _ = ocr(im, use_cls=False)
            parts = [(min(p[0] for p in b), (t or "").strip())
                     for b, t, _c in (res or []) if (t or "").strip()]
            parts.sort()
            r["base"] = " ".join(t for _, t in parts)

        # ★★★ 第二个对照: 把现成 OCR 输出里的**拼音段直接删掉**再比。
        #
        #   因为一定会有人问(经理大概率会问):
        #       "既然拼音读出来是拉丁字母, 那我把拉丁的去掉不就完了? 干嘛训模型?"
        #
        #   页面级别试过, 不成立(读出的字段数 2.0 -> 2.0, 一点没变好)。
        #   但段级别没试过, 所以这里量一下, 把这条路**用数堵死或者让开**。
        latin_run = re.compile(r"[a-zA-ZÀ-ɏ]{2,}")
        for r in recs:
            r["base_strip"] = re.sub(r"\s{2,}", " ",
                                     latin_run.sub("", r.get("base", ""))).strip()

        def cer(rs, key):
            """返回 (字错率, 全对率)。★ 字错率**可以超过 100%** ——
            读出来的比真值还多(把拼音也读出来了), 编辑距离就会大于真值长度。
            这时候硬报"准确率"会是负数, 反而看不懂, 所以直接报错率。"""
            if not rs:
                return 0.0, 0.0
            ed = sum(edit_distance(r.get(key, ""), r["gt"]) for r in rs)
            tot = sum(max(1, len(r["gt"])) for r in rs)
            ex = sum(1 for r in rs if r.get(key, "") == r["gt"])
            return ed / tot, ex / len(rs)

        groups = [("ALL", recs)]
        if any(r["pin"] is not None for r in recs):
            groups.append(("with pinyin", [r for r in recs if r["pin"] is True]))
            groups.append(("no pinyin", [r for r in recs if r["pin"] is False]))

        print("    字错率 CER, **越低越好**; 超过 100% 表示读出来的比真值还多")
        print(f"    {'group':<13}{'n':>7} {'rapidocr':>10}{'+去拉丁':>10}{'ours':>9}")
        for name, sub in groups:
            if not sub:
                continue
            b, _ = cer(sub, "base")
            s, _ = cer(sub, "base_strip")
            o = sum(edit_distance(r["hyp"], r["gt"]) for r in sub) / max(
                1, sum(max(1, len(r["gt"])) for r in sub))
            print(f"    {name:<13}{len(sub):>7,} {b:>9.1%}{s:>10.1%}{o:>9.1%}")
        print()
        print("    exact match")
        for name, sub in groups:
            if not sub:
                continue
            _, bx = cer(sub, "base")
            _, sx = cer(sub, "base_strip")
            ox = sum(1 for r in sub if r["hyp"] == r["gt"]) / len(sub)
            print(f"    {name:<13}{len(sub):>7,} {bx:>9.1%}{sx:>10.1%}{ox:>9.1%}")

    # ---------- 核对图 ----------
    if a.sheet:
        from make_check_sheet import _cjk_font
        font = _cjk_font(21)
        wrong = [r for r in recs if r["hyp"] != r["gt"]]
        right = [r for r in recs if r["hyp"] == r["gt"]]
        random.seed(3)
        pick = (random.sample(right, min(a.sheet // 2, len(right)))
                + random.sample(wrong, min(a.sheet - a.sheet // 2, len(wrong))))
        random.shuffle(pick)
        ims, W = [], 0
        for r in pick:
            im = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.copyMakeBorder(im, 3, 3, 6, 6, cv2.BORDER_CONSTANT,
                                    value=(215, 215, 215))
            ims.append((im, r))
            W = max(W, im.shape[1])
        TW = 900 if font else 0
        padded = [cv2.copyMakeBorder(m, 0, 0, 0, W - m.shape[1] + TW,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
                  for m, _ in ims]
        out = np.vstack(padded)
        if font:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            dr = ImageDraw.Draw(pil)
            y = 0
            for m, (_im, r) in zip(padded, ims):
                h = m.shape[0]
                good = r["hyp"] == r["gt"]
                dr.line([(W, y), (W, y + h)], fill=(190, 190, 190))
                dr.text((W + 10, y + 2), "真 " + r["gt"][:22], font=font,
                        fill=(60, 60, 60))
                dr.text((W + 430, y + 2), ("出 " if good else "错 ") + r["hyp"][:22],
                        font=font, fill=(0, 130, 0) if good else (190, 0, 0))
                y += h
            out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        dst = a.data / "_check_rec.png"
        cv2.imencode(".png", out)[1].tofile(str(dst))
        print()
        print(f"  sheet -> {dst}")
        print("  (一半是读对的, 一半是读错的, 不是随机抽 —— 随机抽全是对的就看不出毛病)")


if __name__ == "__main__":
    main()
