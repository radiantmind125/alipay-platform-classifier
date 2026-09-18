"""带拼音的回单截图 -> 文字。一个**自带检测**的完整识别管线。

为什么是"自带检测"而不是只给一个识别模型
----------------------------------------
经理说的是"估计要单独训练一个"。单给一个识别模型没法用 ——
调用方还得自己有办法把文本行框出来, 而现成检测器在拼音页上框出来的东西
恰恰是不对的(它会把拼音也当成一行文本去框)。

★★★ 我们这边不用检测模型: 行和段是**按版面几何算出来的**,
   而且这套几何已经在 13.4 万行上验过(切段人工核 16/16 对得上)。
   所以整条链是自洽的, 不依赖任何第三方检测。

一张图是怎么变成文字的
----------------------
    1. 取字掩膜, 逐连通块判哪些是注音字        erase_pinyin.annotation_labels
    2. 注音块聚成**拼音带**                    make_pinyin_pairs._pinyin_bands
    3. 排除拼音带之后按纵向重叠聚出**正文行**  make_pinyin_pairs._rows
    4. 每行两个 y 区间:
         不含拼音的(算段用)  含拼音的(送模型)
    5. 在**不含拼音**的那张上按空白切**段**    split_segments.segments
       ★ 必须在不含拼音的那张上切 —— 拼音比汉字宽, 会把该断的地方连起来
    6. 段的 x 区间套到**含拼音**的那张上, 批量送识别器

★ 和训练时走的是同一套几何, 一行代码都没换 —— 训练怎么裁的, 上线就怎么裁。

用法
----
    python pinyin_ocr.py --src 图或目录 --model rec_best.pt
    python pinyin_ocr.py --src 一张图 --model rec_best.pt --sheet out.png
    python pinyin_ocr.py --src 目录 --onnx rec.onnx --json out.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import (EXTS, annotation_labels, local_background,  # noqa: E402
                          text_mask)
from make_pinyin_pairs import (FALLBACK_SHIFT_RATIO, LABEL_PAD,  # noqa: E402
                               MIN_ROW_H, _pinyin_bands, _rows)
from split_segments import MIN_SEG_W, PAD, segments  # noqa: E402
from train_rec import IMG_H, MIN_W, MAX_W, Charset, prep  # noqa: E402


def layout(img: np.ndarray) -> list[dict]:
    """把一张图拆成段。返回 [{row, seg, x0, x1, y_label, y_input, has_pinyin}]。

    ★ 这一步不碰任何模型, 纯几何。
    """
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = text_mask(gray, local_background(gray))
    _n, _lab, st, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    _labels, kept, _cnt, big = annotation_labels(mask)
    bands = _pinyin_bands(kept, st, big)
    rows = _rows(big, bands)
    if not rows:
        return []
    char_h = float(np.median([b - a for a, b in rows]))
    fallback = int(round(FALLBACK_SHIFT_RATIO * char_h))

    out = []
    for ri, (y0, y1) in enumerate(rows):
        above = [(x, y) for x, y in bands if y <= y0 + 2 and (y0 - y) < (y1 - y0)]
        la = max(0, y0 - LABEL_PAD)
        if above:
            la = min(y0, max(la, max(y for _, y in above) + 1))
        lb = min(H, y1 + LABEL_PAD)
        if lb - la < MIN_ROW_H:
            continue
        ia = max(0, (min(x for x, _ in above) - 1) if above else (y0 - fallback))

        lab_crop = img[la:lb, :]
        for si, (x0, x1) in enumerate(segments(lab_crop)):
            a = max(0, x0 - PAD)
            b = min(img.shape[1], x1 + PAD + 1)
            if b - a < MIN_SEG_W:
                continue
            out.append({"row": ri, "seg": si, "x0": a, "x1": b,
                        "y_label": (la, lb), "y_input": (ia, lb),
                        "has_pinyin": bool(above)})
    return out


class Recognizer:
    def __init__(self, model: Path | None, onnx: Path | None):
        if onnx:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(onnx),
                                             providers=["CPUExecutionProvider"])
            chars = [c for c in (onnx.parent / "charset.txt").read_text(
                encoding="utf-8").split("\n") if c]
            self.cs = Charset(chars)
            self.kind = "onnx"
        else:
            import torch
            from train_rec import CRNN
            ck = torch.load(model, map_location="cpu")
            self.cs = Charset(ck["chars"])
            self.net = CRNN(len(self.cs))
            self.net.load_state_dict(ck["model"])
            self.net.eval()
            self.torch = torch
            self.kind = "torch"

    def __call__(self, crops: list[np.ndarray], batch: int = 32) -> list[str]:
        texts: list[str] = []
        # ★ 按宽度排一下再成批, 补的空白少很多(训练时也是这么干的)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
        res: dict[int, str] = {}
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xs = [prep(crops[j]) for j in idx]
            W = max(x.shape[2] for x in xs)
            x = np.empty((len(xs), 3, IMG_H, W), np.float32)
            widths = []
            for k, im in enumerate(xs):
                w = im.shape[2]
                x[k, :, :, :w] = im
                if w < W:
                    x[k, :, :, w:] = im[:, :, -1:]
                widths.append(w)
            if self.kind == "onnx":
                logits = self.sess.run(None, {"x": x})[0]
            else:
                with self.torch.no_grad():
                    logits = self.net(self.torch.from_numpy(x)).numpy()
            ids = logits.argmax(2)
            for k, j in enumerate(idx):
                res[j] = self.cs.decode(ids[k][: max(1, widths[k] // 4)])
        for i in range(len(crops)):
            texts.append(res.get(i, ""))
        return texts


def read_image(path: Path, rec: Recognizer) -> list[dict]:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    segs = layout(img)
    if not segs:
        return []
    crops = [img[s["y_input"][0]:s["y_input"][1], s["x0"]:s["x1"]] for s in segs]
    for s, t in zip(segs, rec(crops)):
        s["text"] = t
        s["y0"], s["y1"] = s["y_input"]
        s.pop("y_label")
        s.pop("y_input")
    # ★★ 读出来是空的那些**保留**, 不在这里悄悄扔掉。
    #    扔掉的话, 模型要是坏了(比如全输出空), 打出来是"0 段", 看着像版面没切出来 ——
    #    实际上版面好好的, 是识别器哑了。两种毛病得分得开。
    #    调用方要过滤自己过滤, 这里只管**如实报**。
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None,
                    help="把结果画到图上(只对单张图有意义)")
    a = ap.parse_args()
    if not a.model and not a.onnx:
        print("要给 --model 或 --onnx")
        return

    rec = Recognizer(a.model, a.onnx)
    files = ([a.src] if a.src.is_file()
             else sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS))
    if a.limit:
        files = files[: a.limit]
    print(f"  {len(files)} images,  recognizer={rec.kind}")

    t0 = time.time()
    n_seg = n_txt = 0
    fh = a.json.open("w", encoding="utf-8") if a.json else None
    for i, p in enumerate(files, 1):
        segs = read_image(p, rec)
        n_seg += len(segs)
        n_txt += sum(1 for s in segs if s["text"].strip())
        if fh:
            fh.write(json.dumps({"file": p.name, "segments": segs},
                                ensure_ascii=False) + "\n")
        if i % 50 == 0:
            print(f"    {i}/{len(files)}   {n_seg:,} segs   "
                  f"{i/(time.time()-t0):.1f} img/s", flush=True)
    if fh:
        fh.close()
    el = time.time() - t0
    print()
    print("=" * 52)
    print(f"  images {len(files):,}   segments {n_seg:,}   "
          f"{n_seg/max(1,len(files)):.1f} per image")
    # ★ 切出来多少段 和 读出文字的有多少, 分开报 —— 版面坏了和识别器哑了是两回事
    print(f"  with text {n_txt:,}  ({n_txt/max(1,n_seg):.0%})"
          f"{'   <-- 识别器没出东西, 但版面是好的' if n_seg and not n_txt else ''}")
    print(f"  {el:.1f}s total   {len(files)/max(el,1e-9):.2f} img/s")
    print("=" * 52)
    if a.json:
        print(f"  -> {a.json}")

    if a.sheet and len(files) == 1:
        from make_check_sheet import _cjk_font
        img = cv2.imdecode(np.fromfile(str(files[0]), np.uint8), cv2.IMREAD_COLOR)
        segs = read_image(files[0], rec)
        canvas = np.full((img.shape[0], img.shape[1] + 620, 3), 255, np.uint8)
        canvas[:, : img.shape[1]] = img
        font = _cjk_font(20)
        if font:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            dr = ImageDraw.Draw(pil)
            # ★ 一行里的几段**拼成一条**再画, 不能各画各的 ——
            #   同一行的段 y0 相同, 各画各的会叠在一起, 画出来是
            #       账单详情 + 全部账单  ->  账部账情
            #   看着像模型读错了, 其实是这张图画坏了。
            by_row: dict[int, list] = {}
            for s in segs:
                dr.rectangle([s["x0"], s["y0"], s["x1"], s["y1"]],
                             outline=(0, 140, 255), width=2)
                by_row.setdefault(s["row"], []).append(s)
            for _ri, group in sorted(by_row.items()):
                group.sort(key=lambda s: s["x0"])
                line = "  |  ".join(s["text"] for s in group if s["text"].strip())
                if line:
                    dr.text((img.shape[1] + 12, group[0]["y0"]), line[:38],
                            font=font, fill=(160, 0, 0))
            canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        cv2.imencode(".png", canvas)[1].tofile(str(a.sheet))
        print(f"  sheet -> {a.sheet}")


if __name__ == "__main__":
    main()
