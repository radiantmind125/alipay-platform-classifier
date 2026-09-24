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

# ★★★★★ 喂给模型的图, 底下要比行边界**再多留 4 像素**。
#
#   为什么会缺字: 行的上下界取的是成员**中位**(不是并集), `订` 这种
#   带勾带捺的字, 笔画伸到中位线底下就被切掉了。
#
#   先拿 `订单号` 那 8 例试的哪一边:
#       原样 0/8   左+8 0/8   左+16 1/8        <- 左边不是原因
#       上+16 0/8  上+32 0/8                   <- 上边更糟, 出的全是上一行的字
#       下+8  8/8  下+16 6/8   四边+16 4/8      <- 是下边, 而且留多了反而坏
#
#   ★★ 上边**千万别放宽** —— 上+32 读出来是 '淘购' '福' '键',
#      那是**上一行的字**。行挨得很紧, 上边界一松就串行。
#
#   ★★★★★ 但 8 是拿**一个字段的 8 个样本**调出来的, 放到 15 个字段上
#      **调过头了**。400 张全量扫了一遍:
#
#        pad       0       4       6       8
#        字段/张  6.96    7.08    7.07    7.05      <- 4 最高
#        汉字/张 125.5   126.1   125.8   125.4      <- 4 最高
#        读不出      6       5       6       6      <- 4 最少
#
#      4 对 0: 订单号+6 全部账单+2 支付时间+2 账单详情+1 转账备注+1 账户余额+1,
#              只有 计入收支 -1
#      8 对 0: 净涨更少, 还多跌了 创建时间 和 付款方式
#
#      ★ **窄样本调出来的值要拿全量复核**, 不然就会像 8 这样顾此失彼。
#
# ★★★★★ 注意这是**推理这一侧的补救**。根子在 y_label 和 y_input 共用同一个
#   底边 lb —— 底边切掉 `订`, 连 label 也是 OCR 出来的 `单号`,
#   于是训练时那一对就是 (缺字的图 -> '单号')。**模型是照着错标签学的。**
#   下一次重新出训练数据时, 底边要一起放宽, 否则标签里那批错的还在。
INPUT_BOT_PAD = 4

# ★★★ 下面两个是**只拿来做 A/B 的开关**, 默认值 = 以前的行为, 一字不差
#   (80 张图逐字节比过 mask / annotation / layout, 20 张比过整条 read_image)。
#   sweep_layout.py 会临时改它们、跑完再还原。**量过之前不要改默认值。**

# 版面那一步(找拼音带、切行)用的掩膜阈值。None = 沿用 DIFF_THRESHOLD(28)。
#
# ★★★★★ 已经量过: **提高这个阈值不是修蓝图拼音检测的办法**, 别再往这上面试。
#   详见 erase_pinyin.text_mask 的说明。要点:
#     - 蓝图顶部拼音认不出, 主因是大小那一关, 不是粘连; 阈值调高 big_h 变小, 反而更糟
#     - 当初那一页是白字描黑边, 黑边离底色约 106, 阈值 100 以下都拆不开
#     - 它还会让整页重新排版: 行、行边界、喂给模型的图、切段全跟着变
#       (切段自己的阈值没变, 但它拿到的那张裁图变了), 底边还会往上缩,
#       等于把 INPUT_BOT_PAD 那次修好的 订 又部分切回去
LAYOUT_MASK_THR: int | None = None

# 版面那一步的注音判法要不要用 local_ratio(和配对的那个大块比, 不和整页比)。
#   False = 以前的行为。拿来端到端试**大小那一关**: 蓝图顶部的字比卡片字大
#   25~40%, 顶部拼音卡在 0.55~0.8 倍 big_h 的空档里。配合
#   erase_pinyin.LOCAL_BIG_MIN 一起试(那个默认 1.3, 而顶部汉字只有
#   big_h 的 1.1~1.25 倍, 正好够不上)。
#   ★ 它只改拼音带, **不改 big**, 所以对版面的扰动比改阈值小得多。
LAYOUT_LOCAL_RATIO: bool = False


def layout(img: np.ndarray) -> list[dict]:
    """把一张图拆成段。返回 [{row, seg, x0, x1, y_label, y_input, has_pinyin}]。

    ★ 这一步不碰任何模型, 纯几何。
    """
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = text_mask(gray, local_background(gray), diff_thr=LAYOUT_MASK_THR)
    _n, _lab, st, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    _labels, kept, _cnt, big = annotation_labels(mask, local_ratio=LAYOUT_LOCAL_RATIO)
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
            # ★ 只放宽**喂给模型**那一份的底边; y_label 不动,
            #   免得和已经出好的训练标签对不上
            out.append({"row": ri, "seg": si, "x0": a, "x1": b,
                        "y_label": (la, lb),
                        "y_input": (ia, min(H, lb + INPUT_BOT_PAD)),
                        "has_pinyin": bool(above)})
    return out


class Recognizer:
    def __init__(self, model: Path | None, onnx: Path | None):
        # ★★★★★ 模型和字表**必须配套**。字表换了(哪怕字数一样), 解出来是
        #   一堆看着像话的乱码, **不报错**。交付出去之后这种错最难查 ——
        #   别人只会觉得"这模型不准", 不会想到是两个文件没配对。
        #   所以在这里**主动核一次**, 对不上就直接抛错, 不让它往下跑。
        self._checked = False
        if onnx:
            import onnxruntime as ort
            cs_path = onnx.parent / "charset.txt"
            if not cs_path.exists():
                raise FileNotFoundError(
                    f"字表不在: {cs_path}\n"
                    f"  模型和 charset.txt 必须放在同一个目录, 而且是同一次训练出来的")
            self.sess = ort.InferenceSession(str(onnx),
                                             providers=["CPUExecutionProvider"])
            chars = [c for c in cs_path.read_text(
                encoding="utf-8").split("\n") if c]
            self.cs = Charset(chars)
            self.kind = "onnx"
            self._where = f"{onnx.name} + {cs_path.name}"
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
            # ★ torch 那一路字表是**存在权重文件里**的, 天然配套, 不会错配
            self._where = f"{model.name} (字表在权重里)"

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
            # ★★★★★ 第一次推理时核一下模型输出的类别数和字表对不对得上。
            #   放在这里而不是加载时, 是因为 onnx 的输出形状可能是动态的,
            #   静态读未必读得到; 真跑一次拿到的形状**一定是准的**。
            #   只核一次, 不影响速度。
            if not self._checked:
                self._checked = True
                n_out = int(logits.shape[-1])
                # ★★ 注意 len(Charset) **本身已经算上 CTC 空白符了**
                #   (Charset.__len__ 返回 len(chars)+1), 而模型是
                #   CRNN(len(cs)) 建的, 所以两者应当**正好相等**, 不要再加 1。
                #   我第一版写成 len(cs)+1, 那样每次推理都会误报, 整条链都跑不动。
                n_need = len(self.cs)
                if n_out != n_need:
                    raise RuntimeError(
                        f"模型和字表对不上, 解出来会是乱码, 已经停下。\n"
                        f"  用的是   {self._where}\n"
                        f"  模型输出 {n_out} 类\n"
                        f"  字表     {len(self.cs.chars)} 个字, 加上 CTC 空白符"
                        f"应当是 {n_need} 类\n"
                        f"  ★ 这两个文件必须是**同一次训练**出来的, 一起拷贝")
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
