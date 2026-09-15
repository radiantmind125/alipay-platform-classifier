"""裁一小块出来做 A/B 对照 —— 关键是**长边不超过上限**。

为什么要这个
------------
检测那一步会把整页按"最长边 = 上限"缩小。手机截图是细高条, 最长边是高,
于是竖向只剩 0.4 倍上下 —— 拼音行和正文行之间那 1 个像素的空隙**直接消失**,
两行在送进网络之前就已经粘在一起了。

裁一块**长边本来就小于上限**的区域出来, 缩放比例保持 1.0, 那个空隙就还在。
同一个二进制、同一个密封包, 只换输入, 就能看出"是不是分辨率害的"。

    python crop_for_ab.py --src 一张图 --out 输出目录 --at-y 0.44 --limit 上限值

★ --at-y 是想看的那一行在整页高度里的相对位置(0~1)。订单号一般在 0.4~0.6。
★ 不传 --at-y 就裁正中间那一块。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def ink_span_x(gray: np.ndarray) -> tuple[int, int]:
    """这一条带里, 文字横向从哪到哪。用的是和局部底色的差, 和别处一致。"""
    H, W = gray.shape
    s = cv2.resize(gray, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    k = min(21, min(s.shape[0], s.shape[1]))
    if k % 2 == 0:
        k -= 1
    if k >= 3:
        s = cv2.medianBlur(s, k)
    bg = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = (cv2.absdiff(gray, bg) > 28).astype(np.uint8)
    col = mask.sum(axis=0)
    nz = np.nonzero(col > max(1, int(H * 0.004)))[0]
    if nz.size == 0:
        return 0, W
    return int(nz[0]), int(nz[-1]) + 1


def crop(src: Path, out_dir: Path, at_y: float | None, limit: int,
         height: int | None) -> Path | None:
    data = np.fromfile(str(src), dtype=np.uint8)      # 中文路径
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"读不了: {src}")
        return None
    H, W = img.shape[:2]

    ch = min(limit, H) if height is None else min(height, limit, H)
    cy = H // 2 if at_y is None else int(H * at_y)
    y0 = max(0, min(H - ch, cy - ch // 2))

    # ★ 横向不能一律从 0 开始 —— 值在右边, 从 0 起会把值的尾巴切掉。
    #   先看这条带里文字实际横跨到哪, 再把窗口套在文字上。
    gray = cv2.cvtColor(img[y0:y0 + ch], cv2.COLOR_BGR2GRAY)
    ix0, ix1 = ink_span_x(gray)
    need = ix1 - ix0
    cw = min(limit, W)
    if need <= cw:
        # 文字整体放得下, 把窗口居中套在文字上
        mid = (ix0 + ix1) // 2
        x0 = max(0, min(W - cw, mid - cw // 2))
    else:
        # 放不下, 只能靠右对齐保住值的尾巴(值比标签重要)
        x0 = max(0, min(W - cw, ix1 - cw))
        print(f"★ 这条带里文字横跨 {need} px, 超过上限 {limit} —— "
              f"靠右对齐保住值的尾巴, 左边的标签可能被切")

    piece = img[y0:y0 + ch, x0:x0 + cw]
    ph, pw = piece.shape[:2]
    if max(ph, pw) > limit:
        print(f"★ 裁出来还是超了 {pw}x{ph}, 超过 {limit} —— 调小 --width-frac")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_crop_{x0}_{y0}_{pw}x{ph}.png"
    ok, buf = cv2.imencode(".png", piece)
    if not ok:
        print("编码失败")
        return None
    buf.tofile(str(dst))
    print(f"原图 {W}x{H}  ->  裁出 {pw}x{ph}  (长边 {max(pw, ph)}, 上限 {limit})")
    print(f"  取的是 y={y0}..{y0 + ch}, x={x0}..{x0 + cw}")
    print(f"  -> {dst}")
    print()
    print("★ 长边没超上限, 所以检测那一步的缩放比例是 1.0, 行与行之间的空隙保住了。")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--at-y", type=float, default=None,
                    help="想看的那一行在整页高度里的相对位置 0~1, 订单号一般 0.4~0.6")
    ap.add_argument("--limit", type=int, required=True,
                    help="检测那一步的最长边上限, 按实际配置填")
    ap.add_argument("--height", type=int, default=None,
                    help="裁多高(像素), 不传就用 limit。想只看一两行时传 300 左右")
    a = ap.parse_args()
    if not a.src.exists():
        print(f"图不在: {a.src}")
        return
    crop(a.src, a.out, a.at_y, a.limit, a.height)


if __name__ == "__main__":
    main()
