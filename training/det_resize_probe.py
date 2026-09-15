"""把整页按检测那一步的方式缩小, 量拼音行和正文行之间那道空隙还剩多深。

为什么要这个
------------
检测前会把整页按"最长边 = 上限"缩小。手机截图竖向只剩 0.4 倍左右,
拼音行和正文行之间那一两个像素的空隙会被压到亚像素 —— 而且用的是两点线性采样,
不是盒平均。空隙一旦填平, 检测那一步就没有任何依据把两行分开了。

这个脚本拿真图实测, **不需要模型, 不需要跑 OCR。**

    python det_resize_probe.py --src 目录 --limit 上限值 --sample 80

★ --limit 填检测那一步的最长边上限, 按实际配置填。

怎么量
------
量的是**归一化墨量剖面上空隙的深度**: 空隙处的墨量 ÷ 两侧峰值。
0 = 完全断开, 1 = 完全糊在一起。

★★ 为什么不数"切出来几行": 试过, **那个量法有尺度偏置**。
   行切分的噪点门槛若是固定值, 低分辨率下每行墨量本来就少, 会凭空多切出行来,
   量出来的结论是反的。空隙深度是比值, 没有这个问题。

★★ 诚实说明: 这个脚本量的是**图像层面**的可分性, 不等于 DB 检测网络的行为 ——
   DB 切的是学出来的概率图, 不是墨量投影。空隙在图像上被填平, 是"检测无从分开"
   的**必要条件**, 不是充分条件。要坐实还得拿真模型跑 A/B。
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIFF_THRESHOLD = 28


def align32(value: int) -> int:
    """按 32 对齐, 银行家舍入, 下限 32 —— 和 C# 的 AlignDetectionAxis 逐位一致。

    已用真 .NET 核对过: 429->416, 436->448, 48->64, 80->64, 112->128, 144->128。
    """
    q = value / 32.0
    f = np.floor(q)
    d = q - f
    if d > 0.5:
        r = f + 1
    elif d < 0.5:
        r = f
    else:
        r = f if f % 2 == 0 else f + 1      # 正好 .5 取偶
    return max(int(r) * 32, 32)


def det_resize(img: np.ndarray, limit: int) -> tuple[np.ndarray, float, float]:
    """复刻检测前那次缩放: 最长边压到 limit, 每轴再按 32 对齐, 双线性。

    ★ 按 32 对齐是**分轴**做的, 所以横竖缩放比并不相等(实测差 2.7%~3.2%)。
    """
    H, W = img.shape[:2]
    longest = max(H, W)
    ratio = limit / longest if longest > limit else 1.0
    nw, nh = align32(int(W * ratio)), align32(int(H * ratio))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, nw / W, nh / H


def ink_profile(gray: np.ndarray) -> np.ndarray:
    """每行的墨量, 归一化成"这一行有多少比例是墨"。和局部底色做差, 不分正负。"""
    H, W = gray.shape
    s = cv2.resize(gray, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    k = min(21, min(s.shape[0], s.shape[1]))
    if k % 2 == 0:
        k -= 1
    if k >= 3:
        s = cv2.medianBlur(s, k)
    bg = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
    m = (cv2.absdiff(gray, bg) > DIFF_THRESHOLD).astype(np.float32)
    return m.sum(axis=1) / W                # 归一化 -> 没有尺度偏置


def find_pairs(prof: np.ndarray, min_run: int = 3):
    """找"矮带紧压在高带上方"的配对。返回 (矮带, 高带, 空隙宽度)。"""
    thr = 0.004
    bands, y, H = [], 0, len(prof)
    while y < H:
        if prof[y] <= thr:
            y += 1
            continue
        y0 = y
        while y < H and prof[y] > thr:
            y += 1
        if y - y0 >= min_run:
            bands.append((y0, y))
    pairs = []
    for i in range(len(bands) - 1):
        a, b = bands[i], bands[i + 1]
        ha, hb, gap = a[1] - a[0], b[1] - b[0], b[0] - a[1]
        if ha <= 0.6 * hb and 0 < gap <= 0.6 * hb:
            pairs.append((a, b, gap))
    return pairs


def gap_depth(prof: np.ndarray, a, b) -> float:
    """空隙深度 = 空隙处墨量 ÷ 两侧峰值。0 断开, 1 糊死。"""
    peak = max(prof[a[0]:a[1]].max(), prof[b[0]:b[1]].max())
    if peak <= 0:
        return 1.0
    valley = prof[a[1]:b[0]].max() if b[0] > a[1] else peak
    return float(valley / peak)


def probe(path: Path, limit: int, max_pairs: int):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    pf = ink_profile(img)
    pairs = find_pairs(pf)
    if not pairs:
        return None
    small, sx, sy = det_resize(img, limit)
    ps = ink_profile(small)
    out = []
    for a, b, _ in pairs[:max_pairs]:
        before = gap_depth(pf, a, b)
        a2 = (int(a[0] * sy), max(int(a[1] * sy), int(a[0] * sy) + 1))
        b2 = (int(b[0] * sy), max(int(b[1] * sy), int(b[0] * sy) + 1))
        after = 1.0 if b2[0] <= a2[1] else gap_depth(ps, a2, b2)
        out.append({"file": path.name, "W": img.shape[1], "H": img.shape[0],
                    "sy": round(sy, 4), "before": round(before, 4),
                    "after": round(after, 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--max-pairs", type=int, default=6, help="每张最多看几对")
    ap.add_argument("--max-scale", type=float, default=0.6,
                    help="只统计竖向缩放小于这个值的(本来就不缩的图没有意义)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    files = ([a.src] if a.src.is_file() else
             [p for p in a.src.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
    random.seed(a.seed)
    if a.sample and len(files) > a.sample:
        files = random.sample(files, a.sample)

    rows = []
    for i, p in enumerate(files, 1):
        r = probe(p, a.limit, a.max_pairs)
        if r:
            rows.extend(x for x in r if x["sy"] < a.max_scale)
        if i % 20 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    if not rows:
        print("没有可统计的配对")
        return

    b = np.array([r["before"] for r in rows])
    af = np.array([r["after"] for r in rows])
    pages = len({r["file"] for r in rows})
    print(f"\n{pages} 张图, {len(rows)} 处配对, 竖向缩放中位 "
          f"{np.median([r['sy'] for r in rows]):.3f}, 上限 {a.limit}")
    print()
    print("空隙深度(0 = 完全断开, 1 = 完全糊在一起)")
    for name, arr in (("原分辨率", b), ("缩小后  ", af)):
        print(f"  {name}  中位 {np.median(arr):.3f}   均值 {arr.mean():.3f}   "
              f">0.5 占 {(arr > 0.5).mean():.1%}   糊死占 {(arr >= 0.999).mean():.1%}")
    print()
    print(f"★ 缩放后明显变浅(深度涨 0.05 以上)的: {(af > b + 0.05).mean():.1%}")
    print(f"★ 缩放后彻底糊死的: {(af >= 0.999).mean():.1%}  "
          f"(原分辨率 {(b >= 0.999).mean():.1%})")
    print()
    print("★ 注意: 这量的是图像层面的可分性, 不等于检测网络的行为。")
    print("  空隙被填平是'检测无从分开'的必要条件, 不是充分条件。")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n明细 -> {a.out}")


if __name__ == "__main__":
    main()
