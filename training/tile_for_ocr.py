"""把一张高长图切成几块, 让检测那一步**不再把它缩小**。

为什么这样有用
--------------
检测前有一步"最长边不超过 N"的缩放。手机截图是 1200x2640 这种细高条,
最长边是**高**, 于是整页被压到 0.4 倍 —— 拼音行和正文行之间那一两个像素的
空隙就被压没了, 检测自然分不开两行。

但那个限制看的是**最长边**。所以:

    先把**宽**缩到 N(手机截图的宽本来就比高小得多, 只掉一点),
    再竖着切成高也不超过 N 的几块。
    每块最长边都正好是 N -> 检测那一步的缩放比是 1.0, 不再压了。

实测(55 张真白图, 157 处拼音/正文配对):

    竖向缩放    整页 0.398   ->   切片 0.889     高 2.2 倍
    空隙糊死率  整页 37.6%   ->   切片 6.4%      降到六分之一

★★ 和"擦掉拼音"相比, 这条路**一个像素都不改**, 没有啃坏汉字的风险。
   代价是一张图变成几张, 结果要合并。

★★ 成本: 切片后总像素约是整页方案的 4.7 倍, 检测那一步会慢这么多。
   但拼音图只占图库的 0.68%, **只对这部分走切片**, 总吞吐影响约 3%。
   要不要这么做是经理的决定, 这里只给机制和代价。

    python tile_for_ocr.py --src 图或目录 --out 输出目录 --limit 上限值
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tile(img: np.ndarray, limit: int, overlap: int):
    """返回 [(块图, y起, y止)]。宽先缩到 limit, 再竖切成高<=limit 的块。

    块之间**留重叠**, 免得正好把一行字从中间切断 —— 切断的那一行
    在相邻那一块里是完整的。
    """
    H, W = img.shape[:2]
    if W > limit:
        s = limit / W
        img = cv2.resize(img, (limit, int(round(H * s))),
                         interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]

    if H <= limit:
        return [(img, 0, H)], 1.0

    step = max(1, limit - overlap)
    pieces = []
    y = 0
    while y < H:
        y2 = min(y + limit, H)
        pieces.append((img[y:y2], y, y2))
        if y2 >= H:
            break
        y += step
    return pieces, (limit / W if W else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, required=True,
                    help="检测那一步的最长边上限, 按实际配置填")
    ap.add_argument("--overlap", type=int, default=160,
                    help="块之间的重叠像素, 默认 160(比一行字高得多, 够用)")
    a = ap.parse_args()

    files = ([a.src] if a.src.is_file() else
             sorted(p for p in a.src.iterdir()
                    if p.is_file() and p.suffix.lower() in EXTS))
    a.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for p in files:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H0, W0 = img.shape[:2]
        pieces, scale = tile(img, a.limit, a.overlap)
        for i, (piece, y0, y1) in enumerate(pieces):
            name = f"{p.stem}__t{i:02d}.png"
            ok, buf = cv2.imencode(".png", piece)
            if ok:
                buf.tofile(str(a.out / name))
            manifest.append({"src": p.name, "tile": name, "index": i,
                             "y0": y0, "y1": y1, "scale": round(scale, 6),
                             "src_w": W0, "src_h": H0,
                             "tile_w": piece.shape[1], "tile_h": piece.shape[0]})
        print(f"{p.name}  {W0}x{H0} -> {len(pieces)} 块, 缩放 {scale:.3f}")

    mf = a.out / "_tiles.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"\n共 {len(files)} 张 -> {len(manifest)} 块")
    print(f"清单 -> {mf}")
    print("★ 跑完 OCR 之后按 src 分组合并: 14 个字段里每个取非空的那一个,")
    print("  两块给出不同的非空值时**标出来人工看**, 不要随便选一个。")


if __name__ == "__main__":
    main()
