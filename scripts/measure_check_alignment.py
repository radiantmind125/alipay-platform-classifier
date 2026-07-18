"""客观测量“对勾圆圈”与“转账成功”文字的竖直对齐偏移，验证经理的判据。

思路：在表头带里用白色内容掩膜，找到最左的圆圈块（勾）和右侧文字块，分别求竖直质心，
比较二者偏移（按文字高度归一）。若 iOS 与安卓的偏移分布明显不同，则判据成立。
仅用 numpy + PIL。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.cache_regions import load_upright_rgb  # noqa: E402


def measure(img: np.ndarray) -> dict | None:
    h, w = img.shape[:2]
    band = img[int(0.05 * h) : int(0.14 * h), int(0.18 * w) : int(0.82 * w)]
    r, g, b = band[:, :, 0], band[:, :, 1], band[:, :, 2]
    mask = (r > 150) & (g > 150) & (b > 150)  # 蓝底上的白色内容
    if mask.sum() < 50:
        return None
    rowsum = mask.sum(axis=1)
    thr = 0.12 * rowsum.max()
    rows = np.where(rowsum > thr)[0]
    if rows.size == 0:
        return None
    r0, r1 = rows.min(), rows.max() + 1
    hdr = mask[r0:r1]  # 表头这一行的内容
    colsum = hdr.sum(axis=0)
    cols = np.where(colsum > 0.1 * colsum.max())[0]
    if cols.size == 0:
        return None
    # 从最左开始的第一段连续内容列 = 勾圆圈；之后有间隙，再往右是文字。
    first = cols[0]
    end = first
    for c in range(first, hdr.shape[1]):
        if colsum[c] > 0.06 * colsum.max():
            end = c
        elif c - end > max(4, hdr.shape[1] * 0.02):  # 出现足够宽的间隙
            break
    check_cols = (first, end + 1)
    text_start = None
    for c in range(end + 1, hdr.shape[1]):
        if colsum[c] > 0.1 * colsum.max():
            text_start = c
            break
    if text_start is None:
        return None
    text_cols = (text_start, cols[-1] + 1)

    def centroid_y(col0: int, col1: int) -> tuple[float, float]:
        sub = hdr[:, col0:col1]
        ys, _ = np.where(sub)
        if ys.size == 0:
            return float("nan"), 0.0
        return float(ys.mean()), float(ys.max() - ys.min() + 1)

    check_cy, check_h = centroid_y(*check_cols)
    text_cy, text_h = centroid_y(*text_cols)
    if not (text_h > 0):
        return None
    offset_px = check_cy - text_cy               # >0：勾比文字中心偏下
    offset_norm = offset_px / text_h             # 按文字高度归一
    return {"offset_px": round(offset_px, 1), "offset_norm": round(offset_norm, 3),
            "check_h": round(check_h, 1), "text_h": round(text_h, 1)}


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("gold/gold_seed_v1.json"))
    ap.add_argument("--image-root", type=Path, default=Path("../raw_images"))
    args = ap.parse_args(argv)
    labels = json.loads(args.gold.read_text(encoding="utf-8"))["labels"]

    by = {"ios": [], "android": []}
    print(f"{'平台':<8}{'分辨率':<12}{'勾-文字偏移px':<14}{'归一偏移':<10}{'文件'}")
    for row in labels:
        try:
            img = load_upright_rgb(args.image_root / row["file"])
        except Exception:
            continue
        m = measure(img)
        if m is None:
            print(f'{row["platform"]:<8}{row.get("res",""):<12}{"(测量失败)":<14}{"":<10}{row["file"][:40]}')
            continue
        by[row["platform"]].append(m["offset_norm"])
        print(f'{row["platform"]:<8}{row.get("res",""):<12}{m["offset_px"]:<14}{m["offset_norm"]:<10}{row["file"][:40]}')

    print("\n== 按平台汇总（归一偏移，>0=勾偏下，<0=勾偏上）==")
    for k in ("ios", "android"):
        v = np.array(by[k], dtype=float)
        if v.size:
            print(f"  {k:<8} n={v.size:<3} 均值={v.mean():+.3f} 标准差={v.std():.3f} 范围=[{v.min():+.3f},{v.max():+.3f}]")
    if by["ios"] and by["android"]:
        gap = abs(np.mean(by["ios"]) - np.mean(by["android"]))
        pooled = (np.std(by["ios"]) + np.std(by["android"])) / 2 + 1e-6
        print(f"\n  两类均值差={gap:.3f}，合并标准差≈{pooled:.3f}，可分性(差/标准差)≈{gap/pooled:.2f}")
        print("  经验判断：可分性>1.5 才算判据成立；接近 0 则说明勾与文字在两平台都基本对齐。")


if __name__ == "__main__":
    main()
