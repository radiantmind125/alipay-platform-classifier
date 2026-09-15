"""把支付宝截图分成**蓝图**(转账成功页)和**白图**(账单详情页)。

为什么要这个
------------
订单号只长在**白图**(账单详情)上。蓝图是转账完成那一屏, 上面根本没有订单号字段。
所以拿"拼音图"整批去量订单号读得全不全, 会混进大量**本来就没有订单号**的蓝图,
空值会被误读成 OCR 失败。

判法: 蓝图顶部有一大片高饱和的蓝。只看**上半屏**, 避开底部的广告位和推荐卡
(那些在白图上也是彩色的)。

    python split_blue_white.py --src <目录> --out split.csv
    python split_blue_white.py --src <目录> --out split.csv --link-white <白图目录>
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 支付宝那个蓝大概 H=210 度上下(OpenCV 的 H 是 0~179, 所以 105 左右)
HUE_LO, HUE_HI = 95, 120
SAT_MIN = 90       # 够不够"蓝", 不是灰
VAL_MIN = 80       # 不算太暗
BLUE_MIN = 0.18    # 上半屏里蓝占比超过这个就算蓝图


def blue_fraction(path: Path) -> float | None:
    """上半屏里"支付宝蓝"占的比例。读不了返回 None。"""
    data = np.fromfile(str(path), dtype=np.uint8)      # 中文路径用 imread 会失败
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h = img.shape[0]
    top = img[: max(1, h // 2)]                        # 只看上半屏
    # 缩一下, 纯粹为了快; 比例不受影响
    if top.shape[1] > 400:
        s = 400 / top.shape[1]
        top = cv2.resize(top, (400, max(1, int(top.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= HUE_LO) & (hsv[:, :, 0] <= HUE_HI)
         & (hsv[:, :, 1] >= SAT_MIN) & (hsv[:, :, 2] >= VAL_MIN))
    return float(m.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=BLUE_MIN)
    ap.add_argument("--link-white", type=Path, default=None,
                    help="把判成白图的硬链接到这个目录(同盘才行), 给后续诊断当输入")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(p for p in a.src.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    if a.limit:
        files = files[: a.limit]

    if a.link_white:
        a.link_white.mkdir(parents=True, exist_ok=True)

    rows = []
    n_blue = n_white = n_bad = 0
    for i, p in enumerate(files, 1):
        f = blue_fraction(p)
        if f is None:
            n_bad += 1
            rows.append((p.name, "", "坏图"))
            continue
        kind = "蓝图" if f >= a.threshold else "白图"
        if kind == "蓝图":
            n_blue += 1
        else:
            n_white += 1
            if a.link_white:
                dst = a.link_white / p.name
                if not dst.exists():
                    try:
                        os.link(p, dst)                 # 硬链接, 不占额外空间
                    except OSError:
                        import shutil
                        shutil.copy2(p, dst)            # 跨盘就退回拷贝
        rows.append((p.name, f"{f:.4f}", kind))
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["文件名", "蓝占比", "判定"])
        w.writerows(rows)

    total = n_blue + n_white
    print(f"\n总计 {len(files)} 张, 坏图 {n_bad}")
    if total:
        print(f"  蓝图 {n_blue}  ({n_blue / total:.1%})  <- 没有订单号字段")
        print(f"  白图 {n_white}  ({n_white / total:.1%})  <- 订单号长在这上面")
    print(f"明细 -> {a.out}")
    if a.link_white:
        print(f"白图已归集 -> {a.link_white}")


if __name__ == "__main__":
    main()
