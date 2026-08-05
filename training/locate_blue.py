r"""蓝底转账页的金额定位(白字蓝底)—— 补上局部检测器最大的覆盖缺口。

背景: `engine_b_tamper.locate_amount` 找的是**浅底深字**(白底账单详情页那种), 蓝图上完全失效 ——
它看不见白字蓝底的真金额, 反而会误锁到红包促销卡(深字浅底), 造成系统性误杀(实测最高分真图里就有)。
所以蓝图现在被 `--skip-blue` 直接跳过 = 约 36% 的提交没有局部覆盖。本模块就是来补这块的。

思路(和白图那套镜像):
- 蓝图金额是**整页最大的一块白字**, 位置偏上(约图高 14%~18%), 字高约图高 4%;
- 而 "转账成功"、"回首页"、"收款方/付款方式" 这些白字都明显更小(约 1.7%~2%);
- 红包促销卡虽然也是白底, 但**又宽又高**(宽>半屏, 高>12%图高), 用宽高上限就能排掉。
所以: 取上部区域里"近白"的连通块 -> 按字高过滤 -> 留最高的那一档 -> 合成一个框。

先目视验证再上量(强烈建议):
  python training/locate_blue.py --src D:\download\TempFakeImages --viz D:\probe\bluebox --n 20
然后打开 D:\probe\bluebox 里的图, 看红框是不是正好框住金额。框歪了告诉我调参数, 别直接拿去造几千张。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_blue_page(rgb: np.ndarray) -> bool:
    """蓝底转账页判定(与 predict_tiled 的 _is_blue_page 同口径)。"""
    top = rgb[: max(1, rgb.shape[0] // 3)]
    r, g, b = (float(top[:, :, i].mean()) for i in range(3))
    return b > r + 25 and b > g + 15


def locate_amount_blue(rgb: np.ndarray):
    """蓝底页的金额框。返回 (x0, y0, x1, y1, comps) 或 None。

    与 locate_amount 的返回格式保持一致, 方便两边共用下游代码。
    """
    h, w = rgb.shape[:2]
    y0b, y1b = int(h * 0.05), int(h * 0.45)          # 金额在偏上位置, 只在这一带找
    band = rgb[y0b:y1b].astype(np.int16)
    mx, mn = band.max(2), band.min(2)
    white = ((mn > 175) & ((mx - mn) < 45)).astype(np.uint8)   # 近白: 三通道都亮且不偏色

    nlab, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    comps = []
    for i in range(1, nlab):
        x, y, ww, hh, area = stats[i]
        # 字高下限排掉"转账成功/收款方"这些小字; 上限+宽度上限排掉红包卡和大白块
        if hh < 0.025 * h or hh > 0.12 * h or area < 30 or ww > 0.5 * w:
            continue
        comps.append([int(x), int(y + y0b), int(ww), int(hh), int(area)])
    if not comps:
        return None

    maxh = max(c[3] for c in comps)
    amt = [c for c in comps if c[3] >= 0.6 * maxh]    # 只留最高的那一档 = 金额数字
    if len(amt) < 2:                                  # 至少要有两个字符
        return None
    # 同一行: 以最高块的竖直中心为基准, 排掉不在一行上的
    cy = np.median([c[1] + c[3] / 2 for c in amt])
    amt = [c for c in amt if abs((c[1] + c[3] / 2) - cy) < 0.6 * maxh]
    if len(amt) < 2:
        return None

    x0 = min(c[0] for c in amt); x1 = max(c[0] + c[2] for c in amt)
    y0 = min(c[1] for c in amt); y1 = max(c[1] + c[3] for c in amt)
    if (x1 - x0) < 0.08 * w:                          # 太窄不像金额
        return None
    return x0, y0, x1, y1, amt


def locate_amount_auto(rgb: np.ndarray):
    """按页型自动选定位器。返回 (loc, page_type); loc 为 None = 没定位到。

    白图用 engine_b_tamper.locate_amount(浅底深字), 蓝图用本模块的 locate_amount_blue(白字蓝底)。
    两者返回格式一致, 下游可以不关心页型。
    """
    if is_blue_page(rgb):
        return locate_amount_blue(rgb), "blue"
    from engine_b_tamper import locate_amount
    return locate_amount(rgb), "white"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="蓝图金额定位(先目视验证再上量)")
    ap.add_argument("--src", type=Path, required=True, help="图库目录(会自动挑蓝图)")
    ap.add_argument("--viz", type=Path, required=True, help="把画了框的图存到这里, 供目视检查")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [p for p in args.src.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(args.seed).shuffle(files)
    args.viz.mkdir(parents=True, exist_ok=True)

    seen = ok = 0
    for p in files:
        if seen >= args.n:
            break
        try:
            arr = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        except Exception:
            continue
        if not is_blue_page(arr):
            continue                                   # 只看蓝图
        seen += 1
        loc = locate_amount_blue(arr)
        vis = arr.copy()
        if loc:
            x0, y0, x1, y1 = loc[0], loc[1], loc[2], loc[3]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 6)
            ok += 1
            tag = "OK"
        else:
            tag = "FAIL"
        Image.fromarray(vis).save(args.viz / f"{tag}_{seen:03d}_{p.stem[:40]}.jpg", quality=88)

    print(f"看了 {seen} 张蓝图, 定位成功 {ok} 张({ok / max(1, seen) * 100:.0f}%) -> {args.viz}")
    print("**请打开几张 OK_ 开头的图看红框是不是正好框住金额**; 框歪/框到别的东西就告诉我调参数。")
    print("FAIL_ 开头的是没定位到的, 也看两张确认是不是版式特殊。")


if __name__ == "__main__":
    main()
