"""不用 OCR, 数出白图上最长那串数字有几位。

为什么要这个
------------
识别流程里有一个**单行文本的长度上限**参数。订单号要是比那个上限长,
那**每一张**都会被截, 跟拼音无关 —— 这是最便宜也最容易被跳过的一种病因。
但"订单号有多少位"光靠肉眼只看得了几张, 样本太小, 撑不起这个判断。

★ 上限具体是多少, 用 `--max-len` 传进来(按你那边的实际配置填),
  脚本里不写死。

怎么数
------
数字在二值化之后是**一个个独立的连通块**, 所以不用认字, 数块就行:
  1. 和局部底色做差拿到文字掩膜(和 PinyinCheck / TextRegion 同一套, 不分正负)
  2. 连通块
  3. 按 y 重叠聚成行
  4. 每行只看**右半边**(值在右, 标签在左), 数等高等宽的块
  5. 全页取最长的那一串

★ 会**低估**: 两个数字粘在一起就少数一个。所以输出的是下界。
★ 商家订单号(23 位)也很长, 所以同一页可能有两串长的, 取最长的那个。

    python count_digits.py --src <白图目录> --out counts.csv --limit 300
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIFF_THRESHOLD = 28          # 和 PinyinCheck / TextRegion 保持一致


def text_mask(gray: np.ndarray) -> np.ndarray:
    """和局部底色的绝对差 -> 文字掩膜。不分正负, 蓝底白字也切得出来。"""
    H, W = gray.shape
    s = cv2.resize(gray, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    k = min(21, min(s.shape[0], s.shape[1]))
    if k % 2 == 0:
        k -= 1
    if k >= 3:
        s = cv2.medianBlur(s, k)
    bg = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
    return ((cv2.absdiff(gray, bg) > DIFF_THRESHOLD).astype(np.uint8)) * 255


def longest_digit_run(path: Path) -> tuple[int, int, float] | None:
    """返回(最长串的块数, 那一行的 y, 块高中位数)。读不了返回 None。"""
    data = np.fromfile(str(path), dtype=np.uint8)      # 中文路径
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    H, W = img.shape
    if H < 64 or W < 64:
        return None

    mask = text_mask(img)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # 太小的是噪点, 太大的是图标/卡片边框
        if h < 8 or h > H * 0.05 or w > W * 0.25 or area < 12:
            continue
        # 值在右半边, 标签在左边
        if x < W * 0.22:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return None

    # 按 y 重叠聚行
    boxes.sort(key=lambda b: b[1])
    rows: list[list[tuple[int, int, int, int]]] = []
    for b in boxes:
        placed = False
        for r in rows:
            my = float(np.median([t[1] + t[3] / 2 for t in r]))
            mh = float(np.median([t[3] for t in r]))
            if abs((b[1] + b[3] / 2) - my) <= 0.6 * mh:
                r.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    best = (0, 0, 0.0)
    for r in rows:
        if len(r) < 8:                       # 短行不可能是订单号
            continue
        hs = np.array([t[3] for t in r], dtype=float)
        mh = float(np.median(hs))
        # 只留和中位高度接近的(滤掉混进来的标点/图标)
        same = [t for t, h in zip(r, hs) if 0.6 * mh <= h <= 1.5 * mh]
        if len(same) < 8:
            continue
        # 按 x 排序, 找间距均匀的最长连续段
        same.sort(key=lambda t: t[0])
        gaps = [same[i + 1][0] - (same[i][0] + same[i][2]) for i in range(len(same) - 1)]
        if not gaps:
            continue
        mg = float(np.median(gaps))
        lim = max(mg * 2.5, mh * 0.8)

        # ★★ 直接数块, 得到的是**下界**。
        #   试过按宽度拆粘连块(宽 40 / 单字 18 = 2 个), **不可靠**:
        #   数字"1"只有"0"的一半宽, 单字宽本身就不是个稳定的数。
        #   实测真值 32 位的那张被拆成了 36 位。
        #   而我们要回答的只是"是不是超过 25", 下界就够了, 而且不会把话说过头。
        run, longest = 1, 1
        for g in gaps:
            if g <= lim:                     # 间距突然变大 = 换字段了, 断开
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        if longest > best[0]:
            best = (longest, int(np.median([t[1] for t in same])), mh)
    return best if best[0] else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default=None, help="只跑文件名含这个串的")
    ap.add_argument("--max-len", type=int, required=True,
                    help="识别流程的单行长度上限, 按实际配置填。超过它的会被标出来")
    a = ap.parse_args()

    files = sorted(p for p in a.src.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    if a.only:
        files = [p for p in files if a.only in p.name]
    if a.limit:
        files = files[: a.limit]

    rows, counts = [], Counter()
    bad = 0
    for i, p in enumerate(files, 1):
        r = longest_digit_run(p)
        if r is None:
            bad += 1
            rows.append((p.name, "", "", ""))
            continue
        run, y, mh = r
        counts[run] += 1
        rows.append((p.name, run, y, f"{mh:.1f}"))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["文件名", "最长串块数", "行y", "块高中位"])
        w.writerows(rows)

    print(f"\n共 {len(files)} 张, 数不出来的 {bad}")
    print("\n最长数字串的长度分布(是**下界**, 粘连会少数):")
    over = 0
    for k in sorted(counts):
        flag = ""
        if k > a.max_len:
            over += counts[k]
            flag = "  <- 超过上限"
        print(f"  {k:>4} 位  {counts[k]:>5} 张  {'#' * min(40, counts[k])}{flag}")
    tot = sum(counts.values())
    if tot:
        print(f"\n★ 超过上限({a.max_len})的: {over}/{tot} = {over/tot:.1%}")
    print(f"明细 -> {a.out}")


if __name__ == "__main__":
    main()
