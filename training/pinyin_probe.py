r"""判断截图里是不是带**拼音标注**。

背景
----
经理 2026-09-11: 有个用户的单子被系统拒了, 原因是"字体变形 没有识别到 真实的订单号"。
实际看图并不是字体变形 —— 是安卓的**拼音标注**功能: 每个汉字上方多出一行小字拼音。
这会把 OCR 的行切分搞乱(每两行真文字之间多插一行小字), 于是订单号读不出来,
系统按"订单号不符合规则"直接拒单。真实交易被误拒。

经理的原话: "帮我写个算法识别图片里是不是有拼音", 并且说
"如果无法用算法识别就不用处理了 我这边再想办法"。

怎么判
------
不做 OCR, 只看**版面结构**, 因为拼音的几何特征非常干净:

  011.jpg(带拼音)的块高分布是**双峰**的:
      小块 峰值在 8 像素   <- 拼音字母
      大块 峰值在 21 像素  <- 汉字
  而且每个拼音块的**正下方紧贴着一个汉字块**, 水平方向还重叠。

普通截图里也有小字(时间戳、说明文字), 但它们**不会系统性地压在大字正上方**。
实测三张对照真图, 这种"小压大"的配对数是 0。

判据: 统计有多少小块满足"正下方紧贴一个大块且水平重叠",
      再除以小块总数, 得到 pinyin_ratio。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def measure(path: str) -> dict | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    if H < 200 or W < 200:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = cv2.threshold(gray, 169, 255, cv2.THRESH_BINARY_INV)[1]
    n, _, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    cs = [(int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3]))
          for i in range(1, n)
          if st[i][4] >= 8 and st[i][3] < 0.05 * H and st[i][2] < 0.4 * W]
    if len(cs) < 60:
        return None

    hs = np.array([c[3] for c in cs], float)
    big_h = float(np.percentile(hs, 75))      # 汉字那一档
    if big_h < 8:
        return None
    # 小块 = 明显比正文矮的; 大块 = 正文高度上下
    small = [c for c in cs if c[3] <= 0.55 * big_h]
    big = [c for c in cs if c[3] >= 0.8 * big_h]
    if len(small) < 20 or len(big) < 20:
        return {"n_comp": len(cs), "big_h": big_h, "n_small": len(small),
                "n_big": len(big), "stacked": 0, "pinyin_ratio": 0.0}

    # 按 x 建桶, 免得 O(N^2)
    buckets: dict[int, list] = {}
    for b in big:
        for k in range(b[0] // 50, (b[0] + b[2]) // 50 + 1):
            buckets.setdefault(k, []).append(b)

    stacked = 0
    for x, y, w, h in small:
        hit = False
        for k in range(x // 50, (x + w) // 50 + 1):
            for bx, by, bw, bh in buckets.get(k, ()):
                gap = by - (y + h)
                if gap < -2 or gap > 0.7 * bh:       # 必须紧贴在上方
                    continue
                ov = min(x + w, bx + bw) - max(x, bx)
                if ov > 0.5 * min(w, bw):            # 水平要压住
                    hit = True
                    break
            if hit:
                break
        stacked += hit

    return {"n_comp": len(cs), "big_h": round(big_h, 1), "n_small": len(small),
            "n_big": len(big), "stacked": stacked,
            "pinyin_ratio": round(stacked / len(small), 4)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="判断截图里有没有拼音标注")
    ap.add_argument("input", type=Path, help="单张图或一个目录")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-ratio", type=float, default=0.30,
                    help="超过这个比例判为带拼音")
    args = ap.parse_args()

    files = [args.input] if args.input.is_file() else [
        p for p in sorted(args.input.rglob("*")) if p.suffix.lower() in _EXTS]
    if args.limit:
        files = files[: args.limit]
    print(f"扫 {len(files):,} 张")

    rows = []
    for i, p in enumerate(files, 1):
        if i % 5000 == 0:
            print(f"  ...{i:,}", flush=True)
        try:
            d = measure(str(p))
        except Exception:
            d = None
        if d:
            d["name"] = p.name
            rows.append(d)

    hits = [r for r in rows if r["pinyin_ratio"] >= args.min_ratio]
    print(f"量到 {len(rows):,} 张, 判为带拼音 {len(hits)} 张 "
          f"({len(hits) / max(1, len(rows)) * 100:.3f}%)")
    if rows:
        v = sorted(r["pinyin_ratio"] for r in rows)
        n = len(v)
        q = lambda x: v[min(n - 1, int(n * x))]
        print(f"  pinyin_ratio  中位 {q(.5):.4f}  p90 {q(.9):.4f}  "
              f"p99 {q(.99):.4f}  p99.9 {q(.999):.4f}  最大 {v[-1]:.4f}")
    for r in sorted(hits, key=lambda r: -r["pinyin_ratio"])[:20]:
        print(f"    {r['pinyin_ratio']:.3f}  小块 {r['n_small']:>4} 压住 {r['stacked']:>4}  {r['name'][:52]}")

    if args.out and rows:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["name", "pinyin_ratio", "stacked",
                                              "n_small", "n_big", "big_h", "n_comp"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
