r"""探查: 线路A 的高分是不是被**完全平坦的小块**带上去的(只读源图)。

线索
----
线路A 榜首那 14 张(钉住阈值的就是它们)里, **12 张含有能量为 0 的 32x32 小块** ——
也就是**像素完全一致、一点纹理都没有**的方块。而 SSP 的取块法恰恰是
"随机取 64 块 -> 排序 -> **只留最简单的那一块**", 于是这些图交给模型的
几乎必然是一块**纯色方块**。

猜想: 模型把"完全没有噪声"读成了"不是真实拍摄/编码出来的" -> 打高分。
若成立, 那么**榜首的误杀不是因为它们像假图, 而是因为它们太干净** ——
而干净恰恰是**无损截图的正常特征**。

★ **但只看榜首不能下结论。** 低分图多半也有大片白边。
   必须**跨分数段对比**: 平坦块在高分图里是不是**明显更多/更平**。
   这个脚本就是做这个对比的。

判读
----
- 高分段的"零能量块数"**显著高于**低分段 -> 猜想成立, 榜首误杀是**平坦块效应**,
  不是池子被污染。修法是取块时跳过退化块, 或把无损截图补进训练集 ——
  **两者都会改变分数分布, 必须重标定**, 所以是 v1 之后的事。
- 两段差不多 -> 猜想不成立, 榜首高分另有原因, 别乱改取块逻辑。

用法
----
  python training/flat_patch_probe.py --csv D:\probe\accept_seeded\_lineA\summary.csv ^
      --search-root D:\probe\gen100k_imgs --per-band 60 --out D:\probe\flatprobe

**只读源图**: 只往 --out 写。
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_BANDS = [(0.0, 0.01, "≈0"), (0.01, 0.1, "0.01~0.1"), (0.1, 0.3, "0.1~0.3"),
          (0.3, 0.6, "0.3~0.6"), (0.6, 0.8, "0.6~0.8"), (0.8, 1.01, "0.8~1.0")]


def flatness(p: Path, P: int = 32) -> tuple[int, int, int] | None:
    """返回 (最简块能量, 零能量块数, 总块数)。与 SSP 的 compute() 同一套差分。"""
    try:
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.int64)
    except Exception:
        return None
    H, W = a.shape[:2]
    if H < P * 2 or W < P * 2:
        return None
    dh = np.abs(a[:, :-1] - a[:, 1:]).sum(2)
    dv = np.abs(a[:-1, :] - a[1:, :]).sum(2)
    nh, nw = (H - 1) // P, (W - 1) // P
    e = (np.add.reduceat(np.add.reduceat(dh[:nh * P, :nw * P], np.arange(0, nh * P, P), 0),
                         np.arange(0, nw * P, P), 1)
         + np.add.reduceat(np.add.reduceat(dv[:nh * P, :nw * P], np.arange(0, nh * P, P), 0),
                           np.arange(0, nw * P, P), 1))
    return int(e.min()), int((e == 0).sum()), int(e.size)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="探查平坦小块与线路A 高分的关系(只读源图)")
    ap.add_argument("--csv", type=Path, required=True, help="线路A summary.csv")
    ap.add_argument("--search-root", type=Path, required=True)
    ap.add_argument("--col", default="final_ai_score")
    ap.add_argument("--per-band", type=int, default=60, help="每个分数段抽多少张")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"扫描 {args.search_root} 建索引...", flush=True)
    idx: dict[str, Path] = {}
    for p in args.search_root.rglob("*"):
        if p.suffix.lower() in _EXTS and p.is_file():
            idx.setdefault(p.name, p)
    print(f"  {len(idx):,} 个文件名")

    buckets: dict[str, list[tuple[str, float]]] = {b[2]: [] for b in _BANDS}
    with open(args.csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            nm = (r.get("image_name") or "").strip()
            v = (r.get(args.col) or "").strip()
            if not nm or not v:
                continue
            if "scores_json" in r and (r.get("scores_json") or "").strip() in ("", "{}"):
                continue
            try:
                s = float(v)
            except ValueError:
                continue
            for lo, hi, lbl in _BANDS:
                if lo <= s < hi:
                    buckets[lbl].append((nm, s))
                    break

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\n{'分数段':>10s}{'池内张数':>10s}{'抽样':>7s}{'最简块能量中位':>15s}"
          f"{'零能量块数中位':>15s}{'有零能量块占比':>15s}")
    print("-" * 74)
    for lo, hi, lbl in _BANDS:
        pool = buckets[lbl]
        if not pool:
            continue
        pick = rng.sample(pool, min(args.per_band, len(pool)))
        mins, zeros, hasz = [], [], 0
        for nm, s in pick:
            p = idx.get(nm)
            if p is None:
                continue
            f = flatness(p)
            if f is None:
                continue
            mn, nz, tot = f
            mins.append(mn); zeros.append(nz); hasz += (nz > 0)
            rows.append({"image_name": nm, "score": f"{s:.6f}", "band": lbl,
                         "最简块能量": mn, "零能量块数": nz, "总块数": tot})
        if not mins:
            continue
        print(f"{lbl:>10s}{len(pool):>10,d}{len(mins):>7d}{st.median(mins):>15,.0f}"
              f"{st.median(zeros):>15,.0f}{hasz / len(mins) * 100:>14.1f}%")

    sp = args.out / "flatness.csv"
    if rows:
        with open(sp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\n-> {sp}")
    print("\n判读: 高分段的'零能量块数'**显著高于**低分段 -> 榜首误杀是**平坦块效应**, 不是池子被污染;")
    print("      修法(取块跳过退化块 / 无损截图补进训练)**都会改分数分布, 必须重标定**, 属 v1 之后。")
    print("      两段差不多 -> 猜想不成立, **别动取块逻辑**。")


if __name__ == "__main__":
    main()
