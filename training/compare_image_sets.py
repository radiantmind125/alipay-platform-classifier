"""比两批图在**像素层面**有什么不一样。

为什么要这个
------------
2026-09-15 实测碰到的怪事: 同一个线上交付包,
    D:\\alipay-ai-data\\normal      的图 -> 五个字段全检测到, PASS
    D:\\download2\\pinyin_hits      的图 -> 五个字段一个都检测不到
    D:\\download2\\OtherImages      的图 -> 同样一个都检测不到

两边文件名格式一模一样(`s3_voucher_GWCZ<id>_<时间戳>`), 是同一个来源导出的。
那差别就只能在**图本身**上。这个脚本把能量的都量一遍, 看差在哪。

量这些
------
    尺寸           检测器画布是 864x1536 letterbox, 尺寸不对会被压得很厉害
    每像素字节数   压得狠不狠, 数值低说明重压过
    亮度/对比度    整体偏亮偏暗都可能让检测器失灵
    墨占比         有多少非底色像素, 太少说明可能是空白页或者裁过
    长宽比         手机截图一般 0.45 左右, 偏离说明裁过或者转过

    python compare_image_sets.py --set A=目录1 --set B=目录2 --limit 60
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def measure(p: Path) -> dict | None:
    try:
        buf = np.fromfile(str(p), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 底色取众数附近, 墨 = 和底色差得远的像素
        bg = int(np.bincount(gray.ravel()).argmax())
        ink = float(np.mean(np.abs(gray.astype(np.int16) - bg) > 28))
        return {
            "w": w, "h": h,
            "bytes": int(buf.size),
            "bpp": buf.size / (w * h),          # 每像素字节数
            "aspect": w / h,
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "bg": bg,
            "ink": ink,
            "ext": p.suffix.lower(),
        }
    except Exception:                            # noqa: BLE001
        return None


def summarise(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{name}: 一张都没量到")
        return
    def med(k):
        return st.median(r[k] for r in rows)
    sizes = {}
    for r in rows:
        sizes[f"{r['w']}x{r['h']}"] = sizes.get(f"{r['w']}x{r['h']}", 0) + 1
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:3]
    exts: dict[str, int] = {}
    for r in rows:
        exts[r["ext"]] = exts.get(r["ext"], 0) + 1

    print(f"--- {name}  ({len(rows)} 张) ---")
    print(f"  尺寸中位       {med('w'):.0f} x {med('h'):.0f}")
    print(f"  最常见尺寸     " + ",  ".join(f"{k} ({v})" for k, v in top))
    print(f"  格式           " + ",  ".join(f"{k} {v}" for k, v in sorted(exts.items())))
    print(f"  文件大小中位   {med('bytes')/1024:.0f} KB")
    print(f"  每像素字节     {med('bpp'):.4f}")
    print(f"  长宽比中位     {med('aspect'):.4f}")
    print(f"  灰度均值       {med('mean'):.1f}")
    print(f"  灰度标准差     {med('std'):.1f}")
    print(f"  底色           {med('bg'):.0f}")
    print(f"  墨占比         {med('ink'):.4f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="名字=目录, 至少两个")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    sets = []
    for spec in a.set:
        if "=" not in spec:
            print(f"格式应当是 名字=目录: {spec}")
            return
        name, path = spec.split("=", 1)
        d = Path(path)
        if not d.exists():
            print(f"目录不在: {d}")
            return
        files = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in EXTS][: a.limit]
        rows = [r for r in (measure(p) for p in files) if r]
        sets.append((name, rows))

    print("=" * 70)
    for name, rows in sets:
        summarise(name, rows)

    if len(sets) >= 2:
        print("=" * 70)
        print("★ 差得最多的几项(第一组 对 第二组)")
        print("=" * 70)
        (n1, r1), (n2, r2) = sets[0], sets[1]
        if r1 and r2:
            for k, label in [("w", "宽"), ("h", "高"), ("bytes", "文件大小"),
                             ("bpp", "每像素字节"), ("aspect", "长宽比"),
                             ("mean", "灰度均值"), ("std", "灰度标准差"),
                             ("ink", "墨占比")]:
                a1 = st.median(r[k] for r in r1)
                a2 = st.median(r[k] for r in r2)
                if a2 == 0:
                    continue
                ratio = a1 / a2
                mark = "   ★ 差很多" if (ratio > 1.3 or ratio < 0.77) else ""
                print(f"  {label:<12} {a1:>12.4f}  {a2:>12.4f}   {ratio:>6.2f} 倍{mark}")
        print()
        print("★ 检测器画布是 864x1536 letterbox。图进去之前会先按 max-side-1600 校正,")
        print("  所以尺寸和长宽比差太多的话, 版面在画布上的位置会差很远。")


if __name__ == "__main__":
    main()
