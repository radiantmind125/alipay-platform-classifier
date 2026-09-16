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


def make_sheet(sets, sheet: Path, per_set: int = 6,
               crop: tuple[float, float] = (0.0, 0.22)) -> None:
    """每组抽几张并排贴出来, 肉眼比**版面**有什么不一样。

    ★★★★★ 为什么必须有这个: 上面那些统计量(尺寸/亮度/墨占比)全是**全局**的,
       两组可能每一项都一样, 但**版面完全不同** —— 比如一组是转账给个人,
       另一组是付款给商家, 字段根本不是同一套。全局统计看不见这个。

    默认裁页面**最上面那一段**(0~22%): 回单类型(转账成功/支付成功/到账成功)
    和金额都在那儿, 一眼就能看出是不是同一种单子。
    """
    cells, labels = [], []
    for name, rows in sets:
        if not rows:
            continue
        step = max(1, len(rows) // per_set)
        picked = [r for r in rows[::step]][:per_set]
        for r in picked:
            p = Path(r["path"])
            if not p.exists():
                continue
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h = img.shape[0]
            c = img[int(h * crop[0]): int(h * crop[1]), :]
            if c.size == 0:
                continue
            c = cv2.resize(c, (460, max(1, int(460 * c.shape[0] / c.shape[1]))))
            cv2.putText(c, name[:10], (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cells.append(c)
            labels.append(name)
    if len(cells) < 2:
        print("图不够, 贴不出来")
        return
    hh = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in cells]
    per_row = per_set
    rows_img = [np.hstack(cells[i:i + per_row])
                for i in range(0, len(cells) - len(cells) % per_row, per_row)]
    if not rows_img:
        return
    grid = np.vstack(rows_img)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", grid)[1].tofile(str(sheet))
    print(f"贴了 {len(cells) - len(cells) % per_row} 张 -> {sheet}")
    print("★ 每一行是一组。看**单子的类型**一不一样(转账成功 / 支付成功 / 到账成功),")
    print("  还有字段是不是同一套(收款方 对 商家)。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="名字=目录, 至少两个")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--sheet", type=Path, default=None,
                    help="★ 每组抽几张并排贴出来 —— 全局统计看不见版面差别")
    ap.add_argument("--crop", type=str, default="0.0,0.22",
                    help="裁页面哪一段, 默认最上面(回单类型和金额在那儿)")
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
        rows = []
        for p in files:
            r = measure(p)
            if r:
                r["path"] = str(p)          # 贴图时要按路径回读原图
                rows.append(r)
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
        print()
        print("★★ 上面这些全是**全局**统计。两组每一项都一样, 也可能版面完全不同")
        print("   (比如一组转账给个人, 一组付款给商家)。要看版面得加 --sheet 贴图。")

    if a.sheet:
        try:
            c1, c2 = (float(x) for x in a.crop.split(","))
        except Exception:                        # noqa: BLE001
            print("--crop 格式应当是 0.0,0.22")
            return
        print()
        make_sheet(sets, a.sheet, crop=(c1, c2))


if __name__ == "__main__":
    main()
