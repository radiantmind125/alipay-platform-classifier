r"""YenCheck v2(2026-10-25)和 AmountFontCheck(2026-10-26)在普查清单上过一遍: 量 + 按 C# 的口径判, 看真图会不会被报。

为什么要跑
----------
YenCheck v2 加了两样东西: 不传机型时按 iPhone 原生截图尺寸认苹果, 以及苹果上的"¥高比"判据。
本机只有 45 张 iPhone 尺寸的苹果蓝图真图, ¥高比这条要在几千张上过一遍才放心。
AmountFontCheck(金额数字字体: 0 的宽 / 小数点的高)本机标定的 2,273 页里安卓占了绝大多数, 苹果也要多过一遍。
普查清单(page_type.csv, 分图程序的输出)里每张图都有页型(B1 转账成功 / B2 到账成功 / B3 支付成功 ...),
正好按页型分开看。

用法(服务器)
------------
  python -u training\yen_census.py E:\_move_logs\pagetype_census_v4\page_type.csv ^
         E:\_move_logs\pagetype_pinyin_v4\page_type.csv --out E:\_move_logs\check_census_1026 --procs 8

  输出(都在 --out 下): yen_census.csv 明细(两个检查的量和结论都在里面) / summary.txt 汇总 /
  flagged_sheet.png ¥ 报出的金额行拼图 / flagged_font_sheet.png 字体报出的拼图 / _DONE(跑完才写)。

★ 只读图片, 只往 --out 写。清单里路径不存在的图记成"找不到", 不中断。
★ 判定只调 yen_scan.judge() 和 amount_font_scan.judge(), 那是两个 C# 类的逐条翻版, 不在这里另写一套。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

# ★ 服务器 24 核, 每个进程再开多线程会顶到提交内存上限(2026-10 踩过), 先把线程数压成 1
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import amount_font_scan  # noqa: E402
import yen_scan  # noqa: E402

COLS = ["path", "cls", "W", "H", "iphone_size", "verdict", "platform", "reason", "skip",
        "digit_h", "n_digit", "d_stroke", "y_stem", "stem_ratio", "yen_h_ratio", "bar_ratio",
        "bot_gap", "stroke_norm",
        "font_verdict", "font_reason", "font_skip", "font_digit_h", "n_zero", "zero_w", "zero_hole_w",
        "dot_h", "dot_w"]


def _init():
    cv2.setNumThreads(1)


def work(item):
    path, cls = item
    row = {"path": path, "cls": cls}
    if not os.path.exists(path):
        row.update(skip="找不到", verdict=0, platform="", reason="找不到",
                   font_verdict=0, font_reason="找不到", font_skip="找不到")
        return row
    # ---- 金额字体(和机型无关) ----
    try:
        fm, fwhy = amount_font_scan.measure(path)
    except Exception as e:  # noqa: BLE001
        fm, fwhy = None, f"出错 {type(e).__name__}"
    fv, freason = amount_font_scan.judge(fm)
    row.update(font_verdict=fv, font_reason=freason, font_skip=fwhy or "")
    if fm:
        row.update(font_digit_h=fm["digit_h"], n_zero=fm["n_zero"], zero_w=fm["zero_w"],
                   zero_hole_w=fm["zero_hole_w"], dot_h=fm["dot_h"], dot_w=fm["dot_w"])
    try:
        m, why = yen_scan.measure(path)
    except Exception as e:  # noqa: BLE001
        m, why = None, f"出错 {type(e).__name__}"
    if m is None:
        row["skip"] = why
        try:
            from PIL import Image
            with Image.open(path) as im:
                row["W"], row["H"] = im.size
        except Exception:  # noqa: BLE001
            pass
    else:
        row.update(m)
    v, plat, reason = yen_scan.judge(m)
    row.update(verdict=v, platform=plat or "", reason=reason)
    if "W" in row and "H" in row:
        row["iphone_size"] = int((row["W"], row["H"]) in yen_scan.IPHONE_SIZES)
    return row


def _amount_tile(path, label):
    """报出来的图: 裁金额行, 缩到统一高度, 左上角写编号。裁不出来就返回 None。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        loc = yen_scan.locate_amount_blue(rgb)
        if not loc:
            return None
        x0, y0, x1, y1 = loc[:4]
        pad = max(4, int((y1 - y0) * 0.3))
        padx = max(pad, int((x1 - x0) * 0.35))
        H, W = rgb.shape[:2]
        crop = rgb[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - padx):min(W, x1 + pad)]
        th = 90
        tile = cv2.resize(crop, (max(1, int(crop.shape[1] * th / crop.shape[0])), th),
                          interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        cv2.putText(tile, label, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return tile
    except Exception:  # noqa: BLE001
        return None


def _sheet(rows, out_png, limit=80):
    tiles = []
    for i, r in enumerate(rows[:limit], 1):
        t = _amount_tile(r["path"], str(i))
        if t is not None:
            tiles.append(t)
    if not tiles:
        return 0
    wmax = 900
    lines, cur, curw = [], [], 0
    for t in tiles:
        t = t[:, :wmax]
        if cur and curw + t.shape[1] > wmax:
            lines.append(cur)
            cur, curw = [], 0
        cur.append(t)
        curw += t.shape[1] + 6
    if cur:
        lines.append(cur)
    rows_img = []
    for ln in lines:
        w = sum(t.shape[1] + 6 for t in ln)
        canvas = np.full((96, max(w, wmax), 3), 255, np.uint8)
        x = 0
        for t in ln:
            canvas[3:3 + t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + 6
        rows_img.append(canvas[:, :wmax])
    sheet = np.vstack(rows_img)
    ok, buf = cv2.imencode(".png", sheet)
    if ok:
        buf.tofile(str(out_png))      # cv2.imwrite 遇到中文路径会静默失败
    return len(tiles)


def _q(v, qs=(0, 0.1, 1, 50, 99, 99.9, 100)):
    v = np.asarray(v, float)
    if not len(v):
        return "(无)"
    return "  ".join(f"p{q:g}={np.percentile(v, q):.4f}" for q in qs)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="YenCheck v2 + AmountFontCheck 普查: 真图会不会被报")
    ap.add_argument("lists", nargs="+", type=Path, help="page_type.csv(分图程序的输出), 可以给多个")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--all-types", action="store_true", help="不只看蓝图(B 开头), 清单里的都过一遍")
    a = ap.parse_args()

    items, seen = [], set()
    for f in a.lists:
        if not f.exists():
            raise SystemExit(f"!! 清单不存在: {f}")
        with f.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                p, cls = r.get("path", ""), r.get("cls", "")
                if not p or p in seen:
                    continue
                if not a.all_types and not cls.startswith("B"):
                    continue
                seen.add(p)
                items.append((p, cls))
    print(f"清单 {len(a.lists)} 份, 要过的图 {len(items):,} 张", flush=True)
    if not items:
        raise SystemExit("!! 清单里一张蓝图都没有")

    a.out.mkdir(parents=True, exist_ok=True)
    done_flag = a.out / "_DONE"
    if done_flag.exists():
        done_flag.unlink()
    tmp = a.out / "yen_census.csv.tmp"
    rows = []
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh, Pool(a.procs, initializer=_init) as pool:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(work, items, chunksize=8), 1):
            w.writerow(row)
            rows.append(row)
            if i % 1000 == 0:
                print(f"  {i:,}/{len(items):,}", flush=True)
    os.replace(tmp, a.out / "yen_census.csv")

    lines = []
    say = lines.append
    miss = sum(1 for r in rows if r.get("skip") == "找不到")
    say(f"共 {len(rows):,} 张, 找不到 {miss:,} 张")
    say("\n== 按页型 x 是否 iPhone 原生尺寸: 量到 / 正常 / 可疑 / 判不了")
    tab = collections.defaultdict(collections.Counter)
    for r in rows:
        key = (r["cls"], "iPhone尺寸" if r.get("iphone_size") == 1 else "别的尺寸")
        tab[key]["共"] += 1
        tab[key]["量到"] += int(not r.get("skip"))
        tab[key][{0: "判不了", 1: "正常", 2: "可疑"}[r["verdict"]]] += 1
    for key in sorted(tab):
        c = tab[key]
        say(f"  {key[0]:<4}{key[1]:<8} 共 {c['共']:>6,}  量到 {c['量到']:>6,}  正常 {c['正常']:>6,}  "
            f"可疑 {c['可疑']:>4,}  判不了 {c['判不了']:>6,}")

    iph = [r for r in rows if r.get("iphone_size") == 1 and not r.get("skip")]
    say(f"\n== iPhone 原生尺寸、量到的蓝图 {len(iph):,} 张, 各项分布(按页型)")
    for cls in sorted({r["cls"] for r in iph}):
        g = [r for r in iph if r["cls"] == cls]
        say(f"  [{cls}] n={len(g)}")
        for k in ("stem_ratio", "yen_h_ratio", "stroke_norm", "digit_h"):
            say(f"     {k:<12} {_q([r[k] for r in g])}")
    if iph:
        yh = np.array([r["yen_h_ratio"] for r in iph if r["digit_h"] >= yen_scan.MIN_DIGIT_H_FOR_YEN_H])
        st = np.array([r["stem_ratio"] for r in iph if r["digit_h"] >= yen_scan.MIN_DIGIT_H])
        say(f"\n  ¥高比 < {yen_scan.APPLE_YEN_H_LOW} 的: {int((yh < yen_scan.APPLE_YEN_H_LOW).sum())} / {len(yh)}"
            f"    > {yen_scan.APPLE_YEN_H_HIGH} 的: {int((yh > yen_scan.APPLE_YEN_H_HIGH).sum())}")
        for hi in (0.62, 0.63, 0.65):
            say(f"  竖笔比 > {hi} 的: {int((st > hi).sum())} / {len(st)}")

    flagged = sorted([r for r in rows if r["verdict"] == 2], key=lambda r: (r["cls"], -r.get("stem_ratio", 0)))
    say(f"\n== 报出来的(可疑) {len(flagged):,} 张(拼图里的编号就是这里的序号)")
    for i, r in enumerate(flagged, 1):
        say(f"  {i:>3} {r['cls']:<3} {r['W']}x{r['H']} 竖笔 {r['stem_ratio']:.4f} ¥高 {r['yen_h_ratio']:.4f} "
            f"数字笔画 {r['stroke_norm']:.4f} 数字高 {r['digit_h']}  {r['reason']}  {r['path']}")
    n_tiles = _sheet(flagged, a.out / "flagged_sheet.png")
    say(f"\n拼图 flagged_sheet.png: {n_tiles} 张(最多 80)")

    # ================= 金额字体(AmountFontCheck) =================
    say("\n\n######## 金额字体 AmountFontCheck(和机型无关) ########")
    say("\n== 按页型 x 是否 iPhone 原生尺寸: 正常 / 可疑 / 判不了")
    ftab = collections.defaultdict(collections.Counter)
    for r in rows:
        key = (r["cls"], "iPhone尺寸" if r.get("iphone_size") == 1 else "别的尺寸")
        ftab[key]["共"] += 1
        ftab[key][{0: "判不了", 1: "正常", 2: "可疑"}[r.get("font_verdict", 0)]] += 1
    for key in sorted(ftab):
        c = ftab[key]
        say(f"  {key[0]:<4}{key[1]:<8} 共 {c['共']:>6,}  正常 {c['正常']:>6,}  可疑 {c['可疑']:>4,}  判不了 {c['判不了']:>6,}")
    fm = [r for r in rows if r.get("font_digit_h") not in (None, "")]
    for grp, sel in (("iPhone 尺寸", lambda r: r.get("iphone_size") == 1), ("别的尺寸", lambda r: r.get("iphone_size") != 1)):
        g = [r for r in fm if sel(r)]
        zw = [r["zero_w"] for r in g if r.get("zero_w") is not None]
        dt = [r["dot_h"] for r in g if r.get("dot_h") is not None and r["font_digit_h"] >= amount_font_scan.MIN_DIGIT_H_FOR_DOT]
        say(f"\n  [{grp}] 量到 {len(g):,} 张")
        say(f"     0 宽/数字高      {_q(zw)}   n={len(zw)}")
        say(f"     小数点高/数字高  {_q(dt)}   n={len(dt)}")
        say(f"     0 宽 > {amount_font_scan.ZERO_WIDTH_MAX}: {sum(1 for v in zw if v > amount_font_scan.ZERO_WIDTH_MAX)}"
            f"   > 0.63: {sum(1 for v in zw if v > 0.63)}"
            f"   小数点 < {amount_font_scan.DOT_HEIGHT_MIN}: {sum(1 for v in dt if v < amount_font_scan.DOT_HEIGHT_MIN)}"
            f"   < 0.17: {sum(1 for v in dt if v < 0.17)}")
    near = sorted([r for r in fm if r.get("zero_w") is not None], key=lambda r: -r["zero_w"])[:10]
    say("\n  0 最胖的 10 张: " + "  ".join(f"{r['zero_w']:.4f}" for r in near))
    for r in near[:5]:
        say(f"     {r['zero_w']:.4f} {r['cls']} {r.get('W')}x{r.get('H')}  {r['path']}")
    flat = sorted([r for r in fm if r.get("dot_h") is not None and r["font_digit_h"] >= amount_font_scan.MIN_DIGIT_H_FOR_DOT],
                  key=lambda r: r["dot_h"])[:10]
    say("  小数点最扁的 10 张: " + "  ".join(f"{r['dot_h']:.4f}" for r in flat))
    for r in flat[:5]:
        say(f"     {r['dot_h']:.4f} {r['cls']} {r.get('W')}x{r.get('H')}  {r['path']}")

    fflag = sorted([r for r in rows if r.get("font_verdict") == 2], key=lambda r: (r["cls"], r["path"]))
    say(f"\n== 字体报出来的(可疑) {len(fflag):,} 张(拼图 flagged_font_sheet.png 里的编号就是这里的序号)")
    for i, r in enumerate(fflag, 1):
        say(f"  {i:>3} {r['cls']:<3} {r.get('W')}x{r.get('H')} 数字高 {r.get('font_digit_h')}  {r['font_reason']}  {r['path']}")
    n_ft = _sheet(fflag, a.out / "flagged_font_sheet.png")
    say(f"\n拼图 flagged_font_sheet.png: {n_ft} 张(最多 80)")

    # ---- 给服务器 PowerShell 块看的几行纯 ASCII 汇总(块里不放中文, 免得 5.1 按 ANSI 读坏) ----
    def _cnt(sel):
        return sum(1 for r in rows if sel(r))
    iph_m = [r for r in rows if r.get("iphone_size") == 1 and not r.get("skip")]
    lines.append("")
    lines.append(f"KEY total={len(rows)} missing={miss} iphone_measured={len(iph_m)}")
    lines.append(f"KEY yen_flagged={_cnt(lambda r: r['verdict'] == 2)} "
                 f"yen_h_below_{yen_scan.APPLE_YEN_H_LOW}={sum(1 for r in iph_m if r['digit_h'] >= yen_scan.MIN_DIGIT_H_FOR_YEN_H and r['yen_h_ratio'] < yen_scan.APPLE_YEN_H_LOW)} "
                 f"stem_above_{yen_scan.APPLE_HIGH}={sum(1 for r in iph_m if r['digit_h'] >= yen_scan.MIN_DIGIT_H and r['stem_ratio'] > yen_scan.APPLE_HIGH)}")
    for grp, sel in (("iphone", lambda r: r.get("iphone_size") == 1), ("other", lambda r: r.get("iphone_size") != 1)):
        g = [r for r in fm if sel(r)]
        zw = [r["zero_w"] for r in g if r.get("zero_w") is not None]
        dt = [r["dot_h"] for r in g if r.get("dot_h") is not None and r["font_digit_h"] >= amount_font_scan.MIN_DIGIT_H_FOR_DOT]
        lines.append(f"KEY font_{grp} measured={len(g)} flagged={sum(1 for r in g if r.get('font_verdict') == 2)} "
                     f"zero_max={max(zw) if zw else 'na'} zero_over_0.63={sum(1 for v in zw if v > 0.63)} "
                     f"dot_min={min(dt) if dt else 'na'} dot_under_0.17={sum(1 for v in dt if v < 0.17)}")

    text = "\n".join(lines)
    (a.out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    done_flag.write_text("ok", encoding="utf-8")
    print(f"\nYEN_CENSUS_DONE FONT_CENSUS_DONE  -> {a.out}")


if __name__ == "__main__":
    main()
