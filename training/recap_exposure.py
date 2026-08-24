r"""标定池排除名单的**暴露面**分析 —— 那 1,277 张里, 哪些拒掉是对的, 哪些才是真误杀。

为什么需要这一份
----------------
经理说**翻拍先不做单独通道**, 于是标定池当初剔掉的那批图会混进同一条流。
第一反应是"那把 1,277 张全放回去重标阈值" —— **这一步是错的**, 因为那 1,277 张
不是一个群体, 是**三个判据**各自剔出来的, 而三者"拒掉算不算误杀"的答案完全不同:

  `screenshot_filter.py` 复用 `_is_clean_screenshot`, 三条判据彼此独立:
    (a) 有相机 EXIF(曝光/光圈/ISO/焦距)  -> 是**拍摄的照片**, 即翻拍
    (b) 短边 > 1500                       -> 不像手机截图(平板/桌面?)
    (c) 长宽比不在 1.6~2.6                -> 不是竖屏截图版式

  **(a) 拒掉是功能正确, 不是误杀。** `screenshot_filter.py` 文件头原话:
  "这些被模型拒掉不是误杀。SSP 最初就是翻拍检测器…它拒掉一张翻拍是功能正确, 不是判错。"

  **(b)(c) 就不一定了。** 平板截图、异形比例的收据**可能是正经提交**。
  `DEPLOY_SPEC.md` 自己写着: "那是**业务范围判断不是技术判断**, 线上如果收平板截图,
  这 842 张应该留在分母里。**这一条要经理点头。**"

所以真正要回答的是两个不同的问题:
  1. 经理量到的"误杀"里, 有多少其实是**正确拒掉的翻拍**? -> 这是**度量口径**问题, 不是模型问题
  2. 平板/异形那批**该不该进分母**? -> 这是**业务范围**问题, 要他拍板

这份脚本把两个问题各自的**数量级**摆出来, 让那次拍板有数可依。

只读
----
只读 summary.csv 和**图片文件头**(不解码像素), 不写任何东西, 不碰模型。

用法
----
  python training/recap_exposure.py --csv E:\SSP_Work\probe\gen100k_white\summary.csv ^
      --roots E:\SSP_Work\probe\gen100k_imgs D:\download2\TempFakeImages
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# 与 gen_local_ai_edit / build_*_dataset 里 _is_clean_screenshot 用的是**同一组**光学 EXIF 标签
_OPT = (33434, 33437, 34855, 37386)   # 曝光 / 光圈 / ISO / 焦距


def classify(path: str):
    """返回 (是不是干净截图, 原因列表)。**只读文件头, 不解码像素。**

    三条判据独立, 一张图可能同时命中多条 —— 所以返回列表不是单值。
    """
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
            try:
                ifd = im.getexif().get_ifd(0x8769)
            except Exception:
                ifd = {}
    except Exception as e:
        return None, [f"读不出来({type(e).__name__})"]
    short = min(w, h)
    if not short:
        return False, ["尺寸为 0"]
    aspect = max(w, h) / short
    why = []
    if any(t in ifd for t in _OPT):
        why.append("相机EXIF")
    if short > 1500:
        why.append("短边>1500")
    if not (1.6 <= aspect <= 2.6):
        why.append("长宽比异常")
    return (not why), why


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="排除名单的暴露面分析(只读)")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="真图 summary.csv, 可多个")
    ap.add_argument("--roots", type=Path, nargs="+", required=True, help="到哪儿找图(会建文件名索引)")
    ap.add_argument("--a-col", default="final_ai_score", help="线路A 分数列名")
    ap.add_argument("--name-col", default="", help="文件名列名, 留空自动猜")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="只看前 N 行(调试用)")
    args = ap.parse_args()

    sys.path.insert(0, str(_HERE))
    from ssp_decide import load_config                      # noqa: E402
    cfg = load_config(args.config or (_HERE / "ssp_config.json"))
    A_S, A_R = cfg["line_a"]["strict"], cfg["line_a"]["review"]
    print(f"线路A 阈值: 自动拒 {A_S:.4f} / 人工复核 {A_R:.4f}\n", flush=True)

    # ---- 建文件名 -> 路径索引 ----
    idx: dict[str, str] = {}
    for r in args.roots:
        if not r.is_dir():
            print(f"  跳过(不存在) {r}", flush=True)
            continue
        for dp, _, fns in os.walk(r):
            for fn in fns:
                idx.setdefault(fn, os.path.join(dp, fn))
    print(f"图片索引 {len(idx)} 个\n", flush=True)
    if not idx:
        raise SystemExit("!! --roots 下一张图都没找到")

    # ---- 读分数 ----
    rows = []
    for c in args.csv:
        if not c.is_file():
            print(f"  跳过(不存在) {c}", flush=True)
            continue
        with c.open(encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            name_col = args.name_col or next(
                (k for k in cols if k.lower() in ("file", "filename", "name", "image", "img")), "")
            if not name_col or args.a_col not in cols:
                print(f"  !! {c.name} 缺列。表头 = {cols}", flush=True)
                continue
            for r in rd:
                try:
                    rows.append((r[name_col], float(r[args.a_col])))
                except (KeyError, TypeError, ValueError):
                    pass
        print(f"  {c.name} -> 累计 {len(rows)} 行", flush=True)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("!! 一行分数都没读到, 检查 --a-col 和 --name-col")

    # ---- 逐张分类 ----
    print(f"\n逐张读文件头分类 {len(rows)} 张(不解码像素)...", flush=True)
    buckets: dict[str, list[float]] = {}
    missing = 0
    for i, (nm, sc) in enumerate(rows, 1):
        p = idx.get(nm)
        if not p:
            missing += 1
            continue
        ok, why = classify(p)
        if ok is None:
            key = "读不出来"
        elif ok:
            key = "干净截图"
        else:
            key = "+".join(why)          # 命中多条就拼起来, 别硬塞进单一桶
        buckets.setdefault(key, []).append(sc)
        if i % 20000 == 0:
            print(f"  已查 {i}/{len(rows)}...", flush=True)
    if missing:
        print(f"  ({missing} 张在 --roots 下找不到, 已跳过)", flush=True)

    total = sum(len(v) for v in buckets.values())
    if not total:
        raise SystemExit("!! 一张都没配上, 检查 --roots")

    # ---- 报表 ----
    # 拒掉算不算误杀: 只看是不是**纯**相机 EXIF 那一类
    def verdict(key: str) -> str:
        if key == "干净截图":
            return "算(正常进件)"
        if key == "读不出来":
            return "查不了"
        if "相机EXIF" in key:
            return "**不算**(翻拍是目标)"
        return "**要经理定**(平板/异形?)"

    print("\n" + "=" * 96)
    print(f"{'类别':<22}{'张数':>7}{'占比':>8}{'自动拒':>8}{'人工复核':>9}   拒掉算误杀吗")
    print("-" * 96)
    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
        v = buckets[key]
        n = len(v)
        hard = sum(1 for s in v if s >= A_S)
        rev = sum(1 for s in v if A_R <= s < A_S)
        print(f"{key:<22}{n:>7}{100.0*n/total:>7.2f}%{hard:>8}{rev:>9}   {verdict(key)}")
    print("=" * 96)

    clean = buckets.get("干净截图", [])
    exif_keys = [k for k in buckets if "相机EXIF" in k]
    fmt_keys = [k for k in buckets if k not in ("干净截图", "读不出来") and "相机EXIF" not in k]
    exif = [s for k in exif_keys for s in buckets[k]]
    fmt = [s for k in fmt_keys for s in buckets[k]]

    def per_wan(v: list[float], thr: float) -> float:
        return 10000.0 * sum(1 for s in v if s >= thr) / len(v) if v else 0.0

    print("\n三组各自的自动拒率(每万张):")
    print(f"  干净截图        n={len(clean):>6}   {per_wan(clean, A_S):>7.1f}/万")
    print(f"  相机EXIF(翻拍)  n={len(exif):>6}   {per_wan(exif, A_S):>7.1f}/万   <- 这些拒掉是**对的**")
    print(f"  平板/异形       n={len(fmt):>6}   {per_wan(fmt, A_S):>7.1f}/万   <- 这些**可能是真误杀**")

    if clean and fmt:
        print("\n★ 若经理确认平板/异形也算正常进件, 按它在进件里的占比折算, 自动拒率会变成:")
        base = per_wan(clean, A_S)
        for p in (0.005, 0.01, 0.02, 0.03, 0.05):
            blended = (1 - p) * base + p * per_wan(fmt, A_S)
            print(f"     占 {p*100:>4.1f}%  ->  {blended:>6.2f}/万   (现在对外报的是 {base:.2f}/万)")
        print("\n  注: 相机EXIF 那批**不进这个折算** —— 拒掉翻拍是功能正确, 不该计进误杀分母。")
        print("      但经理那边量真实误杀率时, **必须把翻拍单独标出来**, 否则会把正确拒掉的也算成误杀。")


if __name__ == "__main__":
    main()
