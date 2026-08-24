r"""三种"分母口径"下的误杀率 —— 给经理那个"平板算不算正常进件"的问题备好答案。

背景
----
经理说**翻拍先不做单独通道**。标定池当初用 `not_screenshot.txt`(1,277 张)剔掉了
"不是干净截图"的图, 而那份名单是**三条判据**拼出来的, 三者性质不同:

  相机 EXIF   -> **翻拍**。拒掉是功能正确, `screenshot_filter.py` 文件头写得很清楚:
                 "这些被模型拒掉不是误杀。SSP 最初就是翻拍检测器…拒掉一张翻拍是功能正确"
  短边 > 1500 -> 平板/桌面截图?  **可能是正经进件**
  长宽比异常  -> 异形版式?        **可能是正经进件**

所以"该不该进误杀分母"有三种口径, 本脚本**三种全算出来**, 经理一句话就能选:

  A 现状      : 1,277 张全剔掉        <- 现在对外报的 2.7/万 就是这个
  B 平板算进件: 只把**格式类**放回去   <- 相机EXIF 那批仍不进分母(拒它是对的)
  C 全算进件  : 1,277 张全放回去      <- 最悲观的上界

★ 自检
------
**方案 A 必须复现已知的 24 张自动拒 / 88,645(= 2.7/万)。** 复现不了就说明
输入或规则接错了, 此时 B/C 的数字一律不能信 —— 脚本会直接停下来而不是继续往下算。

只读
----
只读 CSV 和名单文件, 只读图片**文件头**(判版式, 不解码像素)。不写任何东西。

用法
----
  python training/recalib_scenarios.py ^
      --a-csv E:\SSP_Work\probe\accept_seeded\_lineA\summary.csv ^
      --b-csv E:\SSP_Work\probe\accept_seeded\_lineB\summary.csv ^
      --exclude E:\SSP_Work\probe\exclude_big3.txt ^
      --not-screenshot E:\SSP_Work\probe\not_screenshot.txt ^
      --roots E:\SSP_Work\probe D:\download D:\download2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OPT = (33434, 33437, 34855, 37386)      # 曝光 / 光圈 / ISO / 焦距


def _norm(s: str) -> str:
    return os.path.basename(s.strip().replace("\\", "/"))


def _read_names(p: Path) -> set[str]:
    if not p or not p.is_file():
        return set()
    # utf-8-sig: PowerShell 写出来的名单常带 BOM, 不吃掉第一行永远匹配不上
    return {_norm(ln) for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t", "是")


def _load(p: Path, cols: list[str]) -> dict[str, tuple]:
    """读 summary.csv -> {文件名: (col1, col2, ...)}。缺列直接报错, 不猜。"""
    out: dict[str, tuple] = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        have = rd.fieldnames or []
        key = "image_name" if "image_name" in have else ("image" if "image" in have else "")
        miss = [c for c in cols if c not in have]
        if not key or miss:
            raise SystemExit(f"!! {p} 缺列 {miss or '文件名列'}; 表头 = {have}")
        for r in rd:
            out[_norm(r[key])] = tuple(r[c] for c in cols)
    return out


def classify(path: str) -> list[str]:
    """三条判据, 只读文件头。返回命中的判据(空 = 干净截图)。"""
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
            try:
                ifd = im.getexif().get_ifd(0x8769)
            except Exception:
                ifd = {}
    except Exception:
        return ["读不出来"]
    short = min(w, h)
    if not short:
        return ["尺寸为0"]
    why = []
    if any(t in ifd for t in _OPT):
        why.append("相机EXIF")
    if short > 1500:
        why.append("短边>1500")
    if not (1.6 <= max(w, h) / short <= 2.6):
        why.append("长宽比异常")
    return why


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="三种分母口径下的误杀率(只读)")
    ap.add_argument("--a-csv", type=Path, required=True)
    ap.add_argument("--b-csv", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, required=True, help="合并后的排除名单")
    ap.add_argument("--not-screenshot", type=Path, required=True)
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--expect-hard", type=int, default=24, help="方案A 应当复现的自动拒张数")
    ap.add_argument("--expect-n", type=int, default=88645, help="方案A 应当复现的分母")
    args = ap.parse_args()

    sys.path.insert(0, str(_HERE))
    from ssp_decide import decide, load_config           # noqa: E402
    cfg = load_config(args.config or (_HERE / "ssp_config.json"))
    la, lb = cfg["line_a"], cfg["line_b"]
    print(f"阈值 A {la['strict']:.4f}/{la['review']:.4f}  B {lb['strict']:.4f}/{lb['review']:.4f}\n",
          flush=True)

    a = _load(args.a_csv, ["final_ai_score"])
    b = _load(args.b_csv, ["tile_top3", "roi_amount_located"])
    print(f"线路A {len(a)} 行, 线路B {len(b)} 行", flush=True)

    excl = _read_names(args.exclude)
    nots = _read_names(args.not_screenshot)
    print(f"排除名单 {len(excl)} 个, not_screenshot {len(nots)} 个", flush=True)

    # ---- 把 not_screenshot 按判据分成两拨 ----
    idx: dict[str, str] = {}
    for r in args.roots:
        if not r.is_dir():
            continue
        for dp, _, fns in os.walk(r):
            for fn in fns:
                idx.setdefault(fn, os.path.join(dp, fn))
    print(f"图片索引 {len(idx)} 个, 开始分类 not_screenshot ...", flush=True)

    exif_set, fmt_set, unk = set(), set(), 0
    for nm in nots:
        p = idx.get(nm)
        if not p:
            unk += 1
            continue
        why = classify(p)
        (exif_set if "相机EXIF" in why else fmt_set).add(nm)
    print(f"  相机EXIF(翻拍) {len(exif_set)}   格式类(平板/异形) {len(fmt_set)}"
          f"   找不到图 {unk}\n", flush=True)

    def run(pool: set[str], label: str) -> tuple[int, int, int, int]:
        hard = rev = 0
        hard_exif = 0
        for nm in pool:
            av = a.get(nm)
            if not av:
                continue
            try:
                a_t = (float(av[0]), True)
            except (TypeError, ValueError):
                a_t = None
            bv = b.get(nm)
            b_t = None
            if bv:
                try:
                    b_t = (float(bv[0]), _truthy(bv[1]))
                except (TypeError, ValueError):
                    b_t = None
            ext = os.path.splitext(nm)[1].lower()
            d, _ = decide(a_t, b_t, cfg, ext, None)
            if d == "自动拒":
                hard += 1
                if nm in exif_set:
                    hard_exif += 1
            elif d == "人工复核":
                rev += 1
        n = sum(1 for nm in pool if nm in a)
        print(f"{label:<26}分母 {n:>6}   自动拒 {hard:>4} = {10000.0*hard/n:>5.2f}/万"
              f"   人工复核 {rev:>4} = {10000.0*rev/n:>5.2f}/万"
              + (f"   (其中翻拍 {hard_exif} 张)" if hard_exif else ""), flush=True)
        return n, hard, rev, hard_exif

    all_names = set(a)
    base = all_names - excl

    print("=" * 104)
    nA, hA, rA, _ = run(base, "A 现状(1,277 全剔)")

    # ★ 自检: A 必须复现已知结果, 否则后面两行没有意义
    if abs(nA - args.expect_n) > max(200, args.expect_n * 0.01) or abs(hA - args.expect_hard) > 4:
        print("=" * 104)
        print(f"\n!! 自检没过: 方案A 应当接近 分母 {args.expect_n} / 自动拒 {args.expect_hard},")
        print(f"   实际是 分母 {nA} / 自动拒 {hA}。")
        print("   **B/C 的数字在这种情况下不可信, 已停止。** 可能原因:")
        print("     1. --exclude 用错了名单(试试 exclude_v5.txt / exclude_v4_full.txt)")
        print("     2. --a-csv / --b-csv 不是验收那一次的产物")
        print("     3. 配置里的阈值和当初标定时不一致")
        raise SystemExit(1)
    print("   ^ 自检通过: 与已知的 24 / 88,645(2.7/万)吻合\n")

    nB, hB, rB, eB = run(base | fmt_set, "B 平板算进件")
    nC, hC, rC, eC = run(base | fmt_set | exif_set, "C 全部算进件")
    print("=" * 104)

    print("\n判读:")
    print(f"  A -> B  自动拒 {10000.0*hA/nA:.2f} -> {10000.0*hB/nB:.2f}/万"
          f"   ({(hB/nB)/(hA/nA)-1:+.1%})   <- 经理若说平板算正常进件, 报这个")
    print(f"  A -> C  自动拒 {10000.0*hA/nA:.2f} -> {10000.0*hC/nC:.2f}/万"
          f"   ({(hC/nC)/(hA/nA)-1:+.1%})   <- 上界; 但其中 {eC} 张是翻拍, **拒它是对的**")
    print("\n  ★ C 那一档不建议对外报成'误杀' —— 翻拍被拒是功能正确。")
    print("    真要报, 必须把翻拍那部分单独列出来, 否则等于把做对的事算成做错。")


if __name__ == "__main__":
    main()
