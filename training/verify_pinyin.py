"""把归集好的拼音图逐张核一遍, 把**可疑的挑出来**给人看。

为什么还要再核一遍
------------------
挑图用的分数是个**比例**: 判为注音的连通块 / 总连通块。
这个数只说"有多少块像注音", 不说"它们**长在哪**"。

2026-09-16 实测栽过: 蓝图上的小徽章(充值金+15 / 学分+50 / 里程币+20)
和白图上打码的星号(**浩 / 150 **** **02), 形状同样是"小字压在大字上面凑成一排",
比例照样不低, 但**一张拼音都没有**。

★★★★★ 真拼音和这些假的, 差在**分布**上:

    真拼音     每一行汉字头上都有 -> 覆盖率高, 纵向铺满整页
    小徽章     只在某一条带里     -> 覆盖率低, 纵向挤成一团

所以这里量两个新的:

    coverage   有注音的行 / 总行数
    spread     有注音的行在纵向上铺开的范围(占页高的比例)

再加一个截图判据(平整度), 把拍屏的照片也挑出来 ——
低分段里混进过城市风景照和拍屏。

用法
----
    python verify_pinyin.py --root 归集目录 --out 结果.csv --workers 8
    python verify_pinyin.py --out 结果.csv --report
    python verify_pinyin.py --out 结果.csv --sheet 可疑的.png --worst 12
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import EXTS, annotation_labels, local_background, text_mask  # noqa: E402

# 行分组的容差: 中心 y 差在这个比例的字高之内, 算同一行
ROW_TOL = 0.6

# ★★★ 判可疑的线, 是在已知两组上量出来的, 不是拍的:
#
#     40 张已知拼音图    覆盖率中位 48%, 纵向铺开中位 81%
#     61 张已知非拼音图  覆盖率中位  0%, 纵向铺开中位  0%
#
#   61 张负样本**全部**落在 10% 以下, 所以 10% 这条线负样本一个都不漏。
#   但 40 张正样本里也有 6 张掉在 10% 以下 —— 所以这条线**会误伤**,
#   不能拿来自动判真假。
#
# ★★★★★ 这个数的正确用法是**排序**, 不是分类:
#   覆盖率最低的排最前面, 人从最可疑的开始看。
#   一开始我设成 0.35, 那会把 40 张真拼音里的 14 张标成可疑, 太狠了。
SUSPECT_COVERAGE = 0.10
SUSPECT_SPREAD = 0.15

# 截图平整度(和 check_screenshot.py 同一套): 原始截图是精确的 0
SUSPECT_FLAT = 0.01


def _rows_from(boxes: list[tuple[int, int, int, int]]) -> list[tuple[float, float]]:
    """把字块按纵向聚成行, 返回每行的 (中心y, 高)。"""
    if not boxes:
        return []
    hs = [h for _, _, _, h in boxes]
    med_h = float(np.median(hs))
    items = sorted(((y + h / 2.0, h) for _, y, _, h in boxes))
    rows: list[list[tuple[float, float]]] = []
    for cy, h in items:
        if rows and abs(cy - rows[-1][-1][0]) <= ROW_TOL * med_h:
            rows[-1].append((cy, h))
        else:
            rows.append([(cy, h)])
    return [(float(np.mean([c for c, _ in r])), float(np.mean([h for _, h in r])))
            for r in rows]


def measure(path_str: str) -> dict:
    rec: dict = {"file": Path(path_str).name, "err": None,
                 "coverage": None, "spread": None, "flat": None,
                 "rows": 0, "pinyin_rows": 0,
                 "big_ratio": None, "big_frac": None}
    try:
        gray = cv2.imdecode(np.fromfile(path_str, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rec["err"] = "decode"
            return rec
        H, W = gray.shape

        # --- 截图判据: 小块标准差的下四分位, 原始截图精确等于 0 ---
        k = 16
        hh, ww = H // k, W // k
        if hh >= 4 and ww >= 4:
            blocks = (gray[: hh * k, : ww * k]
                      .reshape(hh, k, ww, k).swapaxes(1, 2))
            sds = blocks.reshape(hh * ww, k * k).std(axis=1)
            rec["flat"] = float(np.percentile(sds, 25))

        # --- 拼音分布 ---
        mask = text_mask(gray, local_background(gray))
        labels, kept, total, big_boxes = annotation_labels(mask)
        if not big_boxes:
            rec["coverage"] = 0.0
            rec["spread"] = 0.0
            return rec

        rows = _rows_from(big_boxes)
        rec["rows"] = len(rows)

        # ★★★★★ 是不是一张回单 —— 光看"有没有拼音"判不出来。
        #
        #   2026-09-16 实测发现: 拼音那个显示模式是**整机生效**的,
        #   所以浏览器页面、系统弹窗、聊天记录, 只要是那台手机截的, 每一行都有拼音。
        #   覆盖率会很高(实测有一张浏览器页面 cov=45%), 上面那两个指标一个都抓不住。
        #
        #   回单有个浏览器页面没有的东西: **一个很大的金额**。
        #   正文字高大概是页高的 1%, 金额那一行能到 3~5%。
        #   所以量"最高的那一行字 除以 正文字高", 回单会明显大。
        hs = sorted((h for _, _, _, h in big_boxes), reverse=True)
        if hs:
            body = float(np.median(hs))
            rec["big_ratio"] = (hs[0] / body) if body > 0 else 0.0
            rec["big_frac"] = hs[0] / H

        if not rows or not kept:
            rec["coverage"] = 0.0
            rec["spread"] = 0.0
            return rec

        # 每个注音块归到它**正下方**最近的那一行
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        hit_rows = set()
        ys = []
        for idx in kept:
            if idx >= len(st):
                continue
            y, h = int(st[idx][1]), int(st[idx][3])
            cy = y + h
            ys.append(y + h / 2.0)
            best, bestd = None, None
            for i, (ry, rh) in enumerate(rows):
                d = (ry - rh / 2.0) - cy          # 行顶 减 注音块底
                if d < -rh * 0.5:                 # 在行里面或更下面, 不算
                    continue
                if bestd is None or d < bestd:
                    best, bestd = i, d
            if best is not None and bestd is not None and bestd <= rows[best][1] * 1.5:
                hit_rows.add(best)

        rec["pinyin_rows"] = len(hit_rows)
        rec["coverage"] = len(hit_rows) / len(rows)
        rec["spread"] = (max(ys) - min(ys)) / H if len(ys) > 1 else 0.0
    except Exception as e:                        # noqa: BLE001
        rec["err"] = type(e).__name__
    return rec


def iter_images(root: Path):
    for dirpath, _, names in os.walk(root):
        for n in names:
            if Path(n).suffix.lower() in EXTS:
                yield str(Path(dirpath) / n)


def read_rows(out: Path) -> list[dict]:
    rows = []
    with out.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for k in ("coverage", "spread", "flat", "big_ratio", "big_frac"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
            for k in ("rows", "pinyin_rows"):
                r[k] = int(r[k]) if r.get(k) not in (None, "", "None") else 0
            rows.append(r)
    return rows


def why_suspect(r: dict) -> str:
    if r.get("err"):
        return f"读不了({r['err']})"
    bits = []
    if r["coverage"] is not None and r["coverage"] < SUSPECT_COVERAGE:
        bits.append(f"覆盖率只有{r['coverage']:.0%}")
    if r["spread"] is not None and r["spread"] < SUSPECT_SPREAD:
        bits.append(f"只集中在{r['spread']:.0%}的页高里")
    if r["flat"] is not None and r["flat"] > SUSPECT_FLAT:
        bits.append(f"不是原始截图(平整度{r['flat']:.3f})")
    return ",  ".join(bits)


def report(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if not r.get("err")]
    bad = [r for r in rows if r.get("err")]
    print("=" * 74)
    print(f"核了 {len(rows):,} 张    读不了 {len(bad)}")
    print("=" * 74)
    if not ok:
        return []

    cov = np.array([r["coverage"] for r in ok if r["coverage"] is not None])
    spr = np.array([r["spread"] for r in ok if r["spread"] is not None])
    print(f"  覆盖率  中位 {np.median(cov):.0%}   25分位 {np.percentile(cov,25):.0%}"
          f"   5分位 {np.percentile(cov,5):.0%}")
    print(f"  纵向铺开 中位 {np.median(spr):.0%}   25分位 {np.percentile(spr,25):.0%}"
          f"   5分位 {np.percentile(spr,5):.0%}")
    print()
    print("  覆盖率分布:")
    for lo, hi in [(0, 0.1), (0.1, 0.35), (0.35, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        n = int(((cov >= lo) & (cov < hi)).sum())
        tag = "  <- ★ 可疑" if hi <= SUSPECT_COVERAGE else ""
        print(f"    {lo:.0%} - {hi:.0%}   {n:6,d}{tag}")
    print()

    suspect = [r for r in rows if why_suspect(r)]
    print("=" * 74)
    print(f"★★ 可疑的 {len(suspect):,} 张  (占 {len(suspect)/len(rows):.1%})")
    print("=" * 74)
    suspect.sort(key=lambda r: (r["coverage"] if r["coverage"] is not None else -1))
    for r in suspect[:20]:
        print(f"  {r['file'][:54]:<56} {why_suspect(r)}")
    if len(suspect) > 20:
        print(f"  ...还有 {len(suspect)-20:,} 张")
    print()
    print(f"★ 剩下 {len(rows)-len(suspect):,} 张看着是干净的。")
    print("★★ 可疑的那些要贴出来肉眼核 —— 数字永远不算证据。")
    return suspect


def make_sheet(rows: list[dict], root: Path, sheet: Path, n: int,
               worst: bool) -> None:
    pool = [r for r in rows if why_suspect(r)] if worst else \
           [r for r in rows if not why_suspect(r)]
    if not pool:
        print("这一类一张都没有")
        return
    pool.sort(key=lambda r: (r["coverage"] if r["coverage"] is not None else -1),
              reverse=not worst)
    step = max(1, len(pool) // n)
    picked = pool[::step][:n]

    cells = []
    for r in picked:
        p = root / r["file"]
        if not p.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h = img.shape[0]
        c = img[int(h * 0.03): int(h * 0.30), :]
        if c.size == 0:
            continue
        c = cv2.resize(c, (500, max(1, int(500 * c.shape[0] / c.shape[1]))))
        cv2.putText(c, f"cov={r['coverage']:.0%} spr={r['spread']:.0%}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cells.append(c)
    if not cells:
        print("图都读不出来")
        return
    hh = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in cells]
    per = 3
    usable = len(cells) - len(cells) % per
    if usable == 0:
        usable, per = len(cells), len(cells)
    grid = np.vstack([np.hstack(cells[i:i + per]) for i in range(0, usable, per)])
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", grid)[1].tofile(str(sheet))
    what = "可疑的" if worst else "看着干净的"
    print(f"贴了 {usable} 张{what} -> {sheet}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--worst", type=int, default=0, help="贴最可疑的这么多张")
    ap.add_argument("--best", type=int, default=0, help="贴看着最干净的这么多张")
    a = ap.parse_args()

    if a.report or a.sheet:
        if not a.out.exists():
            print(f"{a.out} 不在, 先跑一遍 --root")
            return
        rows = read_rows(a.out)
        if a.report or not a.sheet:
            report(rows)
        if a.sheet:
            root = a.root or Path(rows[0].get("dir", ".")) if rows else Path(".")
            if a.root is None:
                print("贴图要给 --root(图在哪个目录)")
                return
            n = a.worst or a.best or 12
            make_sheet(rows, a.root, a.sheet, n, worst=bool(a.worst) or not a.best)
        return

    if not a.root:
        print("要给 --root")
        return
    todo = list(iter_images(a.root))
    if not todo:
        print("没找到图")
        return
    print(f"要核 {len(todo):,} 张,  {a.workers} 个进程")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(measure, todo, chunksize=50))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "coverage", "spread", "flat",
                                          "big_ratio", "big_frac",
                                          "rows", "pinyin_rows", "err"])
        w.writeheader()
        w.writerows(rows)
    print(f"结果 -> {a.out}\n")
    report(rows)


if __name__ == "__main__":
    main()
