r"""timeline_scan.py 扫完之后: 按分辨率拆开看可疑的都是什么, 再把要人眼看的原图拷出来。

为什么要按分辨率拆
------------------
可疑的如果全都出在同一个分辨率, 而且这个分辨率下所有的图都是这个偏移,
那多半是这款手机(或某个版本的支付宝)本来就这样排版, 不是假图 —— 规则会把这款手机的用户全判可疑。
可疑的如果只占这个分辨率的一小部分, 其余同分辨率的图都是正常的 +0.2 左右, 那才是假图。
所以每个出现可疑的分辨率, 都拿同分辨率的正常图来比。

用法
----
    python -u training/timeline_look.py --csv E:/SSP_Work/urgent112/tl_other.csv --copy-to E:/SSP_Work/urgent112/look
拷出来三类(文件名前缀): S_ 全部可疑, C_ 全部"正常但偏移 < 0.10", O_ 每个可疑分辨率下 6 张正常图做对照。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import shutil

import numpy as np


def day_of(path):
    m = re.search(r"_(20\d{6})\d{6}\.", os.path.basename(path))
    return m.group(1) if m else "?"


def bucket(o):
    if o <= -0.01:
        return "susp"
    if o < 0.10:
        return "close"
    if o < 0.30:
        return "normal"
    return "high"


def main():
    ap = argparse.ArgumentParser(description="按分辨率看时间轴扫描结果, 拷出要看的原图")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--copy-to", default="")
    ap.add_argument("--ok-per-res", type=int, default=6)
    ap.add_argument("--prev", default="", help="上一次扫描的 csv: 列出判定变了的图, 和可疑有关的拷出来(前缀 X_)")
    a = ap.parse_args()

    changed = []
    if a.prev:
        with open(a.prev, encoding="utf-8-sig", newline="") as f:
            prev = {r["path"]: r for r in csv.DictReader(f) if r.get("verdict") not in ("error", "unreadable")}
        with open(a.csv, encoding="utf-8-sig", newline="") as f:
            cur = {r["path"]: r for r in csv.DictReader(f) if r.get("verdict") not in ("error", "unreadable")}
        trans = collections.Counter()
        for p in set(prev) | set(cur):
            v0 = prev[p]["verdict"] if p in prev else "NoTimeline"
            v1 = cur[p]["verdict"] if p in cur else "NoTimeline"
            if v0 != v1:
                trans[(v0, v1)] += 1
                if "Suspicious" in (v0, v1):
                    changed.append((v0, v1, cur.get(p) or prev[p], prev.get(p, {}).get("offset", ""), cur.get(p, {}).get("offset", "")))
        print("==== 0. changes against the previous scan (previous -> now) ====")
        for (v0, v1), n in sorted(trans.items(), key=lambda kv: -kv[1]):
            print(f"  {v0:>15} -> {v1:<15} {n:,}")
        for v0, v1, r, o0, o1 in sorted(changed, key=lambda t: t[2]["path"]):
            print(f"    {v0} -> {v1}   offset {o0 or '-'} -> {o1 or '-'}   {r['W']}x{r['H']}   {os.path.basename(r['path'])}")
        print()

    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("verdict") in ("Ok", "Suspicious")]
    for r in rows:
        r["o"] = float(r["offset"])
        r["res"] = f"{r['W']}x{r['H']}"
        r["b"] = bucket(r["o"])
    print(f"measured timeline pages: {len(rows):,}")

    by = collections.defaultdict(list)
    for r in rows:
        by[r["res"]].append(r)
    susp_res = {r["res"] for r in rows if r["b"] == "susp"}
    close_res = {r["res"] for r in rows if r["b"] == "close"}

    print("\n==== 1. by resolution: every resolution that has a Suspicious or close page, plus the 15 most common ====")
    common = [k for k, _ in sorted(by.items(), key=lambda kv: -len(kv[1]))[:15]]
    show = sorted(set(common) | susp_res | close_res, key=lambda k: -len(by[k]))
    print(f"  {'resolution':<12}{'pages':>7}{'susp':>6}{'close':>6}{'normal':>7}{'high':>5}   "
          f"{'min':>7}{'p1':>7}{'p50':>7}{'max':>7}   susp share")
    for k in show:
        v = np.array([r["o"] for r in by[k]])
        c = collections.Counter(r["b"] for r in by[k])
        share = c["susp"] / len(v) * 100
        print(f"  {k:<12}{len(v):>7,}{c['susp']:>6}{c['close']:>6}{c['normal']:>7,}{c['high']:>5}   "
              f"{v.min():>+7.3f}{np.percentile(v, 1):>+7.3f}{np.median(v):>+7.3f}{v.max():>+7.3f}   {share:5.1f}%")

    sus = sorted([r for r in rows if r["b"] == "susp"], key=lambda r: r["o"])
    print(f"\n==== 2. the {len(sus)} Suspicious pages: offset in px x resolution x label (1 = has 处理进度 label) ====")
    cc = collections.Counter((r["offset_px"], r["res"], r["label"]) for r in sus)
    for (px, res, lab), n in sorted(cc.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:>4} pages   offset {int(px):>3} px   {res:<12} label {lab}")

    print("\n==== 3. by upload day, for each resolution with Suspicious pages: Suspicious / all pages that day ====")
    for k in sorted(susp_res, key=lambda k: -len(by[k])):
        days = collections.defaultdict(lambda: [0, 0])
        for r in by[k]:
            d = days[day_of(r["path"])]
            d[1] += 1
            d[0] += r["b"] == "susp"
        line = "  ".join(f"{d[4:]}:{s}/{t}" for d, (s, t) in sorted(days.items()) if s or t >= 1)
        print(f"  {k}  {line}")

    print("\n==== 4. the offset histogram at each resolution with Suspicious pages (px, all pages at that resolution) ====")
    for k in sorted(susp_res, key=lambda k: -len(by[k])):
        h = collections.Counter(int(r["offset_px"]) for r in by[k])
        print(f"  {k}  " + "  ".join(f"{px:+d}px:{n}" for px, n in sorted(h.items())))

    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
        picks = []
        for i, r in enumerate(sus, 1):
            picks.append((f"S_{i:03d}_{r['o']:+.3f}_{r['res']}_", r))
        close = sorted([r for r in rows if r["b"] == "close"], key=lambda r: r["o"])
        for i, r in enumerate(close, 1):
            picks.append((f"C_{i:03d}_{r['o']:+.3f}_{r['res']}_", r))
        for k in sorted(susp_res):
            ok = sorted([r for r in by[k] if r["b"] == "normal"], key=lambda r: r["path"])
            step = max(1, len(ok) // max(1, a.ok_per_res))
            for i, r in enumerate(ok[::step][: a.ok_per_res], 1):
                picks.append((f"O_{k}_{i:02d}_{r['o']:+.3f}_", r))
        for i, (v0, v1, r, o0, o1) in enumerate(changed, 1):
            picks.append((f"X_{i:03d}_{v0[:4]}-{v1[:4]}_", r))
        done = miss = 0
        for pre, r in picks:
            try:
                shutil.copy2(r["path"], os.path.join(a.copy_to, pre + os.path.basename(r["path"])))
                done += 1
            except OSError:
                miss += 1
        print(f"\ncopied {done} images to {a.copy_to}" + (f"   ({miss} could not be copied)" if miss else ""))


if __name__ == "__main__":
    main()
