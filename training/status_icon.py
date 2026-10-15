r"""蓝图页头的图标判状态(纯看图, 不做 OCR): 到账失败 / 提交成功(还没到账) 的页头和成功页长得不一样。

为什么(2026-10-15)
------------------
经理 10/14 要把 到账失败 这类异常页挑出来。status_scan.py 用 OCR 读页头能挑, 但服务器上一张要 3~6 秒,
89 万张蓝图扫一遍要 30 多小时; 交给经理的检查(TimelineCheck 那些)也都是只看图的 C#, 拿不到 OCR 的字。
看了样本(1.8 万张随机图里的 8 张 到账失败、12 张 提交成功, 和成功页比):
    成功页(转账成功 到账成功 支付成功 代付成功 授权成功): 标题左边一个空心细圈里打勾, 标题下面是大号白字金额 ¥100.00
    到账失败: 标题正上方一个实心白圆, 圆里是蓝色感叹号 !, 标题下面一行灰字 资金将原路返回你的付款账户, 页头里没有大金额
    提交成功: 同样的实心白圆, 圆里是蓝色时钟指针, 下面一行 预计2小时内到账 查看到账进度
量出来(样本 8 + 12 张): 实心白圆横向正中(圆心 0.50 W), 直径 0.05~0.066 W, 圆顶在 0.058~0.076 H,
    白色占外接框 0.69~0.72, 填掉圆里的字以后占 0.74~0.76(正圆是 0.785);
    圆里的蓝字: 感叹号 高/宽 4.5~6.3、两块(一竖 + 一点); 时钟 高/宽 1.7 左右、一块、偏右
成功页标题左边的细圈: 白色只占外接框 0.11~0.12(空心), 圆心在 0.40 W 左右 —— 和实心圆分得很开。

判法
----
    页头不是支付宝蓝(5%~20% 高度平均色 B > 150 且 B > R + 60)   -> CannotDetermine(不是蓝图)
    在 2%~17% 高度里找白色连通块(三个通道都 > 215), 挑 横向正中 / 直径 0.035~0.09 W / 近正方 / 实心 的
        找到, 圆里是感叹号(高/宽 >= 3, 至少两块)  -> Suspicious, 到账失败一类
        找到, 圆里是时钟(高/宽 1.2~2.4)           -> Suspicious(可选), 提交成功 还没到账
        找到, 圆里认不出                          -> CannotDetermine, 记下来
    没找到实心圆, 找到成功页标题左边的空心细圈     -> Ok
    都没有(通知条盖住了页头等)                     -> CannotDetermine
★ 授权成功 / 代付成功 的页头和成功页一模一样, 只能靠字分(status_scan)。
★ 通知条盖住页头的(1.8 万张里 8 张 到账失败 有 1 张), 这里判不了, 交给 status_scan(读 资金将原路返回)。

用法
----
    python training/status_icon.py --src 目录或名单 [...] --out 输出目录 --workers 20
    -> icon.csv(每张一行: verdict kind 圆的位置和大小 圆里字的形状)、summary.txt
    全库: 加 --flagged-only, icon.csv 只写 Suspicious 和蓝图上判不了的
本地: 样本 到账失败 8 张 7 张判出(1 张通知条盖住页头, 判不了), 提交成功 12/12; 成功页、代付、授权 40 张 0 误报;
    本地全部 1,925 张蓝图 0 误报(判出的 2 张都是真的 提交成功 拼音页); 8 进程 209 张/秒(OCR 的 35 倍)
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROI = (0.02, 0.17)              # 找图标的高度范围(占页面高度)
WHITE = 215                     # 三个通道都大于这个算白
# 实心白圆(到账失败 / 提交成功)
DISK_CX = 0.04                  # 圆心离页面正中不超过 0.04 W
DISK_D = (0.035, 0.09)          # 直径占页宽
DISK_TOP = (0.035, 0.13)        # 圆顶占页高
DISK_ASPECT = (0.85, 1.18)
DISK_WHITE = (0.55, 0.80)       # 白色占外接框
DISK_FILLED = (0.68, 0.84)      # 填掉圆里的字以后占外接框(正圆 0.785)
BANG_HW, BANG_PARTS = 3.0, 2    # 感叹号: 高/宽 >= 3, 至少两块
CLOCK_HW = (1.2, 2.4)           # 时钟指针: 高/宽
# 成功页标题左边的空心细圈
RING_CX = (0.25, 0.48)
RING_D = (0.025, 0.065)
RING_WHITE = (0.05, 0.30)
RING_FILLED = (0.68, 0.86)

KEYS = ["path", "W", "H", "verdict", "kind", "disk", "glyph", "secs", "src"]


def blue_header(bgr) -> bool:
    h = bgr.shape[0]
    b, g, r = bgr[int(h * 0.05):int(h * 0.20)].reshape(-1, 3).mean(0)
    return b > 150 and b > r + 60


def _shape(lab, k, x, y, w, h, area):
    """一个白色连通块: (白色占外接框, 填掉里面的洞以后占外接框, 洞的掩膜)。"""
    comp = (lab[y:y + h, x:x + w] == k).astype(np.uint8)
    filled = comp.copy()
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return area / (w * h), float(filled.sum()) / (w * h), (filled == 1) & (comp == 0)


def _glyph(hole, w, h):
    """圆里的蓝字: (高/宽, 块数, 中心横向偏移占直径)。"""
    ys, xs = np.nonzero(hole)
    if len(xs) < 4:
        return 0.0, 0, 0.0
    gw, gh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    n, _l, st, _c = cv2.connectedComponentsWithStats(hole.astype(np.uint8), 8)
    parts = sum(1 for j in range(1, n) if st[j][4] >= 0.01 * len(xs))
    return gh / gw, parts, (xs.mean() - w / 2) / w


def check(bgr) -> dict:
    """-> {verdict: Ok / Suspicious / CannotDetermine, kind: 到账失败 / 提交成功 / 成功 / ..., disk, glyph}"""
    H, W = bgr.shape[:2]
    out = {"verdict": "CannotDetermine", "kind": "", "disk": "", "glyph": ""}
    if W < 300 or H < 400:
        out["kind"] = "图太小"
        return out
    if not blue_header(bgr):
        out["kind"] = "不是蓝图"
        return out
    y0 = int(H * ROI[0])
    roi = bgr[y0:int(H * ROI[1])]
    white = (roi.min(2) > WHITE).astype(np.uint8)
    n, lab, st, _c = cv2.connectedComponentsWithStats(white, 8)
    ring = None
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in st[k])
        d, cx, top = w / W, (x + w / 2) / W, (y + y0) / H
        if not (DISK_ASPECT[0] <= w / max(h, 1) <= DISK_ASPECT[1]) or d < min(DISK_D[0], RING_D[0]) or d > DISK_D[1]:
            continue
        fw, fa, hole = _shape(lab, k, x, y, w, h, area)
        if (abs(cx - 0.5) <= DISK_CX and DISK_D[0] <= d <= DISK_D[1] and DISK_TOP[0] <= top <= DISK_TOP[1]
                and DISK_WHITE[0] <= fw <= DISK_WHITE[1] and DISK_FILLED[0] <= fa <= DISK_FILLED[1]):
            hw, parts, off = _glyph(hole, w, h)
            out["disk"] = f"cx {cx:.3f} top {top:.3f} d {d:.3f} white {fw:.2f} filled {fa:.2f}"
            out["glyph"] = f"h/w {hw:.2f} parts {parts} off {off:+.2f}"
            if hw >= BANG_HW and parts >= BANG_PARTS:
                out.update(verdict="Suspicious", kind="到账失败")
            elif CLOCK_HW[0] <= hw <= CLOCK_HW[1]:
                out.update(verdict="Suspicious", kind="提交成功")
            else:
                out.update(kind="实心圆里认不出")
            return out
        if (RING_CX[0] <= cx <= RING_CX[1] and RING_D[0] <= d <= RING_D[1]
                and RING_WHITE[0] <= fw <= RING_WHITE[1] and RING_FILLED[0] <= fa <= RING_FILLED[1]):
            ring = f"cx {cx:.3f} top {top:.3f} d {d:.3f} white {fw:.2f} filled {fa:.2f}"
    if ring:
        out.update(verdict="Ok", kind="成功", disk=ring)
    else:
        out["kind"] = "没找到图标"
    return out


def work(task):
    path, src = task
    t0 = time.time()
    r = {"path": path, "src": src}
    try:
        buf = np.fromfile(path, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if bgr is None:
            r.update(verdict="CannotDetermine", kind="读不出")
        else:
            r.update(W=bgr.shape[1], H=bgr.shape[0], **check(bgr))
    except Exception as e:  # noqa: BLE001
        r.update(verdict="CannotDetermine", kind=f"出错 {type(e).__name__}")
    r["secs"] = round(time.time() - t0, 3)
    return r


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import page_type as pt
    ap = argparse.ArgumentParser(description="蓝图页头图标判状态(只看图)")
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--flagged-only", action="store_true",
                    help="icon.csv 只写 Suspicious 和蓝图上判不了的(全库几十万张时用, 汇总照样数全部)")
    a = ap.parse_args()
    todo = pt.collect(a.src)
    todo, dups = pt.dedupe_gwcz(todo)
    if a.n and a.n < len(todo):
        random.Random(a.seed).shuffle(todo)
        todo = todo[:a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"  图 {len(todo):,} 张   进程 {a.workers}   去重跳过 {dups:,}", flush=True)
    t0 = time.time()
    rows = []
    step = max(2000, len(todo) // 40)
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(work, todo, chunksize=16), 1):
            rows.append(r)
            if i % step == 0 or i == len(todo):
                el = time.time() - t0
                print(f"    {i:,}/{len(todo):,}   {i / max(el, 1e-9):.1f} 张/秒   "
                      + " ".join(f"{k}:{v:,}" for k, v in Counter(f"{x['verdict']}/{x['kind']}" for x in rows).most_common(8)), flush=True)
    wall = time.time() - t0
    tmp = a.out / "icon.csv.tmp"
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r for r in rows if not a.flagged_only
                    or not (r.get("verdict") == "Ok" or r.get("kind") == "不是蓝图"))
    os.replace(tmp, a.out / "icon.csv")
    c = Counter(f"{x['verdict']}/{x['kind']}" for x in rows)
    lines = ["=" * 70, f"  蓝图页头图标 status_icon   图 {len(rows):,} 张   {wall:.0f} 秒   {len(rows) / max(wall, 1e-9):.1f} 张/秒", "=" * 70]
    lines += [f"  {k:<36}{v:>9,}" for k, v in c.most_common()]
    lines.append("=" * 70)
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
