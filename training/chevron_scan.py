r"""返回箭头异常扫描 —— 不用模型, 只量左上角那个 `<` 的尺寸和虚实。

背景
----
账单详情页左上角的返回箭头是**固定素材**。iOS 的导航栏图标是按点数渲染的,
所以**苹果各机型不管分辨率多少, 这个箭头都是 56x33 像素**, 而且边缘很利。
实测 1179x2556 / 1290x2796 / 1170x2532 / 1320x2868 / 1206x2622 五种分辨率,
98~99% 都是 56x33。

伪造工具里如果烘死了一个尺寸不对的箭头素材, 就会**偏小而且发虚** ——
发虚是因为它是缩放出来的, 原生渲染无论多大都是利的。

2026-09-03 在经理给的三张假图上实测: 全部 47x28, 虚实比 0.73~0.75,
而真图的虚实比中位只有 0.237, p99 也才 0.589。
真图最小的一簇是 54x32(占 1.8%), 和假图的 47x28 之间有 7 个像素的空档。

★ 这条**完全在像素里**, 抹元数据没有用 —— 那正是元数据这条线失效之后缺的东西。

★★ 自标定, 不写死阈值
--------------------
分辨率不同, 箭头的像素尺寸也不同; app 改版也可能换素材。
所以**不写死 56x33**, 而是**从你给的这批图里自己统计出该分辨率的常态**, 再挑偏离的。
这样换时间段、换机型都不用改代码。

覆盖边界
--------
- 只对**有这个返回箭头的页面**有效(账单详情类)。量不到就跳过, 不会瞎报。
- **安卓基本不适用**: 1080x2400 实测只有 14% 能量出箭头(返回图标不一样), 常态也只有 51x31。
  自标定会给它单独定常态, 但样本质量差, 别指望这条在安卓上有用。
- **蓝底转账页等于没覆盖**: 深色页头把取样窗口糊满, 98% 的蓝图量不出来。
  "蓝图不报"是没覆盖, **不能当成特异性证据**。
- 微信/QQ 转发过的图(长边缩到 1280)箭头会缩成一半大, 但它们分辨率五花八门,
  每种都凑不够 --min-group, **会被整组跳过**, 不会误报。
- 只认"箭头偏小且发虚"这一种痕迹。**换个正确尺寸的素材就绕过去了。**
- 判的是"这张图的箭头和同分辨率的其它图不一样", **不是**"这张图一定是假的"。

用法
----
  python training/chevron_scan.py D:\download2\OtherImages --out D:\probe\chev.csv
  # 先小样本试跑
  python training/chevron_scan.py D:\download2\OtherImages --limit 20000 --out D:\probe\chev.csv
  # 只看某个时期(基线和被判的图要同期)
  python training/chevron_scan.py D:\download2\OtherImages --since 20260801 --out D:\probe\chev9.csv

★ 输出里有一栏**按月的常态箭头**。各月都一样 = 素材没变过, 可以放心混着judge;
  某个月不一样 = app 换过素材, **那就要用 --since/--until 分月各判各的**。

**只读**: 只读图片, 只往 --out 写。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{8})\d{6}")   # 文件名里的时间戳, 形如 _20260902211530


def header_size(path):
    """只读文件头拿宽高, 不解码整张图 —— 几十万张时这一步省下大量时间。"""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"\xff\xd8":
                raw = head + f.read(200000)
                i = 2
                while i < len(raw) - 9:
                    if raw[i] != 0xFF:
                        break
                    m = raw[i + 1]
                    if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                        i += 2
                        continue
                    ln = struct.unpack(">H", raw[i + 2:i + 4])[0]
                    if m in (0xC0, 0xC1, 0xC2):
                        h, w = struct.unpack(">HH", raw[i + 5:i + 9])
                        return (w, h)
                    i += 2 + ln
    except Exception:
        pass
    return None


def measure(path, size, dark=170, core_thr=100, mid_thr=200):
    """量返回箭头。返回 (高, 宽, 虚实比) 或 None。

    虚实比 = 半灰边缘像素数 / 实心笔画像素数。
    ★ 原生渲染的图标边缘很干净, 这个比值小; 缩放过的会糊出一圈半灰, 比值明显变大。
    """
    W, H = size
    # ★ 窗口上下都要留出余量。原来上边界取 0.080H, 正好压在箭头顶边上,
    #   于是下面那道"贴边就不算"的检查把 2,492/2,504 张全否掉了 —— 必须留够边距。
    x0, x1 = int(W * 0.025), int(W * 0.102)
    y0, y1 = int(H * 0.068), int(H * 0.116)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").crop((x0, y0, x1, y1)), dtype=np.int16)
    except Exception:
        return None
    d = a < dark
    if int(d.sum()) < 20:
        return None
    ys, xs = np.where(d)
    # 贴到窗口边缘 = 裁歪了或者这页根本没有箭头, 这种不给结论
    if ys.min() == 0 or xs.min() == 0 or ys.max() == a.shape[0] - 1 or xs.max() == a.shape[1] - 1:
        return None
    core = int((a < core_thr).sum())
    if core == 0:
        return None
    mid = int(((a >= core_thr) & (a < mid_thr)).sum())
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1), mid / core


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="扫描返回箭头异常(自标定)")
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="只抽这么多张, 0 = 全扫")
    ap.add_argument("--min-group", type=int, default=300,
                    help="某个分辨率不够这么多张就不给它定常态(定不准)。默认 300")
    ap.add_argument("--size-frac", type=float, default=0.92,
                    help="高和宽都低于该分辨率常态的这个比例才算偏小。默认 0.92")
    ap.add_argument("--blur-mult", type=float, default=1.5,
                    help="虚实比超过该分辨率中位数的几倍才算发虚。默认 1.5")
    ap.add_argument("--since", type=str, default=None, help="只看这个日期(含)之后的, YYYYMMDD")
    ap.add_argument("--until", type=str, default=None, help="只看这个日期(含)之前的, YYYYMMDD")
    ap.add_argument("--emit-table", action="store_true",
                    help="把标定出来的常态打成 C# 字面量, 贴进 ChevronCheck.cs")
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    for _k, _v in (("--since", args.since), ("--until", args.until)):
        if _v is not None and not (len(_v) == 8 and _v.isdigit()):
            raise SystemExit(f"!! {_k} 要写成 YYYYMMDD 八位数字, 收到的是 {_v!r}")

    files = []
    for dp, _, fns in os.walk(args.input):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                files.append(os.path.join(dp, fn))
    print(f"候选 {len(files):,} 张", flush=True)

    # ★ 按文件名日期筛。基线和要判的图**必须是同一个时期** ——
    #   app 换过素材的话, 拿旧月份的常态去量新月份会整批误判。
    if args.since or args.until:
        kept, nodate, outside = [], 0, 0
        for p in files:
            m = _TS.search(os.path.basename(p))
            if not m:
                nodate += 1
                continue
            d = m.group(1)
            if (args.since and d < args.since) or (args.until and d > args.until):
                outside += 1
                continue
            kept.append(p)
        rng = f"{args.since or '不限'} ~ {args.until or '不限'}"
        print(f"按日期筛({rng}): 留下 **{len(kept):,}** 张, 不在范围内 {outside:,}, "
              f"取不到日期 {nodate:,}", flush=True)
        if not kept:
            raise SystemExit("!! 这个日期范围里一张都没有")
        files = kept

    if args.limit and args.limit < len(files):
        import random
        random.Random(args.seed).shuffle(files)
        files = files[:args.limit]

    rows = []
    nosize = nomeas = 0
    for i, p in enumerate(files, 1):
        sz = header_size(p)
        if not sz:
            nosize += 1
            continue
        m = measure(p, sz)
        if not m:
            nomeas += 1
            continue
        _m = _TS.search(os.path.basename(p))
        rows.append([os.path.basename(p), sz[0], sz[1], m[0], m[1], round(m[2], 4),
                     _m.group(1)[:6] if _m else ""])
        if i % 5000 == 0:
            print(f"  {i:,}/{len(files):,}  量到 {len(rows):,}", flush=True)

    print(f"\n量到 **{len(rows):,}** 张; 读不到尺寸 {nosize:,}; 没有箭头/量不到 {nomeas:,}")
    if not rows:
        raise SystemExit("!! 一张都没量到 —— 先确认这批图是不是账单详情页")

    # ---- 按分辨率自标定 ----
    by = collections.defaultdict(list)
    for r in rows:
        by[(r[1], r[2])].append(r)

    flagged = []
    norms = {}          # 分辨率 -> (常态高, 常态宽, 张数), 供 --emit-table 用
    print(f"\n{'分辨率':>14}{'张数':>8}{'常态箭头':>12}{'虚实比中位':>12}{'报出':>8}")
    for sz in sorted(by, key=lambda k: -len(by[k])):
        grp = by[sz]
        if len(grp) < args.min_group:
            continue
        mode_h, _ = collections.Counter(r[3] for r in grp).most_common(1)[0]
        mode_w, _ = collections.Counter(r[4] for r in grp).most_common(1)[0]
        med_blur = float(np.median([r[5] for r in grp]))
        norms[sz] = (mode_h, mode_w, len(grp))
        # ★ 用"占常态的比例"而不是"比常态小就算" —— 后者太松。
        #   实测同一分辨率下真图有两簇(56x33 占 98.1%, 54x32 占 1.8%),
        #   只要求"比常态小"会把第二簇整个报出来。而假图是 47x28, 只有常态的 0.84,
        #   两簇之间有 7 个像素的空档, 卡在 0.92 两边都留得很宽。
        hits = [r for r in grp
                if r[3] < mode_h * args.size_frac
                and r[4] < mode_w * args.size_frac
                and r[5] > med_blur * args.blur_mult]
        flagged.extend(hits)
        print(f"{str(sz):>14}{len(grp):>8}{f'{mode_h}x{mode_w}':>12}"
              f"{med_blur:>12.3f}{len(hits):>8}")

        # ★★ 按月再拆一次常态。**这一栏是用来看 app 有没有换过素材的** ——
        #    如果各月的常态箭头不一样, 说明素材变过, 那就必须按月分别定常态,
        #    否则拿混合出来的常态去判, 会把某一整个月都判成异常。
        hit_ids = {id(r) for r in hits}
        bym = collections.defaultdict(list)
        for r in grp:
            if r[6]:
                bym[r[6]].append(r)
        if len(bym) > 1:
            print(f"{'':>14}{'按月:':>8}")
            for mo in sorted(bym):
                g2 = bym[mo]
                h2 = collections.Counter(r[3] for r in g2).most_common(1)[0]
                w2 = collections.Counter(r[4] for r in g2).most_common(1)[0]
                b2 = float(np.median([r[5] for r in g2]))
                n2 = sum(1 for r in g2 if id(r) in hit_ids)
                print(f"{'':>14}{mo:>8}{len(g2):>8}  {h2[0]}x{w2[0]}"
                      f" ({100.0*h2[1]/len(g2):.0f}%)  虚实比 {b2:.3f}  报出 {n2}")

    n_judged = sum(len(g) for g in by.values() if len(g) >= args.min_group)
    print(f"\n**报出 {len(flagged)} 张 / 实际判定 {n_judged:,} 张 "
          f"= {100.0 * len(flagged) / max(n_judged, 1):.4f}%**")
    print(f"  判据: 箭头比同分辨率常态**又小又虚**"
          f"(高和宽都低于常态的 {args.size_frac:.0%}, 且虚实比超过中位的 {args.blur_mult} 倍)")
    for r in sorted(flagged, key=lambda r: -r[5])[:30]:
        print(f"    {r[3]}x{r[4]} 虚实比 {r[5]:.3f}  {r[1]}x{r[2]}  {r[0][:56]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "W", "H", "chev_h", "chev_w", "blur", "month", "flagged"])
            fset = {id(r) for r in flagged}
            for r in rows:
                w.writerow(r + [1 if id(r) in fset else 0])
        print(f"\n明细 -> {args.out}")
        print("★ 报出来的**一定要人工看一眼**再下结论。")



    # ---- 把标定结果打成 ChevronCheck.cs 能直接贴的字面量 ----
    if args.emit_table and norms:
        print()
        print("=" * 72)
        print(f"// 贴进 ChevronCheck.cs。标定自 {args.input}, "
              f"日期 {args.since or '不限'}~{args.until or '不限'}")
        sizes = collections.Counter((h, w) for h, w, _n in norms.values())
        top, top_n = sizes.most_common(1)[0]
        odd = {sz: v for sz, v in norms.items() if (v[0], v[1]) != top}
        print(f"//   常态 {top[0]}x{top[1]} 覆盖 {top_n}/{len(norms)} 个分辨率")
        if odd:
            # ChevronCheck.cs 假设所有苹果机型共用一个尺寸(导航图标按点数渲染)。
            # 这里只要打印出东西, 就说明那个前提在这批数据上不成立。
            print("// ★ 下列分辨率的常态和上面不一样, '所有苹果机型同尺寸'这个前提不成立,")
            print("//   要么把它们从白名单去掉, 要么改成按分辨率查表:")
            for sz, v in sorted(odd.items(), key=lambda kv: -kv[1][2]):
                print(f"//     {sz[0]}x{sz[1]}  常态 {v[0]}x{v[1]}  n={v[2]:,}")
        print(f"        public const int NormalHeight = {top[0]};")
        print(f"        public const int NormalWidth  = {top[1]};")
        print("        static readonly HashSet<(int, int)> Supported = new()")
        print("        {")
        for sz, v in sorted(norms.items(), key=lambda kv: -kv[1][2]):
            if (v[0], v[1]) == top:
                print(f"            ({sz[0]}, {sz[1]}),   // n={v[2]:,}")
        print("        };")


if __name__ == "__main__":
    main()
