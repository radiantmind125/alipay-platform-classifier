r"""把可疑图和**对照组**一起拷出来, 供人工看。

为什么一定要有对照组
--------------------
★ 只看报出来的图, 没法判断"这些看着不对劲"是真的不对劲, 还是**支付宝的图本来就长这样**。
这个项目已经因此错过三次:
  - 「第二个造假工具」 -> 其实是商家账单页, 真页面
  - 「八月翻 6 倍的欺诈活动」 -> 其实是页型占比变了
  - 「¥ 探测器」 -> 报出的 8 张全部带 ¥, 量的根本不是字体
**所以这个脚本一定同时拷一批随机真图(CTRL_ 前缀), 混在一起看。**

判据(和 dot_scan.py / MinusCheck.cs / DotCheck.cs 对齐)
-------------------------------------------------------
  VPOSONLY_ ★ 负号竖直位置越界, **而且负号宽比和小数点都没报** —— 新增覆盖, 最该看
  VPOSHI_   负号坐得过高: bar_vpos > 0.620
  VPOSLO_   负号坐得过低: bar_vpos < 0.515
  NEU_      ★ 金额墨色不是中性灰: max(BGR) - min(BGR) > 20
  DOT_      小数点是圆的: 填充率 < 0.90, 仅 png、非翻拍、账单详情页
  MINUSHI_  负号偏宽: >= 0.78   —— 已上线的那条
  MINUSLO_  负号偏窄: < 0.625
  BOTH_     多条同时命中 —— 置信度最高
  GATETAIL_ ★ 被商家页闸挡下、但图标长宽比不在 0.97~1.03 的 —— 疑似误闸的详情页
  CTRL_     随机真图对照组

★ 商家账单页(左上角 X 关闭图标)在判据里一律排除 —— 那个页型金额字体不一样,
  负号天生更宽。但 GATETAIL_ 是**专门**去看被闸错的那批。

用法
----
  python training/collect_suspects.py --csv probe\inv1.csv ^
      --src D:\download2\OtherImages --out probe\look --controls 12
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BAR_HIGH = 0.78
BAR_LOW = 0.625
DOT_FILL_LOW = 0.90
MIN_DIGIT_HEIGHT = 60
VPOS_LOW, VPOS_HIGH = 0.515, 0.620      # 绝对阈值, 实测 16.88 倍富集
NEU_HIGH = 20                            # 墨色通道极差
GATE_LO, GATE_HI = 0.97, 1.03            # 真正的叉号页集中在这个区间


def num(r, k, d=0.0):
    try:
        return float(r[k] or 0)
    except (KeyError, ValueError, TypeError):
        return d


def is_photo(w, h):
    """疑似翻拍: 相机分辨率, 或长宽比不像手机屏。"""
    if not w or not h:
        return False
    return w * h >= 6_000_000 or max(w, h) / min(w, h) < 1.7


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="把可疑图和对照组一起拷出来")
    ap.add_argument("--csv", type=Path, required=True, help="dot_scan.py 的 --out CSV")
    ap.add_argument("--src", type=Path, required=True, help="图库根目录")
    ap.add_argument("--out", type=Path, required=True, help="拷到哪里")
    ap.add_argument("--controls", type=int, default=12, help="随机真图对照组拷几张")
    ap.add_argument("--max-per-kind", type=int, default=10, help="每类最多拷几张")
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    print(f"{args.csv.name}: 读 {len(rows):,} 行")
    has_vpos = "bar_vpos" in rows[0]
    has_ink = "ink_b" in rows[0]
    if not has_vpos:
        print("★ 这份 CSV 没有 bar_vpos 列, 跳过 VPOS 那几类")

    ok, tag, gate_tail = [], {}, []
    for r in rows:
        w, h = num(r, "W"), num(r, "H")
        if num(r, "mh") < MIN_DIGIT_HEIGHT or is_photo(w, h):
            continue

        # 被商家页闸挡下、但图标比例不在真叉号那个区间 -> 疑似误闸
        if r.get("page") == "close":
            iw, ih = num(r, "icon_w"), num(r, "icon_h")
            if iw > 0 and ih > 0 and not (GATE_LO <= ih / iw <= GATE_HI):
                gate_tail.append(r["name"])
            continue

        if r.get("page") != "back":
            continue
        ok.append(r)

        bar = num(r, "bar_ratio")
        why = []
        if bar >= BAR_HIGH:
            why.append("MINUSHI")
        elif 0 < bar < BAR_LOW:
            why.append("MINUSLO")

        # 小数点只在 png 且非翻拍上算
        fill = num(r, "dot_fill")
        if fill and fill < DOT_FILL_LOW and r.get("fmt") == "png":
            why.append("DOT")

        if has_ink:
            neu = (max(num(r, "ink_b"), num(r, "ink_g"), num(r, "ink_r"))
                   - min(num(r, "ink_b"), num(r, "ink_g"), num(r, "ink_r")))
            if neu > NEU_HIGH:
                why.append("NEU")

        if has_vpos:
            v = num(r, "bar_vpos")
            if v and v > VPOS_HIGH:
                why.append("VPOSHI")
            elif v and v < VPOS_LOW:
                why.append("VPOSLO")

        if why:
            tag[r["name"]] = why

    print(f"过闸可判 {len(ok):,} 张, 命中 {len(tag)} 张, 疑似误闸 {len(gate_tail)} 张")
    for k, c in Counter("+".join(v) for v in tag.values()).most_common(12):
        print(f"    {k:<26} {c}")

    # ★ 最该看的: 只有新判据抓到、老判据全漏的
    vpos_only = [n for n, v in tag.items()
                 if any(x.startswith("VPOS") for x in v)
                 and not any(x in ("MINUSHI", "MINUSLO", "DOT") for x in v)]
    both = [n for n, v in tag.items() if len(v) >= 2]
    print(f"★ 只有负号竖直位置抓到、老判据全漏的 {len(vpos_only)} 张 —— 这批是新增覆盖, 最该看")
    print(f"★ 多条同时命中的 {len(both)} 张, 置信度最高")

    clean = [r["name"] for r in ok if r["name"] not in tag]
    rng = random.Random(args.seed)
    rng.shuffle(clean)
    rng.shuffle(vpos_only)
    rng.shuffle(gate_tail)
    controls = clean[: args.controls]
    print(f"对照组随机抽 {len(controls)} 张真图(CTRL_ 前缀), **混在一起看, 别只看报出来的**")

    print("\n建索引(图库大的话要等一会)...", flush=True)
    idx = {}
    for p in args.src.rglob("*"):
        if p.suffix.lower() in _EXTS:
            idx.setdefault(p.name, p)
    print(f"图库 {len(idx):,} 个文件")

    args.out.mkdir(parents=True, exist_ok=True)
    per = Counter()
    got = miss = 0

    def copy(name, prefix):
        nonlocal got, miss
        src = idx.get(name)
        if src is None:
            miss += 1
            return
        shutil.copy2(src, args.out / (prefix + name))
        got += 1

    for name in vpos_only[: args.max_per_kind]:
        copy(name, "VPOSONLY_"); per["VPOSONLY"] += 1
    for name in both[: args.max_per_kind]:
        copy(name, "BOTH_"); per["BOTH"] += 1
    for name, v in tag.items():
        if name in vpos_only or len(v) >= 2:
            continue
        k = v[0]
        if per[k] >= args.max_per_kind:
            continue
        copy(name, k + "_"); per[k] += 1
    for name in gate_tail[: args.max_per_kind]:
        copy(name, "GATETAIL_"); per["GATETAIL"] += 1
    for name in controls:
        copy(name, "CTRL_"); per["CTRL"] += 1

    print(f"\n拷了 {got} 张 -> {args.out}   图库里找不到的 {miss} 张")
    for k, c in per.most_common():
        print(f"    {k:<12} {c}")
    print("\n★ 文件名前缀就是命中原因。VPOSONLY_ 和 BOTH_ 先看, CTRL_ 是真图对照。")
    print("★ 看的时候不要只看报出来的 —— 对照组长什么样才是判断的基准。")


if __name__ == "__main__":
    main()
