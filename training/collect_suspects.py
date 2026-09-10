r"""把可疑图和**对照组**一起拷出来, 供人工看。

为什么一定要有对照组
--------------------
★ 只看报出来的图, 没法判断"这些看着不对劲"是真的不对劲, 还是**支付宝的图本来就长这样**。
这个项目已经因此错过三次:
  - 「第二个造假工具」 -> 其实是商家账单页, 真页面
  - 「八月翻 6 倍的欺诈活动」 -> 其实是页型占比变了
  - 「¥ 探测器」 -> 报出的 8 张全部带 ¥, 量的根本不是字体
**所以这个脚本一定同时拷一批随机真图(CTRL_ 前缀), 混在一起看。**

判据(和 dot_scan.py / MinusCheck.cs / DotCheck.cs 对齐, 只留真正在跑的)
--------------------------------------------------------------------
  MINUSHI_  负号偏宽: >= 0.78
  MINUSLO_  负号偏窄: < 0.625
  DOT_      小数点是圆的: 填充率 < 0.90, 仅 png、非翻拍、账单详情页
  BOTH_     多条同时命中 —— 置信度最高
  CTRL_     随机真图对照组

★ 这一轮否掉的不再取图: 负号竖直位置(8.43 倍, 加进去让整体变差)、
  墨色中性度(88% 命中是绿色金额这个合法变体)。商家页闸门已定案收紧到 0.97~1.03。

★ 商家账单页(左上角 X 关闭图标, 长宽比 0.97~1.03)一律排除 —— 那个页型金额字体不一样,
  负号天生更宽。

用法
----
  python training/collect_suspects.py --csv probe\inv2.csv ^
      --src D:\download2\OtherImages --out probe\look2
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
    ap.add_argument("--controls", type=int, default=30, help="随机真图对照组拷几张")
    ap.add_argument("--max-per-kind", type=int, default=30, help="每类最多拷几张")
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    print(f"{args.csv.name}: 读 {len(rows):,} 行")

    ok, tag = [], {}
    for r in rows:
        w, h = num(r, "W"), num(r, "H")
        if num(r, "mh") < MIN_DIGIT_HEIGHT or is_photo(w, h):
            continue

        # 商家账单页字体不一样, 整体不判
        iw, ih = num(r, "icon_w"), num(r, "icon_h")
        if iw > 0 and ih > 0 and GATE_LO <= ih / iw <= GATE_HI:
            continue
        if r.get("page") == "other":
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

        if why:
            tag[r["name"]] = why

    print(f"过闸可判 {len(ok):,} 张, 命中 {len(tag)} 张")
    for k, c in Counter("+".join(v) for v in tag.values()).most_common(12):
        print(f"    {k:<26} {c}")

    both = [n for n, v in tag.items() if len(v) >= 2]
    print(f"★ 多条同时命中的 {len(both)} 张, 置信度最高")

    clean = [r["name"] for r in ok if r["name"] not in tag]
    rng = random.Random(args.seed)
    rng.shuffle(clean)
    rng.shuffle(both)
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

    for name in both[: args.max_per_kind]:
        copy(name, "BOTH_"); per["BOTH"] += 1
    for name, v in tag.items():
        if len(v) >= 2:
            continue
        k = v[0]
        if per[k] >= args.max_per_kind:
            continue
        copy(name, k + "_"); per[k] += 1
    for name in controls:
        copy(name, "CTRL_"); per["CTRL"] += 1

    print(f"\n拷了 {got} 张 -> {args.out}   图库里找不到的 {miss} 张")
    for k, c in per.most_common():
        print(f"    {k:<12} {c}")
    print("\n★ 文件名前缀就是命中原因。BOTH_ 先看, CTRL_ 是真图对照。")
    print("★ 看的时候不要只看报出来的 —— 对照组长什么样才是判断的基准。")


if __name__ == "__main__":
    main()
