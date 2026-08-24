r"""**不用源图**审裁块: 直接比 `ai/X` 和 `nature/X` 像不像 —— 补上审不了的那 71.5%。

为什么要绕开源图
----------------
`build_crop_dataset --audit-tags` 靠**源图**判"这组是不是蓝图源图走了白图定位器"。
2026-08-24 实跑: 四个源根(含 `D:\download2\BlueImages`)全都在、索引了 61 万张图,
**九组本地 VAE 裁块(13,496 对 = 71.5%)仍然全是"源图查不到, 判不了"**。
多半是短板 0c 那件事的后果 —— 原始 `gen100k_p0..p3` 被删过, 只找回 90.4%。

但**要查的那个毛病根本不需要源图**:

> 短板 11 原文: 那块区域在合成图里**根本没被改过**, 所以 `ai/` 和 `nature/` 两张裁块
> **几乎逐像素相同却挂着相反的标签** —— 这正是"标签噪声过半 -> 训练卡在瞎猜(loss 0.69)"。

`ai/X` 和 `nature/X` 就在手上, **直接对着比**即可。源图只能告诉我们**为什么**切错,
而这份告诉我们**切没切错**, 后者才是训练要的答案。

自检
----
四个已知该剔的组(`apiqwentrain` `apiqwentrain2` `apiseedblue` `apiwanblue`)**必须**
被判成"几乎相同"。判不出来就说明这个方法不成立, **那九组的结论也不能信** ——
脚本会明说, 而不是让人拿着一堆数字自己猜。

只读
----
只读裁块图片, 不写任何东西, 不碰模型也不碰数据集。

用法
----
  python training/crop_pair_audit.py --crops E:\SSP_Work\localcrops --sample 120
"""
from __future__ import annotations

import argparse
import random
import re
import statistics as st
import sys
from pathlib import Path

# 已知该剔的四组 —— 拿它们当"阳性对照"验方法本身
KNOWN_BAD = {"apiqwentrain", "apiqwentrain2", "apiseedblue", "apiwanblue"}

_TAG = re.compile(r"^([A-Za-z][A-Za-z0-9\-]*?)_\d")


def tag_of(name: str) -> str:
    """裁块文件名形如 `crop_<tag>_00042.jpg` 或 `<tag>_00042.jpg`。"""
    s = name
    if s.startswith("crop_"):
        s = s[5:]
    m = _TAG.match(s)
    return m.group(1) if m else "(未知)"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="不用源图的裁块配对审计(只读)")
    ap.add_argument("--crops", type=Path, required=True, help="含 ai/ 与 nature/ 的目录")
    ap.add_argument("--sample", type=int, default=120, help="每组抽多少对")
    ap.add_argument("--diff-thr", type=int, default=8, help="单像素算'变了'的阈值(0~255)")
    ap.add_argument("--same-pct", type=float, default=2.0,
                    help="改动像素占比低于这个百分数, 就算'几乎相同'(疑似标签反了)")
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    import numpy as np                                    # noqa: E402
    from PIL import Image                                 # noqa: E402

    aid, nad = args.crops / "ai", args.crops / "nature"
    if not aid.is_dir() or not nad.is_dir():
        raise SystemExit(f"!! {args.crops} 下没有 ai/ 和 nature/")

    nature = {p.name for p in nad.iterdir() if p.is_file()}
    groups: dict[str, list[str]] = {}
    for p in aid.iterdir():
        if p.is_file() and p.name in nature:
            groups.setdefault(tag_of(p.name), []).append(p.name)
    if not groups:
        raise SystemExit("!! ai/ 与 nature/ 没有同名配对")
    print(f"配对成功 {sum(len(v) for v in groups.values())} 对, 共 {len(groups)} 组\n", flush=True)

    rng = random.Random(args.seed)
    rows = []
    for tag in sorted(groups, key=lambda t: -len(groups[t])):
        names = groups[tag]
        pick = names if len(names) <= args.sample else rng.sample(names, args.sample)
        pcts, mads, bad = [], [], 0
        for nm in pick:
            try:
                a = np.asarray(Image.open(aid / nm).convert("RGB"), np.int16)
                b = np.asarray(Image.open(nad / nm).convert("RGB"), np.int16)
            except Exception:
                continue
            if a.shape != b.shape:
                continue
            d = np.abs(a - b)
            pct = float((d.max(axis=2) > args.diff_thr).mean() * 100.0)
            pcts.append(pct)
            mads.append(float(d.mean()))
        if not pcts:
            continue
        med = st.median(pcts)
        same = sum(1 for x in pcts if x < args.same_pct)
        rows.append((tag, len(names), len(pcts), med, st.median(mads),
                     100.0 * same / len(pcts)))
        print(f"  {tag:<20} {len(names):>6} 对  抽 {len(pcts):>3}"
              f"   改动像素中位 {med:>6.2f}%   平均绝对差 {st.median(mads):>6.2f}"
              f"   几乎相同占 {100.0*same/len(pcts):>5.1f}%", flush=True)

    # ---- 自检: 四个已知坏组必须被判出来 ----
    print("\n" + "=" * 96)
    known = [r for r in rows if r[0] in KNOWN_BAD]
    other = [r for r in rows if r[0] not in KNOWN_BAD]
    if not known:
        print("!! 没找到那四个已知坏组, 无法自检 —— 下面的结论不要信。")
        raise SystemExit(1)
    k_same = st.median([r[5] for r in known])
    print(f"自检: 四个已知该剔的组, '几乎相同'占比中位 = {k_same:.1f}%")
    if k_same < 50.0:
        print("!! 自检没过: 已知坏组并没有表现为'几乎相同'。")
        print("   说明这个判据不成立(或者 --same-pct / --diff-thr 定得不对),")
        print("   **那九组本地 VAE 的结论同样不能信。**")
        raise SystemExit(1)
    print("   ^ 通过: 判据能认出已知的坏组, 可以拿它去看那九组\n")

    print("按'几乎相同'占比排(越高越可疑):")
    for tag, n, s, med, mad, same in sorted(rows, key=lambda r: -r[5]):
        mark = " <- 已知该剔" if tag in KNOWN_BAD else (" ★ 新发现" if same >= k_same * 0.6 else "")
        print(f"  {tag:<20} {n:>6} 对   几乎相同 {same:>5.1f}%   改动中位 {med:>6.2f}%{mark}")
    print("=" * 96)

    susp = [r for r in other if r[5] >= k_same * 0.6]
    if susp:
        tot = sum(r[1] for r in susp)
        print(f"\n★ 除已知四组外, 另有 {len(susp)} 组表现相近, 合计 **{tot} 对**:")
        print("   " + " ".join(r[0] for r in susp))
        print("   -> 这些也该进 --exclude-tags。**先人工看几张确认再剔, 别只信一个数。**")
    else:
        print("\n★ 除已知四组外, 没有别的组表现出'几乎相同' —— "
              "那 71.5% 审不了的部分, 至少在这个判据上是干净的。")


if __name__ == "__main__":
    main()
