r"""裁块里**还剩多少信号** —— 拿"重压缩基线"当尺子, 判 `ai/X` 到底比 `nature/X` 多了什么。

要解决的问题
------------
`crop_pair_audit` 直接比 `ai` 和 `nature` 的像素差, **判据不成立**:
四个已知该剔的组差了 29~68% 的像素, 一点不像"几乎逐像素相同"。

原因: `gen_api_local_edit.py:282` 把整张合成图**重存成 JPEG q95**,
而 `nature` 裁块来自原图。**所以哪怕那块区域一个像素都没改, 两张也会差一截** ——
差的是重压缩噪声, 不是内容。拿绝对差当判据, 等于在量压缩强度。

尺子: 每一对**自带**的重压缩基线
--------------------------------
对同一张 `nature` 裁块自己做一次 q95 重压再比, 得到**这一对自己的**基线 `d_base`。
再看实际差 `d_actual = |ai - nature|`:

    ratio = d_actual / d_base

  ratio >> 1  -> 差的部分**超出**重压缩能解释的范围 -> 那块**真的被改过**, 标签对
  ratio ~= 1  -> 差的部分**全都能被重压缩解释** -> 那块**没被改过**(或改动已被压没)
                 -> **这一对提供不了信号**, 不管标签在哲学上对不对

★ 为什么"提供不了信号"就够定性:
  SSP 吃的是**高频纹理**, 而 JPEG 重压缩**恰恰就是把高频抹掉**。
  差异若不超过重压缩基线, 高频那一层**已经没有可学的东西了**。

自检(阳性对照)
---------------
源图审计里判定"**裁块对**"且 60/60 都判得出来的那几组
(`recqwenblue*` `recwanblue` `recseedblue` `apiwan` `apiqwenwhite` `apiseed` `recqwenwhite`)
**必须**给出明显 > 1 的 ratio。给不出来就说明这把尺子本身有问题, 脚本会停下,
而不是让人拿着不可信的数字去删几千对训练数据。

只读
----
只读裁块图片, 不写任何东西。

用法
----
  python training/crop_signal_probe.py --crops E:\SSP_Work\localcrops --sample 100
"""
from __future__ import annotations

import argparse
import io
import random
import re
import statistics as st
import sys
from pathlib import Path

# 源图审计里 60/60 判得出来、且结论是"裁块对"的组 —— 拿它们验尺子
KNOWN_GOOD = {"recqwenblue3", "recqwenblue", "recqwenblue2", "recwanblue", "recseedblue",
              "apiwan", "apiqwenwhite", "apiseed", "recqwenwhite"}
KNOWN_BAD = {"apiqwentrain", "apiqwentrain2", "apiseedblue", "apiwanblue"}

_TAG = re.compile(r"^([A-Za-z][A-Za-z0-9\-]*?)_\d")


def tag_of(name: str) -> str:
    s = name[5:] if name.startswith("crop_") else name
    m = _TAG.match(s)
    return m.group(1) if m else "(未知)"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="裁块信号强度(以重压缩为基线, 只读)")
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--quality", type=int, default=95, help="基线用的 JPEG 质量, 与造样本时一致")
    ap.add_argument("--flat-ratio", type=float, default=1.5,
                    help="ratio 低于它就算'差异被重压缩解释光了'")
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    import numpy as np                                    # noqa: E402
    from PIL import Image                                 # noqa: E402

    aid, nad = args.crops / "ai", args.crops / "nature"
    if not aid.is_dir() or not nad.is_dir():
        raise SystemExit(f"!! {args.crops} 下没有 ai/ 和 nature/")

    nat = {p.name for p in nad.iterdir() if p.is_file()}
    groups: dict[str, list[str]] = {}
    for p in aid.iterdir():
        if p.is_file() and p.name in nat:
            groups.setdefault(tag_of(p.name), []).append(p.name)
    print(f"配对 {sum(len(v) for v in groups.values())} 对, {len(groups)} 组"
          f"  (基线 JPEG q{args.quality})\n", flush=True)

    rng = random.Random(args.seed)
    rows = []
    for tag in sorted(groups, key=lambda t: -len(groups[t])):
        names = groups[tag]
        pick = names if len(names) <= args.sample else rng.sample(names, args.sample)
        ratios, acts, bases = [], [], []
        for nm in pick:
            try:
                a = np.asarray(Image.open(aid / nm).convert("RGB"), np.float32)
                n_im = Image.open(nad / nm).convert("RGB")
                n = np.asarray(n_im, np.float32)
            except Exception:
                continue
            if a.shape != n.shape:
                continue
            # 这一对自己的重压缩基线: nature 再存一次 q95 再读回来
            buf = io.BytesIO()
            n_im.save(buf, "JPEG", quality=args.quality)
            buf.seek(0)
            n2 = np.asarray(Image.open(buf).convert("RGB"), np.float32)
            if n2.shape != n.shape:
                continue
            d_act = float(np.abs(a - n).mean())
            d_base = float(np.abs(n2 - n).mean())
            if d_base < 1e-6:            # 原图本来就是无损的, 基线为 0, 比值没意义
                continue
            ratios.append(d_act / d_base)
            acts.append(d_act)
            bases.append(d_base)
        if not ratios:
            continue
        med = st.median(ratios)
        flat = 100.0 * sum(1 for r in ratios if r < args.flat_ratio) / len(ratios)
        rows.append((tag, len(names), len(ratios), med, st.median(acts), st.median(bases), flat))
        print(f"  {tag:<20} {len(names):>6} 对  抽 {len(ratios):>3}"
              f"   ratio 中位 {med:>7.2f}   实际差 {st.median(acts):>5.2f}"
              f"   基线 {st.median(bases):>5.2f}   无信号占 {flat:>5.1f}%", flush=True)

    print("\n" + "=" * 100)
    good = [r for r in rows if r[0] in KNOWN_GOOD]
    if not good:
        print("!! 没找到已知'裁块对'的组, 无法验尺子 —— 下面不要信。")
        raise SystemExit(1)
    g_med = st.median([r[3] for r in good])
    print(f"自检: 已知'裁块对'的 {len(good)} 组, ratio 中位 = {g_med:.2f}")
    if g_med < 2.0:
        print("!! 自检没过: 确认切对的组也没给出明显 > 1 的 ratio。")
        print("   说明这把尺子量不出'改过 vs 没改过', 结论一律不可信。")
        print("   可能是 --quality 和造样本时不一致(造样本用的是 q95)。")
        raise SystemExit(1)
    print(f"   ^ 通过: 确认切对的组 ratio {g_med:.2f}, 尺子能分辨'改过'\n")

    print("按'无信号占比'排(越高越可疑):")
    for tag, n, s, med, act, base, flat in sorted(rows, key=lambda r: -r[6]):
        mk = ""
        if tag in KNOWN_BAD:
            mk = "  <- 源图审计判该剔"
        elif tag in KNOWN_GOOD:
            mk = "  (已知切对)"
        elif flat >= 50.0:
            mk = "  ★ 新发现"
        print(f"  {tag:<20} {n:>6} 对   无信号 {flat:>5.1f}%   ratio 中位 {med:>7.2f}{mk}")
    print("=" * 100)

    dead = [r for r in rows if r[6] >= 50.0 and r[0] not in KNOWN_BAD]
    if dead:
        tot = sum(r[1] for r in dead)
        print(f"\n★ 除源图审计已点名的四组外, 另有 {len(dead)} 组过半数**无信号**, 合计 **{tot} 对**:")
        for r in dead:
            print(f"     {r[0]:<20} {r[1]:>6} 对   无信号 {r[6]:.1f}%   ratio {r[3]:.2f}")
        print("\n   判读: 这些对里 `ai` 和 `nature` 的差异**不超过重压缩噪声**,")
        print("   而 SSP 学的正是被重压缩抹掉的那层高频 —— **等于没有可学的东西**。")
        print("   ★ 但**先人工看几张再决定剔不剔** —— 一个统计量不该单独决定删几千对数据。")
    else:
        print("\n★ 没有别的组过半数无信号。那 71.5% 审不了的部分, 在这把尺子下是有信号的。")

    if any(r[0] in KNOWN_BAD for r in rows):
        kb = [r for r in rows if r[0] in KNOWN_BAD]
        print(f"\n★ 顺带: 源图审计点名该剔的四组, 无信号占比 "
              f"{'/'.join(f'{r[6]:.0f}%' for r in kb)}, ratio 中位 "
              f"{'/'.join(f'{r[3]:.1f}' for r in kb)}。")
        print("   若它们**并不是**无信号, 说明源图审计那个'该剔'结论(只有 8/60 判得出来)")
        print("   在像素层面得不到支持, 剔之前值得再确认。")


if __name__ == "__main__":
    main()
