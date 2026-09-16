"""出一份**训练用的名单**: 每张图带上判据结果和建议, 不删文件。

为什么是打标不是删
------------------
★★★ 卡 big_ratio 这一条, 会为了去掉约 200 张不是回单的, 连带丢掉约 360 张真回单。
   而"要不要那些没有目标的图", 取决于训练脚本怎么处理无目标样本 ——
   对**检测**来说, 没有目标的图不是废数据, 是负样本。

   这件事在这里定不了, 所以**不删**, 只在名单上标清楚, 让用的人自己筛。

列
--
    file          文件名
    pinyin_score  挑图时的分数(来自 _manifest.csv)
    source        原始路径
    coverage      有注音的行 / 总行数
    spread        注音行纵向铺开多少
    flat          小块标准差下四分位(原始截图 = 0)
    big_ratio     最高那行字 / 正文字高(回单有大金额)
    tier          A / B / C  见下
    reason        为什么不是 A

分档
----
    A   三条全过: 有拼音 + 原始截图 + 像回单       -> 拿来就能用
    B   有拼音 + 原始截图, 但 big_ratio 偏低       -> 混着约三成不是回单的
    C   拼音判据或截图判据没过                    -> 别用

用法
----
    python make_training_list.py --manifest 归集目录\\_manifest.csv `
        --verify verify.csv --out 训练名单.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# 这三条线都是在已知数据上量出来的, 出处见 verify_pinyin.py 的注释
MIN_COVERAGE = 0.10      # 61 张已知非拼音图全部落在这条线以下
MAX_FLAT = 0.01          # 939 张真实截图里 934 张精确等于 0
MIN_BIG_RATIO = 3.0      # 回单金额中位是正文字高的 4.5 倍


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--verify", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    for p in (a.manifest, a.verify):
        if not p.exists():
            print(f"不在: {p}")
            return

    man = {}
    with a.manifest.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            man[r["file"]] = r

    rows = []
    with a.verify.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # ★ 两份对不上就停 —— 硬算出来的名单是假的
    missing = [r["file"] for r in rows if r["file"] not in man]
    if missing:
        print(f"★★ 有 {len(missing)} 张在核查结果里但不在归集清单里, 例如 {missing[:3]}")
        print("   两份对不上, 先查清楚再出名单。")
        return

    out = []
    tally = {"A": 0, "B": 0, "C": 0}
    for r in rows:
        cov = fnum(r.get("coverage"))
        flat = fnum(r.get("flat"))
        big = fnum(r.get("big_ratio"))
        why = []
        if r.get("err"):
            why.append(f"读不了({r['err']})")
        if cov is None or cov < MIN_COVERAGE:
            why.append("拼音覆盖率太低")
        if flat is None or flat > MAX_FLAT:
            why.append("不是原始截图")

        if why:
            tier = "C"
        elif big is None or big < MIN_BIG_RATIO:
            tier = "B"
            why.append("没有明显的大金额, 可能不是回单")
        else:
            tier = "A"
        tally[tier] += 1

        m = man[r["file"]]
        out.append({
            "file": r["file"],
            "tier": tier,
            "pinyin_score": m.get("score", ""),
            "coverage": r.get("coverage", ""),
            "spread": r.get("spread", ""),
            "flat": r.get("flat", ""),
            "big_ratio": r.get("big_ratio", ""),
            "reason": ",  ".join(why),
            "source": m.get("source", ""),
        })

    out.sort(key=lambda r: (r["tier"], -(fnum(r["coverage"]) or 0)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n = len(out)
    print("=" * 66)
    print(f"训练名单 -> {a.out}    共 {n:,} 张")
    print("=" * 66)
    print(f"  A  三条全过, 拿来就能用          {tally['A']:6,d}")
    print(f"  B  有拼音是截图, 但不像回单      {tally['B']:6,d}")
    print(f"  C  拼音或截图判据没过, 别用      {tally['C']:6,d}")
    print()
    ab = tally["A"] + tally["B"]
    print(f"  ★ A          {tally['A']:6,d}   离一万差 {max(0, 10000-tally['A']):,}")
    print(f"  ★ A + B      {ab:6,d}   离一万差 {max(0, 10000-ab):,}")
    print()
    print("★ B 那一档抽样看过: 12 张里 7 张是真回单, 4 张不是(浏览器页/空白页),")
    print("  1 张说不准。要不要用, 看训练脚本吃不吃没有目标的图。")


if __name__ == "__main__":
    main()
