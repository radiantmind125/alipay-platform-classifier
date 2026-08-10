r"""把整个工作区从头到尾核对一遍 —— 目录搬动/中断之后先跑这个, 再动任何东西。

为什么要有这个
--------------
服务器上的目录被搬动过一次, 之后我连着用 PowerShell 手写核查脚本, 犯了三个错:
1. 应有值算错(把 10,997 写成 9,997), 白虚惊一场;
2. 用了嵌套数组, PowerShell 在管道里会摊平, `$_[0]` 取到的是字符串的第一个字符, 整节误报"文件不存在";
3. **列目录时写了 `Select-Object -First 20`, 核查本身被截断** —— 这条最糟,
   **一个会截断的完整性检查, 比不检查更坏, 因为它给的是虚假的安心。**

所以固化成脚本: **应有值写在一处、不截断、只读不改、能反复跑。**

用法
----
  python training/verify_workspace.py
  python training/verify_workspace.py --root D:\        # 换个盘符

**只读**: 除了打印什么都不做, 不删不移不建。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# ---- 应有值集中写在这里, 改数据只改这一处 ----

# 目录 -> 应有的文件数(含 manifest.csv 之类的话写实际总数; None = 只查存在不查数量)
_DIR_EXPECT: dict[str, int | None] = {
    r"probe\genuine_20k":       20000,   # 标定误杀用的真图池
    r"probe\api_qwen_full":       151,   # 千问 干净留出测试集(150 图 + manifest)
    r"probe\api_wan_test":        201,   # 万相 干净留出测试集(200 图 + manifest)
    r"probe\api_seed_test":      None,   # 豆包 干净留出测试集(还在造, 目标 200)
    r"probe\api_local_seed":      398,   # 豆包 旧测试集(**训练测试同源, 已弃用**)
    r"probe\api_local_wan":       401,   # 万相 旧测试集(**训练测试同源, 已弃用**)
    r"probe\localedit_heldout":   300,   # 白图 留一测试集(sd-vae-ft-ema, 未进训练)
    r"probe\localedit_blue":      300,   # 蓝图 留一测试集
}

# 源图池: 数量会变(还在收集), 所以查**下限**而不是精确值。
# ★教训: 第一版把这两个写成 None(无基准), 结果 download2 只剩 1 个文件却报"全部通过" ——
#   **没有基准的检查等于没检查, 还会给出虚假的安心。宁可给个粗略下限, 也不要留 None。**
_DIR_MIN: dict[str, int] = {
    r"download\TempFakeImages":  50000,   # 造样本的源图池
}
# **`download2` 不在这里, 是故意的。** 它不是图池, 而是下载器 `B6.AI.Tools.Download.exe`
# 的安装目录, 底下的 `TempFakeImages` 是**会被清空的暂存输出**(实测只剩 1 张 2026-08-01 的新图)。
# 拿它当"池子"去查下限只会天天误报。
#
# ★但由此暴露一个更要紧的问题: 既然 download2 的东西会被搬走(很可能并进 download),
#   那"训练用一个池、测试用另一个池"**从来就不是真正的隔离保证**。
#   **判断训练测试有没有重叠, 只能靠比对 manifest 里的源文件名, 不能靠池子的名字。**
#   万相已经这样比过(重叠 0); 千问/豆包也必须比过才能信它们的召回数。

# 打分结果 CSV -> 应有行数
_CSV_EXPECT: dict[str, int] = {
    r"probe\genuine_20k_ld4\summary.csv": 20000,
    r"probe\genuine_20k_v7\summary.csv":  20000,
    r"probe\wanclean_ld4\summary.csv":      200,
    r"probe\qwen_ld4\summary.csv":          150,
    r"probe\seed_ld4\summary.csv":          397,
    r"probe\white_ld3\summary.csv":         300,
    r"probe\blue_ld3\summary.csv":          300,
    r"probe\wan_full_v7\summary.csv":        50,
    r"probe\qwen_v7\summary.csv":           299,
    r"probe\qwenreal_v7\summary.csv":       150,
    r"probe\qwencomp_v7\summary.csv":       150,
    r"probe\seedcomp_v7\summary.csv":       397,
}

# 必须存在的模型和代码
_MUST_EXIST = [
    r"SSP",
    r"SSP-AI-Generated-Image-Detection-main",
    r"SSP-AI-Generated-Image-Detection-main\snapshot\localdet3\Net_epoch_best.pth",
    r"SSP-AI-Generated-Image-Detection-main\snapshot\localdet4\Net_epoch_best.pth",
    r"SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth",
    r"ssp_aigen_v5",
    r"probe\exclude_evidence.txt",
]

# 花过 API 钱的目录 —— **不可再生, 绝不能删**
_PAID = ["api_full", "api_full_qwen", "api_full_qwen_train", "api_full_qwen_train2",
         "api_full_wan_test", "api_full_seed_test", "localcrops"]

_EXCLUDE_LINES = 25


def count_files(p: Path) -> int:
    """数目录下的文件(不递归)。用 scandir, 十几万个文件也很快。"""
    n = 0
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.is_file():
                    n += 1
    except Exception:
        return -1
    return n


def mark(ok: bool | None) -> str:
    return "OK  " if ok is True else ("??  " if ok is None else "!!!!")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="工作区完整性核查(只读)")
    ap.add_argument("--root", default="D:\\", help="数据根目录, 默认 D:\\")
    args = ap.parse_args()
    R = Path(args.root)
    bad = 0

    print("=" * 78)
    print(f"工作区核查  根目录 {R}")
    print("=" * 78)

    print("\n[1] 代码 / 模型 / 关键文件")
    for rel in _MUST_EXIST:
        p = R / rel
        ok = p.exists()
        bad += (not ok)
        print(f"  {mark(ok)} {rel}")

    print("\n[2] 训练裁块池 (ai 和 nature 必须一样多, 否则配对会被静默丢弃)")
    ai, nat = R / "localcrops" / "ai", R / "localcrops" / "nature"
    n_ai, n_nat = count_files(ai), count_files(nat)
    same = (n_ai == n_nat and n_ai > 0)
    bad += (not same)
    print(f"  {mark(same)} ai={n_ai}  nature={n_nat}")
    if n_ai > 0:
        from collections import Counter
        tags = Counter()
        with os.scandir(ai) as it:
            for e in it:
                if e.is_file():
                    parts = Path(e.name).stem.split("_")
                    tags[parts[1] if len(parts) > 1 else "(无tag)"] += 1
        print(f"      分组 (共 {sum(tags.values())} 对):")
        for t, c in sorted(tags.items(), key=lambda kv: -kv[1]):
            print(f"        {t:22s} {c:6d}")

    print("\n[3] 图片目录")
    for rel, exp in _DIR_EXPECT.items():
        p = R / rel
        if not p.exists():
            print(f"  !!!! {rel:30s} 目录不存在")
            bad += 1
            continue
        n = count_files(p)
        ok = None if exp is None else (n == exp)
        bad += (ok is False)
        print(f"  {mark(ok)} {rel:30s} 实际 {n:7d}" + (f"  应有 {exp:7d}" if exp else "  (无基准)"))

    print("\n[3b] 源图池 (查下限, 不查精确值 —— 池子还在增长)")
    for rel, floor in _DIR_MIN.items():
        p = R / rel
        if not p.exists():
            print(f"  !!!! {rel:30s} 目录不存在")
            bad += 1
            continue
        n = count_files(p)
        deep = sum(1 for _ in p.rglob("*") if _.is_file()) if n < floor else n
        ok = (max(n, deep) >= floor)
        bad += (not ok)
        extra = "" if n == deep else f"  (递归数 {deep} —— 文件在子目录里)"
        print(f"  {mark(ok)} {rel:30s} 实际 {n:7d}  下限 {floor:7d}{extra}")

    print("\n[4] 打分结果 CSV 行数")
    for rel, exp in _CSV_EXPECT.items():
        p = R / rel
        if not p.exists():
            print(f"  !!!! {rel:38s} 文件不存在")
            bad += 1
            continue
        try:
            with open(p, encoding="utf-8-sig", newline="") as f:
                n = sum(1 for _ in csv.DictReader(f))
        except Exception as e:
            print(f"  !!!! {rel:38s} 读不了: {e}")
            bad += 1
            continue
        ok = (n == exp)
        bad += (not ok)
        print(f"  {mark(ok)} {rel:38s} 实际 {n:6d}  应有 {exp:6d}")

    print("\n[5] 排除名单")
    ex = R / "probe" / "exclude_evidence.txt"
    if ex.exists():
        n = len([ln for ln in ex.read_text(encoding="utf-8-sig").splitlines() if ln.strip()])
        ok = (n == _EXCLUDE_LINES)
        bad += (not ok)
        print(f"  {mark(ok)} exclude_evidence.txt {n} 行  应有 {_EXCLUDE_LINES}")
    else:
        print("  !!!! exclude_evidence.txt 不存在")
        bad += 1

    print("\n[6] 花过 API 钱的目录在哪 (不可再生, 绝不能删)")
    stray = R / "SSP_work_images"
    for name in _PAID:
        at_root = (R / name).exists()
        at_stray = (stray / name).exists()
        where = "根目录" if at_root else ("**还在 SSP_work_images 里**" if at_stray else "!!! 两处都找不到")
        base = (R / name) if at_root else (stray / name)
        # **要递归数** —— localcrops 底下是 ai/ 和 nature/ 两个子目录, 直接数会得到 0, 看着像空的
        n = sum(1 for _ in base.rglob("*") if _.is_file()) if (at_root or at_stray) else -1
        flag = "OK  " if at_root else ("??  " if at_stray else "!!!!")
        bad += (not at_root and not at_stray)
        print(f"  {flag} {name:24s} {where:28s} 文件 {n}")

    print("\n[7] SSP_work_images 里还剩什么 (**全部列出, 不截断**)")
    if not stray.exists():
        print("  OK   目录不存在 —— 说明已经全部挪回")
    else:
        items = sorted(stray.iterdir(), key=lambda x: x.name)
        print(f"  共 {len(items)} 项:")
        for p in items:
            n = count_files(p) if p.is_dir() else 1
            paid = "  <- 花过钱" if p.name in _PAID else ""
            print(f"    {p.name:34s} {n:8d} 个文件{paid}")

    print("\n[8] 有没有意外嵌套 (挪回时撞名会造成 probe\\probe 这种)")
    nested = 0
    for d in ("probe", "localcrops", "download", "download2"):
        for sub in (d, "SSP_work_images"):
            p = R / d / sub
            if p.exists():
                print(f"  !!!! 发现嵌套: {d}\\{sub}")
                nested += 1
    if not nested:
        print("  OK   没有嵌套")
    bad += nested

    print()
    print("=" * 78)
    if bad:
        print(f"**有 {bad} 处对不上 —— 先解决再往下跑任何东西。**")
    else:
        print("全部通过。可以继续。")
    print("=" * 78)


if __name__ == "__main__":
    main()
