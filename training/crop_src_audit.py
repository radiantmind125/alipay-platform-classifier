r"""训练裁块的源图 vs 测试集源图 —— 按源图文件名求交集(只读)。

为什么必须有这个
----------------
`verify_workspace.py:53-56` 已经写死了这条教训: **池子的名字不构成隔离保证。**
训练用一个池、测试用另一个池, 只是个约定; 目录被搬动、合并、清空过之后, 约定就不作数了。
**判断训练测试有没有重叠, 只能靠比对 manifest 里的源图文件名。**

万相比过(重叠 0), **千问和豆包从来没比过**。而 localdet6 这一轮新补进训练的,
正好就是千问白/蓝和豆包蓝。如果这些测试集的源图和训练裁块同源,
那么它们的召回率测的是**记忆效应**, 不是检测能力, 数字不能报。

★ 这和"版式混淆"是同一类错误: **拿一个看起来合理的约定当证据, 而没有去查。**

join 链
-------
    localcrops\ai\crop_<tag>_<NNNNN>.jpg          裁块(训练侧)
      -> apilocal_<tag>_<NNNNN>.jpg               同一个 made 计数器命名(gen_api_local_edit.py:175, 249)
      -> probe\*\manifest.csv 里 file 列匹配的行
      -> 该行的 src 列                            = 源图文件名(表头见 :169, 写行见 :237)

测试侧: 各测试集自己的 manifest.csv 的 src 列。两边按源图 basename 求交集。

顺带查一个静默数据损坏
----------------------
`made` 计数器**每次运行都从 0 重数**(gen_api_local_edit.py:171)。
同一个 `--tag` 跑两次, 第二次会**成对覆盖**第一次的 ai/nature 裁块 ——
`ai` 和 `nature` 依然一样多, `verify_workspace.py:134` 那条断言照样报 OK, 但样本已经少了。
manifest 是追加模式, 所以**同一个 file 名出现多行 = 发生过覆盖**。这里一并报出来。

用法
----
  python training/crop_src_audit.py --test D:\probe\api_qwen_full D:\probe\api_wan_test
  python training/crop_src_audit.py --test D:\probe\api_qwen_blue_test --crops D:\localcrops\ai

**只读**: 只读 CSV 和文件名, 不改任何东西。重叠 >0 时退出码为 1。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_crop_name(stem: str) -> tuple[str, str] | None:
    """`crop_<tag>_<NNNNN>` -> (tag, NNNNN)。**从右边切**, 这样 tag 里带下划线也不会切错。"""
    if not stem.startswith("crop_"):
        return None
    body = stem[len("crop_"):]
    tag, sep, idx = body.rpartition("_")
    if not sep or not idx.isdigit():
        return None
    return tag, idx


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="查训练裁块与测试集是否同源(只读)")
    ap.add_argument("--crops", type=Path, default=Path(r"D:\localcrops\ai"),
                    help="训练裁块目录(ai 那一侧就够, ai/nature 是成对的)")
    ap.add_argument("--manifest-root", type=Path, default=Path(r"D:\probe"),
                    help="到哪里去找 manifest.csv(扫一级子目录)")
    ap.add_argument("--test", type=Path, nargs="+", required=True,
                    help="测试集的**输入目录**(里面要有 manifest.csv), 可多个")
    ap.add_argument("--show", type=int, default=20, help="重叠时最多列出几个源图名")
    ap.add_argument("--exclude-tags", nargs="*", default=None, metavar="TAG",
                    help="建数据集时 --exclude-tags 剔掉的那些组。**必须和建数据集时一致** —— "
                         "没进训练的裁块不该参与同源判定, 否则会把干净的测试集误报成大面积泄漏。"
                         "(实测: 豆包蓝曾因此被误报 147/147 全泄漏, 真实只有 12。)")
    args = ap.parse_args()
    ex_tags = set(args.exclude_tags) if args.exclude_tags else set()

    # ---- 1. 训练裁块 -> 生成图文件名 ----
    if not args.crops.is_dir():
        raise SystemExit(f"裁块目录不存在: {args.crops}")
    owner: dict[str, str] = {}          # apilocal_xxx.jpg -> tag
    per_tag: dict[str, int] = defaultdict(int)
    n_crop = n_bad = 0
    for p in args.crops.iterdir():
        if not p.is_file() or p.suffix.lower() not in _EXT:
            continue
        n_crop += 1
        parsed = split_crop_name(p.stem)
        if not parsed:
            n_bad += 1
            continue
        tag, idx = parsed
        if tag in ex_tags:
            continue                      # 没进训练的组, 不参与同源判定
        owner[f"apilocal_{tag}_{idx}.jpg"] = tag
        per_tag[tag] += 1
    print("=" * 96)
    print(f"训练裁块 {n_crop} 个 | 文件名可解析 {len(owner)} 个 | 解析不了 {n_bad} 个"
          + (f" | 已按 --exclude-tags 排除 {len(ex_tags)} 组" if ex_tags else ""))
    if not ex_tags:
        print("  注: 没传 --exclude-tags。若建数据集时剔过组, 这里会把它们也算进同源, **会误报**。")
    if n_bad:
        print("  (解析不了的多半是 VAE 那几组, 命名规则不同, 本来就不经过 manifest)")

    # ---- 2. 扫所有 manifest, 把生成图名映射回源图名 ----
    train_src: dict[str, set[str]] = defaultdict(set)
    prov: dict[str, set[str]] = defaultdict(set)
    dup: dict[str, set[str]] = defaultdict(set)     # file -> 多个 src = 发生过覆盖
    n_manifest = 0
    for mp in sorted(args.manifest_root.glob("*/manifest.csv")):
        n_manifest += 1
        try:
            rows = list(csv.DictReader(open(mp, encoding="utf-8-sig")))
        except Exception as exc:  # noqa: BLE001
            print(f"  读不了 {mp}: {exc}")
            continue
        for r in rows:
            f = (r.get("file") or "").strip()
            s = (r.get("src") or "").strip()
            if not f or not s:
                continue
            dup[f].add(Path(s).name)
            tag = owner.get(f)
            if tag:
                train_src[tag].add(Path(s).name)
                prov[tag].add(mp.parent.name)
    print(f"扫到 manifest {n_manifest} 份")

    collided = {f: v for f, v in dup.items() if len(v) > 1}
    if collided:
        print()
        print(f"!!!! {len(collided)} 个生成图文件名对应**多个不同源图** —— 同一个 tag 跑过两次, "
              f"第二次把第一次的裁块成对覆盖了(made 计数器每次从 0 重数)")
        for f, v in list(collided.items())[: args.show]:
            print(f"       {f}  ->  {', '.join(sorted(v))}")
        print("     后果: ai 和 nature 依然一样多, verify_workspace 那条断言照样报 OK, 但样本少了。")

    # ---- 3. 逐 tag 的可追溯情况 ----
    print()
    print("每个训练 tag 的源图追溯情况:")
    for tag in sorted(per_tag):
        where = ", ".join(sorted(prov[tag])) or "**没找到 manifest -> 这组源图不可追溯**"
        print(f"  {tag:26s} 裁块 {per_tag[tag]:6d}  追到源图 {len(train_src[tag]):6d}  来源: {where}")
    allsrc: set[str] = set()
    for v in train_src.values():
        allsrc |= v
    print(f"\n训练侧源图合计 {len(allsrc)} 张(去重后)")

    # ---- 4. 与各测试集求交 ----
    print()
    print("-" * 96)
    print(f"  {'测试集':34s} {'行数':>7s} {'有src':>7s} {'重叠':>6s}   判读")
    worst = 0
    untraceable = 0
    for t in args.test:
        mp = t / "manifest.csv"
        if not mp.exists():
            print(f"  {t.name:34s} {'':7s} {'':7s} {'':6s}   !!!! 没有 manifest.csv, 源图不可追溯")
            untraceable += 1
            continue
        rows = list(csv.DictReader(open(mp, encoding="utf-8-sig")))
        src = [Path((r.get("src") or "").strip()).name for r in rows if (r.get("src") or "").strip()]
        ov = sorted({s for s in src if s in allsrc})
        worst = max(worst, len(ov))
        note = "干净" if not ov else "**重叠 -> 这个召回数是记忆效应, 不能报**"
        if not src:
            note = "!!!! manifest 没有 src 列, 查不了"
            untraceable += 1
        print(f"  {t.name:34s} {len(rows):7d} {len(src):7d} {len(ov):6d}   {note}")
        for s in ov[: args.show]:
            print(f"        {s}")
        if len(ov) > args.show:
            print(f"        ... 还有 {len(ov) - args.show} 个")

    print()
    print("判读:")
    print("  重叠 = 0        -> 这个测试集干净, 召回数可以信。")
    print("  重叠 > 0        -> 测的是记忆效应, **这个数不能报**, 要重建干净留出集。")
    print("  没有 manifest   -> **不等于干净**, 只是查不了。报数时必须标注'未验证同源'。")
    print()
    print("注: 这里只查'同一张源图'。'不同源图但同一个模板'(近重复)查不出来, ")
    print("    要查近重复用 scripts/mine_fake_families.py, 别用 64 位 dHash —— ")
    print("    支付宝版式高度雷同, 64 位 dHash 主要反映版式, 会把整批并成一族(grouping.py:56-58)。")
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main()
