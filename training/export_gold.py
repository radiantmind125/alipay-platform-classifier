r"""从验证集里按固定种子抽几百段, 连图带现在的标签导出来, 拿去人工逐条改成"金标准" —— 用来量**真实**准确率。

为什么要金标准(2026-10-12)
--------------------------
eval_rec 报的"整段全对率"是跟**老师 OCR 读出来的标签**比的, 标签本身就有错:
逐条看 40 条白图"错误", 8 条其实是标签错、模型对; 40 条蓝图里还有 15 条是裁图和标签根本对不上。
所以 93% / 81% 既不是模型的真实水平, 也没法拿来比两个模型谁更准。
人工改过的几百段当考卷, 以后每次重训都在同一张考卷上量, 数才说得清。

做法
----
1. 从 <data>/val.txt 里按目录挑(验证集 = 没参加过训练的原图), 每个目录按固定种子抽 N 段
2. 把输入裁图(带拼音那张, 模型看到的就是它)拷到 <out>/img/0001.png ...
3. 写 <out>/_gold_candidates.csv: id, 文件名, 原路径, 来源目录, 现在的标签(老师读的)
4. --models 给了的话, 把这些模型(onnx + 同目录字表)拷到 <out>/models/<名字>/, 本机就能量

用法
----
    python training/export_gold.py --data E:\alipay-ai-data\train-n1009 --out E:\_move_logs\gold ^
        --pick pinyin-pairs-n1009-blue:400 pinyin-pairs-n1009-white:300 ^
        --models E:\alipay-ai-data\rec_pinyin.onnx E:\alipay-ai-data\model-n1009\rec_pinyin.onnx
★ 已经导过(out 里有 _gold_candidates.csv)就不再导, 免得抽到另一批。
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="有 val.txt 的训练目录")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pick", nargs="+", required=True, help="目录名:段数, 例 pinyin-pairs-n1009-blue:400")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--models", type=Path, nargs="*", default=[], help="要一起拷出来的 onnx(同目录的 charset.txt 跟着拷)")
    a = ap.parse_args()

    man = a.out / "_gold_candidates.csv"
    if man.exists():
        print(f"  {man} 已经在了, 不再导(要重导先把 {a.out} 挪走)")
        return
    lines = []
    for ln in (a.data / "val.txt").open(encoding="utf-8"):
        ln = ln.rstrip("\n")
        if "\t" in ln:
            p, t = ln.split("\t", 1)
            lines.append((p, t))
    print(f"  val.txt {len(lines):,} 段")

    picked = []
    for spec in a.pick:
        name, n = spec.rsplit(":", 1)
        key = name.lower()
        pool = [(p, t) for p, t in lines if Path(p).parent.parent.name.lower() == key]
        rnd = random.Random(f"{a.seed}:{key}")
        got = rnd.sample(pool, min(int(n), len(pool)))
        print(f"  {name}: 验证集里 {len(pool):,} 段, 抽 {len(got):,}")
        picked += [(name, p, t) for p, t in got]

    img = a.out / "img"
    img.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, p, t) in enumerate(picked, 1):
        src = Path(p)
        if not src.exists():
            print(f"  ★ 图不在, 跳过: {src}")
            continue
        fn = f"{i:04d}{src.suffix}"
        shutil.copy2(src, img / fn)
        rows.append({"id": i, "file": fn, "src_path": str(src), "dir": name, "label": t})
    with man.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "file", "src_path", "dir", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"  拷了 {len(rows):,} 段 -> {img}")
    print(f"  清单 -> {man}")

    for m in a.models:
        cs = m.parent / "charset.txt"
        if not m.exists() or not cs.exists():
            print(f"  ★ 模型或字表不在: {m}")
            continue
        tag = m.parent.name if m.parent.name.startswith("model") else "delivered"
        d = a.out / "models" / tag
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(m, d / m.name)
        shutil.copy2(cs, d / "charset.txt")
        print(f"  模型 {m} -> {d}")


if __name__ == "__main__":
    main()
