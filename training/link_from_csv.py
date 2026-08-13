r"""把某个 summary.csv 里**恰好那一批图**硬链接到一个新目录, 好让另一条线打同一批图。

为什么需要
----------
并集误杀/并集召回要求**两条线打的是同一批图** —— `combined_threshold` 是按裸文件名对齐的,
对不上的那些直接不算数(实测踩过: 线路A 用了旧目录的分数, 交集为 0, 三格并集召回全成了 0)。

线路B 的大标定池是 `predict_tiled --sample 42000` 随机抽出来的。
线路A 的 `predict_all_models` 没有 `--sample`, 而且**不能指望两个脚本用同一个种子就抽到同一批**:
两边的扩展名过滤集合稍有出入, 候选列表就不一样, `random.sample` 的结果自然也不一样。

**所以别去复现抽样, 直接把 B 已经打过分的那批图链出来。** 名单从 CSV 的 `image` 列来, 一张不多一张不少。

用硬链接不是拷贝: 同一个卷上瞬间完成, 不占额外空间。跨卷时自动退回拷贝。

用法
----
  python training/link_from_csv.py --csv D:\probe\gen100k_blue\summary.csv D:\probe\gen100k_white\summary.csv ^
      --out D:\probe\gen100k_imgs
  # 然后线路A 直接打这个目录:
  #   cd D:\SSP
  #   python predict_all_models.py --model_root <aigen_v7.pth> --input D:\probe\gen100k_imgs --output_dir ... --device cuda

**只读源图**: 只建链接, 不改也不删原图。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="按 summary.csv 的 image 列把那批图链到一个目录(只读源图)")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="predict_tiled 出的 summary.csv, 可多个")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy", action="store_true", help="强制拷贝而不是硬链接(跨卷时会自动退回拷贝)")
    args = ap.parse_args()

    want: dict[str, Path] = {}          # basename -> 源路径
    dup = 0
    for c in args.csv:
        n = 0
        for r in csv.DictReader(open(c, encoding="utf-8-sig")):
            p = (r.get("image") or "").strip()
            if not p:
                continue
            n += 1
            nm = Path(p).name
            if nm in want and want[nm] != Path(p):
                dup += 1                # 不同目录下同名 —— 会互相覆盖, 必须报出来
                continue
            want[nm] = Path(p)
        print(f"  {c} -> {n} 行")
    if dup:
        print(f"  !!!! {dup} 个**同名但不同路径**的文件, 已只保留先出现的那个。"
              f"并集是按裸文件名对齐的, 同名本身就会让两条线对错 —— 值得回头查一下。")
    if not want:
        raise SystemExit("CSV 里没读到 image 列")

    args.out.mkdir(parents=True, exist_ok=True)
    linked = copied = missing = existed = 0
    for nm, src in want.items():
        dst = args.out / nm
        if dst.exists():
            existed += 1
            continue
        if not src.exists():
            missing += 1
            continue
        try:
            if args.copy:
                shutil.copy2(src, dst); copied += 1
            else:
                os.link(src, dst); linked += 1
        except OSError:
            try:
                shutil.copy2(src, dst); copied += 1      # 跨卷 / 不支持硬链接
            except Exception:
                missing += 1
        if (linked + copied) % 20000 == 0 and (linked + copied):
            print(f"  已处理 {linked + copied}...", flush=True)

    print(f"\n目标 {len(want)} 张 -> {args.out}")
    print(f"  硬链接 {linked} | 拷贝 {copied} | 已存在 {existed} | 源图找不到 {missing}")
    if missing:
        print("  !!!! **源图找不到的那些不会进目录**, 两条线的图就对不齐了 —— 先查清楚再往下跑")
    print()
    print("下一步(线路A 打同一批图):")
    print("  cd D:\\SSP")
    print(f"  python predict_all_models.py --model_root <aigen_v7 的 .pth> --input {args.out} "
          f"--output_dir <线路A 输出目录> --device cuda")


if __name__ == "__main__":
    main()
