r"""从**没进过标定池**的真图里抽一份留出样本 —— 拿来做**样本外**误杀率验证。

为什么需要它
------------
现在对外的 **2.7/万 和 17.4/万 都是样本内的数**: 阈值取的就是**那个池子里第 k+1 高的分**,
所以在同一个池子上量, 天然合身。**从没在别的图上验过。**

2026-08-26 盘出来: `D:\download2` 下面一共 **1,360,090** 张真图,
而标定池只用了其中 **90,377 张(6.6%)**。也就是说有 **126 万张从没打过分** ——
**足够做一次真正的留出验证。**

  `BlueImages`  487,045 张, 在池 32,378(6.6%)  -> 蓝图
  `OtherImages` 873,045 张, 在池 57,999(6.6%)  -> 白图
  (57,999 与 `gen100k_white/summary.csv` 的行数**一模一样**, 版式对应关系由此坐实)

怎么抽才有可比性
----------------
1. **只抽池外的** —— 在池子里的一律排除, 否则就不是留出了
2. **按池子的版式比例抽**(蓝 35.8% / 白 64.2%) —— 比例变了, 分数分布跟着变,
   到时候分不清是"阈值过拟合"还是"版式配比不同"
3. **已知污染先排掉** —— 已有的排除名单(确认假图/水印命中/造样本源图)按文件名剔,
   剩下的翻拍和 AIGC 元数据, 打完分之后再用 `screenshot_filter` / `watermark_scan` 走一遍,
   **和当初标定池同一套流程**, 否则两个数不可比

★ **一条必须写明的偏差**: 标定池经过**人工看榜首**清洗过, 而这份没有。
`DEPLOY_SPEC` 短板 0b 说过, 那次人工清洗里**有一张肉眼完全看不出破绽**。
所以留出样本里**必然残留一些查不出来的 AI 假图**, 它们会把误杀率**顶高**。
-> **留出的数偏高一点是预期之内的, 别把它直接当成"阈值过拟合"的证据。**

只读源图
--------
只在 `--out` 下建**硬链接**(不占额外磁盘, 与 `stage_pool.py` 同一套做法), 不动源目录。
硬链接要求 `--out` 和源图**在同一个盘**; 跨盘会退回复制并给出提醒。

用法
----
  python training/sample_holdout.py --n 30000 --out D:\holdout30k ^
      --pool-csv E:\SSP_Work\probe\accept_seeded\_lineA\summary.csv ^
      --exclude E:\SSP_Work\probe\exclude_big3.txt E:\SSP_Work\probe\used_srcs.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 版式 -> 目录。比例按标定池实测: 蓝 32,378 / 白 57,999
_ROOTS = [
    ("蓝图", Path(r"D:\download2\BlueImages"), 32378),
    ("白图", Path(r"D:\download2\OtherImages"), 57999),
]


def _norm(s: str) -> str:
    return os.path.basename(s.strip().replace("\\", "/"))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="抽池外真图做留出验证(只建硬链接)")
    ap.add_argument("--n", type=int, default=30000, help="总共抽多少张")
    ap.add_argument("--out", type=Path, required=True, help="放硬链接的目录(**要和源图同盘**)")
    ap.add_argument("--pool-csv", type=Path, required=True, help="标定池的线路A summary.csv")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="已有的排除名单, 可多个")
    ap.add_argument("--roots", type=Path, nargs="*", default=None,
                    help="覆盖默认的两个目录(顺序: 蓝图 白图)")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--dry-run", action="store_true", help="只统计不建链接")
    args = ap.parse_args()

    # ---- 池内名单 ----
    pool: set[str] = set()
    with args.pool_csv.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        have = rd.fieldnames or []
        key = "image_name" if "image_name" in have else "image"
        if key not in have:
            raise SystemExit(f"!! {args.pool_csv} 没有文件名列; 表头 = {have}")
        for r in rd:
            pool.add(_norm(r[key]))
    print(f"标定池 {len(pool)} 张(这些要排除, 否则就不是留出)", flush=True)

    # ---- 已知污染名单 ----
    bad: set[str] = set()
    for p in args.exclude:
        if p.is_file():
            n0 = len(bad)
            bad |= {_norm(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()}
            print(f"  排除名单 {p.name}: +{len(bad) - n0}", flush=True)
        else:
            print(f"  跳过(不存在) {p}", flush=True)
    print(f"已知污染合计 {len(bad)} 个\n", flush=True)

    roots = _ROOTS
    if args.roots:
        if len(args.roots) != 2:
            raise SystemExit("!! --roots 要正好两个(蓝图 白图)")
        roots = [(_ROOTS[i][0], args.roots[i], _ROOTS[i][2]) for i in range(2)]

    tot_w = sum(w for _, _, w in roots)
    rng = random.Random(args.seed)
    picked: list[tuple[str, Path]] = []

    for label, root, w in roots:
        want = round(args.n * w / tot_w)
        if not root.is_dir():
            print(f"!! {root} 不存在, 跳过", flush=True)
            continue
        print(f"扫 {label} {root} ...", flush=True)
        cand: list[str] = []
        seen = 0
        for dp, _, fns in os.walk(root):
            for fn in fns:
                seen += 1
                if os.path.splitext(fn)[1].lower() not in _EXTS:
                    continue
                if fn in pool or fn in bad:
                    continue
                cand.append(os.path.join(dp, fn))
            if seen and seen % 200000 == 0:
                print(f"    已扫 {seen} 个条目, 候选 {len(cand)}", flush=True)
        print(f"  {label}: 候选(池外且不在排除名单) {len(cand)} 张, 目标抽 {want} 张", flush=True)
        if len(cand) < want:
            print(f"  !! 候选不够, 全取 {len(cand)} 张", flush=True)
            want = len(cand)
        for s in rng.sample(cand, want):
            picked.append((label, Path(s)))

    if not picked:
        raise SystemExit("!! 一张都没抽到")

    nb = sum(1 for l, _ in picked if l == "蓝图")
    print(f"\n共抽 {len(picked)} 张: 蓝 {nb} ({100.0*nb/len(picked):.1f}%) / "
          f"白 {len(picked)-nb} ({100.0*(len(picked)-nb)/len(picked):.1f}%)")
    print(f"  (标定池是 蓝 {100.0*32378/90377:.1f}% / 白 {100.0*57999/90377:.1f}%)", flush=True)

    if args.dry_run:
        print("\n--dry-run: 不建链接, 到此为止")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    linked = copied = failed = 0
    for _, src in picked:
        dst = args.out / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)              # 硬链接: 不占额外磁盘
            linked += 1
        except OSError:
            try:
                import shutil
                shutil.copy2(src, dst)     # 跨盘只能复制
                copied += 1
            except Exception:
                failed += 1
    print(f"\n-> {args.out}")
    print(f"   硬链接 {linked}   复制 {copied}   失败 {failed}")
    if copied:
        print("   ⚠ 有复制发生 = --out 和源图**不在同一个盘**, 会实实在在占磁盘。")
        print("     想省磁盘就把 --out 放到 D: 上再来一次。")

    print("\n接下来(和当初标定池**同一套流程**, 否则两个数不可比):")
    print("  1. 打分: predict_seeded(线路A) + predict_tiled(线路B)")
    print("  2. screenshot_filter 出翻拍/异形名单")
    print("  3. watermark_scan 出 AIGC 元数据名单")
    print("  4. ssp_decide --exclude 吃上面两份名单, 出自动拒/人工复核比例")
    print("\n★ 记住那条偏差: 这份**没有经过人工看榜首**那道清洗,")
    print("  所以必然残留查不出来的 AI 假图, 误杀率**天然偏高一点** —— 别直接当成阈值过拟合。")


if __name__ == "__main__":
    main()
