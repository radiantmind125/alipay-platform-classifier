r"""按 CSV 里的文件名, 把标定池**用硬链接**重新搭起来(只读源图)。

为什么需要
----------
标定用的 `D:\probe\gen100k_p0..p3` 四个目录(共 99,990 张)已经被删掉了 —— 应该是为了腾磁盘。
但 CSV 里存的是**原始下载名**, 而 `D:\download2` 底下的原图还在, 所以按名字能找回来。

★ **用硬链接, 不复制。** 同一个卷上的硬链接**不占额外空间**(只多一个目录项),
  而复制一遍是 **40GB**。找不到硬链接可用时**默认拒绝复制**, 免得把磁盘塞满 ——
  真想复制得显式加 `--allow-copy`。

为什么要重搭
------------
线路A 现在改成了**按文件内容定种子**(见 `predict_seeded.py`), 分数会略有变化,
所以 2.7/万 和 18.4/万 必须**重跑一遍验收**确认还站得住。而重跑就要有图。
更长远地说: **池子在, 才谈得上重新标定阈值**; 池子没了, DEPLOY_SPEC 里每个数字
都成了只能引用、无法复现也无法改进的历史值。

用法
----
  python training/stage_pool.py --from-csv D:\probe\gen100k_v7.csv ^
      --search-root D:\download2 --out D:\probe\gen100k_imgs

  # 之后重跑验收
  python training/ssp_decide.py --input D:\probe\gen100k_imgs --score ^
      --exclude D:\probe\exclude_v4_full.txt --out D:\probe\accept_seeded

**只读源图**: 只往 --out 建链接, 一个字节都不改原图。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="按 CSV 用硬链接重搭标定池(只读源图)")
    ap.add_argument("--from-csv", type=Path, required=True, help="里面要有 image_name 这一列")
    ap.add_argument("--search-root", type=Path, required=True, help="到这个目录底下按文件名找原图")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-copy", action="store_true",
                    help="硬链接不可用时改为**复制**。10 万张约 40GB, 所以默认不允许, 要用得显式打开")
    ap.add_argument("--limit", type=int, default=0, help="只搭前 N 张(调试用)")
    args = ap.parse_args()

    # ---- 1. CSV 里要哪些名字 ----
    want: list[str] = []
    with open(args.from_csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            nm = (r.get("image_name") or "").strip() or Path(r.get("image") or "").name
            if nm:
                want.append(nm)
    if not want:
        raise SystemExit(f"{args.from_csv} 里没读到 image_name")
    uniq = sorted(set(want))
    print(f"CSV 里 {len(want):,} 行, 去重后 {len(uniq):,} 个文件名")
    if len(want) != len(uniq):
        print(f"  (有 {len(want) - len(uniq):,} 行重名 —— 重名在这个仓库里坑过四次, 记一笔)")
    if args.limit:
        uniq = uniq[:args.limit]
        print(f"  --limit: 只搭前 {len(uniq):,} 张")

    # ---- 2. 按文件名给 search-root 建索引 ----
    print(f"\n扫描 {args.search_root} 建索引...", flush=True)
    idx: dict[str, Path] = {}
    dup: dict[str, int] = {}
    for p in args.search_root.rglob("*"):
        if p.suffix.lower() in _EXTS and p.is_file():
            if p.name in idx:
                dup[p.name] = dup.get(p.name, 1) + 1
                continue
            idx[p.name] = p
    print(f"  索引到 {len(idx):,} 个文件名" + (f"; **{len(dup):,} 个名字有重名**" if dup else ""))

    # ---- 3. 搭链接 ----
    args.out.mkdir(parents=True, exist_ok=True)
    linked = copied = missing = existed = 0
    miss_eg: list[str] = []
    dup_hit = 0
    for i, nm in enumerate(uniq, 1):
        src = idx.get(nm)
        if src is None:
            missing += 1
            if len(miss_eg) < 5:
                miss_eg.append(nm)
            continue
        if nm in dup:
            dup_hit += 1
        tgt = args.out / nm
        if tgt.exists():
            existed += 1
        else:
            try:
                os.link(src, tgt)
                linked += 1
            except OSError as exc:
                if not args.allow_copy:
                    raise SystemExit(
                        f"\n硬链接建不了({type(exc).__name__}: {str(exc)[:90]})\n"
                        f"  源 {src}\n  目标 {tgt}\n"
                        f"  多半是**源和目标不在同一个卷上**(硬链接跨卷不行)。\n"
                        f"  换一个和 {args.search_root} 同卷的 --out, 或者显式加 --allow-copy;\n"
                        f"  但复制 {len(uniq):,} 张约 **40GB**, 先看看磁盘够不够 —— "
                        f"当初那四个目录就是因为占地方被删的。")
                shutil.copy2(src, tgt)
                copied += 1
        if i % 10000 == 0:
            print(f"  已处理 {i:,}/{len(uniq):,}", flush=True)

    print(f"\n{'=' * 58}")
    print(f"  新建硬链接 {linked:,} | 已存在 {existed:,}" + (f" | 复制 {copied:,}" if copied else ""))
    print(f"  **找不到原图 {missing:,}**" + (f"  例如 {miss_eg[0]}" if miss_eg else ""))
    if dup_hit:
        print(f"  !!!! 其中 {dup_hit:,} 张的文件名在 {args.search_root} 底下**不止一处**, "
              f"只取了排在前面的那一个 —— **未必是当初打分的那一张**")
    print(f"  -> {args.out}")

    got = linked + existed + copied
    if missing:
        print(f"\n  覆盖率 {got / len(uniq) * 100:.1f}% —— 少了 {missing:,} 张。"
              f"\n  少几张不影响'重跑验收看阈值还站不站得住'(总量口径是比例, 不是绝对张数),"
              f"\n  但**重新标定阈值时要留意**: 池子和当初不完全是同一批。")
    print(f"\n下一步 —— 先验明正身(线路B 是确定性的, 拿它当指纹), 再重跑验收:")
    print(f"  python training/ssp_stability.py --from-csv {args.from_csv} \\")
    print(f"      --search-root {args.search_root} --verify-b-csv <历史的线路B summary.csv> \\")
    print(f"      --min-score 0.3 --max-n 200 --k 1 --out D:\\probe\\verify_only")
    print(f"  python training/ssp_decide.py --input {args.out} --score \\")
    print(f"      --exclude <排除名单> --out D:\\probe\\accept_seeded")


if __name__ == "__main__":
    main()
