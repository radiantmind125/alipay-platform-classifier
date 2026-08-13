r"""把现成的假图**另存为 JPEG** 再打一遍分 —— 测最省事的那种绕过手法(只读源图)。

为什么这一步必须做
------------------
真图标定池按**扩展名**分开看, 分布差得离谱(D:\download2\BlueImages 抽 4865 张):

| 扩展名 | 张数 | 中位数 | 99分位 | 最高分 | >=0.94 |
|---|---|---|---|---|---|
| .png  | 2707 | 0.0001 | 0.0064 | 0.9260 | 0.00% |
| .jpg  | 2147 | 0.0000 | **0.4152** | 0.9988 | 0.09% |
| .jpeg |   11 | 0.5751 | 0.9710 | 0.9710 | 9.09% |

**99分位差了 65 倍。** 当初测"蓝图/白图要不要分开定线"的时候, 两边只差 0.0002, 于是判定不用分。
格式的影响比版式**大了三个数量级** —— 真图 PNG 的最高分 0.9260 连 1/1000 的线(0.8974)都够不到,
也就是说: **我们量到的每一次误杀, 都是 .jpg 或 .jpeg。整条工作点是被 JPEG 那一半顶上去的。**

反过来就有一个从来没测过的问题: **假图存成 JPEG 之后还抓得到吗?**

线路B 认的是局部重编辑留下的**高频残差**, 而 JPEG 量化干的就是抹高频。
JPEG 编码能把真图的 99分位抬高 65 倍, 说明它对这个信号影响极大 —— 方向是抬是压, 只能实测。

- 假图存成 jpg 分数**掉下来** -> "另存为 jpg" 是零成本绕过, 必须写进短板, 而且要写进对外口径
- 假图存成 jpg 分数**扛得住** -> 这是一条很强的稳健性结论, 可以放进给经理的报告

**这个测试不花 API 钱**: 假图都是现成的, 只是换个编码重打一遍分。

用法
----
  # 先看假图现在都是什么格式(不写任何文件)
  python training/reencode_jpeg.py --csv D:\probe\_clean_ld9fix\*_clean.csv --formats

  # 再把某一格另存成三档质量
  python training/reencode_jpeg.py --csv D:\probe\_clean_ld9fix\wanblue_ld9fix_clean.csv ^
      --out D:\probe\jpegtest\wanblue --quality 95 90 80

**只读源图**: 只往 --out 写新文件, 不改不删原图。

注意
----
另存之后**扩展名变了, 文件名也就变了**。并集(combined_threshold)是按裸文件名对齐的,
所以这一批只能**单独看线路B 的召回**, 不要拿去算并集。要的结论也就是单线召回掉不掉。
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def _rows(patterns: list[str]) -> list[Path]:
    """从一批 summary.csv / *_clean.csv 里取出图片路径。"""
    files: list[str] = []
    for p in patterns:
        hit = glob.glob(p)
        if not hit:
            print(f"  !!!! {p} 一个文件都没匹配上")
        files.extend(hit)
    if not files:
        raise SystemExit("--csv 没匹配到任何文件 —— 先确认路径和通配符")

    out: list[Path] = []
    for c in sorted(files):
        n = 0
        with open(c, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                p = (r.get("image") or r.get("path") or "").strip()
                if p:
                    out.append(Path(p))
                    n += 1
        print(f"  {c} -> {n} 行")
    if not out:
        raise SystemExit("CSV 里没有 image 列 —— 确认传的是 predict_tiled 出的 summary.csv 或 *_clean.csv")
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="假图另存为 JPEG 再打分, 测'另存为 jpg'能不能绕过(只读源图)")
    ap.add_argument("--csv", nargs="+", required=True, help="假图的 summary.csv / *_clean.csv, 支持通配符")
    ap.add_argument("--out", type=Path, default=None, help="输出根目录; 每档质量一个子目录 q95/q90/...")
    ap.add_argument("--quality", type=int, nargs="+", default=[95, 90, 80],
                    help="JPEG 质量档位。手机截图转发/重存常见落在 80~95, 所以默认测这三档")
    ap.add_argument("--formats", action="store_true",
                    help="只统计源图的扩展名分布然后退出, **不写任何文件**")
    ap.add_argument("--limit", type=int, default=0, help="每档最多转这么多张(0=全部), 想先快速看一眼时用")
    args = ap.parse_args()

    src = _rows(args.csv)

    ext = Counter(p.suffix.lower() for p in src)
    print(f"\n源图 {len(src)} 张, 扩展名分布:")
    for k, v in ext.most_common():
        print(f"  {k or '(无扩展名)':10s} {v:6d}  {v / len(src) * 100:5.1f}%")
    if len(ext) == 1:
        only = next(iter(ext))
        print(f"\n  !!!! 假图**全是 {only}**, 而真图池是 png/jpg 混着的。")
        print(f"       这意味着现在这张召回表里, **格式和标签是混淆的** ——")
        print(f"       分不清模型学到的是'被改过'还是'是 {only}'。这一轮转码正是为了拆开这两者。")

    if args.formats:
        print("\n(--formats: 只统计, 没写文件)")
        return
    if not args.out:
        raise SystemExit("要写文件就得给 --out(或者加 --formats 只看统计)")

    for q in args.quality:
        d = args.out / f"q{q}"
        d.mkdir(parents=True, exist_ok=True)
        ok = fail = 0
        todo = src[: args.limit] if args.limit else src
        for i, p in enumerate(todo, 1):
            try:
                with Image.open(p) as im:
                    im.convert("RGB").save(d / (p.stem + ".jpg"), "JPEG",
                                           quality=q, subsampling=0, optimize=True)
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
            if i % 200 == 0:
                print(f"  q{q}: {i}/{len(todo)}...", flush=True)
        print(f"q{q} -> {d}  成功 {ok} | 失败 {fail}")

    print()
    print("下一步(每档都要重打分, 然后和原始那一格并排比):")
    print(r"  $snap='D:\SSP-AI-Generated-Image-Detection-main\snapshot'")
    for q in args.quality:
        d = args.out / f"q{q}"
        print(f"  python training\\predict_tiled.py --ssp-repo D:\\SSP --model \"$snap\\localdet9\\Net_epoch_best.pth\" "
              f"--input {d} --output_dir {d}_scored --roi-amount --amount-pad 0 --blue-locator "
              f"--agg top3 --roi-top 0.6 --device cuda")
        print(f"  Copy-Item {d}_scored\\summary.csv {args.out}\\q{q}.csv")
    print()
    print("  # 同一批真图当分母, 把原始那一格和各档 jpg 一起传进去, 表格里就是并排的几列")
    print(r"  python training\autoreject_threshold.py --col tile_top3 --require-located "
          r"--genuine D:\probe\gen100k_blue\summary.csv D:\probe\gen100k_white\summary.csv "
          r"--exclude D:\probe\exclude_big2.txt --budgets 5000 1000 "
          f"--fake <原始那一格的 _clean.csv> {args.out}\\q95.csv {args.out}\\q90.csv {args.out}\\q80.csv")
    print()
    print("怎么读: **同一行(同一条分数线)上横着比**。")
    print("  各档 jpg 的召回和原始那一格**差不多**   -> 信号扛得住重编码, 这是稳健性结论, 可以对外讲")
    print("  越压召回掉得越狠                        -> '另存为 jpg' 是零成本绕过, 必须如实写进短板")
    print("★ 别只看 q95。真要绕过的人会往死里压, q80 那一档才是下限。")


if __name__ == "__main__":
    main()
