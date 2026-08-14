r"""验证"无损 PNG 对线路A 是分布外"这个假设 —— 重训上游模型之前先花半小时确认(只读源图)。

假设从哪来
----------
`.jpeg` 剔掉之后, 线路A 真图 `>=0.94` 的构成:
PNG 占池 49.8% 却占高分的 **74.9%**(超配 1.50 倍), **前 15 高分里 13 张是 `.png`**。
也就是说**硬拒那条线几乎完全由无损 PNG 真图定死**。

而阈值有多值钱这一轮刚量过: **线降 0.0376(0.9440->0.9064), 万相整图 +40.0 个点。**
所以"把无损 PNG 真图加进线路A 训练, 把线再压下去"看着很有搞头。

**但那是重训上游 `aigen_v7`, 是真项目。本项目已经四次改进尝试全被实测否掉,
所以先花半小时验证假设, 再谈要不要投入。**

怎么测才算数
------------
**注意: 光换容器毫无意义。** 模型吃的是解码后的 RGB, 把一张 jpg 原样存成 png,
像素一模一样, 分数必然一模一样。所以"PNG 是分布外"**不可能**指文件格式,
只可能指**像素统计**: 从没被 JPEG 量化过的图, 高频成分和量化过的不一样。

于是真正的测法是: **把 PNG 真图做一次 JPEG 量化, 看分数掉不掉。**

**而且必须有对照**: 拿**本来就是 jpg** 的图做同样一次量化。
它们的高频早就被压掉了, 再压一次应该没多大变化。

- PNG 掉得多 而 JPG 几乎不动 -> **假设成立**, 无损高频正是把分数顶高的原因,
  把无损真图加进训练能把线压下去, 重训值得立项
- 两边掉得差不多 -> 掉分只是"又压了一道"的普遍效应, **和 PNG 无关**, 别去重训
- PNG 也不掉 -> 那些图是**内容本身难**, 不是格式问题, 重训救不了

四组各取高分组和随机组: 高分组是**真正定线的那一撮**, 随机组给出基线漂移量。

用法
----
  # 第一步: 挑图 + 转码(只读源图, 只往 --out 写)
  python training/png_ood_probe.py --a D:\probe\gen100k_v7.csv --exclude D:\probe\exclude_big3.txt ^
      --out D:\probe\pngood --n 200

  # 第二步: 用线路A 给四个目录各打一遍分(命令脚本会打印出来)

  # 第三步: 对比
  python training/png_ood_probe.py --out D:\probe\pngood --compare
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_GROUPS = ("png_high", "png_rand", "jpg_high", "jpg_rand")


def _read(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _make(args) -> None:
    drop: set[str] = set()
    if args.exclude and args.exclude.exists():
        drop = {ln.strip().lstrip("\ufeff")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        print(f"(排除名单 {len(drop)} 个)")

    # 路径来源: 线路A 的 CSV 里那一列指向的是**当初打分时用的目录**,
    # 而那批目录很可能是 link_from_csv 建的临时硬链接目录, 早就删掉了。
    # 所以允许用 --paths 从别的 CSV(比如线路B 的 summary.csv, 指向原始下载目录)取路径, 按裸文件名对齐。
    path_of: dict[str, str] = {}
    if args.paths:
        for pc in args.paths:
            n = 0
            for r in _read(pc):
                nm = (r.get("image_name") or "").strip()
                p = (r.get("image") or "").strip()
                if nm and p:
                    path_of[nm] = p
                    n += 1
            print(f"  路径来源 {pc} -> {n} 条")

    rows: list[tuple[float, str, str]] = []
    for r in _read(args.a):
        nm = (r.get("image_name") or "").strip()
        p = path_of.get(nm) or (r.get("image") or "").strip()
        v = (r.get(args.col) or "").strip()
        if not (nm and p and v) or nm in drop:
            continue
        try:
            rows.append((float(v), nm, p))
        except ValueError:
            pass
    if not rows:
        raise SystemExit(f"{args.a} 里没读到 {args.col} —— 确认这是线路A 的 CSV")
    print(f"真图 {len(rows):,} 张")

    # **先探 30 条路径**再干活 —— 全都打不开的话早点停, 别转码 800 张全失败还照样打印下一步命令
    probe = rows[:: max(1, len(rows) // 30)][:30]
    alive = sum(1 for _, _, p in probe if Path(p).exists())
    print(f"  路径抽查: {alive}/{len(probe)} 条能找到文件")
    if alive == 0:
        raise SystemExit(
            f"\n**一条路径都找不到** —— {args.a} 的 image 列多半指向已经删掉的临时目录\n"
            f"(线路A 当初是打 link_from_csv 建的硬链接目录, 那批目录腾空间时清掉了)。\n"
            f"改法: 加 --paths 指向线路B 的 summary.csv, 从那里取原始路径, 按裸文件名对齐 ——\n"
            f"  --paths D:\\probe\\gen100k_blue\\summary.csv D:\\probe\\gen100k_white\\summary.csv")

    by_ext: dict[str, list[tuple[float, str, str]]] = {"png": [], "jpg": []}
    for t in rows:
        e = Path(t[1]).suffix.lower()
        if e == ".png":
            by_ext["png"].append(t)
        elif e == ".jpg":
            by_ext["jpg"].append(t)

    rng = random.Random(args.seed)
    picks: dict[str, list[tuple[float, str, str]]] = {}
    for ext, lst in by_ext.items():
        if len(lst) < args.n * 2:
            raise SystemExit(f".{ext} 只有 {len(lst)} 张, 不够挑 {args.n}x2")
        lst.sort(key=lambda t: -t[0])
        picks[f"{ext}_high"] = lst[: args.n]
        picks[f"{ext}_rand"] = rng.sample(lst[args.n:], args.n)
        print(f"  .{ext:4s} {len(lst):6d} 张 | 高分组分数 {lst[0][0]:.4f}~{lst[args.n-1][0]:.4f} "
              f"| 随机组中位 {np.median([t[0] for t in picks[f'{ext}_rand']]):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    mf = open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig")
    mw = csv.writer(mf)
    mw.writerow(["group", "new_name", "orig_name", "orig_score"])
    total_ok = 0
    for g in _GROUPS:
        d = args.out / g
        d.mkdir(exist_ok=True)
        ok = 0
        why: list[str] = []
        for s, nm, p in picks[g]:
            new = Path(nm).stem + ".jpg"
            try:
                with Image.open(p) as im:
                    # 和两个生成器 `_save_im` 一致的存法, 不另设参数, 免得引入新变量
                    im.convert("RGB").save(d / new, "JPEG", quality=args.quality)
                mw.writerow([g, new, nm, f"{s:.6f}"])
                ok += 1
            except Exception as exc:  # noqa: BLE001
                if len(why) < 3:          # **把原因打出来** —— 只报个失败数等于什么都没说
                    why.append(f"{type(exc).__name__}: {str(exc)[:110]}")
        total_ok += ok
        print(f"  {g:9s} -> {d}  成功 {ok} | 失败 {len(picks[g]) - ok}")
        for w in why:
            print(f"      {w}")
    mf.close()
    if total_ok == 0:
        raise SystemExit("\n**一张都没转成功**, 下一步的命令就不打印了 —— 先看上面的失败原因。")

    print(f"\n-> {args.out / 'manifest.csv'}")
    print("\n下一步: 用线路A 给四个目录各打一遍分")
    print("  cd D:\\SSP")
    for g in _GROUPS:
        print(f"  python predict_all_models.py --model_root <aigen_v7 的 .pth> "
              f"--input {args.out / g} --output_dir {args.out / (g + '_scored')} --device cuda")
    print(f"\n再回来对比: python training/png_ood_probe.py --out {args.out} --compare")


def _compare(args) -> None:
    man = _read(args.out / "manifest.csv")
    if not man:
        raise SystemExit("manifest.csv 是空的 —— 先跑第一步")
    after: dict[str, float] = {}
    for g in _GROUPS:
        sp = args.out / f"{g}_scored" / "summary.csv"
        if not sp.exists():
            print(f"  !!!! 缺 {sp} —— **这一组没打分**, 表会不完整")
            continue
        for r in _read(sp):
            nm = (r.get("image_name") or "").strip()
            v = (r.get(args.col) or "").strip()
            if nm and v:
                try:
                    after[f"{g}|{nm}"] = float(v)
                except ValueError:
                    pass

    print(f"{'组':10s}{'n':>5s}{'转码前中位':>12s}{'转码后中位':>12s}{'中位变化':>10s}"
          f"{'>=0.9064 前':>12s}{'后':>8s}")
    print("-" * 74)
    out: dict[str, tuple[float, float]] = {}
    for g in _GROUPS:
        rs = [r for r in man if r["group"] == g and f"{g}|{r['new_name']}" in after]
        if not rs:
            print(f"{g:10s}{'(没有可比的行)':>30s}")
            continue
        b = np.asarray([float(r["orig_score"]) for r in rs])
        a = np.asarray([after[f"{g}|{r['new_name']}"] for r in rs])
        out[g] = (float(np.median(b)), float(np.median(a)))
        print(f"{g:10s}{len(rs):>5d}{np.median(b):>12.4f}{np.median(a):>12.4f}"
              f"{np.median(a) - np.median(b):>+10.4f}{(b >= 0.9064).mean()*100:>11.1f}%"
              f"{(a >= 0.9064).mean()*100:>7.1f}%")

    print()
    if "png_high" in out and "jpg_high" in out:
        dp = out["png_high"][1] - out["png_high"][0]
        dj = out["jpg_high"][1] - out["jpg_high"][0]
        print(f"高分组中位变化: PNG {dp:+.4f}   JPG {dj:+.4f}")
        print()
        print("怎么判:")
        print("  PNG 掉得明显比 JPG 多  -> **假设成立**, 无损高频就是把分数顶高的原因,")
        print("                           把无损真图加进线路A 训练能把硬拒线压下去, 重训值得立项")
        print("  两边掉得差不多        -> 只是'又压了一道'的普遍效应, **和 PNG 无关**, 别去重训")
        print("  PNG 几乎不掉          -> 那些图是**内容本身难**, 格式不是原因, 重训救不了")
    print()
    print("★ 随机组是基线: 它的变化量代表'转码本身'带来的漂移, 高分组要减掉这个才算净效应。")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="验证无损 PNG 是否是线路A 的分布外输入(只读源图)")
    ap.add_argument("--a", type=Path, default=None, help="线路A 真图 CSV(第一步用)")
    ap.add_argument("--paths", type=Path, nargs="*", default=None,
                    help="**从这些 CSV 取图片路径**(按裸文件名对齐), 而不是用 --a 里的 image 列。"
                         "线路A 当初打的是 link_from_csv 建的临时硬链接目录, 那批目录早清掉了, "
                         "所以要用线路B 的 summary.csv 拿原始下载路径。")
    ap.add_argument("--col", default="final_ai_score")
    ap.add_argument("--exclude", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200, help="每组取多少张")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="第三步: 读回打分结果做对比")
    args = ap.parse_args()

    if args.compare:
        _compare(args)
    elif args.a:
        _make(args)
    else:
        raise SystemExit("第一步要给 --a(线路A 的 CSV); 第三步用 --compare")


if __name__ == "__main__":
    main()
