r"""用**新模型**把上一版打过分的那些批次原样重打一遍 —— 输入目录从旧 CSV 里推出来, 不靠记忆(只读源图)。

为什么要有这个
--------------
换模型做对比, 前提是**两次打的必须是同一批图**。可批次目录散在 `D:\probe` 下面、名字也不统一,
靠记忆手敲九条命令, 迟早会敲错一条 —— 而**敲错的那条不会报错**, 只会安静地给出一个不能比的数。
本项目已经踩过一次: 线路A 用了旧目录的分数, 按裸文件名对齐后交集为 0, 三格并集召回全成了 0。

所以输入目录**不猜**: 直接读旧的 `<批次><旧标记>\summary.csv` 的 `image` 列, 取它们所在的目录。
打完再核对行数, 对不上就大声报出来。

用法
----
  # 默认只看不跑, 先确认它打算跑什么
  python training/rescore_model.py --probe D:\probe --ref-suffix _ld9fix --new-suffix _ld11 ^
      --model D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet11\Net_epoch_best.pth ^
      --ssp-repo D:\SSP

  # 确认无误再真跑
  ... --run

**只读源图**: 只往 `<批次><新标记>` 目录写打分结果。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

# 和 localdet9 那一轮逐字一致 —— 换模型时**只能换模型**, 其余参数一个都不能动,
# 否则就不是单变量对比了(当初把 localdet10 误判成回退, 正是因为同时动了定位器和源图池)。
_FLAGS = ["--roi-amount", "--amount-pad", "0", "--blue-locator",
          "--agg", "top3", "--roi-top", "0.6"]


def _rows(csv_path: Path) -> list[str]:
    with open(csv_path, encoding="utf-8-sig") as fh:
        return [(r.get("image") or "").strip() for r in csv.DictReader(fh)]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="用新模型把旧批次原样重打一遍(输入目录从旧 CSV 推出)")
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--ref-suffix", default="_ld9fix", help="旧标记, 用来找参照批次")
    ap.add_argument("--new-suffix", required=True, help="新标记, 输出目录 = <批次><新标记>")
    ap.add_argument("--model", type=Path, required=True, help="新模型的 .pth")
    ap.add_argument("--ssp-repo", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", default=None, help="只跑这几个批次(给不带后缀的名字)")
    ap.add_argument("--run", action="store_true", help="**真的执行**; 不给就只打印")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"模型不存在: {args.model} —— 先确认训练存到哪儿了")

    refs = sorted(p for p in args.probe.glob(f"*{args.ref_suffix}") if (p / "summary.csv").exists())
    if not refs:
        raise SystemExit(f"{args.probe} 下面没有 *{args.ref_suffix}\\summary.csv —— 先确认 --ref-suffix")

    jobs: list[tuple[str, Path, Path, int]] = []
    for ref in refs:
        cell = ref.name[: -len(args.ref_suffix)]
        if args.only and cell not in args.only:
            continue
        paths = [p for p in _rows(ref / "summary.csv") if p]
        if not paths:
            print(f"  !!!! {ref.name}: summary.csv 里没有 image 列, 跳过")
            continue
        dirs = Counter(str(Path(p).parent) for p in paths)
        if len(dirs) > 1:
            print(f"  !!!! {ref.name}: 这批图来自 **{len(dirs)} 个不同目录**, 直接按目录重打会跑错范围。")
            for d, c in dirs.most_common(5):
                print(f"         {c:6d}  {d}")
            print(f"       -> 用 link_from_csv.py 先把这批图硬链到一个目录再打, **这一格先跳过**")
            continue
        jobs.append((cell, Path(next(iter(dirs))), args.probe / f"{cell}{args.new_suffix}", len(paths)))

    if not jobs:
        raise SystemExit("没有可跑的批次")

    print(f"\n打算重打 {len(jobs)} 个批次, 模型 {args.model.name}:")
    for cell, src, out, n in jobs:
        print(f"  {cell:22s} {n:6d} 张  {src}")
        print(f"  {'':22s}        -> {out}")

    if not args.run:
        print("\n(没给 --run, 只打印没执行。确认无误后加 --run)")
        return

    here = Path(__file__).resolve().parent
    ok, bad = 0, []
    for cell, src, out, n in jobs:
        cmd = [sys.executable, str(here / "predict_tiled.py"),
               "--ssp-repo", str(args.ssp_repo), "--model", str(args.model),
               "--input", str(src), "--output_dir", str(out),
               *_FLAGS, "--device", args.device]
        print(f"\n===== {cell} ({n} 张) =====", flush=True)
        r = subprocess.run(cmd)
        got = len(_rows(out / "summary.csv")) if (out / "summary.csv").exists() else 0
        if r.returncode != 0 or got != n:
            bad.append((cell, n, got, r.returncode))
            print(f"  !!!! {cell}: 参照 {n} 行, 这次 {got} 行, 退出码 {r.returncode}")
        else:
            ok += 1
            print(f"  {cell}: {got} 行, 与参照一致")

    print(f"\n完成 {ok}/{len(jobs)} 个批次")
    if bad:
        print("\n!!!! 下面这几格**行数对不上或跑失败, 不能拿去做对比**:")
        for cell, n, got, rc in bad:
            print(f"     {cell:22s} 参照 {n} -> 这次 {got} (退出码 {rc})")
        print("     行数不一致多半是目录里混进了别的图 —— 用 link_from_csv.py 精确取那一批再打。")
    print()
    print("下一步: 用 union_matrix / autoreject_threshold 把新旧两版在**同一个真图池**上横比。")


if __name__ == "__main__":
    main()
