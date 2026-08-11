r"""把 gen_local_ai_edit --save-crops 产出的配对裁块, 组装成官方 SSP 训练目录结构。

输入(配对): <crops>/ai/crop_000123.jpg 与 <crops>/nature/crop_000123.jpg
  —— 同一张原图 同一位置, 内容一致, 只差有没有被 AI 动过。
输出(官方 loader 硬编码的结构):
  <out>/imagenet_ai_0419_sdv4/{train,val}/{nature,ai}

**按配对切分**: 同一个编号的 ai 和 nature 必须落在同一侧(都进 train 或都进 val)。
否则同一块内容既在训练集又在验证集 -> 验证分数虚高(泄漏)。

用法:
  python training/build_crop_dataset.py --crops D:\localcrops --out D:\ssp_local --val-frac 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_DIR = "imagenet_ai_0419_sdv4"
_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="配对裁块 -> SSP 训练目录")
    ap.add_argument("--crops", type=Path, required=True, help="含 ai/ 与 nature/ 的目录")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-tags", nargs="*", default=None, metavar="TAG",
                    help="按 tag 剔掉某几组裁块(文件名形如 crop_<tag>_<编号>.jpg)。"
                         "**2026-08-11 加**: API 那条路的蓝图裁块切的是红包促销卡不是金额区"
                         "(gen_api_local_edit 当时写死用白图定位器), 要把那几组剔掉重训。"
                         "本地 VAE 那条路一直用的 locate_amount_auto, 蓝图裁块是对的, 不用剔。")
    ap.add_argument("--only-tags", nargs="*", default=None, metavar="TAG",
                    help="反过来, 只保留这几组(和 --exclude-tags 互斥)")
    ap.add_argument("--list-tags", action="store_true", help="只列出有哪些 tag 和各自多少对, 不建数据集")
    args = ap.parse_args()
    if args.exclude_tags and args.only_tags:
        raise SystemExit("--exclude-tags 和 --only-tags 只能给一个")

    ai_dir, nat_dir = args.crops / "ai", args.crops / "nature"
    if not ai_dir.is_dir() or not nat_dir.is_dir():
        raise SystemExit(f"{args.crops} 下面要有 ai/ 和 nature/ 两个目录(先跑 gen_local_ai_edit --save-crops)")

    ai_names = {p.name for p in ai_dir.iterdir() if p.suffix.lower() in _EXTS}
    nat_names = {p.name for p in nat_dir.iterdir() if p.suffix.lower() in _EXTS}
    pairs = sorted(ai_names & nat_names)          # 只用配得上对的
    if not pairs:
        raise SystemExit("没有配对的裁块(ai/ 与 nature/ 里没有同名文件)")
    lonely = len(ai_names ^ nat_names)

    def tag_of(nm: str) -> str:
        """crop_<tag>_<编号>.jpg -> tag。**从右边切**, tag 里带下划线也不会切错。"""
        stem = Path(nm).stem
        if not stem.startswith("crop_"):
            return "(命名不认识)"
        return stem[len("crop_"):].rpartition("_")[0] or "(命名不认识)"

    from collections import Counter
    tags = Counter(tag_of(nm) for nm in pairs)
    if args.list_tags:
        print(f"配对 {len(pairs)} 对, 共 {len(tags)} 组:")
        for t, c in sorted(tags.items(), key=lambda kv: -kv[1]):
            print(f"  {t:26s} {c:6d} 对")
        print("\n要剔掉某几组: --exclude-tags <tag1> <tag2> ...")
        return

    if args.exclude_tags:
        drop = set(args.exclude_tags)
        unknown = drop - set(tags)
        if unknown:
            raise SystemExit(f"这几个 tag 在裁块里不存在, 先用 --list-tags 看一眼: {', '.join(sorted(unknown))}")
        before = len(pairs)
        pairs = [nm for nm in pairs if tag_of(nm) not in drop]
        print(f"剔掉 {len(drop)} 组 tag, 去掉 {before - len(pairs)} 对, 剩 {len(pairs)} 对")
    elif args.only_tags:
        keep = set(args.only_tags)
        unknown = keep - set(tags)
        if unknown:
            raise SystemExit(f"这几个 tag 在裁块里不存在, 先用 --list-tags 看一眼: {', '.join(sorted(unknown))}")
        pairs = [nm for nm in pairs if tag_of(nm) in keep]
        print(f"只保留 {len(keep)} 组 tag, 剩 {len(pairs)} 对")
    if not pairs:
        raise SystemExit("剔完之后没有裁块了")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    k = max(1, int(len(pairs) * args.val_frac))
    val_set, train_set = pairs[:k], pairs[k:]     # 按"对"切, 同一对不跨集合(防泄漏)

    base = args.out / _DIR
    dirs = {(sp, cl): base / sp / cl for sp in ("train", "val") for cl in ("nature", "ai")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for split, names in (("train", train_set), ("val", val_set)):
        for nm in names:
            shutil.copy2(ai_dir / nm, dirs[(split, "ai")] / nm)
            shutil.copy2(nat_dir / nm, dirs[(split, "nature")] / nm)

    def n(d: Path) -> int:
        return sum(1 for _ in d.iterdir())

    print(f"配对 {len(pairs)} 对(未配对的忽略 {lonely} 个)")
    used = Counter(tag_of(nm) for nm in pairs)
    print("  各组: " + "  ".join(f"{t}={c}" for t, c in sorted(used.items(), key=lambda kv: -kv[1])))
    print(f"数据集 -> {base}")
    print(f"  train/nature={n(dirs[('train','nature')])}  train/ai={n(dirs[('train','ai')])}")
    print(f"  val/nature  ={n(dirs[('val','nature')])}  val/ai  ={n(dirs[('val','ai')])}")
    print("按'对'切分: 同一块内容不会既在 train 又在 val(防泄漏)。")
    print(f"下一步: cd D:\\SSP-AI-Generated-Image-Detection-main 然后 "
          f"python train_val.py --image_root {args.out} --gpu_id 0 --save_path .\\snapshot\\localdet\\ "
          f"--jpg_prob 0.5 --blur_prob 0.1")


if __name__ == "__main__":
    main()
