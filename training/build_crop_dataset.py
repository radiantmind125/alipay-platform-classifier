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
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_DIR = "imagenet_ai_0419_sdv4"
_EXTS = {".jpg", ".jpeg", ".png"}


def audit_tags(crops: Path, manifest_root: Path, src_roots: list[Path], sample: int,
               tag_of, pairs: list[str]) -> None:
    """按**像素**判定每组裁块切得对不对 —— 不看 tag 名字。

    为什么不能看名字: 实测 `apiwantest` 这组其实是**蓝图**, 名字里却没有 blue。
    本项目已经在"拿目录名/参数名当证据"上栽过好几次, 这里一律用像素。

    判据(**精确, 不是颜色启发式**):
      裁块是 `src[sy0+4:sy1-4, sx0+4:sx1-4]`(gen_api_local_edit.py:241), 所以
        裁块高 = 框高 - 8, 裁块宽 = 框宽 - 8
      定位器是确定性的, 于是把源图重新跑一遍两个定位器, 拿框的尺寸和裁块实际尺寸对:
      对得上哪个, 当初用的就是哪个。

      源图是蓝图 + 当初用的是白图定位器 -> **切到红包促销卡了, 该剔**
      源图是蓝图 + 当初用的是蓝图定位器 -> 蓝图金额区裁块, 对
      源图是白图 + 用白图定位器          -> 白图金额区裁块, 对

    (最早写的是"看裁块底色是不是蓝"—— 那个不靠谱: 紧贴金额框的裁块几乎全是白色数字笔画,
     蓝底占比很小, 判不出来。尺寸对比是精确的。)
    """
    import numpy as np
    from PIL import Image

    from engine_b_tamper import locate_amount
    from locate_blue import is_blue_page, locate_amount_blue

    def load(p: Path):
        try:
            return np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            return None

    # 裁块 -> 生成图名 -> manifest 的 src
    src_of: dict[str, str] = {}
    for mp in sorted(manifest_root.glob("*/manifest.csv")):
        try:
            for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
                f, s = (r.get("file") or "").strip(), (r.get("src") or "").strip()
                if f and s:
                    src_of[f] = Path(s).name
        except Exception:
            continue
    index: dict[str, Path] = {}
    for root in src_roots:
        if root.is_dir():
            for cur, _d, files in __import__("os").walk(root):
                for f in files:
                    if Path(f).suffix.lower() in _EXTS:
                        index.setdefault(f, Path(cur) / f)
    print(f"manifest 记录 {len(src_of)} 条 | 源图池索引 {len(index)} 张")

    by_tag: dict[str, list[str]] = defaultdict(list)
    for nm in pairs:
        by_tag[tag_of(nm)].append(nm)

    def box_size(loc) -> tuple[int, int] | None:
        """定位框换算成裁块应有的尺寸(四周各让 4 像素, 见 gen_api_local_edit.py:241)。"""
        if not loc:
            return None
        return (loc[3] - loc[1] - 8, loc[2] - loc[0] - 8)

    print()
    print(f"  {'组':26s} {'对数':>7s} {'源图蓝图':>9s} {'用白定位':>9s} {'用蓝定位':>9s} {'判不了':>7s}   判定")
    print("  " + "-" * 104)
    bad: list[str] = []
    for tag in sorted(by_tag, key=lambda t: -len(by_tag[t])):
        names = by_tag[tag]
        pick = names if len(names) <= sample else random.Random(0).sample(names, sample)
        sb = sn = used_w = used_b = amb = 0
        for nm in pick:
            a = load(crops / "ai" / nm)
            gen = f"apilocal_{tag}_{Path(nm).stem.rpartition('_')[2]}.jpg"
            sp = index.get(src_of.get(gen, ""))
            if a is None or sp is None:
                amb += 1
                continue
            s = load(sp)
            if s is None:
                amb += 1
                continue
            sn += 1
            sb += int(is_blue_page(s))
            csz = a.shape[:2]
            wsz, bsz = box_size(locate_amount(s)), box_size(locate_amount_blue(s))
            if wsz == csz and bsz != csz:
                used_w += 1
            elif bsz == csz and wsz != csz:
                used_b += 1
            else:
                amb += 1
        sbp = sb * 100.0 / max(1, sn)
        if sn == 0:
            verdict = "源图查不到, 判不了 —— **不等于没问题**, 先把 --src-root 指对"
        elif used_w + used_b == 0:
            verdict = "尺寸都对不上, 多半不是 --crop-min 0 造的, 需人工看"
        elif sbp > 60 and used_w > used_b:
            verdict = "**蓝图却用了白图定位器 -> 切到红包卡, 该剔**"
            bad.append(tag)
        elif sbp > 60:
            verdict = "蓝图金额区裁块, 对"
        else:
            verdict = "白图金额区裁块, 对"
        print(f"  {tag:26s} {len(names):7d} {sbp:8.0f}% {used_w:9d} {used_b:9d} {amb:7d}   {verdict}")

    print()
    if bad:
        print("  按像素判定该剔掉的组:")
        print("    --exclude-tags " + " ".join(bad))
        print(f"  (共 {sum(len(by_tag[t]) for t in bad)} 对)")
    else:
        print("  没有判定为错区域的组。若源图大面积查不到, 说明源图池路径给得不对, 用 --src-root 指对再跑。")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="配对裁块 -> SSP 训练目录")
    ap.add_argument("--crops", type=Path, required=True, help="含 ai/ 与 nature/ 的目录")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出的数据集目录。--list-tags / --audit-tags 只看不建, 可以不给")
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
    ap.add_argument("--audit-tags", action="store_true",
                    help="**按像素**判每组裁块切得对不对(源图版式 vs 裁块底色), 并直接给出该剔哪几组。"
                         "别看 tag 名字 —— 实测 apiwantest 其实是蓝图, 名字里没有 blue")
    ap.add_argument("--manifest-root", type=Path, default=Path(r"D:\probe"))
    ap.add_argument("--src-root", type=Path, nargs="+",
                    default=[Path(r"D:\download\TempFakeImages"), Path(r"D:\download2\OtherImages")])
    ap.add_argument("--audit-sample", type=int, default=40, help="每组抽多少个裁块做判定")
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

    tags = Counter(tag_of(nm) for nm in pairs)
    if args.audit_tags:
        audit_tags(args.crops, args.manifest_root, list(args.src_root),
                   args.audit_sample, tag_of, pairs)
        return
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
    if args.out is None:
        raise SystemExit("要真的建数据集就得给 --out(只想看的话用 --list-tags 或 --audit-tags)")

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
