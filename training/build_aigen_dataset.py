r"""组装"AI 生成检测"数据集(修掉第一版让 AI 召回=0 的三个坑)。

第一版坑 -> 本脚本的修法:
1. 过度压缩把 AI 指纹洗掉 -> 这里**不缩尺寸**, 之后 reencode 只做轻度质量对齐(--max-side 0 --q 95)。
2. val 里零 AI(全翻拍)-> 这里 ai 主体就是 AI 假图, 且**留一个生成器只进 val**, 专测"没见过的生成器"泛化。
3. (选块盲区归 train_val/patch, 不在本脚本)。

布局(官方 loader 硬编码):<out>/imagenet_ai_0419_sdv4/{train,val}/{nature, ai}
- nature(标签1)= 真截图(排除相机/翻拍)。
- ai(标签0)= AI 生成假图(gen_ai_fakes 多生成器造, 原生分辨率)。
  * held-out 生成器(--holdout-tag)的假图 **全部进 val/ai** -> 测泛化到没见过的生成器。
  * 其余生成器 train/val 切分。文件名保留(含生成器 tag), 便于 val 里分"见过/没见过"分别报召回。

用法(服务器):
  python training/build_aigen_dataset.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages ^
    --aigen-root D:\ai_fakes --holdout-tag sdxl-vae-fp16-fix --out D:\ssp_aigen --n-nature 9000 --val-frac 0.15
  之后: python training/reencode_uniform.py --root D:\ssp_aigen\imagenet_ai_0419_sdv4 --max-side 0 --q 95   (轻度对齐,保指纹)
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)   # 光学拍摄参数;有=相机照片,不当 nature
_DIR = "imagenet_ai_0419_sdv4"


def _is_clean_screenshot(im: Image.Image) -> bool:
    w, h = im.size
    short = min(w, h)
    if not short:
        return False
    aspect = max(w, h) / short
    try:
        ifd = im.getexif().get_ifd(0x8769)
    except Exception:
        ifd = {}
    return (not any(t in ifd for t in _OPT)) and short <= 1500 and 1.6 <= aspect <= 2.6


def _collect_genuine(roots, need, seed):
    files = []
    for r in roots:
        files += [p for p in r.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(seed).shuffle(files)
    out = []
    for p in files:
        if len(out) >= need:
            break
        try:
            with Image.open(p) as im:
                if _is_clean_screenshot(im):
                    out.append(p)
        except Exception:
            continue
    return out


def _tag(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def _split(items, val_frac, rng):
    it = list(items); rng.shuffle(it)
    k = max(1, int(len(it) * val_frac)) if it else 0
    return it[k:], it[:k]   # train, val


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="组装 AI 生成检测数据集(held-out 生成器进 val)")
    ap.add_argument("--genuine-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--aigen-root", type=Path, required=True, help="gen_ai_fakes 产出目录(含 manifest.csv)")
    ap.add_argument("--holdout-tag", required=True, help="留作 val 的生成器 tag(如 sdxl-vae-fp16-fix)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-nature", type=int, default=9000)
    ap.add_argument("--n-ai", type=int, default=0, help="ai 总量上限(0=全用)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = args.out / _DIR
    dirs = {(sp, cl): base / sp / cl for sp in ("train", "val") for cl in ("nature", "ai")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ---- nature ----
    nat = _collect_genuine(args.genuine_roots, args.n_nature, args.seed)
    if len(nat) < 10:
        print(f"真图太少:{len(nat)}"); return
    rng = random.Random(args.seed)
    n_tr, n_va = _split(nat, args.val_frac, rng)
    for f in n_tr:
        shutil.copy2(f, dirs[("train", "nature")] / f.name)
    for f in n_va:
        shutil.copy2(f, dirs[("val", "nature")] / f.name)

    # ---- ai(读 manifest 拿生成器身份)----
    mpath = args.aigen_root / "manifest.csv"
    rows = []
    if mpath.exists():
        for r in csv.DictReader(open(mpath, encoding="utf-8-sig")):
            rows.append((r["file"], _tag(r.get("model", ""))))
    else:   # 没 manifest 就从文件名兜底(aivae_<tag>_<i>.jpg)
        for p in args.aigen_root.iterdir():
            if p.suffix.lower() in _EXTS:
                parts = p.stem.split("_")
                rows.append((p.name, "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"))
    rows = [(fn, tg) for fn, tg in rows if (args.aigen_root / fn).exists()]
    if args.n_ai > 0:
        random.Random(args.seed + 1).shuffle(rows); rows = rows[:args.n_ai]

    holdout = [r for r in rows if r[1] == args.holdout_tag or args.holdout_tag in r[1]]
    seen = [r for r in rows if r not in holdout]
    if not holdout:
        print(f"!! 没匹配到 held-out 生成器 '{args.holdout_tag}';现有 tag: {sorted(set(t for _, t in rows))}")
    s_tr, s_va = _split(seen, args.val_frac, rng)
    for fn, _ in s_tr:
        shutil.copy2(args.aigen_root / fn, dirs[("train", "ai")] / fn)
    for fn, _ in s_va:
        shutil.copy2(args.aigen_root / fn, dirs[("val", "ai")] / fn)
    for fn, _ in holdout:                 # held-out 生成器全进 val/ai
        shutil.copy2(args.aigen_root / fn, dirs[("val", "ai")] / fn)

    def n(d):
        return sum(1 for _ in d.iterdir())
    from collections import Counter
    val_ai_tags = Counter(t for _, t in s_va) + Counter({args.holdout_tag: len(holdout)})
    print(f"数据集 -> {base}")
    print(f"  train/nature={n(dirs[('train','nature')])}  train/ai={n(dirs[('train','ai')])}")
    print(f"  val/nature  ={n(dirs[('val','nature')])}  val/ai  ={n(dirs[('val','ai')])}")
    print(f"  ai 生成器: 训练用 {sorted(set(t for _, t in seen))}  |  held-out(仅val) '{args.holdout_tag}' x{len(holdout)}")
    print(f"  val/ai 各生成器: {dict(val_ai_tags)}")
    if n(dirs[("val", "ai")]) == 0 or len(holdout) == 0:
        print("  !! val/ai 空 或 没 held-out -> 无法测泛化, 检查 --aigen-root/--holdout-tag")
    print("下一步(保指纹): python training/reencode_uniform.py --root %s --max-side 0 --q 95" % base)
    print("诚实: 召回按 val/ai 里**held-out 生成器**单独报 = 对'没见过的 AI'的真实识别率。")


if __name__ == "__main__":
    main()
