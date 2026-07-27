r"""为官方 SSP(train_val.py)重训在服务器 GPU 上组装数据集,按审计配方,避开所有坑。

产出布局(官方 loader 硬编码这些名字,一个字都不能改):
  <out>/imagenet_ai_0419_sdv4/
    train/{nature, ai}
    val/{nature, ai}
- nature(标签1)= 真图支付宝截图(排除相机/翻拍子集)。
- ai(标签0)= 假图:真翻拍(gold 里核过的)+ 合成翻拍(recapture_sim 在"另一批"真图上造,不与 nature 重叠)
  + redteam 攻击图 + 任何 AI 生成假图(--extra-ai)。
坑规避:val/ai 必须非空(否则 train_val.py 的 val 除零/NaN);按源图切分不泄漏;文件夹严格叫 nature/ai。
诚实:正样本几乎全合成,周一分数偏乐观,必须单独在真翻拍上报召回。

服务器用法(GPU 机,真图在本地):
  python training/build_ssp_dataset.py --genuine-roots D:\download\TempFakeImages D:\download2\TempFakeImages \
    --gold platform-classifier/gold/recapture_gold.jsonl --gold-img-root D:\download\TempFakeImages \
    --redteam-dir redteam_prod/attacked --out D:\ssp_alipay --n-nature 8000 --n-recap-synth 4000 --val-frac 0.15
之后编辑 SSP options.py: choices 默认改 [2,2,2,2,1,2,2,2];pip install scipy==1.10.1;
  python train_val.py --image_root D:\ssp_alipay --gpu_id 0 --save_path ./snapshot/alipay/
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.recapture_sim import simulate_recapture  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)  # 光学拍摄参数;有 = 相机照片,不当 nature
_DIR = "imagenet_ai_0419_sdv4"       # 复用 sdv4 槽(配 options.choices=[2,2,2,2,1,2,2,2])


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


def _collect_genuine(roots: list[Path], need: int, seed: int) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        files += [p for p in r.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(seed).shuffle(files)
    out: list[Path] = []
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


def _save_recapture(src: Path, dst: Path, seed: int, cap: int = 1400) -> bool:
    try:
        with Image.open(src) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = rgb.shape[:2]
        s = cap / max(h, w)
        if s < 1.0:
            import cv2
            rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        Image.fromarray(simulate_recapture(rgb, seed=seed)).save(dst, quality=92)
        return True
    except Exception:
        return False


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="组装官方 SSP nature/ai 数据集")
    ap.add_argument("--genuine-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--gold", type=Path, help="recapture_gold.jsonl(真翻拍清单)")
    ap.add_argument("--gold-img-root", type=Path, help="gold 里 file 名所在目录")
    ap.add_argument("--redteam-dir", type=Path, help="redteam 攻击图目录")
    ap.add_argument("--extra-ai", type=Path, help="额外 AI 生成假图目录(GPU 上 diffusers 生成的)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-nature", type=int, default=8000)
    ap.add_argument("--n-recap-synth", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = args.out / _DIR
    dirs = {(sp, cl): base / sp / cl for sp in ("train", "val") for cl in ("nature", "ai")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    genuine = _collect_genuine(args.genuine_roots, args.n_nature + args.n_recap_synth, args.seed)
    if len(genuine) < 10:
        print(f"真图太少:{len(genuine)}"); return
    nat = genuine[:args.n_nature]
    recap_src = genuine[args.n_nature:args.n_nature + args.n_recap_synth]  # 与 nat 无交集 -> 不泄漏
    rng = random.Random(args.seed)

    def split(items: list) -> tuple[list, list]:
        it = list(items); rng.shuffle(it)
        k = max(1, int(len(it) * args.val_frac)) if it else 0
        return it[k:], it[:k]  # train, val

    n_tr, n_va = split(nat)
    for f in n_tr:
        shutil.copy2(f, dirs[("train", "nature")] / f.name)
    for f in n_va:
        shutil.copy2(f, dirs[("val", "nature")] / f.name)

    # --- ai 类 ---
    cnt = {"real_recap": 0, "synth_recap": 0, "redteam": 0, "extra_ai": 0}
    # 真翻拍(gold),约一半进 val/ai 保证非空
    reals: list[Path] = []
    if args.gold and args.gold.exists() and args.gold_img_root:
        for line in args.gold.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("label") == "recapture":
                p = args.gold_img_root / rec["file"]
                if p.exists():
                    reals.append(p)
    rng.shuffle(reals)
    half = len(reals) // 2
    for i, p in enumerate(reals):
        sp = "val" if i < max(1, half) else "train"
        shutil.copy2(p, dirs[(sp, "ai")] / p.name); cnt["real_recap"] += 1
    # 合成翻拍
    r_tr, r_va = split(recap_src)
    for i, f in enumerate(r_tr):
        if _save_recapture(f, dirs[("train", "ai")] / f"synthrecap_{i:06d}.jpg", seed=i):
            cnt["synth_recap"] += 1
    for i, f in enumerate(r_va):
        if _save_recapture(f, dirs[("val", "ai")] / f"synthrecap_v{i:06d}.jpg", seed=10_000 + i):
            cnt["synth_recap"] += 1
    # redteam + 额外 AI 假图 -> 全进 train/ai
    for tag, src in (("redteam", args.redteam_dir), ("extra_ai", args.extra_ai)):
        if src and src.exists():
            for p in src.rglob("*"):
                if p.suffix.lower() in _EXTS:
                    shutil.copy2(p, dirs[("train", "ai")] / f"{tag}_{p.name}"); cnt[tag] += 1

    def n(d):
        return sum(1 for _ in d.iterdir())
    print(f"数据集 -> {base}")
    print(f"  train/nature={n(dirs[('train','nature')])}  train/ai={n(dirs[('train','ai')])}")
    print(f"  val/nature  ={n(dirs[('val','nature')])}  val/ai  ={n(dirs[('val','ai')])}")
    print(f"  ai 组成: {cnt}")
    if n(dirs[("val", "ai")]) == 0:
        print("  !! val/ai 为空 —— train_val.py 会 NaN/崩;放几张真翻拍进 val/ai")
    print("诚实:ai 几乎全合成,周一分数偏乐观,务必单独在真翻拍上报召回。")


if __name__ == "__main__":
    main()
