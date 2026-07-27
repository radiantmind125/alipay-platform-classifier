r"""为 (a)/(b) 评估造两个"无泄漏"测试集,避开 build_ssp_dataset 里两个坑。

坑1(真图泄漏):训练的 nature 是从两个图库按 seed 洗牌抽的 ~1 万张 —— 直接"取前 1000"
  会和训练集重叠,Pass 率虚高。这里用**磁盘上真实落地的 nature 文件名(stem)**当排除集,
  从没被训练用过的图里抽真图测试集 -> 干净的特异性。
坑2(真造假泄漏):13 张 gold 已进训练 ai 类,拿它们报召回是训练集成绩,乐观。
  这里从 recapture_eval_ext.txt(40 张 held-out 真翻拍)拷,并再挡一层:凡 stem 落在
  训练 nature 里的一律跳过。

用法(服务器):
  python training/sample_eval_sets.py ^
    --dataset-root D:\ssp_alipay\imagenet_ai_0419_sdv4 ^
    --genuine-roots D:\download2\TempFakeImages ^
    --n-genuine 1200 --genuine-out D:\ssp_test\gen_clean ^
    --recap-list training/data/recapture_eval_ext.txt ^
    --recap-src-root D:\download\TempFakeImages ^
    --realfake-out D:\ssp_test\realfake2
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)  # 光学拍摄参数;有=相机照片,不当真图


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


def _used_nature_stems(dataset_root: Path) -> set[str]:
    used: set[str] = set()
    for sp in ("train", "val"):
        d = dataset_root / sp / "nature"
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in _EXTS:
                    used.add(p.stem)
    return used


def _fresh(d: Path, force: bool) -> None:
    if len(d.parts) < 2:
        sys.exit(f"拒绝: 输出目录 {d} 像磁盘根/太浅, 不删。请指到具体子目录(如 D:\\ssp_test\\gen_clean)。")
    if d.exists():
        entries = list(d.iterdir())
        if entries and not force:
            sys.exit(f"输出目录已存在且非空: {d}(共 {len(entries)} 项)。为防误删, 需加 --force 才会清空重建。"
                     f" 请先确认这确实是测试输出目录、不是别的重要目录!")
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="造无泄漏的 (a) 真图 / (b) 真造假 测试集")
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="训练数据集根(含 train/val 的 nature),用来读已用真图名做排除")
    ap.add_argument("--genuine-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--n-genuine", type=int, default=1200)
    ap.add_argument("--genuine-out", type=Path, required=True)
    ap.add_argument("--recap-list", type=Path, required=True, help="recapture_eval_ext.txt")
    ap.add_argument("--recap-src-root", type=Path, required=True, help="翻拍原图所在目录")
    ap.add_argument("--realfake-out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="输出目录已存在且非空时也清空重建(默认拒绝, 防误删)")
    ap.add_argument("--allow-empty-exclusion", action="store_true",
                    help="允许排除集为空(默认拒绝, 防真图测试集混入训练图导致泄漏)")
    args = ap.parse_args()

    used = _used_nature_stems(args.dataset_root)
    print(f"训练 nature 已用真图(stem)数: {len(used)}  <- 这些从测试集里排除")
    if not used and not args.allow_empty_exclusion:
        sys.exit(f"排除集为空: --dataset-root({args.dataset_root}) 可能指错, 或其 train/val 的 nature 目录为空。"
                 f" 这会让训练用过的真图混进 (a) 真图测试集 -> 特异性泄漏虚高, 已中止。"
                 f" 确认路径无误(或就是想不排除)可加 --allow-empty-exclusion。")

    # ---- (b) 真造假(先做,顺便拿 40 张的 stem 也从真图里排除) ----
    recap_names = [ln.strip() for ln in args.recap_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    recap_stems = {Path(nm).stem for nm in recap_names}
    _fresh(args.realfake_out, args.force)
    b_copied = b_leak = b_missing = 0
    for nm in recap_names:
        stem = Path(nm).stem
        if stem in used:            # 泄漏挡板:这张翻拍其实进过训练 nature -> 跳过
            b_leak += 1
            continue
        src = args.recap_src_root / nm
        if not src.exists():
            b_missing += 1
            continue
        shutil.copy2(src, args.realfake_out / nm)
        b_copied += 1
    print(f"(b) 真造假测试集 -> {args.realfake_out}: 拷 {b_copied} 张 "
          f"(泄漏跳过 {b_leak}, 找不到 {b_missing})")

    # ---- (a) 无泄漏真图 ----
    files: list[Path] = []
    for r in args.genuine_roots:
        files += [p for p in r.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(args.seed).shuffle(files)
    _fresh(args.genuine_out, args.force)
    a_copied = a_skip_used = a_skip_recap = a_skip_notclean = 0
    for p in files:
        if a_copied >= args.n_genuine:
            break
        if p.stem in used:          # 训练用过 -> 排除
            a_skip_used += 1
            continue
        if p.stem in recap_stems:   # 已知翻拍(假图)-> 不放进真图集
            a_skip_recap += 1
            continue
        try:
            with Image.open(p) as im:
                if not _is_clean_screenshot(im):
                    a_skip_notclean += 1
                    continue
        except Exception:
            continue
        # 加序号前缀保证目标名唯一:两个图库都叫 TempFakeImages, 同名会互相覆盖 -> 计数虚高
        shutil.copy2(p, args.genuine_out / f"{a_copied:06d}_{p.name}")
        a_copied += 1
    print(f"(a) 无泄漏真图测试集 -> {args.genuine_out}: 拷 {a_copied} 张 "
          f"(训练已用跳过 {a_skip_used}, 翻拍跳过 {a_skip_recap}, 非干净截图跳过 {a_skip_notclean})")
    if a_copied < args.n_genuine:
        sys.exit(f"!! (a) 只凑到 {a_copied}/{args.n_genuine} 张 —— 多半是 --genuine-roots 路径不对, "
                 f"或库里可用干净截图不足。别拿不足量的集当'{args.n_genuine}张'报给经理, 已中止。"
                 f" 真图已拷在 {args.genuine_out}; 若确实想用少量, 自行调低 --n-genuine 重跑。"
                 f"((b) 真造假集已生成, 不受影响。)")
    print("提示:真图库以真为主但混极少数未标欺诈,特异性可能被极轻微低估,可接受。")


if __name__ == "__main__":
    main()
