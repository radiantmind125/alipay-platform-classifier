r"""从已经花过钱的**整图重绘结果**里, 用**修好的定位器**重新做局部改金额样本(不再调接口)。

为什么这条路走得通
------------------
`gen_api_local_edit.py --send full --save-full <dir>` 会把**接口返回的整张重绘图**
原样存下来(:216-217, 文件名 `apifull_<tag>_<NNNNN>.jpg`)。

**这一步在定位之前, 所以它没被那个 bug 污染。**
当时那个 bug 只影响三件事: 哪些源图被跳过、往哪儿贴、裁块从哪儿切 ——
**整张重绘图本身是干净的**, 它就是"这家真实模型对这张真收据的重绘结果"。

于是: 源图还在 + 整图重绘还在 + 定位器已修好 = **可以把局部改金额样本离线重做一遍,
一次接口都不用调。** 已经花掉的钱能捞回来一大部分。

join 链(和裁块那条一样)
    api_full_xxx\apifull_<tag>_<NNNNN>.jpg
      -> probe\<批次>\manifest.csv 里的 apilocal_<tag>_<NNNNN>.jpg
      -> 该行 src = 源图文件名

重做的流程(和 `gen_api_local_edit.py:219-251` 的 full 分支逐行对齐, 只把定位器换成 auto)
    1. 源图定位金额     loc_s, 版式 = locate_amount_auto(src)
    2. 重绘图缩回原尺寸, 再定位金额  loc_g = locate_amount_auto(gen)
    3. 取重绘图的金额块, 缩到 loc_s 的尺寸, 羽化贴回源图
    4. 存合成假图 + 配对裁块(ai 取自合成图, nature 取自源图)

用法
----
  先看有什么(只读, 不写任何东西):
    python training/recover_from_full.py --list --full-root D:\

  再重做某一批:
    python training/recover_from_full.py --full D:\api_full_wan_blue_train ^
        --out D:\probe\bluetrain_wan_recov --save-crops D:\localcrops --tag bluetrainwan ^
        --src-root D:\download\TempFakeImages D:\download2\OtherImages

**注意一个已知偏置**: 当时能进 `api_full_*` 的源图, 是白图定位器**返回了某个框**的那些。
蓝底页上白图定位器经常失败, 所以这批蓝图源是**有选择偏置的子集**, 不是随机样本。
图本身可用, 但覆盖面窄于随机抽样 —— 报数时要说明。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_api_local_edit import _feather_paste        # noqa: E402  同一套羽化贴回, 别另写一份
from locate_blue import is_blue_page, locate_amount_auto  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_full_name(stem: str) -> tuple[str, str] | None:
    """`apifull_<tag>_<NNNNN>` -> (tag, NNNNN)。从右边切, tag 带下划线也不会切错。"""
    if not stem.startswith("apifull_"):
        return None
    body = stem[len("apifull_"):]
    tag, sep, idx = body.rpartition("_")
    if not sep or not idx.isdigit():
        return None
    return tag, idx


def load_src_map(manifest_root: Path) -> dict[str, str]:
    """apilocal_<tag>_<NNNNN>.jpg -> 源图 basename。扫 manifest_root 下所有 manifest。"""
    out: dict[str, str] = {}
    for mp in sorted(manifest_root.glob("*/manifest.csv")):
        try:
            for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
                f, s = (r.get("file") or "").strip(), (r.get("src") or "").strip()
                if f and s:
                    out[f] = Path(s).name
        except Exception:
            continue
    return out


def build_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for cur, _d, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in _EXTS:
                    index.setdefault(f, Path(cur) / f)
    return index


def load_rgb(p: Path):
    try:
        return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
    except Exception:
        return None


def do_list(full_root: Path, manifest_root: Path, src_roots: list[Path], sample: int) -> None:
    src_map = load_src_map(manifest_root)
    index = build_index(src_roots)
    print("=" * 104)
    print(f"整图重绘目录盘点   manifest 记录 {len(src_map)} 条 | 源图池索引 {len(index)} 张")
    print("=" * 104)
    dirs = sorted(p for p in full_root.iterdir() if p.is_dir() and p.name.startswith("api_full"))
    if not dirs:
        print(f"  {full_root} 下没有 api_full* 目录")
        return
    print(f"  {'目录':34s} {'图片':>7s} {'组(tag)':22s} {'可溯源':>7s} {'源图蓝图':>9s}")
    print("  " + "-" * 100)
    for d in dirs:
        names = [e.name for e in os.scandir(d)
                 if e.is_file() and os.path.splitext(e.name)[1].lower() in _EXTS]
        tags = Counter()
        traced = 0
        blue = seen = 0
        for i, nm in enumerate(names):
            parsed = split_full_name(Path(nm).stem)
            if not parsed:
                continue
            tag, idx = parsed
            tags[tag] += 1
            sn = src_map.get(f"apilocal_{tag}_{idx}.jpg")
            if sn and sn in index:
                traced += 1
                if seen < sample:
                    a = load_rgb(index[sn])
                    if a is not None:
                        seen += 1
                        blue += int(is_blue_page(a))
        tg = ", ".join(f"{t}({c})" for t, c in tags.most_common(3)) or "(命名不认识)"
        bp = f"{blue*100.0/seen:7.0f}%" if seen else "     --"
        print(f"  {d.name:34s} {len(names):7d} {tg:22s} {traced:7d} {bp:>9s}")
    print()
    print("怎么读:")
    print("  可溯源 = 能从 manifest 找回源图、且源图还在池子里的张数。**只有这些能重做。**")
    print("  源图蓝图 = 抽样判的版式。**蓝底那几批就是能白捡的**: 整图重绘没被定位器 bug 影响,")
    print("             源图还在, 定位器已修好 -> 局部改金额样本可以离线重做, 不用再调接口。")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="用修好的定位器, 从已付费的整图重绘里离线重做局部改金额样本")
    ap.add_argument("--list", action="store_true", help="只盘点有什么, 不写任何文件")
    ap.add_argument("--full-root", type=Path, default=Path(r"D:\\"), help="--list 时扫哪个盘")
    ap.add_argument("--full", type=Path, default=None, help="要重做的那个整图重绘目录")
    ap.add_argument("--out", type=Path, default=None, help="合成假图输出目录")
    ap.add_argument("--save-crops", type=Path, default=None, help="配对裁块输出目录(含 ai/ 和 nature/)")
    ap.add_argument("--tag", default="", help="新 tag(默认用原 tag 加 _rec)")
    ap.add_argument("--manifest-root", type=Path, default=Path(r"D:\probe"))
    ap.add_argument("--src-root", type=Path, nargs="+",
                    default=[Path(r"D:\download\TempFakeImages"), Path(r"D:\download2\OtherImages")])
    ap.add_argument("--sample", type=int, default=40, help="--list 时每个目录抽几张判版式")
    ap.add_argument("--limit", type=int, default=0, help="最多重做几张(0=全部)")
    args = ap.parse_args()

    if args.list:
        do_list(args.full_root, args.manifest_root, list(args.src_root), args.sample)
        return
    if not args.full or not args.out:
        raise SystemExit("要重做就得给 --full 和 --out(只想看用 --list)")

    src_map = load_src_map(args.manifest_root)
    index = build_index(list(args.src_root))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        (args.save_crops / "ai").mkdir(parents=True, exist_ok=True)
        (args.save_crops / "nature").mkdir(parents=True, exist_ok=True)

    mp = args.out / "manifest.csv"
    new = not mp.exists()
    mf = open(mp, "a", newline="", encoding="utf-8-sig")
    mw = csv.writer(mf)
    if new:
        mw.writerow(["file", "model", "src"])

    made = 0
    page_n: Counter[str] = Counter()
    why: Counter[str] = Counter()
    files = sorted(e.name for e in os.scandir(args.full)
                   if e.is_file() and os.path.splitext(e.name)[1].lower() in _EXTS)
    print(f"整图重绘 {len(files)} 张 -> 开始离线重做(不调接口)", flush=True)

    for nm in files:
        if args.limit and made >= args.limit:
            break
        parsed = split_full_name(Path(nm).stem)
        if not parsed:
            why["文件名不认识"] += 1
            continue
        tag, idx = parsed
        sn = src_map.get(f"apilocal_{tag}_{idx}.jpg")
        sp = index.get(sn) if sn else None
        if sp is None:
            why["源图查不到"] += 1
            continue
        src, gen_im = load_rgb(sp), None
        try:
            gen_im = ImageOps.exif_transpose(Image.open(args.full / nm)).convert("RGB")
        except Exception:
            pass
        if src is None or gen_im is None:
            why["图读不了"] += 1
            continue

        loc_s, page = locate_amount_auto(src)          # ★ 修好的自动分派定位器
        if not loc_s:
            why["源图定位不到金额"] += 1
            continue
        sx0, sy0, sx1, sy1 = loc_s[0], loc_s[1], loc_s[2], loc_s[3]

        gen = np.asarray(gen_im.resize((src.shape[1], src.shape[0]), Image.LANCZOS))
        loc_g, _ = locate_amount_auto(gen)
        if not loc_g:
            why["重绘图定位不到金额(版面漂了)"] += 1
            continue
        gx0, gy0, gx1, gy1 = loc_g[0], loc_g[1], loc_g[2], loc_g[3]
        if gx1 - gx0 < 4 or gy1 - gy0 < 4:
            why["重绘图金额框太小"] += 1
            continue

        piece = cv2.resize(gen[gy0:gy1, gx0:gx1], (sx1 - sx0, sy1 - sy0),
                           interpolation=cv2.INTER_LANCZOS4)
        out = _feather_paste(src, piece, (sx0, sy0, sx1, sy1))

        newtag = args.tag or (tag + "_rec")
        fn = f"apilocal_{newtag}_{idx}.jpg"
        Image.fromarray(out).save(args.out / fn, quality=95)
        mw.writerow([fn, f"recovered_from:{tag}", sn])
        mf.flush()

        if args.save_crops:
            a0, b0, a1, b1 = sy0 + 4, sx0 + 4, sy1 - 4, sx1 - 4
            if a1 - a0 >= 32 and b1 - b0 >= 32:
                Image.fromarray(out[a0:a1, b0:b1]).save(
                    args.save_crops / "ai" / f"crop_{newtag}_{idx}.jpg", quality=95)
                Image.fromarray(src[a0:a1, b0:b1]).save(
                    args.save_crops / "nature" / f"crop_{newtag}_{idx}.jpg", quality=95)
            else:
                why["金额框太小,裁块没存"] += 1

        made += 1
        page_n[page] += 1
        if made % 50 == 0:
            print(f"  已重做 {made} 张...", flush=True)

    mf.close()
    print(f"完成: 重做 {made} 张 -> {args.out}")
    print(f"版式: 蓝图 {page_n.get('blue', 0)} 张, 白图 {page_n.get('white', 0)} 张(按像素判)")
    if why:
        print("没做成的原因:")
        for k, v in why.most_common():
            print(f"  {k:26s} {v}")
    print()
    print("下一步: 用 edit_region_audit 验一下改动确实落在金额区, 再决定要不要进训练集")
    print(f"  python training/edit_region_audit.py --fake {args.out} "
          f"--src-root {' '.join(str(r) for r in args.src_root)} --limit 40")


if __name__ == "__main__":
    main()
