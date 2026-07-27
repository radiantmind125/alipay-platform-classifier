r"""从图库里挖"造假模板家族" —— 不用问经理要假图,也用来判断真实欺诈是"模板改字"还是"AI 生成"。

原理:造假截图 App 复用同一套模板,只改金额/姓名/时间 → 产出一堆版面近乎一模一样的图。
真实用户的截图各不相同(状态栏/头像/优惠位/机型分辨率都在变)。所以**又大又紧的近重复簇**
= 疑似造假家族。挖出来人工抽看,既能得到真假图评估集,又能看清欺诈到底长什么样。

方法:每图算 dHash(g×g 网格,g*g 位)→ 分带 LSH 找候选 → leader 聚类(Hamming ≤ 半径)。
只读原图。输出簇大小分布 + 大簇的样例路径供人工核对。

用法:
  python scripts\mine_fake_families.py --roots C:\...\TempFakeImages C:\...\white\TempFakeImages \
    --limit 40000 --grid 12 --radius 8 --min-cluster 8 --out runs\fake_families.jsonl --save-dir runs\fam_samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash_bits(im: Image.Image, g: int) -> np.ndarray:
    a = np.asarray(im.convert("L").resize((g + 1, g), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()          # g*g 位布尔


def _bands(bits: np.ndarray, nb: int) -> list[int]:
    """把位向量切成 nb 段,每段打包成一个 int(LSH 键)。"""
    out = []
    for chunk in np.array_split(bits, nb):
        v = 0
        for b in chunk:
            v = (v << 1) | int(b)
        out.append(v)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="挖造假模板家族(近重复聚类)")
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=40000, help="每池抽样上限")
    ap.add_argument("--grid", type=int, default=12, help="dHash 网格 g(位数=g*g)")
    ap.add_argument("--radius", type=int, default=8, help="同簇的最大 Hamming(越小越紧越像造假)")
    ap.add_argument("--nbands", type=int, default=12)
    ap.add_argument("--min-cluster", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("runs/fake_families.jsonl"))
    ap.add_argument("--save-dir", type=Path, default=None, help="给每个大簇存几张样例(拷贝)")
    args = ap.parse_args()

    paths: list[Path] = []
    for r in args.roots:
        fs = sorted(p for p in r.rglob("*") if p.suffix.lower() in _EXTS)
        step = max(1, len(fs) // args.limit)
        paths += fs[::step][:args.limit]
    print(f"抽样 {len(paths)} 张,算 dHash(g={args.grid},{args.grid**2}位)…")

    bits_list: list[np.ndarray] = []
    kept: list[Path] = []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                bits_list.append(dhash_bits(im, args.grid))
            kept.append(p)
        except Exception:
            continue
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(paths)}")
    if not bits_list:
        print("没算出 hash"); return
    H = np.stack(bits_list)                              # (N, g*g) bool
    n = len(kept)

    # leader 聚类 + 分带候选
    rep_bits: list[np.ndarray] = []
    rep_members: list[list[int]] = []
    band_index: dict[tuple[int, int], list[int]] = {}    # (band_no, val) -> rep ids
    for idx in range(n):
        b = H[idx]
        cand: set[int] = set()
        keys = _bands(b, args.nbands)
        for bn, v in enumerate(keys):
            cand.update(band_index.get((bn, v), ()))
        found = -1
        for rid in cand:
            if int(np.count_nonzero(b != rep_bits[rid])) <= args.radius:
                found = rid
                break
        if found >= 0:
            rep_members[found].append(idx)
        else:
            rid = len(rep_bits)
            rep_bits.append(b)
            rep_members.append([idx])
            for bn, v in enumerate(keys):
                band_index.setdefault((bn, v), []).append(rid)

    clusters = sorted(rep_members, key=len, reverse=True)
    big = [c for c in clusters if len(c) >= args.min_cluster]
    in_big = sum(len(c) for c in big)
    print(f"\n簇总数 {len(clusters)};大簇(≥{args.min_cluster})共 {len(big)} 个,"
          f"覆盖 {in_big}/{n} = {in_big/n:.1%} 的图(近重复占比高 = 造假模板家族多)")
    print("最大的若干簇(大小 + 样例):")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for ci, c in enumerate(big):
            ex = [str(kept[m]) for m in c[:6]]
            fh.write(json.dumps({"cluster": ci, "size": len(c), "examples": ex}, ensure_ascii=False) + "\n")
            if ci < 20:
                print(f"  #{ci:2d} size={len(c):5d}  e.g. {Path(ex[0]).name}")
            if args.save_dir and ci < 15:
                import shutil
                d = args.save_dir / f"cluster_{ci:02d}_n{len(c)}"
                d.mkdir(parents=True, exist_ok=True)
                for m in c[:4]:
                    try:
                        shutil.copy2(kept[m], d / kept[m].name)
                    except Exception:
                        pass
    print(f"\n簇清单 -> {args.out}" + (f";样例 -> {args.save_dir}" if args.save_dir else ""))
    print("怎么读:又大又紧的簇=同一造假模板批量产出(疑似假图家族);人工看几张确认。"
          "若大簇多=欺诈以'模板改字'为主(该重投字形/模板一致性);若几乎没有大簇=更可能真实分散/AI 生成。")


if __name__ == "__main__":
    main()
