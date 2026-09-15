"""看一批图**是不是原始截图** —— 拍屏的、转存过的、重压过的都挑出来。

为什么要这个
------------
拿去训练的图必须是原始截图。混进这几种就会教坏模型:
    拍屏     手机拍手机, 有传感器噪声、摩尔纹、光照不匀、透视变形
    转存     微信/QQ 传一道, 重新压过, 笔画糊边
    重压     多次 JPEG, 底色长出块状噪点

判据
----
★★★ **截图的纯色底是一个数值精确重复, 方差就是 0。**
   拍屏或转存过的, 底色一定有噪点, 方差不可能是 0。

量法: 把图切成 16x16 的小块, 每块算标准差, 取**最平的那 5%** 的中位数。

本机量过 301 张真实截图:

    299 张  平整度 = 0.0000      <- 精确的 0, 不是"接近 0"
      1 张  平整度 = 0.5970      <- 账单详情四个字糊边, 底色有噪点
      1 张  平整度 = 1.4882      <- 顶上一条深色照片带

★ 所以阈值不用调, **大于 0.01 就是可疑的**。这是个几乎二值的信号。

★★ 那 0.5970 的文件名是 `r0.397_f0.38_s3_voucher_...` —— 带前缀,
   本来就是处理流程的产物, 不是原图。这条旁证说明判据抓对了东西。

怎么用
------
    python check_screenshot.py --root 图目录 --workers 8
    python check_screenshot.py --root 图目录 --sheet 出图.png
    python check_screenshot.py --files a.jpg b.jpg c.jpg      # 只看指定几张
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ★ 截图是精确的 0, 所以这个阈值不是调出来的, 是"大于零"的容错余量
SUSPECT = 0.01

BLOCK = 16

# ★★★ 取**四分位**而不是"最平的那几块"。
#   一开始取的是最平的 5%, 写测试时发现能被骗: 图里只要有**任何一块**是
#   纯色的(比如一条实心色块), 最平的 5% 就全落在那上面, 报出来是 0,
#   哪怕别处噪点很重。
#   真实拍屏每个像素都有噪声, 所以实际上不会踩到; 但判据不该立在
#   "整张图没有任何一块是纯色的"这种运气上。
#   截图里底色本来就占一多半的块, 四分位照样是 0, 不损失灵敏度。
FLAT_PCT = 25


def flatness(path_str: str) -> dict:
    """小块标准差的下四分位。原始截图 = 0.0000。"""
    rec = {"path": path_str, "flat": None, "err": None}
    try:
        img = cv2.imdecode(np.fromfile(path_str, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rec["err"] = "decode"
            return rec
        h, w = img.shape
        hh, ww = h // BLOCK, w // BLOCK
        if hh < 4 or ww < 4:
            rec["err"] = "too_small"
            return rec
        blocks = (img[: hh * BLOCK, : ww * BLOCK]
                  .reshape(hh, BLOCK, ww, BLOCK).swapaxes(1, 2))
        sds = blocks.reshape(hh * ww, BLOCK * BLOCK).std(axis=1)
        rec["flat"] = float(np.percentile(sds, FLAT_PCT))
        rec["wh"] = [int(w), int(h)]
    except Exception as e:                      # noqa: BLE001
        rec["err"] = type(e).__name__
    return rec


def iter_images(root: Path):
    for dirpath, _, names in os.walk(root):
        for n in names:
            if Path(n).suffix.lower() in EXTS:
                yield str(Path(dirpath) / n)


def report(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r.get("flat") is not None]
    bad = [r for r in rows if r.get("flat") is None]
    if not ok:
        print("一张都没量到")
        return []

    a = np.array([r["flat"] for r in ok])
    exact = int((a == 0.0).sum())
    suspect = [r for r in ok if r["flat"] > SUSPECT]

    print("=" * 70)
    print(f"量到 {len(ok)} 张    读不了 {len(bad)}")
    print("=" * 70)
    print(f"  平整度精确等于 0 的:  {exact}  ({exact/len(ok):.2%})   <- 原始截图")
    print(f"  大于 {SUSPECT} 的:        {len(suspect)}  ({len(suspect)/len(ok):.2%})   "
          "<- ★ 可疑")
    print()
    for lo, hi, what in [(0.0, 0.0001, "= 0      原始截图"),
                         (0.0001, 0.01, "很小     大概率还是截图"),
                         (0.01, 0.1, "偏大     转存/重压过"),
                         (0.1, 1.0, "明显     糊边或有噪点"),
                         (1.0, 1e9, "很大     多半是拍屏")]:
        n = int(((a >= lo) & (a < hi)).sum())
        if n:
            print(f"    {lo:>7.4f} - {hi:<8.2f} {n:6d}   {what}")
    print()

    if suspect:
        suspect.sort(key=lambda r: -r["flat"])
        print("=" * 70)
        print(f"★★ 可疑的 {len(suspect)} 张(最不平的排前面)")
        print("=" * 70)
        for r in suspect[:30]:
            print(f"  {r['flat']:8.4f}   {Path(r['path']).name[:66]}")
        if len(suspect) > 30:
            print(f"  ...还有 {len(suspect)-30} 张")
        print()
        print("★ 拿去训练之前, 这些应当挑出来人工看一眼。")
    else:
        print("★★★ **一张可疑的都没有** —— 全是原始截图, 可以直接用。")
    print()
    if bad:
        print(f"读不了的 {len(bad)} 张:")
        for r in bad[:10]:
            print(f"  [{r.get('err')}]  {Path(r['path']).name[:66]}")
    return suspect


def make_sheet(rows: list[dict], sheet: Path, n: int = 6) -> None:
    """把最不平的几张的**底色区域放大**贴出来, 好看清有没有噪点。"""
    ok = [r for r in rows if r.get("flat") is not None]
    if not ok:
        return
    ok.sort(key=lambda r: -r["flat"])
    picked = ok[:n // 2] + ok[-(n - n // 2):]      # 最不平的 + 最平的对照
    cells = []
    for r in picked:
        p = Path(r["path"])
        if not p.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        crop = img[int(h * 0.06): int(h * 0.12), int(w * 0.1): int(w * 0.5)]
        if crop.size == 0:
            continue
        # 放大用最近邻, 保住原始像素, 不要插值把噪点抹平
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        crop = crop[:260, :560]
        crop = cv2.copyMakeBorder(crop, 34, 6, 6, 6,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(crop, f"flat={r['flat']:.4f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255) if r["flat"] > SUSPECT else (0, 140, 0), 2)
        cells.append(crop)
    if len(cells) < 2:
        return
    hh = max(c.shape[0] for c in cells)
    ww = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, ww - c.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in cells]
    per = 2
    usable = len(cells) - len(cells) % per
    grid = np.vstack([np.hstack(cells[i:i + per]) for i in range(0, usable, per)])
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", grid)[1].tofile(str(sheet))
    print(f"贴了 {usable} 张 -> {sheet}")
    print("★ 红的是可疑的, 绿的是正常截图。放大 4 倍, 看底色有没有噪点、笔画糊不糊。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None, help="要看的图目录(递归)")
    ap.add_argument("--files", nargs="*", default=None, help="或者直接给几个文件")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=Path, default=None, help="结果存成 jsonl")
    ap.add_argument("--sheet", type=Path, default=None, help="把最不平的贴成一张图")
    a = ap.parse_args()

    if a.files:
        todo = [str(Path(f)) for f in a.files]
    elif a.root:
        if not a.root.exists():
            print(f"目录不在: {a.root}")
            return
        todo = list(iter_images(a.root))
    else:
        print("要给 --root 或者 --files")
        return
    if not todo:
        print("没找到图")
        return

    print(f"要量 {len(todo):,} 张,  {a.workers} 个进程")
    if len(todo) > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(flatness, todo, chunksize=100))
    else:
        rows = [flatness(todo[0])]
    print()

    report(rows)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"结果存了 -> {a.out}")
    if a.sheet:
        print()
        make_sheet(rows, a.sheet)


if __name__ == "__main__":
    main()
