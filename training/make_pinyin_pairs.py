"""给拼音 OCR 出训练配对: 一行出**两张**裁图。

    label/  不含拼音 —— 拿去给现成 OCR 读, 读出来的字当**标签**
    input/  含拼音   —— 学生看到的图, 当**输入**

★★★★★ 两套不能反。学生要学的是"**看见拼音也能读对**",
   所以输入必须**带拼音**; 只有标签才来自把拼音排除掉之后读出来的文字。
   输入要是也不含拼音, 就训成了一个普通 OCR, 白训。

为什么标签不能直接 OCR 原图
---------------------------
实测(rapidocr, 每组 25 张白图), 数固定字段标签能读出几个:

    带拼音    中位 2.0 个
    不带拼音  中位 9.0 个        <- 比值 0.22

而且不是没检出来, 是**认成了别的字**: `账户余额` 读成 `芦荼额`。
拼音笔画混在识别的那条线里, 字就认错。

把拼音**裁在框外**之后:

    按行裁    中位 7.0 个        <- 追回不带拼音那档的 78%

★ 也试过"把拼音擦掉再读", 不行: 擦得只剩骨架 OCR 照样读得出来,
  8 张图整页送 OCR, 原图 129 行纯拉丁, 擦完还有 72 行。**像素占比低 ≠ OCR 看不见。**

行怎么切 —— 两条关键, 缺一不可
------------------------------
1. **按纵向重叠聚行**, 不能按"离得近就合并"。
   回单是左标签右取值两栏, 两栏基线不在同一高度;
   按距离链式合并会把整页连成一条(实测 8 张裁出来 OCR 读出 608 行, 纯拉丁 259 行)。

2. **边界取成员的中位**, 不能取并集。
   `〈` 返回箭头这类**很高的元件**跨越整行, 取并集会从它的顶开始,
   一路顶进上面的拼音带(实测裁出来每一条都还带着拼音)。

★ 也别用水平投影切行: 拼音有 g q y j 这些下伸笔画,
  一行拼音横跨一千多像素, 只要一个笔画探进汉字行, 投影就没有空行。
  实测空隙中位只有 3 像素, 25% 的地方不到 2 像素。

用法
----
    python make_pinyin_pairs.py --src 图目录 --out 输出目录 [--limit N] [--workers N]

出完之后再跑 OCR 填标签(另一个脚本), 这样重跑 OCR 不用重裁。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import EXTS, annotation_labels, local_background, text_mask  # noqa: E402

# 纵向重叠超过这个比例算同一行。两栏基线错开一点仍然大幅重叠, 上下两行几乎不重叠。
OVERLAP = 0.5

# 一行至少这么高才要 —— 更矮的多半是分隔线或噪声
MIN_ROW_H = 10

# 标签裁图上下留的余量。★ 不能留大, 留多了会把上面那行拼音带进来。
LABEL_PAD = 1

# 输入裁图: 上面没找到拼音块时, 按这个倍数的字高往上扩。
# 0.46 是量出来的(800 张白图: 框顶被拼音顶高中位 12 像素, 正文字高中位 26)。
FALLBACK_SHIFT_RATIO = 0.46


def _pinyin_bands(kept, st, big_boxes):
    """把判成注音的那些块按高度聚成**带**, 返回 [(y0, y1)]。

    ★★★★★ 为什么要有这个: `annotation_labels` 把连通块切成两档 ——
       small(<=0.55*big_h) 和 big(>=0.8*big_h), 两档**互不相交**。
       判注音只看 small 那一档。

       但拼音块不是都那么矮。`zhàng dān xiáng qíng` 这种,
       几乎每个音节**上面有声调符号、下面有下伸笔画**, 整个块顶到底
       和汉字一样高, 于是落进 big 那一档 —— 判注音时看不见它,
       聚行时它却**自己凑成一行**, 行顶就落在拼音带里。

       实测后果: 标签裁图裁出来 OCR 读成
           Sueuz uanxiangqing quan buzhangdan 立 ?
       整条都是拼音, 汉字反而被切掉了。

    ★ 所以按**矮拼音块的高度**先把带划出来, 聚行时凡是中心落在带里的
      一律不要 —— 高拼音和矮拼音在同一个高度上, 用矮的就能把高的一起圈掉。

    ★★ 聚带也得**按纵向重叠**, 不能按"离得近就并"。按距离并是链式的 ——
       整页拼音一条条往下挨着, 并到最后成了一条覆盖半页的带, 行全被吃掉。
       (这个坑聚行时已经踩过一次, 见模块开头。实测: 按距离并之后
        判出有拼音的行从 64% 掉到 45%, 漏进标签的反而从 8.6% 涨到 13.1%。)
    """
    if not kept:
        return []
    ys = []
    for idx in kept:
        if idx < len(st):
            ys.append((int(st[idx][1]), int(st[idx][1]) + int(st[idx][3])))
    if not ys:
        return []

    bands: list[list] = []
    for a, b in sorted(ys):
        for g in bands:
            inter = min(g[1], b) - max(g[0], a)
            shorter = min(g[1] - g[0], b - a)
            if shorter > 0 and inter >= OVERLAP * shorter:
                g[0] = min(g[0], a)
                g[1] = max(g[1], b)
                break
        else:
            bands.append([a, b])

    # ★★★ 试过再加一步"把大部分身子在带里的高块吸进带", 想把带撑到实际范围。
    #   **不能加** —— 它会把汉字也吸进去。汉字一旦进了带就不参与聚行,
    #   行里只剩 〈 这类更高的元件, 中位反而**往上**跑, 比不排除还糟:
    #       订单号 ...           -> Sln 单号 uan nlao      顶 1399 -> 1389
    #       芭芭农场 限时翻倍     -> ... baba nongchang     顶 1371 -> 1340
    #   实测 23 行变坏, 只有 9 行变好。
    #
    #   本来也不需要: 高拼音是"矮拼音 + 上面声调 + 下面下伸笔画",
    #   它的**中心**照样落在矮拼音划出来的这条带里, 聚行时按中心判就圈得掉。
    return [(a, b) for a, b in bands]


def _rows(big_boxes, bands=()):
    """按纵向重叠聚行, 边界取成员中位。返回 [(y0, y1)]。

    bands 是拼音带, 中心落在带里的块**不参与聚行** —— 见 _pinyin_bands。
    """
    if not big_boxes:
        return []
    items = sorted((y, y + h) for _, y, _, h in big_boxes
                   if not any(a <= y + h // 2 <= b for a, b in bands))
    if not items:
        return []
    groups: list[list] = []
    for y0, y1 in items:
        placed = False
        for g in groups:
            inter = min(g[1], y1) - max(g[0], y0)
            shorter = min(g[1] - g[0], y1 - y0)
            if shorter > 0 and inter >= OVERLAP * shorter:
                g[0] = min(g[0], y0)
                g[1] = max(g[1], y1)
                g[2].append((y0, y1))
                placed = True
                break
        if not placed:
            groups.append([y0, y1, [(y0, y1)]])
    groups.sort()

    out = []
    for g in groups:
        members = g[2]
        if len(members) < 2:
            continue                       # 孤零零一个块不算一行
        y0 = int(np.median([a for a, _ in members]))
        y1 = int(np.median([b for _, b in members]))
        if y1 - y0 >= MIN_ROW_H:
            out.append((y0, y1))
    return out


def process(args) -> list[dict]:
    path_str, out_dir = args
    p = Path(path_str)
    rows_out: list[dict] = []
    try:
        img = cv2.imdecode(np.fromfile(path_str, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return rows_out
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = text_mask(gray, local_background(gray))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        _, kept, _, big = annotation_labels(mask)
        bands = _pinyin_bands(kept, st, big)
        rows = _rows(big, bands)
        if not rows:
            return rows_out

        char_h = float(np.median([b - a for a, b in rows]))
        fallback = int(round(FALLBACK_SHIFT_RATIO * char_h))

        label_dir = Path(out_dir) / "label"
        input_dir = Path(out_dir) / "input"
        stem = p.stem

        for i, (y0, y1) in enumerate(rows):
            # 压在这一行头上的那条拼音带: 带底在行顶之上, 而且离得不到一行高
            above = [(a, b) for a, b in bands
                     if b <= y0 + 2 and (y0 - b) < (y1 - y0)]
            has_py = bool(above)

            # --- 标签裁图: 中位边界, 再往下夹到拼音带底下 ---
            #
            # ★★ 夹**不得越过行顶** —— 越过就等于切汉字。实测踩过:
            #   不设上限的话, 拼音带和汉字挨得紧的行会被夹掉一半,
            #   `账单详情 全部账单` 裁出来读成 `单 立 k`。
            #   行顶本身已经由 _rows 排除拼音带算出来了, 这一夹只是收掉那点余量。
            la = max(0, y0 - LABEL_PAD)
            if above:
                la = min(y0, max(la, max(b for _, b in above) + 1))
            lb = min(H, y1 + LABEL_PAD)
            if lb - la < MIN_ROW_H:
                continue                    # 夹完太薄就不要这一行了

            # --- 输入裁图: 往上扩到把这一行的拼音包进来 ---
            #   优先用**这一行实测的**拼音带顶, 找不到才退回常数 ——
            #   常数是全局中位, 具体到某一行可能偏大或偏小。
            ia = max(0, (min(a for a, _ in above) - 1) if above
                     else (y0 - fallback))
            ib = lb

            lcrop = img[la:lb, :]
            icrop = img[ia:ib, :]
            if lcrop.size == 0 or icrop.size == 0:
                continue

            name = f"{stem}_r{i:02d}.png"
            label_dir.mkdir(parents=True, exist_ok=True)
            input_dir.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", lcrop)[1].tofile(str(label_dir / name))
            cv2.imencode(".png", icrop)[1].tofile(str(input_dir / name))

            rows_out.append({
                "file": name,
                "source": path_str,
                "row": i,
                "label_y0": la, "label_y1": lb,
                "input_y0": ia, "input_y1": ib,
                # ★ 这一行上面到底有没有拼音。没有的行也留着(模型也要会读普通行),
                #   但训练时可以按这个字段挑。
                "has_pinyin": int(has_py),
                "text": "",          # 留给下一步 OCR 填
            })
    except Exception:                      # noqa: BLE001
        return rows_out
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()

    if not a.src.exists():
        print(f"目录不在: {a.src}")
        return
    files = [str(p) for p in sorted(a.src.iterdir())
             if p.suffix.lower() in EXTS]
    if a.limit:
        files = files[: a.limit]
    if not files:
        print("没找到图")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    print(f"要处理 {len(files):,} 张,  {a.workers} 个进程")

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, rows in enumerate(ex.map(process,
                                        [(f, str(a.out)) for f in files],
                                        chunksize=10), 1):
            all_rows.extend(rows)
            if i % 200 == 0:
                print(f"  {i:,}/{len(files):,}   裁出 {len(all_rows):,} 行")

    man = a.out / "_pairs.csv"
    with man.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "source", "row", "label_y0", "label_y1",
            "input_y0", "input_y1", "has_pinyin", "text"])
        w.writeheader()
        w.writerows(all_rows)

    with_py = sum(r["has_pinyin"] for r in all_rows)
    print()
    print("=" * 60)
    print(f"共裁出 {len(all_rows):,} 行")
    print(f"  上面有拼音的  {with_py:,}  ({with_py/max(1,len(all_rows)):.0%})")
    print(f"  上面没拼音的  {len(all_rows)-with_py:,}")
    print(f"  平均每张      {len(all_rows)/max(1,len(files)):.1f} 行")
    print("=" * 60)
    print(f"标签裁图 -> {a.out / 'label'}")
    print(f"输入裁图 -> {a.out / 'input'}")
    print(f"清单     -> {man}")
    print()
    print("★ 下一步: 对 label/ 跑 OCR 把 text 填上, 再人工核几十行对不对。")
    print("★★ 核过之前**别**跑全量 —— 7159 张约 12 万行, OCR 要跑好几个小时。")


if __name__ == "__main__":
    main()
