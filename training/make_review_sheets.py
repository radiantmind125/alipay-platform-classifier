"""把拼音扫描的结果拼成联系表, 供人工核对。

为什么不是"把命中的全看一遍"
----------------------------
全量扫下来命中上千张, 一张张开图不现实, 而且没必要。
人工核对要回答的是两个数, 各自该看哪些图是不同的:

  准确率(判为带拼音的里面有多少真的带)  -> 看命中的
  漏判(真带拼音的里面漏了多少)          -> 看**没命中但接近线**的

而且命中里也不是每张都值得看:
  ★ 错误集中在**贴着线**的那一段。远高于线的基本不会错(实测中位 0.45, 最高 0.73),
    贴线的那些才是真正要看的。
所以分三组:

  A 贴线的命中   比例 0.25~0.32 全看        -> 最容易出错的一段
  B 随机命中     从全部命中里随机抽         -> 用来估准确率(无偏)
  C 差一点的     比例 0.15~0.25 随机抽      -> 用来找漏判

★ B 和 C 必须**随机**抽, 不能取前 N 张 —— 文件名带时间戳, 取头部只会取到一天。

用法
----
    python make_review_sheets.py 扫描出来的.csv 图片目录 输出目录
"""
import csv
import io
import os
import random
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CELL_W = 430          # 每格宽度
COLS = 4              # 每行几格
PER_SHEET = 12        # 每张表几格 —— 格子少一点, 看的人不容易看漏
SEED = 20260912


def find_images(roots):
    """建一张 文件名 -> 完整路径 的表。"""
    idx = {}
    for root in roots:
        for dirpath, _, names in os.walk(root):
            for nm in names:
                if os.path.splitext(nm)[1].lower() in ('.jpg', '.jpeg', '.png'):
                    idx.setdefault(nm, os.path.join(dirpath, nm))
    return idx


def cell(path, label, ok_color=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    # 上面这一段(标题 + 户名 + 金额)拼音最好认
    crop = img[int(H * 0.05):int(H * 0.32), :]
    if crop.size == 0:
        return None
    sc = CELL_W / crop.shape[1]
    crop = cv2.resize(crop, (CELL_W, max(1, int(crop.shape[0] * sc))),
                      interpolation=cv2.INTER_AREA)
    bg = (205, 235, 205) if ok_color else (200, 215, 255)
    bar = np.full((26, CELL_W, 3), bg, np.uint8)
    cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (10, 10, 10), 1, cv2.LINE_AA)
    return np.vstack([bar, crop])


def sheets(items, out_dir, tag, ok_color=True):
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for s in range(0, len(items), PER_SHEET):
        chunk = items[s:s + PER_SHEET]
        cells = []
        for label, path in chunk:
            c = cell(path, label, ok_color)
            if c is not None:
                cells.append(c)
        if not cells:
            continue
        ch = max(c.shape[0] for c in cells)
        rows = (len(cells) + COLS - 1) // COLS
        canvas = np.full((rows * (ch + 6) + 6, COLS * (CELL_W + 6) + 6, 3), 242, np.uint8)
        for i, c in enumerate(cells):
            r, k = divmod(i, COLS)
            y, x = 6 + r * (ch + 6), 6 + k * (CELL_W + 6)
            canvas[y:y + c.shape[0], x:x + CELL_W] = c
        p = os.path.join(out_dir, f'{tag}_{made:02d}.png')
        cv2.imwrite(p, canvas)
        made += 1
    print(f'  {tag}: {len(items)} 张 -> {made} 张联系表')
    return made


def main():
    csv_path, out_dir = sys.argv[1], sys.argv[-1]
    roots = sys.argv[2:-1]
    idx = find_images(roots)
    print(f'图片索引 {len(idx):,} 张')

    rows = []
    with io.open(csv_path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                r['pinyin_ratio'] = float(r['pinyin_ratio'])
                r['stacked'] = int(r['stacked'])
                r['flat'] = float(r['flat'])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    print(f'CSV 里 {len(rows):,} 行')

    def hit(r):
        return r['pinyin_ratio'] >= 0.25 and r['stacked'] >= 15 and r['flat'] >= 0.15

    hits = [r for r in rows if hit(r)]
    near = [r for r in rows if not hit(r) and 0.15 <= r['pinyin_ratio'] < 0.25]
    print(f'命中 {len(hits):,} 张, 差一点的 {len(near):,} 张')

    rnd = random.Random(SEED)

    # A: 贴线的命中, 全看
    A = sorted([r for r in hits if r['pinyin_ratio'] < 0.32],
               key=lambda r: r['pinyin_ratio'])
    # B: 从全部命中里随机抽 120 张估准确率
    B = hits[:]
    rnd.shuffle(B)
    B = B[:120]
    # C: 差一点的里面随机抽 60 张找漏判
    C = near[:]
    rnd.shuffle(C)
    C = C[:60]

    def to_items(rs):
        out = []
        for r in rs:
            p = idx.get(r['name'])
            if p:
                out.append((f"{r['pinyin_ratio']:.3f} f{r['flat']:.2f}", p))
        return out

    print('\n出联系表:')
    sheets(to_items(A), os.path.join(out_dir, 'A_borderline'), 'A')
    sheets(to_items(B), os.path.join(out_dir, 'B_random'), 'B')
    sheets(to_items(C), os.path.join(out_dir, 'C_nearmiss'), 'C', ok_color=False)
    print(f'\n都在 {out_dir}')
    print('  A = 贴着线的命中, 全看, 看有没有不带拼音的混进来')
    print('  B = 随机抽的命中, 用来估准确率')
    print('  C = 没命中但接近线的, 看有没有真带拼音的被漏掉')


if __name__ == '__main__':
    main()
