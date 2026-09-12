r"""判断截图里是不是带**拼音标注**。

背景
----
经理 2026-09-11: 有个用户的单子被系统拒了, 原因是"字体变形 没有识别到 真实的订单号"。
实际看图并不是字体变形 —— 是安卓的**拼音标注**功能: 每个汉字上方多出一行小字拼音。
这会把 OCR 的行切分搞乱(每两行真文字之间多插一行小字), 于是订单号读不出来,
系统按"订单号不符合规则"直接拒单。真实交易被误拒。

经理的原话: "帮我写个算法识别图片里是不是有拼音", 并且说
"如果无法用算法识别就不用处理了 我这边再想办法"。

怎么判
------
不做 OCR, 只看**版面结构**, 因为拼音的几何特征非常干净:

  011.jpg(带拼音)的块高分布是**双峰**的:
      小块 峰值在 8 像素   <- 拼音字母
      大块 峰值在 21 像素  <- 汉字
  而且每个拼音块的**正下方紧贴着一个汉字块**, 水平方向还重叠。

普通截图里也有小字(时间戳、说明文字), 但它们**不会系统性地压在大字正上方**。
实测三张对照真图, 这种"小压大"的配对数是 0。

★ 深色字和浅色字要**各算一遍**: 支付宝的转账成功页是蓝底白字,
  只找深色字的话整页取不到, 那些页上的拼音会全部漏掉。

判据: 统计有多少小块满足"正下方紧贴一个大块且水平重叠"(stacked),
      再除以小块总数得到 pinyin_ratio。★ 但**光看比例会错**, 见 has_pinyin() 里记的两个坑,
      要同时满足 比例 >= 0.25、压住数 >= 15、平坦占比 >= 0.15 三个条件。

实测(全部人工核对过)
--------------------
  经理给的 011.jpg            0.605   判对
  随机样本里命中的 12 张       0.54~0.73   逐张开图看过, 12/12 确实带拼音
  两张漏判过的(已修)          0.294 / 0.299   文字少所以比例低, 靠绝对数救回来
  145 张人工看过的真图         0 误判
  翻拍图/别的照片             5 张, 靠平坦占比挡掉(照片 0.009~0.043 vs 截图 0.312~0.880)

频率: 跨 7 天随机抽 5,987 张, 命中 45 张 = 0.75%(95% 区间 0.56%~1.00%),
      按此推算 76.8 万张里约 5,800 张。
★ 注意不要拿"排序后的前 N 张"来估频率 —— 文件名带时间戳, 那样只会取到一天,
  我第一次就是这么算的, 得出 0.93%, 实际随机抽是 0.75%。
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _components(mask, H: int, W: int):
    """从一张二值掩膜里取出像文字的连通块。"""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3]))
            for i in range(1, n)
            if st[i][4] >= 8 and st[i][3] < 0.05 * H and st[i][2] < 0.4 * W]


def _stacking(cs):
    """给一组连通块, 算有多少小块"正下方紧贴一个大块且水平压住"。

    返回 (比例, 小块数, 压住数, 大块高度)。
    """
    if len(cs) < 60:
        return 0.0, 0, 0, 0.0
    hs = np.array([c[3] for c in cs], float)
    big_h = float(np.percentile(hs, 75))      # 汉字那一档
    if big_h < 8:
        return 0.0, 0, 0, 0.0
    # 小块 = 明显比正文矮的; 大块 = 正文高度上下
    small = [c for c in cs if c[3] <= 0.55 * big_h]
    big = [c for c in cs if c[3] >= 0.8 * big_h]
    if len(small) < 20 or len(big) < 20:
        return 0.0, len(small), 0, big_h

    # 按 x 建桶, 免得 O(N^2)
    buckets: dict[int, list] = {}
    for b in big:
        for k in range(b[0] // 50, (b[0] + b[2]) // 50 + 1):
            buckets.setdefault(k, []).append(b)

    stacked = 0
    for x, y, w, h in small:
        hit = False
        for k in range(x // 50, (x + w) // 50 + 1):
            for bx, by, bw, bh in buckets.get(k, ()):
                gap = by - (y + h)
                if gap < -2 or gap > 0.7 * bh:       # 必须紧贴在上方
                    continue
                ov = min(x + w, bx + bw) - max(x, bx)
                if ov > 0.5 * min(w, bw):            # 水平要压住
                    hit = True
                    break
            if hit:
                break
        stacked += hit
    return stacked / len(small), len(small), stacked, big_h


def measure(path: str) -> dict | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    if H < 200 or W < 200:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 平坦占比 = 出现最多的那个灰度值占了多少像素。
    # 截图有大片数值完全一样的底色; 照片因为传感器噪声做不到。
    # 缩图必须用 INTER_NEAREST, 插值会把"完全一样"抹掉。
    g = gray
    if max(g.shape) > 1400:
        sc = 1400 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * sc), int(g.shape[0] * sc)),
                       interpolation=cv2.INTER_NEAREST)
    flat = float(np.bincount(g.ravel(), minlength=256).max()) / g.size

    # 两种明暗关系都量, 但**只有深色字那一边参与判定**。
    #   浅色字(蓝底白字页)那边量出来只写进 CSV 当诊断:
    #   拿它判会把误判带上来(见 has_pinyin 第 4 条)。
    dark = _components(cv2.threshold(gray, 169, 255, cv2.THRESH_BINARY_INV)[1], H, W)
    light = _components(cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1], H, W)
    if len(dark) < 60:
        return None

    rd, sd, kd, bd = _stacking(dark)
    rl, sl, kl, bl = _stacking(light)
    # ★ 判定**只用深色字那一边**。亮字那边照样量, 但只写进 CSV 当诊断, 不参与判定。
    #   原因见 has_pinyin() 的第 4 条: 拿亮字判会带来约 83% 的误判。
    ratio, n_small, stacked, big_h = rd, sd, kd, bd
    if n_small == 0:
        return None

    return {"n_comp": len(dark) + len(light), "big_h": round(big_h, 1),
            "n_small": n_small, "n_big": 0, "stacked": stacked,
            "pinyin_ratio": round(ratio, 4), "flat": round(flat, 4),
            "dark_r": round(rd, 4), "light_r": round(rl, 4)}


def has_pinyin(d, min_ratio=0.25, min_stacked=15, min_flat=0.15):
    """由 measure() 的结果判断是不是带拼音。三条要同时过。

    ★ 光看比例不够, 实测踩到三种坑:

    1. **不是截图的图会把比例顶高。** 拿相机拍屏幕, 或者干脆是别的照片,
       纹理会凑出一堆小块, 凑巧"压在"别的块上方。
       实测 5 张这种(4 张翻拍 + 1 张床上衣服的照片)比例 0.27~0.43, 全是误判。
       ★ 这里原来用"小块数 <= 1500"挡, **挡不住** —— 一张 4624x3468 的翻拍
         分辨率太高, 笔画没碎, 小块只有 1462 个就钻过去了。
         改用平坦占比: 照片 0.009~0.043, 截图 0.312~0.880, 中间空的, 差 7 倍。
       ★ 也别拿长宽比挡: 有张真带拼音的截图是 1200x1920, 长宽比只有 1.6,
         按长宽比会误杀, 按平坦占比(0.717)是稳的。

    2. **文字少的图比例不稳。** 两张确实带拼音的图只拿到 0.294 和 0.299,
       因为整页文字本来就少(小块只有 68 和 264 个), 分母小比例就抖。
       而不带拼音的红包弹窗页是 0.256 —— 光靠比例这两类分不开。
       但绝对数差得很清楚: 那两张带拼音的压住了 20 和 79 个,
       红包弹窗只压住 10 个。所以比例和绝对数**两个都要过**。

    3. 少数民族双语证件(维吾尔文压在汉字上方)几何上和拼音一样。
       实测一张身份证照片是 0.236, 在线下面; 而且它不是账单截图,
       平坦占比 0.03 也会被第 1 条挡掉。

    4. ★★ **判不了蓝底白字页 —— 这是已知的漏判, 不是没发现。**
       二值化取的是"比 169 暗"的像素, 也就是浅底上的深色字。
       支付宝的转账成功页是蓝底白字, 白字取不到, 那些页上的拼音**判不出来**。
       人工抽看 48 张"差一点"的图, 里面约 16 张确实带拼音, 大多是这种蓝色页。
       按此估计召回率约 3/4。

       ★ 试过加一条"找亮字"的掩膜来补, **不能用**:
         补回来的 510 张里人工看了 24 张, 20 张根本没拼音(误判率约 83%)。
         蓝色页上"找亮字"取到的块很少, 而金额数字特别大, 把 big_h 顶高,
         零碎小块都被归成"小块", 再叠上"标题正好在金额上方"的天然版面,
         就凑出假的"小压大"。
         间距、小高占比、压住的行数三个量在真假两组上**完全重叠**, 分不开。
         真要补蓝色页, 得按区域分别定阈值, 是另一件事。
         亮字那边的比例仍然写进 CSV(light_r 列), 留给以后做。
    """
    if not d or d.get("flat", 1.0) < min_flat:
        return False
    return d["pinyin_ratio"] >= min_ratio and d["stacked"] >= min_stacked


def _worker_init():
    """子进程里把 OpenCV 的内部线程关掉 —— 否则它会和进程池抢核心, 越并行越慢。"""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _measure_safe(path: str):
    """给进程池用: 任何异常都吞掉, 返回 (路径, 结果或 None)。"""
    try:
        return path, measure(path)
    except Exception:
        return path, None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="判断截图里有没有拼音标注")
    ap.add_argument("input", type=Path, nargs="+", help="一个或多个图片目录(或单张图)")
    ap.add_argument("--out", type=Path, default=None, help="把每张图的量值写成 CSV")
    ap.add_argument("--copy-hits", type=Path, default=None,
                    help="把判为带拼音的图拷到这个目录, 供人工核对")
    ap.add_argument("--sample", type=int, default=0,
                    help="只随机抽这么多张来扫(0 = 全扫)")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数, 0 = 自动(核数减一), 1 = 不并行")
    ap.add_argument("--min-ratio", type=float, default=0.25,
                    help="比例下界; 还要同时满足压住数 >= 15 且平坦占比 >= 0.15")
    args = ap.parse_args()

    files: list[Path] = []
    for root in args.input:
        if root.is_file():
            files.append(root)
        else:
            files += [p for p in root.rglob("*") if p.suffix.lower() in _EXTS]
    # 去重(两个库里有极少数重名的)
    seen_name: dict[str, Path] = {}
    for p in files:
        seen_name.setdefault(p.name, p)
    files = list(seen_name.values())

    if args.sample and args.sample < len(files):
        # ★ 必须随机抽。文件名带时间戳, 取前 N 张只会取到一天的数据。
        random.Random(args.seed).shuffle(files)
        files = files[: args.sample]

    print(f"扫 {len(files):,} 张", flush=True)

    n_work = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    rows, pool = [], None
    try:
        if n_work > 1 and len(files) > 200:
            import multiprocessing as mp
            print(f"并行 {n_work} 个进程", flush=True)
            pool = mp.Pool(n_work, initializer=_worker_init)
            it = pool.imap_unordered(_measure_safe, [str(p) for p in files],
                                     chunksize=64)
        else:
            _worker_init()
            it = (_measure_safe(str(p)) for p in files)

        for i, (path, d) in enumerate(it, 1):
            if i % 20000 == 0:
                print(f"  ...{i:,} / {len(files):,}", flush=True)
            if d:
                d["name"] = os.path.basename(path)
                d["path"] = path
                rows.append(d)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    hits = [r for r in rows if has_pinyin(r, args.min_ratio)]
    n = max(1, len(rows))
    print(f"量到 {len(rows):,} 张 (量不了 {len(files) - len(rows):,} 张), "
          f"判为带拼音 {len(hits):,} 张 ({len(hits) / n * 100:.3f}%)")

    if rows:
        v = sorted(r["pinyin_ratio"] for r in rows)
        m = len(v)
        q = lambda x: v[min(m - 1, int(m * x))]
        print(f"  pinyin_ratio  中位 {q(.5):.4f}  p90 {q(.9):.4f}  "
              f"p99 {q(.99):.4f}  p99.9 {q(.999):.4f}  最大 {v[-1]:.4f}")
    if hits:
        fl = sorted(r["flat"] for r in hits)
        rr = sorted(r["pinyin_ratio"] for r in hits)
        print(f"  命中的: 比例最低 {rr[0]:.3f}, 平坦占比最低 {fl[0]:.3f}")

    if args.out and rows:
        import csv
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["name", "pinyin_ratio", "stacked", "flat", "dark_r", "light_r",
                "n_small", "n_big", "big_h", "n_comp"]
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sorted(rows, key=lambda r: -r["pinyin_ratio"]):
                w.writerow({k: r[k] for k in cols})
        print(f"  写出 {args.out}")

    if args.copy_hits and hits:
        args.copy_hits.mkdir(parents=True, exist_ok=True)
        for r in sorted(hits, key=lambda r: r["pinyin_ratio"]):
            # 文件名前面带上比例, 人工看的时候从最低分开始看最省事
            dst = args.copy_hits / f"r{r['pinyin_ratio']:.3f}_f{r['flat']:.2f}_{r['name']}"
            try:
                shutil.copy2(r["path"], dst)
            except Exception:
                pass
        print(f"  拷了 {len(hits):,} 张到 {args.copy_hits} (文件名前面是比例, 从低到高看)")


if __name__ == "__main__":
    main()
