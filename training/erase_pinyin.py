"""把拼音标注**擦掉**, 还原成一张普通截图。

思路
----
经理要的是"拼音的图 OCR 读不全"这件事。与其让模型去学会无视拼音,
不如**在送进 OCR 之前先把拼音擦掉** —— 擦干净了它就不再是拼音图,
现成的 OCR 直接就能用, 不需要训练任何模型。

怎么擦
------
1. 先判这一页到底带不带拼音(整页比例, 和 pinyin_probe 同一套判据)
2. 水平投影切行
3. 找"矮行紧压在高行上方"的配对 —— 矮的那行就是注音行
4. **只擦注音行里的墨点**, 用局部底色填回去

★★ 第 4 步是**外科式**的: 不是把整条带刷成白色, 而是只把被判成文字的像素
   换成"这个位置本来的底色"。所以卡片边框、彩色块、右边同一高度的别的内容
   都不会被动到。

★★ 安全阀: 默认**只对判定带拼音的页**动手(`--force` 才跳过这个检查)。
   因为"矮行压在高行上"这个形状, 拼音和普通版面(标题压金额、字段名压值)
   **是一样的** —— 之前实测过, 单看一处根本分不开, 必须整页判定说了算。

    python erase_pinyin.py --src 目录或单图 --out 输出目录
    python erase_pinyin.py --src 一张图 --out 目录 --debug   # 同时存一张对照图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pinyin_probe import has_pinyin, measure  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIFF_THRESHOLD = 28            # 和 pinyin_probe / TextRegion 一致
ANNOT_HEIGHT_RATIO = 0.6       # 注音行相对下面那行的高度上限
ANNOT_GAP_RATIO = 0.5          # 注音行和下面那行的间距上限
MIN_RUN = 4                    # 同一高度上至少要凑成这么多个才算一排拼音


def local_background(gray: np.ndarray) -> np.ndarray:
    """局部底色估计。降采样 -> 中值 -> 升采样, 和取字用的是同一套。"""
    H, W = gray.shape
    s = cv2.resize(gray, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    k = min(21, min(s.shape[0], s.shape[1]))
    if k % 2 == 0:
        k -= 1
    if k >= 3:
        s = cv2.medianBlur(s, k)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


def text_mask(gray: np.ndarray, bg: np.ndarray) -> np.ndarray:
    return (cv2.absdiff(gray, bg) > DIFF_THRESHOLD).astype(np.uint8)


def annotation_labels(mask: np.ndarray):
    """逐**连通块**判哪些是注音字。

    返回 (标签图, 注音块的标签号, 总块数, 汉字块的外接框列表)。

    ★★ 为什么不按"整行"判: 回单是左标签右取值的两栏版面, 而且两栏的基线
       **不在同一高度**。整幅宽做水平投影会把左边标签的拼音和右边取值的拼音
       糊成一条带, 于是"矮行压在高行上"这个形状就不成立了 ——
       实测就是这样漏掉了一半(付款方式、账户余额、支付奖励都没擦掉)。
       改成逐连通块判, 左右两栏各判各的, 互不干扰。

    判据和 pinyin_probe._stacking 完全一致: 小块正下方紧贴一个大块, 且水平压住。
    """
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    comps = [(i, int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3]))
             for i in range(1, n)
             if st[i][4] >= 8 and st[i][3] < 0.05 * H and st[i][2] < 0.4 * W]
    if len(comps) < 60:
        return labels, [], len(comps), []

    hs = np.array([c[4] for c in comps], float)
    big_h = float(np.percentile(hs, 75))          # 汉字那一档
    if big_h < 8:
        return labels, [], len(comps), []
    small = [c for c in comps if c[4] <= 0.55 * big_h]
    big = [c for c in comps if c[4] >= 0.8 * big_h]
    if len(small) < 20 or len(big) < 20:
        return labels, [], len(comps), []

    buckets: dict[int, list] = {}
    for b in big:
        for k in range(b[1] // 50, (b[1] + b[3]) // 50 + 1):
            buckets.setdefault(k, []).append(b)

    hit_labels = []
    for idx, x, y, w, h in small:
        for k in range(x // 50, (x + w) // 50 + 1):
            done = False
            for _, bx, by, bw, bh in buckets.get(k, ()):
                if by < y:
                    continue
                if min(x + w, bx + bw) - max(x, bx) <= 0.5 * min(w, bw):
                    continue
                gap = by - (y + h)
                if gap < -2 or gap > ANNOT_GAP_RATIO * bh:
                    continue
                hit_labels.append(idx)
                done = True
                break
            if done:
                break
    # ★★★★ 到这里为止的判据**不足以用来擦**。
    #   检测(数比例)和擦除(改像素)对误判的容忍度完全不同:
    #   数比例时混进几个误判无所谓, 整页比例照样高;
    #   擦除时**每一个误判都是一个被啃坏的汉字**。
    #
    #   实测踩到的坑: 汉字自己的**分离笔画**和拼音长得一模一样 ——
    #       方 = 亠 + 万   上面那一横是独立连通块, 底下紧跟一个大块
    #       注 = 氵 + 主   三点水最上面那点同理
    #   于是 付款方式 -> 付款万式, 转账备注 -> 转账备汪。
    #
    #   区分办法: **真拼音是一整排**。一行拼音是很多个小块并排在同一高度上,
    #   而汉字的散笔画在它那个高度上是**孤零零一个**。
    #   所以只留"同一高度上凑得成一排"的那些。
    groups: dict[int, list] = {}
    for idx in hit_labels:
        c = next(x for x in small if x[0] == idx)
        cy = c[2] + c[4] // 2
        key = cy // max(4, int(big_h * 0.25))      # 按高度分桶
        groups.setdefault(key, []).append(idx)

    kept = []
    for key, members in groups.items():
        # 相邻桶也算同一排(避免正好卡在桶边界上)
        n = len(members) + len(groups.get(key - 1, [])) + len(groups.get(key + 1, []))
        if n >= MIN_RUN:
            kept.extend(members)

    big_boxes = [(c[1], c[2], c[3], c[4]) for c in big]
    return labels, kept, len(comps), big_boxes


def erase(img: np.ndarray, force: bool = False,
          meas: dict | None = None) -> tuple[np.ndarray, dict]:
    """返回(擦完的图, 统计)。判定不带拼音且没 --force 时原样返回。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bg = local_background(gray)
    mask = text_mask(gray, bg)
    labels, hits, n_comp, big_boxes = annotation_labels(mask)

    info = {"components": n_comp, "annotation_glyphs": len(hits),
            "erased_px": 0, "acted": False}
    if not hits:
        return img, info
    if not force and meas is not None and not has_pinyin(meas):
        info["skipped_not_pinyin"] = True
        return img, info

    # 把判成注音的那些块做成一张掩膜
    erase_mask = np.isin(labels, hits).astype(np.uint8)

    # ★ 要膨胀一点才能盖住笔画边缘的抗锯齿像素(那些和底色的差没到阈值, 不在掩膜里,
    #   不盖住就留一圈灰影)。
    # ★★ 但**绝对不能向下膨胀**。拼音离它底下那个汉字只有 1-3 像素,
    #    向下多吃几像素就会啃掉汉字的第一笔 —— 实测 3x3 膨胀 2 次之后:
    #        付款方式 -> 付款万式      (方 的头一横被擦了)
    #        对方账户 -> 对万账户
    #        转账备注 -> 转账备汪      (注 的三点水上面那点被擦了)
    #    **擦掉拼音是为了让 OCR 读对, 把汉字啃坏比留点残影严重得多。**
    #    所以用一个顶行为 0 的核: 只往上和左右长, 不往下长。
    kernel_up = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)
    erase_mask = cv2.dilate(erase_mask, kernel_up, iterations=2)

    # ★ 再加一道保险: 凡是落在"汉字块"范围里的像素一律不擦。
    #   膨胀方向已经限制过了, 这条是兜底 —— 宁可留残影, 不许动汉字。
    if big_boxes:
        guard = np.zeros_like(erase_mask)
        for bx, by, bw, bh in big_boxes:
            guard[by:by + bh, bx:bx + bw] = 1
        erase_mask = erase_mask & (1 - guard)

    out = img.copy()
    ys, xs = np.nonzero(erase_mask)
    if ys.size:
        if out.ndim == 3:
            for c in range(out.shape[2]):
                out[ys, xs, c] = bg[ys, xs]
        else:
            out[ys, xs] = bg[ys, xs]
    info["erased_px"] = int(ys.size)
    info["acted"] = True
    return out, info


def process(src: Path, out_dir: Path, force: bool, debug: bool,
            no_erase: bool = False) -> dict | None:
    data = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if no_erase:
        # ★★★★★ 对照组: **一个像素都不改**, 但走**完全一样**的解码和存盘。
        #
        #   为什么非要有这一组: 原图是 jpg, 擦完存的是 png。
        #   直接拿"原图 jpg"和"擦完 png"去比, 比的是**擦拼音 + 换编码**两件事,
        #   分不清哪件起的作用。png 无损而 jpg 有压缩痕迹, OCR 结果本来就可能不一样。
        #
        #   这个坑之前踩过一次: 拿 jpg 解码的图和 BGR 存成 png 再解码的图比,
        #   报出来 40 张全被擦坏了, 其实**一张都没坏**, 差的全是编码。
        #
        #   有了这一组, 三列一比就分得清楚:
        #       原图jpg -> 原图png   这一步的差 = 换编码带来的
        #       原图png -> 擦完png   这一步的差 = **真正擦拼音带来的**
        # acted 这个键主循环要用来计数, 不能少
        cleaned, info = img, {"acted": False, "reason": "no_erase 对照组"}
    else:
        meas = None if force else measure(str(src))
        cleaned, info = erase(img, force=force, meas=meas)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_clean.png"
    ok, buf = cv2.imencode(".png", cleaned)
    if ok:
        buf.tofile(str(dst))
    if debug:
        pair = np.hstack([img, cleaned])
        ok2, b2 = cv2.imencode(".png", pair)
        if ok2:
            b2.tofile(str(out_dir / f"{src.stem}_before_after.png"))
    info["file"] = src.name
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="跳过整页拼音判定, 见得到注音行就擦。**慎用**")
    ap.add_argument("--debug", action="store_true", help="另存一张左右对照图")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-erase", action="store_true",
                    help="★ 对照组: 一个像素都不改, 但走一样的解码和存盘。"
                         "拿来把'换编码'和'擦拼音'两件事分开。")
    a = ap.parse_args()
    if a.force and a.no_erase:
        print("--force 和 --no-erase 是反的, 不能一起给")
        return

    files = ([a.src] if a.src.is_file() else
             sorted(p for p in a.src.iterdir()
                    if p.is_file() and p.suffix.lower() in EXTS))
    if a.limit:
        files = files[: a.limit]

    acted = skipped = failed = 0
    for i, p in enumerate(files, 1):
        info = process(p, a.out, a.force, a.debug, a.no_erase)
        if info is None:
            failed += 1
        elif info["acted"]:
            acted += 1
        else:
            skipped += 1
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    print(f"\n共 {len(files)} 张: 擦了 {acted}, 跳过 {skipped}, 读不了 {failed}")
    print(f"输出 -> {a.out}")


if __name__ == "__main__":
    main()
