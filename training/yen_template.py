r"""蓝图 ¥ 模板生成器 —— 经理提的"图片定位"那条路的标定工具。

经理 2026-09-16: "不行直接图片定位" "把所有的蓝图符号截图" "但是也要分安卓和苹果"。

做什么
------
从语料里挑真图的 ¥, 按金额字高归一化, **按平台各求一张平均模板**,
再把模板量化成 uint8 打成 base64, 直接贴进 `demo/YenTemplateCheck.cs`。

★★★★★ 选模板尺寸**只用真图, 不用假图**
------------------------------------
第一次做这个的时候, 我是按"真图和假图的空档最大"去挑画布尺寸的。
七种尺寸的空档是 -0.002 / 0.028 / 0.034 / 0.061 / 0.061 / 0.066 / 0.107,
**差 50 倍还穿过零** —— 那等于**拿两张假图在挑超参**, 换两张新假图结论就变。
`dotcheck-split` 里写过: "选择指标本身可能是错的, 而且错得看不出来"。

所以改成: **挑尺寸的判据是"苹果真图和安卓真图分得开不开"**, 全是真图, 一张假图不用。
    margin = 苹果留出真图的最低分 − 安卓真图的最高分
假图只在最后**验一下**, 不参与任何选择。

阈值同理: 按**真图分位**定(和 MinusCheck 一个路子), 不是按"刚好卡住那两张假图"定。

用法
----
  python training/yen_template.py c:\projects\China\TempFakeImages --limit 4000 --emit-cs

**只读**: 只读图片, 只往 stdout 打。
"""
from __future__ import annotations

import argparse
import base64
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import yen_scan as ys  # noqa: E402

# Apple 官方分辨率表(竖屏)。★ 故意剔掉 1080x1920 / 1080x2340, 那两个和安卓撞车。
IPHONE = {(640, 960), (640, 1136), (750, 1334), (1242, 2208), (1125, 2436), (828, 1792),
          (1242, 2688), (1170, 2532), (1284, 2778), (1179, 2556), (1290, 2796),
          (1206, 2622), (1320, 2868)}
AMBIG = {(1080, 1920), (1080, 2340)}
# 整组都长得像苹果的分辨率 = 被裁过/缩过的 iPhone 截图, 建安卓模板时要剔掉
RESCALED = {(1260, 2736), (1280, 2781), (1280, 2774), (1280, 2769), (960, 2079), (3072, 4096)}

# 候选画布 (高, 宽, 归一化后的金额字高)
SIZES = [((110, 90), 100.0), ((66, 54), 60.0), ((55, 45), 50.0), ((44, 36), 40.0),
         ((36, 30), 33.0), ((33, 27), 30.0), ((27, 22), 25.0)]


def raw_yen(path):
    """取原始 ¥ 灰度块 + 金额字高 + 分辨率。缩放留到后面, 同一批图才能喂给所有尺寸。"""
    rgb = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    if not ys.is_blue_page(rgb):
        return None
    loc = ys.locate_amount_blue(rgb)
    if not loc:
        return None
    glyphs, gcrop, _fg = ys._reextract_mask(rgb, (loc[0], loc[1], loc[2], loc[3]))
    if len(glyphs) < 4:
        return None
    yen, rest = glyphs[0], glyphs[1:]
    tall = max(k["h"] for k in rest)
    cw = gcrop.shape[1]
    digits = [k for k in rest if k["h"] >= 0.80 * tall and k["x"] > 0 and k["x"] + k["w"] < cw]
    if len(digits) < 3:
        return None
    d_h = max(k["y"] + k["h"] for k in digits) - min(k["y"] for k in digits)
    if d_h < 20 or not (ys.YEN_H_LOW <= yen["h"] / d_h <= ys.YEN_H_HIGH):
        return None
    return yen["gray"].astype(np.float32), float(d_h), (rgb.shape[1], rgb.shape[0])


def render(raw, dh_norm, can):
    """按给定归一字高和画布渲染成一张可比对的贴片; 放不下就返回 None。"""
    g, d_h, _ = raw
    sc = dh_norm / d_h
    g = cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX)
    nh, nw = max(3, int(round(g.shape[0] * sc))), max(3, int(round(g.shape[1] * sc)))
    if nh > can[0] or nw > can[1]:
        return None
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros(can, np.float32)
    out[(can[0] - nh) // 2:(can[0] - nh) // 2 + nh, (can[1] - nw) // 2:(can[1] - nw) // 2 + nw] = g
    return out


def ncc(a, b):
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def plat_of(sz):
    if sz in IPHONE:
        return "apple"
    if sz in AMBIG or sz in RESCALED:
        return None          # 认不准的一律不用来标定
    return "android"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="生成蓝图 ¥ 平均模板并打成 C# 常量")
    ap.add_argument("input", type=Path)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--bank", type=int, default=80, help="每个平台用多少张图求平均模板")
    ap.add_argument("--emit-cs", action="store_true", help="打印能贴进 .cs 的常量")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(args.input):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in ys._EXTS:
                files.append(os.path.join(dp, fn))
    random.Random(args.seed).shuffle(files)
    files = files[:args.limit]
    print(f"候选 {len(files):,} 张", flush=True)

    raws = {"apple": [], "android": []}
    for i, p in enumerate(files, 1):
        try:
            r = raw_yen(p)
        except Exception:
            continue
        if r is None:
            continue
        pl = plat_of(r[2])
        if pl:
            raws[pl].append(r)
        if i % 1000 == 0:
            print(f"  {i:,}/{len(files):,}  苹果 {len(raws['apple']):,} 安卓 {len(raws['android']):,}",
                  flush=True)
    print(f"取到 ¥: 苹果 {len(raws['apple']):,}   安卓 {len(raws['android']):,}")
    if min(len(raws["apple"]), len(raws["android"])) < args.bank + 100:
        raise SystemExit("!! 样本不够, 加大 --limit")

    bank = {k: v[:args.bank] for k, v in raws.items()}
    held = {k: v[args.bank:] for k, v in raws.items()}

    fakes = [(r"C:\projects\China\2.png", "假 2.png"), (r"C:\projects\China\013.jpg", "假 013")]
    fake_raw = []
    for fp, nm in fakes:
        try:
            fr = raw_yen(fp)
            if fr:
                fake_raw.append((fr, nm))
        except Exception:
            pass

    # ---- 选尺寸: 判据只用真图 ----
    print(f"\n★ 选画布尺寸 —— 判据是**苹果真图 vs 安卓真图**分得开不开, 不用假图")
    print(f"{'画布':<12}{'字节':>7}{'苹果min':>9}{'安卓max':>9}{'margin':>9}   (假图只做验证)")
    best = None
    for can, dh in SIZES:
        tpl = {}
        for k in ("apple", "android"):
            rs = [render(r, dh, can) for r in bank[k]]
            rs = [x for x in rs if x is not None]
            if len(rs) < 20:
                tpl = None
                break
            tpl[k] = np.mean(np.stack(rs), axis=0)
        if tpl is None:
            print(f"{str(can):<12} 渲染不出来")
            continue
        ha = np.array([ncc(x, tpl["apple"]) for x in
                       (render(r, dh, can) for r in held["apple"]) if x is not None])
        hd = np.array([ncc(x, tpl["apple"]) for x in
                       (render(r, dh, can) for r in held["android"]) if x is not None])
        if len(ha) < 100 or len(hd) < 100:
            print(f"{str(can):<12} 留出不够")
            continue
        margin = float(ha.min() - hd.max())
        fk = [ncc(render(r, dh, can), tpl["apple"]) for r, _ in fake_raw]
        print(f"{str(can):<12}{can[0]*can[1]:>7}{ha.min():>9.4f}{hd.max():>9.4f}{margin:>9.4f}"
              f"   假 {['%.4f' % x for x in fk]}")
        if best is None or margin > best[0]:
            best = (margin, can, dh, tpl, ha, hd, fk)

    margin, can, dh, tpl, ha, hd, fk = best
    print(f"\n★★ 选中 {can}, 归一字高 {dh:.0f}, margin {margin:.4f}")
    print(f"   苹果留出真图 n={len(ha)}  min {ha.min():.4f}  p0.1 {np.percentile(ha,0.1):.4f}  "
          f"p1 {np.percentile(ha,1):.4f}  p50 {np.percentile(ha,50):.4f}")
    print(f"   安卓真图     n={len(hd)}  max {hd.max():.4f}  p99.9 {np.percentile(hd,99.9):.4f}")
    print(f"   ★ 假图(只验证不参与选择): {['%.4f' % x for x in fk]}")

    # ---- 阈值: 按真图分位定, 不按假图定 ----
    # ★ 两个平台各用**自己的**模板和自己的真图分布定阈值。
    thr = {}
    for k, other in (("apple", "android"), ("android", "apple")):
        own = np.array([ncc(x, tpl[k]) for x in
                        (render(r, dh, can) for r in held[k]) if x is not None])
        oth = np.array([ncc(x, tpl[k]) for x in
                        (render(r, dh, can) for r in held[other]) if x is not None])
        t = float(np.floor(np.percentile(own, 0.1) * 1000) / 1000.0)
        thr[k] = t
        fkk = [ncc(render(r, dh, can), tpl[k]) for r, _ in fake_raw]
        print(f"\n★★★ {k} 阈值 = 自己真图的 p0.1 向下取整 = {t:.3f}   (n={len(own)})")
        print(f"   本平台真图  min {own.min():.4f}  p0.1 {np.percentile(own,0.1):.4f}  "
              f"p50 {np.percentile(own,50):.4f}   低于阈值 {int((own<t).sum())} 张 "
              f"= {100.0*(own<t).sum()/len(own):.3f}%")
        print(f"   另一平台真图 max {oth.max():.4f}   低于阈值 {int((oth<t).sum())}/{len(oth)} "
              f"= {100.0*(oth<t).sum()/len(oth):.1f}%  (本来就该低)")
        print(f"   ★ 假图(都是苹果): {['%.4f' % x for x in fkk]}   "
              f"低于阈值 {sum(1 for x in fkk if x < t)}/{len(fkk)}")

    if args.emit_cs:
        print("\n" + "=" * 74)
        for k in ("apple", "android"):
            q = np.clip(tpl[k] * 255.0, 0, 255).astype(np.uint8)
            b64 = base64.b64encode(q.tobytes()).decode()
            print(f"// {k} 平均模板  {can[0]}x{can[1]}  归一字高 {dh:.0f}  "
                  f"{q.nbytes} 字节  标定自 {len(bank[k])} 张真图")
            print(f"const string {k.capitalize()}TemplateB64 =")
            for i in range(0, len(b64), 100):
                end = ";" if i + 100 >= len(b64) else " +"
                print(f'    "{b64[i:i+100]}"{end}')
        print(f"public const int TemplateH = {can[0]}, TemplateW = {can[1]};")
        print(f"public const double NormDigitHeight = {dh:.0f};")
        print(f"public const double AppleMinScore = {thr['apple']:.3f};")
        print(f"public const double AndroidMinScore = {thr['android']:.3f};")


if __name__ == "__main__":
    main()
