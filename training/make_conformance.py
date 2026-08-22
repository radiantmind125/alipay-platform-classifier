r"""生成**分阶段**一致性测试向量, 给 .NET 那边逐级核对(只写 --out, 不碰任何真实图)。

为什么必须分阶段
----------------
只给"最后那个分数"是不够的。对不上的时候, 只知道"结果不一样",
不知道是**随机数不对**、**坐标算错**、**选块规则反了**, 还是**喂给模型的数据不对** ——
只能从头猜。这条线上已经为这种猜法付过一次代价了。

所以这里按 **12 级台阶**给(前 11 级要图, **第 12 级判定层不用图**), 每一级都能单独对:

  线路A(整图)
    1. 发生器      给定种子, xorshift32 的前 8 个输出
    2. 坐标        第 0 轮的全部 64 个 (y, x)
    3. 选块        16 轮各自选中的 (y, x) 和它的纹理能量
    4. 模型        16 个块各自的 ai_score
    5. 汇总        16 个分数的平均 = final_ai_score

  线路B(局部改金额) —— ★ 只有传了 `--b-onnx` 才生成
    6. 版式        white 还是 blue
    7. 定位        金额框坐标; 定不到是 null
    8. 裁剪        外扩后的裁剪框 + 切块网格 (2x2 还是 3x6)
    9. 切块        每一块在裁剪图里的坐标
   10. 模型        每块的 ai_score
   11. 汇总        最高 3 个分数的平均 = tile_top3

  判定层(`decision_cases`) —— ★ **不需要图**, 是一张真值表
   12. 判定       (A分, B分, B定没定位到, 扩展名, 线路C铁证) -> 自动拒 / 人工复核 / 放行

**哪一级先对不上, 问题就在那一级。**

★ 为什么判定层要单独做一份**不用图**的:
  那 7 张合成图**走不到**判定层的大多数分支 —— 它们没有 AIGC 元数据(线路C 永远空),
  而且**全是 .png**(`.jpeg` 免自动拒永远不触发)。
  所以只对到 stage11 的话, **判定规则本身一条都没被验过**。
  真值表覆盖了几处最容易写错的:
    - `>=` 是闭区间: 分数**等于**严格线就该拒
    - 线路B **没定位到时不出信号**, 哪怕分数 0.9999 也一样
    - **线路C 命中会短路**, `.jpeg` 免自动拒**管不到它**
    - `.jpg` 不等于 `.jpeg`

★ 台阶 8 最要紧: `crop_box` 验得出 **pad 是不是 8**, `tile_grid` 验得出
  **定位成功时是不是 2x2** —— 这两处正是我们自己写错过的, 光对最后的分数看不出来。

★ 测试图是**程序生成的合成图**, 不是真实收据 ——
  真实收据不能给外部, 而合成图两边都能拿到一模一样的字节, 反而更适合做一致性核对。

用法
----
  python training/make_conformance.py --onnx E:\SSP_Work\onnx\aigen_v7.onnx ^
      --out E:\SSP_Work\onnx\conformance

生成 `images/*.png` 和 `vectors.json`, 两样一起给对方。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def make_images(out: Path) -> list[Path]:
    """造几张**收据模样**的合成图, 覆盖不同版式和尺寸。

    刻意包含: 大片纯色(考验"并列取最后一个")、密集文字(高能量块)、
    渐变(蓝图那种)、以及一张窄长图(考验坐标范围)。
    """
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260821)          # 固定种子 -> 谁跑都得到同样的图
    specs = [
        ("case1_white_text", 420, 900, "white"),
        ("case2_blue_grad", 400, 880, "blue"),
        ("case3_mostly_flat", 360, 760, "flat"),
        ("case4_dense", 480, 1040, "dense"),
        ("case5_narrow", 200, 1200, "white"),
        # ★ 真实进件是 1080x2400 这个量级, 上面几张都太小。
        #   坐标是 next() % (H-31), 图越大用到的随机数位数越多 ——
        #   拿一张**真实尺寸**的进来, 免得实现方在小图上全对、一上真图就偏。
        ("case6_fullsize", 1080, 2400, "white"),
        # ★ 上面六张**都能定位到金额**, 于是全走 2x2 那条分支。
        #   这一张故意**定位不到**(只有小字, 没有更高的金额行), 用来覆盖另一条分支:
        #   退回 roi-top 0.6 只保留上部, 并切 3x6。**两条分支都要有用例。**
        ("case7_noamount", 380, 820, "noloc"),
    ]
    paths = []
    for name, w, h, kind in specs:
        if kind == "blue":
            a = np.zeros((h, w, 3), np.uint8)
            for y in range(h):                      # 竖直渐变, 模仿蓝底转账页
                a[y] = (30 + y * 40 // h, 90 + y * 60 // h, 200 + y * 40 // h)
        else:
            a = np.full((h, w, 3), 248 if kind != "flat" else 255, np.uint8)
        # ★ 每一行都画成**若干个分开的小块**, 不是一整条。
        #   金额定位器要求"同一行里至少 2 个连通域", 画成一整条的话整行只有 1 个连通域,
        #   定位必然失败 —— 第一版就是这样, 六张图定位率 0%, 于是**线路B 那段代码根本没被走到**,
        #   两个 bug(pad 下限 8 / 定位成功切 2x2)在合成图上完全看不出来。
        def row(y0: int, hgt: int, x0: int, x1: int, n: int, lo: int, hi: int) -> None:
            gap = max(2, (x1 - x0) // (n * 4))
            bw = max(3, ((x1 - x0) - gap * (n - 1)) // n)
            for k in range(n):
                bx = x0 + k * (bw + gap)
                if bx + bw <= x1:
                    a[y0:y0 + hgt, bx:bx + bw] = rng.integers(lo, hi, (1, 1, 3), dtype=np.uint8)

        # ★ 蓝图要画**近白**的字(蓝底白字), 因为蓝图定位器找的是
        #   `min(R,G,B) > 175 且 max-min < 45` 的连通域。画成深色字它一个都找不到 ——
        #   而真实的蓝底转账页本来就是白字。
        lo, hi = (232, 256) if kind == "blue" else (0, 70)
        alo, ahi = (238, 256) if kind == "blue" else (0, 40)
        if kind != "flat":
            nrow = 22 if kind == "dense" else 9
            for i in range(nrow):                   # 正文小字: 每行 6~8 个小块
                y0 = int(h * (0.08 + 0.82 * i / nrow))
                row(y0, max(6, h // 90), int(w * 0.10), int(w * (0.55 + 0.35 * ((i * 7) % 5) / 5)),
                    6 + (i % 3), lo, hi)
        # 金额那一行: **明显更高**的 6 个块, 让"取中位高度最大的一行"能选中它。
        # noloc 那张**故意不画**这一行 -> 所有行高度一样, 没有"更高的一行" -> 定位失败。
        if kind != "noloc":
            row(int(h * 0.30), int(h * 0.055), int(w * 0.28), int(w * 0.72), 6, alo, ahi)
        p = out / f"{name}.png"
        Image.fromarray(a).save(p)
        paths.append(p)
    return paths


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="生成分阶段一致性测试向量")
    ap.add_argument("--onnx", type=Path, required=True, help="线路A 的 ONNX")
    ap.add_argument("--b-onnx", type=Path, default=None,
                    help="线路B 的 ONNX。给了才会出线路B 的台阶。**强烈建议给** —— "
                         "线路B 正是刚查出两个 bug 的地方(pad 下限 8 / 定位成功切 2x2)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--trainsize", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=16)
    args = ap.parse_args()

    import importlib.util
    if not importlib.util.find_spec("onnxruntime"):
        raise SystemExit("缺 onnxruntime —— pip install onnxruntime")
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX 不存在: {args.onnx}")

    import onnxruntime as ort                                   # noqa: E402
    from patch_select import (energy, select_patches,           # noqa: E402
                              select_positions, xorshift32)
    from predict_tiled import _richest_patch, _tiles            # noqa: E402
    from locate_blue import locate_amount_auto                  # noqa: E402
    from ssp_decide import load_config, decide                  # noqa: E402
    _cfg = load_config(_HERE / "ssp_config.json")
    _f = list(_cfg["line_b"].get("flags", []))
    def _flag(n, d):
        return _f[_f.index(n) + 1] if n in _f and _f.index(n) + 1 < len(_f) else d
    b_pad, b_roi_top = float(_flag("--amount-pad", "0")), float(_flag("--roi-top", "0.6"))
    sb = (ort.InferenceSession(str(args.b_onnx), providers=["CPUExecutionProvider"])
          if args.b_onnx else None)

    npatch = (args.trainsize // args.patch_size) ** 2
    print(f"生成合成测试图 -> {args.out / 'images'}", flush=True)
    imgs = make_images(args.out / "images")
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    cases = []
    for p in imgs:
        raw = p.read_bytes()
        from PIL import Image
        rgb = np.asarray(Image.open(p).convert("RGB"))
        h, w = rgb.shape[:2]
        seed = zlib.crc32(raw) & 0x7FFFFFFF

        # 台阶 1: 发生器本身
        s, first8 = (seed & 0xFFFFFFFF) or 1, []
        for _ in range(8):
            s = xorshift32(s); first8.append(s)

        # 台阶 2: 第 0 轮的 64 个坐标
        rounds = select_positions(h, w, seed, args.patch_size, npatch, args.repeat)

        # 台阶 3~5
        patches, chosen = select_patches(rgb, seed, args.patch_size, npatch, args.repeat)
        scores = sess.run(["ai_score"], {"patches": patches})[0]

        cases.append({
            "image": p.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "width": int(w), "height": int(h),
            "stage1_seed": int(seed),
            "stage1_xorshift_first8": [int(v) for v in first8],
            "stage2_round0_positions": [[int(y), int(x)] for y, x in rounds[0]],
            "stage3_chosen_positions": [[int(y), int(x)] for y, x in chosen],
            "stage3_chosen_energies": [int(energy(q)) for q in patches],
            "stage4_patch_scores": [float(v) for v in scores],
            "stage5_ai_score": float(scores.mean()),
        })

        # ---- 线路B 的台阶(6~11)。这一段必须和 predict_tiled 逐行一致 ----
        if sb is not None:
            loc, page = locate_amount_auto(rgb)
            arr, cols, rows = rgb, 3, 6
            crop = [0, 0, int(w), int(h)]
            if loc:
                bx0, by0, bx1, by1 = loc[0], loc[1], loc[2], loc[3]
                pad = int(max(8, (by1 - by0) * b_pad))      # ★ amount_pad=0 -> pad=8
                cx0, cy0 = max(0, bx0 - pad), max(0, by0 - pad)
                cx1, cy1 = min(w, bx1 + pad), min(h, by1 + pad)
                arr, crop, (cols, rows) = rgb[cy0:cy1, cx0:cx1], [cx0, cy0, cx1, cy1], (2, 2)
            elif 0 < b_roi_top < 1.0:
                cy1 = max(32, int(h * b_roi_top))
                arr, crop = rgb[:cy1], [0, 0, int(w), cy1]
            tiles_xy = _tiles(arr.shape[1], arr.shape[0], cols, rows, 0.15)
            tp = [np.asarray(_richest_patch(arr[ty0:ty1, tx0:tx1], args.patch_size), np.uint8)
                  for (tx0, ty0, tx1, ty1) in tiles_xy]
            ts = sb.run(["ai_score"], {"patches": np.stack(tp)})[0]
            cases[-1].update({
                "stage6_page": page,
                "stage7_locate_box": None if not loc else [int(v) for v in loc[:4]],
                "stage8_crop_box_after_pad": [int(v) for v in crop],
                "stage8_tile_grid": [cols, rows],
                "stage9_tiles": [[int(v) for v in t] for t in tiles_xy],
                "stage10_tile_scores": [float(v) for v in ts],
                "stage11_tile_top3": float(np.sort(ts)[-3:].mean()),
            })
            print(f"  {p.name:24s} {w}x{h}  seed={seed}  A={scores.mean():.6f}  "
                  f"B={np.sort(ts)[-3:].mean():.6f} ({page}, "
                  f"{'定位' if loc else '未定位'}, {cols}x{rows}, {len(tiles_xy)} 块)", flush=True)
        else:
            print(f"  {p.name:24s} {w}x{h}  seed={seed}  A={scores.mean():.6f}  "
                  f"(没给 --b-onnx, 线路B 台阶未生成)", flush=True)

    algo = {
        "prng": "xorshift32: s^=s<<13; s^=s>>17; s^=s<<5  (全程 uint32)",
        "seed": "crc32(整个文件字节) & 0x7FFFFFFF; 若为 0 则取 1",
        "order": "每块先取 y = next() % (H-P+1), 再取 x = next() % (W-P+1)",
        "select": "每轮 64 块取纹理能量**最大**的; **并列取最后出现的那个**",
        "energy": "水平+垂直+两条对角线的相邻像素绝对差之和, int64",
        "onnx_input": "uint8 (N,32,32,3) NHWC RGB; 归一化/放大已包在图里",
        "aggregate": "16 个 ai_score 求平均 = final_ai_score",
        "patch_size": args.patch_size, "num_patch": npatch, "repeat": args.repeat,
    }
    tol = {"stage1_2_3": "必须完全相等(整数)", "stage4_5": 1e-4}

    # ★ 线路B 的规则也写进 json —— 对方要是只对着这个文件做, 不该还得回头翻文档。
    #   之前 tolerance 只有 stage1_2_3 / stage4_5 两条, 台阶 6~11 一条容差都没有,
    #   只对 json 的人根本不知道 stage10/11 该按 1e-4 收。
    if sb is not None:
        algo["line_b_locate"] = "先判版式(white/blue), 再找金额行: 同一行需 >=2 个连通域, 取中位高度最大的那行"
        algo["line_b_pad"] = (f"裁剪框外扩 pad = max(8, 金额框高 * amount_pad); "
                              f"当前 amount_pad={b_pad}, 所以 pad 恒为 8")
        algo["line_b_grid"] = "定位成功 -> 裁剪框切 2x2 共 4 块; 定位失败 -> 上 roi_top 部分切 3x6 共 18 块"
        algo["line_b_roi_top"] = b_roi_top
        algo["line_b_tile_overlap"] = 0.15
        # ★★ 并列规则和线路A **相反**, 别写成"规则同线路A"(2026-08-21 修正: 原文就是这么写错的)
        algo["line_b_patch"] = ("每一块内部再取纹理能量最大的一个 patch 送模型。"
                                "用积分图一次算完所有窗口, 取 argmax。"
                                "★ 并列时取**第一个**(行优先扫描的第一个), "
                                "这与线路A 的'并列取最后一个'**正好相反**, 不要照搬")
        algo["line_b_patch_integral"] = ("积分图必须用 float64 或 int64 累加, **不能用 float32** —— "
                                         "真实尺寸的块积分和会超过 2^24, float32 精确表示整数到此为止")
        algo["line_b_upscale"] = ("块任一边 < 32 时先放大: s = max(32/h, 32/w), "
                                  "目标尺寸 = (max(32, int(w*s+0.5)), max(32, int(h*s+0.5))), "
                                  "双线性、半像素中心对齐(OpenCV INTER_LINEAR), 结果可以不是正方形")
        algo["line_b_aggregate"] = "所有块的分数取最高 3 个求平均 = tile_top3"
        tol["stage6_7_8_9"] = "必须完全相等(版式字符串 + 整数坐标 + 网格)"
        tol["stage10_11"] = 1e-4

    # ---- 台阶 12: 判定层的真值表(**不需要图**) ----
    # 为什么必须单独做: 那 7 张合成图**走不到**判定层的大多数分支 ——
    # 它们没有 AIGC 元数据(线路C 永远是空的), 而且**全是 .png**(免自动拒永远不触发)。
    # 也就是说向量到 stage11 为止, **判定规则本身一条都没被验过**。
    # 期望值一律**当场调用 decide() 算出来**, 不手写 —— 手写等于再引入一个错误来源。
    la_, lb_ = _cfg["line_a"], _cfg["line_b"]
    A_S, A_R, B_S, B_R = la_["strict"], la_["review"], lb_["strict"], lb_["review"]
    _specs = [
        ("两条线都很低",                        0.10, 0.10, True,  ".png",  []),
        ("A 刚好压在严格线上",                  A_S,  0.10, True,  ".png",  []),
        ("A 差一点点到严格线",                  A_S - 1e-4, 0.10, True, ".png", []),
        ("A 刚好压在复核线上",                  A_R,  0.10, True,  ".png",  []),
        ("A 差一点点到复核线",                  A_R - 1e-4, 0.10, True, ".png", []),
        ("A 远高于严格线",                      0.95, 0.10, True,  ".png",  []),
        ("B 定位到且高过严格线",                0.10, 0.99, True,  ".png",  []),
        ("★ B 分极高但**没定位到**(应当无意见)", 0.10, 0.9999, False, ".png", []),
        ("B 定位到, 落在复核与严格之间",         0.10, 0.90, True,  ".png",  []),
        ("★ B 没定位到, 分落在复核区",          0.10, 0.90, False, ".png",  []),
        ("A 和 B 都高过严格线",                 0.95, 0.99, True,  ".png",  []),
        ("★ A 高过严格线, 但扩展名是 .jpeg",     0.95, 0.10, True,  ".jpeg", []),
        ("★ B 高过严格线, 但扩展名是 .jpeg",     0.10, 0.99, True,  ".jpeg", []),
        ("★ .jpg 不等于 .jpeg, 照常自动拒",      0.95, 0.10, True,  ".jpg",  []),
        ("★ 线路C 铁证 + 两条线分数都很低",       0.10, 0.10, True,  ".png",  ["C2PA生成"]),
        ("★★ 线路C 铁证 + .jpeg(免拒管不到线路C)", 0.10, 0.10, True, ".jpeg", ["C2PA生成"]),
        ("线路C 多个铁证同时命中",               0.10, 0.10, True,  ".png",  ["C2PA生成", "TC260国标"]),
        ("两条线都没有分数(打分失败)",           None, None, False, ".png",  []),
        ("只有 A 有分, B 没分",                 0.95, None, False, ".png",  []),
        ("只有 B 有分且定位到, A 没分",          None, 0.99, True,  ".png",  []),
    ]
    dcases = []
    for _nm, _a, _b, _loc, _ext, _ch in _specs:
        d_, why_ = decide((float(_a), True) if _a is not None else None,
                          (float(_b), bool(_loc)) if _b is not None else None,
                          _cfg, _ext, set(_ch) if _ch else None)
        dcases.append({
            "_名称": _nm,
            "line_a_score": None if _a is None else round(float(_a), 6),
            "line_b_score": None if _b is None else round(float(_b), 6),
            "line_b_located": bool(_loc),
            "ext": _ext,
            "line_c_hard": sorted(_ch),
            "expected_decision": d_,
            "expected_reason": why_,
        })
    tol["stage12_decision"] = "字符串, 必须完全相等"

    doc = {
        "_说明": ("线路A + 线路B 分阶段一致性向量。逐级核对: 哪一级先对不上, 问题就在那一级。"
                  "图是程序生成的合成图, 不是真实收据。"
                  "★ decision_cases 是判定层的真值表, 不需要图, 单独对。"
                  + ("" if sb is not None else
                     " ★本次没传 --b-onnx, 所以只有台阶 1~5, 线路B 那 6 级没有生成。")),
        "algorithm": algo,
        "thresholds": {"line_a_strict": A_S, "line_a_review": A_R,
                       "line_b_strict": B_S, "line_b_review": B_R,
                       "never_auto_reject_ext": _cfg.get("never_auto_reject_ext", [])},
        "tolerance": tol,
        "cases": cases,
        "decision_cases": dcases,
    }
    vp = args.out / "vectors.json"
    vp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {vp}")
    print(f"-> 图 {len(imgs)} 张在 {args.out / 'images'}")
    print("\n给对方的用法: 先对台阶 1(纯整数, 不用图), 再 2, 再 3 —— 前三级都是整数, 必须**完全相等**;")
    print("台阶 4/5 是浮点, 容差 1e-4(跨设备本身就有 1e-4 量级差异)。")
    if sb is not None:
        print("线路B 接着对台阶 6~9(版式/定位框/裁剪框/切块坐标), 也全是**完全相等**;")
        print("台阶 10/11 是浮点, 同样 1e-4。★ 台阶 8 最要紧: 验 pad 是不是 8, 网格是不是 2x2。")
    else:
        print("★ 这次没传 --b-onnx, 只出了台阶 1~5。要覆盖线路B 必须带上 --b-onnx 重跑。")
    # ★ 判定层是**不用图**的, 所以无论传没传 --b-onnx 都会生成 —— 但之前不打印,
    #   操作的人得再跑一条 python -c 才知道有没有出来。这里直接报出来。
    print(f"台阶 12 判定层真值表 {len(dcases)} 条(不用图), 字段 decision_cases, 判定字符串必须完全相等。")
    print(f"tolerance 一共 {len(tol)} 项: {' '.join(sorted(tol))}")


if __name__ == "__main__":
    main()
