r"""生成**分阶段**一致性测试向量, 给 .NET 那边逐级核对(只写 --out, 不碰任何真实图)。

为什么必须分阶段
----------------
只给"最后那个分数"是不够的。对不上的时候, 只知道"结果不一样",
不知道是**随机数不对**、**坐标算错**、**选块规则反了**, 还是**喂给模型的数据不对** ——
只能从头猜。这条线上已经为这种猜法付过一次代价了。

所以这里按**五级台阶**给, 每一级都能单独对:

    1. 发生器      给定种子, xorshift32 的前 8 个输出
    2. 坐标        第 0 轮的全部 64 个 (y, x)
    3. 选块        16 轮各自选中的 (y, x) 和它的纹理能量
    4. 模型        16 个块各自的 ai_score
    5. 汇总        16 个分数的平均 = final_ai_score

**哪一级先对不上, 问题就在那一级。**

★ 测试图是**程序生成的合成图**, 不是真实收据 ——
  真实收据不能给外部, 而合成图两边都能拿到一模一样的字节, 反而更适合做一致性核对。

用法
----
  python training/make_conformance.py --onnx E:\SSP_Work\onnx\aigen_v7.onnx ^
      --out E:\SSP_Work\onnx\conformance

生成 `images/*.png` 和 `lineA_vectors.json`, 两样一起给对方。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


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
    ]
    paths = []
    for name, w, h, kind in specs:
        if kind == "blue":
            a = np.zeros((h, w, 3), np.uint8)
            for y in range(h):                      # 竖直渐变, 模仿蓝底转账页
                a[y] = (30 + y * 40 // h, 90 + y * 60 // h, 200 + y * 40 // h)
        else:
            a = np.full((h, w, 3), 248 if kind != "flat" else 255, np.uint8)
        if kind != "flat":
            nrow = 22 if kind == "dense" else 9
            for i in range(nrow):                   # 文字条: 高能量区
                y0 = int(h * (0.08 + 0.82 * i / nrow))
                x0, x1 = int(w * 0.10), int(w * (0.55 + 0.35 * ((i * 7) % 5) / 5))
                a[y0:y0 + max(6, h // 90), x0:x1] = rng.integers(0, 70, (1, 1, 3), dtype=np.uint8)
        a[int(h * 0.30):int(h * 0.36), int(w * 0.25):int(w * 0.75)] = \
            rng.integers(0, 40, (1, 1, 3), dtype=np.uint8)      # 金额那一行
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
    ap.add_argument("--onnx", type=Path, required=True)
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
        print(f"  {p.name:24s} {w}x{h}  seed={seed}  ai_score={scores.mean():.6f}", flush=True)

    doc = {
        "_说明": "线路A 分阶段一致性向量。逐级核对: 哪一级先对不上, 问题就在那一级。"
                 "图是程序生成的合成图, 不是真实收据。",
        "algorithm": {
            "prng": "xorshift32: s^=s<<13; s^=s>>17; s^=s<<5  (全程 uint32)",
            "seed": "crc32(整个文件字节) & 0x7FFFFFFF; 若为 0 则取 1",
            "order": "每块先取 y = next() % (H-P+1), 再取 x = next() % (W-P+1)",
            "select": "每轮 64 块取纹理能量**最大**的; **并列取最后出现的那个**",
            "energy": "水平+垂直+两条对角线的相邻像素绝对差之和, int64",
            "onnx_input": "uint8 (N,32,32,3) NHWC RGB; 归一化/放大已包在图里",
            "aggregate": "16 个 ai_score 求平均 = final_ai_score",
            "patch_size": args.patch_size, "num_patch": npatch, "repeat": args.repeat,
        },
        "tolerance": {"stage1_2_3": "必须完全相等(整数)", "stage4_5": 1e-4},
        "cases": cases,
    }
    vp = args.out / "lineA_vectors.json"
    vp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {vp}")
    print(f"-> 图 {len(imgs)} 张在 {args.out / 'images'}")
    print("\n给对方的用法: 先对台阶 1(纯整数, 不用图), 再 2, 再 3 —— 前三级都是整数, 必须**完全相等**;")
    print("台阶 4/5 是浮点, 容差 1e-4(跨设备本身就有 1e-4 量级差异)。")


if __name__ == "__main__":
    main()
