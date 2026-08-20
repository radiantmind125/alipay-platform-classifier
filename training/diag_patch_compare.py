r"""诊断: **真正的** `patch_img` 和我写的 `select_patches` 到底选了什么(只读源图)。

为什么要这个
------------
换成 lcg 取块之后, 597 张图的线路A 分数**中位涨了 +0.1615, 97.7% 的图都涨**。
这不是"换了一次抽样"该有的样子(那应该正负均衡), 是**系统性偏差** —— 我的实现有问题。

而我之前那次"头对头核对"是**拿我的实现去比我自己写的原版模拟** ——
如果我对原版的理解本身就是错的, 两边会一起错, 那个比对什么也证明不了。**这是循环论证。**

所以这里直接 import SSP 仓库里**真正的** `patch_img`, 和我的实现选出来的块逐个比:
坐标、能量、是不是纯色、平均色。**哪一步开始分叉, 一眼就能看出来。**

用法
----
  python training/diag_patch_compare.py --ssp-repo E:\SSP_Work\SSP ^
      --input E:\SSP_Work\probe\speedtest --n 6
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="对比真正的 patch_img 与 select_patches(只读源图)")
    ap.add_argument("--ssp-repo", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=16)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--trainsize", type=int, default=256)
    ap.add_argument("--onnx", type=Path, default=None,
                    help="给了就把两边的块都过一遍模型, 直接看**分数**差多少 —— "
                         "光看块的性质还隔一层, 分数才是最终要对的东西")
    args = ap.parse_args()

    import importlib.util
    miss = [m for m in ("torch", "torchvision") if not importlib.util.find_spec(m)]
    if miss:
        raise SystemExit(f"缺 {miss} —— 真正的 patch_img 要靠 torchvision")

    # ★ 每一步都要吱声。这个脚本 import torch + 载入 94MB ONNX + 每张图跑 1024 次
    #   原版取块, 静默期能有半分钟 —— 而"静默"和"卡死"从外面是分不出来的。
    #   这条已经踩过四次了(--n 10 的进度、线路C、模型路径、还有这次)。
    print("加载 torch 和 SSP 的取块模块...", flush=True)
    import torch                                            # noqa: E402
    from PIL import Image                                   # noqa: E402
    sys.path.insert(0, str(args.ssp_repo))
    from utils.patch import patch_img, compute              # noqa: E402
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from patch_select import select_patches, energy         # noqa: E402

    sess = None
    if args.onnx:
        print(f"加载 ONNX {args.onnx.name} (94MB, 要几秒)...", flush=True)
        import onnxruntime as ort                           # noqa: E402
        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    def score(arrs) -> float:
        if sess is None:
            return float("nan")
        x = np.stack(arrs).astype(np.uint8)
        return float(sess.run(["ai_score"], {"patches": x})[0].mean())

    imgs = [p for p in sorted(args.input.rglob("*")) if p.suffix.lower() in _EXTS][:args.n]
    if not imgs:
        raise SystemExit(f"{args.input} 底下没找到图")

    def describe(arrs: list[np.ndarray]) -> str:
        es = [energy(a) for a in arrs]
        uni = sum(1 for a in arrs if a.min() == a.max())        # 三通道全同一个值
        cols = {tuple(int(v) for v in a.reshape(-1, 3).mean(0).round()) for a in arrs}
        return (f"能量 中位{sorted(es)[len(es)//2]:>8,d} 最小{min(es):>7,d} | "
                f"纯色块 {uni:>2d}/{len(arrs)} | 不同均色 {len(cols):>2d}")

    n_draw = args.repeat * (args.trainsize // args.patch_size) ** 2
    print(f"\n共 {len(imgs)} 张, 每张要跑 {n_draw:,} 次原版取块, 慢是正常的\n", flush=True)
    print(f"{'图':40s}  来源")
    print("-" * 104, flush=True)
    for i, p in enumerate(imgs, 1):
        print("  第 %d/%d 张 处理中..." % (i, len(imgs)), flush=True)
        img = Image.open(p).convert("RGB")
        seed = zlib.crc32(p.read_bytes()) & 0x7FFFFFFF

        # ---- 真正的 patch_img(和线上完全一样: 先按内容定种子, 再连叫 repeat 次) ----
        torch.manual_seed(seed)
        real = [np.asarray(patch_img(img, args.patch_size, args.trainsize), dtype=np.uint8)
                for _ in range(args.repeat)]

        # ---- 我的实现 ----
        mine, _ = select_patches(np.asarray(img), seed,
                                 patch_size=args.patch_size,
                                 num_patch=(args.trainsize // args.patch_size) ** 2,
                                 repeat=args.repeat)
        mine = [m for m in mine]

        sr, sm = score(real), score(mine)
        tail = lambda v: "" if v != v else f" | **分数 {v:.4f}**"
        print(f"{p.name[:38]:40s}  真正的 patch_img : {describe(real)}{tail(sr)}", flush=True)
        print(f"{'':40s}  我的 select_patch : {describe(mine)}{tail(sm)}")
        # 逐块比能量: 若我的**总是更小**, 说明我选得"更简单", 那分歧就在取块本身
        lower = sum(1 for a, b in zip(mine, real) if energy(a) < energy(b))
        higher = sum(1 for a, b in zip(mine, real) if energy(a) > energy(b))
        print(f"{'':40s}  -> 我选的更简单 {lower}/{args.repeat} 块, 更复杂 {higher}/{args.repeat} 块")

        # 用原版 compute 复核我的能量函数 —— 排除"我的能量算错"这条可能
        bad = sum(1 for a in mine if compute(Image.fromarray(a)) != energy(a))
        if bad:
            print(f"{'':40s}  !!!! 我的能量函数和原版 compute 对不上 {bad}/{len(mine)} 块")

    print("\n判读:")
    print("  两边'纯色块数'差不多  -> 取块没分叉, 问题在别处(喂给模型那一段)")
    print("  我的纯色块**明显更多** -> 我的取块确实更偏向纯色, 分歧就在取块")
    print("  '我选的更简单'占压倒多数 -> 同上; 但要注意**原版也是取最小能量**,")
    print("     所以正常情况下两边应当**互有高低**, 一边倒就说明有一边没在取最小值")
