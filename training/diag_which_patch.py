r"""查清楚**服务器上真正在跑的** `patch_img` 到底是什么、在哪、干了什么(只读源图)。

为什么要这个
------------
实测: 真正的 `patch_img` 返回的块能量 **250,000~350,000, 0/16 是纯色**;
我按源码复刻的实现返回**能量 0 的纯色块**, 而且**六张图 16/16 全都比它更简单**。

可我读到的源码明明写着取最小:

    patch_list.sort(key=lambda x: compute(x), reverse=False)
    new_img = patch_list[0]

**两者必有一个不成立。** 可能是:
1. 服务器上的 `utils/patch.py` 和我读的那份**不是同一个版本**;
2. `sys.path` 上有**另一个 `utils` 包**把它顶掉了(同名遮蔽);
3. `RandomCrop` 的行为和我理解的不一样。

这个脚本把三条**一次全查掉**, 并且**直接做实验**: 自己取 64 个随机块算能量,
再看 `patch_img` 返回的那块**是不是这 64 个里最小的**。
—— 不管源码写的是什么, **行为**骗不了人。

用法
----
  python training/diag_which_patch.py --ssp-repo E:\SSP_Work\SSP ^
      --input E:\SSP_Work\probe\speedtest
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="查清真正在跑的 patch_img(只读源图)")
    ap.add_argument("--ssp-repo", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--trainsize", type=int, default=256)
    args = ap.parse_args()

    print("加载 torch / torchvision / SSP ...", flush=True)
    import torch                                            # noqa: E402
    import torchvision                                      # noqa: E402
    from torchvision import transforms                      # noqa: E402
    from PIL import Image                                   # noqa: E402
    sys.path.insert(0, str(args.ssp_repo))
    from utils import patch as patchmod                     # noqa: E402
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from patch_select import energy                         # noqa: E402

    print(f"\ntorch {torch.__version__} | torchvision {torchvision.__version__}")
    print(f"\n★ 真正加载的模块文件: **{patchmod.__file__}**")
    print(f"  (若不是 {args.ssp_repo}\\utils\\patch.py, 说明被**同名包遮蔽**了)")

    print("\n===== patch_img 源码 =====")
    print(inspect.getsource(patchmod.patch_img))
    print("===== compute 源码 =====")
    print(inspect.getsource(patchmod.compute))

    # ---- 行为实验: 源码怎么写不重要, 看它实际返回的是不是最小 ----
    imgs = [p for p in sorted(args.input.rglob("*")) if p.suffix.lower() in _EXTS][:3]
    if not imgs:
        raise SystemExit(f"{args.input} 底下没找到图")
    P, T = args.patch_size, args.trainsize
    n_draw = (T // P) ** 2
    print(f"===== 行为实验: patch_img 返回的块, 是不是它自己那 {n_draw} 个候选里最小的 =====")
    for p in imgs:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        # 自己按同样方式取 n_draw 个随机块, 看能量分布
        rp = transforms.RandomCrop(P)
        torch.manual_seed(20260820)
        cand = [np.asarray(rp(img), dtype=np.uint8) for _ in range(n_draw)]
        ce = sorted(energy(c) for c in cand)
        # 同一个种子再喂给 patch_img, 它应当在同样的候选里挑
        torch.manual_seed(20260820)
        got = energy(np.asarray(patchmod.patch_img(img, P, T), dtype=np.uint8))
        rank = sum(1 for e in ce if e < got)
        print(f"\n  {p.name[:46]}  {w}x{h}")
        print(f"    {n_draw} 个随机候选的能量: 最小 {ce[0]:,} | 中位 {ce[len(ce)//2]:,} | 最大 {ce[-1]:,}")
        print(f"    候选里能量为 0 的: {sum(1 for e in ce if e == 0)} 个")
        print(f"    **patch_img 实际返回的能量: {got:,}**  -> 排在第 {rank + 1}/{n_draw} 位")
        if rank == 0:
            print(f"    -> 取的是**最小**(和源码一致)")
        elif rank >= n_draw - 1:
            print(f"    -> 取的是**最大** —— 和源码写的相反!")
        else:
            print(f"    -> **既不是最小也不是最大**, 说明候选集和我这里取的不是同一批")

    print("\n判读:")
    print("  返回值排第 1 位  -> 确实取最小, 那分歧在别处(我的取块另有 bug)")
    print("  返回值排最后一位 -> 实际取的是**最大**, 我按源码复刻的方向就是反的")
    print("  排在中间        -> 它的候选集和我这里不一样(种子/取块方式对不上)")


if __name__ == "__main__":
    main()
