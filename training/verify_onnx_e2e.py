r"""端到端核对: ONNX 那条路算出来的分, 和**产物里已有的分**对不对得上(只读源图)。

为什么单有 export_onnx 的核对还不够
------------------------------------
`export_onnx.py` 验的是"**ONNX 忠实于我写的那个包装层**" ——
同一批随机小块喂进 torch 包装层和 onnxruntime, 输出差 1e-6。这没问题。

**但它没有验"那个包装层忠实于真正在跑的打分脚本"。** 两件事不一样:

    包装层里的归一化是我照着 torchvision 的 ToTensor + Normalize **手写的**;
    真正的打分走的是 torchvision 本身。两者**应该**一样, 但"应该"不是"验过"。

这个脚本把这一环补上: 用**和线上完全一样的取块方式**(直接 import SSP 的 `patch_img`,
种子也按 `predict_seeded` 的规则来), 把小块喂给 ONNX, 求平均, 再和
`summary.csv` 里已经算好的 `final_ai_score` 逐张比。

对得上 -> ONNX 可以原样替掉 torch 那一半, 阈值继续有效。
对不上 -> 预处理有出入, **别往下做**。

★ 注意这里验的仍然只是"模型那一半"。取块这一步还是 Python 在做,
  .NET 那边要自己实现取块的话, 是另一个独立的风险, 见 export_onnx 的说明。

用法
----
  python training/verify_onnx_e2e.py --onnx E:\SSP_Work\onnx\aigen_v7.onnx ^
      --csv E:\SSP_Work\probe\accept_seeded\_lineA\summary.csv ^
      --search-root D:\download2 --ssp-repo E:\SSP_Work\SSP --n 60
"""

from __future__ import annotations

import argparse
import csv
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
    ap = argparse.ArgumentParser(description="端到端核对 ONNX 与已有分数(只读源图)")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True, help="线路A summary.csv(已有分数)")
    ap.add_argument("--search-root", type=Path, required=True, help="原图在哪儿")
    ap.add_argument("--ssp-repo", type=Path, required=True)
    ap.add_argument("--col", default="final_ai_score")
    ap.add_argument("--n", type=int, default=60, help="抽多少张来比")
    ap.add_argument("--repeat", type=int, default=16, help="要和 predict_all_models 的 --repeat 一致")
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--trainsize", type=int, default=256)
    ap.add_argument("--seed-sample", type=int, default=0)
    args = ap.parse_args()

    import importlib.util
    miss = [m for m in ("onnxruntime", "torch", "torchvision") if not importlib.util.find_spec(m)]
    if miss:
        raise SystemExit(f"缺 Python 包 {miss} —— 取块要 torchvision(SSP 的 patch_img 用它), "
                         f"核对要 onnxruntime")
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX 不存在: {args.onnx}")

    import onnxruntime as ort                                  # noqa: E402
    import torch                                               # noqa: E402
    from PIL import Image                                      # noqa: E402
    sys.path.insert(0, str(args.ssp_repo))
    from utils.patch import patch_img                          # noqa: E402

    # ---- 已有分数 ----
    want: dict[str, float] = {}
    with open(args.csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            nm = (r.get("image_name") or "").strip()
            v = (r.get(args.col) or "").strip()
            if not nm or not v:
                continue
            if "scores_json" in r and (r.get("scores_json") or "").strip() in ("", "{}"):
                continue                                       # 打分失败的行不参与
            try:
                want[nm] = float(v)
            except ValueError:
                continue
    if not want:
        raise SystemExit(f"{args.csv} 里没读到 {args.col}")

    print(f"扫描 {args.search_root} 建索引...", flush=True)
    idx: dict[str, Path] = {}
    for p in args.search_root.rglob("*"):
        if p.suffix.lower() in _EXTS and p.is_file():
            idx.setdefault(p.name, p)
    print(f"  {len(idx):,} 个文件名")

    import random
    names = sorted(n for n in want if n in idx)
    if not names:
        raise SystemExit("CSV 里的图一张都没在 --search-root 底下找到")
    random.Random(args.seed_sample).shuffle(names)
    names = names[:args.n]
    print(f"抽了 {len(names)} 张来比(CSV 有 {len(want):,} 张, 能找到原图的 "
          f"{sum(1 for n in want if n in idx):,} 张)\n")

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    diffs, rows = [], []
    for i, nm in enumerate(names, 1):
        p = idx[nm]
        # ★ 种子规则必须和 predict_seeded.py 一模一样, 否则取的块不同, 比出来的差是假的
        torch.manual_seed(zlib.crc32(p.read_bytes()) & 0x7FFFFFFF)
        img = Image.open(p).convert("RGB")
        patches = [np.asarray(patch_img(img, args.patch_size, args.trainsize), dtype=np.uint8)
                   for _ in range(args.repeat)]
        x = np.stack(patches)                                  # (repeat, 32, 32, 3) uint8
        got = float(sess.run(["ai_score"], {"patches": x})[0].mean())
        d = abs(got - want[nm])
        diffs.append(d)
        rows.append((nm, want[nm], got, d))
        if i % 20 == 0:
            print(f"  已比 {i}/{len(names)}", flush=True)

    diffs_sorted = sorted(diffs)
    worst = max(diffs)
    print(f"\n{'=' * 62}")
    print(f"  比了 {len(diffs)} 张 | 差值 中位 {diffs_sorted[len(diffs) // 2]:.2e} | "
          f"p95 {diffs_sorted[int(len(diffs) * .95)]:.2e} | **最大 {worst:.2e}**")
    for nm, a, b, d in sorted(rows, key=lambda t: -t[3])[:5]:
        print(f"    {nm[:44]:46s} CSV {a:.6f}  ONNX {b:.6f}  差 {d:.2e}")

    if worst < 1e-4:
        print(f"\n  -> **对得上**。ONNX 可以原样替掉 torch 那一半, 阈值继续有效。")
        print(f"     (参照: 严格线 0.8031 与它上下相邻分数相差 0.02~0.03, 差 {worst:.0e} 动不了判定。)")
    else:
        print(f"\n  !!!! **对不上, 先别往下做。** 预处理有出入, 常见原因:")
        print(f"       通道顺序(RGB/BGR) / 有没有除 255 / 归一化常数 / --repeat 与当初不一致")
        print(f"       也可能是这批图的分数不是用 predict_seeded 算的(种子规则对不上)。")

    print("\n★ 提醒: 这里验的仍然只是**模型那一半**。取块还在 Python 里 ——")
    print("  .NET 要自己实现取块的话, 那是另一个独立风险, 见 export_onnx 的说明。")


if __name__ == "__main__":
    main()
