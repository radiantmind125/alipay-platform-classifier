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
    ap.add_argument("--same-device", action="store_true",
                    help="CSV 里的分数**和这次一样是 CPU 算的**。加上它就用严格判据(全体都比), "
                         "因为跨设备那层干扰不存在了 —— 高分图反而是最灵敏的样本。"
                         "不加则只敢看低分图(跨设备的差在低分区会自动消失)。")
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

    # ★ 判读要看**中位数**, 不是最大值 —— 这一点第一版写错了, 差点把好的结果判成坏的。
    #   预处理若写错(通道顺序 / 没除 255 / 常数不对), 是**每一张都错**, 中位数会很大。
    #   而"CSV 在 GPU 上算、这次在 CPU 上算"只会让**少数**图差到 1e-3 量级,
    #   绝大多数(分数接近 0 的那些)照样完全对得上, 中位数仍然是 1e-6。
    #   实测跨设备的分布: 中位 2.76e-04 / p95 2.35e-03 / 最大 1.20e-02(858 张高分图)。
    # ★ 更稳的判据: **只看低分图**。
    #   分数接近 0 时 sigmoid 已经饱和, logit 差一点点在分数上几乎看不出来 ->
    #   **跨设备的差在低分区会自动消失**。而预处理写错是改了模型的输入,
    #   logit 会整体偏掉, **低分区照样看得见**。
    #   (第一版只看全体中位数, 若样本恰好全是高分图, 跨设备也会被误判成预处理出错 ——
    #    实测跨设备在高分图上的中位是 2.76e-04, 远超 1e-5。)
    low = [d for nm, a, b, d in rows if a < 0.05]
    med_low = sorted(low)[len(low) // 2] if low else None
    print(f"  其中低分图(CSV<0.05) {len(low)} 张: 中位差 "
          f"{med_low:.2e}" if low else "  (样本里没有低分图, 判据受限)")
    med = diffs_sorted[len(diffs) // 2]
    if args.same_device:
        # 同设备就没有跨设备那层干扰, **全体都能拿来判**, 而且高分图最灵敏。
        # 门槛放在 1e-4: ONNX 自身的浮点差约 1e-6, 留两个数量级余量。
        print(f"\n  (--same-device: CSV 与本次同为 CPU, 用严格判据, 全体 {len(diffs)} 张都算)")
        if worst < 1e-4:
            print(f"  -> **对得上**。最大差 {worst:.2e}, 其中高分图也在内 —— "
                  f"ONNX 可以原样替掉 torch 那一半。")
        else:
            print(f"  !!!! **最大差 {worst:.2e} 偏大。** 同设备下不该有这么多, 查预处理。")
    elif med_low is None:
        print("\n  !!!! 这批样本全是高分图, 而 CSV 可能是别的设备算的, **判不了**。")
        print("       两条路: 加 --same-device(确认 CSV 也是 CPU 算的), 或换一批含低分图的样本。")
    elif med_low < 1e-5:
        print(f"\n  -> **预处理是对的**。低分区中位差 {med_low:.2e}(全体中位 {med:.2e}),")
        print(f"     与 ONNX 对 torch 的浮点差同量级 —— 说明输入喂对了。")
        if worst > 1e-4:
            print(f"\n     尾部到 {worst:.2e} —— **几乎肯定是 CSV 与本次不在同一个设备上算的**")
            print(f"     (CSV 若是 --device cuda 跑的, 这次 ONNX 走 CPU, 实测跨设备最大可到 1.2e-02)。")
            print(f"     ★ 要坐实, 拿一份**同设备(CPU)算的 CSV** 再比一次, 尾部应当塌到 1e-6。")
        print(f"\n     操作影响: 严格线 0.8031 上下相邻分数相差 0.02~0.03, 差 {worst:.0e} 只能撼动")
        print(f"     恰好贴线 {worst:.0e} 以内的图; 实测跨设备在 858 张边界图里只翻了 **2 张**。")
    else:
        print(f"\n  !!!! **低分区中位差 {med_low:.2e} 太大 —— 预处理确实有出入, 先别往下做。**")
        print(f"       (低分区跨设备的差会自动消失, 所以这里大就只能是输入喂错了。) 常见原因:")
        print(f"       通道顺序(RGB/BGR) / 有没有除 255 / 归一化常数 / --repeat 与当初不一致")
        print(f"       也可能是这批图的分数不是用 predict_seeded 算的(种子规则对不上)。")

    print("\n★ 提醒: 这里验的仍然只是**模型那一半**。取块还在 Python 里 ——")
    print("  .NET 要自己实现取块的话, 那是另一个独立风险, 见 export_onnx 的说明。")


if __name__ == "__main__":
    main()
