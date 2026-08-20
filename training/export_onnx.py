r"""把 SSPNet 导成 ONNX, 并**逐位核对** torch 与 onnxruntime 的输出是否一致。

为什么要把预处理也包进图里
--------------------------
经理那边要用 .NET 直接调 ONNX。**跨语言重写预处理是最容易悄悄出错的地方**:
通道顺序(RGB/BGR)、除不除 255、归一化常数、双线性放大的实现差异 ——
这几样每一项写错都**不会报错**, 只会让分数偏一点, 而阈值就是按分数定的。

所以这里把 **归一化 + 通道顺序 + 放大到 256** 全部包进 ONNX 图:

    输入: uint8, 形状 (N, 32, 32, 3), **RGB**, 就是原图上裁下来的小块
    输出: float32, 形状 (N,), 每块的 ai_score(越高越像 AI 生成)

于是 .NET 那边只剩一件事要做对: **裁哪些小块**(见下)。

★ 还剩的那一半: 取块
--------------------
线路A 的分数 = 取块 + 模型前向。**ONNX 只能带走模型那一半。**
取块现在用的是 PyTorch 的随机数发生器, .NET 不可能逐位复现 ——
若两边取的块不一样, 分数就不一样, 我们标的阈值也就不作数了。
(实测: 取块不同 -> 阈值附近标准差 0.0305, 判定不一致 46.9%。这不是舍入误差, 是量级差异。)

**所以 ONNX 这条路必须配一个"两边都能实现的取块算法"** —— 那是另一步的事,
这个脚本只负责把模型那一半做对, 并给出可核对的测试向量。

用法
----
  python training/export_onnx.py --weights E:\SSP_Work\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth ^
      --ssp-repo E:\SSP_Work\SSP --out E:\SSP_Work\onnx\aigen_v7.onnx

  # 线路B 同样导一份
  python training/export_onnx.py --weights ...\snapshot\localdet9\Net_epoch_best.pth ^
      --ssp-repo E:\SSP_Work\SSP --out E:\SSP_Work\onnx\localdet9.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def load_net(ssp_repo: Path, weights: Path):
    """从 SSP 仓库里**只取模型定义和加载函数**, 不整个 import。

    `predict_all_models.py` 顶上 `from torchvision import transforms`,
    而导出 ONNX **根本用不到 transforms**(归一化已经写进包装层了)。
    整个 import 会把 torchvision 变成硬依赖, 没装就直接跑不了 —— 没必要。
    """
    import ast
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    sys.path.insert(0, str(ssp_repo))
    from networks.resnet import resnet50                    # noqa: E402
    from networks.srm_conv import SRMConv2d_simple          # noqa: E402

    # utf-8-sig: SSP 那个文件开头带 BOM, 用 utf-8 读会在第一行留个 U+FEFF, ast.parse 直接报错
    src = (ssp_repo / "predict_all_models.py").read_text(encoding="utf-8-sig")
    want = {"SSPNoDownload", "clean_state_dict", "torch_load_compatible", "load_ssp_model"}
    keep = [n for n in ast.parse(src).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in want]
    got = {n.name for n in keep}
    if got != want:
        raise SystemExit(f"{ssp_repo}/predict_all_models.py 里没找到 {sorted(want - got)} —— "
                         f"SSP 仓库版本对不上, 先确认路径")
    ns = {"torch": torch, "nn": nn, "F": F, "resnet50": resnet50,
          "SRMConv2d_simple": SRMConv2d_simple, "Path": Path, "Dict": dict}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<pam>", "exec"), ns)
    return ns["load_ssp_model"](weights, torch.device("cpu"))


def build_wrapper(net, ai_label: int):
    """把预处理包进模型, 让 ONNX 的输入就是**原始 uint8 小块**。"""
    import torch
    import torch.nn as nn

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("m", torch.tensor(_MEAN).view(1, 3, 1, 1))
            self.register_buffer("s", torch.tensor(_STD).view(1, 3, 1, 1))
            self.ai_label = ai_label

        def forward(self, patches_uint8):        # (N, 32, 32, 3) uint8 RGB
            x = patches_uint8.to(torch.float32) / 255.0
            x = x.permute(0, 3, 1, 2)            # -> NCHW
            x = (x - self.m) / self.s
            logit = self.net(x).view(-1)         # net 内部 interpolate 到 256x256
            p = torch.sigmoid(logit)
            return p if self.ai_label == 1 else (1.0 - p)

    return Wrapped().eval()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="导出 SSPNet 到 ONNX 并核对与 torch 的一致性")
    ap.add_argument("--weights", type=Path, required=True, help=".pth 权重")
    ap.add_argument("--ssp-repo", type=Path, required=True, help="SSP 仓库(要 import networks)")
    ap.add_argument("--out", type=Path, required=True, help="输出 .onnx")
    ap.add_argument("--ai-label", type=int, default=0, choices=[0, 1],
                    help="与 predict_all_models 的 --ai_label 一致。默认 0, 即 ai_score = 1 - sigmoid")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--n-vectors", type=int, default=8, help="导出多少组测试向量供 .NET 逐位核对")
    args = ap.parse_args()

    # ★ 先把依赖和路径查一遍再动手 —— 这是 ssp_decide 的 preflight 一开始就在做的事,
    #   写这个脚本时却忘了照做, 结果是**加载完 94MB 权重、导出跑到一半才报"没装 onnx"**。
    #   缺什么要在第一秒一次性说全, 不要修一个再暴露下一个。
    import importlib
    missing = [m for m in ("onnx", "onnxruntime") if not importlib.util.find_spec(m)]
    bad = []
    if missing:
        bad.append(f"缺 Python 包 {missing} —— 装一下: pip install {' '.join(missing)}\n"
                   f"      (torch 的导出器要 onnx 才能写文件, 核对那一步要 onnxruntime)")
    if not args.weights.is_file():
        bad.append(f"权重文件不存在: {args.weights}")
    if not (args.ssp_repo / "predict_all_models.py").is_file():
        bad.append(f"{args.ssp_repo} 里没有 predict_all_models.py(--ssp-repo 指对了吗)")
    if bad:
        raise SystemExit("跑不了, 先解决这些:\n  " + "\n  ".join(bad))

    import torch                                            # noqa: E402
    import onnxruntime as ort                               # noqa: E402

    print(f"加载权重 {args.weights}")
    net = load_net(args.ssp_repo, args.weights)
    net.eval()
    wrapped = build_wrapper(net, args.ai_label)

    # ---- 导出 ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8)
    kw = dict(input_names=["patches"], output_names=["ai_score"],
              dynamic_axes={"patches": {0: "n"}, "ai_score": {0: "n"}},
              opset_version=args.opset, do_constant_folding=True)
    try:
        # dynamo=False 用的是老的 TorchScript 追踪导出器。对这种纯 CNN 它更稳,
        # 而且**不需要额外装 onnxscript**(torch 2.6 起新导出器成了默认, 会硬要那个包)。
        # dynamic_axes 也正是老导出器的参数, 两者配套。
        torch.onnx.export(wrapped, (dummy,), str(args.out), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(wrapped, (dummy,), str(args.out), **kw)   # 老版本没有 dynamo 这个参数
    print(f"-> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    # ---- 逐位核对: 同样的输入, torch 和 onnxruntime 必须给同样的输出 ----
    # **这一步不能省。** 导出成功不等于导对了 —— 算子替换、精度、align_corners
    # 任何一处不同都只会体现在小数点后几位, 不会报错。
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (1, 3, 16, 64):                    # 同时验**动态 batch** 是不是真的能用
        x = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        with torch.no_grad():
            t = wrapped(torch.from_numpy(x)).numpy()
        o = sess.run(["ai_score"], {"patches": x})[0]
        d = float(np.abs(t - o).max())
        worst = max(worst, d)
        print(f"  batch={n:>3d}  最大差 {d:.3e}   torch[0]={t[0]:.6f}  ort[0]={o[0]:.6f}")

    print(f"\n最大差 **{worst:.3e}**")
    if worst < 1e-5:
        print("  -> 一致(1e-5 以内)。**这个量级不会影响判定** —— 换设备本身就有 1e-4 量级差异,")
        print("     而实测跨设备 858 张里只有 2 张翻边。")
    else:
        print("  !!!! **差得太多, 不能用**。导出有问题, 先别往下走。")

    # ---- 测试向量: 给 .NET 那边逐位核对用 ----
    x = rng.integers(0, 256, (args.n_vectors, 32, 32, 3), dtype=np.uint8)
    y = sess.run(["ai_score"], {"patches": x})[0]
    vec = args.out.with_suffix(".vectors.json")
    vec.write_text(json.dumps({
        "_说明": "给 .NET 逐位核对用。patches 是 (N,32,32,3) uint8 RGB, 展平成一维; "
                 "喂进 ONNX 应当得到 expected_ai_score。**对不上就别接着往下做**, "
                 "多半是通道顺序或数据类型不对。",
        "shape": list(x.shape), "dtype": "uint8", "layout": "NHWC-RGB",
        "patches_flat": x.reshape(-1).tolist(),
        "expected_ai_score": [float(v) for v in y],
        "tolerance": 1e-5,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"-> 测试向量 {vec}  ({args.n_vectors} 组)")

    print("\n★ 还没解决的一半: **取块**。")
    print("  ONNX 只带走了模型, 取块还在 Python 里, 而且用的是 PyTorch 的随机数。")
    print("  .NET 自己写取块 -> 取到的块不一样 -> 分数不一样 -> 阈值不作数。")
    print("  下一步要把取块换成一个**两边都能一模一样实现**的算法, 然后重标一次阈值。")


if __name__ == "__main__":
    main()
