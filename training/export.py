"""把训练好的 checkpoint 导出为 ONNX（默认 FP32），可选 INT8 动态量化。

说明（依据评审）：模型本身很小，CPU 推理耗时主要在 JPEG 解码而非前向；INT8 对这么小的
模型收益有限，且可能在“稀疏安卓栏 / 缩放 iPhone”这类微妙边界上移动决策面。所以默认导出
FP32 ONNX；只有在真实 CPU 机器上做过延迟评测确实需要时才用 INT8，且量化后必须在硬骨头
子集 + 金标上重新验证，不能只看 FP32。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from model import build_model
from preprocess import CANVAS_H, CANVAS_W


class _WithSoftmax(nn.Module):
    """在图内接一个 Softmax：ML.NET 端直接读概率 prob[1]=P(苹果)，省掉外部再算一步。"""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core
        self.sm = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sm(self.core(x))


def export_onnx(checkpoint: Path, out: Path, *, opset: int = 17, softmax: bool = False) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(num_classes=len(payload.get("classes", ["android", "ios"])), width=float(payload.get("width", 1.0)))
    model.load_state_dict(payload["model_state"])
    model.eval()
    export_model: nn.Module = _WithSoftmax(model) if softmax else model
    out_name = "prob" if softmax else "logits"
    dummy = torch.zeros(1, 3, CANVAS_H, CANVAS_W)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        opset_version=opset, input_names=["strip"], output_names=[out_name],
        dynamic_axes={"strip": {0: "batch"}, out_name: {0: "batch"}},
    )
    try:
        # 旧版 TorchScript 导出器：图更干净、算子最基础，ML.NET / onnxruntime 兼容性最好
        torch.onnx.export(export_model, dummy, out.as_posix(), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(export_model, dummy, out.as_posix(), **kwargs)  # 老版本 torch 没有 dynamo 参数
    print(f"ONNX 已导出：{out}  (classes={payload.get('classes')}, 输入 strip=Bx3x{CANVAS_H}x{CANVAS_W}, 输出 {out_name}, opset={opset}, softmax={softmax})")


def quantize_int8(onnx_fp32: Path, out: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as e:  # noqa: BLE001
        raise SystemExit("需要 onnxruntime：pip install onnxruntime") from e
    quantize_dynamic(onnx_fp32.as_posix(), out.as_posix(), weight_type=QuantType.QInt8)
    print(f"INT8 ONNX 已导出：{out}（务必在硬骨头子集+金标上重新验证后再上线）")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("training/runs/statusbar_v1/model_fp32.onnx"))
    ap.add_argument("--int8", action="store_true", help="额外导出 INT8 动态量化（默认不导出）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--mlnet", action="store_true", help="ML.NET 友好：图内含 Softmax(输出概率 prob) + opset 13")
    args = ap.parse_args(argv)
    opset = 13 if args.mlnet else args.opset
    export_onnx(args.checkpoint, args.out, opset=opset, softmax=args.mlnet)
    if args.int8:
        quantize_int8(args.out, args.out.with_name(args.out.stem + "_int8.onnx"))


if __name__ == "__main__":
    main()
