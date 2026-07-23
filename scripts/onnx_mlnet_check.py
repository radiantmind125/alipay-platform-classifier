r"""检验导出的 ONNX 是否 ML.NET 可用(Python 侧高置信预检)。

ML.NET 底层用的就是 onnxruntime——所以:能在 onnxruntime 加载+跑通、算子都是基础算子、
opset 不过高、输入输出名固定 → ML.NET 极大概率能加载。真正的 .NET/ML.NET 加载,请最后在
.NET 侧确认一次(本脚本不替代那一步,但把该看的都替你看了)。

打印:onnx.checker、opset、用到的算子、输入输出(名/类型/形状)、onnxruntime 跑一遍的输出;
给出"ML.NET 就绪"判断。

用法(带 onnx/onnxruntime 的 venv):
  D:\alipay-ai-data\alipay-ai-inference\.venv\Scripts\python.exe scripts\onnx_mlnet_check.py --onnx D:\download\statusbar_device.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ML.NET(onnxruntime)常见能跑的基础算子;超出这个集合的标出来供人工确认
_SAFE_OPS = {
    "Conv", "BatchNormalization", "Relu", "LeakyRelu", "Clip", "Add", "Mul", "Sub", "Div",
    "GlobalAveragePool", "AveragePool", "MaxPool", "ReduceMean", "Gemm", "MatMul",
    "Softmax", "Reshape", "Flatten", "Transpose", "Concat", "Constant", "Identity",
    "Squeeze", "Unsqueeze", "Shape", "Gather", "Slice", "Cast", "Pad", "Resize", "Sigmoid",
}


def _shape(value_info) -> list:
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return dims


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="ONNX 的 ML.NET 就绪预检")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--opset-warn", type=int, default=17, help="超过此 opset 就提示老版 ML.NET 可能不支持")
    args = ap.parse_args()

    try:
        import onnx
        from onnx import TensorProto
    except Exception as exc:  # noqa: BLE001
        print(f"缺 onnx:先 pip install -r requirements-export.txt ({exc})")
        return

    model = onnx.load(str(args.onnx))
    try:
        onnx.checker.check_model(model)
        checker = "通过"
    except Exception as exc:  # noqa: BLE001
        checker = f"失败:{exc}"

    opsets = {(o.domain or "ai.onnx"): o.version for o in model.opset_import}
    ops = sorted({n.op_type for n in model.graph.node})

    def dtype_name(elem: int) -> str:
        try:
            return TensorProto.DataType.Name(elem)
        except Exception:  # noqa: BLE001
            return str(elem)

    def io(vi) -> tuple:
        return (vi.name, dtype_name(vi.type.tensor_type.elem_type), _shape(vi))

    inputs = [io(i) for i in model.graph.input]
    outputs = [io(o) for o in model.graph.output]

    print(f"ONNX: {args.onnx}")
    print(f"  onnx.checker : {checker}")
    print(f"  opset        : {opsets}")
    print(f"  producer     : {model.producer_name} {model.producer_version}")
    print(f"  输入         : {inputs}")
    print(f"  输出         : {outputs}")
    print(f"  算子({len(ops)}): {', '.join(ops)}")

    # onnxruntime 跑一遍(ML.NET 底层就是它)
    ran = False
    run_info = "(未跑)"
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        x = np.zeros(shape, dtype=np.float32)
        outs = sess.run(None, {inp.name: x})
        run_info = "; ".join(
            f"{o.name}={list(np.asarray(v).shape)}(sum={float(np.sum(v)):.3f})"
            for o, v in zip(sess.get_outputs(), outs)
        )
        ran = True
    except Exception as exc:  # noqa: BLE001
        run_info = f"失败:{exc}"
    print(f"  onnxruntime 跑一遍: {run_info}")

    unsafe = [o for o in ops if o not in _SAFE_OPS]
    high_opset = {d: v for d, v in opsets.items() if v > args.opset_warn}

    print("\n== ML.NET 就绪判断 ==")
    if checker == "通过" and ran and not unsafe and not high_opset:
        print("  ✅ 极大概率可用:onnxruntime 加载+推理通过、算子全是基础算子、opset 不过高、输入输出名固定。")
        print(f"     ML.NET 侧:按输入名 {inputs[0][0]!r}(float32,形状 {inputs[0][2]})喂张量,读输出 {outputs[0][0]!r} 概率。")
        print("     最后在 .NET 里加载确认一次即可。")
    else:
        print("  ⚠ 有几点需要人工确认:")
        if checker != "通过":
            print(f"     - onnx.checker {checker}")
        if not ran:
            print(f"     - onnxruntime 跑不通:{run_info}")
        if unsafe:
            print(f"     - 非基础算子(ML.NET 需确认支持):{unsafe}")
        for domain, version in high_opset.items():
            print(f"     - opset {domain}={version} 偏高;老版 ML.NET 可能不支持,可用导出 --opset 13 重导一版")


if __name__ == "__main__":
    main()
