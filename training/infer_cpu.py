"""CPU 端推理（ONNX Runtime）：原图 -> 状态栏条 -> tiny-CNN -> 设备。只需 numpy/PIL/ORT，无需 torch。

这是上线端的 Tier-1：只对分辨率判不了（弃权）的图调用；分辨率能判的 ~79% 在 Tier-0 直接出结果。
先验用 logit 调整重定向（默认关闭），设备判定 = argmax；置信度低于阈值标 uncertain，
并与分辨率/EXIF 先验做一致性交叉校验（不一致 -> inconsistent，交欺诈评分）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import preprocess_original  # noqa: E402

CLASSES = ("android", "ios")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class DeviceClassifier:
    def __init__(self, onnx_path: str, *, prior_logit: float = 0.0, conf: float = 0.75) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.prior_logit = prior_logit   # 对 ios 这一类的 logit 调整（重定向先验），默认 0
        self.conf = conf

    def predict(self, original_rgb: np.ndarray) -> dict:
        x = preprocess_original(original_rgb)[None].astype(np.float32)
        logits = self.sess.run(None, {self.input_name: x})[0][0]
        logits = logits.copy()
        logits[1] += self.prior_logit
        p = _softmax(logits)
        p_ios = float(p[1])
        c = max(p_ios, 1 - p_ios)
        device = "uncertain" if c < self.conf else CLASSES[int(np.argmax(p))]
        return {"device": device, "p_ios": round(p_ios, 4), "conf": round(c, 3)}


def load_upright_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", type=Path, required=True, help="单张图或目录")
    ap.add_argument("--out", type=Path, default=Path("cpu_device_predictions.jsonl"))
    ap.add_argument("--prior-logit", type=float, default=0.0)
    ap.add_argument("--conf", type=float, default=0.75)
    args = ap.parse_args(argv)

    clf = DeviceClassifier(args.onnx, prior_logit=args.prior_logit, conf=args.conf)
    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    with args.out.open("w", encoding="utf-8") as f:
        for p in paths:
            try:
                r = clf.predict(load_upright_rgb(p))
                r["file"] = p.name
            except Exception as e:  # noqa: BLE001
                r = {"file": p.name, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(paths)} 张，结果写入 {args.out}")


if __name__ == "__main__":
    main()
