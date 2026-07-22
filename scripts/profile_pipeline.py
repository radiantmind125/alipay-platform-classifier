r"""流水线分速:定位瓶颈,回答"0.7s/张到底花在哪"。

做两件事:
  (1) 本脚本——只跑检测阶段(不带 OCR,单进程,便于在进程内给叶子函数套计时壳),
      给出 解码 / 矫正 / 检测 / 设备 各步的 ms/张 与占比;不改动生产代码,跑完即恢复。
  (2) OCR 占比——脚本末尾会提示:另跑两条 Measure-Command(--ocr none vs --ocr paddle),
      取墙钟差即得 OCR 阶段占总时间的比例(用的是生产的双进程路径,最准)。

必须用带 torch 的那个 venv 跑(alipay-ai-inference 的 .venv)。

用法(服务器,PowerShell 一行):
  D:\alipay-ai-data\alipay-ai-inference\.venv\Scripts\python.exe `
    <platform-classifier>\scripts\profile_pipeline.py `
    --hx-src D:\Hx.AI.py\src `
    --input D:\download\TempFakeImages --limit 200 `
    --platform-checkpoint D:\Hx.AI.py\checkpoints\statusbar_device_v1\best.pt `
    --device cuda
（检测权重默认自动取 receipt_lrcnn_v1;定位失败再显式加 --checkpoint ...\best.pt）
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="流水线分速:检测阶段各步 ms/张 + 占比")
    ap.add_argument("--hx-src", type=Path, required=True, help=r"Hx.AI.py 的 src 目录,如 D:\Hx.AI.py\src")
    ap.add_argument("--input", type=Path, required=True, help="图片目录")
    ap.add_argument("--limit", type=int, default=200, help="取多少张(默认200,够稳又快)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="检测权重;默认自动取 receipt_lrcnn_v1")
    ap.add_argument("--platform-checkpoint", type=Path, default=None, help="设备权重(可选,想一并测设备耗时就带上)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.hx_src))
    try:
        import transfer_receipt_ai.pipeline as pipeline_mod
        from transfer_receipt_ai.model import LRCNNPredictor
        from transfer_receipt_ai.infer import run_inference
        from receipt_inference.models import resolve_checkpoint
    except Exception as exc:  # noqa: BLE001
        print(f"导入失败(--hx-src 指向 Hx.AI.py\\src 了吗?用带 torch 的 venv 跑了吗?):{type(exc).__name__}: {exc}")
        raise

    checkpoint = args.checkpoint
    if checkpoint is None:
        try:
            checkpoint = resolve_checkpoint("receipt_lrcnn_v1", None)
        except Exception as exc:  # noqa: BLE001
            print(f"自动定位检测权重失败:{exc}")
            print(r"请显式传 --checkpoint D:\Hx.AI.py\checkpoints\receipt_lrcnn_v1\best.pt")
            return

    acc: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)

    def install(owner: object, name: str, key: str) -> None:
        """给 owner.name 套一层计时壳(不改源码,运行时替换)。"""
        try:
            original = getattr(owner, name)
        except AttributeError:
            print(f"  (跳过 {key}:未找到 {name})")
            return

        def timed(*a, **k):  # type: ignore[no-untyped-def]
            start = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                acc[key] += time.perf_counter() - start
                cnt[key] += 1

        setattr(owner, name, timed)

    install(pipeline_mod, "load_upright_rgb", "解码")
    install(pipeline_mod, "rectify_receipt", "矫正")
    install(LRCNNPredictor, "predict", "检测")
    if args.platform_checkpoint:
        try:
            from transfer_receipt_ai.device_statusbar import StatusBarDeviceClassifier

            install(StatusBarDeviceClassifier, "classify", "设备")
        except Exception as exc:  # noqa: BLE001
            print(f"  (设备计时未装:{exc})")
    # 拆开"其它"里最可疑的两块:标注绘制 与 存图/JPEG 编码
    install(pipeline_mod, "draw_original_circles", "标注绘制")
    install(pipeline_mod, "draw_rectified_circles", "标注绘制")
    install(pipeline_mod, "save_rgb", "存图编码")

    out = Path(tempfile.mkdtemp(prefix="profile_"))
    print(f"检测权重:{checkpoint}")
    print(f"跑检测阶段(不含 OCR),最多 {args.limit} 张,临时输出 {out} …\n")
    wall_start = time.perf_counter()
    outputs = run_inference(
        checkpoint=Path(checkpoint),
        input_path=args.input,
        output_dir=out,
        device=args.device,
        use_ocr=False,
        continue_on_error=True,
        limit=args.limit,
        platform_checkpoint=args.platform_checkpoint,
    )
    total = time.perf_counter() - wall_start
    n = max(1, len(outputs))

    print(f"\n检测阶段:成功 {len(outputs)} 张,总墙钟 {total:.1f}s = {total / n * 1000:.0f} ms/张\n")
    measured = 0.0
    for key in ("解码", "矫正", "检测", "设备", "标注绘制", "存图编码"):
        if key not in acc:
            continue
        measured += acc[key]
        print(f"  {key}: {acc[key] / n * 1000:7.1f} ms/张  占 {acc[key] / total * 100:5.1f}%  (累计 {acc[key]:.1f}s, 调用 {cnt[key]})")
    other = max(0.0, total - measured)
    print(f"  其它/JSON/开销: {other / n * 1000:7.1f} ms/张  占 {other / total * 100:5.1f}%")

    print("\n== OCR 占比另测(用生产双进程路径,最准)==")
    print(f"同样 {args.limit} 张各跑一次,取墙钟差(PowerShell):")
    print(f"  Measure-Command {{ <你平时的 receipt_inference.cli 命令> --ocr none   --limit {args.limit} }}")
    print(f"  Measure-Command {{ <你平时的 receipt_inference.cli 命令> --ocr paddle --limit {args.limit} }}")
    print("  OCR 占比 ≈ (paddle 秒 - none 秒) / paddle 秒")


if __name__ == "__main__":
    main()
