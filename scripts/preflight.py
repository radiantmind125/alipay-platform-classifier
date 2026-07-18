r"""上服务器前的预检：Python/依赖/GPU/金标/图片池，避免跑几小时才发现环境问题。

用法：python scripts\preflight.py --input D:\download\raw_images
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _check(name: str, fn, required: bool = True) -> bool:
    try:
        ok, msg = fn()
    except Exception as e:  # noqa: BLE001
        ok, msg = False, f"{type(e).__name__}: {e}"
    tag = "OK " if ok else ("×  " if required else "!  ")
    print(f"[{tag}] {name}: {msg}")
    return ok or not required


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="全量图片池目录（抽查一张能否读）")
    ap.add_argument("--gold", type=Path, default=Path("gold/gold_seed_v2.json"))
    args = ap.parse_args(argv)

    results: list[bool] = []
    results.append(_check("Python >= 3.10", lambda: (sys.version_info >= (3, 10), sys.version.split()[0])))

    for m in ("numpy", "PIL"):
        results.append(_check(f"import {m}（数据准备必需）",
                              lambda m=m: (True, getattr(importlib.import_module(m), "__version__", "?"))))

    def torch_check():
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "不可用（训练必需，数据准备不需要）"
        return cuda, f"torch {torch.__version__}, CUDA={name}"
    _check("torch + CUDA（训练用）", torch_check, required=False)

    _check("onnxruntime（上线用）",
           lambda: (True, importlib.import_module("onnxruntime").__version__), required=False)

    sys.path.insert(0, "src")
    results.append(_check("alipay_platform 可导入",
                          lambda: (True, importlib.import_module("alipay_platform").__version__)))
    results.append(_check("金标文件存在", lambda: (args.gold.exists(), str(args.gold))))

    if args.input:
        def pool_check():
            from PIL import Image
            first = None
            for p in args.input.rglob("*"):
                if p.suffix.lower() in _EXTS:
                    first = p
                    break
            if first is None:
                return False, "目录下没找到图片"
            with Image.open(first) as im:
                size = im.size
            return True, f"可读，示例 {first.name} {size}"
        results.append(_check("图片池可读", pool_check))

    ok = all(results)
    print("\n" + ("必需项全部通过，可以按 RUNBOOK 开跑。" if ok else "有必需项未通过（×），请先解决。"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
