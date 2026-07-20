r"""对一批新图一条命令出设备标签(可选顺带标注)。用于经理拿最新蓝图验证。

流程 = 生产两层:先读分辨率(Tier-0,零解码,~78% 直接判);判不了的再用状态栏 CNN(Tier-1)。
自带裁条,不需要先跑 inspect/缓存。需 torch(GPU;没 GPU 会退回 CPU)。

用法:
  python training\classify_batch.py --input <新图目录> --model training\runs\statusbar_v2\best.pt `
      --output <结果目录> --annotate 300
产出:<结果目录>\final_device.jsonl(+ 若 --annotate 则 <结果目录>\annotated\ 红字标注样例)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # training/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))           # src/
from preprocess import preprocess_original  # noqa: E402
from model import build_model  # noqa: E402
from alipay_platform.bootstrap import resolution_platform  # noqa: E402
from alipay_platform.metadata_seed import read_metadata_facts  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(root: Path):
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _EXTS:
            yield p


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="新图目录")
    ap.add_argument("--model", type=Path, required=True, help="best.pt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.5, help="默认 0.5 = 强制判定(无不确定);要保留不确定用 0.75")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--annotate", type=int, default=0, help=">0 则顺带随机标注这么多张(红字面板)")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    payload = torch.load(args.model, map_location=device, weights_only=False)
    model = build_model(2, width=float(payload.get("width", 1.0))).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    root = args.input if args.input.is_dir() else args.input.parent
    imgs = list(_iter_images(args.input))
    print(f"共 {len(imgs)} 张,先按分辨率分流…", flush=True)

    # Tier-0:分辨率(零解码)
    result: dict[Path, dict] = {}
    abstain: list[Path] = []
    for p in imgs:
        try:
            f = read_metadata_facts(p)
        except Exception:
            continue
        plat = resolution_platform(f.width, f.height)
        rel = p.relative_to(root).as_posix()
        if plat in ("ios", "android"):
            result[p] = {"file": rel, "device": plat, "source": "resolution", "confidence": 0.99}
        else:
            abstain.append(p)
    print(f"  分辨率直接判 {len(result)} 张;需状态栏模型 {len(abstain)} 张,推理中…", flush=True)

    # Tier-1:状态栏 CNN(分批)
    for i in range(0, len(abstain), args.batch_size):
        chunk = abstain[i : i + args.batch_size]
        xs, keep = [], []
        for p in chunk:
            try:
                rgb = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
                xs.append(torch.from_numpy(preprocess_original(rgb)))
                keep.append(p)
            except Exception:
                continue
        if not xs:
            continue
        with torch.no_grad():
            probs = torch.softmax(model(torch.stack(xs).to(device)), 1)[:, 1].cpu().numpy()
        for p, pv in zip(keep, probs):
            pv = float(pv)
            conf = max(pv, 1 - pv)
            dev = "uncertain" if conf < args.conf else ("ios" if pv > 0.5 else "android")
            result[p] = {"file": p.relative_to(root).as_posix(), "device": dev, "source": "cnn",
                         "confidence": round(conf, 3), "p_ios": round(pv, 4)}
        if (i // args.batch_size) % 4 == 0:
            print(f"    CNN {min(i + args.batch_size, len(abstain))}/{len(abstain)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output / "final_device.jsonl"
    cnt: Counter[str] = Counter()
    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in imgs:
            if p in result:
                f.write(json.dumps(result[p], ensure_ascii=False) + "\n")
                cnt[result[p]["device"]] += 1
    print(f"\n完成 {sum(cnt.values())} 张:{dict(cnt)}\n结果 -> {out_jsonl}", flush=True)

    if args.annotate > 0:
        ann = Path(__file__).resolve().parents[1] / "scripts" / "annotate_results.py"
        subprocess.run([sys.executable, str(ann), "--results", str(out_jsonl), "--image-root", str(args.input),
                        "--output", str(args.output / "annotated"), "--limit", str(args.annotate)], check=False)


if __name__ == "__main__":
    main()
