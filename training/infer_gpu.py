"""GPU 批量推理：对清单（如 predict.jsonl）里已缓存的状态栏条跑 CNN，输出设备预测。

比 infer_cpu.py 快很多——直接用缓存的小状态栏条 + GPU，不重新解码整张原图。
用来给“模棱两可桶”出 Tier-1 结果，再交给 merge_device_labels 合并。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import StripDataset  # noqa: E402
from model import build_model  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", type=Path, required=True, help="要分类的清单（每行含 strip 路径与 file）")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.75)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ds = StripDataset(args.predict, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(num_classes=len(payload.get("classes", ["android", "ios"]))).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    total = len(ds)
    print(f"开始推理 {total} 张（{device}），单线程加载约几分钟，会每几批打印进度…", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    i = n_ios = n_and = n_unc = 0
    with torch.no_grad(), args.out.open("w", encoding="utf-8") as fo:
        for bi, (x, _) in enumerate(dl):
            probs = torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
            for pv in probs:
                pv = float(pv)
                conf = max(pv, 1 - pv)
                if conf < args.conf:
                    dev = "uncertain"; n_unc += 1
                elif pv > 0.5:
                    dev = "ios"; n_ios += 1
                else:
                    dev = "android"; n_and += 1
                r = ds.rows[i]
                fo.write(json.dumps({"file": r["file"], "device": dev, "p_ios": round(pv, 4), "conf": round(conf, 3)}, ensure_ascii=False) + "\n")
                i += 1
            if (bi + 1) % 5 == 0 or i >= total:
                print(f"  已处理 {i}/{total}（ios={n_ios} android={n_and} uncertain={n_unc}）", flush=True)
    print(f"完成 {i} 张（{device}）：ios={n_ios} android={n_and} uncertain={n_unc} -> {args.out}")


if __name__ == "__main__":
    main()
