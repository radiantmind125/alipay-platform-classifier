r"""分块打分(tiled) —— 试图补上"只改一小块"的结构盲区, **不用重训**。

问题(已实测坐实): SSP 打分时是在**整张图**里随机取 64 个 32x32 小块, 只留**纹理最丰富的那一个**。
收据上最"丰富"的通常是底部彩色缩略图/图标, 所以金额那一小块几乎永远选不中 ->
只改金额的假图 300 张 100% 漏检(median 0.002 = 和真图一样), 而整图重绘的对照组 98.7% 拦下。

思路: 把图切成若干块, **每块单独打一次分**, 再取最高分(或 top3 均值)当整图分数。
块内的"最富纹理块"就只能在这一块里选 -> 改动区所在的那一块就有机会被看到。
好处: 复用现有 v6 模型, 无需重训。
代价: 打分次数变多, **误杀风险上升**(机会多了) -> 必须按经理"绝对不能误杀"的红线, 在真图上重测阈值。

产出 summary.csv 与 predict_all_models.py 兼容, 可直接喂 eval_summary.py。
CSV 里同时存了 max / mean / top3 三种聚合, 事后调阈值不用重跑。

用法(服务器):
  python training/predict_tiled.py --ssp-repo D:\SSP ^
    --model D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v6\Net_epoch_best.pth ^
    --input D:\probe\localedit --output_dir D:\probe\localedit_tiled --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _tiles(w: int, h: int, cols: int, rows: int, overlap: float):
    """切成 cols x rows 块, 相邻块按 overlap 比例重叠(防改动区正好被切成两半)。"""
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tw - ox)); y0 = max(0, int(r * th - oy))
            x1 = min(w, int((c + 1) * tw + ox)); y1 = min(h, int((r + 1) * th + oy))
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                out.append((x0, y0, x1, y1))
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="分块打分(补局部改动盲区, 不重训)")
    ap.add_argument("--ssp-repo", required=True, help="SSP 预测仓库路径(取 networks/ 和 utils/), 如 D:\\SSP")
    ap.add_argument("--model", required=True, help="Net_epoch_best.pth")
    ap.add_argument("--input", required=True, help="图片目录")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=6, help="截图是竖长条, 行多列少更合理")
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--repeat", type=int, default=4, help="每块重复取块次数(块内已经小, 不用 16)")
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--trainsize", type=int, default=256)
    ap.add_argument("--agg", default="max", choices=["max", "top3", "mean"], help="整图分数怎么聚合各块")
    ap.add_argument("--reject_threshold", type=float, default=0.90)
    ap.add_argument("--review_threshold", type=float, default=0.60)
    ap.add_argument("--ai_label", type=int, default=0, choices=[0, 1])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.ssp_repo)))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageFile
    from torchvision import transforms
    from networks.resnet import resnet50
    from networks.srm_conv import SRMConv2d_simple
    from utils.patch import patch_img
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class SSPNet(nn.Module):          # 与官方 networks/ssp.py 同构(pretrained=False 免下载)
        def __init__(self):
            super().__init__()
            self.srm = SRMConv2d_simple()
            self.disc = resnet50(pretrained=False)
            self.disc.fc = nn.Linear(2048, 1)

        def forward(self, x):
            x = F.interpolate(x, (256, 256), mode="bilinear")
            return self.disc(self.srm(x))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("没有 CUDA, 改用 CPU"); args.device = "cpu"
    device = torch.device(args.device)

    sd = torch.load(args.model, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model = SSPNet()
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    root = Path(args.input)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in _EXTS) if root.is_dir() else [root]
    if not imgs:
        print("没找到图片"); return
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"图片 {len(imgs)} 张 | 切块 {args.cols}x{args.rows} 重叠 {args.overlap} | 每块重复 {args.repeat} | 聚合 {args.agg}",
          flush=True)

    rows_out = []
    for i, ip in enumerate(imgs, 1):
        try:
            im = Image.open(ip).convert("RGB")
            w, h = im.size
            tile_scores = []
            for (x0, y0, x1, y1) in _tiles(w, h, args.cols, args.rows, args.overlap):
                crop = im.crop((x0, y0, x1, y1))
                batch = [tf(patch_img(crop, args.patch_size, args.trainsize)) for _ in range(args.repeat)]
                with torch.no_grad():
                    logits = model(torch.stack(batch).to(device)).view(-1)
                    s = torch.sigmoid(logits)
                    s = (1.0 - s) if args.ai_label == 0 else s
                tile_scores.append(float(s.mean()))       # 该块的分数(块内重复取平均)
            tile_scores.sort(reverse=True)
            s_max = tile_scores[0]
            s_top3 = sum(tile_scores[:3]) / min(3, len(tile_scores))
            s_mean = sum(tile_scores) / len(tile_scores)
            final = {"max": s_max, "top3": s_top3, "mean": s_mean}[args.agg]
            dec = ("Reject" if final >= args.reject_threshold else
                   "ManualReview" if final >= args.review_threshold else "PassToBusinessVerify")
            rows_out.append({"image": str(ip), "image_name": ip.name,
                             "final_ai_score": round(final, 6),
                             "tile_max": round(s_max, 6), "tile_top3": round(s_top3, 6),
                             "tile_mean": round(s_mean, 6), "n_tiles": len(tile_scores),
                             "decision": dec,
                             "scores_json": json.dumps({"tiled": round(final, 6)})})
            if i % 50 == 0:
                print(f"  已算 {i}/{len(imgs)}...", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  失败 {ip.name}: {str(exc)[:150]}", flush=True)

    sp = out_dir / "summary.csv"
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=["image", "image_name", "final_ai_score", "tile_max",
                                             "tile_top3", "tile_mean", "n_tiles", "decision", "scores_json"])
        wcsv.writeheader(); wcsv.writerows(rows_out)
    print(f"完成 {len(rows_out)} 张 -> {sp}")
    print("下一步: python training/eval_summary.py <上面的 summary.csv> --kind fake|genuine")
    print("提示: CSV 里存了 tile_max/tile_top3/tile_mean, 想换聚合方式不用重跑, 改 --agg 或直接看这几列。")


if __name__ == "__main__":
    main()
