r"""增强一致性检查(无人工、无独立信号也能用)。

思路:一个判得对、判得稳的模型,对"保标签"的扰动应当给出同样的结论。于是给每张图做
几种**不改变状态栏语义**的扰动(降采样=模拟被缩放的低清源、JPEG 重压缩=模拟转发、
亮度、轻裁剪),看预测翻不翻:不翻=稳(可信),翻=脆(该进不确定带 / 是该补训的难例)。

**绝不做水平翻转**——那会把 iOS 的左时钟/右电池镜像过去,改变语义、不是保标签扰动。

用途(据对抗验证收窄):**只作弃权门 / 难例挑选**——不稳的样本转"不确定"(用覆盖换精度),
脆弱子集给"该补新信息的难例"做参考。**注意边界**:一致性≠正确性,只抓方差型错、抓不到
系统性错,在难例上会**高估**准确率,所以**不能当准确率估计**;且**绝不能同时既当训练正则
又当置信度**(会表征坍缩)。扰动必须真保标签、且其分布最好保密(否则可被构造成"只在已知
扰动下稳定"的假栏)。需 torch + 模型;只读,不动模型文件、不写结果库。

用法:
  python training\consistency_check.py --input D:\download\TempFakeImages --model training\runs\statusbar_v2\best.pt --out-dir runs\pool_full\consistency --limit 3000
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))                     # training/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))         # src/
from preprocess import crop_status_strip, strip_to_canvas, normalize  # noqa: E402
from model import build_model  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def augment_strip(strip: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """裁好的状态栏条(uint8 HxWx3)-> 一组保标签变体的 512x64 画布。"""
    h, w = strip.shape[:2]
    outs: list[tuple[str, np.ndarray]] = [("原图", strip_to_canvas(strip))]
    for f in (0.6, 0.8):  # 降采样再由画布放大 = 细节损失,模拟低清/被缩放的源
        small = np.asarray(Image.fromarray(strip).resize((max(8, int(w * f)), max(4, int(h * f)))))
        outs.append((f"降采样{f}", strip_to_canvas(small)))
    for q in (40, 65):  # JPEG 重压缩,模拟支付宝/微信转发
        buf = io.BytesIO()
        Image.fromarray(strip).save(buf, format="JPEG", quality=q)
        outs.append((f"JPEG{q}", strip_to_canvas(np.asarray(Image.open(buf).convert("RGB")))))
    for b in (0.85, 1.15):  # 亮度
        outs.append((f"亮度{b}", strip_to_canvas(np.clip(strip.astype(np.float32) * b, 0, 255).astype(np.uint8))))
    dh, dw = int(h * 0.05), int(w * 0.05)  # 轻裁剪 5%
    if h - 2 * dh > 2 and w - 2 * dw > 2:
        outs.append(("裁剪5%", strip_to_canvas(strip[dh:h - dh, dw:w - dw])))
    return outs


@torch.no_grad()
def score_image(model, device: str, path: Path) -> dict:
    rgb = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    variants = augment_strip(crop_status_strip(rgb))
    x = torch.stack([torch.from_numpy(normalize(c)) for _, c in variants]).to(device)
    p = torch.softmax(model(x), 1)[:, 1].cpu().numpy()  # 每个变体的 P(苹果)
    orig = int(p[0] > 0.5)
    flips = int(sum(1 for pv in p[1:] if int(pv > 0.5) != orig))
    return {"file": path.name, "pred": "ios" if orig else "android",
            "p_ios_orig": round(float(p[0]), 4), "conf": round(float(max(p[0], 1 - p[0])), 4),
            "n_aug": len(p) - 1, "flips": flips, "stable": flips == 0,
            "p_ios_spread": round(float(p.max() - p.min()), 4)}


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
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    payload = torch.load(args.model, map_location=device, weights_only=False)
    model = build_model(2, width=float(payload.get("width", 1.0))).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    imgs = list(_iter_images(args.input))
    random.seed(args.seed)
    random.shuffle(imgs)
    imgs = imgs[: args.limit]

    rows, fragile = [], []
    for i, p in enumerate(imgs):
        try:
            r = score_image(model, device, p)
        except Exception:
            continue
        rows.append(r)
        if not r["stable"]:
            fragile.append(r)
        if i % 500 == 0 and i:
            print(f"  {i}/{len(imgs)}", flush=True)

    if not rows:
        print("没有可评估的图")
        return
    n = len(rows)
    stable = sum(1 for r in rows if r["stable"])
    hi = [r for r in rows if r["conf"] >= 0.9]        # 模型自认为有把握的
    lo = [r for r in rows if r["conf"] < 0.9]
    hi_stable = sum(1 for r in hi if r["stable"])
    lo_stable = sum(1 for r in lo if r["stable"])
    print(f"\n评估 {n} 张,每张 {rows[0]['n_aug']} 个保标签扰动")
    print(f"  稳定(扰动下判定不翻)      {stable}/{n} = {stable/n:.1%}")
    print(f"  脆弱(至少翻一次)          {n-stable}/{n} = {(n-stable)/n:.1%}  → 应进不确定带/难例")
    print(f"  平均 P(苹果) 波动幅度        {sum(r['p_ios_spread'] for r in rows)/n:.3f}")
    print(f"  自信(conf≥0.9)里仍脆弱的    {len(hi)-hi_stable}/{len(hi) or 1} = {(len(hi)-hi_stable)/max(1,len(hi)):.1%}  (越低越好:说明置信度靠谱)")
    print(f"  不自信(conf<0.9)里稳定的    {lo_stable}/{len(lo) or 1} = {lo_stable/max(1,len(lo)):.1%}")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "consistency.jsonl").open("w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (args.out_dir / "fragile.jsonl").open("w", encoding="utf-8") as w:
            for r in fragile:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n逐图 -> {args.out_dir/'consistency.jsonl'}；脆弱子集({len(fragile)}) -> {args.out_dir/'fragile.jsonl'}")


if __name__ == "__main__":
    main()
