r"""重训 SSP-tiny(经理说的"那几个评估模型重新训练"里,我们自己域上重训的那个)。

服务器 GPU 用 --manifest(jsonl 每行 {"file":..., "label":0/1});本机 CPU 用 --smoke
自动造数据冒烟验证流水线能跑能分:正样本=引擎C合成翻拍(+可选真翻拍),负样本=干净截图。

    冒烟(本机 CPU):
      python training/train_ssp.py --smoke --image-root C:\...\TempFakeImages --n 80 --epochs 10
    正式(服务器):
      python training/train_ssp.py --manifest data/screen_train.jsonl --val data/screen_val.jsonl --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssp_model import build_ssp, count_params  # noqa: E402
from alipay_platform.ssp_patch import npr_residual, select_patches, srm_residual  # noqa: E402
from alipay_platform.recapture_sim import simulate_recapture  # noqa: E402

_OPT = {33434, 33437, 34855, 37386}
PATCH = 32


def _load(p: Path, cap: int) -> np.ndarray:
    with Image.open(p) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = rgb.shape[:2]
    s = cap / max(h, w)
    if s < 1.0:
        import cv2
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return rgb


def feat(rgb: np.ndarray, k: int = 8, stride: int = 24) -> np.ndarray:
    """选块 -> 每块 [SRM3|NPR1] 残差 -> [K,4,32,32](不足补零,残差归一化)。"""
    sel = select_patches(rgb, patch=PATCH, stride=stride, topk=k)
    out = np.zeros((k, 4, PATCH, PATCH), np.float32)
    if sel.residual_null or not sel.coords:
        return out
    gray = rgb[:, :, :3].mean(axis=2).astype(np.float32)
    for i, (y, x) in enumerate(sel.coords[:k]):
        srm = srm_residual(gray[y:y + PATCH, x:x + PATCH])                 # 32,32,3
        npr = npr_residual(gray[y:y + PATCH, x:x + PATCH])[:, :, None]      # 32,32,1
        stk = np.concatenate([srm, npr], axis=2)                           # 32,32,4
        out[i] = np.clip(stk, -64, 64).transpose(2, 0, 1) / 64.0
    return out


def _pick_screenshots(root: Path, n: int, seed: int) -> list[Path]:
    import glob, random
    files = sorted(glob.glob(str(root / "*.jpg")) + glob.glob(str(root / "*.png")))
    random.Random(seed).shuffle(files)
    out: list[Path] = []
    for f in files:
        if len(out) >= n:
            break
        try:
            with Image.open(f) as im:
                w, h = im.size
                ifd = im.getexif().get_ifd(0x8769) if hasattr(im.getexif(), "get_ifd") else {}
                if min(w, h) in (720, 1080, 1170, 1206, 1290, 1320) and 2.0 < max(w, h) / min(w, h) < 2.4 \
                        and not any(t in ifd for t in _OPT):
                    out.append(Path(f))
        except Exception:
            continue
    return out


def build_smoke(root: Path, n: int, cap: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    neg_src = _pick_screenshots(root, n, seed=1)
    pos_src = _pick_screenshots(root, n, seed=999)                          # 另一批截图,合成成翻拍当正样本
    xs, ys = [], []
    print(f"负样本(干净截图) {len(neg_src)},正样本(合成翻拍) {len(pos_src)};提特征中…")
    for p in neg_src:
        try:
            xs.append(feat(_load(p, cap), k)); ys.append(0)
        except Exception:
            pass
    for i, p in enumerate(pos_src):
        try:
            sim = simulate_recapture(_load(p, cap), seed=i)
            xs.append(feat(sim, k)); ys.append(1)
        except Exception:
            pass
    return np.stack(xs).astype(np.float32), np.array(ys, np.int64)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tp = {0: 0, 1: 0}; tot = {0: 0, 1: 0}; correct = n = 0
    for x, y in loader:
        pred = (torch.sigmoid(model(x.to(device))[:, 0]) >= 0.5).long().cpu().numpy()
        for p, t in zip(pred, y.numpy()):
            tot[int(t)] += 1; n += 1
            if p == t:
                correct += 1; tp[int(t)] += 1
    rec = {c: (tp[c] / tot[c] if tot[c] else float("nan")) for c in (0, 1)}
    return {"acc": correct / max(1, n), "balanced_acc": float(np.nanmean([rec[0], rec[1]])),
            "recall_clean": rec[0], "recall_recap": rec[1], "n": n}


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="重训 SSP-tiny(翻拍头冒烟/正式)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--n", type=int, default=80, help="每类多少张(冒烟)")
    ap.add_argument("--cap", type=int, default=512, help="提特征前长边上限(冒烟提速)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("training/runs/ssp_screen_smoke"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.smoke:
        raise SystemExit("正式训练请接 --manifest(留待服务器数据就绪);本机先用 --smoke 验证流水线。")
    if not args.image_root:
        raise SystemExit("--smoke 需要 --image-root 指向截图池")

    torch.manual_seed(args.seed)
    x, y = build_smoke(args.image_root, args.n, args.cap, args.k)
    print(f"特征张量 {x.shape}(样本, K, C, 32, 32),正样本 {int(y.sum())} 负样本 {int((y==0).sum())}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y)); ntr = int(len(y) * 0.75)
    tr, va = idx[:ntr], idx[ntr:]
    xt = torch.from_numpy(x[tr]); yt = torch.from_numpy(y[tr]).float()
    xv = torch.from_numpy(x[va]); yv = torch.from_numpy(y[va])
    counts = np.bincount(y[tr], minlength=2).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    sampler = WeightedRandomSampler([w[t] for t in y[tr]], num_samples=len(tr), replacement=True)
    tl = DataLoader(TensorDataset(xt, yt), batch_size=args.batch_size, sampler=sampler, drop_last=True)
    vl = DataLoader(TensorDataset(xv, yv), batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = build_ssp(head_names=("screen",), cin=4).to(device)
    print(f"设备={device} 参数量={count_params(model):,}")
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(device))[:, 0], yb.to(device))
            loss.backward(); opt.step(); run += float(loss.detach())
        sch.step()
        m = evaluate(model, vl, device)
        print(f"epoch {ep+1}/{args.epochs} loss={run/max(1,len(tl)):.3f} "
              f"val_acc={m['acc']:.3f} bal={m['balanced_acc']:.3f} "
              f"(clean={m['recall_clean']:.2f} recap={m['recall_recap']:.2f})")
        if m["balanced_acc"] > best:
            best = m["balanced_acc"]
            torch.save({"model_state": model.state_dict(), "head_names": ["screen"], "cin": 4}, args.out / "best.pt")
    print(f"冒烟完成 best balanced_acc={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
