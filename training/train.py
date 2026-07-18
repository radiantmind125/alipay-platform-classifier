"""在 GPU 上训练状态栏 tiny-CNN；用金标验证，按平衡准确率存 best。

用法（服务器 GPU）：
    python training/train.py --train training/data/train.jsonl --val training/data/val.jsonl \
        --out training/runs/statusbar_v1 --epochs 20 --batch-size 256 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import StripDataset  # 同目录
from model import PLATFORM_CLASSES, build_model, count_params


def class_balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = [inv[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tp = {0: 0, 1: 0}
    tot = {0: 0, 1: 0}
    correct = n = 0
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        y = y.numpy()
        for p, t in zip(pred, y):
            tot[int(t)] += 1
            n += 1
            if p == t:
                correct += 1
                tp[int(t)] += 1
    recall = {k: (tp[k] / tot[k] if tot[k] else float("nan")) for k in (0, 1)}
    bal = np.nanmean([recall[0], recall[1]])
    return {"acc": correct / max(1, n), "balanced_acc": float(bal),
            "recall_android": recall[0], "recall_ios": recall[1], "n": n}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = StripDataset(args.train, train=True)
    val_ds = StripDataset(args.val, train=False)
    sampler = class_balanced_sampler(train_ds.labels())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.startswith("cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=2, width=args.width).to(device)
    print(f"设备={device} 参数量={count_params(model):,} 训练={len(train_ds)} 验证={len(val_ds)}")

    # 平衡靠采样器，不写死类别权重；用标签平滑抵御 ~1% 自举噪声。
    # 先验(≈60/40 iOS/安卓)在推理端用 logit 调整重定向，不烘焙进权重（分布会随时间漂移）。
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best = -1.0
    history_path = args.out / "history.jsonl"
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics.update({"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), "lr": optimizer.param_groups[0]["lr"]})
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(f"epoch {epoch+1}/{args.epochs} loss={metrics['train_loss']:.4f} "
              f"val_acc={metrics['acc']:.3f} bal={metrics['balanced_acc']:.3f} "
              f"(and={metrics['recall_android']:.2f} ios={metrics['recall_ios']:.2f})")

        payload = {"model_state": model.state_dict(), "classes": list(PLATFORM_CLASSES),
                   "width": args.width, "img_hw": [64, 384], "metrics": metrics}
        torch.save(payload, args.out / "last.pt")
        if metrics["balanced_acc"] > best:
            best = metrics["balanced_acc"]
            torch.save(payload, args.out / "best.pt")
            print(f"  saved best (balanced_acc={best:.3f})")
    print(f"训练完成，best balanced_acc={best:.3f}，checkpoint 在 {args.out}")


if __name__ == "__main__":
    main()
