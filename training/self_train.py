"""门控自训练：用 CNN 对“模棱两可桶”打伪标签扩充训练，每轮用金标+对抗集把关，可回滚。

依据评审的护栏：
- 只对模棱两可桶（predict.jsonl，分辨率判不了的 ~21%）打伪标签。
- 只收“校准置信度 > 阈值(0.95)”的；两类**均衡录入**，避免塌向 ~94% 的安卓多数。
- 冻结的自举标签是锚点；每轮在金标 + 对抗集上评估，若头号指标（对抗集平衡准确率）回退就停并回滚。
- 每轮伪标签带轮次来源，可回滚。

需在 GPU 服务器上、CNN v0 训练完之后运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import StripDataset
from model import build_model
from train import class_balanced_sampler, evaluate


def _train_model(train_manifest: Path, val_manifest: Path, device: str, epochs: int,
                 init_state=None, lr: float = 3e-3, batch_size: int = 256, workers: int = 4):
    train_ds = StripDataset(train_manifest, train=True)
    val_ds = StripDataset(val_manifest, train=False)
    tl = DataLoader(train_ds, batch_size=batch_size, sampler=class_balanced_sampler(train_ds.labels()),
                    num_workers=workers, drop_last=True, pin_memory=device.startswith("cuda"))
    vl = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)
    model = build_model(num_classes=2).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
        m = evaluate(model, vl, device)
        if m["balanced_acc"] > best:
            best, best_state = m["balanced_acc"], {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def _score(model, manifest: Path, device: str, batch_size: int = 256):
    ds = StripDataset(manifest, train=False)
    dl = DataLoader(ds, batch_size=batch_size)
    model.eval()
    p_ios: list[float] = []
    for x, _ in dl:
        p = torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
        p_ios.extend(p.tolist())
    return ds.rows, p_ios


def _admit_pseudo(rows, p_ios, conf_thr: float, max_per_class: int, round_id: int):
    """按置信度筛，两类均衡录入伪标签。"""
    cand = []
    for row, p in zip(rows, p_ios):
        conf = max(p, 1 - p)
        if conf >= conf_thr:
            cand.append({"strip": row["strip"], "file": row["file"], "label": 1 if p > 0.5 else 0,
                         "p_ios": round(float(p), 4), "provenance": f"pseudo_round_{round_id}"})
    ios = [c for c in cand if c["label"] == 1]
    andr = [c for c in cand if c["label"] == 0]
    k = min(len(ios), len(andr), max_per_class)
    # 取每类置信度最高的 k 个（均衡）。
    ios.sort(key=lambda c: -max(c["p_ios"], 1 - c["p_ios"]))
    andr.sort(key=lambda c: -max(c["p_ios"], 1 - c["p_ios"]))
    return ios[:k] + andr[:k], len(ios), len(andr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--predict", type=Path, required=True, help="模棱两可桶（打伪标签的对象）")
    ap.add_argument("--adversarial", type=Path, required=True, help="对抗测试集（头号指标）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--conf-threshold", type=float, default=0.95)
    ap.add_argument("--max-per-class", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    base_train = [json.loads(l) for l in args.train.read_text(encoding="utf-8").splitlines() if l.strip()]

    def adv_score(model) -> float:
        return evaluate(model, DataLoader(StripDataset(args.adversarial, train=False), batch_size=256), device)["balanced_acc"]

    print("训练 v0（仅自举标签）…")
    model, val_bal = _train_model(args.train, args.val, device, args.epochs)
    best_adv = adv_score(model)
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"model_state": best_state, "classes": ["android", "ios"], "round": 0, "adv_balanced": best_adv},
               args.out / "best.pt")
    print(f"  v0: val_bal={val_bal:.3f} 对抗集平衡={best_adv:.3f}")

    for r in range(1, args.rounds + 1):
        rows, p_ios = _score(model, args.predict, device)
        pseudo, n_ios, n_and = _admit_pseudo(rows, p_ios, args.conf_threshold, args.max_per_class, r)
        aug_path = args.out / f"train_round{r}.jsonl"
        with aug_path.open("w", encoding="utf-8") as f:
            for row in base_train + pseudo:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"第 {r} 轮：候选伪标签 ios={n_ios} android={n_and}，均衡录入 {len(pseudo)}；重训…")
        model, val_bal = _train_model(aug_path, args.val, device, args.epochs, init_state=best_state)
        adv = adv_score(model)
        print(f"  round{r}: val_bal={val_bal:.3f} 对抗集平衡={adv:.3f}（历史最好 {best_adv:.3f}）")
        if adv > best_adv:
            best_adv = adv
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "classes": ["android", "ios"], "round": r, "adv_balanced": best_adv},
                       args.out / "best.pt")
            print(f"  ✓ 对抗集提升，保存 best（round {r}）")
        else:
            print("  ✗ 对抗集未提升，停止自训练并回滚到历史最好。")
            model.load_state_dict(best_state)
            break
    print(f"完成。最好对抗集平衡准确率={best_adv:.3f}，best.pt 在 {args.out}")


if __name__ == "__main__":
    main()
