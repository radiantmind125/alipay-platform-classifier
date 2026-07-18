"""用自举标签训练一个“状态栏条 -> 安卓/苹果”的分类器（numpy 逻辑回归），
在金标上验证，并对模棱两可（分辨率弃权）的图给出预测，完成整池 100% 设备标注。

- 训练标签来自分辨率自举（高精度，已验证）；金标文件从训练集中剔除，避免验证泄漏。
- 所有状态栏条缩放到固定尺寸取特征，因此不泄漏分辨率。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.strip_features import FEATURE_NAMES, load_features  # noqa: E402


def _crop_path(crop_dir: Path, file: str) -> Path:
    stem = Path(file).with_suffix("").as_posix().replace("/", "__")
    return crop_dir / f"{stem}.png"


def _seeded_shuffle(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx


def train_lr(X, y, w, *, epochs=1200, lr=0.2, l2=1e-3):
    n, d = X.shape
    W = np.zeros(d)
    b = 0.0
    w = w / w.mean()
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ W + b)))
        grad = (p - y) * w
        W -= lr * (X.T @ grad / n + l2 * W)
        b -= lr * grad.mean()
    return W, b


def predict_proba(X, W, b):
    return 1.0 / (1.0 + np.exp(-(X @ W + b)))


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("runs/pool_20260701/bootstrap_labels.jsonl"))
    ap.add_argument("--crops", type=Path, default=Path("runs/pool_20260701/status_bar"))
    ap.add_argument("--gold", type=Path, default=Path("gold/gold_seed_v1.json"))
    ap.add_argument("--cache", type=Path, default=Path("runs/pool_20260701/strip_features.npz"))
    ap.add_argument("--out-pred", type=Path, default=Path("runs/pool_20260701/device_predictions.jsonl"))
    ap.add_argument("--out-model", type=Path, default=Path("runs/pool_20260701/strip_lr_model.json"))
    ap.add_argument("--conf", type=float, default=0.75, help="判定所需最低置信度，低于则标 uncertain")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.labels.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = {r["file"]: r["platform"] for r in json.loads(args.gold.read_text(encoding="utf-8"))["labels"]}

    # 特征缓存，便于反复调参。
    if args.cache.exists():
        data = np.load(args.cache, allow_pickle=True)
        feats = {k: data[k] for k in data.files}
        files = list(feats["_files"])
        X_all = feats["_X"]
        print(f"从缓存载入特征：{X_all.shape}")
    else:
        files, feat_list = [], []
        n = len(rows)
        for i, r in enumerate(rows):
            p = _crop_path(args.crops, r["file"])
            if not p.exists():
                continue
            try:
                feat_list.append(load_features(str(p)))
                files.append(r["file"])
            except Exception:
                continue
            if (i + 1) % 500 == 0:
                print(f"  提特征 {i+1}/{n}")
        X_all = np.vstack(feat_list)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, _X=X_all, _files=np.array(files, dtype=object))
        print(f"特征已缓存：{X_all.shape}")

    label_of = {r["file"]: r["label"] for r in rows}
    y_map = {"ios": 1.0, "android": 0.0}

    # 训练集 = 分辨率给出 ios/android 且不在金标里的图。
    tr_idx = [i for i, f in enumerate(files) if label_of[f] in ("ios", "android") and f not in gold]
    Xtr_raw = X_all[tr_idx]
    ytr = np.array([y_map[label_of[files[i]]] for i in tr_idx])

    mean, std = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
    Xtr = (Xtr_raw - mean) / std

    # 类别权重（ios 多，安卓少）。
    w = np.where(ytr == 1, 1.0 / (ytr == 1).mean(), 1.0 / (ytr == 0).mean())

    # 80/20 留出评估。
    order = _seeded_shuffle(len(Xtr))
    cut = int(len(order) * 0.8)
    tr, te = order[:cut], order[cut:]
    W, b = train_lr(Xtr[tr], ytr[tr], w[tr])
    pte = predict_proba(Xtr[te], W, b)
    acc_te = ((pte > 0.5) == (ytr[te] > 0.5)).mean()
    # 平衡准确率
    ios_te = (ytr[te] == 1)
    and_te = (ytr[te] == 0)
    bal = 0.5 * (((pte > 0.5) & ios_te).sum() / max(1, ios_te.sum()) + ((pte <= 0.5) & and_te).sum() / max(1, and_te.sum()))
    print(f"\n留出集（{len(te)} 张）：准确率={acc_te:.3f} 平衡准确率={bal:.3f}")

    # 全量训练最终模型。
    Wf, bf = train_lr(Xtr, ytr, w)

    # 金标验证（含模棱两可的 6 张，这是目标域，最有价值）。
    g_correct = g_total = amb_correct = amb_total = 0
    amb_labels = {f: label_of.get(f) for f in gold}
    for i, f in enumerate(files):
        if f not in gold:
            continue
        x = (X_all[i] - mean) / std
        p = predict_proba(x[None, :], Wf, bf)[0]
        pred = "ios" if p > 0.5 else "android"
        truth = gold[f]
        g_total += 1
        g_correct += int(pred == truth)
        if amb_labels[f] == "abstain":   # 分辨率判不了的（模棱两可）金标
            amb_total += 1
            amb_correct += int(pred == truth)
    print(f"金标全部 {g_total} 张：状态栏分类器准确率={g_correct/max(1,g_total):.3f}")
    if amb_total:
        print(f"其中模棱两可 {amb_total} 张（目标域）：准确率={amb_correct/amb_total:.3f}")

    # 对模棱两可的图给出预测，完成整池标注。
    amb_idx = [i for i, f in enumerate(files) if label_of[f] == "abstain"]
    Xamb = (X_all[amb_idx] - mean) / std
    pamb = predict_proba(Xamb, Wf, bf)
    n_ios = int((pamb > 0.5).sum())
    n_and = int((pamb <= 0.5).sum())
    n_unc = int((np.abs(pamb - 0.5) < (args.conf - 0.5)).sum())
    print(f"\n模棱两可 {len(amb_idx)} 张的状态栏预测：ios={n_ios} android={n_and}；其中低置信(<{args.conf})={n_unc} 标 uncertain")

    args.out_pred.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pred.open("w", encoding="utf-8") as fo:
        for j, i in enumerate(amb_idx):
            p = float(pamb[j])
            conf = max(p, 1 - p)
            device = "uncertain" if conf < args.conf else ("ios" if p > 0.5 else "android")
            fo.write(json.dumps({"file": files[i], "device": device, "p_ios": round(p, 4), "conf": round(conf, 3)}, ensure_ascii=False) + "\n")
    args.out_model.write_text(json.dumps({
        "feature_names": list(FEATURE_NAMES), "mean": mean.tolist(), "std": std.tolist(),
        "W": Wf.tolist(), "b": float(bf), "conf_threshold": args.conf,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"预测写入 {args.out_pred}；模型写入 {args.out_model}")


if __name__ == "__main__":
    main()
