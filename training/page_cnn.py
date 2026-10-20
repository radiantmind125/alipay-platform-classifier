"""整页分类小 CNN(只看图, 不做 OCR): 训练 + 评估。数据(缩略图缓存 + 名单)见 page_cnn_data.py。

    python training/page_cnn.py train --cache 缓存目录 --out 输出目录 --views both [--epochs 15] [--device cuda]
    python training/page_cnn.py eval  --cache 缓存目录 --ckpt 输出目录\\best.pt [--target 0.995]
    (train 跑完自动 eval 一次, 结果在 输出目录\\eval)

类别 CLASSES = 规则 + OCR 定得下来、样子也认得出的 10 类。U / B0 不训练(说不清的剩余, 不是一种样子), P 交给翻拍规则。
快速分类 = 只看图的规则先出手(manifest 的 rule 列), 剩下的交给模型。所以:
  门槛只在"规则定不下来"的 val 上定(连 U / B0 一起算: 它们被判成哪类就是那类的错), 让每类精确率 >= --target, 低于门槛记 U;
  报告主要看 test(census: 从凭证池随机抽的, 比例和整池一样)里规则定不下来的那些, 再看规则 + 模型合起来, 再看 test_py(拼音库)。
  每类精确率旁边写 95% 单边下界(Clopper-Pearson): 几十张全对也只说明错误率不高于几个百分点, 别当成 0.5% 读。

模型 PageNet: 每个视图一支深度可分离卷积(model.py 的 DWSep, 总步长 32), 末尾按格子平均(保留大致位置: 蓝图的标题、
银行卡那一行在哪), 拼起来过两层全连接。输入 float NCHW BGR 0~255, 归一化在模型里(以后导 ONNX 时 C# 直接喂 uint8)。
★ 新文件, 不碰设备分类(StatusBarNet)的任何文件和输出目录。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DWSep  # noqa: E402
from page_cnn_data import HEAD_FRAC, HEAD_H, HEAD_W, PAGE_H, PAGE_W  # noqa: E402

CLASSES = ("B1", "B2", "B3", "B4", "W1", "W2", "W3", "W4", "D", "X")    # P(翻拍照片)只看图的规则就定了, 不交给模型
CI = {c: i for i, c in enumerate(CLASSES)}
SIZE = {"page": (PAGE_H, PAGE_W), "head": (HEAD_H, HEAD_W)}
GRID = {"page": (5, 3), "head": (3, 7)}      # 末尾特征图 page 20x9、head 12x14(高 x 宽), 格子要整除
MEAN_BGR = (0.406, 0.456, 0.485)
STD_BGR = (0.225, 0.224, 0.229)


# ---------------------------------------------------------------- 数据
def augment(img, rng: random.Random):
    """训练时的扰动: 缩小再放大(转发、截图被压过)、轻微平移缩放、亮度对比度、JPEG 压缩。不翻转(版面左右不对称)。"""
    h, w = img.shape[:2]
    if rng.random() < 0.3:
        s = rng.uniform(0.5, 0.9)
        small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        sc = rng.uniform(0.97, 1.03)
        dx, dy = rng.uniform(-0.02, 0.02) * w, rng.uniform(-0.02, 0.02) * h
        m = np.float32([[sc, 0, dx + (1 - sc) * w / 2], [0, sc, dy + (1 - sc) * h / 2]])
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.9, 1.1), beta=rng.uniform(-12, 12))
    if rng.random() < 0.5:
        q = int(rng.uniform(40, 95))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img


class PageSet(Dataset):
    def __init__(self, cache: Path, rows: list, views: tuple, train: bool) -> None:
        self.cache, self.rows, self.views, self.train = str(cache), rows, views, train
        self.rng = random.Random(0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, k: int):
        r = self.rows[k]
        out = []
        for v in self.views:
            img = cv2.imdecode(np.fromfile(f"{self.cache}/{v}/{r['id']}.jpg", np.uint8), cv2.IMREAD_COLOR)
            if self.train:
                img = augment(img, self.rng)
            out.append(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1))
        return (*out, CI.get(r["lab"], -1), k)


def _worker_init(wid: int) -> None:
    cv2.setNumThreads(1)
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(info.seed % (2 ** 31))


def load_rows(cache: Path) -> list:
    """读名单, 去掉读不出的; lab = 人眼定的类别(gold 名单里有的), 没有就用 page_type 的。"""
    with (cache / "manifest.jsonl").open(encoding="utf-8") as f:
        rows = [r for r in (json.loads(x) for x in f) if not r["err"]]
    for r in rows:
        r["lab"] = r["truth"].split("/")[0] if r.get("truth") else r["cls"]
    return rows


# ---------------------------------------------------------------- 模型
class Branch(nn.Module):
    def __init__(self, grid: tuple, width: float = 1.0) -> None:
        super().__init__()

        def c(n: int) -> int:
            return max(8, int(round(n * width)))

        self.stem = nn.Sequential(nn.Conv2d(3, c(24), 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c(24)), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            DWSep(c(24), c(48), 2), DWSep(c(48), c(48), 1),
            DWSep(c(48), c(96), 2), DWSep(c(96), c(96), 1),
            DWSep(c(96), c(128), 2), DWSep(c(128), c(128), 1),
            DWSep(c(128), c(160), 2), DWSep(c(160), c(160), 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.out_dim = c(160) * grid[0] * grid[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(x))).flatten(1)


class PageNet(nn.Module):
    def __init__(self, views: tuple, n_cls: int = len(CLASSES), width: float = 1.0) -> None:
        super().__init__()
        self.views = tuple(views)
        self.br = nn.ModuleDict({v: Branch(GRID[v], width) for v in self.views})
        d = sum(b.out_dim for b in self.br.values())
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, n_cls))
        self.register_buffer("mean", torch.tensor(MEAN_BGR).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD_BGR).view(1, 3, 1, 1) * 255.0)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """xs: 按 self.views 的顺序, float NCHW BGR 0~255。"""
        f = [self.br[v]((x - self.mean) / self.std) for v, x in zip(self.views, xs)]
        return self.fc(torch.cat(f, 1))


# ---------------------------------------------------------------- 训练
def class_sampler(rows: list) -> WeightedRandomSampler:
    """按 1/sqrt(类别张数) 抽: 小类(B4 / W4 / D)多见几次, 但不像 1/n 那样一张图一轮重复上百次(会背下来)。"""
    n = Counter(r["lab"] for r in rows)
    w = [1.0 / math.sqrt(n[r["lab"]]) for r in rows]
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(rows), replacement=True,
                                 generator=torch.Generator().manual_seed(0))


@torch.no_grad()
def predict(model, loader, device, n_views: int) -> tuple:
    model.eval()
    probs, idx = [], []
    for b in loader:
        xs = [t.to(device, non_blocking=True).float() for t in b[:n_views]]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = model(*xs).float().softmax(1)
        probs.append(p.cpu())
        idx.append(b[n_views + 1])
    return torch.cat(probs).numpy(), torch.cat(idx).numpy()


def balanced_acc(y: np.ndarray, pred: np.ndarray, min_n: int = 5) -> float:
    rec = [float((pred[y == c] == c).mean()) for c in range(len(CLASSES)) if (y == c).sum() >= min_n]
    return float(np.mean(rec)) if rec else 0.0


def train(a) -> Path:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if a.device == "cuda" and not torch.cuda.is_available():
        print("  ★ 没有可用的 GPU(torch.cuda.is_available() = False), 不在 CPU 上硬跑, 停", flush=True)
        sys.exit(2)
    device = torch.device(a.device)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}", flush=True)
    views = ("page", "head") if a.views == "both" else (a.views,)
    rows = load_rows(a.cache)
    tr = [r for r in rows if r["split"] == "train" and r["lab"] in CI]
    va = [r for r in rows if r["split"] == "val" and r["lab"] in CI]
    if a.limit:
        tr, va = tr[:a.limit], va[:max(200, a.limit // 5)]
    print(f"  视图 {views}   train {len(tr):,}  val {len(va):,}", flush=True)
    print("  train 各类: " + "  ".join(f"{k} {v:,}" for k, v in sorted(Counter(r['lab'] for r in tr).items(), key=lambda x: -x[1])), flush=True)
    # ★ Windows 上每个 DataLoader 子进程都重新 import torch(连 CUDA 的 dll), 吃提交内存(服务器出过 WinError 1455):
    #   只让训练的常驻, val 在主进程里读(几千张, 每轮多十几秒)
    dl_tr = DataLoader(PageSet(a.cache, tr, views, True), batch_size=a.batch, sampler=class_sampler(tr), drop_last=True,
                       num_workers=a.workers, pin_memory=device.type == "cuda", worker_init_fn=_worker_init,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(PageSet(a.cache, va, views, False), batch_size=a.batch, shuffle=False, num_workers=0)
    model = PageNet(views, width=a.width).to(device)
    print(f"  参数 {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    a.out.mkdir(parents=True, exist_ok=True)
    yv = np.array([CI[r["lab"]] for r in va])
    best, hist = -1.0, a.out / "history.jsonl"
    hist.write_text("", encoding="utf-8")
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in dl_tr:
            xs = [t.to(device, non_blocking=True).float() for t in b[:len(views)]]
            y = b[len(views)].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = lossf(model(*xs), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        p, idx = predict(model, dl_va, device, len(views))
        pred = p.argmax(1)
        y = yv[idx]
        acc, bacc = float((pred == y).mean()), balanced_acc(y, pred)
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "val_acc": acc, "val_bacc": bacc, "secs": time.time() - t0}
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"    第 {ep:2d} 轮  loss {rec['loss']:.4f}  val 准确率 {acc:.4f}  各类平均召回 {bacc:.4f}  {rec['secs']:.0f} 秒", flush=True)
        payload = {"model_state": model.state_dict(), "classes": list(CLASSES), "views": list(views), "width": a.width,
                   "page_hw": [PAGE_H, PAGE_W], "head_hw": [HEAD_H, HEAD_W], "head_frac": HEAD_FRAC,
                   "grid": {v: list(GRID[v]) for v in views}, "cache_jpeg_q": 95,
                   "preprocess": "cv2.imdecode(+FFD9) BGR; cv2.resize INTER_AREA; float 0-255, mean/std in model",
                   "epoch": ep, "val_acc": acc, "val_bacc": bacc}
        torch.save(payload, a.out / "last.pt")
        if bacc > best:
            best = bacc
            torch.save(payload, a.out / "best.pt")
    print(f"  最好 val 各类平均召回 {best:.4f} -> {a.out / 'best.pt'}", flush=True)
    return a.out / "best.pt"


# ---------------------------------------------------------------- 评估
def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ 二项(n, p)。对数里加, n 几万也不溢出。"""
    lp, lq = math.log(p), math.log1p(-p)
    c = math.lgamma(n + 1)
    terms = [c - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(terms)
    return math.exp(m) * sum(math.exp(t - m) for t in terms)


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """精确率的 95% 单边下界(Clopper-Pearson): n 张判出来对了 k 张, 真实精确率有 95% 把握不低于它。"""
    if n == 0 or k == 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def thresholds(p: np.ndarray, y: np.ndarray, target: float, min_n: int = 30) -> dict:
    """每类一个门槛: 判成这一类、把握 >= 门槛的那些里精确率 >= target(至少 min_n 张才算数); 定不出来的类不判(1.01)。
    y = -1 的(U / B0 / 不在 10 类里的)判成哪类都算那类的错。"""
    pred, conf = p.argmax(1), p.max(1)
    th = {}
    for c in range(len(CLASSES)):
        m = pred == c
        cand = 1.01
        for t in np.round(np.arange(0.50, 1.0, 0.01), 2).tolist() + [0.995, 0.999]:
            s = m & (conf >= t)
            if s.sum() >= min_n and (y[s] == c).mean() >= target:
                cand = t
                break
        th[CLASSES[c]] = cand
    return th


def apply_th(p: np.ndarray, th: dict) -> list:
    pred, conf = p.argmax(1), p.max(1)
    return [CLASSES[k] if conf[i] >= th[CLASSES[k]] else "U" for i, k in enumerate(pred)]


def evaluate(a, ckpt: Path) -> None:
    pay = torch.load(ckpt, map_location="cpu", weights_only=False)
    views = tuple(pay["views"])
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = PageNet(views, width=pay["width"]).to(device)
    model.load_state_dict(pay["model_state"])
    rows = load_rows(a.cache)
    ev = [r for r in rows if r["split"] in ("val", "test", "test_py")]
    dl = DataLoader(PageSet(a.cache, ev, views, False), batch_size=a.batch, shuffle=False, num_workers=0)
    p, idx = predict(model, dl, device, len(views))
    p = p[np.argsort(idx)]
    out = ckpt.parent / "eval"
    out.mkdir(exist_ok=True)
    sp = np.array([r["split"] for r in ev])
    lab = np.array([r["lab"] for r in ev])
    rule = np.array([r.get("rule", "") for r in ev])
    yv = np.array([CI.get(c, -1) for c in lab])
    mv = (sp == "val") & (rule == "")
    th = thresholds(p[mv], yv[mv], a.target)
    final = apply_th(p, th)
    comb = [rule[i] if rule[i] else final[i] for i in range(len(ev))]       # 规则先出手, 剩下的交给模型
    with (out / "preds.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "set", "split", "label", "truth", "rule", "pred", "p", "final", "combined", "top2", "p2"])
        for i, r in enumerate(ev):
            o = np.argsort(-p[i])
            w.writerow([r["id"], r["path"], r["set"], r["split"], r["lab"], r.get("truth", ""), rule[i], CLASSES[o[0]],
                        f"{p[i, o[0]]:.4f}", final[i], comb[i], CLASSES[o[1]], f"{p[i, o[1]]:.4f}"])
    L = [f"模型 {ckpt}  视图 {views}  第 {pay['epoch']} 轮  val 各类平均召回 {pay['val_bacc']:.4f}",
         f"门槛(规则定不下来的 val {int(mv.sum()):,} 张上, 每类精确率 >= {a.target}, 至少 30 张; 定不出的 1.01 = 不判): "
         + "  ".join(f"{k} {v}" for k, v in th.items()), ""]
    for name in ("test", "test_py"):
        m = sp == name
        if not m.any():
            continue
        res = m & (rule == "")
        L.append(f"==== {name}: {int(m.sum()):,} 张, 规则定不下来、交给模型的 {int(res.sum()):,} 张 ====")
        L.append("  [只看模型: 规则定不下来的那些]")
        L += report(lab[res], [CLASSES[k] for k in p[res].argmax(1)], [final[i] for i in np.where(res)[0]])
        L.append("  [规则 + 模型合起来: 全部]")
        L += report(lab[m], None, [comb[i] for i in np.where(m)[0]])
    (out / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    for name in ("test", "test_py"):
        # 规则定不下来、模型过了门槛还判错的(最自信的在前); 标签是 U / B0 却被判了一类的: 人眼看是标签错还是模型错
        res = np.where((sp == name) & (rule == ""))[0]
        errs = sorted([i for i in res if final[i] != "U" and final[i] != lab[i]], key=lambda i: -p[i].max())
        sheet(a.cache, [ev[i] for i in errs[:48]], [final[i] for i in errs[:48]], [p[i].max() for i in errs[:48]], out / f"{name}_errors.png")
        keep = [i for i in res if lab[i] not in CI and final[i] != "U"]
        sheet(a.cache, [ev[i] for i in keep[:48]], [final[i] for i in keep[:48]], [p[i].max() for i in keep[:48]], out / f"{name}_U_B0_judged.png")
    json.dump({"target": a.target, "thresholds": th, "ckpt": str(ckpt), "epoch": pay["epoch"]},
              open(out / "thresholds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def report(lab, argmax, final: list) -> list:
    """lab: 标签(可能有 U / B0 / P / 空); argmax: 模型最大的那类(None = 不看); final: 最后的判定(U = 不判)。
    判错 = 判了一类(不是 U)但和标签不一样, U / B0 / P 被判成任何一类都算错。"""
    lab = list(lab)
    n = len(lab)
    cov = [i for i in range(n) if final[i] != "U"]
    err = sum(final[i] != lab[i] for i in cov)
    ok = len(cov) - err
    ub = f"{1 - cp_lower(ok, len(cov)):.2%}" if cov else "-"
    L = [f"  {n:,} 张: 判出 {len(cov):,} 张(覆盖 {len(cov) / max(n, 1):.1%}), 判错 {err} 张(错误率 {err / max(len(cov), 1):.2%}, "
         f"95% 把握错误率不高于 {ub}), 记 U {n - len(cov):,} 张"]
    if argmax is not None:
        known = [i for i in range(n) if lab[i] in CI]
        if known:
            L.append(f"  其中 10 类的图 {len(known):,} 张, 只看最大那类的准确率 {sum(argmax[i] == lab[i] for i in known) / len(known):.4f}")
    oth = [i for i in range(n) if lab[i] not in CI]
    if oth:
        c = Counter(final[i] for i in oth)
        L.append(f"  标签是 U / B0 / P / 其它的 {len(oth):,} 张: 记 U {c.get('U', 0):,}, 判成了 "
                 + (" ".join(f"{k} {v}" for k, v in c.most_common() if k != "U") or "(没有)"))
    L.append("  每类: 标签张数 | 判出 / 判对 / 精确率 (95% 下界) / 召回")
    for c in sorted(set(lab) | {f for f in final if f != "U"}, key=lambda x: (x not in CLASSES, x)):
        nl = sum(1 for x in lab if x == c)
        pf = [i for i in range(n) if final[i] == c] if c != "U" else []      # U 是不判, 不算判出
        okc = sum(lab[i] == c for i in pf)
        if not nl and not pf:
            continue
        L.append(f"    {c:3} {nl:6,} | {len(pf):6,} / {okc:6,} / {okc / max(len(pf), 1):.4f} ({cp_lower(okc, len(pf)):.4f}) / {okc / max(nl, 1):.3f}")
    conf = defaultdict(Counter)
    for i in range(n):
        if final[i] != lab[i]:
            conf[lab[i]][final[i]] += 1
    L.append("  没判对的去向(标签 -> 判成): " + "; ".join(
        f"{k} -> " + ",".join(f"{kk} {vv}" for kk, vv in v.most_common(4)) for k, v in sorted(conf.items())))
    L.append("")
    return L


def sheet(cache: Path, rows: list, final: list, conf: list, dst: Path) -> None:
    """判错的格子: page 缩略图 + 一行 '标签 -> 判成 把握'(只用 ASCII, cv2 写不了中文)。"""
    if not rows:
        return
    tiles = []
    for r, f, c in zip(rows, final, conf):
        img = cv2.imdecode(np.fromfile(f"{cache}/page/{r['id']}.jpg", np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (216, 480), interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 44, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, f"#{r['id']} {r['lab']}->{f} {c:.3f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
        cv2.putText(img, r["gid"][-12:], (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        tiles.append(img)
    while len(tiles) % 8:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)])
    cv2.imencode(".png", grid)[1].tofile(str(dst))


class _OnnxWrap(nn.Module):
    """导 ONNX 用: 输入 uint8 NHWC BGR(OpenCvSharp 的 Mat 数据直接喂, 不用在 C# 里归一化), 输出各类概率。"""

    def __init__(self, model: PageNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.permute(0, 3, 1, 2).float()).softmax(1)


def export(a) -> None:
    """单视图模型 -> ONNX(opset 13, 旧导出器, 固定 [1,H,W,3] uint8)+ 对拍 + .contract.json。
    ★ C# ORT 1.23.2 最多认 IR 11, 导出来超了就停; 对拍用缓存里的真缩略图和随机图, 概率差 < 1e-4、最大类一样。"""
    import hashlib
    import onnx
    import onnxruntime as ort
    pay = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    views = tuple(pay["views"])
    if len(views) != 1:
        print("  ★ 只支持单视图模型导 ONNX", flush=True)
        sys.exit(2)
    v = views[0]
    hgt, wid = SIZE[v]
    model = PageNet(views, width=pay["width"])
    model.load_state_dict(pay["model_state"])
    model.eval()
    wrap = _OnnxWrap(model).eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, hgt, wid, 3, dtype=torch.uint8)
    torch.onnx.export(wrap, (dummy,), str(a.out), input_names=["image"], output_names=["probabilities"],
                      opset_version=13, do_constant_folding=True, dynamo=False)
    om = onnx.load(str(a.out))
    onnx.checker.check_model(om)
    print(f"  ONNX {a.out}  IR {om.ir_version}  opset {om.opset_import[0].version}  "
          f"算子 {sorted({n.op_type for n in om.graph.node})}", flush=True)
    if om.ir_version > 10:
        print("  ★ IR 版本 > 10, C# ORT 1.23.2 读不了, 停", flush=True)
        sys.exit(3)
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
    sess = ort.InferenceSession(str(a.out), so, providers=["CPUExecutionProvider"])
    rows = [r for r in load_rows(a.cache) if r["split"] in ("test", "test_py")] if a.cache else []
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), replace=False)] if rows else []
    imgs = [cv2.imdecode(np.fromfile(f"{a.cache}/{v}/{r['id']}.jpg", np.uint8), cv2.IMREAD_COLOR) for r in rows]
    imgs += [rng.integers(0, 256, (hgt, wid, 3), dtype=np.uint8) for _ in range(8)]
    worst, same = 0.0, 0
    with torch.no_grad():
        for img in imgs:
            x = np.ascontiguousarray(img)[None]
            pt = wrap(torch.from_numpy(x)).numpy()[0]
            po = sess.run(None, {"image": x})[0][0]
            worst = max(worst, float(np.abs(pt - po).max()))
            same += int(pt.argmax() == po.argmax())
    print(f"  对拍 {len(imgs)} 张(缓存缩略图 {len(rows)} + 随机 8): 概率最大差 {worst:.2e}, 最大类一样 {same}/{len(imgs)}", flush=True)
    if worst > 1e-4 or same != len(imgs):
        print("  ★ torch 和 ONNX 对不上, 停", flush=True)
        sys.exit(4)
    th = json.load(open(a.thresholds, encoding="utf-8")) if a.thresholds else {}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    contract = {
        "kind": "page_type_head_v1",
        "input": {"name": "image", "shape": [1, hgt, wid, 3], "dtype": "uint8", "layout": "NHWC", "color": "BGR",
                  "preprocess": f"cv2.imdecode(IMREAD_COLOR, 截断的 JPEG 补 FFD9 再解); 取上面 round({HEAD_FRAC} * H) 行; "
                                f"cv2.resize 到 {wid}x{hgt}(宽 x 高), INTER_AREA。归一化在图里, 不用自己做"},
        "output": {"name": "probabilities", "shape": [1, len(CLASSES)], "labels": list(CLASSES)},
        "postprocess": th,
        "checkpoint_sha256": sha(a.ckpt), "onnx_sha256": sha(a.out),
        "ir_version": om.ir_version, "opset": om.opset_import[0].version,
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "parity": {"images": len(imgs), "max_abs_diff": worst},
    }
    Path(str(a.out) + ".contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {a.out}.contract.json", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="整页分类小 CNN")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--views", choices=["both", "page", "head"], default="both")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--width", type=float, default=1.0)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--device", default="cuda")
    t.add_argument("--target", type=float, default=0.995)
    t.add_argument("--limit", type=int, default=0, help="只用前 N 张训练(本地试跑)")
    e = sub.add_parser("eval")
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--device", default="cuda")
    e.add_argument("--target", type=float, default=0.995)
    x = sub.add_parser("export")
    x.add_argument("--ckpt", type=Path, required=True)
    x.add_argument("--out", type=Path, required=True, help="输出 .onnx")
    x.add_argument("--cache", type=Path, default=None, help="缩略图缓存(对拍用真图)")
    x.add_argument("--thresholds", type=Path, default=None, help="门槛 json(写进 contract)")
    x.add_argument("--n", type=int, default=64, help="对拍用几张缓存缩略图")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a)
        return
    if a.cmd == "train":
        best = train(a)
        evaluate(a, best)
    else:
        evaluate(a, a.ckpt)


if __name__ == "__main__":
    main()
