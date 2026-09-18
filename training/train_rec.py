"""训一个能**无视拼音**读中文的识别器 (CRNN + CTC, PyTorch)。

要解决的问题
------------
实测过: 拼音页整页送现成 OCR, 每张只读出 2.0 个固定字段标签, 不带拼音的读出 9.0 个。
而且拆开看过, 坏的**不是检测是识别**:

    检测覆盖率  97%     检测器**找得到**那些行
    识别一致率  61%     找到了却认不对    账户余额 -> 芦荼额

所以要训的是**识别器**, 让它见过"上面压着一行拼音"的版面, 学会只读下面那行汉字。

输入是什么, 别搞反
------------------
★★★★★ 图用 `seg_input/` —— **带拼音**那套。
   文字来自 `seg_label/`(不含拼音)经现成 OCR 读出来的。
   反过来就训成一个普通 OCR 了, 白训。

为什么是"段"不是"行"
--------------------
行图 1080x43, 宽高比 26:1, 而且 69% 的行里有两段以上(左字段名右取值)。
整行压进 320 宽要挤掉 75%, 字全糊。切成段之后 3.8:1, 正常了。
见 split_segments.py。

用法
----
    python train_rec.py --data D:\\alipay-ai-data\\pinyin-pairs --epochs 20
    python train_rec.py --data ... --export rec.onnx        # 训完导出
    python train_rec.py --data ... --limit 2000 --epochs 1  # 冒烟跑一遍
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

IMG_H = 48           # 和常见识别器一致
MAX_W = 640          # 实测 95% 的段缩到高 48 之后不超过 640
MIN_W = 16
DOWN_W = 4           # 网络在宽度方向下采样的倍数


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
class Charset:
    """字表。0 号留给 CTC 的 blank, 真实字符从 1 开始。"""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}
        self.itos = {i + 1: c for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.chars) + 1        # +1 是 blank

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        """CTC 贪心解码: 先去连续重复, 再去 blank。**顺序不能反。**"""
        out, prev = [], -1
        for i in ids:
            i = int(i)
            if i != prev and i != 0:
                out.append(self.itos.get(i, ""))
            prev = i
        return "".join(out)


def load_list(p: Path) -> list[tuple[str, str]]:
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if "\t" not in ln:
            continue
        img, txt = ln.split("\t", 1)
        if txt.strip():
            rows.append((img, txt))
    return rows


def prep(img: np.ndarray) -> np.ndarray:
    """缩到高 IMG_H, 保持长宽比, 宽度夹在 [MIN_W, MAX_W]。"""
    h, w = img.shape[:2]
    nw = max(MIN_W, min(MAX_W, int(round(w * IMG_H / max(1, h)))))
    im = cv2.resize(img, (nw, IMG_H), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32) / 127.5 - 1.0
    return im.transpose(2, 0, 1)          # HWC -> CHW


def image_widths(rows, tag: str) -> list[int]:
    """把每张图缩到高 48 之后会有多宽 —— 只读文件头, 不解码整张图。"""
    from PIL import Image
    out = []
    for i, (path, _t) in enumerate(rows, 1):
        try:
            with Image.open(path) as im:
                w, h = im.size
            out.append(max(MIN_W, min(MAX_W, int(round(w * IMG_H / max(1, h))))))
        except Exception:                      # noqa: BLE001
            out.append(MIN_W)
        if i % 50000 == 0:
            print(f"    量 {tag} 宽度 {i:,}/{len(rows):,}", flush=True)
    return out


class BucketSampler(Sampler):
    """按宽度相近的凑一批。

    ★★★★★ 为什么非要这样: 段的宽度差很多(中位 183, 90% 分位 508, 最大 640),
       随机凑一批的话整批要补到最宽的那张那么宽 —— 实测**补的空白占了张量的 62%**,
       算力有六成花在补的空白上。按宽度分桶之后降到 9%, **省下约 61% 的算力**。

    ★ 但不能直接按宽度全局排序, 那样每一批的内容就固定了, 等于没打乱。
      办法是: 先整体打乱 -> 切成大块 -> **块内**按宽度排 -> 切成批 -> 再把批的顺序打乱。
      这样既有随机性, 同一批里宽度又接近。
    """

    def __init__(self, widths, batch_size, shuffle=True, pool_batches=64, drop_last=False):
        self.widths = widths
        self.bs = batch_size
        self.shuffle = shuffle
        self.pool = batch_size * pool_batches
        self.drop_last = drop_last

    def __iter__(self):
        idx = list(range(len(self.widths)))
        if self.shuffle:
            random.shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.pool):
            part = sorted(idx[i:i + self.pool], key=lambda j: self.widths[j])
            for k in range(0, len(part), self.bs):
                b = part[k:k + self.bs]
                if len(b) == self.bs or not self.drop_last:
                    batches.append(b)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.widths) // self.bs
        return (len(self.widths) + self.bs - 1) // self.bs


class SegDataset(Dataset):
    def __init__(self, rows, cs: Charset, train: bool):
        self.rows, self.cs, self.train = rows, cs, train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        path, txt = self.rows[i]
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((IMG_H, MIN_W, 3), np.uint8)
        x = prep(img)
        y = self.cs.encode(txt)
        if not y:
            y = [0]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), txt


def collate(batch):
    """按**这一批里最宽的**补齐, 不是补到 MAX_W —— 大部分段很窄, 补满纯浪费。

    ★★ 补的内容是**最后一列的复制**, 不是 0。
       0 在归一化之后等于中灰(像素 127), 会在图右边接一大块灰 ——
       CTC 那边虽然按真实长度忽略了这些帧, 但 **BatchNorm 的统计量是算进去的**。
       复制边缘列就没有这条人为的边。
    """
    xs, ys, txts = zip(*batch)
    W = max(x.shape[2] for x in xs)
    W = max(W, DOWN_W * 4)
    out = torch.empty(len(xs), 3, IMG_H, W)
    widths = []
    for i, x in enumerate(xs):
        w = x.shape[2]
        out[i, :, :, :w] = x
        if w < W:
            out[i, :, :, w:] = x[:, :, -1:]      # 拿最后一列铺满
        widths.append(w)
    tgt = torch.cat(ys)
    tgt_len = torch.tensor([len(y) for y in ys], dtype=torch.long)
    # ★ 输入长度按**这张图自己的**宽度算, 不是按补齐后的宽度 ——
    #   按补齐后的算, 等于告诉 CTC 后面那片空白也是有效时间步。
    inp_len = torch.tensor([max(1, w // DOWN_W) for w in widths], dtype=torch.long)
    return out, tgt, inp_len, tgt_len, txts


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class CRNN(nn.Module):
    """CNN 把高压成 1, 宽压成 1/4, 再上双向 LSTM, 最后 CTC。"""

    def __init__(self, n_class: int):
        super().__init__()

        def blk(i, o, pool):
            layers = [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True)]
            if pool:
                layers.append(nn.MaxPool2d(pool, pool))
            return layers

        self.cnn = nn.Sequential(
            *blk(3, 64, (2, 2)),        # 48x W   -> 24 x W/2
            *blk(64, 128, (2, 2)),      #         -> 12 x W/4
            *blk(128, 256, None),
            *blk(256, 256, (2, 1)),     #         ->  6 x W/4
            *blk(256, 512, None),
            *blk(512, 512, (2, 1)),     #         ->  3 x W/4
            nn.Conv2d(512, 512, (3, 1)),  #       ->  1 x W/4
            nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        f = self.cnn(x)                 # B, 512, 1, T
        f = f.squeeze(2).permute(0, 2, 1)   # B, T, 512
        f, _ = self.rnn(f)
        return self.fc(f)               # B, T, n_class


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, loader, cs: Charset, dev, max_batches: int = 0):
    """返回 (字准确率, 整条全对率, 抽几条看看)。

    ★ 字准确率用**编辑距离**算, 不是逐位置比 —— 少一个字会让后面全部错位,
      逐位置比会把"只错一个字"算成"全错"。
    """
    model.eval()
    ed = tot = exact = n = 0
    samples = []
    for bi, (x, _tgt, inp_len, _tl, txts) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        logits = model(x.to(dev))
        pred_ids = logits.argmax(2).cpu().numpy()
        for k, gt in enumerate(txts):
            T = int(inp_len[k])
            hyp = cs.decode(pred_ids[k][:T])
            ed += edit_distance(hyp, gt)
            tot += max(1, len(gt))
            exact += int(hyp == gt)
            n += 1
            if len(samples) < 8:
                samples.append((gt, hyp))
    model.train()
    return 1 - ed / max(1, tot), exact / max(1, n), samples


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="放 train.txt / val.txt / charset.txt 的目录")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只拿这么多条, 冒烟用")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--export", type=Path, default=None, help="训完导出 onnx")
    ap.add_argument("--overfit", action="store_true",
                    help="★ 调试用: 拿训练集当验证集。跑不到接近 100% 就说明有 bug, "
                         "不是数据不够 —— 几十条样本任何正常模型都该背得下来")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = a.out or a.data
    out_dir.mkdir(parents=True, exist_ok=True)

    chars = [c for c in (a.data / "charset.txt").read_text(encoding="utf-8").split("\n") if c]
    cs = Charset(chars)
    tr = load_list(a.data / "train.txt")
    va = load_list(a.data / "val.txt")
    if a.limit:
        tr, va = tr[: a.limit], va[: max(64, a.limit // 10)]
    if a.overfit:
        # 背书测试: 验证集就是训练集。这不是在评估模型好坏,
        # 是在验**这套代码本身通不通** —— 编码解码、CTC 的长度、时间步够不够。
        va = tr

    print("=" * 60)
    print(f"  设备 {dev}    字表 {len(cs)} (含 blank)")
    print(f"  训练 {len(tr):,} 条   验证 {len(va):,} 条")
    print("=" * 60)

    print("  量一遍图宽好按宽度分桶(只读文件头, 不解码)...")
    w_tr = image_widths(tr, "train")
    w_va = image_widths(va, "val")
    dl_tr = DataLoader(SegDataset(tr, cs, True), collate_fn=collate,
                       num_workers=a.workers,
                       batch_sampler=BucketSampler(w_tr, a.batch, True, drop_last=True))
    dl_va = DataLoader(SegDataset(va, cs, False), collate_fn=collate,
                       num_workers=a.workers,
                       batch_sampler=BucketSampler(w_va, a.batch, False))

    model = CRNN(len(cs)).to(dev)
    # ★ zero_infinity: 有的样本标签比时间步还长(超宽的段被夹到 MAX_W),
    #   那一条的 loss 会是 inf。不置零会把整个 batch 的梯度污染掉。
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, a.epochs * len(dl_tr)))

    best = -1.0
    for ep in range(1, a.epochs + 1):
        t0, run, seen = time.time(), 0.0, 0
        for i, (x, tgt, inp_len, tgt_len, _t) in enumerate(dl_tr, 1):
            logits = model(x.to(dev))
            # CTCLoss 要 (T, B, C) 而且要 log_softmax
            lp = logits.permute(1, 0, 2).log_softmax(2)
            T = lp.shape[0]
            loss = crit(lp, tgt.to(dev), inp_len.clamp(max=T), tgt_len)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += float(loss) * x.shape[0]
            seen += x.shape[0]
            if i % 200 == 0:
                print(f"    ep{ep} {i}/{len(dl_tr)}  loss {run/max(1,seen):.3f}",
                      flush=True)

        acc, exact, samples = evaluate(model, dl_va, cs, dev)
        print(f"  ep{ep:>3}  loss {run/max(1,seen):6.3f}   "
              f"字准确率 {acc:6.2%}   整条全对 {exact:6.2%}   "
              f"{time.time()-t0:5.0f}s")
        for gt, hyp in samples[:3]:
            mark = "  " if gt == hyp else "x "
            print(f"      {mark}真: {gt[:26]:<28} 出: {hyp[:26]}")
        # ★ 指标落盘。控制台的中文在服务器上会显示成乱码, 贴回来读不了;
        #   而且窗口一关就什么都没了。落一份**纯 ASCII**的, 事后还查得到。
        with (out_dir / "train_log.txt").open("a", encoding="utf-8") as f:
            f.write("epoch {}\tloss {:.4f}\tchar_acc {:.4f}\texact {:.4f}"
                    "\tsecs {:.0f}\n".format(ep, run / max(1, seen), acc, exact,
                                             time.time() - t0))
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "chars": chars},
                       out_dir / "rec_best.pt")
            print(f"      ★ 存了 {out_dir / 'rec_best.pt'}  (字准确率 {acc:.2%})")

    if a.export:
        ck = torch.load(out_dir / "rec_best.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        model.eval().cpu()
        dummy = torch.randn(1, 3, IMG_H, 320)
        # ★ torch 2.13 默认走 dynamo 那条新路, 要额外装 onnxscript。
        #   这里显式用老的 TorchScript 导出, 少一个依赖; 新路装了也能用。
        kw = dict(input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "B", 3: "W"}, "logits": {0: "B", 1: "T"}},
                  opset_version=13)
        try:
            torch.onnx.export(model, dummy, str(a.export), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, dummy, str(a.export), **kw)
        (a.export.parent / "charset.txt").write_text("\n".join(chars) + "\n",
                                                     encoding="utf-8")
        print(f"\n★ 导出 {a.export}  (字表另存在旁边, 推理时要用同一份)")


if __name__ == "__main__":
    main()
