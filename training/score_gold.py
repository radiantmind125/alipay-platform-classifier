r"""金标准考卷上给模型打分: 跟人工改过的标准答案比, 不是跟老师 OCR 的标签比。

输入
----
<gold>/_gold_candidates.csv     export_gold.py 导出的: id, file, src_path, dir, label(老师读的)
<gold>/_gold.csv                人工改好的: id, gold   (gold 为空 = 没定; NO_TEXT = 裁图里没有字; UNREADABLE = 看不清)
<gold>/img/                     裁图
--onnx 一个或几个模型(同目录要有 charset.txt)

输出
----
每个模型(外加"老师标签"这一行, 看标签本身错多少), 按目录分开:
    整段全对(严)   全半角统一、去空格之后一字不差
    整段全对(宽)   再去掉箭头 > 和 +/十 这类图标字的差别
    字准确率       1 - 编辑距离 / 标准答案字数(宽口径)
NO_TEXT / UNREADABLE 的段不计分, 单独报个数。
--dump 把每段每个模型读出来的存成 csv, 方便逐条看。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_rec import IMG_H, Charset, edit_distance, prep  # noqa: E402

SKIP = {"NO_TEXT", "UNREADABLE"}


def strict(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(c for c in s if not c.isspace())


def lenient(s: str) -> str:
    """再去掉不算错的写法差别: 箭头图标、+ 图标(结尾的)、引号样式、句尾标点、时间后面被读成"支"的支付宝图标。"""
    s = strict(s).replace("十", "+")
    for a in "“”„‟": s = s.replace(a, '"')
    for a in "‘’": s = s.replace(a, "'")
    s = "".join(c for c in s if c not in "><›‹〉〈")
    s = re.sub(r"^(\d{1,2}:\d{2})支$", r"\1", s)
    return s.rstrip("+,，。")


def run_model(onnx: Path, files: list[Path]) -> list[str]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    chars = [c for c in (onnx.parent / "charset.txt").read_text(encoding="utf-8").split("\n") if c]
    cs = Charset(chars)
    out = []
    for p in files:
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            out.append("")
            continue
        x = prep(im)[None].astype(np.float32)
        logits = sess.run(None, {"x": x})[0]
        ids = logits.argmax(2)[0]
        out.append(cs.decode(ids[: max(1, x.shape[3] // 4)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, nargs="*", default=[])
    ap.add_argument("--dump", type=Path, default=None)
    a = ap.parse_args()

    cand = list(csv.DictReader((a.gold / "_gold_candidates.csv").open(encoding="utf-8-sig")))
    gp = a.gold / "_gold.csv"
    gold = {}
    if gp.exists():
        gold = {r["id"]: r["gold"] for r in csv.DictReader(gp.open(encoding="utf-8-sig"))}
    files = [a.gold / "img" / r["file"] for r in cand]
    preds = {"老师标签": [r["label"] for r in cand]}
    for m in a.onnx:
        name = m.parent.name
        print(f"  跑 {name} ...", flush=True)
        preds[name] = run_model(m, files)

    if a.dump:
        with a.dump.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "dir", "gold"] + list(preds))
            for i, r in enumerate(cand):
                w.writerow([r["id"], r["dir"], gold.get(r["id"], "")] + [preds[k][i] for k in preds])
        print(f"  逐条 -> {a.dump}")
    if not gold:
        print("  还没有 _gold.csv, 只出逐条结果")
        return

    dirs = sorted({r["dir"] for r in cand})
    print("=" * 78)
    print("  GOLD SCORE  (vs 人工标准答案)")
    print("=" * 78)
    for d in dirs:
        idx = [i for i, r in enumerate(cand) if r["dir"] == d and gold.get(r["id"], "") not in ("", *SKIP)]
        nskip = sum(1 for r in cand if r["dir"] == d and gold.get(r["id"], "") in SKIP)
        nnone = sum(1 for r in cand if r["dir"] == d and gold.get(r["id"], "") == "")
        print(f"  {d}:  计分 {len(idx)} 段   (裁图没字/看不清 {nskip}, 还没定 {nnone})")
        print(f"    {'':<14}{'全对(严)':>10}{'全对(宽)':>10}{'字准确率':>10}")
        for k, pr in preds.items():
            ex_s = sum(1 for i in idx if strict(pr[i]) == strict(gold[cand[i]["id"]]))
            ex_l = sum(1 for i in idx if lenient(pr[i]) == lenient(gold[cand[i]["id"]]))
            ed = sum(edit_distance(lenient(pr[i]), lenient(gold[cand[i]["id"]])) for i in idx)
            tot = sum(max(1, len(lenient(gold[cand[i]["id"]]))) for i in idx)
            n = max(1, len(idx))
            print(f"    {k:<14}{ex_s / n:>10.1%}{ex_l / n:>10.1%}{1 - ed / tot:>10.1%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
