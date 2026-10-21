"""按页面类型分开训识别器, 值不值? (经理 10/12: 先分类, 再用这一类自己的模型)

做法: 从现在最好的模型(n1009h)接着训(微调), 一组只喂某一类页面的段(蓝图 / 转账成功 / 到账成功 / 支付成功),
另一组喂全部的段(对照: 同样多训几轮, 分得清是"分开训"有用还是"多训了几轮"有用)。最后在两份标准答案上按页面类型比。

三步:
  sets     从一个训练集(train.txt / val.txt / charset.txt)按页面类型挑出子集, 每组一个目录, 字表原样拷(微调要一字不差)。
           段文件名里带 GWCZ 编号 -> 查 page_type 的标签 -> 类别。验证集还是原来那批原图, 一张不多一张不少。
  evalval  几个模型在几个验证集上读一遍(老师标签, 段多), 按页面类型出表 + 每段的读法(配对比)。
  gold     在标准答案上比(本地, 答案不出这台机器): 按页面类型, 加"分流"组合(某类的段交给那一类的模型), 配对符号检验。

用法:
  python training/type_exp.py sets --data E:\\alipay-ai-data\\train-n1009h --labels page_type.csv [...] --out-root E:\\alipay-ai-data\\type-exp
                                   --set blue=B0,B1,B2,B3,B4 --set B1=B1 --set B2=B2 --set B3=B3
  python training/type_exp.py evalval --val 某组\\val.txt --labels page_type.csv [...] --model h=路径.onnx all=... --out 表.csv
  python training/type_exp.py gold --gold 答案目录 [...] --labels ... --model h=... --route 分流名:B1=模型,B2=模型,B3=模型 --base h
                                   [--compare 甲:乙 ...] [--dump 逐段.csv]
  python training/type_exp.py valreport --csv evalval 出的逐段表 --route ... --base h --compare 甲:乙   (老师标签上同样的比法)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_rec import edit_distance, load_list  # noqa: E402

GW = re.compile(r"GWCZ\d{15,}", re.I)


def load_labels(files: list) -> dict:
    """GWCZ 编号 -> 页面类别(page_type.csv; 同一个编号在几个文件里都有, 前面的文件算数)。"""
    lab = {}
    for fn in files:
        with open(fn, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                m = GW.search(r["path"])
                if m:
                    lab.setdefault(m.group().upper(), r["cls"])
    return lab


def seg_type(path: str, lab: dict) -> tuple:
    m = GW.search(Path(path).name)
    g = m.group().upper() if m else ""
    return g, lab.get(g, "?")


def sets(a) -> None:
    lab = load_labels(a.labels)
    print(f"  页面标签 {len(lab):,} 张", flush=True)
    rows = {k: load_list(a.data / f"{k}.txt") for k in ("train", "val")}
    typed = {}
    for k, rs in rows.items():
        t = [(p, txt) + seg_type(p, lab) for p, txt in rs]
        typed[k] = t
        c = Counter(x[3] for x in t)
        pg = defaultdict(set)
        for x in t:
            pg[x[3]].add(x[2])
        print(f"  {k}: {len(t):,} 段, 有页面类别的 {sum(v for c_, v in c.items() if c_ != '?'):,} "
              f"({1 - c['?'] / max(1, len(t)):.1%})", flush=True)
        for c_, v in sorted(c.items(), key=lambda z: -z[1]):
            print(f"      {c_:3} {v:>9,} 段  {len(pg[c_]):>7,} 张图", flush=True)
    # ★ 训练和验证的原图不能有重叠(上游按原图划的; 这里再核一遍, 防止接错数据)
    both = {x[2] for x in typed["train"]} & {x[2] for x in typed["val"]}
    both.discard("")
    if both:
        print(f"  ★ 训练和验证有 {len(both)} 张同一原图, 不做", flush=True)
        sys.exit(3)
    # ★ 先全部核完(够不够多、目录在不在)再一起写, 不留半截目录
    plan = []
    for spec in a.set:
        name, cls = spec.split("=", 1)
        want = set(cls.split(","))
        d = a.out_root / f"data-{name}"
        sel = {k: [x for x in typed[k] if x[3] in want] for k in ("train", "val")}
        print(f"  {name:6} = {sorted(want)}  训练 {len(sel['train']):,} 段  验证 {len(sel['val']):,} 段  -> {d}", flush=True)
        if d.exists():
            print(f"  ★ {d} 已经有了, 不覆盖", flush=True)
            sys.exit(2)
        if len(sel["train"]) < a.min_train or len(sel["val"]) < a.min_val:
            print(f"  ★ {name} 太少了(训练至少 {a.min_train}, 验证至少 {a.min_val}), 不做", flush=True)
            sys.exit(4)
        plan.append((d, sel))
    for d, sel in plan:
        d.mkdir(parents=True)
        for k in ("train", "val"):
            with (d / f"{k}.txt").open("w", encoding="utf-8", newline="\n") as f:
                for p, txt, _g, _c in sel[k]:
                    f.write(f"{p}\t{txt}\n")
        shutil.copyfile(a.data / "charset.txt", d / "charset.txt")
    print("SETS_OK", flush=True)


def _models(specs: list) -> list:
    out = []
    for s in specs:
        k, p = s.split("=", 1)
        out.append((k, Path(p)))
    return out


def evalval(a) -> None:
    from score_gold import lenient, run_model
    lab = load_labels(a.labels)
    rows = []
    for v in a.val:
        rows += load_list(v)
    seen, uniq = set(), []
    for p, t in rows:
        if p not in seen:
            seen.add(p)
            uniq.append((p, t))
    files = [Path(p) for p, _ in uniq]
    types = [seg_type(p, lab)[1] for p, _ in uniq]
    preds = {}
    for k, m in _models(a.model):
        print(f"  读 {k} ({len(files):,} 段) ...", flush=True)
        preds[k] = run_model(m, files)
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "cls", "label"] + list(preds))
        for i, (p, t) in enumerate(uniq):
            w.writerow([p, types[i], t] + [preds[k][i] for k in preds])
    print(f"  逐段 -> {a.out}", flush=True)
    print(f"  {'类别':<6}{'段':>7}  " + "  ".join(f"{k:>16}" for k in preds), flush=True)
    for c in sorted(set(types)):
        idx = [i for i, t in enumerate(types) if t == c]
        if len(idx) < 30:
            continue
        cells = []
        for k in preds:
            ex = sum(1 for i in idx if lenient(preds[k][i]) == lenient(uniq[i][1])) / len(idx)
            ed = sum(edit_distance(lenient(preds[k][i]), lenient(uniq[i][1])) for i in idx)
            tot = sum(max(1, len(lenient(uniq[i][1]))) for i in idx)
            cells.append(f"{ex:7.2%} {1 - ed / tot:7.2%}")
        print(f"  {c:<6}{len(idx):>7,}  " + "  ".join(f"{x:>16}" for x in cells), flush=True)
    print("  (每格: 整段全对 宽口径 / 字准确率; 对的是老师标签, 不是标准答案)", flush=True)


def sign_p(win: int, lose: int) -> float:
    """配对符号检验, 双侧精确 p(平的不算)。"""
    n = win + lose
    if n == 0:
        return 1.0
    k = min(win, lose)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def report(cls: list, rel: list, truth: list, preds: dict, routes: list, base: str, compares: list, title: str) -> None:
    """按页面类型出表 + 分流组合 + 配对符号检验。gold(标准答案)和 valreport(老师标签)共用。
    分流: 名字:B1=模型,B2=模型,... 某类的段交给那一类的模型, 别的类交给 base。preds 里会加上分流的结果。"""
    from score_gold import lenient
    for spec in routes:
        name, rest = spec.split(":", 1)
        rt = dict(x.split("=", 1) for x in rest.split(","))
        preds[name] = [preds[rt.get(c, base)][i] for i, c in enumerate(cls)]
    ok = {k: [lenient(p) == lenient(t) for p, t in zip(v, truth)] for k, v in preds.items()}
    groups = [("B1", lambda i: cls[i] == "B1"), ("B2", lambda i: cls[i] == "B2"), ("B3", lambda i: cls[i] == "B3"),
              ("蓝图B1-3", lambda i: cls[i] in ("B1", "B2", "B3")),
              ("蓝图B1-3 KEY", lambda i: cls[i] in ("B1", "B2", "B3") and rel[i] == "KEY"),
              ("白图", lambda i: not cls[i].startswith("B")), ("全部", lambda i: True)]
    print("=" * 100)
    print(f"  {title}")
    print(f"  {'':<14}{'段':>6}  " + "".join(f"{k:>14}" for k in preds))
    for gname, f in groups:
        idx = [i for i in range(len(cls)) if f(i)]
        if idx:
            print(f"  {gname:<14}{len(idx):>6}  " + "".join(f"{sum(ok[k][i] for i in idx) / len(idx):>14.1%}" for k in preds))
    print("-" * 100)
    print("  配对比(赢:输, 双侧符号检验 p)  只算两边一对一错的段")
    for pair in compares:
        x, y = pair.split(":", 1)
        for gname, f in groups:
            idx = [i for i in range(len(cls)) if f(i)]
            win = sum(1 for i in idx if ok[x][i] and not ok[y][i])
            lose = sum(1 for i in idx if ok[y][i] and not ok[x][i])
            if win + lose:
                print(f"    {x} vs {y}  {gname:<14} {win:>3}:{lose:<3}  p {sign_p(win, lose):.3f}")


def valreport(a) -> None:
    """evalval 出的逐段表(老师标签)上做和 gold 一样的比法。段多(蓝图验证集 3,600 段), 但对的是老师标签, 不是标准答案。"""
    with a.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    names = [k for k in rows[0] if k not in ("path", "cls", "label")]
    preds = {k: [r[k] for r in rows] for k in names}
    report([r["cls"] for r in rows], [""] * len(rows), [r["label"] for r in rows], preds, a.route, a.base, a.compare,
           f"验证集(老师标签) {a.csv.name}: {len(rows):,} 段   (整段全对, 宽口径)")


def gold(a) -> None:
    from score_gold import SKIP, run_model
    lab = load_labels(a.labels)
    segs = []                       # (答案目录, id, 文件, 页面类别, 答案, 重要性)
    for g in a.gold:
        cand = list(csv.DictReader((g / "_gold_candidates.csv").open(encoding="utf-8-sig")))
        ans = {r["id"]: r["gold"] for r in csv.DictReader((g / "_gold.csv").open(encoding="utf-8-sig"))}
        rel = {}
        if (g / "_relevance.csv").exists():
            rel = {r["id"]: r["rel"] for r in csv.DictReader((g / "_relevance.csv").open(encoding="utf-8-sig"))}
        for r in cand:
            if ans.get(r["id"], "") in ("", *SKIP):
                continue
            segs.append((g.name, r["id"], g / "img" / r["file"], seg_type(r["src_path"], lab)[1], ans[r["id"]], rel.get(r["id"], "")))
    files = [s[2] for s in segs]
    preds = {}
    for k, m in _models(a.model):
        print(f"  读 {k} ...", flush=True)
        preds[k] = run_model(m, files)
    report([s[3] for s in segs], [s[5] for s in segs], [s[4] for s in segs], preds, a.route, a.base, a.compare,
           f"标准答案 {', '.join(g.name for g in a.gold)}: 计分 {len(segs):,} 段   (整段全对, 宽口径)")
    if a.dump:
        with a.dump.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["set", "id", "cls", "rel", "gold"] + list(preds))
            for i, s in enumerate(segs):
                w.writerow([s[0], s[1], s[3], s[5], s[4]] + [preds[k][i] for k in preds])
        print(f"  逐段 -> {a.dump}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="按页面类型分开训识别器的实验")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--labels", nargs="+", required=True)
    s.add_argument("--out-root", type=Path, required=True)
    s.add_argument("--set", action="append", required=True, help="名字=类别,类别  比如 blue=B0,B1,B2,B3,B4")
    s.add_argument("--min-train", type=int, default=1000)
    s.add_argument("--min-val", type=int, default=100)
    e = sub.add_parser("evalval")
    e.add_argument("--val", type=Path, nargs="+", required=True)
    e.add_argument("--labels", nargs="+", required=True)
    e.add_argument("--model", nargs="+", required=True, help="名字=onnx 路径(同目录要有 charset.txt)")
    e.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gold")
    g.add_argument("--gold", type=Path, nargs="+", required=True)
    g.add_argument("--labels", nargs="+", required=True)
    g.add_argument("--model", nargs="+", required=True)
    g.add_argument("--route", nargs="*", default=[], help="分流名:B1=模型名,B2=模型名,B3=模型名(别的类用 --base)")
    g.add_argument("--base", default="h")
    g.add_argument("--compare", nargs="*", default=[], help="甲:乙 配对比")
    g.add_argument("--dump", type=Path, default=None)
    v = sub.add_parser("valreport")
    v.add_argument("--csv", type=Path, required=True, help="evalval 出的逐段表")
    v.add_argument("--route", nargs="*", default=[])
    v.add_argument("--base", default="h")
    v.add_argument("--compare", nargs="*", default=[])
    a = ap.parse_args()
    {"sets": sets, "evalval": evalval, "gold": gold, "valreport": valreport}[a.cmd](a)


if __name__ == "__main__":
    main()
