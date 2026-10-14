r"""按页面类型和位置修标签: 每一类页面上固定的字(回首页 / 付款方式 / 收款方 / 立即通知收款人 ...)在哪、长什么样, 从训练数据里自己学出来,
读错一两个字的改回去, 被当成"图和字对不上"扔掉的、其实就是固定字的段捡回来。

为什么(2026-10-14)
------------------
拼音页按类型分好以后(page_type.py, 26,046 张), 在两套标准答案上看新模型(model-n1009d)蓝图还错的段:
    固定的字被读坏   付款方式 -> 付默笑, 交易方式 -> 易 / 菱易努式, 账户余额 -> 账芦茶额, 一键领 -> ——键领
    回首页按钮       裁图切掉一半, 老师读成 支付成功 / 回目贝, recheck 判成对不上扔了 -> 模型干脆输出空
    金额少了 ¥       同一个位置别的页都是 ¥100.00, 这几张标成 100.00
这些都是"同一类页面、同一个位置, 别的几千张页上写的是什么"就能定下来的。fix_labels 的替换表是照着金标准一条条写的,
这里换成从数据里学, 不看金标准(金标准留着量真实效果)。

学什么(每一类页面分开学; 页数 >= MIN_PAGES 的类才学)
----------------------------------------------------
    固定字   在这一类 >= VOCAB_PAGE_FRAC 的页上出现过的标签(按页数算, 一页只算一次), 至少 2 个字;
             带数字的一律不算(金额、日期、满X元可用、积分数都是变的, 满10元可用再常见也不能拿来改满20元可用)
    它的位置 中心横坐标的中位数, 纵坐标(行中心 / 页高)的 5% 和 95% 分位 —— 免费领这种按钮一页好几行, 纵向范围就大
    ¥ 的位置 金额样子的标签(纯数字两位小数), 同一类同一个位置(纵向 2% 一格、左中右三块)有 >= YEN_SHARE 带 ¥ 的格子

怎么改(只改"明确是读错了"的, 拿不准就不动)
------------------------------------------
1. 读错的固定字: 标签不在固定字里、不带数字、这一类里这个标签本身很少见(页占比 < RARE_PAGE_FRAC), 位置落在某个固定字的范围里,
   和它的编辑距离 <= 允许值(3~4 个字 1, 5~7 个字 2, 8 个字以上 3), 而且最近的只有一个 -> 改成那个固定字
2. 捡回来: recheck_segments 判成对不上(_recheck.csv 里 flag=1)扔掉的段, 单独重读的结果按第 1 条的规矩能对上一个固定字
   (位置也对) -> 标签改成那个固定字, 不再扔(bad=0)。别的原因扔掉的(大号红色金额、只有一半的标签、错配 ...)一律不动
3. 补 ¥: 纯数字两位小数的标签, 落在 ¥ 格子里 -> 前面补 ¥
状态栏那一行(行号 0)不动。

用法
----
    python training/type_labels.py --pairs 目录 [目录 ...] --types page_type.csv [...] [--sheet 60 --sheet-dir 输出目录]
    python training/type_labels.py --pairs ... --types ... --apply      # 真改 _segments.csv(先备份 .csv.bak6, 只备份一次)
★ 页面类型按 GWCZ 编号对: 段清单 source 列里的原图路径换过盘也没关系。
★ 先只出报告和核对图, 人眼看过再 --apply。重跑不会重复改: 改过的已经是固定字了。
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

MIN_PAGES = 200           # 一类至少这么多页才学
VOCAB_PAGE_FRAC = 0.10    # 固定字: 这一类里至少一成的页上出现
VOCAB_MIN_PAGES = 30
RARE_PAGE_FRAC = 0.003    # 要改的标签本身在这一类里得少见(真常见的写法不是读错)
X_TOL = 0.08              # 横向: 中心离固定字的中位数不超过页宽的 8%
Y_MARGIN = 0.02           # 纵向: 固定字 5%~95% 分位外面再放 2% 页高
YEN_SHARE = 0.80
YEN_MIN = 50
GW = re.compile(r"GWCZ\d+")
AMOUNT = re.compile(r"^\d{1,7}\.\d{2}$")
YEN_AMOUNT = re.compile(r"^[¥￥]\d{1,7}\.\d{2}$")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(c for c in s if not c.isspace())


def allowed(n: int) -> int:
    if n <= 2:
        return 0
    if n <= 4:
        return 1
    if n <= 7:
        return 2
    return 3


def lev(a: str, b: str, cap: int) -> int:
    """编辑距离, 超过 cap 提前返回 cap + 1。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            best = min(best, cur[j])
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def load_types(paths) -> dict:
    """GWCZ 编号 -> (类别, 宽, 高)。"""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                m = GW.search(r.get("path", ""))
                if m and r.get("W") and r.get("H"):
                    out[m.group(0)] = (r["cls"], int(float(r["W"])), int(float(r["H"])))
    return out


def load_dir(d: Path, types: dict) -> tuple[list[dict], list[dict], list[str]]:
    """读一个目录的段清单, 每段补上 类别 / 相对位置 / 是否被 recheck 扔的 / 重读结果。返回 (原样的行, 记录, 列名)。"""
    rows_y = {}
    pl = d / "_pairs_labeled.csv"
    with pl.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows_y[r["file"]] = (float(r["label_y0"]) + float(r["label_y1"])) / 2
            except (KeyError, ValueError):
                pass
    rc = {}
    rp = d / "_recheck.csv"
    if rp.exists():
        with rp.open(encoding="utf-8-sig", newline="") as f:
            rc = {r["file"]: (r.get("reread", ""), r.get("flag", "0") == "1") for r in csv.DictReader(f)}
    with (d / "_segments.csv").open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        rows = list(rd)
    recs = []
    for i, r in enumerate(rows):
        m = GW.search(r.get("source", "") or r["file"])
        t = types.get(m.group(0)) if m else None
        y = rows_y.get(r.get("row_file", ""))
        rowno = re.search(r"_r(\d+)_s\d+", r["file"])
        if not t or y is None or not rowno:
            continue
        cls, W, H = t
        x0, x1 = float(r["x0"]), float(r["x1"])
        rr, flag = rc.get(r["file"], ("", False))
        recs.append({"i": i, "dir": d.name, "page": m.group(0), "cls": cls, "row": int(rowno.group(1)),
                     "xc": (x0 + x1) / 2 / W, "y": y / H, "text": norm(r.get("text", "")),
                     "bad": r.get("bad") == "1", "rc_flag": flag, "reread": norm(rr), "file": r["file"]})
    return rows, recs, fields


def learn(recs: list[dict]) -> tuple[dict, dict, dict]:
    """每一类的固定字(带位置)、¥ 格子、每个标签的页占比。只用没扔掉的段学。"""
    pages = defaultdict(set)
    seen = defaultdict(lambda: defaultdict(set))
    pos = defaultdict(lambda: defaultdict(list))
    yen = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in recs:
        pages[r["cls"]].add(r["page"])
        if r["bad"] or r["row"] == 0 or not r["text"]:
            continue
        seen[r["cls"]][r["text"]].add(r["page"])
        pos[r["cls"]][r["text"]].append((r["xc"], r["y"]))
        if AMOUNT.match(r["text"]) or YEN_AMOUNT.match(r["text"]):
            cell = (round(r["y"] / 0.02), 0 if r["xc"] < 0.34 else (1 if r["xc"] < 0.66 else 2))
            yen[r["cls"]][cell][0] += 1
            yen[r["cls"]][cell][1] += bool(YEN_AMOUNT.match(r["text"]))
    vocab, yen_cells, share = {}, {}, {}
    for cls, pg in pages.items():
        n = len(pg)
        share[cls] = {t: len(ps) / max(1, n) for t, ps in seen[cls].items()}
        if n < MIN_PAGES:
            continue
        v = {}
        for t, ps in seen[cls].items():
            if len(t) < 2 or any(c.isdigit() for c in t):
                continue
            if len(ps) < VOCAB_MIN_PAGES or len(ps) / n < VOCAB_PAGE_FRAC:
                continue
            xs = sorted(p[0] for p in pos[cls][t])
            ys = sorted(p[1] for p in pos[cls][t])
            q = lambda arr, f: arr[min(len(arr) - 1, int(f * len(arr)))]
            v[t] = {"pages": len(ps), "share": len(ps) / n, "xc": q(xs, 0.5),
                    "y0": q(ys, 0.05) - Y_MARGIN, "y1": q(ys, 0.95) + Y_MARGIN}
        vocab[cls] = v
        yen_cells[cls] = {c for c, (tot, y) in yen[cls].items() if tot >= YEN_MIN and y / tot >= YEN_SHARE}
    return vocab, yen_cells, share


def snap(text: str, xc: float, y: float, voc: dict) -> str | None:
    """位置对得上、编辑距离够小、而且最近的只有一个固定字 -> 返回它; 否则 None。"""
    best, best_d, tie = None, 99, False
    for v, info in voc.items():
        if abs(xc - info["xc"]) > X_TOL or not (info["y0"] <= y <= info["y1"]):
            continue
        cap = allowed(len(v))
        if cap == 0:
            continue
        d = lev(text, v, cap)
        if d > cap:
            continue
        if d < best_d:
            best, best_d, tie = v, d, False
        elif d == best_d:
            tie = True
    return None if tie else best


def decide(r: dict, vocab: dict, yen_cells: dict, share: dict) -> tuple[str | None, bool, str]:
    """返回 (新标签, 新的 bad, 规则); 规则为空 = 不动。"""
    voc = vocab.get(r["cls"])
    if voc is None or r["row"] == 0:
        return None, r["bad"], ""
    t = r["text"]
    if r["bad"]:
        if r["rc_flag"] and r["reread"] and r["reread"] not in voc:
            v = snap(r["reread"], r["xc"], r["y"], voc)
            if v:
                return v, False, "捡回: 重读对上固定字"
        elif r["rc_flag"] and r["reread"] in voc:
            info = voc[r["reread"]]
            if abs(r["xc"] - info["xc"]) <= X_TOL and info["y0"] <= r["y"] <= info["y1"]:
                return r["reread"], False, "捡回: 重读就是固定字"
        return None, True, ""
    if not t or t in voc:
        return None, False, ""
    if AMOUNT.match(t):
        cell = (round(r["y"] / 0.02), 0 if r["xc"] < 0.34 else (1 if r["xc"] < 0.66 else 2))
        if cell in yen_cells.get(r["cls"], ()):
            return "¥" + t, False, "补 ¥"
        return None, False, ""
    if any(c.isdigit() for c in t) or share.get(r["cls"], {}).get(t, 0) >= RARE_PAGE_FRAC:
        return None, False, ""
    v = snap(t, r["xc"], r["y"], voc)
    if v:
        return v, False, "读错的固定字"
    return None, False, ""


def sheet(changes: list[tuple], dst: Path, n: int) -> None:
    """改动的段抽 n 条出核对图: 裁图(带拼音那张) + 原标签 -> 新标签。"""
    import random
    import cv2
    import numpy as np
    from make_check_sheet import _cjk_font
    font = _cjk_font(20)
    pick = changes[:]
    random.Random(7).shuffle(pick)
    pick = pick[:n]
    if not pick:
        return
    ims, W = [], 0
    for d, r, new, rule in pick:
        p = d / "seg_input" / r["file"]
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if p.exists() else None
        if im is None:
            continue
        if im.shape[0] != 60:
            im = cv2.resize(im, (max(1, int(im.shape[1] * 60 / im.shape[0])), 60))
        im = im[:, :640]
        im = cv2.copyMakeBorder(im, 3, 3, 6, 6, cv2.BORDER_CONSTANT, value=(215, 215, 215))
        ims.append((im, r, new, rule))
        W = max(W, im.shape[1])
    if not ims:
        return
    TW = 820
    padded = [cv2.copyMakeBorder(m, 0, 0, 0, W - m.shape[1] + TW, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for m, *_ in ims]
    out = np.vstack(padded)
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    y = 0
    for k, (m, (im, r, new, rule)) in enumerate(zip(padded, ims), 1):
        old = r["text"] if not r["bad"] else f"[扔] {r['reread']}"
        dr.text((W + 6, y + 4), f"{k}. {r['cls']} {rule}", font=font, fill=(90, 90, 90))
        dr.text((W + 6, y + 32), f"{old[:18]}  ->  {new[:18]}", font=font, fill=(190, 0, 0))
        y += m.shape[0]
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", out)[1].tofile(str(dst))
    with dst.with_suffix(".txt").open("w", encoding="utf-8") as f:
        for k, (_im, r, new, rule) in enumerate(ims, 1):
            f.write(f"{k}\t{r['dir']}\t{r['file']}\t{r['cls']}\t{rule}\t{r['text']}\t{r['reread']}\t{new}\n")


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--types", type=Path, nargs="+", required=True, help="page_type.py 出的 page_type.csv")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sheet", type=int, default=0, help="每一类规则抽这么多条出核对图")
    ap.add_argument("--sheet-dir", type=Path, default=None)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--min-pages", type=int, default=MIN_PAGES, help="一类至少这么多页才学(本机小样本测试时调小)")
    ap.add_argument("--vocab-min-pages", type=int, default=VOCAB_MIN_PAGES)
    a = ap.parse_args()
    globals()["MIN_PAGES"], globals()["VOCAB_MIN_PAGES"] = a.min_pages, a.vocab_min_pages

    types = load_types(a.types)
    print("=" * 76)
    print("  TYPE LABELS" + ("  --apply" if a.apply else "  (只看, 不改)") + f"   有类型的页 {len(types):,}")
    print("=" * 76)
    loaded, allrecs = [], []
    for d in a.pairs:
        if not (d / "_segments.csv").exists() or not (d / "_pairs_labeled.csv").exists():
            print(f"  {d}: 没有 _segments.csv / _pairs_labeled.csv, 跳过")
            continue
        rows, recs, fields = load_dir(d, types)
        loaded.append((d, rows, recs, fields))
        allrecs += recs
        print(f"  {d.name}: 段 {len(rows):,}   对上页面类型的 {len(recs):,}")
    vocab, yen_cells, share = learn(allrecs)
    print("-" * 76)
    for cls in sorted(vocab):
        v = vocab[cls]
        top = sorted(v.items(), key=lambda kv: -kv[1]["pages"])
        print(f"  {cls}: 固定字 {len(v)} 个, ¥ 格子 {len(yen_cells.get(cls, ()))} 个")
        print("      " + "  ".join(f"{t}({info['share']:.0%})" for t, info in top[:28]))
    print("-" * 76)
    tot = Counter()
    changes_by_rule = defaultdict(list)
    for d, rows, recs, fields in loaded:
        cnt, ex = Counter(), defaultdict(Counter)
        for r in recs:
            new, bad, rule = decide(r, vocab, yen_cells, share)
            if not rule:
                continue
            cnt[(r["cls"], rule)] += 1
            old = r["text"] if not r["bad"] else "[扔]" + r["reread"]
            ex[(r["cls"], rule)][f"{old} -> {new}"] += 1
            changes_by_rule[(r["cls"], rule)].append((d, r, new, rule))
            if a.apply:
                rows[r["i"]]["text"] = new
                rows[r["i"]]["bad"] = "1" if bad else "0"
        tot.update(cnt)
        n = sum(cnt.values())
        print(f"  {d.name}: 要动 {n:,}")
        for (cls, rule), k in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"      {cls} {rule:<14}{k:>6,}   例: " + "  |  ".join(s for s, _ in ex[(cls, rule)].most_common(a.show)))
        if a.apply and n:
            man = d / "_segments.csv"
            bak = man.with_suffix(".csv.bak6")
            if not bak.exists():
                bak.write_bytes(man.read_bytes())
            if "bad" not in fields:
                fields.append("bad")
            for r in rows:
                if not r.get("bad"):
                    r["bad"] = "0"
            with man.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"      ★ 已改写 {man.name}  (原件备份 {bak.name})")
    print("-" * 76)
    print("  合计: " + "  ".join(f"{cls} {rule} {k:,}" for (cls, rule), k in sorted(tot.items(), key=lambda kv: -kv[1])))
    if a.sheet and a.sheet_dir:
        for (cls, rule), ch in sorted(changes_by_rule.items()):
            name = f"tl_{cls}_{rule.replace(' ', '').replace(':', '').replace('¥', 'yen')}.png"
            sheet(ch, a.sheet_dir / name, a.sheet)
        print(f"  核对图 -> {a.sheet_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
