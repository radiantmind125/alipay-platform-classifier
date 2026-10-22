r"""训练标签体检(只看不改): 把现在的段标签(fix_labels 等五轮清洗之后的)按位置和页面类型导出来, 数几类还没管到的错, 出核对图。

为什么(2026-10-22)
------------------
按页面类型分开训没用(type_exp): 所有模型错的是同一批段, 原因在切图和标签, 不在模型。
金标准上 h 的 KEY 错 47 段里 39 段老师标签也错 —— 但那份老师标签是 10/12 清洗之前导出的,
fix_labels 已经管了一部分(大号红色金额、带网速的状态栏、回首页错读、（（ ))。这里看**现在**的训练标签还剩什么:
    S1 状态栏: 第 0 行而且在页面最上面(行底 <= 0.10 x 页宽) —— 图标 + 小字, 标签多半是渣, 本来就不用读
    S2 标题挂到右边按钮上: 标签是页面标题(支付成功 / 到账成功 ...), 段却在右边 30%(回首页 / 完成 那个位置) —— 切段对错位
    S3 括号半角全角叠了: (6679)） —— fix_labels 只管 （（ )) 和 ））
    S4 金额前面是 ?: ?200.00 —— ¥ 被老师读成 ?
    S5 打码的名字: 双学 / (X航) / (火航) —— ** 被读成 双 / X / 火
    S6 蓝图大金额有没有 ¥: 看老师的写法是不是统一的
    S7 左边一列的字段名: 出频次表, 回来对着已知字段名找花体字的新错读(付默笑 / 高口说明 ...)
输出(都在 --out 下): report.txt, segments_plus.csv(每段: 目录 文件 原图 GWCZ 类别 行 段 x0 x1 页宽 行顶 行底 标签 bad),
    field_names.csv, 核对图 audit_*.png(+ .txt)。
★ 只读, 不改任何 _segments.csv。

用法
----
    python training/label_audit.py --pairs 目录 [...] --types page_type.csv [...] --out 输出目录 [--sheet 60]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from type_exp import load_labels  # noqa: E402

GW = re.compile(r"GWCZ\d{15,}", re.I)
TITLES = {"转账成功", "到账成功", "支付成功", "代付成功", "付款成功", "充值成功", "授权成功", "提交成功", "到账失败",
          "等待对方领取", "订单展示", "转出成功"}
HAN = re.compile(r"[一-鿿]")
CHECKS = {
    "S1_status": "状态栏(第 0 行, 页面最上面)",
    "S2_title_right": "标题挂到右边按钮上",
    "S3_mixed_bracket": "括号半角全角叠了",
    "S4_qmark_amount": "金额前面是 ?",
    "S5_mask": "打码的名字读成 双 / X / 火",
    "S6_amount_no_yen": "蓝图大金额没有 ¥",
}
MIXED = re.compile(r"\)）|）\)")
QMARK = re.compile(r"^[?？]\s*\d")
MASK = re.compile(r"^双[一-鿿]{1,2}([（(]个人[)）])?$|[（(][X火×xＸ][一-鿿][)）]")
AMOUNT = re.compile(r"^[¥￥]?\d[\d,]*\.\d{2}$")


def page_width(d: Path, row_file: str, cache: dict, src: str):
    if src in cache:
        return cache[src]
    from PIL import Image
    w = None
    for sub in ("label", "input"):
        p = d / sub / row_file
        if p.exists():
            try:
                with Image.open(p) as im:
                    w = im.width
                break
            except Exception:  # noqa: BLE001
                pass
    cache[src] = w
    return w


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--types", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sheet", type=int, default=60)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lab = load_labels(a.types)
    hits = defaultdict(list)                 # 检查名 -> [(目录, 行, 旧标签, 提议, "")]
    cnt = defaultdict(Counter)               # 检查名 -> 目录 -> 数
    amount_rows = Counter()                  # (有没有 ¥, 行号) 蓝图大金额
    fields = defaultdict(Counter)            # 蓝 / 白 -> 左边一列字段名 -> 数
    n_all = n_active = n_now = 0
    out_csv = (a.out / "segments_plus.csv").open("w", encoding="utf-8-sig", newline="")
    w = csv.writer(out_csv)
    w.writerow(["dir", "file", "source", "gwcz", "cls", "row", "seg", "x0", "x1", "W", "y0", "y1", "has_pinyin", "text", "bad"])
    for d in a.pairs:
        man, pl = d / "_segments.csv", d / "_pairs_labeled.csv"
        if not man.exists():
            print(f"  {d}: 没有 _segments.csv, 跳过", flush=True)
            continue
        ys = {}
        if pl.exists():
            with pl.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    ys[r["file"]] = (r.get("row", ""), r.get("label_y0", ""), r.get("label_y1", ""))
        wcache = {}
        with man.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            n_all += 1
            m = GW.search(r.get("source", "")) or GW.search(r["file"])
            g = m.group().upper() if m else ""
            cls = lab.get(g, "?")
            row, y0, y1 = ys.get(r.get("row_file", ""), ("", "", ""))
            if row == "":
                mm = re.search(r"_r(\d+)_s\d+", r["file"])
                row = str(int(mm.group(1))) if mm else ""
            W = page_width(d, r.get("row_file", ""), wcache, r.get("source", ""))
            t = (r.get("text") or "").strip()
            bad = r.get("bad", "") == "1"
            w.writerow([d.name, r["file"], r.get("source", ""), g, cls, row, r.get("seg", ""), r.get("x0", ""), r.get("x1", ""),
                        W or "", y0, y1, r.get("has_pinyin", ""), t, "1" if bad else "0"])
            if bad or not t:
                continue
            n_active += 1
            x0 = float(r["x0"]) if r.get("x0") not in (None, "") else None
            x1 = float(r["x1"]) if r.get("x1") not in (None, "") else None
            found = []
            if row == "0" and W and y1 not in ("", None) and float(y1) <= 0.10 * W:
                found.append(("S1_status", "(不进训练)"))
            if t in TITLES and W and x0 is not None and x0 >= 0.70 * W:
                found.append(("S2_title_right", "(不进训练)"))
            if MIXED.search(t):
                found.append(("S3_mixed_bracket", MIXED.sub(lambda z: z.group()[0], t)))
            if QMARK.match(t):
                found.append(("S4_qmark_amount", "¥" + t[1:].lstrip()))
            if MASK.search(t):
                found.append(("S5_mask", "(待定)"))
            if cls.startswith("B") and AMOUNT.match(t):
                has = t[0] in "¥￥"
                amount_rows[("有¥" if has else "没¥", row)] += 1
                if not has:
                    found.append(("S6_amount_no_yen", "¥" + t))
            if W and x1 is not None and x1 <= 0.40 * W and 2 <= len(t) <= 6 and len(HAN.findall(t)) == len(t):
                fields["蓝" if cls.startswith("B") else "白"][t] += 1
            for k, prop in found:
                cnt[k][d.name] += 1
                hits[k].append((d, r, t, prop, ""))
        n_now += 1
        print(f"  {d.name}: 段 {len(rows):,}  页宽读到 {sum(1 for v in wcache.values() if v):,}/{len(wcache):,} 张图", flush=True)
    out_csv.close()
    L = [f"段 {n_all:,}   在训练里的(bad=0 有字) {n_active:,}   目录 {n_now}", ""]
    for k, name in CHECKS.items():
        tot = sum(cnt[k].values())
        L.append(f"{k}  {name}: {tot:,}  ({tot / max(1, n_active):.2%})   " + "  ".join(f"{dd} {v:,}" for dd, v in cnt[k].most_common()))
        ex = Counter(h[2] for h in hits[k]).most_common(25)
        L.append("    常见标签: " + "  |  ".join(f"{s} x{v}" for s, v in ex))
    L.append("")
    L.append("蓝图大金额 ¥ 写法(有没有 ¥, 第几行): " + "  ".join(f"{k[0]} 行{k[1]} {v:,}" for k, v in amount_rows.most_common(12)))
    (a.out / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    with (a.out / "field_names.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fw = csv.writer(f)
        fw.writerow(["side", "label", "count"])
        for side, c in fields.items():
            for s, v in c.most_common(600):
                fw.writerow([side, s, v])
    if a.sheet:
        from consensus_labels import sheet
        for k, ch in hits.items():
            sheet(ch, a.out / f"audit_{k}.png", a.sheet)
    print(f"AUDIT_OK {n_active}", flush=True)


if __name__ == "__main__":
    main()
