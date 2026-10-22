r"""第六轮标签清洗: label_audit 查出来、核对图看过的几类(fix_labels 等五轮之后还剩的)。只动"明确知道错在哪"的。

为什么(2026-10-22)
------------------
按页面类型分开训没用; 金标准上模型 h 的 KEY 错 47 段, 33 段是**现在的**训练标签本身就错(模型照着学)。
label_audit(1dfadaa)在 51 万段现在的标签上数过、核对图逐张看过:
    R1 状态栏(第 0 行, 行底 <= 0.10 x 页宽)里不是纯时间的: 图标、电量、运营商、(!) -> 不进训练(纯时间 HH:MM 标签是对的, 留着)
    R2 卡号括号里有 / 或 +: (4769) 读成 4/69, (3477) 读成 34++, (1376) 读成 7316 —— / 和 + 有时是 7 有时是 1, 拿不准 -> 不进训练
    R3 括号半角全角叠了 (6679)） : 核对图 60/60 是括号后面的 > 箭头被读成 ）-> 只留一个
    R4 ? 开头: ?200.00 是 ¥ -> ¥200.00;  ?244m步行 是定位图标 -> 去掉 ?(另有 204 条本来就没有 ?)
    R5 蓝图大金额没 ¥(100.00 / 500.00): 核对图上都有 ¥, 老师漏了 -> 加 ¥。
       只加: 蓝图页、第 0~5 行、金额 >= 10 —— 小金额是 -¥0.07 立减 / 3.08元 券 / 8.10 网速, 不加
    R6 打码的名字读成 火 / × / x / X / 双: (**强) -> (火强), (*亮) -> (×亮), 星号几个看不出来 -> 不进训练
       (标签里 (*X) / (**X) 写对的有 6,654 条, 模型从那些学)
    R7 手机号 / 邮箱打码: 155×*****37 -> 155******37(× 一个顶一个 *), stexxx@ -> ste***@(支付宝邮箱都是 前 3 位 + ***)
    R8 花体字段名的错读(整段等于才换): 文付实励 -> 支付奖励, 爱易芳式 -> 交易方式, 付默笑 -> 付款方式, 高口说明 -> 商品说明 ...
       段内 汀单号 -> 订单号。只收"不是正常词、而且一看就是哪个字段名"的; 半截的(单号 / 建时 / 易)不补, 裁图可能真的只有半截
    R9 代付成功页(page_type 读到 代付)的标题标成了 支付成功 -> 代付成功
金标准: decide() 也拿来标出"这条规则管得到"的金标准段, 新旧模型在这些段上比(预先写好的标准, 见 .claude\LABEL-FIX-2026-10-22.md)。

用法
----
    python training/label_rules.py --pairs 目录 [...] --types page_type.csv [...] [--sheet 60 --sheet-dir 目录]   # 只看
    python training/label_rules.py --pairs ... --types ... --apply      # 真改 _segments.csv(先备份 .csv.bak9, 只备份一次)
★ 先只看(出报告和核对图), 看过再 --apply。重跑不会重复改: 改过的已经不匹配了, 扔掉的 bad=1 不再看。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GW = re.compile(r"GWCZ\d{15,}", re.I)
TIME = re.compile(r"^\d{1,2}[:：]\d{2}支?$")
CARD_ODD = re.compile(r"[(（](?=[\d/+]{3,5}[)）])(?=[^)）]*[/+])(?=(?:[^)）]*\d){2})[\d/+]{3,5}[)）]")
MIXED = re.compile(r"\)）|）\)")
Q_AMOUNT = re.compile(r"^[?？](\d[\d,]*\.\d{2})$")
Q_WALK = re.compile(r"^[?？](\d+m步行)")
AMOUNT = re.compile(r"^\d[\d,]*\.\d{2}$")
NAME_MASK = re.compile(r"[（(][\*xX×Ｘ火]{0,2}[xX×Ｘ火][\*xX×Ｘ火]{0,2}[一-鿿]{1,2}[)）]|^双[一-鿿]{1,2}([（(]个人[)）])?$")
PHONE_X = re.compile(r"\d[\d\*×]*×[\d\*×]*\d")   # ★ 只改带 * 的(打码); 550ml×12 这种规格没有 *, 不动
EMAIL_X = re.compile(r"(?<=[A-Za-z0-9])[xX×]{3}(?=@)")
FIELD_FIX = {
    "文付实励": "支付奖励", "文何奖励": "支付奖励", "麦付奠励": "支付奖励",
    "爱易芳式": "交易方式", "易号式": "交易方式", "美易罗式": "交易方式",
    "待款芳式": "付款方式", "付默笑": "付款方式",
    "高口说明": "商品说明",
    "列兆时间": "创建时间", "创捷时问": "创建时间",
    "账軍營理": "账单管理",
}
FIELD_SUB = [("汀单号", "订单号")]


def decide(text: str, row: str, W, y1, cls: str, sub: str) -> tuple:
    """一段标签 -> (动作, 新标签, 规则)。动作: "" 不动 / "drop" 不进训练 / "fix" 换成新标签。
    row: 第几行(字符串); W: 页宽; y1: 行底(像素); cls / sub: page_type 的类别 / 细分。"""
    t = (text or "").strip()
    if not t:
        return "", t, ""
    if row == "0" and W and y1 not in ("", None) and float(y1) <= 0.10 * float(W) and not TIME.match(t):
        return "drop", t, "R1状态栏"
    if CARD_ODD.search(t):
        return "drop", t, "R2卡号括号里有斜杠或加号"
    if NAME_MASK.search(t):
        return "drop", t, "R6名字打码读成火x"
    new, rules = t, []
    if MIXED.search(new):
        new = MIXED.sub(lambda m: m.group()[0], new)
        rules.append("R3括号叠了")
    m = Q_AMOUNT.match(new)
    if m:
        new = "¥" + m.group(1)
        rules.append("R4问号是¥")
    elif Q_WALK.match(new):
        new = new[1:]
        rules.append("R4问号是定位图标")
    if cls.startswith("B") and AMOUNT.match(new) and row not in ("", None) and int(row) <= 5:
        try:
            v = float(new.replace(",", ""))
        except ValueError:
            v = 0.0
        if v >= 10:
            new = "¥" + new
            rules.append("R5大金额加¥")
    if any("*" in m.group() for m in PHONE_X.finditer(new)) or EMAIL_X.search(new):
        new = PHONE_X.sub(lambda m: m.group().replace("×", "*") if "*" in m.group() else m.group(), new)
        new = EMAIL_X.sub("***", new)
        rules.append("R7手机邮箱打码")
    if new in FIELD_FIX:
        new = FIELD_FIX[new]
        rules.append("R8字段名")
    else:
        for a_, b_ in FIELD_SUB:
            if a_ in new:
                new = new.replace(a_, b_)
                rules.append("R8字段名")
    if sub == "代付成功" and new == "支付成功":
        new = "代付成功"
        rules.append("R9代付成功")
    if new != t:
        return "fix", new, "|".join(rules)
    return "", t, ""


def load_types(files: list) -> dict:
    """GWCZ -> (类别, 细分)。前面的文件算数。"""
    out = {}
    for fn in files:
        with open(fn, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                m = GW.search(r["path"])
                if m:
                    out.setdefault(m.group().upper(), (r.get("cls", ""), r.get("sub", "")))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--types", nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sheet", type=int, default=0)
    ap.add_argument("--sheet-dir", type=Path, default=None)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    from label_audit import page_width
    types = load_types(a.types)
    print("=" * 76)
    print("  LABEL RULES (R1-R9)" + ("  --apply" if a.apply else "  (只看, 不改)"))
    print("=" * 76)
    tot, by_rule = Counter(), defaultdict(list)
    for d in a.pairs:
        man, pl = d / "_segments.csv", d / "_pairs_labeled.csv"
        if not man.exists():
            print(f"  {d}: 没有 _segments.csv, 跳过")
            continue
        ys = {}
        if pl.exists():
            with pl.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    ys[r["file"]] = (r.get("row", ""), r.get("label_y1", ""))
        with man.open(encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            rows = list(rd)
        if "bad" not in fields:
            fields.append("bad")
        wcache, cnt, ex = {}, Counter(), defaultdict(Counter)
        for r in rows:
            if r.get("bad") == "1" or not (r.get("text") or "").strip():
                continue
            row, y1 = ys.get(r.get("row_file", ""), ("", ""))
            if row == "":
                mm = re.search(r"_r(\d+)_s\d+", r["file"])
                row = str(int(mm.group(1))) if mm else ""
            m = GW.search(r.get("source", "")) or GW.search(r["file"])
            cls, sub = types.get(m.group().upper(), ("", "")) if m else ("", "")
            W = page_width(d, r.get("row_file", ""), wcache, r.get("source", "")) if row == "0" else None
            act, new, rule = decide(r.get("text", ""), row, W, y1, cls, sub)
            if not act:
                continue
            old = r["text"]
            for k in rule.split("|"):
                cnt[k] += 1
                ex[k][f"{old} -> {'(不进训练)' if act == 'drop' else new}"] += 1
                by_rule[k].append((d, r, old, "(不进训练)" if act == "drop" else new, ""))
            if a.apply:
                if act == "drop":
                    r["bad"] = "1"
                else:
                    r["text"] = new
        for r in rows:
            if not r.get("bad"):
                r["bad"] = "0"
        n = sum(cnt.values())
        tot.update(cnt)
        print(f"  {d.name}: 段 {len(rows):,}   要动 {n:,}   " + "  ".join(f"{k} {v:,}" for k, v in sorted(cnt.items())))
        for k, c in sorted(cnt.items()):
            print(f"      {k} {c:>6,}   例: " + "  |  ".join(s for s, _ in ex[k].most_common(a.show)))
        if a.apply and n:
            bak = man.with_suffix(".csv.bak9")
            if not bak.exists():
                bak.write_bytes(man.read_bytes())
            with man.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"      ★ 已改写 {man.name}  (原件备份 {bak.name})")
    print("-" * 76)
    print("  合计: " + "  ".join(f"{k} {v:,}" for k, v in sorted(tot.items())))
    print(f"  TOTAL_CHANGES {sum(tot.values())}")
    if a.sheet and a.sheet_dir:
        from consensus_labels import sheet
        for k, ch in sorted(by_rule.items()):
            sheet(ch, a.sheet_dir / f"rule_{k}.png", a.sheet)
        print(f"  核对图 -> {a.sheet_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
