r"""把老师 OCR 的**系统性**读错改回来(或者不让它进训练) —— 照着 700 段人工金标准查出来的错改。

为什么(2026-10-12)
------------------
金标准上量出来: 模型的准确率基本就等于标签的准确率 —— 它把老师的错一起学了。
蓝图 79 条错里 38 条是"模型和错标签一模一样", 白图 25 条里 16 条。
这些错大多很有规律, 同一个错在几万条标签里反复出现, clean_labels(只纠差一个汉字、高频的)纠不到:
    回首页 -> 口国贝 / 回自贝          付款方式 -> 得默式          支付有福利 -> 变得省福利
    蓝图大号红色金额(券的面额): 标签裁图只截到数字的下半截, 老师基本是猜的
        金标准上 13 段这种"金额+元"的标签错了 11 段: 0.9元 -> U.7元, 2元 -> L元 / 一元, 12元 -> 1元
        单独重读也一样不靠谱(2.24元 -> 224元), 所以一律不让进训练, 不拿重读去改
    裁图里有两样东西, 标签只有其中一样(支付宝打车 + 可叠加商家券使用, 只标了后半)
    立即领取3积分 -> 立即领取了积分(数字读成"了"或者读丢)
    卡号括号: 兴业银行储蓄卡(7428) -> 兴业银行储蓄卡（（428)   (数字读成括号)
    日期、订单号里的 0 读成 D(2D26-09-18)
    状态栏时间连着网速: 1:02|38.8K/s -> 1:02138.8K/s
单独的状态栏时间(01:54、01:54支)标签是对的, 不动。

改法(只动"明确知道是错、而且知道对的是什么"的; 拿不准的就不让进训练, 不瞎改)
---------------------------------------------------------------------------
1. 积分: 立即领取[了/|]积分 -> 重读里有"立即领取N积分"就用 N; 没有就不让进训练
2. 不让进训练(bad=1):
     大号红色金额: 整段只是"金额+元"(2元、U.7元、L元、元), 或者 ¥ 开头的金额里有字母(￥ZUU.UU)
     状态栏连着网速: 有 K/s、B/s、5G、4G、HD, 汉字不超过 1 个
     括号叠了: （（ 或 ((   (卡号数字读成了括号, 原来是几不知道)
     标签只是裁图的一部分: 标签整个出现在重读结果里, 而且重读多出 3 个以上汉字
3. 整段替换表 FIX: 整段等于左边才换成右边(固定界面文字的老错读)
4. 段内替换表 SUB: 段里出现左边这串就换(汉字错读, 左边这串在正常文字里不会出现)
5. 日期/订单号: 整段只有数字、D、-、:、空格, 而且至少 4 个数字 -> D 换成 0

用法
----
    python training/fix_labels.py --pairs 目录 [目录 ...]          # 只看会改什么、改多少
    python training/fix_labels.py --pairs 目录 [目录 ...] --apply  # 真改 _segments.csv(先备份 .csv.bak5)
★ 要先跑过 recheck_segments(目录里要有 _recheck.csv); 没有的话积分只能扔、"标签只是一部分"那条跳过。
★ 重跑不会重复改: 改过的文字已经不在替换表里了; 已经 bad=1 的段不看。
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

FIX = {
    "口国贝": "回首页", "回自贝": "回首页", "回首贝": "回首页", "回自页": "回首页", "回页页": "回首页", "日贝": "回首页",
    "得默式": "付款方式", "得款式": "付款方式", "符款式": "付款方式", "美易式": "交易方式", "易男式": "交易方式",
    "变得省福利": "支付有福利", "变得有福利": "支付有福利", "斐得省福利": "支付有福利",
    "斐得省礼": "支付有礼", "变得省礼": "支付有礼", "麦得礼": "支付有礼",
    "布费顾": "免费领", "宽书领": "免费领", "宽客领": "免费领", "宽费领": "免费领", "免弗领": "免费领", "免负领": "免费领",
    "第费领": "免费领", "免货领": "免费领", "免蕾领": "免费领", "光费领": "免费领", "色花饭": "免费领", "色费顾": "免费领",
    "满费领": "免费领", "免费锁": "免费领", "色费饰": "免费领", "兔费领": "免费领", "免帮领": "免费领", "可费领": "免费领",
    "迈利网": "返利网", "活甜钜鸭脖": "甜辣鸭脖",
}
SUB = [
    ("数学人民吊", "数字人民币"), ("生活激费", "生活缴费"), ("激水电燃费", "缴水电燃费"),
    ("支付奕励", "支付奖励"), ("支付突励", "支付奖励"), ("支付实励", "支付奖励"), ("支付笑励", "支付奖励"),
    ("支村奖励", "支付奖励"), ("支村奕励", "支付奖励"), ("支时奖励", "支付奖励"), ("文付奖励", "支付奖励"),
    ("账单洋情", "账单详情"), ("金部账单", "全部账单"), ("计人收支", "计入收支"), ("汁入收支", "计入收支"),
    ("清選择", "请选择"), ("清选择", "请选择"), ("收款方金称", "收款方全称"), ("银行人账成功", "银行入账成功"),
    ("古付时间", "支付时间"), ("有付时间", "支付时间"), ("立郎领取", "立即领取"), ("立即领最", "立即领取"),
    ("立郎领最", "立即领取"), ("预药转账", "预约转账"), ("颈约转账", "预约转账"), ("立即遍知", "立即通知"),
    ("付款芳式", "付款方式"), ("付款罗式", "付款方式"), ("得款方式", "付款方式"), ("符款方式", "付款方式"),
    ("消费图鉴少", "消费图鉴"), ("请选择入", "请选择"), ("预约转账入", "预约转账"), ("去查看入", "去查看"),
    ("积务", "积分"), ("））", "）"), ("))", ")"),
]
JUNK = "ULJZIOTXuDl一"
BIG_AMOUNT = re.compile(r"^[￥¥]?[0-9" + JUNK + r".,，]*[元完]$")
YEN_JUNK = re.compile(r"^[￥¥][0-9" + JUNK + r".,，]+$")
POINTS = re.compile(r"立即领取[了/|]?积分")
POINTS_RE = re.compile(r"立即领取(\d+)积分")
DATE_LIKE = re.compile(r"^[0-9D:\- ]+$")
STATUS = re.compile(r"[KkMm]?B/s|K/s|(?<![A-Za-z])[45]G(?![A-Za-z])|(?<![A-Za-z])HD(?![A-Za-z])", re.I)
HAN = re.compile(r"[一-鿿]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(c for c in s if c.isalnum() or c == "¥")


def decide(text: str, reread: str | None) -> tuple[str | None, str]:
    """返回 (新文字, 规则); 新文字 None = 不让进训练; 规则为空 = 不动。"""
    t = (text or "").strip()
    if not t:
        return t, ""
    m = POINTS.search(t)
    if m:
        d = POINTS_RE.search(reread or "")
        if d:
            return t[:m.start()] + f"立即领取{d.group(1)}积分" + t[m.end():], "积分数字用重读"
        return None, "积分数字读丢"
    if BIG_AMOUNT.match(t) or (YEN_JUNK.match(t) and any(c in JUNK for c in t[1:])):
        return None, "大号红色金额"
    if STATUS.search(t) and len(HAN.findall(t)) <= 1:
        return None, "状态栏连着网速"
    if "（（" in t or "((" in t:
        return None, "括号叠了"
    if t in FIX:
        return FIX[t], "整段替换"
    new = t
    for _ in range(3):                      # ））） 这种一遍换不干净
        old = new
        for a, b in SUB:
            new = new.replace(a, b)
        if new == old:
            break
    if new != t:
        return new, "段内替换"
    if DATE_LIKE.match(t) and "D" in t and sum(c.isdigit() for c in t) >= 4:
        return t.replace("D", "0"), "日期里D换0"
    if reread is not None:
        nt, nr = norm(t), norm(reread)
        if nt and nt != nr and nt in nr and len(HAN.findall(reread)) - len(HAN.findall(t)) >= 3:
            return None, "标签只是裁图的一部分"
    return t, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    print("=" * 72)
    print("  FIX LABELS" + ("  --apply" if a.apply else "  (只看, 不改)"))
    print("=" * 72)
    total, total_live = Counter(), 0
    for d in a.pairs:
        man = d / "_segments.csv"
        if not man.exists():
            print(f"  {d}: 没有 _segments.csv")
            continue
        rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
        rr = None
        rp = d / "_recheck.csv"
        if rp.exists():
            rr = {r["file"]: r["reread"] for r in csv.DictReader(rp.open(encoding="utf-8-sig"))}
        cnt, ex = Counter(), defaultdict(Counter)
        live = 0
        for r in rows:
            if r.get("bad") == "1":
                continue
            live += 1
            new, rule = decide(r.get("text", ""), None if rr is None else rr.get(r["file"], ""))
            if not rule:
                continue
            cnt[rule] += 1
            ex[rule][f"{r.get('text', '')} -> {new if new is not None else 'X'}"] += 1
            if a.apply:
                if new is None:
                    r["bad"] = "1"
                else:
                    r["text"] = new
        changed = sum(cnt.values())
        total.update(cnt)
        total_live += live
        print(f"  {d.name}: 能用的段 {live:,}   要动 {changed:,} ({changed / max(1, live):.2%})"
              + ("" if rr is not None else "   ★ 没有 _recheck.csv"))
        for rule, n in cnt.most_common():
            print(f"      {rule:<12}{n:>7,}   例: " + "  |  ".join(k for k, _ in ex[rule].most_common(a.show)))
        if a.apply and changed:
            bak = man.with_suffix(".csv.bak5")
            if not bak.exists():
                bak.write_bytes(man.read_bytes())
            fields = list(rows[0].keys())
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
    print("-" * 72)
    n = sum(total.values())
    print(f"  合计: 能用的段 {total_live:,}   要动 {n:,} ({n / max(1, total_live):.2%})")
    for rule, k in total.most_common():
        print(f"      {rule:<12}{k:>8,}")
    print("  (X = 不让进训练)")
    print("=" * 72)


if __name__ == "__main__":
    main()
