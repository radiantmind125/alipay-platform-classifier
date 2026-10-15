r"""标签只标了一半的段, 拿单独重读(recheck_segments 的 _recheck.csv)补全 —— 两边一致改标签(consensus_labels)补不到的那一批。

为什么(2026-10-15)
------------------
两套标准答案 1,356 段里, 老师标签只是正确答案一部分的: 蓝图 45 段(6.0%), 白图 10 段(1.7%)。
所有模型在这 55 段上只读对 40%~45%(其余段 91%), 把"只读一半"的习惯学走了:
    2026-09-04 00:22:32 -> 00:22:32      09-09 13:17 -> 09-09      鼎策快递 满0.1元可用 -> 满0.1元可用
    1688彩白通用提亮焕新 -> 彩白通用提亮焕新      订单号 -> 单号      工商银行储蓄卡(6571) -> 商银行储蓄卡(6571)
来历: split_segments 按空白切段、和老师整行读出的几段一一对上; 老师漏检了一段(比如日期)而段数碰巧还对得上,
这一段的标签就只有一半。
consensus_labels 补不到它们: 那边要"交叉读的模型和重读一致", 而模型已经学会了只读一半, 一致不了。
那边改的 5,499 条(估计对 3,590 条、错 45 条)全是模型本来就读得对的, 所以重训以后分数不动(model-n1009g = e = f)。
这里只用重读一个人的读法, 所以闸要严:
    A 长补全: 宽口径下 标签 是 重读 里连着的一段, 重读多出来的部分至少 3 个真字(汉字/数字/字母), 没有图标渣字
    B 补一两个字: 多出来的是开头或结尾 1~2 个汉字, 而且补全后的写法(数字当 #)在全部标签里 >= 50 次、
       是缺字写法的 3 倍以上 (单号 -> 订单号, 备 -> 备注, 商银行储蓄卡(#) -> 工商银行储蓄卡(#))
    都不动: 状态栏那一行(_r00_); 补全后会被 fix_labels 判成不让进训练的(大号红色金额、状态栏连着网速...)
          补出来那段开头或结尾是标点(，）蜜雪冰城… / （充）充值金…: 图标读成的渣)
          (10/15 服务器上只看一遍后加) 补出来的有字母(拼音没擦干净: 10lun18笔… 外卖满减券WIO E YIS; 订单号前缀 ALW2026 / P100*** 除外);
          数字前面又接数字而结果不是日期时间(图标读成 1: 109-0113:02, 14119943974889); 旧标签只有一两个字而补全后的写法
          别处没见过(我的 -> 我的渭费鉴(少)); 去掉头或尾一个字是常见写法、带上反而少见(干开通记账本…: 干 是加号图标)
          头尾的箭头图标(<账单详情)不进标签
          (10/15 核对 224 行以后再加) 数字前面补数字只许是订单号前半截(补 >= 5 位、整串 >= 16 位); 补出来括号不配对的不补;
          旧标签 3 个字以内也要补全后的写法别处见过; 头上的 ? 和记账本 ¥ 图标去掉
    C 已知错读: 花体字里 账户余额 被老师读成 账芦茶额(个人余额 -> 个大茶额), 核对图 12/12 都是 账户余额 -> 所有活段一起改
标准答案上量过(本地重读, 比服务器上的粗; 整个工具跑过): 23 处改动 18 条从错变对、0 条从对变错、5 条还是错但都更近。
★ bad=1 的段也看: fix_labels 当初把"标签只是裁图的一部分"的段直接扔了(没补), 这一类图就再也没有完整标签的样本;
  补得上的拉回来(bad 改成 0)。

用法
----
    python training/complete_labels.py --pairs 目录 [...] [--sheet 60 --sheet-dir 输出目录]   # 只出报告和核对图
    python training/complete_labels.py --pairs ... --apply      # 真改 _segments.csv(先备份 .csv.bak8, 只备份一次)
★ 先只出报告和核对图, 看过再 --apply。重跑不会重复改: 补过的标签已经和重读一样长了。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consensus_labels import cmp, sheet  # noqa: E402
from fix_labels import decide  # noqa: E402

REAL = re.compile(r"[一-鿿0-9A-Za-z]")
OK_PUNCT = set("·…:-,，¥￥+%*()（）/ 年月日")          # 补出来的部分里允许夹着的标点(cmp 之后的写法)
ICON_HAN = set("一二三")                                # 文档 / 打勾图标常读成的字, 不算真字
EDGE_JUNK = ",，、)）(（·…:"                              # 补出来那段的开头(或结尾)是这些 = 图标渣, 不补
A_MIN_REAL = 3
LATIN_OK = re.compile(r"[A-Z]+(?=[0-9*])")              # 补出来的字母只许是订单号前缀(ALW2026… P100******), 后面紧跟数字或 *
DATETIME = re.compile(r"^(\d{4}-)?\d{2}-\d{2}\d{2}:\d{2}(:\d{2})?$")   # cmp 之后(空格去掉)的 日期时间
SHORT_OLD = 3                                           # 旧标签只有 1~3 个字时, 补全后的写法别处要见过
SHORT_MIN_FREQ = 3
LONG_ID_ADD, LONG_ID_LEN = 5, 16                        # 数字前面补数字: 补 >= 5 位、整串 >= 16 位的是订单号/交易号前半截, 可以
KNOWN = [("账芦茶额", "账户余额"), ("个大茶额", "个人余额")]   # 花体字的已知错读(10/15 核对图 12/12 都是 账户余额), 所有活段都改
B_MIN_FREQ = 50
B_RATIO = 3


def skel(s: str) -> str:
    return re.sub(r"\d", "#", s)


def is_han(c: str) -> bool:
    return "一" <= c <= "鿿"


def propose(old: str, rr: str, freq: Counter) -> tuple[str, str]:
    """返回 (规则, 新标签); 规则为空 = 不补。"""
    o, n = cmp(old), cmp(rr)
    if not o or not n or len(n) <= len(o) or o not in n:
        return "", ""
    i = n.find(o)
    pre, suf = n[:i], n[i + len(o):]
    if (pre and pre[0] in EDGE_JUNK) or (suf and suf[-1] in ",，、(（:"):     # 结尾的省略号 … 是真的(名字被截断)
        return "", ""                                      # ，）蜜雪冰城… / （充）充值金…: 补出来的头尾是图标渣
    add = pre + suf
    if re.search(r"[A-Za-z]", LATIN_OK.sub("", pre) + LATIN_OK.sub("", suf)):
        return "", ""                                      # 拼音没擦干净读成字母(10lun18笔… 外卖满减券WIO E YIS), logo(tooNt)
    if pre and pre[-1].isdigit() and o[0].isdigit() and not DATETIME.match(n):
        head = re.search(r"\d+$", pre).group(0)
        whole = re.match(r"\d+", n[len(pre) - len(head):]).group(0)
        if not (head == pre and len(head) >= LONG_ID_ADD and len(whole) >= LONG_ID_LEN):
            return "", ""                                  # 数字前面又接几位数字: 图标读成 1(109-0113:02, 14119943974889)
    if n.count("(") != n.count(")") and o.count("(") == o.count(")"):
        return "", ""                                      # 补出来括号不配对(…去查看）, …有限公司(嵩级认证)
    if len(o) <= SHORT_OLD and freq[skel(n)] < SHORT_MIN_FREQ:
        return "", ""                                      # 旧标签就一两个字, 补全后的写法别处没见过(我的 -> 我的渭费鉴(少))
    for cut in (n[1:], n[:-1]):
        if len(cut) >= 2 and freq[skel(cut)] >= 20 and freq[skel(n)] * 10 < freq[skel(cut)]:
            return "", ""                                  # 去掉头(尾)一个字是常见写法, 带上反而少见: 那个字是图标渣(干开通记账本…)
    real = [c for c in add if REAL.match(c)]
    new = re.sub(r"^[<〈‹?？]+|^[¥￥](?=[^\d\-+])|[>〉›]+$", "", rr.strip())   # 头尾的图标不进标签(<账单详情, ?银行处理中)
    for a_, b_ in KNOWN:
        new = new.replace(a_, b_)
    if all(REAL.match(c) or c in OK_PUNCT for c in add) and len([c for c in real if c not in ICON_HAN]) >= A_MIN_REAL:
        return "A长补全", new
    if 1 <= len(add) <= 2 and all(is_han(c) for c in add) and (not pre or not suf):
        fn, fo = freq[skel(n)], freq[skel(o)]
        if fn >= B_MIN_FREQ and fn >= B_RATIO * fo:
            return "B补一两个字", new
    return "", ""


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sheet", type=int, default=0)
    ap.add_argument("--sheet-dir", type=Path, default=None)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    print("=" * 76)
    print("  COMPLETE LABELS" + ("  --apply" if a.apply else "  (只看, 不改)"))
    print("=" * 76)
    loaded, freq = [], Counter()
    for d in a.pairs:
        man, rp = d / "_segments.csv", d / "_recheck.csv"
        if not man.exists() or not rp.exists():
            print(f"  {d}: 没有 _segments.csv 或 _recheck.csv, 跳过")
            continue
        with man.open(encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            rows = list(rd)
        with rp.open(encoding="utf-8-sig", newline="") as f:
            rr = {r["file"]: r.get("reread", "") for r in csv.DictReader(f)}
        for r in rows:
            if r.get("bad") != "1":
                freq[skel(cmp(r.get("text", "")))] += 1
        loaded.append((d, rows, fields, rr))
    tot, by_kind = Counter(), defaultdict(list)
    for d, rows, fields, rr in loaded:
        cnt, ex = Counter(), defaultdict(Counter)
        skipped = Counter()
        for r in rows:
            if re.search(r"_r00_s\d+", r["file"]):
                continue
            old, re_ = r.get("text", ""), rr.get(r["file"], "")
            rule, new = propose(old, re_, freq)
            if not rule and r.get("bad") != "1" and any(a_ in old for a_, _ in KNOWN):
                new = old
                for a_, b_ in KNOWN:
                    new = new.replace(a_, b_)
                rule = "C已知错读"
            if not rule:
                continue
            nt, why = decide(new, re_)
            if nt is None or nt != new:
                skipped[why or "fix_labels 会再改"] += 1      # 补全后会被扔 / 会被别的规则再改的, 不碰
                continue
            revive = r.get("bad") == "1"
            k = rule + ("(拉回)" if revive else "")
            cnt[k] += 1
            ex[k][f"{old} -> {new}"] += 1
            by_kind[("蓝" if "blue" in d.name else "白", k)].append((d, r, old, new, re_))     # 核对图按 规则 x 拉回没拉回 分开
            if a.apply:
                r["text"] = new
                if revive:
                    r["bad"] = "0"
        n = sum(cnt.values())
        tot.update(cnt)
        print(f"  {d.name}: 段 {len(rows):,}   要补 {n:,}   " + "  ".join(f"{k} {v:,}" for k, v in cnt.most_common())
              + ("   (补完会被 fix_labels 扔掉的不碰: " + "  ".join(f"{k} {v}" for k, v in skipped.most_common()) + ")" if skipped else ""))
        for k, c in cnt.most_common():
            print(f"      {k} {c:>6,}   例: " + "  |  ".join(s for s, _ in ex[k].most_common(a.show)))
        if a.apply and n:
            man = d / "_segments.csv"
            bak = man.with_suffix(".csv.bak8")
            if not bak.exists():
                bak.write_bytes(man.read_bytes())
            with man.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"      ★ 已改写 {man.name}  (原件备份 {bak.name})")
    print("-" * 76)
    print("  合计: " + "  ".join(f"{k} {v:,}" for k, v in tot.most_common()))
    print(f"  TOTAL_CHANGES {sum(tot.values())}")             # 给服务器脚本读的(纯 ASCII)
    if a.sheet and a.sheet_dir:
        for (color, k), ch in sorted(by_kind.items()):
            sheet(ch, a.sheet_dir / f"cp_{color}_{k}.png", a.sheet)
        print(f"  核对图 -> {a.sheet_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
