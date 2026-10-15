r"""两边一致改标签: 交叉读(crossfit.py, 没背过这条标签的模型) 和 单独重读(recheck_segments 的 _recheck.csv) 读得一样,
却和标签不一样的段, 标签换成模型读的 —— 专改名字、商家、卡号尾号、只标了一半这类规则管不到的错。

为什么(2026-10-14)
------------------
在两套标准答案上量过(model-n1009e 读验证集 + 单独重读): 这样改的 42 条里 39 条从错变对、0 条从对变错、3 条还是错。
改对的有 哈啰顺风车 -> 哈啰顺风车新客专属5折优惠(标签只有一半)、小羔 -> 小志、(0551) -> (0531)、单号 -> 订单号、支付奕励 -> 支付奖励。
两个读的人是独立的: 一个是 RapidOCR 只识别(裁掉拼音的那张), 一个是我们自己的模型(带拼音的那张、而且没训过这一段)。

规矩(只改活着的段; 状态栏那一行不动)
------------------------------------
    比的时候用宽口径(和 score_gold.lenient 一样): 全半角统一、去空格、箭头图标不算、十 当 +、引号样式不算、句尾 + , 。 不算
    模型读的 == 重读的 != 标签  ->  标签换成模型读的
    保险: 标签是很常见的写法(全部目录里 >= COMMON 次), 两边一致读出来的却少见(不到标签次数的 1/10) -> 不改
          (两个读的人一起把 付款方式 读成 付款方 这种, 信常见的标签)
    10/15 训练集上抽 360 条对着图核过(补全 109 对 1 错, 换字 89 对 7 错, 删字 50 对 13 错), 错的几种加了闸(guard):
          银行名被改(农业->农亚)、卡号尾号不再是 4 位数字、数字少了或变成字母(满1元->满元, 2026->2D26, 手写 1 -> /)、
          删的不是箭头 / 加号 / 文档图标读成的字(进群领->进群, 晟逸卫坊->逸卫坊)、删汉字后剩下的不是常见写法、
          图标读成 上 / 三 / 一 加在常见写法前面、常见三字词被改成少见的(已入账->已人账)、编号里 O/0 I/1 对调、逗号改句号;
          省略号几个点、定位图标 ?、记账本图标 ¥ 比的时候都不算。模型读出来尾巴多个逗号(箭头读成的)而重读没有的, 写进标签前去掉。
分三种出报告和核对图: 补全(读出来更长, 多半是标签只有一半)、换字(一样长)、删字(读出来更短)。

用法
----
    python training/consensus_labels.py --pairs 目录 [...] --pred cf_pred_A.csv cf_pred_B.csv [--sheet 60 --sheet-dir 输出目录]
    python training/consensus_labels.py --pairs ... --pred ... --apply     # 真改 _segments.csv(先备份 .csv.bak7, 只备份一次)
★ --pred 是 crossfit.py predict 出的 csv(path, label, pred); path 里的目录名和文件名对上 _segments.csv 的段。
★ 先只出报告和核对图, 看过再 --apply。重跑不会重复改: 改过的标签已经和读出来的一样了。
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

COMMON = 50


def cmp(s: str) -> str:
    """宽口径: 和 score_gold.lenient 一样。"""
    s = unicodedata.normalize("NFKC", s or "")
    s = "".join(c for c in s if not c.isspace()).replace("十", "+")
    for q in "“”„‟":
        s = s.replace(q, '"')
    for q in "‘’":
        s = s.replace(q, "'")
    s = "".join(c for c in s if c not in "><›‹〉〈")
    s = re.sub(r"^(\d{1,2}:\d{2})支$", r"\1", s)
    s = re.sub(r"(?<!\d)[.·•]+(?!\d)|[.·•]{2,}", "…", s)      # 省略号几个点都算一样(··· / .. / ·)
    s = re.sub(r"^[?？]", "", s)                                 # 定位图标读成的 ?
    s = re.sub(r"^[¥￥](?=[^\d\-+])", "", s)                    # 记账本图标读成的 ¥(后面不是数字)
    return s.rstrip("+,，。")


TRAIL_JUNK = set("入人干)）〉>/|!！")         # 结尾: 箭头 > 常读成 入 人 ) /, 加号读成 干
LEAD_JUNK = set("三司白兰二」「?？(（)）一V")   # 开头: 文档图标常读成 三 司 白 兰 二, 打勾图标读成 ) 一 V
#                                             (删开头的汉字要剩下的是常见写法: 一支付成功 -> 支付成功 可以, 一键领 -> 键领 不行)
MID_JUNK = set("「」|/、~")                   # 中间: 纯标点的零碎
BANKS = ("农业银行", "工商银行", "建设银行", "中国银行", "招商银行", "交通银行", "光大银行", "邮政储蓄", "邮储银行", "兴业银行",
         "浦发银行", "民生银行", "中信银行", "平安银行", "华夏银行", "广发银行", "农商银行", "农村信用", "农信")
CONFUSE = {("O", "0"), ("0", "O"), ("I", "1"), ("1", "I"), ("l", "1"), ("1", "l"), ("Z", "2"), ("2", "Z"),
           ("S", "5"), ("5", "S"), ("B", "8"), ("8", "B")}
TAIL = re.compile(r"(?:卡|行)[（(]([^）)]*)[）)]")


def trigrams(s: str) -> list[str]:
    return [s[i:i + 3] for i in range(max(0, len(s) - 2))]


def is_han(c: str) -> bool:
    return "一" <= c <= "鿿"


def guard(old: str, new: str, freq: Counter, cnt3: Counter) -> str:
    """两边一致了也不改的几种(10/15 训练集上抽 360 条人眼核过以后加的), 返回理由; 空 = 可以改。"""
    import difflib
    o, n = cmp(old), cmp(new)
    for b in BANKS:
        if b in o and b not in n:
            return "银行名被改"
    to, tn = TAIL.search(o), TAIL.search(n)
    if to and re.fullmatch(r"\d{4}", to.group(1)) and (not tn or not re.fullmatch(r"\d{4}", tn.group(1))):
        return "卡号尾号不再是4位数字"
    ops = [op for op in difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes() if op[0] != "equal"]
    only_conf = only_comma_to_dot = bool(ops)
    for tag, i1, i2, j1, j2 in ops:
        os_, ns_ = o[i1:i2], n[j1:j2]
        arrow7 = tag == "delete" and i2 == len(o) and os_ == "7" and i1 > 0 and is_han(o[i1 - 1])   # 请选择7: 箭头读成 7
        if not arrow7 and sum(c.isdigit() for c in ns_) < sum(c.isdigit() for c in os_):
            return "数字少了或变成别的"
        if tag == "delete" and not arrow7:
            if i2 == len(o) and all(c in TRAIL_JUNK for c in os_):
                pass                                            # 结尾的箭头 / 加号
            elif i1 == 0 and all(c in LEAD_JUNK or "a" <= c <= "z" for c in os_):
                if any(is_han(c) for c in os_) and freq[n] < 20:
                    return "删开头的汉字后剩下的不常见"
            elif not all(c in MID_JUNK for c in os_):
                return "删掉的不是图标读成的字"
        if tag == "insert" and i1 == 0 and ns_ in ("上", "三", "一") and freq[o] >= 20:
            return "图标读成一个字加在前面"
        if tag == "replace" and len(os_) == len(ns_):
            if not all((a, b) in CONFUSE for a, b in zip(os_, ns_)):
                only_conf = False
            if not all(a in ",，" and b in ".。" for a, b in zip(os_, ns_)):
                only_comma_to_dot = False
            for k in range(i1, i2):
                if not is_han(o[k]):
                    continue                            # 只看汉字; 数字(卡号尾号)不按常见不常见判
                jk = j1 + (k - i1)
                t_old, t_new = o[max(0, k - 1):k + 2], n[max(0, jk - 1):jk + 2]
                if len(t_old) == 3 and cnt3[t_old] >= 20 and cnt3[t_new] * 10 < cnt3[t_old]:
                    return "常见的词被改成少见的"
        else:
            only_conf = only_comma_to_dot = False
    if only_conf and any("A" <= c <= "Z" or "a" <= c <= "z" for c in n):
        return "编号里字母和数字对调"
    if only_comma_to_dot:
        return "逗号改成句号"
    return ""


def kind(old: str, new: str) -> str:
    a, b = len(cmp(old)), len(cmp(new))
    return "补全" if b > a else ("换字" if b == a else "删字")


def load_preds(paths) -> dict:
    """(目录名, 文件名) -> 模型读的。"""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                q = Path(r["path"])
                out[(q.parent.parent.name, q.name)] = r["pred"]
    return out


def sheet(changes, dst: Path, n: int) -> None:
    import random
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from make_check_sheet import _cjk_font
    font = _cjk_font(20)
    pick = changes[:]
    random.Random(11).shuffle(pick)
    pick = pick[:n]
    ims, W = [], 0
    for d, r, old, new, rr in pick:
        p = d / "seg_input" / r["file"]
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if p.exists() else None
        if im is None:
            continue
        if im.shape[0] != 60:
            im = cv2.resize(im, (max(1, int(im.shape[1] * 60 / im.shape[0])), 60))
        im = cv2.copyMakeBorder(im[:, :900], 3, 3, 6, 6, cv2.BORDER_CONSTANT, value=(215, 215, 215))
        ims.append((im, d, r, old, new, rr))
        W = max(W, im.shape[1])
    if not ims:
        return
    TW = 900
    padded = [cv2.copyMakeBorder(m[0], 0, 0, 0, W - m[0].shape[1] + TW, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for m in ims]
    out = np.vstack(padded)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    y = 0
    for k, (m, (_im, d, r, old, new, rr)) in enumerate(zip(padded, ims), 1):
        dr.text((W + 6, y + 4), f"{k}. 标签 {old[:30]}", font=font, fill=(90, 90, 90))
        dr.text((W + 6, y + 32), f"-> {new[:30]}", font=font, fill=(190, 0, 0))
        y += m.shape[0]
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", out)[1].tofile(str(dst))
    with dst.with_suffix(".txt").open("w", encoding="utf-8") as f:
        for k, (_im, d, r, old, new, rr) in enumerate(ims, 1):
            f.write(f"{k}\t{d.name}\t{r['file']}\t{old}\t{rr}\t{new}\n")


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sheet", type=int, default=0)
    ap.add_argument("--sheet-dir", type=Path, default=None)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    preds = load_preds(a.pred)
    print("=" * 76)
    print("  CONSENSUS LABELS" + ("  --apply" if a.apply else "  (只看, 不改)") + f"   交叉读的段 {len(preds):,}")
    print("=" * 76)
    loaded = []
    freq = Counter()
    for d in a.pairs:
        man = d / "_segments.csv"
        if not man.exists():
            print(f"  {d}: 没有 _segments.csv, 跳过")
            continue
        with man.open(encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            rows = list(rd)
        rr = {}
        rp = d / "_recheck.csv"
        if rp.exists():
            with rp.open(encoding="utf-8-sig", newline="") as f:
                rr = {r["file"]: r.get("reread", "") for r in csv.DictReader(f)}
        for r in rows:
            if r.get("bad") != "1":
                freq[cmp(r.get("text", ""))] += 1
        loaded.append((d, rows, fields, rr))
    cnt3 = Counter()
    for t, c in freq.items():
        for g in trigrams(t):
            cnt3[g] += c
    gwhy = Counter()
    tot, guard_tot = Counter(), 0
    by_kind = defaultdict(list)
    for d, rows, fields, rr in loaded:
        cnt, ex = Counter(), defaultdict(Counter)
        seen = blocked = 0
        for r in rows:
            if r.get("bad") == "1" or re.search(r"_r00_s\d+", r["file"]):
                continue
            p = preds.get((d.name, r["file"]))
            if p is None:
                continue
            seen += 1
            old = r.get("text", "")
            re_ = rr.get(r["file"], "")
            if not cmp(p) or cmp(p) != cmp(re_) or cmp(p) == cmp(old) or not cmp(old):   # 读出来只剩标点的(。)不算
                continue
            if freq[cmp(old)] >= COMMON and freq[cmp(p)] < freq[cmp(old)] / 10:
                blocked += 1
                gwhy["常见标签少见读法"] += 1
                continue
            why = guard(old, p, freq, cnt3)
            if why:
                blocked += 1
                gwhy[why] += 1
                continue
            if re.search(r"[，,。]$", p) and not re.search(r"[，,。]$", re_):
                p = re.sub(r"[，,。]+$", "", p)          # 箭头读成的逗号: 尾巴跟重读走
            k = kind(old, p)
            cnt[k] += 1
            ex[k][f"{old} -> {p}"] += 1
            by_kind[("蓝" if "blue" in d.name else "白", k)].append((d, r, old, p, re_))
            if a.apply:
                r["text"] = p
        n = sum(cnt.values())
        tot.update(cnt)
        guard_tot += blocked
        print(f"  {d.name}: 交叉读到的活段 {seen:,}   要改 {n:,} ({n / max(1, seen):.2%})   保险挡下 {blocked:,}")
        for k, c in cnt.most_common():
            print(f"      {k} {c:>6,}   例: " + "  |  ".join(s for s, _ in ex[k].most_common(a.show)))
        if a.apply and n:
            man = d / "_segments.csv"
            bak = man.with_suffix(".csv.bak7")
            if not bak.exists():
                bak.write_bytes(man.read_bytes())
            with man.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"      ★ 已改写 {man.name}  (原件备份 {bak.name})")
    print("-" * 76)
    print("  合计: " + "  ".join(f"{k} {v:,}" for k, v in tot.most_common()) + f"   保险挡下 {guard_tot:,}")
    print("  挡下的理由: " + "  ".join(f"{k} {v:,}" for k, v in gwhy.most_common()))
    if a.sheet and a.sheet_dir:
        for (color, k), ch in sorted(by_kind.items()):
            sheet(ch, a.sheet_dir / f"cl_{color}_{k}.png", a.sheet)
        print(f"  核对图 -> {a.sheet_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
