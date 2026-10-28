r"""训练清单里老师读错的固定文字(step 50, 2026-10-29)。先 scan(只读, 出候选和样图, 人看过), 再 apply(只按看过的清单改, 写新目录)。

为什么
------
step 49 发现训练清单里有成批的老师读错: 农亚银行储蓄卡 111 条、工商银行诸蓄卡 49 条、农业银行储蓄华 32 条、翠镶聘端酱等 15 条
(手写体的 农业银行储蓄卡)、开通记账本 前面多一个 ￥ / 羊、本月转账红包是最大支出类日 ... 模型把这些也学会了(gold3 的 1862、531)。
读完以后再改(step 49)只救回一部分, 根上要把训练标签改对再重训。

scan(只读; 只写 --out)
    python training/relabel_known.py scan --lists <train-n1009k> <train-n1009i> --known <known_texts> --out <目录> [--samples 3] [--per-rule 150]
    对两份清单(train.txt + val.txt)里每个不同的标签按下面的规则出"候选", 不改任何东西:
      text    不带数字 / *, >= 4 个字, 和一条常见文字(--known 里 >= 50 次的, 见 targets)同长只差一个字, 那条 >= 5 倍常见
      pre     >= 6 个字, 和某条常见文字的开头那段同长只差一个字(被切断的长句), 那条 >= 5 倍常见
      affix   常见文字(>= 5 个字)前面或后面多了一个字, 那条 >= 5 倍常见
      plus    常见文字后面多了一个 干(+ 图标读错; 十 打分时本来就算 +)
      bank    卡号行(括号里 4 位数字)括号前那一截不在银行白名单里、但带银行字样: 白名单里编辑距离最近的(<= 2), 一样近的都列出来
      bankjunk 卡号行括号前那一截一个银行字样都没有(翠镶聘端酱等 这种乱码)-> 候选是整条不要
      slash   只由 数字 - : 空格 / 组成(>= 6 个字)且带 / -> 候选是 / 换成 1
      mask    名字少了星号: (字) / 字(个人) / (##字) -> 候选是整条不要(三套答案 3,445 段里一次都没有对的)
    写 <out>/candidates.tsv(rule, old, new, n_k_train, n_k_val, n_i_train, n_i_val, sample files)和每个候选最多 --samples 张样图
    (从第一份清单的图拷, 名字 c<序号>_<n>.png), 打印 RELABEL_SCAN_STATS / RELABEL_SCAN_OK。
apply(只按人审过的决定改; 原清单只读, 写两个新目录)
    python training/relabel_known.py apply --lists <train-n1009k> <train-n1009i> --out <train-n1009m> <train-n1009ic>
           --scan <step50_scan> --decisions training/relabel_decisions_50.tsv
    决定文件(公开仓库里, 只有候选编号 / 银行名 / 界面固定文字)第一行 pin 的是 candidates.tsv 的 sha256, 不一样就停。
    规则: rule slash|mask|bankjunk = 这条规则的候选全照办(keep 的除外); bankprefix 旧 新 = 银行候选里括号前正好是旧的换成新的;
    accept <cid> = 照候选的 new 改; rewrite <cid> <新标签> = 改成给的; 没写到的候选一律不动。
    新目录: train.txt / val.txt(行的顺序不变, 只有决定里的标签改了或整行去掉)+ charset.txt(原样拷)+ _relabel_applied.tsv + _relabel_version.txt。
    打印 RELABEL_APPLY_STATS(每份清单 进 / 出 / 改 / 去掉, 按规则)和 RELABEL_APPLY_OK。
每个子命令最后打印一行纯 ASCII 的 XXX_OK; 不对就打印原因并以非 0 退出。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
DIGIT = re.compile(r"[0-9０-９]")
BAD_TARGET = re.compile(r"[0-9０-９*@￥¥]")
CARD = re.compile(r"^(.*?)([.。·…]*)([(（]\s*\d{4}\s*[)）]?.*)$")
SHAPE = re.compile(r"^[\d\-: /]{6,}$")
# 斜线换成 1 以后得是支付宝里的样子(空格去掉以后): 年月日 时分秒 / 月日 时分 / 时分秒 / 年月日 / 12 位以上的单号; 2026/09/12 这种不动
FORMATS = re.compile(r"\d{4}-\d{2}-\d{2}(\d{2}:\d{2}(:\d{2})?)?|\d{2}-\d{2}\d{2}:\d{2}(:\d{2})?|\d{1,2}:\d{2}(:\d{2})?|\d{12,}")
MASK = [re.compile(r"[(（][一-鿿][)）]"), re.compile(r"^[一-鿿][(（]个人[)）]"), re.compile(r"[(（]#+[一-鿿]")]
BANKISH = ("银行", "农信", "农村", "农商", "信用", "储蓄", "邮政", "商业", "联合")
# 银行白名单(公开的银行名, 从 known_texts/card_prefix.csv 里 >= 5 次的挑出来, 读错 / 乱码 / 切掉开头的不要; 和 step 49 的一样)
BANKS = """上海银行储蓄卡 东莞银行 东莞银行储蓄卡 中信银行 中信银行信用卡 中信银行储蓄卡 中原银行 中国光大银行 中国农业银行 中国工商银行
中国建设银行 中国民生银行 中国邮政储蓄 中国邮政储蓄银行 中国银行 中国银行信用卡 中国银行储蓄卡 乌鲁木齐银行储蓄卡 九江银行储蓄卡
云南农信储蓄卡 云南省农村信 云南省农村信用社 交通银行 交通银行信用卡 交通银行储蓄卡 光大银行储蓄卡 兰州银行 兰州银行储蓄卡 兴业银行
兴业银行储蓄卡 内蒙古农商银 农业银行 农业银行储蓄卡 北京农商银行储蓄卡 北京银行 北京银行储蓄卡 华夏银行 华夏银行储蓄卡 南京银行储蓄卡
南海农商银行储蓄卡 台州银行储蓄卡 吉林省农村信用社联合社 吉林银行储蓄卡 四川农信储蓄卡 四川省农村信 四川省农村信用社联合社
国民村镇银行储蓄卡 威海银行储蓄卡 宁夏银行 宁夏银行储蓄卡 宁波银行 宁波银行储蓄卡 安徽省农村信用社 山东农信储蓄卡 山西农信 工商银行
工商银行信用卡 工商银行储蓄卡 平安银行 平安银行储蓄卡 广东农信储蓄卡 广东华润银行储蓄卡 广东省农村信 广东省农村信用社联合社 广发银行
广发银行信用卡 广发银行储蓄卡 广西农信储蓄卡 广西农商联合 广西农商联合银行 建设银行 建设银行储蓄卡 成都农商银行储蓄卡 成都银行 招商银行
招商银行信用卡 招商银行储蓄卡 新疆农信储蓄卡 新疆农村信用 新疆农村信用社 新疆银行储蓄卡 柳州银行储蓄卡 民生银行储蓄卡
江南农村商业银行储蓄卡 江苏银行 江西农信储蓄卡 河南农商银行 浙商银行储蓄卡 浙江农商联合 浙江农商联合银行 浙江农商联合银行储蓄卡 浦发银行
浦发银行储蓄卡 海南农商储蓄卡 深圳农商银行储蓄卡 湖北省农信社 甘肃农信储蓄卡 甘肃农商银行 甘肃银行储蓄卡 石嘴山银行 福建农信储蓄卡
福建省农村信用社联合社 网商银行储蓄卡 苏州银行 贵州农信储蓄卡 贵州农村商业 贵州农村商业联合银行股份有限公司 贵州银行储蓄卡 贵阳银行
贵阳银行储蓄卡 辽宁农商银行 辽宁农商银行储蓄卡 邮储银行 邮储银行储蓄卡 重庆农商行储蓄卡 重庆农村商业 重庆农村商业银行 长沙银行
长沙银行储蓄卡 陕西省农信社 青海农信储蓄卡 青海银行储蓄卡 黄河农商银行""".split()


def die(tag: str, msg: str) -> None:
    print(f"  ★ {msg}", flush=True)
    print(tag, flush=True)
    sys.exit(2)


def norm(s: str) -> str:
    """打分用的写法(score_gold.lenient): 全半角、空格、箭头、十 和 +、结尾标点都不算差别。"""
    s = "".join(c for c in unicodedata.normalize("NFKC", s or "") if not c.isspace()).replace("十", "+")
    for a in "“”„‟":
        s = s.replace(a, '"')
    for a in "‘’":
        s = s.replace(a, "'")
    s = "".join(c for c in s if c not in "><›‹〉〈")
    s = re.sub(r"^(\d{1,2}:\d{2})支$", r"\1", s)
    return s.rstrip("+,，。")


def ed(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sub1(a: str, b: str) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def read_list(d: Path, which: str) -> list:
    p = d / f"{which}.txt"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if "\t" in ln:
                path, txt = ln.split("\t", 1)
                rows.append((path, txt))
    return rows


def targets(known: Path, charset: set) -> dict:
    """常见文字: known_texts/texts.csv 里按打分写法合并以后 >= 50 次的, 不带数字 / * / @ / ￥, 不带 公司(商户名是值), 字都在字表里。
    返回 {写法: (最常见的原文, 次数)}。"""
    merged, best = Counter(), {}
    with (known / "texts.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t, n = r["text"], int(r["total"])
            k = norm(t)
            merged[k] += n
            if k not in best or n > best[k][1]:
                best[k] = (t, n)
    tg = {k: (best[k][0], n) for k, n in merged.items()
          if n >= 50 and k and not BAD_TARGET.search(best[k][0]) and "公司" not in k and all(ch in charset for ch in best[k][0])}
    # 老师自己的读错也会 >= 50 次(羊开通记账本... 114 次): 是另一条 >= 5 倍常见的文字只差一个字 / 前后多一个字 / 后面多个 干 的, 不算常见文字
    bad = set()
    for k, (_t, n) in tg.items():
        for kk, (_tt, c) in tg.items():
            if kk == k or c < 5 * n:
                continue
            if (len(k) >= 4 and sub1(k, kk)) or (len(kk) >= 5 and len(k) == len(kk) + 1 and (k[1:] == kk or k[:-1] == kk)) \
                    or (k.endswith("干") and k[:-1] == kk):
                bad.add(k)
                break
    return {k: v for k, v in tg.items() if k not in bad}


def candidates(label: str, tg: dict, freq: Counter) -> list:
    """一个标签的候选 [(rule, new)]; new = "" 表示整条不要。"""
    out = []
    for m in MASK:
        if m.search(label):
            return [("mask", "")]
    if "/" in label and SHAPE.match(label):
        new = label.replace("/", "1")
        return [("slash", new)] if FORMATS.fullmatch(new.replace(" ", "")) else []
    m = CARD.match(label)
    if m and CJK.search(m.group(1)):
        p = m.group(1).strip()
        if p in BANKS:
            return []
        if not any(w in p for w in BANKISH):
            return [("bankjunk", "")]
        d = sorted((ed(p, b), b) for b in BANKS)
        best = [b for e, b in d if e == d[0][0] and e <= 2]
        return [("bank", b + m.group(2) + m.group(3)) for b in best]
    k = norm(label)
    if not k or k in tg:
        return []
    n = max(1, freq[k])
    if label[-1:] == "干" and norm(label[:-1]) in tg:
        return [("plus", tg[norm(label[:-1])][0] + "+")]
    if DIGIT.search(label) or "*" in label:
        return []
    by_len = tg_by_len(tg)
    if len(k) >= 4:
        out += [("text", t) for kk, (t, c) in by_len.get(len(k), []) if sub1(k, kk) and c >= 5 * n]
    if not out and len(k) >= 6:
        pre = set()
        for L, items in by_len.items():
            if L <= len(k):
                continue
            for kk, (t, c) in items:
                if sub1(k, kk[: len(k)]) and c >= 5 * n:
                    # 用原文的开头(全角逗号这些照原样), 打分写法和 kk 的开头一样的那一段
                    raw = next((t[:j] for j in range(1, len(t) + 1) if norm(t[:j]) == kk[: len(k)]), None)
                    if raw:
                        pre.add(raw)
        out += [("pre", p) for p in sorted(pre)]
    if not out:
        out += [("affix", t) for kk, (t, c) in by_len.get(len(k) - 1, []) if len(kk) >= 5 and (k[1:] == kk or k[:-1] == kk) and c >= 5 * n]
    return out


_BY_LEN = {}


def tg_by_len(tg: dict) -> dict:
    key = id(tg)
    if key not in _BY_LEN:
        d = defaultdict(list)
        for kk, v in tg.items():
            d[len(kk)].append((kk, v))
        _BY_LEN[key] = d
    return _BY_LEN[key]


def cmd_scan(a) -> None:
    if a.out.exists():
        die("RELABEL_SCAN_EXISTS", f"{a.out} 已经在了, 不再写(要重写先挪走)")
    if len(a.lists) != 2:
        die("RELABEL_SCAN_ARGS", "--lists 要正好两份: v3 那份(k)在前, 老切图那份(i)在后")
    charset = set(c for c in (a.lists[0] / "charset.txt").read_text(encoding="utf-8").split("\n") if c)
    if not charset:
        die("RELABEL_SCAN_NO_CHARSET", f"{a.lists[0] / 'charset.txt'} 是空的")
    tg = targets(a.known, charset)
    data = {}
    for li, d in zip("ki", a.lists):
        for which in ("train", "val"):
            data[(li, which)] = read_list(d, which)
            if which == "train" and not data[(li, which)]:
                die("RELABEL_SCAN_EMPTY", f"{d / 'train.txt'} 没有能用的行")
    freq = Counter(norm(t) for _p, t in data[("k", "train")] + data[("k", "val")])
    labels = sorted({t for rows in data.values() for _p, t in rows})
    cand, off_charset = {}, 0
    for t in labels:
        c = candidates(t, tg, freq)
        keep = [(r, new) for r, new in c if all(ch in charset for ch in new)]   # 改成的字都得在字表里, 不然训练时会被悄悄丢掉
        off_charset += len(c) - len(keep)
        if keep:
            cand[t] = keep
    count = defaultdict(Counter)
    for (li, which), rows in data.items():
        for p, t in rows:
            if t in cand:
                count[t][(li, which)] += 1
    # 样图: 每条规则只给最常见的 --per-rule 个候选各拷 --samples 张(打星的名字可能有几千个不同的)
    with_samples = set()
    for rule in sorted({r for c in cand.values() for r, _n in c}):
        ts = sorted((t for t, c in cand.items() if c[0][0] == rule), key=lambda t: (-sum(count[t].values()), t))
        with_samples |= set(ts[: a.per_rule])
    paths = defaultdict(list)
    for p, t in data[("k", "train")] + data[("k", "val")]:
        if t in with_samples and len(paths[t]) < a.samples:
            paths[t].append(p)
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "img").mkdir(parents=True)
    rows_out, missing = [], 0
    order = sorted(cand, key=lambda t: (cand[t][0][0], -sum(count[t].values()), t))
    for ci, t in enumerate(order, 1):
        names = []
        for j, p in enumerate(paths[t], 1):
            src = Path(p)
            if src.is_file():
                fn = f"c{ci:04d}_{j}{src.suffix or '.png'}"
                shutil.copyfile(src, tmp / "img" / fn)
                names.append(fn)
            else:
                missing += 1
        for rule, new in cand[t]:
            rows_out.append([ci, rule, t, new, count[t][("k", "train")], count[t][("k", "val")], count[t][("i", "train")],
                             count[t][("i", "val")], " ".join(names)])
    with (tmp / "candidates.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cid", "rule", "old", "new", "n_k_train", "n_k_val", "n_i_train", "n_i_val", "samples"])
        w.writerows(rows_out)
    os.replace(tmp, a.out)
    by = Counter()
    for ci, rule, t, new, a1, a2, a3, a4, _s in rows_out:
        by[rule] += a1 + a2
    print(f"RELABEL_SCAN_STATS lines_k {len(data[('k', 'train')]) + len(data[('k', 'val')])} lines_i "
          f"{len(data[('i', 'train')]) + len(data[('i', 'val')])} distinct_labels {len(labels)} targets {len(tg)} "
          f"candidate_labels {len(cand)} candidate_rows {len(rows_out)} off_charset {off_charset} sample_missing {missing}", flush=True)
    print("RELABEL_SCAN_RULES " + " ".join(f"{r}={n}" for r, n in sorted(by.items())), flush=True)
    print("RELABEL_SCAN_OK", flush=True)


def sha256_of(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_decisions(p: Path) -> dict:
    d = {"pin": None, "rules": set(), "keep": set(), "bank": {}, "accept": set(), "rewrite": {}}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        f = ln.split("\t")
        if f[0] == "pin" and len(f) == 2:
            d["pin"] = f[1]
        elif f[0] == "rule" and len(f) == 2 and f[1] in ("slash", "mask", "bankjunk"):
            d["rules"].add(f[1])
        elif f[0] == "keep" and len(f) == 2:
            d["keep"].add(f[1])
        elif f[0] == "bankprefix" and len(f) == 3 and f[1] and f[2] in BANKS:
            d["bank"][f[1]] = f[2]
        elif f[0] == "accept" and len(f) == 2:
            d["accept"].add(f[1])
        elif f[0] == "rewrite" and len(f) == 3 and f[2]:
            d["rewrite"][f[1]] = f[2]
        else:
            die("RELABEL_APPLY_BAD_DECISION", f"看不懂的决定: {ln!r}")
    if not d["pin"]:
        die("RELABEL_APPLY_BAD_DECISION", "决定文件里没有 pin")
    return d


def build_plan(rows: list, dec: dict) -> dict:
    """候选表 + 人审过的决定 -> {旧标签: (规则, 新标签 或 None = 整条不要)}。没写到的候选一律不动; 同一个旧标签两个不同的决定就停。"""
    plan = {}
    for x in rows:
        cid, rule, old, new = x["cid"], x["rule"], x["old"], x["new"]
        if cid in dec["keep"]:
            continue
        if cid in dec["rewrite"]:
            act = (rule, dec["rewrite"][cid])
        elif rule == "slash" and "slash" in dec["rules"]:
            act = (rule, new)
        elif rule in ("mask", "bankjunk") and rule in dec["rules"]:
            act = (rule, None)
        elif rule == "bank":
            m = CARD.match(old)
            p = m.group(1).strip() if m else ""
            if p not in dec["bank"]:
                continue
            act = (rule, dec["bank"][p] + m.group(2) + m.group(3))
        elif rule in ("text", "pre", "affix", "plus") and cid in dec["accept"]:
            act = (rule, new)
        else:
            continue
        if old in plan and plan[old] != act:
            die("RELABEL_APPLY_CONFLICT", f"同一个标签两个不同的决定: {old!r} {plan[old]} {act}")
        plan[old] = act
    return plan


def cmd_apply(a) -> None:
    if len(a.lists) != 2 or len(a.out) != 2:
        die("RELABEL_APPLY_ARGS", "--lists 和 --out 都要正好两个(v3 那份在前)")
    for o in a.out:
        if o.exists() or o.with_name(o.name + ".tmp").exists():
            die("RELABEL_APPLY_EXISTS", f"{o} 已经在了, 不覆盖")
    dec = load_decisions(a.decisions)
    cand_file = a.scan / "candidates.tsv"
    have = sha256_of(cand_file)
    if have != dec["pin"]:
        die("RELABEL_APPLY_PIN", f"{cand_file} 的 sha256 {have} 和决定文件里的 {dec['pin']} 不一样: 不是审过的那份候选表")
    with cand_file.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    plan = build_plan(rows, dec)
    charset = set(c for c in (a.lists[0] / "charset.txt").read_text(encoding="utf-8").split("\n") if c)
    bad = [new for _r, new in plan.values() if new is not None and not all(ch in charset for ch in new)]
    if bad:
        die("RELABEL_APPLY_OFF_CHARSET", f"{len(bad)} 个新标签有字表外的字, 例如 {bad[:3]}")
    if any(new == "" for _r, new in plan.values()):
        die("RELABEL_APPLY_EMPTY_LABEL", "有新标签是空的")
    for src, out in zip(a.lists, a.out):
        if (src / "charset.txt").read_bytes() != (a.lists[0] / "charset.txt").read_bytes():
            die("RELABEL_APPLY_CHARSET_DIFFERS", f"{src} 的字表和 {a.lists[0]} 的不一样")
        tmp = out.with_name(out.name + ".tmp")
        tmp.mkdir(parents=True)
        stats = Counter()
        applied = Counter()
        for which in ("train", "val"):
            p = src / f"{which}.txt"
            if not p.exists():
                continue
            lines_in = p.read_text(encoding="utf-8").splitlines()
            lines_out, paths_in, paths_out = [], [], []
            for ln in lines_in:
                if "\t" not in ln:
                    lines_out.append(ln)
                    continue
                path, txt = ln.split("\t", 1)
                paths_in.append(path)
                act = plan.get(txt)
                if act is None:
                    lines_out.append(ln)
                    paths_out.append(path)
                    continue
                rule, new = act
                applied[(rule, txt, new, which)] += 1
                if new is None:
                    stats[f"{which}_drop_{rule}"] += 1
                    continue
                stats[f"{which}_rewrite_{rule}"] += 1
                lines_out.append(f"{path}\t{new}")
                paths_out.append(path)
            dropped = sum(v for k, v in stats.items() if k.startswith(f"{which}_drop_"))
            if len(lines_out) != len(lines_in) - dropped or not set(paths_out) <= set(paths_in):
                die("RELABEL_APPLY_COUNT", f"{p}: 进 {len(lines_in)} 行, 去掉 {dropped}, 出 {len(lines_out)} 行 对不上")
            (tmp / f"{which}.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
            stats[f"{which}_in"] = len(lines_in)
            stats[f"{which}_out"] = len(lines_out)
        shutil.copyfile(src / "charset.txt", tmp / "charset.txt")
        if sha256_of(tmp / "charset.txt") != sha256_of(src / "charset.txt"):
            die("RELABEL_APPLY_CHARSET_COPY", "拷过去的字表和原来的不一样")
        with (tmp / "_relabel_applied.tsv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["rule", "old", "new", "list", "n"])
            for (rule, old, new, which), n in sorted(applied.items()):
                w.writerow([rule, old, "DROP" if new is None else new, which, n])
        (tmp / "_relabel_version.txt").write_text(f"decisions {sha256_of(a.decisions)}\ncandidates {have}\nsource {src}\n", encoding="utf-8")
        # --expect: 本机用同一张候选表算好的总数(每份清单都一样), 对不上就不留这个目录
        if a.expect:
            got = {f"{w}_{act}": sum(v for k, v in stats.items() if k.startswith(f"{w}_{act}_")) for w in ("train", "val") for act in ("rewrite", "drop")}
            want = dict(kv.split("=") for kv in a.expect)
            diff = {k: (got.get(k), int(v)) for k, v in want.items() if got.get(k) != int(v)}
            if diff:
                shutil.rmtree(tmp)
                die("RELABEL_APPLY_UNEXPECTED", f"{out.name}: 改动的行数和本机算的不一样 {diff}")
        os.replace(tmp, out)
        print(f"RELABEL_APPLY_STATS {out.name} " + " ".join(f"{k}={v}" for k, v in sorted(stats.items())), flush=True)
    print(f"RELABEL_APPLY_PLAN labels {len(plan)} rewrite {sum(1 for r, n in plan.values() if n is not None)} "
          f"drop {sum(1 for r, n in plan.values() if n is None)}", flush=True)
    print("RELABEL_APPLY_OK", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--lists", type=Path, nargs="+", required=True, help="v3 清单(train-n1009k)在前, 老切图清单(train-n1009i)在后")
    s.add_argument("--known", type=Path, required=True, help="known_texts.py 写的目录(texts.csv)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--samples", type=int, default=3, help="每个候选拷几张样图")
    s.add_argument("--per-rule", type=int, default=150, help="每条规则只给最常见的这么多个候选拷样图")
    s = sub.add_parser("apply")
    s.add_argument("--lists", type=Path, nargs="+", required=True, help="和 scan 一样: v3 清单在前, 老切图清单在后(只读)")
    s.add_argument("--out", type=Path, nargs="+", required=True, help="两个新清单目录(不能已经存在), 顺序和 --lists 一样")
    s.add_argument("--scan", type=Path, required=True, help="scan 写的目录(candidates.tsv)")
    s.add_argument("--decisions", type=Path, required=True, help="人审过的决定(training/relabel_decisions_50.tsv)")
    s.add_argument("--expect", nargs="*", default=[], help="每份清单该改 / 去掉多少行, 例如 train_rewrite=3381 train_drop=1609 val_rewrite=208 val_drop=71")
    a = ap.parse_args()
    {"scan": cmd_scan, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
