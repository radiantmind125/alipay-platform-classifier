r"""训练清单里常见的整段文字(只读): 字段名、银行名、固定句子这些"固定文字"各出现多少次, 给 OCR 读完以后对照用(step 49)。

为什么(2026-10-28)
------------------
gold3 上新流程剩下的 59 条重要字段错误里, 15 条是固定文字读错: 交易方式 读成 娄易武, 付款方式 读成 符式, 平安 读成 苹安,
余额宝(转出资金付款) 读乱。这类文字在训练清单里出现成千上万次, 拿它们的次数做一张"认得的文字"表, 读完以后用模型自己的分数
在表里挑, 比只靠贪心解码稳。表只从训练清单来, 不看任何答案集。

    python training/known_texts.py --list E:\alipay-ai-data\train-n1009k --out E:\_move_logs\known_texts [--min 5]

写 <out>/texts.csv        text, blue, white, other, total            整段文字, total >= --min
   <out>/card_prefix.csv  prefix, dots, blue, white, other, total    像 "农业银行储蓄卡(1234)张三" 的段: 括号里 4 位数字前面那一截,
                                                                     dots = 那一截后面有没有省略号(银行名太长被截断时手机上会加)
只读清单, 只写 --out 目录(先写到 .tmp 再改名)。最后打印一行纯 ASCII 的 KNOWN_TEXTS_OK; 不对就打印原因并以非 0 退出。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

CARD = re.compile(r"^(.*?)([.。·…]*)[(（]\s*\d{4}\s*[)）]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def colour(path: str) -> str:
    d = Path(path.replace("\\", "/")).parent.parent.name.lower()
    hits = [c for c in ("blue", "white") if c in d]
    return hits[0] if len(hits) == 1 else "other"


def write_csv(p: Path, cols: list, rows: list) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True, help="训练清单目录(里面有 train.txt, 可以有 val.txt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min", type=int, default=5, help="整段文字至少出现几次才写进表")
    a = ap.parse_args()
    if a.out.exists():
        print(f"  ★ {a.out} 已经在了, 不再写(要重写先挪走)", flush=True)
        print("KNOWN_TEXTS_EXISTS", flush=True)
        sys.exit(2)
    if not (a.list / "train.txt").is_file():
        print(f"  ★ 找不到 {a.list / 'train.txt'}", flush=True)
        print("KNOWN_TEXTS_NO_LIST", flush=True)
        sys.exit(2)
    texts, cards = Counter(), Counter()
    n = bad = 0
    for which in ("train", "val"):
        p = a.list / f"{which}.txt"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\r\n")
                if "\t" not in ln:
                    bad += 1
                    continue
                path, txt = ln.split("\t", 1)
                txt = txt.strip()
                if not txt:
                    bad += 1
                    continue
                n += 1
                c = colour(path)
                texts[(txt, c)] += 1
                m = CARD.match(txt)
                if m and CJK.search(m.group(1)):
                    cards[(m.group(1).strip(), int(bool(m.group(2))), c)] += 1
    if n == 0:
        print("  ★ 清单里一行能用的都没有", flush=True)
        print("KNOWN_TEXTS_EMPTY", flush=True)
        sys.exit(2)

    def table(counter, key_len):
        by = {}
        for k, v in counter.items():
            by.setdefault(k[:key_len], Counter())[k[key_len]] += v
        return by

    tt = table(texts, 1)
    rows_t = sorted(([k[0], v["blue"], v["white"], v["other"], sum(v.values())] for k, v in tt.items() if sum(v.values()) >= a.min),
                    key=lambda r: (-r[4], r[0]))
    cc = table(cards, 2)
    rows_c = sorted(([k[0], k[1], v["blue"], v["white"], v["other"], sum(v.values())] for k, v in cc.items() if sum(v.values()) >= a.min),
                    key=lambda r: (-r[5], r[0], r[1]))
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    write_csv(tmp / "texts.csv", ["text", "blue", "white", "other", "total"], rows_t)
    write_csv(tmp / "card_prefix.csv", ["prefix", "dots", "blue", "white", "other", "total"], rows_c)
    os.replace(tmp, a.out)
    print(f"KNOWN_TEXTS_STATS lines={n} skipped={bad} distinct={len(tt)} kept={len(rows_t)} card_lines={sum(cards.values())} "
          f"card_prefixes_kept={len(rows_c)} min={a.min}", flush=True)
    print("KNOWN_TEXTS_OK", flush=True)


if __name__ == "__main__":
    main()
