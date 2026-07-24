r"""收款方现状分析,只用我们自己的 OCR 输出,不依赖同事的模型。

从全量 OCR 结果里把收款方文本拿出来,按 姓名 / 表情emoji / 银行卡尾号 分类,看:
  各占多少、Paddle 自己的置信度多高、字符有多杂,生僻字比例多少。
用来估哪种子类小模型会难,好先把路由的大方向定下来。这是能自己做的部分;
自训识别器 vs Paddle 的精确对比,还是得要同事那份训好的模型。只读。

用法,指向全量 ocr paddle 的输出目录:
  python scripts\recipient_composition_report.py --results-dir D:\download\ann10k
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U00002B50\U00002764\U0000203C\U00002049"
    "]"
)
_BANK = re.compile(r"尾号|银行|储蓄卡|信用卡|借记卡|\(\s*\d{3,4}\s*\)|（\s*\d{3,4}\s*）")


def subtype(text: str) -> str:
    if _EMOJI.search(text):
        return "表情emoji"
    if _BANK.search(text):
        return "银行卡尾号"
    return "姓名"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="收款方现状分析,只用自己的 OCR 输出")
    ap.add_argument("--results-dir", type=Path, required=True, help="全量 ocr paddle 输出目录")
    ap.add_argument("--rare-threshold", type=int, default=2, help="全局出现次数不超过它就算生僻字")
    ap.add_argument("--low-conf", type=float, default=0.90, help="低于它算低置信")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    recips: list[tuple[str, float | None]] = []
    for f in args.results_dir.rglob("*.json"):
        if f.name.startswith("inference_"):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec = (d.get("fields") or {}).get("recipient") or {}
        raw = rec.get("raw")
        if isinstance(raw, str) and raw.strip():
            conf = rec.get("ocr_confidence")
            recips.append((raw.strip(), conf if isinstance(conf, (int, float)) else None))

    if not recips:
        print(f"没找到带 OCR 文本的收款方(要用全量 --ocr paddle 的输出):{args.results_dir}")
        return

    char_freq: Counter[str] = Counter(c for text, _ in recips for c in text)
    rare = {c for c, n in char_freq.items() if n <= args.rare_threshold}

    buckets: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    for text, conf in recips:
        buckets[subtype(text)].append((text, conf))

    n_all = len(recips)
    print(f"收款方共 {n_all} 条,来自我们自己的 OCR 输出,按子类拆:\n")
    print(f"  {'子类':<10}{'条数':>7}{'占比':>7}{'均置信':>8}{'低置信':>8}{'含生僻':>8}{'均长':>6}{'字符种数':>9}")
    print("  " + "-" * 64)
    for st in ("姓名", "银行卡尾号", "表情emoji"):
        b = buckets.get(st, [])
        if not b:
            continue
        n = len(b)
        confs = [c for _, c in b if c is not None]
        mean_conf = sum(confs) / len(confs) if confs else float("nan")
        low = sum(1 for c in confs if c < args.low_conf) / len(confs) if confs else float("nan")
        rare_rate = sum(1 for t, _ in b if any(ch in rare for ch in t)) / n
        mean_len = sum(len(t) for t, _ in b) / n
        uniq = len({ch for t, _ in b for ch in t})
        print(f"  {st:<10}{n:>7}{n / n_all:>7.1%}{mean_conf:>8.3f}{low:>8.1%}{rare_rate:>8.1%}{mean_len:>6.1f}{uniq:>9}")

    print("\n各子类低置信样例(Paddle 自己都没把握的,任何模型都难):")
    for st in ("姓名", "银行卡尾号", "表情emoji"):
        low_ex = sorted(
            (item for item in buckets.get(st, []) if item[1] is not None and item[1] < args.low_conf),
            key=lambda x: x[1],
        )[: args.examples]
        if low_ex:
            print(f"  [{st}]")
            for text, conf in low_ex:
                print(f"    {conf:.3f}  {text!r}")

    print(
        "\n怎么读:含生僻字比例高、字符种数多的子类,小模型很可能难,倾向走 Paddle;"
        "\n  结构化、生僻字少的,比如银行卡尾号,小模型大概率能扛。"
        "\n  Paddle 自己置信度就低的,是数据本身难,换谁都读不准,那种要么走 Paddle 要么直接留空待复核。"
        "\n  这里给的是问题规模和难度的估计;小模型 vs Paddle 的精确精度,还要拿同事的模型跑 ocr_evaluate 才有。"
    )


if __name__ == "__main__":
    main()
