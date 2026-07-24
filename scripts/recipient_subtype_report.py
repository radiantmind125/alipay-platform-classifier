r"""收款方(recipient)分子类精度分析:把 ocr_evaluate 出的 comparisons.jsonl 里 recipient 那些,
按 姓名 / 表情emoji / 银行卡尾号 三类分开,各算准确率——定位到底哪种子类拖后腿,好定路由
(比如:姓名+emoji 走 Paddle,银行卡尾号那种结构化的小模型也许扛得住)。只读。

用法:
  python scripts\recipient_subtype_report.py --comparisons D:\download\ocr_eval\comparisons.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 表情 / 符号:常见 emoji 段 + 变体选择符(全部用 \U 转义,避免字面字符出问题)
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoticons / symbols / transport / supplemental
    "\U00002600-\U000027BF"   # 杂项符号 + dingbats
    "\U00002190-\U000021FF"   # 箭头
    "\U00002B00-\U00002BFF"   # 杂项符号与箭头
    "\U0000FE00-\U0000FE0F"   # 变体选择符
    "\U00002B50\U00002764\U0000203C\U00002049"  # 零散常见符号
    "]"
)
# 银行卡尾号:含"尾号/银行/储蓄卡/信用卡/借记卡"或 括号里 3-4 位数字
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
    ap = argparse.ArgumentParser(description="收款方分子类(姓名/emoji/银行卡)精度分析")
    ap.add_argument("--comparisons", type=Path, required=True, help="ocr_evaluate 输出的 comparisons.jsonl")
    ap.add_argument("--field", default="recipient_field", help="要拆的字段(默认收款方)")
    ap.add_argument("--examples", type=int, default=5, help="每子类打印几条失败样例")
    args = ap.parse_args()

    rows = []
    for line in args.comparisons.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [r for r in rows if r.get("field") == args.field]
    if not rows:
        print(f"comparisons.jsonl 里没有 field={args.field} 的记录")
        return

    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[subtype(str(r.get("reference_text", "")))].append(r)

    print(f"{args.field} 共 {len(rows)} 条,按子类拆(参考文本来自 PaddleOCR):\n")
    print(f"  {'子类':<10}{'条数':>6}{'逐字对':>9}{'语义对':>9}{'含OOV':>9}{'CER':>8}")
    print("  " + "-" * 50)
    for st in ("姓名", "银行卡尾号", "表情emoji"):
        b = buckets.get(st, [])
        if not b:
            continue
        n = len(b)
        raw = sum(bool(r.get("raw_exact")) for r in b)
        sem = sum(bool(r.get("semantic_exact")) for r in b)
        oov = sum(bool(r.get("reference_has_oov_character")) for r in b)
        ref_chars = sum(int(r.get("reference_characters", 0)) for r in b)
        cer = sum(int(r.get("cer_edits", 0)) for r in b) / max(1, ref_chars)
        print(f"  {st:<10}{n:>6}{raw / n:>9.1%}{sem / n:>9.1%}{oov / n:>9.1%}{cer:>8.3f}")

    print("\n各子类失败样例(参考 → 小模型识别):")
    for st in ("姓名", "银行卡尾号", "表情emoji"):
        fails = [r for r in buckets.get(st, []) if not r.get("raw_exact")][: args.examples]
        if fails:
            print(f"  [{st}]")
            for r in fails:
                print(f"    {str(r.get('reference_text',''))!r} -> {str(r.get('candidate_text',''))!r}")

    print(
        "\n提示:逐字对=完全一样;语义对=按字段规则抽取后一致。"
        "\n  若某子类连语义对也很低 -> 那种必须走 Paddle;"
        "\n  若只有'姓名/emoji'低、'银行卡尾号'高 -> 只把姓名/emoji 那类路由到 Paddle 即可。"
    )


if __name__ == "__main__":
    main()
