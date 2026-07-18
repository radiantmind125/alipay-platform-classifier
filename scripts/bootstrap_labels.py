"""用已验证的分辨率规则给整池产出自举标签，并用金标种子核对精度。

输入用 inspect.jsonl（已含每图 width/height），因此无需再解码图片，几秒跑完。
输出 bootstrap_labels.jsonl（file/label）+ 覆盖率；若有金标则给出规则在金标上的精度。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.bootstrap import resolution_platform  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=Path("runs/pool_20260701/inspect.jsonl"))
    ap.add_argument("--gold", type=Path, default=Path("gold/gold_seed_v1.json"))
    ap.add_argument("--output", type=Path, default=Path("runs/pool_20260701/bootstrap_labels.jsonl"))
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in args.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if "error" not in r]

    labeled: list[dict] = []
    counts: Counter[str] = Counter()
    for r in ok:
        label = resolution_platform(r["width"], r["height"])
        counts[label] += 1
        labeled.append({"file": r["file"], "width": r["width"], "height": r["height"], "label": label})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(ok)
    auto = counts["ios"] + counts["android"]
    print(f"总计 {n} 张")
    print(f"  ios       {counts['ios']:<5} = {counts['ios']/n*100:.1f}%")
    print(f"  android   {counts['android']:<5} = {counts['android']/n*100:.1f}%")
    print(f"  abstain   {counts['abstain']:<5} = {counts['abstain']/n*100:.1f}%（交像素模型）")
    print(f"  => 自动高精度标注覆盖 {auto}/{n} = {auto/n*100:.1f}%")
    print(f"标签写入 {args.output}")

    # 用金标核对规则精度。
    if args.gold.exists():
        gold = {row["file"]: row["platform"] for row in json.loads(args.gold.read_text(encoding="utf-8"))["labels"]}
        pred = {row["file"]: row["label"] for row in labeled}
        checked = correct = abstained = 0
        errors: list[str] = []
        for file, truth in gold.items():
            p = pred.get(file)
            if p is None:
                continue
            if p == "abstain":
                abstained += 1
                continue
            checked += 1
            if p == truth:
                correct += 1
            else:
                errors.append(f"{file}: 规则={p} 金标={truth}")
        print("\n== 分辨率规则在金标上的表现 ==")
        print(f"  金标 {len(gold)} 张：规则给出判定 {checked} 张，其中正确 {correct}，弃权 {abstained}")
        if checked:
            print(f"  规则精度（非弃权部分）：{correct}/{checked} = {correct/checked*100:.0f}%")
        for e in errors:
            print("  [错] " + e)


if __name__ == "__main__":
    main()
