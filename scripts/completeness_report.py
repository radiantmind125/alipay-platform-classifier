r"""统计 process-partial(不带 require-complete)输出的识别覆盖:多少张完整、多少只缺时钟、
多少真缺核心字段;各字段留空率;设备分布。给经理看"识别到什么程度"的一页统计。只读。

用法:
  python scripts\completeness_report.py --results-dir D:\download\fail_partial
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CORE_FIELDS = ("amount", "transfer_status", "recipient_field", "payment_method_field")
ALL_FIELDS = ("time", *CORE_FIELDS)
CN = {"time": "时间(时钟)", "amount": "金额", "transfer_status": "转账状态",
      "recipient_field": "收款方", "payment_method_field": "付款方式"}


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True, help="跑出来的结果目录(含每图 .json)")
    args = ap.parse_args(argv)

    files = [f for f in args.results_dir.rglob("*.json") if not f.name.startswith("inference_")]
    n = 0
    full = core_only = partial = 0
    empty = Counter()
    device = Counter()
    partial_missing = Counter()  # 部分桶里"缺了哪几个核心字段"的组合,表征这批到底缺在哪
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        detected = {det.get("label") for det in d.get("detections", []) if isinstance(det, dict)}
        n += 1
        for c in ALL_FIELDS:
            if c not in detected:
                empty[c] += 1
        core_ok = all(c in detected for c in CORE_FIELDS)
        if core_ok and "time" in detected:
            full += 1
        elif core_ok:
            core_only += 1
        else:
            partial += 1
            partial_missing[tuple(c for c in CORE_FIELDS if c not in detected)] += 1
        dev = (d.get("device") or {}).get("platform")
        if dev:
            device[dev] += 1

    if not n:
        print(f"目录里没找到结果 json:{args.results_dir}")
        return
    p = lambda x: f"{x/n*100:.1f}%"
    print(f"识别覆盖统计（{n} 张）\n")
    print(f"  完整(5字段全)          {full:6}  {p(full)}")
    print(f"  核心完整(仅缺时钟,4字段) {core_only:6}  {p(core_only)}")
    print(f"  —— 可用(核心字段齐全)    {full+core_only:6}  {p(full+core_only)}   ← 交易信息完整可用")
    print(f"  部分(核心字段有缺)       {partial:6}  {p(partial)}   ← 通知遮挡/非转账页/动画中,交空、待复核")
    print("\n各字段留空率(越高越常缺):")
    for c in ALL_FIELDS:
        print(f"    {CN[c]:14} {empty[c]:6}  {p(empty[c])}")
    if device:
        tot = sum(device.values())
        print("\n设备分布:", " / ".join(f"{k}={v}({v/tot*100:.0f}%)" for k, v in device.most_common()))
    if partial:
        print(f"\n部分({partial}张)缺了哪几个核心字段(表征这批到底缺在哪、要不要重训):")
        for combo, cnt in partial_missing.most_common(10):
            names = "+".join(CN[c] for c in combo) if combo else "(检测数异常/空)"
            print(f"    缺 {names:32} {cnt:6}  {cnt/partial*100:.1f}%")


if __name__ == "__main__":
    main()
