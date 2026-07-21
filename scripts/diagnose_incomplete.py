r"""诊断 --require-complete 跳过的那批图:到底为什么检测凑不齐 5 个字段。

回答两个问题:
  1. 主要缺哪个字段?(已知 time=状态栏时钟 是最脆的)
  2. 这些失败图是**被裁过的**(顶部状态栏没了 → 时钟本就不在,放宽要求即可)
     还是**完整截图但检测/矫正漏了**(→ 检测那边可优化)?
     用分辨率规则粗判:命中 iPhone/安卓标准分辨率 = 大概率是完整截图;对不上 = 大概率被裁/异形。

另存少量失败图的顶部条,肉眼一看便知状态栏在不在。只读,不改任何东西。

用法:
  python scripts\diagnose_incomplete.py --errors D:\download\TempFakeResults_v2_device\inference_errors.jsonl --out-dir runs\diag_incomplete --sample 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.bootstrap import resolution_platform  # noqa: E402

_MISS = re.compile(r"missing=([^;]+)")


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", type=Path, required=True, help="inference_errors.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None, help="另存失败图顶部条的目录")
    ap.add_argument("--sample", type=int, default=30)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.errors.read_text(encoding="utf-8").splitlines() if l.strip()]
    field_miss: Counter[str] = Counter()
    res_cat: Counter[str] = Counter()
    only_time = 0
    incomplete = 0
    other_err: Counter[str] = Counter()
    saved = 0
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for r in rows:
        msg = str(r.get("message", ""))
        if "incomplete detection" not in msg:
            other_err[str(r.get("error_type", "?"))] += 1  # 不是"缺字段"的失败(另有原因)
            continue
        incomplete += 1
        m = _MISS.search(msg)
        miss = [x.strip() for x in m.group(1).split(",")] if m else []
        for f in miss:
            field_miss[f] += 1
        if miss == ["time"]:
            only_time += 1
        src = r.get("source", "")
        try:
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            w, h = im.size
            res_cat[resolution_platform(w, h)] += 1
            if args.out_dir and saved < args.sample:
                strip = np.asarray(im)[: max(1, int(h * 0.12))]  # 顶部12%,含状态栏+一点下方
                Image.fromarray(strip).save(args.out_dir / f"{Path(src).stem}.jpg", quality=90)
                saved += 1
        except Exception:
            res_cat["<读不了>"] += 1

    print(f"require-complete 失败(缺字段)= {incomplete} 张" + (f";其它错误 {dict(other_err)}" if other_err else ""))
    print("\n【1】各字段缺失次数(越高越脆):")
    for f, c in field_miss.most_common():
        print(f"    {f:24} {c}")
    print(f"    —— 只缺 time(时钟)的 = {only_time}/{incomplete}")
    print("\n【2】失败图的分辨率归类(判是否被裁):")
    tot = sum(res_cat.values()) or 1
    for k, c in res_cat.most_common():
        name = {"ios": "命中iPhone分辨率(完整截图)", "android": "命中安卓面板宽(完整截图)",
                "abstain": "分辨率对不上(疑似被裁/异形)"}.get(k, k)
        print(f"    {name:32} {c}  ({c/tot:.0%})")
    full = res_cat.get("ios", 0) + res_cat.get("android", 0)
    print("\n【判读】")
    if full / tot >= 0.6:
        print("    多数是标准分辨率的完整截图 → 状态栏/时钟其实在,是**检测或矫正漏了**")
        print("    → 优化方向在检测那边(比如矫正别把顶部状态栏裁掉);或放宽 require-complete。")
    else:
        print("    多数分辨率对不上 → 大概率**顶部被裁/异形图**,时钟本就不在")
        print("    → 这类放宽 require-complete(把 time 设为可选)即可救回;它们也偏可疑,可当假图信号。")
    if args.out_dir:
        print(f"\n顶部条样张另存 {saved} 张 -> {args.out_dir}(打开看状态栏在不在)")


if __name__ == "__main__":
    main()
