"""查"字段被裁掉了头尾一个字"到底是怎么发生的。

起因
----
field_recall 拆出来白图有 6 条是这一类, 占全部损失的三分之一:

    订单号    -> 单号      掉了开头的 订   (这个字段的损失**全部**是这一类)
    账单详情  -> 账单详    掉了末尾的 情
    全部账单  -> 账单      掉了开头的 全部

★★★★★ 这一类**不用重训**, 所以值得先修。但修之前得知道是哪一种:

    切成两段了      订 和 单号 各自成段, 拼页面文字时中间插了空格
                   -> 改拼接就行
    段边界压进字里   段的 x0 就在 订 的右边, 那个字根本没进图
                   -> 要放宽边界
    行边界压进字里   整行裁的时候左边就切掉了
                   -> 要改行裁剪, 动的是共用几何

★ 三种修法完全不同, 看一眼坐标就能分清。**别再猜了。**

用法
----
    python why_truncated.py --dump D:\\alipay-ai-data\\dump_white_b.jsonl --field 订单号
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    recs = []
    with a.dump.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    fd = a.field
    print("=" * 70)
    print("  WHY TRUNCATED  (ASCII only - safe to paste)")
    print("=" * 70)
    print(f"  field = {fd}")
    print()

    shown = 0
    for r in recs:
        ours = (r.get("ours") or "").replace(" ", "")
        if fd in ours:
            continue
        segs = r.get("segs") or []
        # 找那个"是字段名真子串"的段
        hit = None
        for s in segs:
            t = (s.get("t") or "").strip()
            if t and t != fd and t in fd and len(t) >= len(fd) - 2:
                hit = s
                break
        if hit is None:
            continue
        shown += 1
        if shown > a.n:
            break
        print("-" * 70)
        print(f"  {r.get('file')}")
        print(f"  读成了 {hit.get('t')!r}   少的是 "
              f"{fd.replace(hit.get('t') or '', '')!r}")
        row = hit.get("row")
        same = [s for s in segs if s.get("row") == row]
        same.sort(key=lambda s: s.get("x0", 0))
        print(f"  这一行(row={row}) 一共切出 {len(same)} 段:")
        for s in same:
            mark = "  <== 就是它" if s is hit else ""
            print(f"    seg{s.get('seg')}  x {s.get('x0'):>5}-{s.get('x1'):>5}"
                  f"  y {s.get('y0'):>5}-{s.get('y1'):>5}"
                  f"  宽 {s.get('x1',0)-s.get('x0',0):>4}"
                  f"  pin={s.get('pin')}  {(s.get('t') or '')[:24]!r}{mark}")
        # ★ 判一判是哪一种
        left = [s for s in same if s.get("x1", 0) <= hit.get("x0", 0)]
        print()
        if left:
            gap = hit.get("x0", 0) - max(s.get("x1", 0) for s in left)
            print(f"  ★ 左边还有段, 最近的那段右边缘离它 {gap} 像素")
            print(f"    -> 少掉的字**可能在左边那一段里**, 那就是**切成两段**了,")
            print(f"       改拼接就行, 不用动几何")
        else:
            print(f"  ★ 左边没有别的段, 段的 x0 = {hit.get('x0')}")
            print(f"    -> 少掉的字**根本没被切出来**, 是边界压进字里了,")
            print(f"       要放宽边界(段边界或行边界)")
    print("-" * 70)
    if not shown:
        print("  这一份 dump 里没有这个字段的截断样本")
    else:
        print(f"  共 {shown} 例")
        print()
        print("  ★★ 三种情况的修法:")
        print("     左边有段且很近   -> 改拼接, 最便宜")
        print("     左边没段         -> 放宽段边界")
        print("     整行都偏         -> 改行裁剪, 动共用几何, 最贵")


if __name__ == "__main__":
    main()
