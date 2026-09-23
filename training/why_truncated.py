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


def _save(a, rec: dict, hit: dict, same: list, field: str) -> None:
    """把这一行连同段框画出来存成图 —— 坐标说不清的时候肉眼一看就清楚。

    ★ 左右各多留 3 个字的宽度, 这样"少掉的那个字"如果在框外也能看见。
    """
    import cv2
    import numpy as np
    if not a.src:
        return
    p = a.src / rec["file"]
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print(f"    (原图读不到: {p})")
        return
    pad = 130
    x0 = max(0, min(s.get("x0", 0) for s in same) - pad)
    x1 = min(im.shape[1], max(s.get("x1", 0) for s in same) + pad)
    y0 = max(0, hit.get("y0", 0) - 30)
    y1 = min(im.shape[0], hit.get("y1", 0) + 30)
    crop = im[y0:y1, x0:x1].copy()
    for s in same:
        c = (0, 0, 255) if s is hit else (0, 160, 0)
        cv2.rectangle(crop, (s.get("x0", 0) - x0, s.get("y0", 0) - y0),
                      (s.get("x1", 0) - x0, s.get("y1", 0) - y0), c, 2)
    a.save.mkdir(parents=True, exist_ok=True)
    out = a.save / f"{field}_{rec['file']}"
    cv2.imencode(".png", crop)[1].tofile(str(out.with_suffix(".png")))
    print(f"    存图 {out.with_suffix('.png').name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--src", type=Path, default=None,
                    help="原图目录。给了才能存图")
    ap.add_argument("--save", type=Path, default=None,
                    help="把这几段连同左右各留一截存成 png, 肉眼看字在不在框里")
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

    # ★★★★★ 一个字多宽, **从这份 dump 自己量**, 不要拿行高凑 ——
    #   行高被开关图标撑大过, 上次拿行高当字宽只救回 35%。
    #   只拿纯汉字的段来量: 宽度 / 字数。
    per = []
    for r in recs:
        for s in r.get("segs") or []:
            t = (s.get("t") or "").strip()
            if len(t) >= 2 and all("一" <= c <= "鿿" for c in t):
                w = s.get("x1", 0) - s.get("x0", 0)
                if w > 0:
                    per.append(w / len(t))
    per.sort()
    ppc = per[len(per) // 2] if per else 40.0
    print(f"  一个汉字约 {ppc:.1f} 像素 (从 {len(per):,} 个纯汉字段量的中位)")
    print()

    verdict: dict[str, int] = {}
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
        # ★★★★★ 判据**不能只看左边有没有段** ——
        #   "左边没有别的段"只证明**没有别的段捡走它**, 不证明**它不在这个段里**。
        #   我第一版就是这么判的, 判出来全是"边界压进字里", 是错的。
        #
        #   真正的判据是**宽度够不够**:
        #       段宽 / 一个字的宽  ≈ 字段名的字数   -> 字就在框里, 是**模型漏读**
        #       段宽 / 一个字的宽  ≈ 读出来的字数   -> 框确实小了, 是**边界**
        w = hit.get("x1", 0) - hit.get("x0", 0)
        t = hit.get("t") or ""
        holds = w / ppc
        print()
        print(f"  ★ 段宽 {w}, 一个字约 {ppc:.1f} 像素 -> 这个框装得下 "
              f"{holds:.1f} 个字")
        print(f"    读出来 {len(t)} 个字, 字段名 {len(fd)} 个字")
        if holds >= len(fd) - 0.45:
            print(f"    ★★★★★ **框够宽, 少掉的字就在框里** —— 是模型漏读了,")
            print(f"       不是边界问题。放宽边界没用, 要看识别。")
            verdict["模型漏读"] = verdict.get("模型漏读", 0) + 1
        elif holds <= len(t) + 0.45:
            print(f"    ★ 框只装得下读出来的那几个字 -> 确实是边界压进字里了")
            verdict["边界太窄"] = verdict.get("边界太窄", 0) + 1
        else:
            print(f"    ★ 介于两者之间, 说不准 -> 要看图")
            verdict["说不准"] = verdict.get("说不准", 0) + 1
        left = [s for s in same if s.get("x1", 0) <= hit.get("x0", 0)]
        if left:
            gap = hit.get("x0", 0) - max(s.get("x1", 0) for s in left)
            print(f"    (左边最近的段离它 {gap} 像素"
                  f"{' —— 太远, 不是同一组文字' if gap > ppc * 3 else ''})")
        if a.save:
            _save(a, r, hit, same, fd)
    print("-" * 70)
    if not shown:
        print("  这一份 dump 里没有这个字段的截断样本")
    else:
        print(f"  共 {shown} 例, 按**框够不够宽**判:")
        for k, n in sorted(verdict.items(), key=lambda kv: -kv[1]):
            fix = {"模型漏读": "框里有字模型没读出来 -> 看识别, 放宽边界没用",
                   "边界太窄": "框确实小了 -> 放宽段边界",
                   "说不准": "要肉眼看图 -> 加 --src 和 --save"}.get(k, "")
            print(f"    {k:<8}{n:>4}   {fix}")
        print()
        print("  ★★★★★ 只看'左边有没有段'是**判不出来**的 ——")
        print("     那只证明没有别的段捡走它, 不证明它不在这个段里。")
        print("     要按宽度判: 框装得下几个字。")


if __name__ == "__main__":
    main()
