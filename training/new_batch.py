r"""经理新传的一批图: 有多少张、有多少是新的、里面有多少张拼音图。

为什么单独写这个
----------------
新图传进了**老路径**(D:\download2\BlueImages、D:\download2\OtherImages),
老图挪到了 E:\BlueImages、E:\OtherImages。要回答的是三件事:

    1. 新传了多少张, 其中多少是**真新的**(老库里没有同名的)
    2. 这批是什么: 日期范围、蓝底还是白底、有没有读不出来的
    3. 里面有多少张拼音图 —— 要和以前的数**能比**

★★★★★ 第 3 条要能比, 就必须和以前**一模一样**地打分、用**各自的阈值**:

    蓝图库  阈值 0.12   以前 487,045 张挑出 3,622 张(0.74%); 0.20 之上 1,909 张(0.39%)
    白图库  阈值 0.06   以前挑出 7,159 张

   两个库的阈值是**分别**按分数段肉眼核出来的, 同一个分数在两个库里意思正好相反
   (蓝图的小徽章冒充拼音, 白图的真拼音被一堆数字稀释)。所以打分直接用
   pick_pinyin.py 原样的那一套, 这里**不另写打分**, 只做清点和汇总。

★★ 结果写到**新的**文件里, 不要沿用老的名单 jsonl ——
   pick_pinyin 的续跑是**按路径**认的, 新图和老图在同一个 D:\download2\... 路径下,
   用老文件会把新旧混在一起。summary 会检查名单里的路径是不是都在要统计的目录里。

★ 这个脚本**只读**图库目录, 不往里面写任何东西。

用法
----
    # 第一步: 清点(几秒到一两分钟)
    python new_batch.py inventory --new D:\download2\BlueImages --old E:\BlueImages
    # 第二步: 用 pick_pinyin 原样打分(几小时, 可中断续跑)
    python pick_pinyin.py --root D:\download2\BlueImages --out D:\alipay-ai-data\newbatch_20260925\blue_scan.jsonl
    # 第三步: 汇总
    python new_batch.py summary --scan D:\alipay-ai-data\newbatch_20260925\blue_scan.jsonl ^
        --new D:\download2\BlueImages --old E:\BlueImages --kind blue
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import EXTS  # noqa: E402

# 以前的数, 拿来对比
OLD = {
    "blue": {"total": 487_045, "hits": {0.12: 3_622, 0.20: 1_909}},
    "white": {"total": None, "hits": {0.06: 7_159}},
}
# 各库自己的阈值(主阈值放第一个)
THR = {"blue": [0.12, 0.20], "white": [0.06]}
DATE = re.compile(r"_(20\d{2})(\d{2})(\d{2})\d{6}")


def walk(root: Path):
    """递归列出所有文件, 返回 (图片路径列表, 其它文件的扩展名计数, 子目录数)。"""
    imgs, other, ndir = [], Counter(), 0
    for dirpath, dirnames, names in os.walk(root):
        ndir += len(dirnames)
        for n in names:
            ext = Path(n).suffix.lower()
            if ext in EXTS:
                imgs.append(os.path.join(dirpath, n))
            else:
                other[ext or "(无扩展名)"] += 1
    return imgs, other, ndir


def page_kind(path: str) -> str:
    """页头颜色: 蓝底 / 白底 / 其它 / 读不出。

    规则和版面默认(pinyin_ocr._blue_header)、以及这一程所有分析分白图蓝图用的完全一样:
    页面 5%~20% 高度那一段的平均颜色 B > 150 且 B > R + 60 算蓝底, 三个通道都 > 200 算白底。
    用 1/4 尺寸解码 —— 只看平均颜色, 不用原图。
    """
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    if im is None:
        return "读不出"
    H = im.shape[0]
    b, g, r = im[int(H * 0.05):int(H * 0.20)].reshape(-1, 3).mean(0)
    if b > 150 and b > r + 60:
        return "蓝底"
    if min(b, g, r) > 200:
        return "白底"
    return "其它"


def months(names) -> Counter:
    c = Counter()
    for n in names:
        m = DATE.search(n)
        c[f"{m.group(1)}-{m.group(2)}" if m else "(文件名里没日期)"] += 1
    return c


def days(names) -> list[str]:
    out = []
    for n in names:
        m = DATE.search(n)
        if m:
            out.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return sorted(out)


def _sha1(path: str) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detail(new_imgs, old_imgs, fresh, overlap, a) -> None:
    """同名的是不是真的同一张图(逐字节比), 以及日期精确到天。

    ★★ "同名 = 同一张图"只是推断。文件名里是凭证号 + 时间戳, 按理同名就是同一张,
       但要是重新截过图、重新压缩过, 同名底下可能是不一样的内容。
       要跟经理说"传上来的基本都是老图", 得先逐字节比过。
    """
    print("  ---- 细看 ----")
    dn, do, dv = days(fresh), days(os.path.basename(p) for p in old_imgs), days(overlap)
    for tag, d in (("老目录", do), ("新目录里同名的", dv), ("真新的", dn)):
        if d:
            print(f"  {tag:<14}日期 {d[0]} 到 {d[-1]}   ({len(d):,} 张)")
    if dn:
        c = Counter(dn)
        print("  真新的按天:")
        for k in sorted(c):
            print(f"    {k}   {c[k]:>6,}")
    print()

    if overlap:
        old_by_name = {}
        for p in old_imgs:
            old_by_name.setdefault(os.path.basename(p), p)
        new_by_name = {}
        for p in new_imgs:
            new_by_name.setdefault(os.path.basename(p), p)
        random.seed(a.seed)
        names = random.sample(overlap, min(a.hash_sample, len(overlap)))
        same = diff = 0
        example = None
        for n in names:
            pn, po = new_by_name[n], old_by_name[n]
            if os.path.getsize(pn) == os.path.getsize(po) and _sha1(pn) == _sha1(po):
                same += 1
            else:
                diff += 1
                if example is None:
                    example = (n, os.path.getsize(pn), os.path.getsize(po))
        print(f"  同名的随机抽 {len(names)} 对逐字节比:  一模一样 {same} 对,  不一样 {diff} 对")
        if example:
            print(f"    不一样的例子: {example[0]}   新 {example[1]:,} 字节  老 {example[2]:,} 字节")
        print()


def cmd_inventory(a) -> None:
    print("=" * 72)
    print(f"  NEW BATCH INVENTORY  (ASCII only - safe to paste)")
    print("=" * 72)
    for d in (a.new, a.old):
        if not d.exists():
            print(f"  目录不在: {d}")
            return
    print(f"  新: {a.new}")
    print("  正在列新目录...", flush=True)
    new_imgs, other, ndir = walk(a.new)
    print("  正在列老目录...", flush=True)
    old_imgs, _o, _d = walk(a.old)
    old_names = {os.path.basename(p) for p in old_imgs}
    new_names = [os.path.basename(p) for p in new_imgs]
    dup_in_new = sum(c - 1 for c in Counter(new_names).values() if c > 1)
    overlap = [n for n in new_names if n in old_names]
    fresh = [n for n in new_names if n not in old_names]

    print()
    print(f"  新目录  图片 {len(new_imgs):>10,} 张    子目录 {ndir}")
    if other:
        print(f"          不是图片的文件 {sum(other.values()):,} 个: "
              + ", ".join(f"{k} {v}" for k, v in other.most_common(6)))
    print(f"  老目录  图片 {len(old_imgs):>10,} 张   ({a.old})")
    print()
    print(f"  ★ 新目录里和老目录**同名**的   {len(overlap):>10,} 张   <- 不是新数据")
    print(f"  ★ **真新的**(老目录里没有)     {len(fresh):>10,} 张")
    if dup_in_new:
        print(f"    新目录里自己重名的(不同子目录)  {dup_in_new:,} 张")
    print()

    mf = months(fresh)
    mo = months(old_names)
    print("  真新的那些, 按文件名里的日期分月:")
    for k in sorted(mf):
        print(f"    {k:<18}{mf[k]:>10,}")
    known = sorted(k for k in mo if k[0].isdigit())
    if known:
        print(f"  (老目录的日期范围: {known[0]} 到 {known[-1]})")
    print()

    if a.detail:
        _detail(new_imgs, old_imgs, fresh, overlap, a)

    # 抽样解码: 读不读得出来、蓝底白底各多少
    random.seed(a.seed)
    pool = [p for p in new_imgs if os.path.basename(p) not in old_names]
    sample = random.sample(pool, min(a.sample, len(pool)))
    if sample:
        kinds = Counter(page_kind(p) for p in sample)
        print(f"  真新的里随机抽 {len(sample)} 张看页头颜色:")
        for k in ("蓝底", "白底", "其它", "读不出"):
            if kinds[k]:
                print(f"    {k:<6}{kinds[k]:>6}   ({kinds[k] / len(sample):.1%})")
        print()

    # 预估打分要多久(以前 48.7 万张约 7 小时, 约 19 张/秒)
    hrs = len(new_imgs) / 19 / 3600
    print(f"  ★ 用 pick_pinyin 全部打分, 按以前的速度估约 {hrs:.1f} 小时(可中断续跑)")
    print("=" * 72)


def cmd_summary(a) -> None:
    rows = []
    with a.scan.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:                   # noqa: BLE001
                continue                        # 半行(上次被砍断的)
    print("=" * 72)
    print(f"  NEW BATCH PINYIN SUMMARY  (ASCII only - safe to paste)   {a.kind}")
    print("=" * 72)
    if not rows:
        print(f"  名单是空的: {a.scan}")
        return

    def norm(p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

    # ★★ 防混一: 名单里的路径必须都在 --new 这个目录**里面**
    #   (要带分隔符比, 不然 BlueImages2 也会被当成 BlueImages 下面的)
    root = norm(str(a.new)).rstrip("\\/") + os.sep
    outside = [r for r in rows if not norm(r["path"]).startswith(root)]
    if outside:
        print(f"  ★★★★★ 名单里有 {len(outside):,} 条不在 {a.new} 下面, 例:")
        print(f"         {outside[0]['path']}")
        print("     这份名单混进了别的目录, 数不能用。换个新文件重扫。")
        return

    # ★★ 防混二: 名单里的文件**现在得还在**。
    #   沿用老名单的话, 老图的路径同样是 D:\download2\BlueImages\...,
    #   第一条挡不住; 但老图已经挪去 E 盘了, 这些路径现在都不存在 —— 靠这个认出来。
    new_imgs, _o, _d = walk(a.new)
    exist = {norm(p) for p in new_imgs}
    gone = [r for r in rows if norm(r["path"]) not in exist]
    if gone:
        frac = len(gone) / len(rows)
        print(f"  ★★ 名单里有 {len(gone):,} 条({frac:.1%})对应的文件现在不在了, 例:")
        print(f"       {gone[0]['path']}")
        if frac > 0.01:
            print("  ★★★★★ 这么多对不上, 多半是**沿用了老名单**(老图已经挪去 E 盘)。")
            print("     数不能用。换个新文件名重扫。")
            return
        print("     少量对不上(扫完之后删过几张?), 这些不计入。")
        gone_set = {id(r) for r in gone}
        rows = [r for r in rows if id(r) not in gone_set]
        print()

    old_names = set()
    if a.old and a.old.exists():
        _imgs, _o, _d = walk(a.old)
        old_names = {os.path.basename(p) for p in _imgs}

    ok = [r for r in rows if r.get("score") is not None]
    bad = Counter(r.get("err") or "?" for r in rows if r.get("score") is None)
    fresh = [r for r in ok if os.path.basename(r["path"]) not in old_names]
    # 该扫的是"真新的"那些(扫的时候用了 --skip-names-from 的话, 老图本来就不扫)
    n_fresh_imgs = sum(1 for p in new_imgs if os.path.basename(p) not in old_names)
    n_fresh_rows = sum(1 for r in rows if os.path.basename(r["path"]) not in old_names)
    print(f"  新目录图片 {len(new_imgs):,} 张, 其中真新的 {n_fresh_imgs:,} 张;  "
          f"名单里 {len(rows):,} 条"
          f"{'  <- 还没扫完' if n_fresh_rows < n_fresh_imgs else ''}")
    print(f"  量到分数的 {len(ok):,}    读不出/太小 {sum(bad.values()):,}"
          + (f"  ({', '.join(f'{k} {v}' for k, v in bad.items())})" if bad else ""))
    print(f"  其中真新的(老目录里没有同名) {len(fresh):,}")
    print()

    thrs = THR[a.kind]
    main_t = thrs[0]
    print(f"  ★ 拼音图(按这个库以前定的阈值, 真新的里面数):")
    for t in thrs:
        h = sum(1 for r in fresh if r["score"] >= t)
        pct = h / len(fresh) if fresh else 0
        old = OLD[a.kind]
        o = old["hits"].get(t)
        cmp = ""
        if o is not None and old["total"]:
            cmp = f"   (以前 {o:,}/{old['total']:,} = {o / old['total']:.2%})"
        elif o is not None:
            cmp = f"   (以前挑出 {o:,} 张)"
        tag = "  <- 主阈值" if t == main_t else ""
        print(f"    >= {t:.2f}   {h:>8,} 张   占 {pct:.2%}{cmp}{tag}")
    print()

    # 分数段: 主阈值上面那一段最容易混进假的, 单独看数
    edges = sorted(set([main_t, 0.15, 0.20, 0.25, 0.35, 1.01]))
    edges = [e for e in edges if e >= main_t]
    a_sc = np.array([r["score"] for r in fresh]) if fresh else np.array([])
    print("  过主阈值的, 按分数段:")
    for lo, hi in zip(edges, edges[1:]):
        n = int(((a_sc >= lo) & (a_sc < hi)).sum())
        print(f"    {lo:.2f} - {hi:.2f}   {n:>8,}")
    print()

    # ★★ 页头颜色: 新传的批次里要是混了另一种页, 阈值就该跟着页走, 不是跟着文件夹走
    hits = [r for r in fresh if r["score"] >= min(THR['white'][0], THR['blue'][0])]
    kinds = Counter()
    by_kind_hit = Counter()
    for r in hits:
        k = page_kind(r["path"])
        kinds[k] += 1
        t = THR["blue"][0] if k == "蓝底" else THR["white"][0]
        if r["score"] >= t:
            by_kind_hit[k] += 1
    if hits:
        print(f"  分数 >= {min(THR['white'][0], THR['blue'][0]):.2f} 的 {len(hits):,} 张, 按页头颜色:")
        for k in ("蓝底", "白底", "其它", "读不出"):
            if kinds[k]:
                t = THR["blue"][0] if k == "蓝底" else THR["white"][0]
                print(f"    {k:<6}{kinds[k]:>8,} 张   按它自己那一类的阈值({t:.2f})过的 "
                      f"{by_kind_hit[k]:>8,} 张")
        tot = sum(by_kind_hit.values())
        print(f"  ★ 阈值跟着页走(蓝底 0.12, 其余 0.06)合计 {tot:,} 张")
        print()

    print("  ★★★★★ 数只是第一步。阈值是在老图上核出来的, 新批次要是版面变了, 精度会变。")
    print("     必须按分数段抽图肉眼核一遍(命令见说明), 真有拼音的比例对得上才算数。")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("inventory", help="清点: 多少张、多少真新的、日期、蓝白")
    p1.add_argument("--new", type=Path, required=True)
    p1.add_argument("--old", type=Path, required=True)
    p1.add_argument("--sample", type=int, default=300, help="抽多少张看页头颜色")
    p1.add_argument("--seed", type=int, default=17)
    p1.add_argument("--detail", action="store_true",
                    help="同名的抽样逐字节比, 日期精确到天")
    p1.add_argument("--hash-sample", type=int, default=200)
    p2 = sub.add_parser("summary", help="pick_pinyin 打完分之后汇总拼音图数")
    p2.add_argument("--scan", type=Path, required=True, help="pick_pinyin 的 --out 那个 jsonl")
    p2.add_argument("--new", type=Path, required=True)
    p2.add_argument("--old", type=Path, default=None)
    p2.add_argument("--kind", choices=["blue", "white"], required=True)
    a = ap.parse_args()
    if a.cmd == "inventory":
        cmd_inventory(a)
    else:
        cmd_summary(a)


if __name__ == "__main__":
    main()
