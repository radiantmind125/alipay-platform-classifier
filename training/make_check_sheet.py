"""把要人工核的行拼成一张图, 顺带打一份紧凑报告。

为什么要这个脚本
----------------
闸门那一步只有人眼能判 —— 工具只知道"读出来了", 不知道"读对了"。
但一张张点开 `input\\` 里的图太慢, 而且没法把结果发出来讨论。

★ 这里把随机抽的几行**竖着拼成一张图**, 左边编号, 再按同样的编号
  把 OCR 读出来的文字打出来。一张图 + 一份文字, 对着看就行。

两张表分别在核什么
------------------
    --kind usable     判成可用的      -> 核**读对了没有**, 以及标签里有没有漏拼音
    --kind rejected   判成不可用的    -> 核**是不是错杀了**

★★ 第二张同样要紧。过滤是"连续 4 个以上拉丁字母就不要", 这条可能
   误伤本来就带英文的行(商户名之类)。错杀多了白白少一批训练数据。

用法
----
    python make_check_sheet.py --pairs 配对目录
    python make_check_sheet.py --pairs 配对目录 --kind rejected
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_pinyin_labels import classify  # noqa: E402

HAN = re.compile(r"[一-鿿]")


def _cjk_font(size: int):
    """找一个能画中文的字体。找不到就返回 None, 退回只出图不写字。"""
    for name in ("msyh.ttc", "simsun.ttc", "msyhl.ttc", "simhei.ttf"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            try:
                from PIL import ImageFont
                return ImageFont.truetype(str(p), size)
            except Exception:                      # noqa: BLE001
                continue
    return None


def stitch(rows, pair_dir: Path, dst: Path) -> tuple[int, int]:
    """把每行的 input 裁图竖着拼起来, 左边写编号, **右边把 OCR 读出来的文字画上去**。

    ★★★★★ 文字一定要画进图里, 不能只打在控制台。
       服务器控制台按 GBK 显示而我们输出 UTF-8, 贴回来是
           鈽呪槄 涓嬩竴姝?*蹇呴』**浜哄伐鏍稿嚑鍗佹潯
       关键的标签文字全糊了, 等于没法核。**画进图里就不过编码这一关。**

    ★ cv2.putText 画不了中文, 所以中文这部分走 PIL + 系统字体。
    """
    font = _cjk_font(22)
    imgs, W = [], 0
    for i, r in enumerate(rows, 1):
        p = pair_dir / "input" / r["file"]
        if not p.exists():
            continue
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.copyMakeBorder(im, 2, 2, 46, 2, cv2.BORDER_CONSTANT,
                                value=(210, 210, 210))
        cv2.putText(im, str(i), (6, im.shape[0] // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 160), 2)
        imgs.append(im)
        W = max(W, im.shape[1])
    if not imgs:
        return 0, 0

    TEXT_W = 760 if font else 0
    imgs = [cv2.copyMakeBorder(m, 0, 0, 0, W - m.shape[1] + TEXT_W,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for m in imgs]
    out = np.vstack(imgs)

    if font:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        y = 0
        for i, (m, r) in enumerate(zip(imgs, rows), 1):
            h = m.shape[0]
            dr.line([(W, y), (W, y + h)], fill=(190, 190, 190), width=1)
            dr.text((W + 10, y + max(0, (h - 26) // 2)),
                    (r.get("text") or "")[:42], font=font, fill=(150, 0, 0))
            y += h
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    cv2.imencode(".png", out)[1].tofile(str(dst))
    return out.shape[1], out.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--kind", choices=["usable", "rejected"], default="usable")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--seed", type=int, default=7,
                    help="固定种子, 这样重跑核的是同一批")
    a = ap.parse_args()

    man = a.pairs / "_pairs_labeled.csv"
    if not man.exists():
        print(f"清单不在: {man}")
        print("   先跑 fill_pinyin_labels.py")
        return
    allrows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
    if not allrows:
        print("清单是空的")
        return

    # ---------- 紧凑报告 ----------
    n = len(allrows)
    use = [r for r in allrows if r.get("usable") == "1"]
    py = sum(1 for r in allrows if r.get("has_pinyin") == "1")
    nlab = len(list((a.pairs / "label").glob("*.png"))) if (a.pairs / "label").exists() else 0
    nin = len(list((a.pairs / "input").glob("*.png"))) if (a.pairs / "input").exists() else 0
    srcs = len({r["source"] for r in allrows})
    # ★ 用**过滤本身**来复查, 不要自己另写一条正则。
    #   自己写过一版"有 4 个以上连续拉丁字母就算漏", 结果把
    #       对方账户 倪萍（薄）dhh**@outlook.com
    #   这种邮箱也数成漏拼音了 —— 邮箱是回单上本来就有的东西, 过滤里
    #   专门按 @ 放行过。两处判据各写一遍就会对不上, 报出假警报。
    leak = sum(1 for r in use if classify(r.get("text") or "") != "可用")
    han = sum(len(HAN.findall(r.get("text") or "")) for r in use)

    print("=" * 58)
    print("  报告 —— 把这一段整个贴回来")
    print("=" * 58)
    print(f"  清单 {n:,} 行   label {nlab:,} 张   input {nin:,} 张")
    print(f"  三个数一样: {'是' if n == nlab == nin else '★ 不一样, 停下来'}")
    print(f"  原图 {srcs:,} 张,  平均每张 {n/max(1,srcs):.1f} 行")
    print(f"  上面有拼音的 {py:,} ({py/n:.0%})")
    print(f"  可用 {len(use):,} ({len(use)/n:.0%})   可用里汉字共 {han:,} 个")
    print(f"  ★ 可用里还带拼音的 {leak}  —— 这个必须是 0")
    print()

    # ---------- 抽样拼图 ----------
    if a.kind == "usable":
        pool = [r for r in allrows
                if r.get("usable") == "1" and r.get("has_pinyin") == "1"]
        what = "判成可用的(核读对了没有)"
    else:
        pool = [r for r in allrows if r.get("usable") != "1"
                and (r.get("text") or "").strip()]
        what = "判成不可用的(核是不是错杀)"
    if not pool:
        print(f"没有可抽的行({what})")
        return

    random.seed(a.seed)
    pick = random.sample(pool, min(a.n, len(pool)))
    dst = a.pairs / f"_check_{a.kind}.png"
    w, h = stitch(pick, a.pairs, dst)

    print("=" * 58)
    print(f"  {what}   抽了 {len(pick)} 行")
    print("=" * 58)
    for i, r in enumerate(pick, 1):
        print(f"  {i:2d}. {r.get('text','')}")
    print()
    print(f"拼好的图 ({w}x{h}) -> {dst}")
    print()
    print("★ 把上面这份文字, 和那张图, 一起贴回来。")


if __name__ == "__main__":
    main()
