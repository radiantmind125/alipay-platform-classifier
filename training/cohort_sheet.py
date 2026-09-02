r"""把**报出来的图**和**同群真图**并排拼成一张图, 让人一眼看出差别。

为什么要这一步
--------------
2026-09-02 `cohort_check` 在八月那 108 张上出了结论:
**90 张的负号比同群 60 张对照全都长**(同群 = 同分辨率 + 同编码器指纹 = 同机型同版本),
跨 45 个群都成立 —— 渲染变体解释不了这个。

**但那仍然只是数字。** 这个项目里每一次结论被推翻, 都是因为**去看了图**:
二维码混进离群榜首、149 连发那组其实是 Windows 提示条、两个最大的重复组是赌博 app 页面。
**统计说"离群", 眼睛说"是不是真的不一样" —— 后者才是能拿给别人看的证据。**

这个脚本做什么
--------------
对每一张报出来的图, 在**同一个群**里挑几张真图, 把金额区**并排**摆出来:

    [报出的那张]   <- 上面
    [同群真图 1]
    [同群真图 2]
    [同群真图 3]

同机型、同版本、同分辨率, **唯一的差别只剩"这一张有没有被改过"**。
负号明显更长 -> 肉眼就能确认; 看不出差别 -> 那这个判据就还不能信。

为什么不直接改 `cohort_check`
-----------------------------
`cohort_check` 要给整池按分辨率建索引, 在服务器上**实测跑了 93.7 分钟**(72.7 万张冷缓存)。
只是为了看几张图再跑一遍不值。
**这里改成随机抽一小部分池子来找对照** —— 常见分辨率在池子里都有几万张,
随机抽 5 万张就足够为每个群凑出十来张对照, 时间从一个半小时降到几分钟。

用法
----
  python training/cohort_sheet.py --flagged D:\probe\cohort_aug.csv ^
      --pool D:\download2\OtherImages --since 20260801 ^
      --top 8 --out D:\probe\cohort_sheet.png

**只读**: 只读图片, 只往 --out 写。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from encoder_fingerprint import fingerprint      # noqa: E402
from locate_blue import locate_amount_auto       # noqa: E402
from minus_outlier import measure as measure_minus  # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{8})\d{6}")


def _amount_crop(path: str, tile_h: int = 78):
    """裁出金额区, 统一高度好并排比。"""
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    loc, _pg = locate_amount_auto(rgb)
    if not loc:
        return None
    x0, y0, x1, y1, _ = loc
    px = int((x1 - x0) * 0.42)
    py = int((y1 - y0) * 0.30)
    sub = rgb[max(0, y0 - py):y1 + py, max(0, x0 - px):x1 + px]
    if sub.size == 0:
        return None
    im = Image.fromarray(sub)
    if im.height < 4:
        return None
    return im.resize((max(1, int(im.width * tile_h / im.height)), tile_h), Image.LANCZOS)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="报出的图 vs 同群真图, 并排拼图")
    ap.add_argument("--flagged", type=Path, required=True,
                    help="cohort_check 出的 csv(要有 image_name / bar_width / 判读)")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--since", type=str, default=None)
    ap.add_argument("--top", type=int, default=8, help="挑几张报出的图来展示")
    ap.add_argument("--ctrl", type=int, default=3, help="每张配几张同群真图")
    ap.add_argument("--sample", type=int, default=50000,
                    help="从池子里随机抽这么多张来找对照(不建全池索引, 省时间)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.flagged.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise SystemExit("!! --flagged 是空的")
    # 优先挑"同群里最离群"的, 按 bar_width 从大到小
    key = "判读" if "判读" in rows[0] else None
    cand = [r for r in rows if (not key or r[key].startswith("★★"))]
    if not cand:
        cand = rows
    cand.sort(key=lambda r: -float(r["bar_width"]))
    cand = cand[: args.top]
    print(f"要展示 {len(cand)} 张(优先取同群最离群的)")

    print(f"清点 {args.pool} ...", flush=True)
    idx = {}
    for dp, _, fns in os.walk(args.pool):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                idx[fn] = os.path.join(dp, fn)
    print(f"池里 {len(idx):,} 张", flush=True)

    flagged_names = {r["image_name"] for r in rows}
    pool_names = [n for n in idx if n not in flagged_names]
    if args.since:
        pool_names = [n for n in pool_names
                      if (_TS.search(n) and _TS.search(n).group(1) >= args.since)]
    random.Random(args.seed).shuffle(pool_names)
    pool_names = pool_names[: args.sample]
    print(f"随机抽 {len(pool_names):,} 张用来找同群对照(不建全池索引)", flush=True)

    # 抽样池按分辨率归档
    by_size = collections.defaultdict(list)
    for n in pool_names:
        try:
            with Image.open(idx[n]) as im:
                by_size[im.size].append(n)
        except Exception:
            continue
    print(f"抽样里共 {len(by_size)} 种分辨率", flush=True)

    blocks = []
    for r in cand:
        nm = r["image_name"]
        p = idx.get(nm)
        if not p:
            print(f"  跳过(池里找不到) {nm[:44]}")
            continue
        try:
            with Image.open(p) as im:
                sz = im.size
        except Exception:
            continue
        fp = fingerprint(p)
        tile = _amount_crop(p)
        if not tile:
            print(f"  跳过(裁不出金额区) {nm[:44]}")
            continue
        ctrls = []
        for cn in by_size.get(sz, []):
            if len(ctrls) >= args.ctrl:
                break
            cp = idx[cn]
            if fingerprint(cp) != fp:
                continue
            m = measure_minus(Path(cp))
            if not (m and m["bar_width"]):
                continue          # 没有负号的不能比
            t = _amount_crop(cp)
            if t:
                ctrls.append((t, float(m["bar_width"])))
        print(f"  {nm[:40]}  宽比 {float(r['bar_width']):.3f}  {sz[0]}x{sz[1]}  "
              f"找到同群对照 {len(ctrls)} 张", flush=True)
        if ctrls:
            blocks.append((r, sz, tile, ctrls))

    if not blocks:
        raise SystemExit("!! 一组都没凑齐, 试试把 --sample 调大")

    # ★ 标注要用中文字体, 否则 PIL 默认字体没有 CJK 字形, 中文全变成方框。
    #   实测本机 msyh.ttc / simsun.ttc 可用, simhei.ttf 不可用(会抛 OSError)。
    #   找不到就退回英文标注 —— 宁可标注是英文, 也不能是一排方框。
    def _font(sz: int):
        for f in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc",
                  r"C:\Windows\Fonts\msyhl.ttc"):
            try:
                return ImageFont.truetype(f, sz)
            except Exception:
                continue
        return None

    fnt = _font(15)
    if fnt is None:
        print("!! 找不到中文字体, 标注改用英文")
    L_HEAD = "{0}x{1}  报出 bar={2:.3f}" if fnt else "{0}x{1}  FLAGGED bar={2:.3f}"
    L_FLAG = "★ 报出的这张" if fnt else "* FLAGGED"
    L_CTRL = "同群真图 bar={0:.3f}" if fnt else "same-cohort genuine bar={0:.3f}"

    PAD, LAB = 8, 300
    W = LAB + max(max(t.width for t in [b[2]] + [c[0] for c in b[3]]) for b in blocks) + PAD * 2
    H = sum((len(b[3]) + 1) * (78 + PAD) + 30 for b in blocks)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    y = 0
    for r, sz, tile, ctrls in blocks:
        d.text((6, y + 4), L_HEAD.format(sz[0], sz[1], float(r["bar_width"])),
               fill=(180, 0, 0), font=fnt)
        y += 22
        sheet.paste(tile, (LAB, y))
        d.text((6, y + 28), L_FLAG, fill=(180, 0, 0), font=fnt)
        y += 78 + PAD
        for t, bw in ctrls:
            sheet.paste(t, (LAB, y))
            d.text((6, y + 28), L_CTRL.format(bw), fill=(0, 0, 0), font=fnt)
            y += 78 + PAD
        y += 8
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"\n拼图 -> {args.out}  ({sheet.size[0]}x{sheet.size[1]})")
    print("\n★ 怎么看: 每一组里, 上面红字标的是报出来的那张, 下面是**同机型同版本**的真图。")
    print("  负号明显更长 -> 判据成立, 而且这是能直接拿给别人看的证据。")
    print("  看不出差别 -> 那这个判据还不能信, 不管统计上多显著。")


if __name__ == "__main__":
    main()
