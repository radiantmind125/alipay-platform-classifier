r"""把标定池里**不是干净截图**的挑出来 —— 翻拍照片、非收据页、尺寸离谱的(只读)。

为什么标定池必须先过这一道
--------------------------
标定池是下载器的**原始产出**, 不是"已核实的真实提交"。人工看了分数最高的 30 张之后发现:

| 类别 | 30 张里 | 例子 |
|---|---|---|
| **翻拍**(拿手机拍另一块屏幕) | 5 张 | 画面里有手、有边框、反光、倾斜 |
| **根本不是收据** | 3 张 | 豆浆机广告页 / 几乎全灰的空图 / 霓虹灯电竞酒店图 |

**这些被模型拒掉不是误杀。** SSP 最初就是**翻拍检测器**(经理那边的口径是"翻拍搞定,
留出 40 张翻拍 100% 抓到"), 它拒掉一张翻拍是功能正确, 不是判错。
空图和广告页更不是"收据提交", 压根不该进误杀分母。

而 1/5000 的阈值就是**第 20 高分**, 前 20 里有 4 张是翻拍 —— 也就是说
**工作点和对外报的误杀率, 有一部分是被不该计入的图顶出来的。**

判据(**不看模型分数**, 所以不构成循环论证)
------------------------------------------
直接复用 `gen_local_ai_edit._is_clean_screenshot`, 也就是当初**挑造样本源图时的同一把尺子**:

- 有相机 EXIF(曝光/光圈/ISO/闪光灯) -> 是拍摄的照片, 不是截图
- 短边 > 1500 -> 不像手机截图
- 长宽比不在 1.6~2.6 -> 不是竖屏截图版式

用同一个判据既选源图又清标定池, 口径才自洽。

用法
----
  python training/screenshot_filter.py --csv D:\probe\gen100k_blue\summary.csv D:\probe\gen100k_white\summary.csv ^
      --out D:\probe\not_screenshot.txt

**只读**: 只读图片, 只写你指定的名单文件。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_local_ai_edit import _OPT, _is_clean_screenshot  # noqa: E402


def why(p: Path) -> str | None:
    """返回不合格的原因; 合格返回 None。"""
    try:
        with Image.open(p) as im:
            w, h = im.size
            short = min(w, h)
            if not short:
                return "尺寸异常"
            aspect = max(w, h) / short
            try:
                ifd = im.getexif().get_ifd(0x8769)
            except Exception:
                ifd = {}
            if any(t in ifd for t in _OPT):
                return "有相机EXIF(翻拍/拍摄)"
            if short > 1500:
                return f"短边 {short} > 1500"
            if not (1.6 <= aspect <= 2.6):
                return f"长宽比 {aspect:.2f} 不在 1.6~2.6"
            return None
    except Exception as exc:  # noqa: BLE001
        return f"读不了({str(exc)[:40]})"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="挑出标定池里不是干净截图的图(只读)")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="真图 summary.csv, 可多个")
    ap.add_argument("--out", type=Path, required=True, help="不合格的文件名写这里, 一行一个")
    ap.add_argument("--require-located", action="store_true",
                    help="只看金额定位成功的那些(线路B 只在定位成功时出信号)")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    items: list[tuple[str, str]] = []       # (image_name, path)
    for c in args.csv:
        n = 0
        for r in csv.DictReader(open(c, encoding="utf-8-sig")):
            if args.require_located and (r.get("roi_amount_located") or "1").strip() != "1":
                continue
            nm = (r.get("image_name") or "").strip()
            p = (r.get("image") or "").strip()
            if nm and p:
                items.append((nm, p))
                n += 1
        print(f"  {c} -> {n} 行")
    if not items:
        raise SystemExit("CSV 里没读到图片")

    print(f"\n逐张检查 {len(items)} 张(只读文件头, 不解码像素, 很快)...", flush=True)
    bad: list[tuple[str, str]] = []
    reasons: Counter[str] = Counter()
    for i, (nm, p) in enumerate(items, 1):
        r = why(Path(p))
        if r:
            bad.append((nm, r))
            reasons[r.split("(")[0].split(" ")[0]] += 1
        if i % 20000 == 0:
            print(f"  已查 {i}/{len(items)}...", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(nm for nm, _ in bad) + ("\n" if bad else ""), encoding="utf-8")

    print(f"\n不合格 {len(bad)} 张 / {len(items)} = {len(bad)/len(items)*100:.2f}%")
    for k, v in reasons.most_common():
        print(f"  {k:24s} {v:6d}")
    print(f"\n-> {args.out}({len(bad)} 行)")
    if bad:
        print(f"\n前 {min(args.show, len(bad))} 个:")
        for nm, r in bad[: args.show]:
            print(f"  {nm:58s} {r}")
    print()
    print("怎么用: 把这份名单和 exclude_evidence / used_srcs / wm_hits 拼起来, 一起传给 --exclude。")
    print("★ 前提: **线上要真的把翻拍单独处理**(拒掉或走人工), 否则这里等于人为压低误杀率。")
    print("  翻拍本来就是 SSP 最早的目标, 拒掉它是功能正确, 不是误杀 —— 但口径要和线上一致。")


if __name__ == "__main__":
    main()
