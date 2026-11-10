r"""数几个目录里各有多少张图从来没扫过(文件名不在 --seen 的扫描名单 / 目录里), 只读, 什么都不写。
给服务器上的整段命令用: 每个目录打一行纯 ASCII, 方便按数字判断新图在哪个目录、有没有到。
    python training/count_new_images.py --roots D:\download2\BlueImages D:\download2\OtherImages --seen <以前的扫描名单 ...>
每个目录一行: NEW_IMAGES <没扫过的张数> TOTAL <图片总数> DATES <文件名日期最早>..<最晚(只算没扫过的)> DIR <目录>
目录不在: NEW_IMAGES 0 TOTAL 0 DATES -..- MISSING <目录>; 最后一行 COUNT_NEW_DONE
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seen_names import EXTS, load as load_seen  # noqa: E402

DATE = re.compile(r"_(20\d{2})(\d{2})(\d{2})\d{6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--seen", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    seen, notes, _w = load_seen(a.seen)
    for s in notes:
        print(f"  seen: {s}")
    for root in a.roots:
        if not root.exists():
            print(f"NEW_IMAGES 0 TOTAL 0 DATES -..- MISSING {root}")
            continue
        n = tot = 0
        days = []
        for dirpath, _d, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() not in EXTS:
                    continue
                tot += 1
                if f in seen:
                    continue
                n += 1
                m = DATE.search(f)
                if m:
                    days.append(m.group(1) + m.group(2) + m.group(3))
        print(f"NEW_IMAGES {n} TOTAL {tot} DATES {min(days) if days else '-'}..{max(days) if days else '-'} DIR {root}", flush=True)
    print("COUNT_NEW_DONE")


if __name__ == "__main__":
    main()
