r"""经理传到服务器上的原图按文件名日期点个数(只读, 什么都不写): 每个目录多少张、多少 GB, 按月分, 10 月 / 11 月按天分,
以及有多少张是最后一次扫描以后才到的(文件名不在 --seen 的扫描名单里)。用来回答"11 月的下完了没有、10 月缺哪几天、放不放得下"。
    python training/download_census.py --roots E:\BlueImages D:\download2\OtherImages --seen <所有 *_scan.jsonl ...> [--days 202610 202611]
输出全是 ASCII:
    ROOT <目录> FILES <张数> GB <大小> NEW <没扫过的张数> NEW_GB <大小> SEEN_NAMES <名单里的文件名数>
    MONTH <yyyymm> FILES <张数> GB <大小> NEW <没扫过的张数>
    DAY <yyyymmdd> FILES <张数> GB <大小> NEW <没扫过的张数>          (只列 --days 给的月份)
    NODATE FILES <文件名里没有日期的张数>
目录不在: ROOT <目录> MISSING; 最后一行 DOWNLOAD_CENSUS_DONE
大小是文件本身的大小; 服务器上 分图_9月 这类目录是硬链接, 删原图腾不出这部分空间(链接还在)。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seen_names import EXTS, load as load_seen  # noqa: E402

DATE = re.compile(r"_(20\d{2})(\d{2})(\d{2})\d{6}")
GB = 1024 ** 3


def iter_images(root: str):
    """(文件名, 字节数); 用 scandir, Windows 上文件大小在目录项里就有, 不用再逐个 stat。"""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False) and os.path.splitext(e.name)[1].lower() in EXTS:
                        yield e.name, e.stat(follow_symlinks=False).st_size
                except OSError:
                    continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--seen", type=Path, nargs="+", required=True)
    ap.add_argument("--days", nargs="*", default=["202610", "202611"], help="按天列的月份(yyyymm)")
    a = ap.parse_args()
    seen, _notes, _w = load_seen(a.seen)
    for root in a.roots:
        if not os.path.isdir(root):
            print(f"ROOT {root} MISSING", flush=True)
            continue
        files = size = new = new_size = nodate = 0
        month = defaultdict(lambda: [0, 0, 0])      # 张数, 字节, 没扫过的张数
        day = defaultdict(lambda: [0, 0, 0])
        for name, b in iter_images(root):
            files += 1
            size += b
            is_new = name not in seen
            if is_new:
                new += 1
                new_size += b
            m = DATE.search(name)
            if not m:
                nodate += 1
                continue
            ym, ymd = m.group(1) + m.group(2), m.group(1) + m.group(2) + m.group(3)
            for key, tab in ((ym, month), (ymd, day)):
                if tab is day and ym not in a.days:
                    continue
                tab[key][0] += 1
                tab[key][1] += b
                tab[key][2] += int(is_new)
        print(f"ROOT {root} FILES {files} GB {size / GB:.1f} NEW {new} NEW_GB {new_size / GB:.1f} SEEN_NAMES {len(seen)}", flush=True)
        for k in sorted(month):
            n, b, nn = month[k]
            print(f"MONTH {k} FILES {n} GB {b / GB:.1f} NEW {nn}")
        for k in sorted(day):
            n, b, nn = day[k]
            print(f"DAY {k} FILES {n} GB {b / GB:.2f} NEW {nn}")
        print(f"NODATE FILES {nodate}", flush=True)
    print("DOWNLOAD_CENSUS_DONE")


if __name__ == "__main__":
    main()
