"""按页名把答案集里某个颜色的整页原图拷出来(只读原图, 只写 --out), 给试用版的整页核对用。和 copy_pages.py 一样的拷法和核对,
只是不经过配对目录的 _pairs.csv(测试目录的段在那里找不到, copy_pages 会报 no_pairs_row): 页名直接按 page_type.csv 的 path 列找,
找到的路径不在了再换盘符(recrop.resolve_path)。
页 = <gold-root>\\<集合>\\_gold3_meta.csv 里 colour 是 --colour 的每一页(page 列), 每页记这一页上编号最小的那段。
    python training/copy_pages_byname.py --gold-root E:\\_move_logs --set gold4 --colour blue --page-index <page_type.csv> [...] --out E:\\_move_logs\\pages_blue3
出 --out\\_pages.csv(set, id, page, how, sha256), 最后一行纯 ASCII: COPY_PAGES_BYNAME_OK n / COPY_PAGES_BYNAME_MISSING n。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recrop import load_index, resolve_path  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-root", type=Path, required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--colour", required=True)
    ap.add_argument("--page-index", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        print(f"{a.out} 已经有东西了, 不覆盖")
        sys.exit(2)
    first = {}
    for x in csv.DictReader(open(a.gold_root / a.set / "_gold3_meta.csv", encoding="utf-8-sig")):
        if x["colour"] == a.colour and (x["page"] not in first or int(x["id"]) < int(first[x["page"]])):
            first[x["page"]] = x["id"]
    index = load_index([str(p) for p in a.page_index])
    print(f"  page_type 索引 {len(index):,} 个文件名, {a.set} {a.colour} 页 {len(first)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, done, missing = [], 0, 0
    for page in sorted(first):
        q = index.get(page.lower())
        if not q:
            rows.append({"set": a.set, "id": first[page], "page": page, "how": "not_in_index", "sha256": ""})
            missing += 1
            continue
        path, how = resolve_path(q, index)
        if not path:
            rows.append({"set": a.set, "id": first[page], "page": page, "how": how, "sha256": ""})
            missing += 1
            continue
        dst = a.out / page
        shutil.copyfile(path, dst)
        h = sha(dst)
        if h != sha(Path(path)):
            print(f"COPY_PAGES_BYNAME_STOP copy differs {page}")
            sys.exit(2)
        done += 1
        rows.append({"set": a.set, "id": first[page], "page": page, "how": how, "sha256": h})
    with open(a.out / "_pages.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "id", "page", "how", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"COPY_PAGES_BYNAME_OK {done}" if not missing else f"COPY_PAGES_BYNAME_MISSING {missing} (copied {done})")


if __name__ == "__main__":
    main()
