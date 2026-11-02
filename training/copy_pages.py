"""把答案集里指定的段对应的原图拷出来(只读原图, 只写 --out), 给试用版的整页核对用。
--sel gold5:12,345  --sel gold6:7,89 ...  答案集编号(<gold-root>\\<集合>\\_gold_candidates.csv 的 src_path + _gold3_meta.csv 的 row_file)。
做法: 段的配对目录 = src_path 往上两级; 在那个目录的 _pairs.csv 里按 row_file 找到 source(原图路径),
原图搬过家的按 page_type.csv 的 path 列按文件名找、再换盘符找(和 recrop.resolve_path 一样)。每张原图只拷一次, 拷完逐字节核对。
    python training/copy_pages.py --gold-root E:\\_move_logs --sel gold5:1,2 --sel gold6:3 --page-index <page_type.csv> [...] --out E:\\_move_logs\\pages68
出 --out\\_pages.csv(set, id, row_file, page, how, sha256), 最后一行纯 ASCII: COPY_PAGES_OK n / COPY_PAGES_MISSING n。
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
    ap.add_argument("--sel", action="append", required=True, help="集合:编号,编号,...")
    ap.add_argument("--page-index", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        print(f"{a.out} 已经有东西了, 不覆盖")
        sys.exit(2)
    want = []
    for s in a.sel:
        st, ids = s.split(":", 1)
        want += [(st, i.strip()) for i in ids.split(",") if i.strip()]
    sets = sorted({st for st, _ in want})
    cand = {st: {x["id"]: x["src_path"] for x in csv.DictReader(open(a.gold_root / st / "_gold_candidates.csv", encoding="utf-8-sig"))} for st in sets}
    meta = {st: {x["id"]: x["row_file"] for x in csv.DictReader(open(a.gold_root / st / "_gold3_meta.csv", encoding="utf-8-sig"))} for st in sets}
    a.out.mkdir(parents=True, exist_ok=True)
    index = load_index([str(p) for p in a.page_index])
    print(f"  page_type 索引 {len(index):,} 个文件名, 要找 {len(want)} 段", flush=True)
    pairs_cache: dict[Path, dict] = {}
    rows, done, missing = [], {}, 0
    for st, i in want:
        src_path, row_file = cand[st].get(i), meta[st].get(i)
        if not src_path or not row_file:
            rows.append({"set": st, "id": i, "row_file": row_file or "", "page": "", "how": "no_such_id", "sha256": ""})
            missing += 1
            continue
        d = Path(src_path).parent.parent
        if d not in pairs_cache:
            pc = d / "_pairs.csv"
            pairs_cache[d] = {x["file"]: x["source"] for x in csv.DictReader(open(pc, encoding="utf-8-sig"))} if pc.exists() else {}
        src = pairs_cache[d].get(row_file)
        if not src:
            rows.append({"set": st, "id": i, "row_file": row_file, "page": "", "how": "no_pairs_row", "sha256": ""})
            missing += 1
            continue
        name = Path(src).name
        if name.lower() in done:
            rows.append({"set": st, "id": i, "row_file": row_file, "page": name, "how": "same_page", "sha256": done[name.lower()]})
            continue
        path, how = resolve_path(src, index)
        if not path:
            rows.append({"set": st, "id": i, "row_file": row_file, "page": name, "how": how, "sha256": ""})
            missing += 1
            continue
        dst = a.out / name
        shutil.copyfile(path, dst)
        h = sha(dst)
        if h != sha(Path(path)):
            print(f"COPY_PAGES_STOP copy differs {name}")
            sys.exit(2)
        done[name.lower()] = h
        rows.append({"set": st, "id": i, "row_file": row_file, "page": name, "how": how, "sha256": h})
    with open(a.out / "_pages.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "id", "row_file", "page", "how", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"COPY_PAGES_OK {len(done)}" if not missing else f"COPY_PAGES_MISSING {missing} (copied {len(done)})")


if __name__ == "__main__":
    main()
