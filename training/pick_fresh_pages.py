r"""从一批新导出的图里, 按固定种子随机挑 N 张从来没用过的蓝图拼音页, 拷到 --out(只读原图, 只写 --out), 给新一轮整页测试用。

页 = 这批新图的 pick_pinyin 扫描名单(--scan, 可以给几个)里 score >= --threshold 的图(蓝图 0.12: 以前每一张蓝图拼音页都是这么挑的,
     页头颜色不再筛, 和 hits_blue 一样)。去掉:
  - 文件名在 --seen 里的: 以前所有批次的扫描名单 / 目录(seen_names.load, 按文件名认);
  - GWCZ 号在 --used 里出现过的: 以前用过的页(训练 / 测试 / 答案集的目录或名单)。同一笔交易换了截图时间, 文件名不一样、GWCZ 号一样, 也算用过。
按文件名排好, random.Random(--seed).sample 挑 --n 张(池子不够就全拿); 每张拷完逐字节核对 sha256。
★ 先核扫完了没有(扫描半路断了, 名单也能用, 但只是这批图的一部分): 给了 --root, 就把这些目录里所有图片里不在 --seen 的,
  一张张对 --coverage-scan(这批的完整扫描名单, 不给就用 --scan): 有一张没扫到就停。白图的池子是 probe 的输出(只含 probe 判有拼音的),
  再给 --probe-all(probe 的全部输出), 核 --coverage-scan 里分数 >= --threshold 的每一张 probe 都判过。
出 --out\_pages.csv(page, score, src, sha256)和 --out\_pool.txt(池子的统计); --out 里已经有东西就不动。
最后一行纯 ASCII: PICK_FRESH_OK n(挑够了)/ PICK_FRESH_SHORT n(池子比 --n 小, 全拿了)/ PICK_FRESH_STOP ...(出错, 什么都没拷完)
    python training/pick_fresh_pages.py --scan E:\alipay-ai-data\newbatch_X\blue_scan.jsonl --seen <以前的扫描名单 ...>
        --used <用过的目录 ...> --n 300 --seed 8181 --out E:\_move_logs\pages_blue4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seen_names import load as load_seen  # noqa: E402

GWCZ = re.compile(r"GWCZ\d+")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scan_rows_any(paths):
    """扫描名单里每一行(有 path 就算, 读不出分数的也算: 那张图扫过了, 只是读不了)。"""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:   # noqa: BLE001
                    continue
                if r.get("path"):
                    out.append(r)
    return out


def scan_rows(paths):
    """扫描名单里量到了分数的行。"""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:   # noqa: BLE001  半行(被砍断的)
                    continue
                if r.get("path") and r.get("score") is not None:
                    out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, nargs="+", required=True)
    ap.add_argument("--seen", type=Path, nargs="+", required=True)
    ap.add_argument("--used", type=Path, nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=8181)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, nargs="*", default=None, help="这批新图所在的目录(核扫完了没有)")
    ap.add_argument("--coverage-scan", type=Path, nargs="*", default=None, help="这批的完整扫描名单(不给 = --scan)")
    ap.add_argument("--probe-all", type=Path, default=None, help="白图: new_batch.py probe 的全部输出(核 probe 判完了没有)")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        print(f"{a.out} 已经有东西了, 不覆盖")
        print("PICK_FRESH_STOP out_not_empty")
        sys.exit(2)
    cov = list(a.coverage_scan or a.scan)
    for p in list(a.scan) + list(a.seen) + list(a.used) + cov + list(a.root or []) + ([a.probe_all] if a.probe_all else []):
        if not p.exists():
            print(f"不在: {p}")
            print("PICK_FRESH_STOP missing_input")
            sys.exit(2)
    seen, notes_s, _w = load_seen(a.seen)
    if a.root:
        cov_names = {os.path.basename(r["path"].replace("\\", "/")) for r in scan_rows_any(cov)}
        want, n_img = set(), 0
        for d in a.root:
            for dirpath, _dirs, fs in os.walk(d):
                for f in fs:
                    if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                        n_img += 1
                        if f not in seen:
                            want.add(f)
        miss = sorted(want - cov_names)
        print(f"核扫完了没有: 目录里图片 {n_img:,} 张, 不在 --seen 里的 {len(want):,} 张, 扫描名单里没有的 {len(miss):,} 张")
        if miss:
            print(f"  例: {miss[:3]}")
            print("PICK_FRESH_STOP scan_incomplete")
            sys.exit(2)
    if a.probe_all:
        cand = {os.path.basename(r["path"].replace("\\", "/")) for r in scan_rows(cov) if r["score"] >= a.threshold}
        probed = {os.path.basename(r["path"].replace("\\", "/")) for r in scan_rows_any([a.probe_all])}
        miss = sorted(cand - probed)
        print(f"核 probe 判完了没有: 分数 >= {a.threshold} 的 {len(cand):,} 张, probe 没判的 {len(miss):,} 张")
        if miss:
            print("PICK_FRESH_STOP probe_incomplete")
            sys.exit(2)
    used_names, notes_u, _w2 = load_seen(a.used)
    used_ids = {m.group(0) for n in used_names for m in [GWCZ.search(n)] if m}
    rows = scan_rows(a.scan)
    by_name = {}
    for r in rows:
        name = os.path.basename(r["path"].replace("\\", "/"))
        if name not in by_name or r["score"] > by_name[name]["score"]:
            by_name[name] = r
    hits = {n: r for n, r in by_name.items() if r["score"] >= a.threshold}
    in_seen = sorted(n for n in hits if n in seen)
    no_id = sorted(n for n in hits if not GWCZ.search(n))
    id_used = sorted(n for n in hits if n not in seen and GWCZ.search(n) and GWCZ.search(n).group(0) in used_ids)
    pool = sorted(n for n in hits if n not in seen and GWCZ.search(n) and GWCZ.search(n).group(0) not in used_ids)
    # 同一个 GWCZ 号在池子里出现两次(同一笔交易截了两张): 只留文件名排第一的那张, 免得两张几乎一样的页都进测试
    first_of_id, dup = {}, []
    for n in pool:
        g = GWCZ.search(n).group(0)
        if g in first_of_id:
            dup.append(n)
        else:
            first_of_id[g] = n
    pool = sorted(first_of_id.values())
    lines = ["扫描名单:"] + [f"  {p}" for p in a.scan] + ["以前见过的(--seen):"] + [f"  {s}" for s in notes_s] + \
            ["用过的(--used):"] + [f"  {s}" for s in notes_u] + [
        f"扫描名单里的图 {len(by_name):,} 张(量到分数的), 分数 >= {a.threshold} 的 {len(hits):,} 张",
        f"  去掉: 文件名以前见过 {len(in_seen):,}, 文件名里没有 GWCZ 号 {len(no_id):,}, GWCZ 号以前用过 {len(id_used):,}, 同一 GWCZ 号重复 {len(dup):,}",
        f"  池子 {len(pool):,} 张; 要挑 {a.n} 张, 种子 {a.seed}"]
    for s in lines:
        print(s)
    pick = random.Random(a.seed).sample(pool, a.n) if len(pool) > a.n else list(pool)
    a.out.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for n in sorted(pick):
        src = Path(hits[n]["path"])
        if not src.exists():
            print(f"原图不在了: {src}")
            print("PICK_FRESH_STOP source_missing")
            sys.exit(2)
        dst = a.out / n
        shutil.copyfile(src, dst)
        h = sha(dst)
        if h != sha(src):
            print(f"拷出来和原图不一样: {n}")
            print("PICK_FRESH_STOP copy_differs")
            sys.exit(2)
        out_rows.append({"page": n, "score": f"{hits[n]['score']:.4f}", "src": str(src), "sha256": h})
    with open(a.out / "_pages.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "score", "src", "sha256"])
        w.writeheader()
        w.writerows(out_rows)
    (a.out / "_pool.txt").write_text("\n".join(lines + [f"挑出 {len(out_rows)} 张"]) + "\n", encoding="utf-8")
    print(f"PICK_FRESH_OK {len(out_rows)}" if len(out_rows) == a.n else f"PICK_FRESH_SHORT {len(out_rows)}")


if __name__ == "__main__":
    main()
