r"""step 45 收回标签抽查: 从每个配对目录的 _revive.csv 抽样, 把 v3 图和旧图拷出来, 拿回本机逐张看图核标签。

只读: 只读各目录的 _revive.csv / _segments.csv / seg_input / seg_input_rc3 和训练清单, 只往 --out 写。
不改 revive_segments.py / recrop.py(那两个的指纹 step 45 的服务器脚本在核对)。

抽什么
------
    revive  收回来进了训练的: 按 (目录, 候选种类) 按比例随机抽 --n 段(每个有的格子至少 1 段)
    relabel / keep / drop  改了 / 没动 / 拿掉的训练段: 全部
    skip    没收的: 按目录按比例随机抽 --skip-n 段(看看漏掉了什么)
    随机数种子固定(--seed), 重跑抽到的一样。
每段: v3 图(seg_input_rc3)和旧图(seg_input)各拷一张, manifest.csv 记 _revive.csv 那一行 + 原图路径
      + 训练清单(train-n1009k)里实际用的标签, 并核对清单和收回规则是否一致:
        revive -> 应在清单里、标签 = new_text(金标准的段和字表外的不进)
        relabel -> 应在清单里、标签 = new_text(字表外的拿掉)   keep -> 应在清单里   drop / skip -> 不应在清单里
输出: <out>\lists\<目录>__revive.csv(九个全文) / img\ / manifest.csv / summary.txt / REVIVE_AUDIT_DONE.txt
      先写 <out>.tmp, 全部写完才改名成 <out>; 已经有 REVIVE_AUDIT_DONE.txt 的不重做(打印上次的统计)。

用法
----
    python training/revive_audit.py --pairs 目录 [...] --list E:\alipay-ai-data\train-n1009k --gold 金标准目录 [...] --out E:\_move_logs\revive_audit
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RC = "seg_input_rc3"
GROUPS = ("revive", "relabel", "keep", "drop", "skip")
MANIFEST = ["idx", "group", "dir", "file", "kind", "old_text", "read", "conf", "status", "is_button", "tall", "still_cut",
            "action", "new_text", "in_list", "list_label", "expect", "list_check", "rc3_png", "old_png", "source", "row_file"]


def nk(s: str) -> str:
    """比较用: 全半角统一(NFKC), 去空白(和 revive_segments.nk 一样)。"""
    return "".join(unicodedata.normalize("NFKC", s or "").split())


def read_csv(p: Path) -> list:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_key(p: str) -> tuple:
    """清单里的路径 -> (配对目录名, 文件名), 小写(和 revive_segments.relist 用的键一样)。"""
    sp = Path(p)
    return sp.parent.parent.name.lower(), sp.name.lower()


def load_lists(d: Path) -> tuple:
    """train-n1009k 的 train.txt / val.txt -> ({键: [(train|val, 标签), ...]}, 总行数)。同一段出现两次也都留着(核对时报出来)。"""
    out, n = defaultdict(list), 0
    for which in ("train", "val"):
        p = d / f"{which}.txt"
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if "\t" in ln:
                path, txt = ln.split("\t", 1)
                out[list_key(path)].append((which, txt))
                n += 1
    return out, n


def gold_keys(golds: list) -> set:
    keys = set()
    for g in golds:
        f = g / "_gold_candidates.csv"
        if f.exists():
            for r in read_csv(f):
                sp = Path(r.get("src_path", ""))
                keys.add((sp.parent.parent.name.lower(), sp.name.lower()))
    return keys


def allocate(cells: dict, n: int) -> dict:
    """按格子大小按比例分 n 个名额(最大余数法), 每个有的格子至少 1 个; 总数不够 n 就全要。"""
    total = sum(len(v) for v in cells.values())
    if total <= n:
        return {k: len(v) for k, v in cells.items()}
    keys = [k for k in sorted(cells) if cells[k]]
    alloc = {k: 1 for k in keys}
    left = n - len(keys)
    if left < 0:                                       # 格子比名额还多: 只按比例(不保底)
        alloc, left = {k: 0 for k in keys}, n
    rest = {k: len(cells[k]) - alloc[k] for k in keys}
    room = sum(rest.values())
    share = {k: left * rest[k] / room if room else 0 for k in keys}
    for k in keys:
        alloc[k] += int(share[k])
    extra = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(share[k] - int(share[k])), k))[:max(0, extra)]:
        alloc[k] += 1
    return {k: min(alloc[k], len(cells[k])) for k in keys}


def expect_of(group: str, r: dict, key: tuple, gk: set, chars: set) -> str:
    """按收回规则(revive_segments.relist), 这一段在训练清单里应该是什么样。"""
    new = r.get("new_text", "")
    if group == "revive":
        if key in gk:
            return "absent(gold)"
        if chars and not all(c in chars for c in new):
            return "absent(charset)"
        return "present:" + new
    if group == "relabel":
        if chars and not all(c in chars for c in new):
            return "absent(charset)"
        return "present:" + new
    if group == "keep":
        return "present"
    return "absent"


def check(expect: str, hits: list) -> str:
    """hits = 这一段在清单里的所有行。同一段出现两次以上本身就不对(比如旧标签那行没去掉)。"""
    if len(hits) > 1:
        return "WRONG_dup"
    hit = hits[0] if hits else None
    if expect.startswith("absent"):
        return "ok" if hit is None else "WRONG_in_list"
    if hit is None:
        return "WRONG_not_in_list"
    if expect.startswith("present:"):
        return "ok" if nk(hit[1]) == nk(expect[len("present:"):]) else "WRONG_label"
    return "ok"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="step 45 收回标签抽查(只读, 只往 --out 写)")
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--list", type=Path, required=True, help="v3 训练清单目录(train-n1009k)")
    ap.add_argument("--gold", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--skip-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=45)
    a = ap.parse_args()
    t0 = time.time()
    out, tmp = a.out, a.out.with_name(a.out.name + ".tmp")
    done = out / "REVIVE_AUDIT_DONE.txt"
    if done.exists():
        print("  already collected - not redone; last summary:", flush=True)
        print((out / "summary.txt").read_text(encoding="utf-8"), flush=True)
        print("REVIVE_AUDIT_OK", flush=True)
        return
    if out.exists():                                   # 以前没跑完留下的: 挪开, 不删
        moved = out.with_name(out.name + "_partial_" + time.strftime("%Y%m%d_%H%M%S"))
        os.replace(out, moved)
        print(f"  old incomplete folder moved to {moved.name}", flush=True)
    for _try in range(40):                             # 自己上次没写完的 .tmp(偶尔被别的程序开着: 等一下再试)
        if not tmp.exists():
            break
        shutil.rmtree(tmp, ignore_errors=True)
        if tmp.exists():
            time.sleep(0.5)
    if tmp.exists():
        print(f"REVIVE_AUDIT_TMP_LOCKED - cannot clear {tmp}; close what uses it and run again", flush=True)
        sys.exit(5)
    for d in a.pairs:
        if not (d / "_revive.csv").exists():
            print(f"REVIVE_AUDIT_MISSING {d}\\_revive.csv - nothing written", flush=True)
            sys.exit(2)
    if not (a.list / "train.txt").exists():
        print(f"REVIVE_AUDIT_MISSING {a.list}\\train.txt - nothing written", flush=True)
        sys.exit(2)
    (tmp / "lists").mkdir(parents=True)
    (tmp / "img").mkdir()

    rows, per_dir = [], {}
    for d in a.pairs:
        shutil.copy2(d / "_revive.csv", tmp / "lists" / f"{d.name}__revive.csv")
        rs = read_csv(d / "_revive.csv")
        per_dir[d.name] = Counter(r.get("action", "") for r in rs)
        for r in rs:
            r["_dir"] = d
            rows.append(r)
    lists, list_lines = load_lists(a.list)
    cs = a.list / "charset.txt"
    chars = set(c for c in cs.read_text(encoding="utf-8").split("\n") if c) if cs.exists() else set()
    gk = gold_keys(a.gold)
    full, full_wrong = Counter(), []                   # 全部 _revive.csv 行都和清单核一遍(不只抽到的)
    for r in rows:
        g = r.get("action", "")
        if g not in GROUPS:
            continue
        key = (r["_dir"].name.lower(), r["file"].lower())
        chk = check(expect_of(g, r, key, gk, chars), lists.get(key, []))
        full[(g, chk == "ok")] += 1
        if chk != "ok":
            full_wrong.append(f"{g} {r['_dir'].name} {r['file']} {chk}")
    dup_keys = sum(1 for v in lists.values() if len(v) > 1)

    rnd = random.Random(a.seed)
    pick = []                                          # (group, row)
    cells = defaultdict(list)
    for r in rows:
        if r.get("action") == "revive":
            cells[(r["_dir"].name, r.get("kind", ""))].append(r)
    for k, m in sorted(allocate(cells, a.n).items()):
        pool = sorted(cells[k], key=lambda r: r["file"])
        pick += [("revive", r) for r in rnd.sample(pool, m)]
    for g in ("relabel", "keep", "drop"):
        pick += [(g, r) for r in sorted((r for r in rows if r.get("action") == g), key=lambda r: (r["_dir"].name, r["file"]))]
    sk = defaultdict(list)
    for r in rows:
        if r.get("action") == "skip" and r.get("status") == "ok":
            sk[r["_dir"].name].append(r)
    for k, m in sorted(allocate(sk, a.skip_n).items()):
        pool = sorted(sk[k], key=lambda r: r["file"])
        pick += [("skip", r) for r in rnd.sample(pool, m)]

    segs = {}                                          # 原图路径(以后要看整页时用)
    need = defaultdict(set)
    for _g, r in pick:
        need[r["_dir"]].add(r["file"])
    for d, files in need.items():
        f = d / "_segments.csv"
        if f.exists():
            with f.open(encoding="utf-8-sig", newline="") as fh:
                for s in csv.DictReader(fh):
                    if s.get("file") in files:
                        segs[(d.name, s["file"])] = (s.get("source", ""), s.get("row_file", ""))

    man, cnt, wrong = [], Counter(), []
    for i, (g, r) in enumerate(pick, 1):
        d = r["_dir"]
        key = (d.name.lower(), r["file"].lower())
        hits = lists.get(key, [])
        hit = hits[0] if hits else None
        exp = expect_of(g, r, key, gk, chars)
        chk = check(exp, hits)
        cnt[(g, chk)] += 1
        if chk != "ok":
            wrong.append(f"{g} {d.name} {r['file']} expect={exp} list={hits}")
        names = {}
        for tag, sub in (("rc3", RC), ("old", "seg_input")):
            src = d / sub / r["file"]
            name = f"{i:04d}_{g}__{tag}.png"
            if src.exists():
                shutil.copy2(src, tmp / "img" / name)
                names[tag] = name
                cnt[f"img_{tag}"] += 1
            else:
                names[tag] = ""
                cnt[f"img_{tag}_missing"] += 1
        src_page, row_file = segs.get((d.name, r["file"]), ("", ""))
        man.append({"idx": f"{i:04d}", "group": g, "dir": d.name, "file": r["file"], "kind": r.get("kind", ""),
                    "old_text": r.get("old_text", ""), "read": r.get("read", ""), "conf": r.get("conf", ""),
                    "status": r.get("status", ""), "is_button": r.get("is_button", ""), "tall": r.get("tall", ""),
                    "still_cut": r.get("still_cut", ""), "action": r.get("action", ""), "new_text": r.get("new_text", ""),
                    "in_list": hit[0] if hit else "", "list_label": hit[1] if hit else "", "expect": exp, "list_check": chk,
                    "rc3_png": names["rc3"], "old_png": names["old"], "source": src_page, "row_file": row_file})
    with (tmp / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST)
        w.writeheader()
        w.writerows(man)

    lines = []
    for name, c in per_dir.items():
        lines.append(f"REVIVE_AUDIT_DIR {name} rows={sum(c.values())} " + " ".join(f"{g}={c.get(g, 0)}" for g in GROUPS))
    tot = Counter()
    for c in per_dir.values():
        tot.update(c)
    picked = Counter(g for g, _r in pick)
    lines.append("REVIVE_AUDIT_TOTAL rows=" + str(sum(tot.values())) + " " + " ".join(f"{g}={tot.get(g, 0)}" for g in GROUPS))
    lines.append("REVIVE_AUDIT_SAMPLE " + " ".join(f"{g}={picked.get(g, 0)}" for g in GROUPS)
                 + f" images_rc3={cnt['img_rc3']} images_old={cnt['img_old']}"
                 + f" missing_rc3={cnt['img_rc3_missing']} missing_old={cnt['img_old_missing']}")
    lines.append("REVIVE_AUDIT_LISTCHECK " + " ".join(f"{g}_ok={cnt[(g, 'ok')]}" for g in GROUPS)
                 + f" wrong={len(wrong)} list_keys={len(lists)} list_lines={list_lines} dup_keys={dup_keys} gold_keys={len(gk)}")
    lines += [f"  WRONG {x}" for x in wrong[:30]]
    lines.append("REVIVE_AUDIT_FULLCHECK " + " ".join(f"{g}_ok={full[(g, True)]}" for g in GROUPS)
                 + f" wrong={len(full_wrong)}")
    lines += [f"  FULLWRONG {x}" for x in full_wrong[:30]]
    lines.append(f"seed={a.seed} n={a.n} skip_n={a.skip_n} list={a.list} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    (tmp / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp / "REVIVE_AUDIT_DONE.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    for _try in range(40):                             # 杀毒 / 索引刚好开着里面的新文件时 Windows 不让改名: 等一下再试
        try:
            os.replace(tmp, out)
            break
        except OSError:
            time.sleep(0.5)
    else:
        print(f"REVIVE_AUDIT_RENAME_FAILED - everything is complete in {tmp}; rename that folder to {out.name}", flush=True)
        sys.exit(5)
    print("\n".join(lines), flush=True)
    print(f"REVIVE_AUDIT_OK {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
