r"""gold3: 在 900 张留出测试页上, 照 gold / gold2 的做法出一份新答案集的候选段。测试数据目录只读。

为什么要新的一份(2026-10-25)
----------------------------
gold + gold2 那 1,400 段上已经做过太多选择 —— i2、jt、k、拼音读取器、"认不出来"的标记规则都是在它上面挑的,
再在上面报数就是自己考自己。测试页(new_batch.py holdout, 种子 17, 每轮白 200 / 蓝 100)从没进过训练,
10-15 以后 OCR 的选择也没再用过它们, 但它们**从来没切过段**: 只有行图和老师标签。

做法 —— 和训练目录出 val.txt 时一样的几步, 只是全写到工作目录
------------------------------------------------------------
    prep      核对每个测试目录: 有行图(label / input)和老师标签(_pairs_labeled.csv), 还没有 _segments.csv / seg_input;
              记快照(顶层文件和大小、子目录文件数和最新修改时间、清单 sha256), 把 _pairs_labeled.csv 拷到工作目录
    (块里)    split_segments --pairs <测试目录> --out <工作目录>  ->  clean_labels --apply  ->  check_alignment --drop
              ★ 这三个工具一个字没改。--out 让切出来的段和改过的清单都落在工作目录, 测试目录一个字节都不写
    gate      每一步之后核一遍: 清单行数 = seg_input = seg_label; 洗完要有 bad 列和两份备份, 错配的不到 10%
    pool      汇总: 空标签 / 错配(bad=1) / 状态栏(第 0 行且下边 <= 0.10 x 页宽, 和 label_audit 同一条) / 排除页 都不要;
              每页都必须是 newbatch_* 的 test_white / test_blue 里同颜色的那一张;
              每段一个固定排序键 sha1("gold3:<目录>:<段文件名>"); 写 _pool.csv(**不带老师标签**)
    export    每个目录按排序键取前 N 段 -> <out>/img/0001.png ... + _gold_candidates.csv(和 export_gold 同样的列, label 留空)
              + _gold3_meta.csv(几何、页名, 不带文字) + _teacher_labels.csv(老师标签单放; 答案定下来之前不看)
    (块里)    旧版重切 recrop_v2_frozen.py(指纹 dcc336dd7add) -> <out>_rc2, 现版 recrop.py(b78370c3e883) -> <out>_rc3
    parity    两份重切的答案目录逐像素比: 拿 gold / gold2 用两版代码各重切一遍, 和当初的 _rc2 / _rc3 一样才说明代码和环境都没变
    check     OLD / _rc2 / _rc3: 段一样, 重切图 = 旧图上下各加几行(旧图原样嵌在里面), 宽度不变, 指纹对
    snapshot  测试目录和 prep 时的快照一模一样(只读的证据)

每个子命令最后打印一行纯 ASCII 的 XXX_OK; 不对就打印原因并以非 0 退出(服务器脚本看这两样)。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

GW = re.compile(r"GWCZ\d+", re.I)
COLOURS = ("blue", "white")
ROW_RE = re.compile(r"_r(\d+)_s\d+")
POOL_COLS = ["dir", "colour", "file", "row_file", "page", "gwcz", "row", "seg", "x0", "x1", "has_pinyin",
             "W", "label_y0", "label_y1", "input_y0", "input_y1", "why", "rank"]


def die(tag: str, msg: str, code: int = 2) -> None:
    print(f"  ★ {msg}", flush=True)
    print(tag, flush=True)
    sys.exit(code)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def colour_of(name: str) -> str:
    hits = [c for c in COLOURS if c in name.lower()]
    if len(hits) != 1:
        die("BAD_DIR_NAME", f"目录名里看不出白 / 蓝: {name}")
    return hits[0]


def read_csv(p: Path) -> list:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, cols: list, rows: list) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def snapshot(d: Path) -> dict:
    """测试目录的快照: 顶层文件(名字, 大小)、每个子目录的文件数和最新修改时间、清单的 sha256。"""
    top, subs = [], {}
    for e in os.scandir(d):
        if e.is_file():
            top.append([e.name, e.stat().st_size])
        elif e.is_dir():
            n, mt = 0, 0.0
            for f in os.scandir(e.path):
                n += 1
                mt = max(mt, f.stat().st_mtime)
            subs[e.name] = [n, mt]
    hs = {fn: sha256_of(d / fn) for fn in ("_pairs_labeled.csv", "_pairs.csv") if (d / fn).exists()}
    return {"top": sorted(top), "subs": subs, "sha256": hs}


def cmd_prep(a) -> None:
    for name in a.dirs:                       # ★ 先把六个目录都核完, 有一个不对就什么都不建
        d, w = a.data / name, a.work / name
        miss = [p.name for p in (d / "_pairs_labeled.csv", d / "label", d / "input") if not p.exists()]
        if miss:
            die("PREP_MISSING", f"{d} 缺 {miss}")
        if (d / "_segments.csv").exists() or (d / "seg_input").exists():
            die("PREP_ALREADY_SEGMENTED", f"{d} 里已经有 _segments.csv 或 seg_input —— 和记录的不一样, 先停下看看", 3)
        if w.exists():
            die("PREP_WORK_EXISTS", f"{w} 已经在了, 不覆盖")
    snaps = {}
    for name in a.dirs:
        d, w = a.data / name, a.work / name
        snap = snapshot(d)
        w.mkdir(parents=True)
        shutil.copyfile(d / "_pairs_labeled.csv", w / "_pairs_labeled.csv")
        if sha256_of(w / "_pairs_labeled.csv") != snap["sha256"]["_pairs_labeled.csv"]:
            die("PREP_COPY_DIFFERS", f"{name}: 拷到工作目录的 _pairs_labeled.csv 和原件不一样")
        rows = read_csv(w / "_pairs_labeled.csv")
        usable = sum(1 for r in rows if r.get("usable") == "1" and (r.get("text") or "").strip())
        pages = len({r["source"] for r in rows})
        snaps[name] = snap
        print(f"PREP_STATS {name} pages={pages} rows={len(rows)} usable={usable} "
              f"label={snap['subs'].get('label', [0])[0]} input={snap['subs'].get('input', [0])[0]}", flush=True)
    (a.work / "_snapshot_prep.json").write_text(json.dumps(snaps, ensure_ascii=False, indent=1), encoding="utf-8")
    print("PREP_OK", flush=True)


def cmd_gate(a) -> None:
    for name in a.dirs:
        w = a.work / name
        if not (w / "_segments.csv").exists():
            die("GATE_FAIL", f"{name}: 没有 _segments.csv")
        rows = read_csv(w / "_segments.csv")
        names = [r["file"] for r in rows]
        si = set(os.listdir(w / "seg_input")) if (w / "seg_input").is_dir() else set()
        sl = set(os.listdir(w / "seg_label")) if (w / "seg_label").is_dir() else set()
        ok = bool(rows) and len(set(names)) == len(names) and set(names) == si == sl
        nb = 0
        if a.stage == "clean":
            nb = sum(1 for r in rows if r.get("bad") == "1")
            ok = ok and "bad" in rows[0] and nb < 0.10 * len(rows)
            ok = ok and (w / "_segments.csv.bak").exists() and (w / "_segments.csv.bak2").exists()
        print(f"GATE_STATS {a.stage} {name} segments={len(rows)} seg_input={len(si)} seg_label={len(sl)} bad={nb}", flush=True)
        if not ok:
            die("GATE_FAIL", f"{name}: {a.stage} 这一步的产物对不上")
    print("GATE_OK", flush=True)


def page_width(d: Path, row_file: str, cache: dict, src: str):
    """和 label_audit.page_width 一样: 行图的宽 = 页宽(先 label 再 input)。"""
    if src in cache:
        return cache[src]
    from PIL import Image
    w = None
    for sub in ("label", "input"):
        p = d / sub / row_file
        if p.exists():
            try:
                with Image.open(p) as im:
                    w = im.width
                break
            except Exception:  # noqa: BLE001
                pass
    cache[src] = w
    return w


def test_pages(roots: list) -> dict:
    """newbatch_* 下 test_white / test_blue 里每张页的名字(小写) -> 颜色。"""
    out = {}
    for r in roots:
        for c in COLOURS:
            d = Path(r) / f"test_{c}"
            if d.is_dir():
                for e in os.scandir(d):
                    if e.is_file() and not e.name.startswith("_"):
                        out[e.name.lower()] = c
    return out


def cmd_pool(a) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    excl = {x.upper() for x in a.exclude_ids}
    (a.out / "exclude_pages.txt").write_text("\n".join(sorted(excl)) + "\n", encoding="utf-8")
    tp = test_pages(a.test_folders)
    print(f"  测试页 {len(tp)} 张  (白 {sum(1 for c in tp.values() if c == 'white')} / 蓝 {sum(1 for c in tp.values() if c == 'blue')})", flush=True)
    out, found = [], set()
    for name in a.dirs:
        d, w = a.data / name, a.work / name
        col = colour_of(name)
        pl = {r["file"]: r for r in read_csv(w / "_pairs_labeled.csv")}
        segs = read_csv(w / "_segments.csv")
        wcache = {}
        why_n = Counter()
        pages, pages_ok, not_test = set(), set(), set()
        for r in segs:
            src = r.get("source", "")
            page = Path(src.replace("\\", "/")).name
            m = GW.search(page) or GW.search(r["file"])
            g = m.group().upper() if m else ""
            pages.add(page)
            if tp.get(page.lower()) != col:
                not_test.add(page)
            row = pl.get(r.get("row_file", ""), {})
            rown = row.get("row", "")
            if rown == "":
                mm = ROW_RE.search(r["file"])
                rown = str(int(mm.group(1))) if mm else ""
            W = page_width(d, r.get("row_file", ""), wcache, src)
            ly1 = row.get("label_y1", "")
            if g in excl:
                found.add(g)
            if not (w / "seg_input" / r["file"]).exists():
                why = "missing"
            elif g in excl:
                why = "excluded"
            elif not (r.get("text") or "").strip():
                why = "empty"
            elif r.get("bad") == "1":
                why = "bad"
            elif rown == "0" and W and ly1 not in ("", None) and float(ly1) <= 0.10 * W:
                why = "status"
            else:
                why = ""
                pages_ok.add(page)
            why_n[why or "eligible"] += 1
            out.append({"dir": name, "colour": col, "file": r["file"], "row_file": r.get("row_file", ""), "page": page, "gwcz": g,
                        "row": rown, "seg": r.get("seg", ""), "x0": r.get("x0", ""), "x1": r.get("x1", ""),
                        "has_pinyin": r.get("has_pinyin", ""), "W": W or "", "label_y0": row.get("label_y0", ""), "label_y1": ly1,
                        "input_y0": row.get("input_y0", ""), "input_y1": row.get("input_y1", ""), "why": why,
                        "rank": hashlib.sha1(f"gold3:{name}:{r['file']}".encode("utf-8")).hexdigest()})
        print(f"POOL_STATS {name} pages={len(pages)} pages_eligible={len(pages_ok)} segments={len(segs)} "
              + " ".join(f"{k}={why_n[k]}" for k in ("eligible", "excluded", "empty", "bad", "status", "missing")), flush=True)
        if not_test:
            die("POOL_NOT_TEST_PAGE", f"{name}: {len(not_test)} 张页不在 {col} 测试页里, 例如 {sorted(not_test)[:3]}")
    print(f"EXCLUDE_FOUND {len(found)} of {len(excl)}", flush=True)
    write_csv(a.out / "_pool.csv", POOL_COLS, out)
    print("POOL_OK", flush=True)


def rename_retry(src: Path, dst: Path) -> None:
    for _ in range(40):
        try:
            os.replace(src, dst)
            return
        except OSError:
            time.sleep(0.5)
    die("RENAME_FAILED", f"{src} 改不了名(被占用?)")


def cmd_export(a) -> None:
    if a.out.exists():
        die("EXPORT_EXISTS", f"{a.out} 已经在了, 不再导(要重导先挪走)")
    want = {}
    for spec in a.n:
        c, n = spec.split(":")
        want[c] = int(n)
    pool = read_csv(a.pool)
    teacher = {}
    for name in a.dirs:
        for r in read_csv(a.work / name / "_segments.csv"):
            teacher[(name, r["file"])] = (r.get("text") or "").strip()
    picked = []
    for name in a.dirs:
        el = sorted((r for r in pool if r["dir"] == name and r["why"] == ""), key=lambda r: r["rank"])
        n = want[colour_of(name)]
        if len(el) < n:
            die("EXPORT_SHORT", f"{name}: 能用的只有 {len(el)} 段, 要 {n}")
        picked += el[:n]
        print(f"EXPORT_STATS {name} eligible={len(el)} picked={n} pages={len({r['page'] for r in el[:n]})}", flush=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "img").mkdir(parents=True)
    cand, meta, teach = [], [], []
    for i, r in enumerate(picked, 1):
        src = a.work / r["dir"] / "seg_input" / r["file"]
        fn = f"{i:04d}.png"
        shutil.copyfile(src, tmp / "img" / fn)
        if sha256_of(tmp / "img" / fn) != sha256_of(src):
            die("EXPORT_COPY_DIFFERS", f"{fn} 和 {src} 不一样")
        cand.append({"id": i, "file": fn, "src_path": str(src), "dir": r["dir"], "label": ""})
        meta.append({"id": i, **{k: r[k] for k in POOL_COLS if k not in ("why",)}})
        teach.append({"id": i, "label": teacher[(r["dir"], r["file"])]})
    write_csv(tmp / "_gold_candidates.csv", ["id", "file", "src_path", "dir", "label"], cand)
    write_csv(tmp / "_gold3_meta.csv", ["id"] + [k for k in POOL_COLS if k != "why"], meta)
    write_csv(tmp / "_teacher_labels.csv", ["id", "label"], teach)
    if len(os.listdir(tmp / "img")) != len(cand):
        die("EXPORT_COUNT", "拷出来的图数和清单对不上")
    rename_retry(tmp, a.out)
    print(f"EXPORT_OK total={len(cand)} pages={len({r['page'] for r in picked})}", flush=True)


def imread(p: Path):
    import cv2
    import numpy as np
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def cmd_parity(a) -> None:
    import numpy as np
    if len(a.a) != len(a.b):
        die("PARITY_ARGS", "--a 和 --b 个数不一样")
    bad = 0
    for x, y in zip(a.a, a.b):
        fx = set(os.listdir(x / "img")) if (x / "img").is_dir() else set()
        fy = set(os.listdir(y / "img")) if (y / "img").is_dir() else set()
        same = differ = 0
        for fn in sorted(fx & fy):
            p, q = imread(x / "img" / fn), imread(y / "img" / fn)
            if p is not None and q is not None and p.shape == q.shape and np.array_equal(p, q):
                same += 1
            else:
                differ += 1
        lx = read_csv(x / "_recrop_log.csv") if (x / "_recrop_log.csv").exists() else None
        ly = read_csv(y / "_recrop_log.csv") if (y / "_recrop_log.csv").exists() else None
        vx = (x / "_recut_version.txt").read_text(encoding="utf-8").strip() if (x / "_recut_version.txt").exists() else ""
        vy = (y / "_recut_version.txt").read_text(encoding="utf-8").strip() if (y / "_recut_version.txt").exists() else ""
        ok = bool(fx) and fx == fy and differ == 0 and lx is not None and lx == ly and vx == vy and vx != ""
        bad += not ok
        print(f"PARITY_STATS {x.name} vs {y.parent.name}\\{y.name} files={len(fx)} pixel_same={same} differ={differ} "
              f"only_a={len(fx - fy)} only_b={len(fy - fx)} log_same={int(lx is not None and lx == ly)} ver_a={vx} ver_b={vy}", flush=True)
    if bad:
        die("PARITY_FAIL", f"{bad} 对目录不一样: 代码或服务器环境变了, 不能往下切")
    print("PARITY_OK", flush=True)


def cmd_check(a) -> None:
    import numpy as np
    cand = read_csv(a.gold / "_gold_candidates.csv")
    if len(a.rc) != len(a.fp):
        die("CHECK_ARGS", "--rc 和 --fp 个数不一样")
    bad = 0
    for rc, fp in zip(a.rc, a.fp):
        c2 = read_csv(rc / "_gold_candidates.csv")
        log = {r["id"]: r for r in read_csv(rc / "_recrop_log.csv")}
        ver = (rc / "_recut_version.txt").read_text(encoding="utf-8").strip() if (rc / "_recut_version.txt").exists() else ""
        emb = changed = 0
        st = Counter()
        for r in cand:
            o, c = imread(a.gold / "img" / r["file"]), imread(rc / "img" / r["file"])
            L = log.get(r["id"])
            if o is None or c is None or L is None:
                st["unreadable_or_no_log"] += 1
                continue
            st[L["status"]] += 1
            up, down = int(L["up"]), int(L["down"])
            changed += (up, down) != (0, 0)
            if up >= 0 and down >= 0 and c.shape[1] == o.shape[1] and c.shape[0] == o.shape[0] + up + down \
                    and np.array_equal(c[up:up + o.shape[0]], o):
                emb += 1
        ok = c2 == cand and ver == fp and emb == len(cand) and st.get("ok", 0) == len(cand)
        bad += not ok
        print(f"CHECK_STATS {rc.name} n={len(cand)} embedded={emb} status_ok={st.get('ok', 0)} changed={changed} "
              f"version={ver} want={fp} same_candidates={int(c2 == cand)}", flush=True)
    if bad:
        die("CHECK_FAIL", "重切的答案目录和原图对不上")
    print("CHECK_OK", flush=True)


def cmd_snapshot(a) -> None:
    old = json.loads((a.work / "_snapshot_prep.json").read_text(encoding="utf-8"))
    bad = []
    for name in a.dirs:
        now = json.loads(json.dumps(snapshot(a.data / name)))
        if now != old.get(name):
            bad.append(name)
    if bad:
        die("SNAPSHOT_CHANGED", f"测试目录和 prep 时不一样: {bad}")
    print(f"SNAPSHOT_SAME {len(a.dirs)} dirs", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", type=Path, required=True, help="测试配对目录所在的根(E:\\alipay-ai-data)")
    p.add_argument("--work", type=Path, required=True, help="工作目录(切出来的段放这里)")
    p.add_argument("--dirs", nargs="+", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--work", type=Path, required=True)
    g.add_argument("--dirs", nargs="+", required=True)
    g.add_argument("--stage", choices=["seg", "clean"], required=True)
    q = sub.add_parser("pool")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--work", type=Path, required=True)
    q.add_argument("--dirs", nargs="+", required=True)
    q.add_argument("--test-folders", nargs="+", required=True, help="newbatch_* 目录(下面有 test_white / test_blue)")
    q.add_argument("--exclude-ids", nargs="*", default=[], help="整页不要的 GWCZ 号")
    q.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("export")
    e.add_argument("--work", type=Path, required=True)
    e.add_argument("--pool", type=Path, required=True)
    e.add_argument("--dirs", nargs="+", required=True, help="按这个顺序编号")
    e.add_argument("--n", nargs="+", required=True, help="每个目录取几段, 按颜色: blue:400 white:300")
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("parity")
    r.add_argument("--a", type=Path, nargs="+", required=True)
    r.add_argument("--b", type=Path, nargs="+", required=True)
    c = sub.add_parser("check")
    c.add_argument("--gold", type=Path, required=True)
    c.add_argument("--rc", type=Path, nargs="+", required=True)
    c.add_argument("--fp", nargs="+", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--dirs", nargs="+", required=True)
    a = ap.parse_args()
    {"prep": cmd_prep, "gate": cmd_gate, "pool": cmd_pool, "export": cmd_export, "parity": cmd_parity,
     "check": cmd_check, "snapshot": cmd_snapshot}[a.cmd](a)


if __name__ == "__main__":
    main()
