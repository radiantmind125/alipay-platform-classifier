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
              (10-28 加 --skip / --not-in / --n all: 先跳过 gold3 取过的那些段, 剩下的再导一份确认用的 gold4)
              + _gold3_meta.csv(几何、页名, 不带文字) + _teacher_labels.csv(老师标签单放; 答案定下来之前不看)
    (块里)    旧版重切 recrop_v2_frozen.py(指纹 dcc336dd7add) -> <out>_rc2, 现版 recrop.py(b78370c3e883) -> <out>_rc3
    parity    两份重切的答案目录逐像素比: 拿 gold / gold2 用两版代码各重切一遍, 和当初的 _rc2 / _rc3 一样才说明代码和环境都没变
    splitparity 切段对照: 用今天的 split_segments 把一个训练目录重切一遍, 段和 x 区间、裁图都得和当初训练用的一样
    check     OLD / _rc2 / _rc3: 段一样, 重切图 = 旧图上下各加几行(旧图原样嵌在里面), 宽度不变, 指纹对; 清单 / 几何表 / 老师标签逐行对得上
    snapshot  测试目录(和对照用的训练目录)每个文件的 名字 / 大小 / 修改时间 和 prep 时一模一样(只读的证据)
★ 几何缺了、排除页没全找到、同一页在两个目录里、参数重复这些都直接停, 不悄悄跳过(10-25 复核 wf_840badab-49a 试出来的)。

gold6(2026-10-31): 训练目录里留作验证的页 —— 测试页的段用完了, 验证页(每个训练目录按页留出的 5%)从来没进过训练
------------------------------------------------------------------------------------------------
    valpool   读每个训练过的清单(认字模型的 train / val, 拼音带的 train / val), 每行对回它的原页:
              任何一张验证页出现在任何一份训练清单里就停 —— "没训练过"是核出来的, 不是记得的;
              候选 = 认字模型 val.txt 里的段; 不要: 用过的答案段(gold / gold2)、状态栏(和 pool 同一条)、
              和训练页是"同一笔交易"的验证页(规则和 gold3 查测试页那次一样: 同一个订单号 / 同一个到秒的时间加少见文字 /
              蓝页同卡号尾号加同金额), 两张验证页是同一笔交易的留排在前面那张;
              候选页是测试页、同一页在两个目录里、清单里有认不出的目录或对不回的段, 都直接停;
              写 _pool.csv(和 pool 同样的列, 排序键 sha1("gold6:<目录>:<段文件名>"))、_twins.csv(只有页名)、_inputs.json(读过的每个文件的 sha256)
    export    同上, --work 指到训练目录的根(只读); 每个目录的颜色从池子里来(目录名里看不出颜色的 pinyin-pairs 也行)
    rcsame    重切的答案图和训练目录里 10-24 切好的 seg_input_rc3 逐像素比: k / m 在验证集上看到的就是这些图
    inputs    valpool 读过的文件(清单、段清单、答案清单)到最后一个字节都没变

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
import unicodedata
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
    """目录的快照: 每个文件(含子目录里的)的 名字 / 大小 / 修改时间(纳秒), 两个清单再加 sha256。
    ★ 只记每个子目录的"最新修改时间"不够: 用保留时间戳的方式覆盖一个文件就看不出来(复核时试出来的)。"""
    files = []
    for e in os.scandir(d):
        if e.is_file():
            st = e.stat()
            files.append([e.name, st.st_size, st.st_mtime_ns])
        elif e.is_dir():
            files.append([e.name + "/", 0, 0])
            for f in os.scandir(e.path):
                st = f.stat()
                files.append([e.name + "/" + f.name, st.st_size, st.st_mtime_ns])
    hs = {fn: sha256_of(d / fn) for fn in ("_pairs_labeled.csv", "_pairs.csv", "_segments.csv") if (d / fn).exists()}
    return {"files": sorted(files), "sha256": hs}


def no_dups(names: list, tag: str) -> None:
    dup = sorted({x for x in names if names.count(x) > 1})
    if dup:
        die(tag, f"重复了: {dup}")


def cmd_prep(a) -> None:
    no_dups(a.dirs, "PREP_DUP_DIRS")
    for name in a.dirs:                       # ★ 先把六个目录都核完, 有一个不对就什么都不建
        d, w = a.data / name, a.work / name
        miss = [p.name for p in (d / "_pairs_labeled.csv", d / "label", d / "input") if not p.exists()]
        if miss:
            die("PREP_MISSING", f"{d} 缺 {miss}")
        cut = [e.name for e in os.scandir(d) if e.name.startswith(("seg_", "_segments"))]
        if cut:
            die("PREP_ALREADY_SEGMENTED", f"{d} 里已经有 {cut} —— 和记录的不一样, 先停下看看", 3)
        if w.exists():
            die("PREP_WORK_EXISTS", f"{w} 已经在了, 不覆盖")
    snaps = {}
    for name in a.also_snap:                  # 只拿来比对、不切的目录(切段对照用的那个训练目录): 也记快照, 证明没写进去
        snaps["also:" + name] = snapshot(a.data / name)
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
        nl = sum(1 for f in snap["files"] if f[0].startswith("label/") and f[0] != "label/")
        ni = sum(1 for f in snap["files"] if f[0].startswith("input/") and f[0] != "input/")
        print(f"PREP_STATS {name} pages={pages} rows={len(rows)} usable={usable} label={nl} input={ni}", flush=True)
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
    """newbatch_* 下 test_white / test_blue 里每张页的名字(小写) -> 颜色。同一个名字出现两次就停(颜色说不清)。"""
    out = {}
    for r in roots:
        for c in COLOURS:
            d = Path(r) / f"test_{c}"
            if d.is_dir():
                for e in os.scandir(d):
                    if e.is_file() and not e.name.startswith("_"):
                        if e.name.lower() in out:
                            die("POOL_DUP_TEST_PAGE", f"测试页 {e.name} 出现了两次")
                        out[e.name.lower()] = c
    return out


def gwcz_of(text: str) -> str:
    m = GW.search(text)
    return m.group().upper() if m else ""


def cmd_pool(a) -> None:
    no_dups(a.dirs, "POOL_DUP_DIRS")
    excl = {x.strip().upper() for x in a.exclude_ids}
    badid = sorted(x for x in excl if not re.fullmatch(r"GWCZ\d+", x))
    if badid:
        die("POOL_BAD_EXCLUDE_ID", f"排除页的号不像 GWCZ 号: {badid}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "exclude_pages.txt").write_text("\n".join(sorted(excl)) + "\n", encoding="utf-8")
    tp = test_pages(a.test_folders)
    print(f"  测试页 {len(tp)} 张  (白 {sum(1 for c in tp.values() if c == 'white')} / 蓝 {sum(1 for c in tp.values() if c == 'blue')})", flush=True)
    # 排除页: 在六个目录的老师标签清单(行, 不是段)里找, 一张页就算一个段都没切出来也找得到; 必须全找到
    page_dir = {}
    for name in a.dirs:
        for r in read_csv(a.work / name / "_pairs_labeled.csv"):
            page = Path(r.get("source", "").replace("\\", "/")).name
            if page_dir.setdefault(page, name) != name:
                die("POOL_DUP_PAGE", f"页 {page} 同时在 {page_dir[page]} 和 {name} 里")
    found = {gwcz_of(p) for p in page_dir} & excl
    print(f"EXCLUDE_FOUND {len(found)} of {len(excl)}", flush=True)
    if found != excl:
        die("POOL_EXCLUDE_NOT_FOUND", f"排除页没全找到: {sorted(excl - found)}")
    out = []
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
            g = gwcz_of(page) or gwcz_of(r["file"])
            pages.add(page)
            if tp.get(page.lower()) != col:
                not_test.add(page)
            row = pl.get(r.get("row_file", ""))
            W = page_width(d, r.get("row_file", ""), wcache, src)
            # ★ 几何缺了就停, 不悄悄当成"不是状态栏"(复核时试出来: 删掉第 0 行的清单行, 状态栏段就混进来了)
            if row is None or row.get("label_y1", "") in ("", None) or not W or not (w / "seg_input" / r["file"]).exists():
                die("POOL_NO_GEOMETRY", f"{name}/{r['file']}: 行清单 / 页宽 / 段图 缺了")
            rown = row.get("row", "")
            if rown == "":
                mm = ROW_RE.search(r["file"])
                rown = str(int(mm.group(1))) if mm else ""
            ly1 = row["label_y1"]
            if g in excl:
                why = "excluded"
            elif not (r.get("text") or "").strip():
                why = "empty"
            elif r.get("bad") == "1":
                why = "bad"
            elif rown == "0" and float(ly1) <= 0.10 * W:
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
              + " ".join(f"{k}={why_n[k]}" for k in ("eligible", "excluded", "empty", "bad", "status")), flush=True)
        if not_test:
            die("POOL_NOT_TEST_PAGE", f"{name}: {len(not_test)} 张页不在 {col} 测试页里, 例如 {sorted(not_test)[:3]}")
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
    no_dups(a.dirs, "EXPORT_DUP_DIRS")
    want, skip = {}, {}
    for spec in a.n:
        c, n = spec.split(":")
        want[c] = None if n == "all" else int(n)        # all = 跳过的后面剩下的全要
        if want[c] is not None and want[c] <= 0:
            die("EXPORT_BAD_N", f"{spec}: 段数要大于 0")
    for spec in a.skip:
        c, n = spec.split(":")
        skip[c] = int(n)
        if skip[c] < 0:
            die("EXPORT_BAD_SKIP", f"{spec}: 跳过的段数不能是负的")
    # --not-in: 以前导过的答案目录。跳过的那几段必须正好是它们导过的(证明清单和排序和当初一样), 取到的段一段都不能和它们重
    done = set()
    for g in a.not_in:
        rows = read_csv(g / "_gold3_meta.csv")
        if not rows:
            die("EXPORT_NOT_IN_EMPTY", f"{g} 的 _gold3_meta.csv 是空的")
        done |= {(r["dir"], r["file"]) for r in rows}
    pool = read_csv(a.pool)
    no_dups([(r["dir"], r["file"]) for r in pool], "EXPORT_DUP_POOL_ROWS")
    # 每个目录的颜色从池子里来(pool 写的就是 colour_of(目录名), 结果和以前一样; valpool 里 pinyin-pairs 这种目录名看不出颜色的也行)
    col_of = {}
    for r in pool:
        col_of.setdefault(r["dir"], set()).add(r["colour"])
    for name in a.dirs:
        if len(col_of.get(name, ())) != 1 or not col_of[name] <= set(COLOURS):
            die("EXPORT_COLOUR", f"{name}: 池子里这个目录的颜色是 {sorted(col_of.get(name, ()))}, 应该正好是 blue / white 里的一种")
    teacher = {}
    for name in a.dirs:
        for r in read_csv(a.work / name / "_segments.csv"):
            teacher[(name, r["file"])] = (r.get("text") or "").strip()
    picked = []
    for name in a.dirs:
        col = next(iter(col_of[name]))
        el = sorted((r for r in pool if r["dir"] == name and r["why"] == ""), key=lambda r: r["rank"])
        s = skip.get(col, 0)
        rest = el[s:]
        n = want[col]
        n = len(rest) if n is None else n
        if n <= 0 or len(rest) < n:
            die("EXPORT_SHORT", f"{name}: 跳过 {s} 段以后能用的只有 {len(rest)} 段, 要 {n}")
        if a.not_in and {(r["dir"], r["file"]) for r in el[:s]} != {k for k in done if k[0] == name}:
            die("EXPORT_SKIP_MISMATCH", f"{name}: 跳过的 {s} 段和 --not-in 里导过的不是同一批(清单或排序变了)")
        hit = [r["file"] for r in rest[:n] if (r["dir"], r["file"]) in done]
        if hit:
            die("EXPORT_OVERLAP", f"{name}: {len(hit)} 段以前导过了, 例如 {hit[:3]}")
        picked += rest[:n]
        print(f"EXPORT_STATS {name} eligible={len(el)} skipped={s} picked={n} pages={len({r['page'] for r in rest[:n]})}", flush=True)
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
    if not p.is_file():
        return None
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
    if not cand:
        die("CHECK_FAIL", "清单是空的")
    files = {r["file"] for r in cand}
    meta = read_csv(a.gold / "_gold3_meta.csv")
    teach = read_csv(a.gold / "_teacher_labels.csv")
    m_ok = [(m["id"], m["dir"]) for m in meta] == [(r["id"], r["dir"]) for r in cand] and [t["id"] for t in teach] == [r["id"] for r in cand]
    i_ok = set(os.listdir(a.gold / "img")) == files
    print(f"CHECK_STATS gold3 n={len(cand)} meta_and_teacher_aligned={int(m_ok)} img_set_same={int(i_ok)}", flush=True)
    bad = 0 if (m_ok and i_ok) else 1
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
        ok = c2 == cand and ver == fp and emb == len(cand) and st.get("ok", 0) == len(cand) and set(os.listdir(rc / "img")) == files
        bad += not ok
        print(f"CHECK_STATS {rc.name} n={len(cand)} embedded={emb} status_ok={st.get('ok', 0)} changed={changed} "
              f"version={ver} want={fp} same_candidates={int(c2 == cand)} img_set_same={int(set(os.listdir(rc / 'img')) == files)}", flush=True)
    if bad:
        die("CHECK_FAIL", "重切的答案目录和原图对不上")
    print("CHECK_OK", flush=True)


def cmd_snapshot(a) -> None:
    old = json.loads((a.work / "_snapshot_prep.json").read_text(encoding="utf-8"))
    bad = []
    keys = list(a.dirs) + ["also:" + x for x in a.also_snap]
    for key in keys:
        now = json.loads(json.dumps(snapshot(a.data / key.replace("also:", ""))))
        if old.get(key) is None or now != old[key]:
            bad.append(key)
    if bad:
        die("SNAPSHOT_CHANGED", f"目录和 prep 时不一样: {bad}")
    print(f"SNAPSHOT_SAME {len(keys)} dirs", flush=True)


def cmd_splitparity(a) -> None:
    """切段对照: 拿一个训练目录用今天的 split_segments 重切到 --fresh, 段的 (文件, 行, 序号, x0, x1) 和当初的 _segments.csv 必须一样,
    seg_input 图逐像素一样。证明今天切测试目录的 x 区间和训练、gold 当初是同一个切法(代码一样, 服务器上的 OpenCV / numpy 也没变)。"""
    import numpy as np
    key = lambda r: (r["file"], r["row_file"], str(r["seg"]), str(r["x0"]), str(r["x1"]))  # noqa: E731
    ref = {key(r) for r in read_csv(a.ref / "_segments.csv")}
    new = read_csv(a.fresh / "_segments.csv")
    got = {key(r) for r in new}
    same = differ = 0
    for r in new:
        p, q = imread(a.fresh / "seg_input" / r["file"]), imread(a.ref / "seg_input" / r["file"])
        if p is not None and q is not None and p.shape == q.shape and np.array_equal(p, q):
            same += 1
        else:
            differ += 1
    print(f"SPLITPARITY_STATS ref={len(ref)} fresh={len(got)} only_ref={len(ref - got)} only_fresh={len(got - ref)} "
          f"crops_same={same} crops_differ={differ}", flush=True)
    if not got or got != ref or differ:
        die("SPLITPARITY_FAIL", "今天切出来的段和当初训练目录的不一样: 切段代码或服务器环境变了, 不能往下切")
    print("SPLITPARITY_OK", flush=True)


# ---------------- gold6: 训练目录的验证页 ----------------
# "同一笔交易"的规则, 和 2026-10-25 查测试页那次(t3_final_twins)一字不差
TW_ORD = re.compile(r"2026\d{16,}")
TW_TS = re.compile(r"20\d\d-\d\d-\d\d\d\d:\d\d:\d\d")
TW_HM = re.compile(r"^(\d{1,2}):(\d\d)")
TW_CARD = re.compile(r"[（(](\d{4})[）)]")
TW_AMT = re.compile(r"(\d+\.\d\d)")
TW_HAN = re.compile(r"[\u4e00-\u9fff*]{4,}")
TW_KEEP = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff¥*.:\-]")


def tw_nf(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").replace("￥", "¥")


def tw_n(s: str) -> str:
    return TW_KEEP.sub("", tw_nf(s))


def tw_feats(rows: list) -> dict:
    """一张页的特征。rows = [(行号, 段号, 老师文字)]: 订单号 / 到秒的时间(段和整行拼起来都找)、4 个字以上的整段文字、
    卡号尾号、金额、4 个字以上的汉字串、状态栏的 时:分(第 0 行第一段)。"""
    byrow = {}
    for r, _s, t in sorted(rows):
        byrow.setdefault(r, []).append(t)
    texts = [t for _r, _s, t in sorted(rows)]
    f = {"O": set(), "S": set(), "tok": set(), "card": set(), "amt": set(), "han": set()}
    for t in texts + ["".join(v) for v in byrow.values()]:
        n = tw_n(t)
        f["O"].update(TW_ORD.findall(n))
        f["S"].update(TW_TS.findall(n))
    for t in texts:
        n = tw_n(t)
        if len(n) >= 4:
            f["tok"].add(n)
        nf = tw_nf(t)
        f["card"].update(TW_CARD.findall(nf))
        f["amt"].update(TW_AMT.findall(nf))
        f["han"].update(TW_HAN.findall(nf))
    r0 = byrow.get(0, [None])[0]
    m = TW_HM.match(tw_n(r0)) if r0 else None
    f["hm"] = f"{int(m.group(1)):02d}:{m.group(2)}" if m else None
    return f


def find_twins(feats: dict, colour: dict, pages) -> dict:
    """pages 里每张页 -> {另一张页: 规则}。O 同一个订单 / 交易号(2026 开头 20 位以上);
    S 同一个到秒的时间, 另外还有一段少见的文字(<= 5 张页有, 时间本身不算); B 这张是蓝页: 同一个卡号尾号 + 同一个金额,
    另外 状态栏同一个 时:分 或 一串少见的汉字(<= 3 张页有)。少见是在 feats 里所有页上数的。"""
    df, hdf, inv = Counter(), Counter(), {}
    for g, f in feats.items():
        df.update(f["tok"])
        hdf.update(f["han"])
        for k in f["O"]:
            inv.setdefault("O" + k, set()).add(g)
        for k in f["S"]:
            inv.setdefault("S" + k, set()).add(g)
        for c in f["card"]:
            for x in f["amt"]:
                inv.setdefault(f"B{c}|{x}", set()).add(g)
    out = {}
    for g in sorted(pages):
        f = feats.get(g)
        if f is None:
            continue
        cand = {}
        for k in f["O"]:
            for h in inv.get("O" + k, ()):
                cand.setdefault(h, set()).add("O")
        for k in f["S"]:
            for h in inv.get("S" + k, ()):
                cand.setdefault(h, set()).add("S?")
        if colour.get(g) == "blue":
            for c in f["card"]:
                for x in f["amt"]:
                    for h in inv.get(f"B{c}|{x}", ()):
                        cand.setdefault(h, set()).add("B?")
        got = {}
        for h, why in cand.items():
            if h == g:
                continue
            hf, rules = feats[h], set()
            if "O" in why:
                rules.add("O")
            if "S?" in why and {x for x in f["tok"] & hf["tok"] if df[x] <= 5} - {x for x in f["tok"] if TW_TS.fullmatch(x)}:
                rules.add("S")
            if "B?" in why and ((f["hm"] and f["hm"] == hf["hm"]) or {x for x in f["han"] & hf["han"] if hdf[x] <= 3}):
                rules.add("B")
            if rules:
                got[h] = "+".join(sorted(rules))
        out[g] = got
    return out


def list_key(p: str) -> tuple:
    """清单里一行的图路径 -> (配对目录名, 段文件名), 都小写。认字模型的清单: <根>\\<目录>\\seg_input[_rc / _rc3]\\<段>;
    拼音带的清单: <带目录>\\img\\<目录>\\<段>。盘符、根目录、斜杠方向都不管。"""
    q = Path(p.strip().replace("\\", "/"))
    d = q.parent.parent.name if q.parent.name.lower().startswith("seg_input") else q.parent.name
    return d.lower(), q.name.lower()


def iter_rows(p: Path, cols: tuple):
    """大清单一行一行读, 只取要的几列(没有的列给空串), 不建字典。"""
    with p.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        head = next(rd, [])
        ix = [head.index(c) if c in head else None for c in cols]
        for r in rd:
            yield tuple((r[i] if i is not None and i < len(r) else "") for i in ix)


def page_name(src: str) -> str:
    return Path((src or "").replace("\\", "/")).name.lower()


def cmd_valpool(a) -> None:
    no_dups([n.lower() for n in a.dirs], "VALPOOL_DUP_DIRS")
    no_dups([x.name.lower() for x in a.lists + a.also_lists], "VALPOOL_DUP_LISTS")
    names = {n.lower(): n for n in a.dirs}
    over = {}
    for spec in a.colour:
        d, _sep, c = spec.rpartition(":")
        if d.lower() not in names or c not in COLOURS:
            die("VALPOOL_BAD_COLOUR", f"--colour {spec}: 目录不在 --dirs 里, 或者颜色不是 blue / white")
        over[d.lower()] = c
    col = {n: over.get(n.lower()) or colour_of(n) for n in a.dirs}
    if a.out.exists():
        die("VALPOOL_EXISTS", f"{a.out} 已经在了, 不覆盖(要重做先挪走)")
    inputs = {}

    def note(p: Path) -> None:
        inputs[str(p)] = sha256_of(p)

    # 1. 九个训练目录: 每页属于哪个目录(同一页在两个目录就停)、每段在哪页、老师文字(只拿来找同一笔交易, 不打印)
    page_dir, seg_page, rows_pl, tw_rows = {}, {}, {}, {}
    for n in a.dirs:
        d = a.data / n
        for fn in ("_segments.csv", "_pairs_labeled.csv"):
            if not (d / fn).is_file():
                die("VALPOOL_MISSING", f"{d / fn} 不在")
            note(d / fn)
        pl = {}
        for rf, src, rown, ly0, ly1, iy0, iy1 in iter_rows(d / "_pairs_labeled.csv",
                                                             ("file", "source", "row", "label_y0", "label_y1", "input_y0", "input_y1")):
            pg = page_name(src)
            if page_dir.setdefault(pg, n) != n:
                die("VALPOOL_DUP_PAGE", f"页 {pg} 同时在 {page_dir[pg]} 和 {n} 里")
            pl[rf] = (rown, ly0, ly1, iy0, iy1)
        rows_pl[n] = pl
        for fn, rf, src, sg, text in iter_rows(d / "_segments.csv", ("file", "row_file", "source", "seg", "text")):
            k = (n.lower(), fn.lower())
            if k in seg_page:
                die("VALPOOL_DUP_SEGMENT", f"{n}/{fn} 在 _segments.csv 里出现了两次")
            pg = page_name(src)
            seg_page[k] = pg
            rown = pl.get(rf, ("",))[0]
            if not rown.isdigit():
                mm = ROW_RE.search(fn)
                rown = mm.group(1) if mm else "-1"
            # 空文字也留着: 状态栏的 时:分 取第 0 行第一段, 它是空的就没有(和 t3 一样)
            tw_rows.setdefault(pg, []).append((int(rown), int(sg) if sg.isdigit() else 0, text or ""))
        print(f"VALPOOL_DIR {n} colour={col[n]} pages={len({p for p, x in page_dir.items() if x == n})} segments={sum(1 for k in seg_page if k[0] == n.lower())}", flush=True)
    # 2. 训练过的每一份清单: 每行对回原页。训练页 T = 所有 train.txt 的页; 候选 = 认字模型 val.txt 里的段
    T, V, cand = set(), {}, {}
    for L, rec in [(x, True) for x in a.lists] + [(x, False) for x in a.also_lists]:
        for which in ("train", "val"):
            p = L / f"{which}.txt"
            if not p.is_file():
                die("VALPOOL_NO_LIST", f"{p} 不在")
            note(p)
            nl = notab = unmapped = 0
            unknown = Counter()
            pages = set()
            with p.open(encoding="utf-8") as f:
                for ln in f:
                    ln = ln.rstrip("\r\n")
                    if not ln.strip():
                        continue
                    nl += 1
                    notab += "\t" not in ln
                    k = list_key(ln.split("\t", 1)[0])
                    if k[0] not in names:
                        unknown[k[0]] += 1
                        continue
                    pg = seg_page.get(k)
                    if pg is None:
                        unmapped += 1
                        continue
                    pages.add(pg)
                    if rec and which == "val":
                        cand.setdefault(k, set()).add(L.name)
            print(f"VALPOOL_LIST {L.name} {which} lines={nl} pages={len(pages)} notab={notab} unknown_dir={sum(unknown.values())} "
                  f"unmapped={unmapped}", flush=True)
            if unknown:
                die("VALPOOL_UNKNOWN_DIR", f"{p}: 有不在 --dirs 里的目录 {dict(unknown.most_common(5))}")
            if unmapped:
                die("VALPOOL_UNMAPPED", f"{p}: {unmapped} 行对不回 _segments.csv 里的段")
            if nl == 0:
                die("VALPOOL_EMPTY_LIST", f"{p} 是空的")
            if which == "train":
                T |= pages
            else:
                V[L.name] = pages
    leak = {k: len(v & T) for k, v in V.items()}
    vall = set().union(*V.values())
    vrec = [V[x.name] for x in a.lists]
    differ = len(set().union(*vrec) - set.intersection(*vrec))
    print(f"VALPOOL_SPLIT train_pages={len(T)} val_pages={len(vall)} val_pages_in_train={sum(leak.values())} "
          f"val_pages_not_in_every_list={differ} pages_in_no_list={len(set(page_dir) - T - vall)}", flush=True)
    if any(leak.values()):
        die("VALPOOL_VAL_PAGE_IN_TRAIN", f"验证页出现在训练清单里: {leak}")
    # 3. 用过的答案段(gold / gold2): 必须是训练目录里认得的段, 它们的页不能是训练页(当初就是从验证集抽的)
    dev = set()
    for g in a.dev:
        p = g / "_gold_candidates.csv"
        note(p)
        for r in read_csv(p):
            k = list_key(r["src_path"])
            if k not in seg_page:
                die("VALPOOL_DEV_UNKNOWN", f"{g.name}/{r['id']}: {r['src_path']} 不是训练目录里的段")
            if seg_page[k] in T:
                die("VALPOOL_DEV_IN_TRAIN", f"{g.name}/{r['id']}: 这段的页在训练清单里")
            dev.add(k)
    print(f"VALPOOL_DEVPAGES dev_pages={len({seg_page[k] for k in dev})} not_val_pages={len({seg_page[k] for k in dev} - vall)}", flush=True)
    # 4. 同一笔交易: 验证页对所有页找(老师文字), 有一张是训练页 -> 整页不要; 两张都是验证页 -> 留名字排在前面那张
    feats = {pg: tw_feats(rs) for pg, rs in tw_rows.items()}
    del tw_rows
    pcol = {pg: col[n] for pg, n in page_dir.items()}
    tw = find_twins(feats, pcol, vall)
    why_page, rule_n, tw_out = {}, Counter(), []
    for v in sorted(vall):
        got = tw.get(v, {})
        tt = sorted(h for h in got if h in T)
        tv = sorted(h for h in got if h in vall and h < v)
        why_page[v] = "twin_train" if tt else ("twin_val" if tv else "")
        for h in tt:
            rule_n[got[h]] += 1
        if got:
            tw_out.append({"page": v, "dir": page_dir.get(v, ""), "why": why_page[v],
                           "partners": ";".join(f"{h}@{page_dir.get(h, '')}[{got[h]}]{'T' if h in T else ('V' if h in vall else '-')}"
                                                for h in sorted(got))})
    print(f"VALPOOL_TWINS val_pages={len(vall)} with_any_twin={len(tw_out)} twin_train={sum(1 for x in why_page.values() if x == 'twin_train')} "
          f"twin_val={sum(1 for x in why_page.values() if x == 'twin_val')} train_pairs_by_rule="
          + ",".join(f"{k}:{v}" for k, v in sorted(rule_n.items())), flush=True)
    # 5. 候选段 -> 池子(和 pool 同样的列)。候选页是测试页就停; 几何缺了就停
    tp = test_pages(a.test_folders)
    out, found = [], 0
    for n in a.dirs:
        d = a.data / n
        wcache, why_n, pages, pages_ok, lists_n = {}, Counter(), set(), set(), Counter()
        for fn, rf, src, sg, x0, x1, hp in iter_rows(d / "_segments.csv", ("file", "row_file", "source", "seg", "x0", "x1", "has_pinyin")):
            k = (n.lower(), fn.lower())
            if k not in cand:
                continue
            found += 1
            pg = seg_page[k]
            if pg in tp:
                die("VALPOOL_TEST_PAGE", f"{n}/{fn}: 这段的页 {pg} 是测试页")
            row = rows_pl[n].get(rf)
            W = page_width(d, rf, wcache, src)
            if row is None or row[2] == "" or not W or not (d / "seg_input" / fn).exists():
                die("VALPOOL_NO_GEOMETRY", f"{n}/{fn}: 行清单 / 页宽 / 段图 缺了")
            rown = row[0]
            if rown == "":
                mm = ROW_RE.search(fn)
                rown = str(int(mm.group(1))) if mm else ""
            if why_page.get(pg):
                why = why_page[pg]
            elif k in dev:
                why = "dev"
            elif rown == "0" and float(row[2]) <= 0.10 * W:
                why = "status"
            else:
                why = ""
                pages_ok.add(pg)
            pages.add(pg)
            why_n[why or "eligible"] += 1
            lists_n[len(cand[k])] += 1
            g = gwcz_of(pg) or gwcz_of(fn)
            out.append({"dir": n, "colour": col[n], "file": fn, "row_file": rf, "page": Path(src.replace("\\", "/")).name, "gwcz": g,
                        "row": rown, "seg": sg, "x0": x0, "x1": x1, "has_pinyin": hp, "W": W, "label_y0": row[1], "label_y1": row[2],
                        "input_y0": row[3], "input_y1": row[4], "why": why,
                        "rank": hashlib.sha1(f"gold6:{n}:{fn}".encode("utf-8")).hexdigest()})
        print(f"VALPOOL_STATS {n} colour={col[n]} pages={len(pages)} pages_eligible={len(pages_ok)} candidates={sum(why_n.values())} "
              + " ".join(f"{x}={why_n[x]}" for x in ("eligible", "dev", "status", "twin_train", "twin_val"))
              + f" in_all_lists={lists_n[len(a.lists)]}", flush=True)
    if found != len(cand):
        die("VALPOOL_CAND_LOST", f"候选 {len(cand)} 段, 在段清单里只找回 {found} 段")
    nd = sum(1 for r in out if r["why"] == "dev")
    print(f"VALPOOL_DEV dev_segments={len(dev)} dev_in_candidates={nd}", flush=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    write_csv(tmp / "_pool.csv", POOL_COLS, out)
    write_csv(tmp / "_twins.csv", ["page", "dir", "why", "partners"], tw_out)
    (tmp / "_inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=1), encoding="utf-8")
    rename_retry(tmp, a.out)
    el = [r for r in out if r["why"] == ""]
    print(f"VALPOOL_OK eligible={len(el)} blue={sum(1 for r in el if r['colour'] == 'blue')} white={sum(1 for r in el if r['colour'] == 'white')} "
          f"pages={len({r['page'] for r in el})}", flush=True)


def cmd_rcsame(a) -> None:
    """重切的答案图 == 训练目录里同一段的 seg_input_rc3(10-24 recut / revive 切的, k / m 训练和验证时看的就是它)。
    训练目录里没有这段重切图的算 missing(那段 k 的清单里用的是旧图), 只报数, 不超过 --max-missing 的比例。"""
    import numpy as np
    cand = read_csv(a.gold / "_gold_candidates.csv")
    same = differ = missing = 0
    bad = []
    for r in cand:
        sp = Path(r["src_path"])
        want = imread(sp.parent.parent / a.rc_name / sp.name)
        if want is None:
            missing += 1
            continue
        got = imread(a.rc / "img" / r["file"])
        if got is not None and got.shape == want.shape and np.array_equal(got, want):
            same += 1
        else:
            differ += 1
            bad.append(r["id"])
    print(f"RCSAME_STATS n={len(cand)} same={same} differ={differ} missing={missing}", flush=True)
    if not cand or not same or differ or missing > a.max_missing * len(cand):
        die("RCSAME_FAIL", f"重切图和训练目录的 {a.rc_name} 对不上: 不一样 {differ} 段(例如 {bad[:5]}), 找不到 {missing} 段")
    print("RCSAME_OK", flush=True)


def cmd_inputs(a) -> None:
    rec = json.loads((a.pool / "_inputs.json").read_text(encoding="utf-8"))
    changed = [p for p, h in rec.items() if not Path(p).is_file() or sha256_of(Path(p)) != h]
    print(f"INPUTS_STATS files={len(rec)} changed={len(changed)}", flush=True)
    if not rec or changed:
        die("INPUTS_CHANGED", f"valpool 读过的文件变了: {changed[:5]}")
    print("INPUTS_SAME", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", type=Path, required=True, help="测试配对目录所在的根(E:\\alipay-ai-data)")
    p.add_argument("--work", type=Path, required=True, help="工作目录(切出来的段放这里)")
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--also-snap", nargs="*", default=[], help="只记快照、不切的目录(切段对照用的训练目录)")
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
    e.add_argument("--n", nargs="+", required=True, help="每个目录取几段, 按颜色: blue:400 white:300; all = 剩下的全要")
    e.add_argument("--skip", nargs="*", default=[], help="按颜色先跳过排在最前面的几段(以前导过的那份), 例如 blue:400 white:300")
    e.add_argument("--not-in", type=Path, nargs="*", default=[], help="以前导过的答案目录: 跳过的段必须正好是它们, 取到的段不能和它们重")
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
    s.add_argument("--also-snap", nargs="*", default=[])
    sp = sub.add_parser("splitparity")
    sp.add_argument("--ref", type=Path, required=True, help="当初切过的训练配对目录(只读)")
    sp.add_argument("--fresh", type=Path, required=True, help="今天用 split_segments --out 重切到的目录")
    v = sub.add_parser("valpool")
    v.add_argument("--data", type=Path, required=True, help="训练配对目录所在的根(E:\\alipay-ai-data), 只读")
    v.add_argument("--dirs", nargs="+", required=True, help="训练配对目录(九个)")
    v.add_argument("--colour", nargs="*", default=[], help="目录名里看不出颜色的: 目录名:white 或 目录名:blue")
    v.add_argument("--lists", type=Path, nargs="+", required=True, help="认字模型的训练集目录(train.txt / val.txt): 候选段从它们的 val.txt 来")
    v.add_argument("--also-lists", type=Path, nargs="*", default=[], help="别的训练过的清单目录(拼音带): 只拿来核训练页")
    v.add_argument("--dev", type=Path, nargs="*", default=[], help="用过的答案目录(gold gold2): 它们的段不要")
    v.add_argument("--test-folders", nargs="+", required=True, help="newbatch_* 目录(下面有 test_white / test_blue)")
    v.add_argument("--out", type=Path, required=True)
    rs = sub.add_parser("rcsame")
    rs.add_argument("--gold", type=Path, required=True)
    rs.add_argument("--rc", type=Path, required=True)
    rs.add_argument("--rc-name", default="seg_input_rc3")
    rs.add_argument("--max-missing", type=float, default=0.01, help="训练目录里没有重切图的段最多占这么多")
    ins = sub.add_parser("inputs")
    ins.add_argument("--pool", type=Path, required=True, help="valpool 的输出目录(里面有 _inputs.json)")
    a = ap.parse_args()
    {"prep": cmd_prep, "gate": cmd_gate, "pool": cmd_pool, "export": cmd_export, "parity": cmd_parity,
     "check": cmd_check, "snapshot": cmd_snapshot, "splitparity": cmd_splitparity, "valpool": cmd_valpool,
     "rcsame": cmd_rcsame, "inputs": cmd_inputs}[a.cmd](a)


if __name__ == "__main__":
    main()
