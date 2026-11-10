r"""清掉服务器上已经扫过、对我们没用的 9 月原图, 腾地方下 10 月的图。分两步: plan 只读出删除清单, apply 照清单删。

为什么要这个
------------
经理 11-11: "把没用的删除一下吧"。9 月的原图(E:\BlueImages 蓝图, D:\download2\OtherImages 白图和别的)都用
pick_pinyin 扫过了, 拼音图都拷进了 newbatch_*\hits_* / test_*(是拷贝, 不是链接), 训练、答案集、第 81 / 88 步的整页
也都是拷贝。剩下的原图对我们没用了, 但占着八百多 GB。

★ 硬链接: E:\分图_9月 / D:\分图_9月 是 9 月原图的硬链接(page_fast folders, 10-22, 2,057,490 个)。只删原图一个字节都
  腾不出来, 同一个文件在 分图_9月 里的链接要一起删。别处还有链接的(链接数对不上)一律不删, 单独报出来。
★ 只删同时满足下面每一条的图, 其余一律留着(按第一条不满足的原因分类报数):
    1  直接在 --roots 目录下面(不进子目录), 是图片, 不是符号链接
    2  文件名末尾有日期(_yyyymmddhhmmss.扩展名), 而且不晚于 --max-date(默认 20260930, 只删 9 月)
    3  在 --seen 的扫描名单里(扫过), 而且每一次扫的分数都低于 --keep-score(默认 0.03; 拼音图的线是 蓝 0.12 / 白 0.06,
       0.03 以上的都留着, 以后按模板要多找拼音页还能翻)。扫的时候读坏了(分数是空的)的也留着
    4  文件名不在 --protect 的名单 / 目录里(hits_* / test_* / pages* 目录、*_check*.jsonl 里出现过的)
    5  修改时间(下载日期)早于 --mtime-before(默认 2026-10-10; 9 月的图最后一次扫是 10-09)
    6  链接数 = 1 + 在 分图_9月 里找到的同一个文件(盘号 + 文件号一样)的个数, 也就是别处没有链接
★ plan 只读图, 只写 --out 清单(先写 .tmp 再改名)。打出清单的 sha256; apply 要 --expect 对上才动手, 而且每一行都按上面
  的条件重新判一遍, 大小、修改时间、文件号、链接数有一样和清单不同就不删。
★ apply 先删 分图_9月 里的链接, 最后删原图(删链接失败就不删原图); 只读属性的先去掉再删。中途断了再跑一遍就行: 删过的跳过,
  原图删了链接还在的接着删。删完把 分图_9月 里删空的子目录去掉。不进回收站(进回收站腾不出空间)。
★ 输出: 路径照原样打(有中文), 其余是 ASCII 关键字行; 最后一行 POOL_CLEANUP_PLAN_DONE / POOL_CLEANUP_APPLY_DONE。

用法(服务器)
----
    python -u training\pool_cleanup.py plan --roots E:\BlueImages D:\download2\OtherImages --seen <所有 *_scan.jsonl>
        --protect <hits_* test_* pages* 目录, *_check*.jsonl> --out E:\_move_logs\cleanup_1111\plan.tsv
    python -u training\pool_cleanup.py apply --plan E:\_move_logs\cleanup_1111\plan.tsv --expect <plan 打的 sha256 前 16 位>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seen_names import EXTS, load as load_names  # noqa: E402

VERSION = "pool_cleanup v1"
TREE = "分图_9月"
POOLS = {"blueimages", "otherimages"}
DATE = re.compile(r"_(20\d{2})(\d{2})(\d{2})\d{6}\.[A-Za-z]+$")
GB = 1024 ** 3
REASONS = ["nodate", "date_after", "not_scanned", "scan_error", "score_keep", "protected", "mtime_after",
           "symlink", "stat_error", "no_file_id", "extra_links"]


def norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def under(p: str, top: str) -> bool:
    """p 在 top 下面(都已 norm); 不同盘 commonpath 会报错, 当不在。"""
    try:
        return os.path.commonpath([p, top]) == top and p != top
    except ValueError:
        return False


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_roots(roots) -> list[str]:
    """只认经理的两个图库目录(名字是 BlueImages / OtherImages), 不认盘根、不认上层目录。"""
    out = []
    for r in roots:
        p = os.path.abspath(r)
        if os.path.dirname(p) == p or os.path.basename(p).lower() not in POOLS:
            raise SystemExit(f"ROOT_REFUSED {p}  (only folders named BlueImages / OtherImages)")
        if not os.path.isdir(p):
            raise SystemExit(f"ROOT_MISSING {p}")
        out.append(p)
    if len({os.path.basename(p).lower() for p in out}) != len(out):
        raise SystemExit("ROOT_REFUSED two roots with the same folder name")
    return out


def check_tree(root: str, tree: str) -> str:
    """分图 树: 路径里要有 分图_9月 这一级, 最后一级和图库同名, 和图库在同一个盘(硬链接不跨盘)。"""
    t = os.path.abspath(tree)
    ok = (TREE in Path(t).parts and os.path.basename(t).lower() == os.path.basename(root).lower()
          and os.path.splitdrive(t)[0].lower() == os.path.splitdrive(root)[0].lower())
    if not ok:
        raise SystemExit(f"TREE_REFUSED {t} for root {root}")
    return t


def find_trees(roots: list[str], given) -> dict[str, str | None]:
    """默认: <图库所在盘>\\分图_9月\\<图库名>, 不在就当没有。给了 --link-trees 就按最后一级的名字对上图库。"""
    trees: dict[str, str | None] = {}
    used = set()
    for r in roots:
        if given:
            hit = [t for t in given if os.path.basename(os.path.abspath(t)).lower() == os.path.basename(r).lower()]
            if not hit:
                trees[r] = None
                continue
            t = check_tree(r, hit[0])
            if not os.path.isdir(t):
                raise SystemExit(f"TREE_MISSING {t}")
            used.add(norm(hit[0]))
            trees[r] = t
        else:
            t = os.path.join(os.path.splitdrive(r)[0] + "\\", TREE, os.path.basename(r))
            trees[r] = check_tree(r, t) if os.path.isdir(t) else None
    if given and len(used) != len({norm(t) for t in given}):
        raise SystemExit("TREE_UNUSED a --link-trees folder matches no root")
    return trees


def load_scores(paths) -> tuple[dict[str, float | None], list[tuple[str, int, int]]]:
    """扫描名单 -> {文件名: 最高分}; 每次都读坏了(分数空)的记 None。按文件名认, 不按路径(图挪过盘)。"""
    best: dict[str, float | None] = {}
    notes = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() != ".jsonl" or not p.is_file():
            raise SystemExit(f"SEEN_MISSING {p}")
        n = bad = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:           # noqa: BLE001  半行(上次被砍断的)
                    bad += 1
                    continue
                path = r.get("path")
                if not path:
                    continue
                name = os.path.basename(str(path).replace("\\", "/"))
                s = r.get("score")
                s = float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else None
                n += 1
                if name not in best or (s is not None and (best[name] is None or s > best[name])):
                    best[name] = s
        notes.append((str(p), n, bad))
    return best, notes


def tree_map(tree: str | None) -> tuple[dict[str, int | list[int]], list[str]]:
    """分图 树里的图: {文件名: 目录序号, 同名在几个目录里才是 [序号...]}, 目录表(省内存, 两百万个文件名)。"""
    m: dict[str, int | list[int]] = {}
    dirs: list[str] = []
    if not tree:
        return m, dirs
    for dirpath, _dn, files in os.walk(tree):
        idx = -1
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                if idx < 0:
                    dirs.append(dirpath)
                    idx = len(dirs) - 1
                old = m.get(f)
                if old is None:
                    m[f] = idx
                elif isinstance(old, list):
                    old.append(idx)
                else:
                    m[f] = [old, idx]
    return m, dirs


def idxs(v) -> list[int]:
    return [] if v is None else (v if isinstance(v, list) else [v])


def judge(name: str, mtime: float, scores, protect, P) -> str | None:
    """不用开文件就能判的几条(2 到 5); None = 还要看链接。"""
    m = DATE.search(name)
    if not m:
        return "nodate"
    if m.group(1) + m.group(2) + m.group(3) > P.max_date:
        return "date_after"
    if name not in scores:
        return "not_scanned"
    s = scores[name]
    if s is None:
        return "scan_error"
    if s >= P.keep_score:
        return "score_keep"
    if name in protect:
        return "protected"
    if mtime >= P.mtime_ts:
        return "mtime_after"
    return None


def same_links(name: str, st, tmap, tdirs) -> list[str]:
    """分图 树里和这个文件是同一个文件(盘号 + 文件号一样)的链接; 同名的拷贝不算。"""
    out = []
    for i in idxs(tmap.get(name)):
        lp = os.path.join(tdirs[i], name)
        try:
            ls = os.stat(lp, follow_symlinks=False)
        except OSError:
            continue
        if (ls.st_dev, ls.st_ino) == (st.st_dev, st.st_ino):
            out.append(lp)
    return out


def all_links(path: str):
    """Windows: 同一个文件的所有链接路径(FindFirstFileNameW), 只拿来举例; 出错返回 None。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        first = k32.FindFirstFileNameW
        first.restype = wintypes.HANDLE
        first.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR]
        nxt = k32.FindNextFileNameW
        nxt.restype = wintypes.BOOL
        nxt.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR]
        close = k32.FindClose
        close.argtypes = [wintypes.HANDLE]
        bad = wintypes.HANDLE(-1).value
        drive = os.path.splitdrive(os.path.abspath(path))[0]
        buf = ctypes.create_unicode_buffer(2048)
        n = wintypes.DWORD(2048)
        h = first(path, 0, ctypes.byref(n), buf)
        if not h or h == bad:
            return None
        out = [drive + buf.value]
        try:
            for _ in range(64):
                n = wintypes.DWORD(2048)
                if not nxt(h, ctypes.byref(n), buf):
                    break
                out.append(drive + buf.value)
        finally:
            close(h)
        return out
    except Exception:                   # noqa: BLE001
        return None


class Params:
    def __init__(self, max_date: str, keep_score: float, mtime_before: str):
        if not re.fullmatch(r"20\d{6}", max_date):
            raise SystemExit(f"BAD max_date {max_date}")
        self.max_date = max_date
        self.keep_score = float(keep_score)
        self.mtime_before = mtime_before
        self.mtime_ts = dt.datetime.strptime(mtime_before, "%Y-%m-%d").timestamp()


def load_protect(paths) -> set[str]:
    try:
        names, _notes, _w = load_names(paths)
    except FileNotFoundError as e:
        raise SystemExit(f"PROTECT_MISSING {e}")
    return names


def cmd_plan(a) -> int:
    t0 = time.time()
    P = Params(a.max_date, a.keep_score, a.mtime_before)
    roots = check_roots(a.roots)
    trees = find_trees(roots, a.link_trees)
    scores, snotes = load_scores(a.seen)
    for p, n, bad in snotes:
        print(f"SEEN {p} ROWS {n} BAD {bad}", flush=True)
    print(f"SEEN_LISTS {len(snotes)} NAMES {len(scores)}", flush=True)
    protect = load_protect(a.protect)
    print(f"PROTECT_SOURCES {len(a.protect or [])} NAMES {len(protect)}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    freed_by_drive: dict[str, int] = defaultdict(int)
    tot_rows = tot_bytes = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as fo:
        fo.write(f"#\tversion\t{VERSION}\n#\tmax_date\t{P.max_date}\n#\tkeep_score\t{P.keep_score}\n"
                 f"#\tmtime_before\t{P.mtime_before}\n")
        for r in roots:
            fo.write(f"#\troot\t{r}\t{trees[r] or '-'}\n")
        for p in a.seen:
            fo.write(f"#\tseen\t{os.path.abspath(p)}\n")
        for p in a.protect or []:
            fo.write(f"#\tprotect\t{os.path.abspath(p)}\n")
        for r in roots:
            tree = trees[r]
            print(f"  reading tree of {r} ({time.time() - t0:.0f} s)", flush=True)
            tmap, tdirs = tree_map(tree)
            tree_files = sum(len(idxs(v)) for v in tmap.values())
            print(f"TREE {tree} FILES {tree_files} DIRS {len(tdirs)}" if tree else f"TREE NONE for {r}", flush=True)
            keep = defaultdict(lambda: [0, 0])
            month = defaultdict(lambda: [0, 0, 0, 0])       # 张数, 字节, 删, 留
            day = defaultdict(lambda: [0, 0])
            pool_names: set[str] = set()
            n_img = b_img = subdirs = other = c_rows = c_bytes = c_links = 0
            dmin = dmax = None
            extra_eg: list[tuple[str, int, int]] = []
            with os.scandir(r) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            subdirs += 1
                            continue
                        if os.path.splitext(e.name)[1].lower() not in EXTS:
                            other += 1
                            continue
                        st0 = e.stat(follow_symlinks=False)
                    except OSError:
                        other += 1
                        continue
                    name = e.name
                    pool_names.add(name)
                    n_img += 1
                    b_img += st0.st_size
                    m = DATE.search(name)
                    ym = m.group(1) + m.group(2) if m else "nodate"
                    if m and ym in a.days:
                        day[ym + m.group(3)][0] += 1
                        day[ym + m.group(3)][1] += st0.st_size
                    month[ym][0] += 1
                    month[ym][1] += st0.st_size
                    why = "symlink" if e.is_symlink() else judge(name, st0.st_mtime, scores, protect, P)
                    links: list[str] = []
                    st = None
                    if why is None:
                        try:
                            st = os.stat(e.path, follow_symlinks=False)
                        except OSError:
                            st = None
                            why = "stat_error"
                    if why is None:
                        if not st.st_ino:
                            why = "no_file_id"
                        else:
                            links = same_links(name, st, tmap, tdirs)
                            if st.st_nlink != 1 + len(links):
                                why = "extra_links"
                                if len(extra_eg) < 5:
                                    extra_eg.append((e.path, st.st_nlink, len(links)))
                    if why:
                        keep[why][0] += 1
                        keep[why][1] += st0.st_size
                        month[ym][3] += 1
                        continue
                    fo.write(f"{e.path}\t{st.st_size}\t{st.st_mtime_ns}\t{st.st_dev}\t{st.st_ino}\t{st.st_nlink}\t"
                             f"{'|'.join(links)}\n")
                    c_rows += 1
                    c_bytes += st.st_size
                    c_links += len(links)
                    month[ym][2] += 1
                    d = m.group(1) + m.group(2) + m.group(3)
                    dmin = d if dmin is None or d < dmin else dmin
                    dmax = d if dmax is None or d > dmax else dmax
                    if c_rows % 200000 == 0:
                        print(f"  ... {r} candidates {c_rows} ({time.time() - t0:.0f} s)", flush=True)
            orphan = orphan_b = 0
            for nm, v in tmap.items():
                if nm not in pool_names:
                    for i in idxs(v):
                        orphan += 1
                        try:
                            orphan_b += os.stat(os.path.join(tdirs[i], nm)).st_size
                        except OSError:
                            pass
            print(f"ROOT {r} IMAGES {n_img} GB {b_img / GB:.1f} SUBDIRS {subdirs} OTHER_FILES {other}", flush=True)
            print(f"CAND FILES {c_rows} GB {c_bytes / GB:.1f} TREE_LINKS {c_links} DATES {dmin} {dmax}")
            for k in REASONS:
                if k in keep:
                    print(f"KEEP {k} FILES {keep[k][0]} GB {keep[k][1] / GB:.1f}")
            for k in sorted(month):
                n, b, c, kp = month[k]
                print(f"MONTH {k} FILES {n} GB {b / GB:.1f} CAND {c} KEEP {kp}")
            for k in sorted(day):
                print(f"DAY {k} FILES {day[k][0]} GB {day[k][1] / GB:.2f}")
            print(f"ORPHAN_TREE_FILES {orphan} GB {orphan_b / GB:.1f}  (in the tree, not in the pool; not touched)")
            for p, nl, nf in extra_eg:
                lk = all_links(p)
                print(f"EXTRA_LINK_SAMPLE {p} NLINK {nl} IN_TREE {nf} ALL {' | '.join(lk) if lk else '?'}")
            freed_by_drive[os.path.splitdrive(r)[0].upper()] += c_bytes
            tot_rows += c_rows
            tot_bytes += c_bytes
    os.replace(tmp, out)
    for drv, b in sorted(freed_by_drive.items()):
        u = shutil.disk_usage(drv + "\\")
        print(f"DISK {drv} FREE_NOW {u.free / GB:.1f} GB FREE_AFTER {(u.free + b) / GB:.1f} GB")
    h = sha256(out)
    print(f"PLAN {out} ROWS {tot_rows} GB {tot_bytes / GB:.1f} SHA256 {h}")
    print(f"TIME {time.time() - t0:.0f} s")
    print("POOL_CLEANUP_PLAN_DONE", flush=True)
    return 0


def remove(p: str) -> str | None:
    """删一个链接; 只读属性(所有链接共用)先去掉再删。返回 None = 删了, 否则是错误说明。"""
    try:
        os.remove(p)
        return None
    except FileNotFoundError:
        return None
    except PermissionError:
        try:
            os.chmod(p, stat.S_IWRITE)
            os.remove(p)
            return None
        except OSError as e:
            return f"{type(e).__name__}: {e}"
    except OSError as e:
        return f"{type(e).__name__}: {e}"


def cmd_apply(a) -> int:
    t0 = time.time()
    plan = Path(a.plan)
    if not plan.is_file():
        raise SystemExit(f"PLAN_MISSING {plan}")
    h = sha256(plan)
    if len(a.expect) < 12 or not h.startswith(a.expect.lower()):
        print(f"SHA_MISMATCH plan {h} expect {a.expect}  (nothing deleted)")
        return 2
    head: dict[str, list] = defaultdict(list)
    with plan.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#\t"):
                break
            k, *v = line.rstrip("\n").split("\t")[1:]
            head[k].append(v)
    if head["version"] != [[VERSION]]:
        raise SystemExit(f"PLAN_VERSION {head['version']}")
    P = Params(head["max_date"][0][0], float(head["keep_score"][0][0]), head["mtime_before"][0][0])
    roots = check_roots([v[0] for v in head["root"]])
    trees = {r: (check_tree(r, v[1]) if v[1] != "-" else None) for r, v in zip(roots, head["root"])}
    root_n = {norm(r) for r in roots}
    tree_n = [norm(t) for t in trees.values() if t]
    scores, _ = load_scores([v[0] for v in head["seen"]])
    protect = load_protect([v[0] for v in head["protect"]])
    print(f"APPLY plan {plan} SHA256 {h} SEEN_NAMES {len(scores)} PROTECT_NAMES {len(protect)}", flush=True)
    cnt = defaultdict(int)
    freed = 0
    errs: list[str] = []
    rows = 0
    with plan.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#\t"):
                continue
            if a.limit and rows >= a.limit:
                break
            rows += 1
            if rows % 100000 == 0:
                print(f"  ... rows {rows} deleted {cnt['deleted']} freed {freed / GB:.1f} GB ({time.time() - t0:.0f} s)",
                      flush=True)
            f = line.rstrip("\n").split("\t")
            if len(f) != 7:
                cnt["bad_row"] += 1
                continue
            path, size, mt, dev, ino, nlink = f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])
            links = [x for x in f[6].split("|") if x]
            name = os.path.basename(path)
            # 只删图库目录下面一层的原图, 链接只删 分图 树里的
            if norm(os.path.dirname(path)) not in root_n or not all(
                    any(under(norm(lp), t) for t in tree_n) for lp in links):
                cnt["outside"] += 1
                continue
            try:
                st = os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                st = None
            except OSError:
                cnt["stat_error"] += 1
                continue
            if st is None:
                # 原图没了(上次删到一半断了, 或者别人删了): 清单里的链接要是还在、还是那个文件、别处没链接, 接着删
                alive = []
                for lp in links:
                    try:
                        ls = os.stat(lp, follow_symlinks=False)
                    except OSError:
                        continue
                    if (ls.st_dev, ls.st_ino) == (dev, ino):
                        alive.append((lp, ls))
                if not alive:
                    cnt["already_gone"] += 1
                    continue
                ls0 = alive[0][1]
                if ls0.st_nlink != len(alive) or ls0.st_size != size or judge(name, ls0.st_mtime, scores, protect, P):
                    cnt["changed"] += 1
                    continue
                bad = [e for e in (remove(lp) for lp, _ in alive) if e]
                if bad:
                    cnt["failed"] += 1
                    if len(errs) < 20:
                        errs.append(f"{path} (links): {bad[0]}")
                    continue
                cnt["resumed"] += 1
                cnt["links_deleted"] += len(alive)
                freed += size
                continue
            if (st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino, st.st_nlink) != (size, mt, dev, ino, nlink) \
                    or stat.S_ISLNK(st.st_mode) or st.st_nlink != 1 + len(links):
                cnt["changed"] += 1
                continue
            why = judge(name, st.st_mtime, scores, protect, P)
            if why:
                cnt["rule_" + why] += 1
                continue
            ok = True
            for lp in links:
                try:
                    ls = os.stat(lp, follow_symlinks=False)
                except OSError:
                    ok = False
                    break
                if (ls.st_dev, ls.st_ino) != (dev, ino):
                    ok = False
                    break
            if not ok:
                cnt["changed"] += 1
                continue
            bad = [e for e in (remove(lp) for lp in links) if e]
            if not bad:
                e = remove(path)
                bad = [e] if e else []
            if bad:
                cnt["failed"] += 1
                if len(errs) < 20:
                    errs.append(f"{path}: {bad[0]}")
                continue
            cnt["deleted"] += 1
            cnt["links_deleted"] += len(links)
            freed += size
    pruned = 0
    for t in trees.values():
        if not t or not os.path.isdir(t):
            continue
        for dirpath, _dn, _fn in os.walk(t, topdown=False):
            if norm(dirpath) == norm(t):
                continue
            try:
                os.rmdir(dirpath)           # 只有空目录才删得掉
                pruned += 1
            except OSError:
                pass
    print(f"ROWS {rows} " + " ".join(f"{k.upper()} {v}" for k, v in sorted(cnt.items())))
    print(f"FREED_GB {freed / GB:.1f} EMPTY_DIRS_REMOVED {pruned}")
    for e in errs[:20]:
        print(f"FAILED_EG {e}")
    for drv in sorted({os.path.splitdrive(r)[0].upper() for r in roots}):
        print(f"DISK {drv} FREE {shutil.disk_usage(drv + chr(92)).free / GB:.1f} GB")
    print(f"TIME {time.time() - t0:.0f} s")
    if cnt["failed"] or cnt["bad_row"] or cnt["outside"] or cnt["stat_error"]:
        print("POOL_CLEANUP_APPLY_INCOMPLETE", flush=True)
        return 3
    if not a.limit:
        done = plan.with_name("apply_done.txt")
        tmpd = done.with_name(done.name + ".tmp")
        tmpd.write_text(f"{h} deleted {cnt['deleted']} resumed {cnt['resumed']} freed_gb {freed / GB:.1f}\n", encoding="utf-8")
        os.replace(tmpd, done)
    print("POOL_CLEANUP_APPLY_DONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="只读: 列出能删的图, 写清单")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--seen", nargs="+", required=True, help="扫描名单 *_scan.jsonl")
    p.add_argument("--protect", nargs="*", default=[], help="这些目录 / 名单里出现的文件名不删")
    p.add_argument("--link-trees", nargs="*", default=None, help="分图 树(默认 <盘>\\分图_9月\\<图库名>)")
    p.add_argument("--max-date", default="20260930")
    p.add_argument("--keep-score", type=float, default=0.03)
    p.add_argument("--mtime-before", default="2026-10-10")
    p.add_argument("--days", nargs="*", default=["202610", "202611"], help="按天点数的月份")
    p.add_argument("--out", required=True)
    q = sub.add_parser("apply", help="照清单删")
    q.add_argument("--plan", required=True)
    q.add_argument("--expect", required=True, help="plan 打的 sha256 前面至少 12 位")
    q.add_argument("--limit", type=int, default=0, help="只做前这么多行(试跑)")
    a = ap.parse_args()
    return cmd_plan(a) if a.cmd == "plan" else cmd_apply(a)


if __name__ == "__main__":
    sys.exit(main())
