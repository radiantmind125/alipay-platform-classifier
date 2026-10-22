r"""切图修正: 每一段按**它自己的字**定上下边, 不再整行共用一条"成员中位"的边。

为什么(2026-10-23)
------------------
make_pinyin_pairs 切行时, 行的上下边取的是这一行所有字块的**中位**(故意的: 防 〈 箭头这种高元件把边顶进上面的拼音带),
再只留 1 像素余量; split_segments 切段时整行共用这条边。于是一行里**比一般字高、或者位置偏高偏低**的那一段就被切掉一截:
    大号红色券面额(0.9元 -> 只剩下半截, 老师读成 U.7元)、蓝图标题行右上角的 回首页(上半截没了)、大一号字体的名字 ...
金标准上模型 h 剩下的错里, 这是最大的一类(标签修完以后)。

做法(只改上下边, 左右不动):
    整页: 和 make_pinyin_pairs.process 一模一样地算 字块 / 拼音块 / 拼音带 / 行 / 每行的 label 边 (la, lb) 和 input 边 (ia, ib)
    每一段 x 区间 [a, b) 里, 挑属于这一段的字块:
        横向至少一半在段里; 纵向和这一行的 label 边 [la, lb] 重叠至少自身高度的 30%(上一行下伸、下一行顶的不算);
        中心在 [la - 0.35 行高, lb + 0.35 行高] 里(行边夹在两行中间时, 上下两行的字中心在外面);
        不是拼音块(annotation_labels 判的), 中心不在拼音带里; 不是超高的竖线 / 框 / 图标列(高 > 2.4 倍行高)、不是又扁又长的横线
        (试过按"填满外接框的比例"去掉实心图标: 不行, 粗体大号金额也填 0.56~0.64, 和图标 0.55 分不开)
    新的 input 边 = 原来的边 和 这些字块的上下沿 取外包(上下各留 2 像素), 最多各往外扩 2 倍行高, 不越过上下邻行
★ 原来的边一律保留(只往外扩, 不往里收): 拼音还在上面, 正常的段和原来一模一样。

用法
----
    python training/recrop.py demo --src 图目录或名单 [...] --out 输出目录 [--n 40]      # 出前后对比核对图
    python training/recrop.py regold --gold 答案目录 [...] --out-root 输出 [--min-ext 0.15 --suffix _rcB]   # 金标准从原图重切
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import annotation_labels, local_background, text_mask  # noqa: E402
from make_pinyin_pairs import FALLBACK_SHIFT_RATIO, LABEL_PAD, MIN_ROW_H, _pinyin_bands, _rows  # noqa: E402

PAD_Y = 2          # 新边上下各留几像素
CAP = 2.0          # 每边最多往外扩几倍行高
V_OVERLAP = 0.3    # 字块和行的 label 边至少重叠自身高度的这么多
CENTER = 0.35      # 字块中心离这一行的 label 边不能超过这么多倍行高(大号金额底对齐, 中心正好在行顶附近)
TALL = 2.4         # 比行高这么多倍还高的是竖线 / 框 / 一列图标, 不算字(核对图: 大号金额最高 2.0 倍, 图标列 2.49 倍)
CONTAINER_H, CONTAINER_N = 1.2, 3   # 比行高 1.2 倍高、外接框里整个装着 >= 3 块别的: 按钮外圈 / 图标 / 头像, 不算字


def layout(img: np.ndarray) -> dict:
    """整页的字块、拼音、行 —— 和 make_pinyin_pairs.process 同一套算法、同一组常数。"""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = text_mask(gray, local_background(gray))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    _, kept, _, big = annotation_labels(mask)
    bands = _pinyin_bands(kept, st, big)
    rows = _rows(big, bands)
    out = {"H": H, "W": W, "st": st, "kept": set(int(k) for k in kept), "bands": bands, "rows": []}
    if not rows:
        return out
    char_h = float(np.median([b - a for a, b in rows]))
    fallback = int(round(FALLBACK_SHIFT_RATIO * char_h))
    for i, (y0, y1) in enumerate(rows):
        above = [(a, b) for a, b in bands if b <= y0 + 2 and (y0 - b) < (y1 - y0)]
        la = max(0, y0 - LABEL_PAD)
        if above:
            la = min(y0, max(la, max(b for _, b in above) + 1))
        lb = min(H, y1 + LABEL_PAD)
        if lb - la < MIN_ROW_H:
            continue
        ia = max(0, (min(a for a, _ in above) - 1) if above else (y0 - fallback))
        out["rows"].append({"row": i, "la": la, "lb": lb, "ia": ia, "ib": lb, "has_py": int(bool(above))})
    return out


def _encloses(st, i: int, near: list) -> int:
    """第 i 块的外接框里整个装着几块别的(按钮外圈、图标、头像里面都装着字或花纹; 字基本不装)。"""
    x, y, w, h = (int(v) for v in st[i][:4])
    n = 0
    for j in near:
        if j == i:
            continue
        xj, yj, wj, hj = (int(v) for v in st[j][:4])
        if xj > x and yj > y and xj + wj < x + w and yj + hj < y + h:
            n += 1
    return n


def seg_extent(L: dict, la: int, lb: int, a: int, b: int):
    """这一段自己的字块的上下沿 (top, bottom); 一个都没有返回 None。(numpy 一次筛完, 51 万段也快)"""
    st, bands = L["st"], L["bands"]
    if "kept_mask" not in L:
        km = np.zeros(len(st), bool)
        km[[k for k in L["kept"] if k < len(st)]] = True
        L["kept_mask"] = km
    row_h = max(1, lb - la)
    X, Y, Wc, Hc, A = (st[:, k].astype(np.int64) for k in range(5))
    idx = np.arange(len(st))
    sel = (idx > 0) & (A >= 6) & ~L["kept_mask"]
    sel &= ~((Hc > TALL * row_h) | ((Wc > 3 * row_h) & (Hc < 0.25 * row_h)))
    sel &= (np.minimum(X + Wc, b) - np.maximum(X, a)) >= 0.5 * Wc
    sel &= (np.minimum(Y + Hc, lb) - np.maximum(Y, la)) >= V_OVERLAP * Hc
    cy = Y + Hc / 2
    for p, q in bands:
        sel &= ~((cy >= p) & (cy <= q))
    sel &= (cy >= la - CENTER * row_h) & (cy <= lb + CENTER * row_h)
    cand = np.nonzero(sel)[0]
    if len(cand) and (Hc[cand] > CONTAINER_H * row_h).any():
        nearm = (idx > 0) & (A >= 6) & (np.minimum(X + Wc, b) > np.maximum(X, a)) & (Y < lb + 3 * row_h) & (Y + Hc > la - 3 * row_h)
        near = np.nonzero(nearm)[0]
        keep = []
        for i in cand:
            if Hc[i] > CONTAINER_H * row_h:
                inside = ((X[near] > X[i]) & (Y[near] > Y[i]) & (X[near] + Wc[near] < X[i] + Wc[i]) & (Y[near] + Hc[near] < Y[i] + Hc[i])
                          & (near != i)).sum()
                if inside >= CONTAINER_N:
                    continue            # 外圈: 按钮框 / 图标 / 头像, 里面装着别的块 —— 不按它扩(10-23: 免费领 92 -> 157 像素, 字被缩小)
            keep.append(i)
        cand = np.array(keep, dtype=np.int64)
    if not len(cand):
        return None
    return int(Y[cand].min()), int((Y[cand] + Hc[cand]).max())


def neighbours(L: dict, la: int, lb: int) -> tuple:
    """这一行上面那行的 label 底、下面那行的 input 顶(含它的拼音)、下面最近的拼音带顶 —— 新边不许越过, 免得把邻行切进来。"""
    rows = sorted(L["rows"], key=lambda r: r["la"])
    up, down = 0, L["H"]
    for r in rows:
        if r["la"] < la:
            up = max(up, r["lb"] + 1)
        elif r["la"] > la:
            down = min(down, r["ia"] - 1)
    # ★ 这一行下面的拼音带一定是**下一行**的(自己的拼音在上面): 往下扩不许进去。
    #   下一行字小、没被聚成行时(芭芭农场 限时翻倍、按钮上的 免费领), 只有它的拼音带拦得住
    for a, b in L["bands"]:
        if a >= lb - 2:                 # 带在这一行底下(行里面的小字拼音不算)
            down = min(down, a - 1)
    return up, down


def new_box(L: dict, la: int, lb: int, ia: int, ib: int, a: int, b: int) -> tuple:
    """这一段新的 input 上下边 (ia2, ib2): 原来的边和自己字块的外包, 只扩不收;
    每边最多扩 CAP 倍行高, 而且不越过上下邻行(2026-10-23 核对图: 图标、按钮底色跨两行, 会把下一行带进来)。"""
    ext = seg_extent(L, la, lb, a, b)
    if ext is None:
        return ia, ib
    row_h = max(1, lb - la)
    top, bot = ext
    up, down = neighbours(L, la, lb)
    ia2 = max(0, int(ia - CAP * row_h), min(up, ia), min(ia, top - PAD_Y))
    ib2 = min(L["H"], int(ib + CAP * row_h), max(down, ib), max(ib, bot + PAD_Y))
    return ia2, ib2


def demo(a) -> None:
    """本地看效果: 每张图按现有算法切行切段, 新旧上下边差 >= 3 像素的段出前后对比图。"""
    import random
    from PIL import Image, ImageDraw
    from make_check_sheet import _cjk_font
    from split_segments import PAD, segments
    import page_type as pt
    font = _cjk_font(18)
    paths = [p for p, _t in pt.collect(a.src)]
    random.Random(a.seed).shuffle(paths)
    paths = paths[: a.n]
    pairs, n_seg, n_changed = [], 0, 0
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        L = layout(img)
        for r in L["rows"]:
            lab = img[r["la"]:r["lb"], :]
            for x0, x1 in segments(lab):
                sa, sb = max(0, x0 - PAD), min(L["W"], x1 + PAD + 1)
                n_seg += 1
                ia2, ib2 = new_box(L, r["la"], r["lb"], r["ia"], r["ib"], sa, sb)
                if (r["ia"] - ia2) + (ib2 - r["ib"]) >= 3:
                    n_changed += 1
                    pairs.append((Path(p).name, r["row"], img[r["ia"]:r["ib"], sa:sb], img[ia2:ib2, sa:sb],
                                  r["ia"] - ia2, ib2 - r["ib"]))
    print(f"  图 {len(paths)}  段 {n_seg:,}  上下边变了的 {n_changed:,} ({n_changed / max(1, n_seg):.1%})", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    random.Random(1).shuffle(pairs)
    per = 30
    for k in range(0, min(len(pairs), a.sheets * per), per):
        part = pairs[k:k + per]
        rowsimg = []
        for name, row, old, new, up, down in part:
            def fit(im):
                s = min(1.0, 520 / max(1, im.shape[1]))
                return cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
            o, n_ = fit(old), fit(new)
            h = max(o.shape[0], n_.shape[0]) + 6
            can = np.full((h + 22, 1100, 3), 255, np.uint8)
            can[22:22 + o.shape[0], 0:o.shape[1]] = o
            can[22:22 + n_.shape[0], 560:560 + n_.shape[1]] = n_
            pil = Image.fromarray(can[:, :, ::-1])
            ImageDraw.Draw(pil).text((4, 1), f"{name[:40]} r{row}   旧(左)  新(右) 上扩 {up} 下扩 {down}", font=font, fill=(200, 0, 0))
            rowsimg.append(np.array(pil)[:, :, ::-1])
        sheet = np.vstack(rowsimg)
        cv2.imencode(".png", sheet)[1].tofile(str(a.out / f"recrop_{k // per:02d}.png"))
    print(f"  核对图 -> {a.out}", flush=True)


def regold(a) -> None:
    """把金标准的段从原图重切一遍(新切法), 出一份同样格式的答案目录(img + 三个 csv 原样拷), 好拿同一套答案比新旧切图。
    每段先按**旧**切法从原图切一遍, 和答案目录里存的裁图逐像素比 —— 一样才说明行、段、原图都对得上。"""
    import csv
    import shutil
    seg_cache, pl_cache = {}, {}
    # ★ _segments.csv 里记的原图路径是切图那天的(比如 D 盘的 alipay-ai-data, 后来整个搬到了 E 盘 —— 2026-10-23 服务器上就是这样报错的)。
    #   找不到时: 按文件名查 page_type.csv(服务器上新跑的, 路径是现在的) -> 再试换盘符 -> 都没有才算找不到(用旧图, 记下来)
    index = {}
    for fn in a.page_index:
        if Path(fn).exists():
            with open(fn, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    index.setdefault(Path(r["path"]).name.lower(), r["path"])
    how = {"原路径": 0, "按文件名找到": 0, "换盘符找到": 0}

    def resolve(src: str):
        if Path(src).exists():
            how["原路径"] += 1
            return src
        q = index.get(Path(src).name.lower())
        if q and Path(q).exists():
            how["按文件名找到"] += 1
            return q
        for drv in "EDCF":
            q = drv + src[1:]
            if len(src) > 2 and src[1] == ":" and Path(q).exists():
                how["换盘符找到"] += 1
                return q
        return None

    def segs_of(d: Path) -> dict:
        if d not in seg_cache:
            with (d / "_segments.csv").open(encoding="utf-8-sig", newline="") as f:
                seg_cache[d] = {r["file"]: r for r in csv.DictReader(f)}
        return seg_cache[d]

    def rows_of(d: Path) -> dict:
        if d not in pl_cache:
            with (d / "_pairs_labeled.csv").open(encoding="utf-8-sig", newline="") as f:
                pl_cache[d] = {r["file"]: r for r in csv.DictReader(f)}
        return pl_cache[d]

    for g in a.gold:
        out = a.out_root / f"{g.name}{a.suffix}"
        if out.exists():
            print(f"  ★ {out} 已经有了, 不覆盖", flush=True)
            sys.exit(2)
        (out / "img").mkdir(parents=True)
        for fn in ("_gold_candidates.csv", "_gold.csv", "_relevance.csv"):
            if (g / fn).exists():
                shutil.copyfile(g / fn, out / fn)
        with (g / "_gold_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
            cand = list(csv.DictReader(f))
        same = differ = missing = changed = 0
        log = {}
        jobs = {}                                     # 原图 -> [(答案行, 段行, 行行)]: 一张原图只解码、只算一次字块
        for r in cand:
            sp = Path(r["src_path"])
            d = sp.parent.parent
            if not (d / "_segments.csv").exists():
                d = a.pairs_root / d.name             # 路径盘符 / 根目录不一样时按目录名找
            try:
                s_ = segs_of(d)[sp.name]
                row = rows_of(d)[s_["row_file"]]
            except (KeyError, FileNotFoundError):
                missing += 1
                shutil.copyfile(g / "img" / r["file"], out / "img" / r["file"])
                log[r["id"]] = (r["id"], r["file"], "段或行找不到, 用旧图", 0, 0)
                continue
            jobs.setdefault(s_["source"], []).append((r, s_, row))
        for src0, items in jobs.items():
            src = resolve(src0)
            img = None
            if src:
                try:
                    img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
                except Exception:  # noqa: BLE001
                    img = None
            L = layout(img) if img is not None else None
            for r, s_, row in items:
                if img is None:
                    missing += 1
                    shutil.copyfile(g / "img" / r["file"], out / "img" / r["file"])
                    log[r["id"]] = (r["id"], r["file"], "原图找不到或读不出, 用旧图", 0, 0)
                    continue
                la, lb = int(row["label_y0"]), int(row["label_y1"])
                ia, ib = int(row["input_y0"]), int(row["input_y1"])
                x0, x1 = int(s_["x0"]), int(s_["x1"])
                old = img[ia:ib, x0:x1]
                stored = cv2.imdecode(np.fromfile(str(g / "img" / r["file"]), np.uint8), cv2.IMREAD_COLOR)
                if stored is not None and stored.shape == old.shape and np.array_equal(stored, old):
                    same += 1
                else:
                    differ += 1
                ia2, ib2 = new_box(L, la, lb, ia, ib, x0, x1)
                if max(ia - ia2, ib2 - ib) < a.min_ext * max(1, lb - la):
                    ia2, ib2 = ia, ib                 # 只差几像素的不动(B 方案: 只改真被切掉一截的段)
                cv2.imencode(".png", img[ia2:ib2, x0:x1])[1].tofile(str(out / "img" / r["file"]))
                if (ia2, ib2) != (ia, ib):
                    changed += 1
                log[r["id"]] = (r["id"], r["file"], "ok" if (stored is not None and stored.shape == old.shape and np.array_equal(stored, old)) else "旧切法对不上",
                                ia - ia2, ib2 - ib)
        log = [log[r["id"]] for r in cand if r["id"] in log]
        with (out / "_recrop_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "file", "status", "up", "down"])
            w.writerows(log)
        print(f"  {g.name}: 原图 {len(jobs)} 张  " + "  ".join(f"{k} {v}" for k, v in how.items()), flush=True)
        for k in how:
            how[k] = 0
        print(f"  {g.name}: 段 {len(cand)}  旧切法和存的裁图逐像素一样 {same}  不一样 {differ}  找不到 {missing}  新切法上下边变了 {changed}"
              f"  -> {out}", flush=True)
        if differ:
            print(f"  ★ {g.name}: 有 {differ} 段旧切法对不上存的裁图(行 / 段 / 原图不一致), 先看看", flush=True)
        # 给服务器脚本读的(纯 ASCII)
        print(f"REGOLD_STATS {g.name} total={len(cand)} same={same} differ={differ} missing={missing} changed={changed}", flush=True)
    print("REGOLD_OK", flush=True)


def load_index(files: list) -> dict:
    """文件名(小写) -> 现在的路径: 从 page_type.csv 的 path 列来(原图搬过家时按文件名找)。"""
    import csv
    index = {}
    for fn in files:
        if Path(fn).exists():
            with open(fn, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    index.setdefault(Path(r["path"]).name.lower(), r["path"])
    return index


def resolve_path(src: str, index: dict) -> tuple:
    """(现在的路径或 None, 怎么找到的)。原路径 -> 按文件名查 index -> 换盘符。"""
    if Path(src).exists():
        return src, "orig"
    q = index.get(Path(src).name.lower())
    if q and Path(q).exists():
        return q, "index"
    if len(src) > 2 and src[1] == ":":
        for drv in "EDCF":
            q = drv + src[1:]
            if Path(q).exists():
                return q, "drive"
    return None, "missing"


_INDEX = {}


def _recut_init(index: dict) -> None:
    global _INDEX
    cv2.setNumThreads(1)
    _INDEX = index


def _recut_page(job) -> list:
    """一张原图: 解码一次、算一次字块, 把它的每一段重切到 seg_input_rc。
    每段先按旧切法切, 和 seg_input 里存的逐像素比; 对不上(行 / 段 / 原图不一致)就**原样拷旧图**, 不拿新切法冒险。
    已经切过的(seg_input_rc 里有了)跳过 —— 停了再跑接着做。"""
    src, items = job
    out = []
    todo = [it for it in items if not (Path(it["dir"]) / "seg_input_rc" / it["file"]).exists()]
    for it in items:
        if it not in todo:
            out.append((it["dir"], it["file"], "done_before", 0, 0))
    if not todo:
        return out
    path, how = resolve_path(src, _INDEX)
    img = None
    if path:
        try:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            img = None
    L = layout(img) if img is not None else None
    for it in todo:
        d = Path(it["dir"])
        dst = d / "seg_input_rc" / it["file"]
        stored_p = d / "seg_input" / it["file"]
        if img is None:
            if stored_p.exists():
                dst.write_bytes(stored_p.read_bytes())
            out.append((it["dir"], it["file"], "page_missing", 0, 0))
            continue
        la, lb, ia, ib, x0, x1 = (int(it[k]) for k in ("la", "lb", "ia", "ib", "x0", "x1"))
        stored = cv2.imdecode(np.fromfile(str(stored_p), np.uint8), cv2.IMREAD_COLOR) if stored_p.exists() else None
        old = img[ia:ib, x0:x1]
        if stored is None or stored.shape != old.shape or not np.array_equal(stored, old):
            if stored_p.exists():
                dst.write_bytes(stored_p.read_bytes())
            out.append((it["dir"], it["file"], "old_mismatch", 0, 0))
            continue
        ia2, ib2 = new_box(L, la, lb, ia, ib, x0, x1)
        cv2.imencode(".png", img[ia2:ib2, x0:x1])[1].tofile(str(dst))
        out.append((it["dir"], it["file"], "ok_" + how, ia - ia2, ib2 - ib))
    return out


def recut(a) -> None:
    """训练集的段全部按新切法重切到 <目录>\\seg_input_rc\\ (旧的 seg_input 一张不动)。只切在训练里的段(bad != 1 而且有字)。
    按原图分组并行; 每段先核对旧切法 == 存的 seg_input(对不上的原样拷旧图); 出 _recut_log.csv 和纯 ASCII 的 RECUT_STATS。"""
    import csv
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    index = load_index(a.page_index)
    print(f"  按文件名找原图的表 {len(index):,} 张", flush=True)
    for d in a.pairs:
        with (d / "_pairs_labeled.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
        jobs = {}
        n_skip = 0
        with (d / "_segments.csv").open(encoding="utf-8-sig", newline="") as f:
            for s in csv.DictReader(f):
                if s.get("bad") == "1" or not (s.get("text") or "").strip():
                    n_skip += 1
                    continue
                r = rows.get(s["row_file"])
                if r is None:
                    n_skip += 1
                    continue
                jobs.setdefault(s["source"], []).append({
                    "dir": str(d), "file": s["file"], "x0": s["x0"], "x1": s["x1"],
                    "la": r["label_y0"], "lb": r["label_y1"], "ia": r["input_y0"], "ib": r["input_y1"]})
        (d / "seg_input_rc").mkdir(exist_ok=True)
        n_seg = sum(len(v) for v in jobs.values())
        print(f"  {d.name}: 原图 {len(jobs):,} 张  要切 {n_seg:,} 段  (不在训练里 / 找不到行 跳过 {n_skip:,})", flush=True)
        cnt, log = Counter(), []
        done = 0
        with ProcessPoolExecutor(a.workers, initializer=_recut_init, initargs=(index,)) as ex:
            for res in ex.map(_recut_page, list(jobs.items()), chunksize=4):
                for dd, fn, st_, up, down in res:
                    cnt[st_.split("_")[0] if st_.startswith("ok") else st_] += 1
                    if st_.startswith("ok") and (up or down):
                        cnt["changed"] += 1
                    log.append((fn, st_, up, down))
                done += 1
                if done % 2000 == 0:
                    print(f"    {done:,}/{len(jobs):,} 张图", flush=True)
        with (d / "_recut_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "status", "up", "down"])
            w.writerows(log)
        print(f"RECUT_STATS {d.name} total={n_seg} ok={cnt['ok']} changed={cnt['changed']} old_mismatch={cnt['old_mismatch']} "
              f"page_missing={cnt['page_missing']} done_before={cnt['done_before']}", flush=True)
    print("RECUT_OK", flush=True)


def relist(a) -> None:
    """训练集清单换成重切的图: 路径里 \\seg_input\\ 换成 \\seg_input_rc\\(新图在才换, 不在就留旧图并记数), 字表原样拷。标签一字不动。"""
    import shutil
    if a.out.exists():
        print(f"  ★ {a.out} 已经有了, 不覆盖", flush=True)
        sys.exit(2)
    a.out.mkdir(parents=True)
    tot = rc = 0
    for name in ("train.txt", "val.txt"):
        lines_out = []
        for ln in (a.src / name).read_text(encoding="utf-8").splitlines():
            if "\t" not in ln:
                continue
            p, txt = ln.split("\t", 1)
            q = p.replace("\\seg_input\\", "\\seg_input_rc\\").replace("/seg_input/", "/seg_input_rc/")
            tot += 1
            if q != p and Path(q).exists():
                rc += 1
                p = q
            lines_out.append(f"{p}\t{txt}")
        (a.out / name).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    shutil.copyfile(a.src / "charset.txt", a.out / "charset.txt")
    print(f"  {a.src.name} -> {a.out.name}: {tot:,} 段, 换成重切图的 {rc:,} ({rc / max(1, tot):.2%})", flush=True)
    print(f"RELIST_STATS total={tot} rc={rc}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="切图修正: 每段按自己的字定上下边")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--src", nargs="+", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--n", type=int, default=40)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--sheets", type=int, default=3)
    g = sub.add_parser("regold")
    g.add_argument("--gold", type=Path, nargs="+", required=True, help="答案目录(gold gold2): img + _gold_candidates.csv ...")
    g.add_argument("--out-root", type=Path, required=True, help="新答案目录放哪(出 <名字>_rc)")
    g.add_argument("--pairs-root", type=Path, default=Path(r"E:\alipay-ai-data"), help="src_path 里的目录找不到时到这里按目录名找")
    g.add_argument("--min-ext", type=float, default=0.0, help="新边比旧边多出不到这么多倍行高就不动(0 = 全按新切法)")
    g.add_argument("--suffix", default="_rc", help="新答案目录名 = 原名 + 这个")
    g.add_argument("--page-index", nargs="*", default=[r"E:\_move_logs\pagetype_pinyin_v4\page_type.csv", r"E:\_move_logs\pagetype_census_v4\page_type.csv"],
                   help="原图搬过家时按文件名找现在的路径(page_type.csv 的 path 列)")
    c = sub.add_parser("recut")
    c.add_argument("--pairs", type=Path, nargs="+", required=True, help="配对目录(里面有 _segments.csv / _pairs_labeled.csv / seg_input)")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--page-index", nargs="*", default=[r"E:\_move_logs\pagetype_pinyin_v4\page_type.csv", r"E:\_move_logs\pagetype_census_v4\page_type.csv"])
    rl = sub.add_parser("relist")
    rl.add_argument("--src", type=Path, required=True, help="原训练集目录(train.txt / val.txt / charset.txt)")
    rl.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "demo":
        demo(a)
    elif a.cmd == "regold":
        regold(a)
    elif a.cmd == "recut":
        recut(a)
    elif a.cmd == "relist":
        relist(a)


if __name__ == "__main__":
    main()
