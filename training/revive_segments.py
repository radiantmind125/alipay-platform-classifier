r"""step 45: 被当成坏段扔掉的券面金额 / 按钮段, 用 v3 切法切全、核过再收回来; 训练里明显是切坏了读出来的券面金额标签, 核过改掉或拿掉。

为什么(2026-10-24)
------------------
服务器上清标签时(recheck_segments), 券面大数字的段: 读法对的(0.5元 / 20元)931 段几乎全被标了 bad ——
图切掉了上半截, 模型读不出, 和老师对不上就被扔了; 训练里只剩 7 段。读法明显是切坏了的(U元 / 元 / U.5元 ...)
倒还有 64 段留在训练里(错标签)。按钮(免费领 / 回首页 ...)也有约 600 段被扔。
结果模型从没见过完整的券面大数字, 金标准上这一类几乎全错(v3 切法把图切全了, 但光有图没有对的标签学不会)。

做法(每个配对目录)
------------------
    候选: (a) bad=1, 文字是券面金额(N元 / ¥N元)                      -> 重读像样就收回, 标签用重读的
          (b) bad=1, 文字是切坏了读的金额(U元 / 元 / 0v元 / U.5元 ...)  -> 重读像样就收回, 标签用重读的
          (c) 还在训练里, 文字是切坏了读的金额                        -> 重读和原标签一样: 不动; 重读像样: 改标签; 否则拿掉
    ★ 按钮不收: 训练里已经有几千段对的(免费领 3,573 / 回首页 1,496 / 一键领 1,023), 按钮的问题是图切掉一截(v3 切全了);
      被扔的按钮又核不了(擦拼音把白字擦坏, v3 认实心按钮的门槛本机 30 段只认出 0 段) —— 收回来不值这个险(10-24)
    每个候选按 v3 切法从原图切(和 recut 一样先逐像素核对旧切法 == seg_input), 存 <目录>\seg_input_rc3\
    重读: 切好的图擦掉拼音(erase_pinyin.erase, force), RapidOCR 只识别不检测(和 recheck_segments 同一套设置)
    ★ 宁可少收, 不收错:
      券面金额的标签一律用 v3 整图的重读 —— 10-24 本机核对: 切掉上半截读的旧标签常错成像样的数(3 -> 5、30 -> 50),
      整图重读数字是对的, 只是 元 常读成 完(擦拼音的残影), 这个改回来; 重读要是 N元 格式、置信度 >= 0.85
      v3 切出来比行高 3 倍还高的不用(多半把下一行的按钮带进来了)
    出 <目录>\_revive.csv 和 REVIVE_STATS, 全部完了 REVIVE_OK

    relist: 从 v2 训练集清单出 v3 训练集 —— 路径换成 seg_input_rc3、按 _revive.csv 改标签 / 拿掉 / 加回来的段
           (加回来的按原图分: 原图在验证集里的进验证集, 其他进训练; 金标准的段一律不要)

用法
----
    python training/revive_segments.py scan   --pairs 目录 [...] --workers 20
    python training/revive_segments.py relist --src E:\alipay-ai-data\train-n1009i --out E:\alipay-ai-data\train-n1009k --pairs 目录 [...] --gold 金标准目录 [...]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recrop  # noqa: E402

RC = "seg_input_rc3"
CONF = 0.85
CONF_CUT = 0.90        # v3 切出来的图最上面一行还有字(还是被切了一截): 置信度要更高(10-24 复核: 9.07 读成 7.07 是 0.842)
STILL_CUT_INK = 10     # 最上面一行和底色差得多的像素 >= 这么多 = 还被切着
TALL_GUARD = 3.0       # v3 切出来的图比行高这么多倍还高: 多半把下一行(比如下面的按钮)带进来了, 不用
BOTTOM_GROW = 0.5      # 往下扩了超过半个行高也不用(10-24 复核: 行高大的券卡上, 金额 + 下面整个 免费领 按钮还不到 3 倍)
COUPON = re.compile(r"^¥?\d+(\.\d+)?元$")
BAD_AMOUNT = re.compile(r"^¥?(0\d|0(\.0+)?元$)")      # 0元 / 0.0元 / 05元 / 010元: 券上没有这种, 是切坏了读出来的
CUTLIKE = re.compile(r"^[A-Za-z一六'/.\d]*元$")       # 切坏了读的金额: U元 V元 L元 J元 ZU元 一元 六元 ... (10-24 复核在服务器标签里数过)
YUAN_LIKE = "完兀π冗"   # 擦拼音后 元 上面常剩一点拼音的残影, RapidOCR 读成这几个(10-24 本机核对图)
SEG_RE = re.compile(r"^(.*)_r\d+_s\d+\.png$", re.I)


def nk(s: str) -> str:
    """比较用: 全半角统一(NFKC), 去空白。"""
    return "".join(unicodedata.normalize("NFKC", s or "").split())


def coupon_read(read: str, conf: float, still_cut: bool) -> str:
    """重读结果当券面金额用: 结尾的 完 / 兀 / π 当 元; 要是 N元 格式、不是 0元 / 05元 这种、置信度够(还被切着要更高)。不行返回空。"""
    if conf < (CONF_CUT if still_cut else CONF):
        return ""
    t = nk(read)
    if t and t[-1] in YUAN_LIKE:
        t = t[:-1] + "元"
    return t if COUPON.match(t) and not BAD_AMOUNT.match(t) else ""


def kind_of(text: str, bad: bool):
    """这一段是不是候选、是哪一种: revive_coupon / relabel_bad / relabel_active / None。
    还在训练里的: 切坏了读的金额, 和已经是 N元 样子的(10-24 复核: 老师在切掉一截的图上读的 N元 约四成是错的数)。"""
    t = nk(text)
    if not t:
        return None
    coupon = bool(COUPON.match(t))
    cut = bool(CUTLIKE.match(t)) and not coupon
    if bad:
        if coupon:
            return "revive_coupon"
        if cut:
            return "relabel_bad"
        return None
    return "relabel_active" if (cut or coupon) else None


def decide(kind: str, label: str, read: str, conf: float, status: str, is_button: bool, tall: bool, still_cut: bool = False) -> tuple:
    """(动作, 新标签)。动作: revive / relabel / keep / drop / skip。宁可少收, 不收错。
    ★ 券面金额: 标签一律用 v3 整图的重读(10-24 本机核对: 切掉上半截读的旧标签常错成像样的数, 3 -> 5、30 -> 50;
      整图重读数字是对的, 只是 元 常读成 完)。重读不像样或置信度不够 -> 不收 / 拿掉。
    ★ 还在训练里的: 重读和原标签一字不差 -> 不动(比如真的就是 一元); 重读是像样的金额 -> 改; 否则拿掉(宁可少一段, 不留错标签)。
    is_button 只记下来看(按钮不收, 见文件头)。"""
    active = kind == "relabel_active"
    if status != "ok" or tall:
        return ("drop", "") if active else ("skip", "")
    cr = coupon_read(read, conf, still_cut)
    if active:
        if nk(read) == nk(label) or (cr and cr == nk(label)):
            return "keep", label
        return ("relabel", cr) if cr else ("drop", "")
    return ("revive", cr) if cr else ("skip", "")


def code_version() -> str:
    """收回规则的指纹: 这个文件的源码 + v3 切法指纹。train-n1009k 里记一份, 服务器脚本跳过时核对。"""
    import hashlib
    src = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n") + recrop.recut_version()
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]


_INDEX, _OCR = {}, None


def _init(index: dict) -> None:
    global _INDEX
    cv2.setNumThreads(1)
    _INDEX = index


def _read(img: np.ndarray) -> tuple:
    """擦掉拼音, RapidOCR 只识别不检测 -> (读出, 置信度, 出错信息)。出错兜住: 置信度 -1, 并带回错误文字(scan 里计数并打印, 闸门看)。"""
    global _OCR
    try:
        from erase_pinyin import erase
        im, _info = erase(img, force=True)
        if _OCR is None:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        res, _ = _OCR(im, use_det=False, use_cls=False)
        if not res:
            return "", 0.0, ""
        return (res[0][0] or "").strip(), float(res[0][1]), ""
    except Exception as e:  # noqa: BLE001
        return "", -1.0, f"{type(e).__name__}: {e}"[:200]


def _still_cut(crop: np.ndarray) -> bool:
    """最上面一行有没有字(和整张图的底色差 > 40 的像素够多): 有 = 字的顶还被切着。"""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return int((np.abs(g[0] - int(np.median(g))) > 40).sum()) >= STILL_CUT_INK


def _scan_page(job) -> list:
    """一张原图: 解码一次、算一次字块; 每个候选段按 v3 切法切、核对旧切法、存图、擦拼音重读。"""
    src, items = job
    out = []
    path, _how = recrop.resolve_path(src, _INDEX)
    img = None
    if path:
        try:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            img = None
    L = recrop.layout(img) if img is not None else None
    for it in items:
        d = Path(it["dir"])
        if img is None:
            out.append((it, "", 0.0, "page_missing", False, False, False, ""))
            continue
        la, lb, ia, ib, x0, x1 = (int(it[k]) for k in ("la", "lb", "ia", "ib", "x0", "x1"))
        sp = d / "seg_input" / it["file"]
        stored = cv2.imdecode(np.fromfile(str(sp), np.uint8), cv2.IMREAD_COLOR) if sp.exists() else None
        old = img[ia:ib, x0:x1]
        if stored is None or stored.shape != old.shape or not np.array_equal(stored, old):
            out.append((it, "", 0.0, "old_mismatch", False, False, False, ""))
            continue
        L["_btn_ext"] = None                          # new_box 里 seg_extent 会写; 先清掉, 免得用上一段剩下的
        ia2, ib2 = recrop.new_box(L, la, lb, ia, ib, x0, x1)
        is_button = L.get("_btn_ext") is not None     # v3 用了实心按钮的洞 = 这段图确实是按钮
        row_h = max(1, lb - la)
        tall = (ib2 - ia2) > TALL_GUARD * row_h or (ib2 - ib) > BOTTOM_GROW * row_h
        crop = img[ia2:ib2, x0:x1]
        dst = d / RC / it["file"]
        if not recrop._done(dst):
            tmp = dst.with_name(dst.name + ".tmp")
            cv2.imencode(".png", crop)[1].tofile(str(tmp))
            os.replace(tmp, dst)
        txt, conf, err = _read(crop)
        out.append((it, txt, conf, "ok", is_button, tall, _still_cut(crop), err))
    return out


def scan(a) -> None:
    from concurrent.futures import ProcessPoolExecutor
    ver = recrop.recut_version()
    print(f"  切法指纹 {ver}", flush=True)
    for d in a.pairs:                                 # 和 recut 共用 seg_input_rc3: 指纹不对整个停
        rc = d / RC
        vf = rc / "_recut_version.txt"
        if rc.exists() and any(q.suffix.lower() == ".png" for q in rc.iterdir()):
            have = vf.read_text(encoding="utf-8").strip() if vf.exists() else "none"
            if have != ver:
                print(f"REVIVE_VERSION_MISMATCH {d.name} have={have} code={ver}", flush=True)
                sys.exit(3)
    index = recrop.load_index(a.page_index)
    for d in a.pairs:
        with (d / "_pairs_labeled.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
        jobs, kinds = {}, Counter()
        with (d / "_segments.csv").open(encoding="utf-8-sig", newline="") as f:
            for s in csv.DictReader(f):
                k = kind_of(s.get("text", ""), s.get("bad") == "1")
                if k is None:
                    continue
                r = rows.get(s["row_file"])
                if r is None:
                    kinds["row_missing"] += 1
                    continue
                kinds[k] += 1
                jobs.setdefault(s["source"], []).append({
                    "dir": str(d), "file": s["file"], "text": s["text"], "kind": k, "x0": s["x0"], "x1": s["x1"],
                    "la": r["label_y0"], "lb": r["label_y1"], "ia": r["input_y0"], "ib": r["input_y1"]})
        rc = d / RC
        rc.mkdir(exist_ok=True)
        (rc / "_recut_version.txt").write_text(ver + "\n", encoding="utf-8")
        for t in rc.glob("*.tmp"):
            t.unlink()
        n = sum(len(v) for v in jobs.values())
        print(f"  {d.name}: 候选 {n:,} 段  {dict(kinds)}", flush=True)
        cnt, recs, first_err = Counter(), [], ""
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(index,)) as ex:
            for res in ex.map(_scan_page, list(jobs.items()), chunksize=2):
                for it, txt, conf, status, is_button, tall, still_cut, err in res:
                    act, new = decide(it["kind"], it["text"], txt, conf, status, is_button, tall, still_cut)
                    cnt[act] += 1
                    cnt[status] += status != "ok"
                    cnt["too_tall"] += tall
                    cnt["read_error"] += conf < 0
                    first_err = first_err or err
                    recs.append({"file": it["file"], "kind": it["kind"], "old_text": it["text"], "read": txt, "conf": f"{conf:.3f}",
                                 "status": status, "is_button": int(is_button), "tall": int(tall), "still_cut": int(still_cut),
                                 "action": act, "new_text": new})
        tmp = d / "_revive.csv.tmp"
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "kind", "old_text", "read", "conf", "status", "is_button", "tall", "still_cut", "action", "new_text"])
            w.writeheader()
            w.writerows(recs)
        os.replace(tmp, d / "_revive.csv")
        if first_err:
            print(f"REVIVE_READ_ERROR {d.name} {first_err}", flush=True)
        print(f"REVIVE_STATS {d.name} candidates={n} revive={cnt['revive']} relabel={cnt['relabel']} keep={cnt['keep']} "
              f"drop={cnt['drop']} skip={cnt['skip']} too_tall={cnt['too_tall']} page_missing={cnt['page_missing']} "
              f"old_mismatch={cnt['old_mismatch']} row_missing={kinds['row_missing']} read_error={cnt['read_error']}", flush=True)
    print("REVIVE_OK", flush=True)


def _gold_keys(golds: list) -> set:
    keys = set()
    for g in golds:
        with (g / "_gold_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                sp = Path(r["src_path"])
                keys.add((sp.parent.parent.name.lower(), sp.name.lower()))
    return keys


def _page(fname: str) -> str:
    m = SEG_RE.match(fname)
    return (m.group(1) if m else fname).lower()


def relist(a) -> None:
    """v2 训练集清单 -> v3: 路径换 seg_input_rc3(切好了才换), 按 _revive.csv 改标签 / 拿掉, 加回收回来的段。
    换成 v3 图的不到 --min-rc 就什么都不写(RELISTK_LOW)。train.txt 最后写。"""
    import shutil
    if (a.out / "train.txt").exists():
        print(f"  ★ {a.out} 已经有了, 不覆盖", flush=True)
        sys.exit(2)
    chars = set(c for c in (a.src / "charset.txt").read_text(encoding="utf-8").split("\n") if c)
    gk = _gold_keys(a.gold)
    acts = {}
    for d in a.pairs:
        with (d / "_revive.csv").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                acts[(d.name.lower(), r["file"].lower())] = (r["action"], r["new_text"], d, r["file"])
    cnt = Counter()
    lists, val_pages = {}, set()
    for name in ("train", "val"):
        out = []
        for ln in (a.src / f"{name}.txt").read_text(encoding="utf-8").splitlines():
            if "\t" not in ln:
                continue
            p, txt = ln.split("\t", 1)
            sp = Path(p)
            key = (sp.parent.parent.name.lower(), sp.name.lower())
            if name == "val":
                val_pages.add((key[0], _page(sp.name)))
            cnt["total"] += 1
            act = acts.get(key)
            if act and act[0] == "drop":
                cnt["drop"] += 1
                continue
            if act and act[0] == "relabel":
                if all(c in chars for c in act[1]):
                    txt = act[1]
                    cnt["relabel"] += 1
                else:
                    cnt["drop"] += 1             # 新标签里有字表外的字: 不冒险, 拿掉
                    cnt["charset_skip"] += 1
                    continue
            q = p.replace("\\seg_input\\", f"\\{RC}\\").replace("/seg_input/", f"/{RC}/")     # 和 recrop.relist 一样换
            if q != p and recrop._done(Path(q)):
                cnt["rc"] += 1
                p = q
            out.append(f"{p}\t{txt}")
        lists[name] = out
    for key, (act, new, d, fname) in sorted(acts.items()):
        if act != "revive":
            continue
        if key in gk:
            cnt["gold"] += 1
            continue
        if not all(c in chars for c in new):
            cnt["charset_skip"] += 1
            continue
        q = d / RC / fname
        if not recrop._done(q):
            cnt["revive_noimg"] += 1
            continue
        which = "val" if (key[0], _page(fname)) in val_pages else "train"
        lists[which].append(f"{q}\t{new}")
        cnt[f"revive_{which}"] += 1
    kept = cnt["total"] - cnt["drop"]
    cov = cnt["rc"] / max(1, kept)
    print(f"RELISTK_STATS total={cnt['total']} rc={cnt['rc']} relabel={cnt['relabel']} drop={cnt['drop']} "
          f"revive_train={cnt['revive_train']} revive_val={cnt['revive_val']} gold={cnt['gold']} "
          f"charset_skip={cnt['charset_skip']} revive_noimg={cnt['revive_noimg']}", flush=True)
    if cov < a.min_rc:
        print(f"RELISTK_LOW cov={cov:.4f} min={a.min_rc} - nothing written", flush=True)
        sys.exit(4)
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.src / "charset.txt", a.out / "charset.txt")
    (a.out / "_revive_version.txt").write_text(code_version() + "\n", encoding="utf-8")   # 跳过这一步时服务器脚本核对
    (a.out / "val.txt").write_text("\n".join(lists["val"]) + "\n", encoding="utf-8")
    tmp = a.out / "train.txt.tmp"
    tmp.write_text("\n".join(lists["train"]) + "\n", encoding="utf-8")
    os.replace(tmp, a.out / "train.txt")
    print("RELISTK_OK", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="券面金额 / 按钮段: v3 切全、核过收回; 切坏了读的金额标签改掉或拿掉")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--pairs", type=Path, nargs="+", required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--page-index", nargs="*", default=[r"E:\_move_logs\pagetype_pinyin_v4\page_type.csv", r"E:\_move_logs\pagetype_census_v4\page_type.csv"])
    r = sub.add_parser("relist")
    r.add_argument("--src", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--pairs", type=Path, nargs="+", required=True)
    r.add_argument("--gold", type=Path, nargs="+", required=True)
    r.add_argument("--min-rc", type=float, default=0.97)
    sub.add_parser("version", help="打印收回规则的指纹(REVIVE_CODE xxx), 给服务器脚本核对")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a)
    elif a.cmd == "relist":
        relist(a)
    else:
        print(f"REVIVE_CODE {code_version()}", flush=True)


if __name__ == "__main__":
    main()
