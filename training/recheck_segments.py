r"""把每一段的标签裁图(seg_label)单独再读一遍, 和配给它的标签比 —— 找出"图和文字根本不是一对"的段。

为什么要这一步(2026-10-12)
--------------------------
逐条看蓝图验证集的错误, 40 条里 11 条是**裁图和标签对不上**:
    红色"免费领"按钮的裁图, 标签却是旁边那张券的"满3元可用"
    一个图标的裁图, 标签是"支付宝打车"
这种段哪个模型都学不会、也考不过; 而且它们很有规律, 模型会照着学 ——
实测新模型对"免费领"按钮的裁图真的输出了"满1.元可用"。

来历: 标签是**整行**读出来的, split_segments 再按空隙把整行文字切成几段、按顺序配给几何上的几块。
块数和段数一样就配上 —— 但按钮、图标也算一块的时候, 顺序会错开一位。
check_alignment 按"宽度/字数"挡掉了一部分, 宽度凑巧对得上的挡不住。

怎么查
------
每一段的 seg_label(拼音已经裁在框外)单独用同一个 RapidOCR **只做识别、不做检测**再读一遍(一段 0.04 秒),
看标签里的字有多少出现在重读结果里:
    重叠 = 标签和重读结果共有的字数(按个数算) / 标签字数      只比汉字、字母、数字, 标点和全半角不算
    重叠 < --thr(默认 0.34): 标签里三分之二以上的字图上都没有 -> 判成对不上
读错一两个字的不会被判中(重叠还很高), 只有"根本不是这几个字"的才会。

用法
----
    python training/recheck_segments.py --pairs 目录 [目录 ...]                  # 只出报告 + _recheck.csv
    python training/recheck_segments.py --pairs 目录 [目录 ...] --sheet 60       # 再出核对图(判中的)
    python training/recheck_segments.py --pairs 目录 [目录 ...] --drop           # 在 _segments.csv 里标 bad=1
★ 重读结果存在每个目录的 _recheck.csv, 再跑(比如加 --drop)直接用, 不重读。
★ --drop 改之前把 _segments.csv 备份成 .csv.bak4(只备份一次)。
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

THR = 0.34
_OCR = None


def norm(s: str) -> str:
    """全半角统一, 只留汉字、字母、数字和 ¥ —— 标点、箭头、括号不算。"""
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(c for c in s if c.isalnum() or c == "¥")


def overlap(label: str, reread: str) -> float | None:
    a, b = norm(label), norm(reread)
    if not a:
        return None                     # 标签只有标点, 不判
    ca, cb = Counter(a), Counter(b)
    return sum(min(n, cb[c]) for c, n in ca.items()) / len(a)


def _init() -> None:
    cv2.setNumThreads(1)


def _read(path: str) -> tuple[str, float]:
    """只识别不检测。出错就地兜住, 返回空。"""
    global _OCR
    try:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "", -1.0
        if _OCR is None:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        res, _ = _OCR(img, use_det=False, use_cls=False)
        if not res:
            return "", 0.0
        t, conf = res[0][0], float(res[0][1])
        return (t or "").strip(), conf
    except Exception:                    # noqa: BLE001
        return "", -1.0


def load_cache(p: Path) -> dict:
    if not p.exists():
        return {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def run_dir(d: Path, a, pool) -> tuple[list[dict], dict]:
    man = d / "_segments.csv"
    rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
    cache_p = d / "_recheck.csv"
    cache = load_cache(cache_p)
    todo = [r for r in rows if r["file"] not in cache]
    t0 = time.time()
    if todo:
        paths = [str(d / "seg_label" / r["file"]) for r in todo]
        for i, (r, (txt, conf)) in enumerate(zip(todo, pool.map(_read, paths, chunksize=64)), 1):
            cache[r["file"]] = {"file": r["file"], "reread": txt, "conf": f"{conf:.3f}"}
            if i % 20000 == 0:
                print(f"    {i:,}/{len(todo):,}   {i / max(1e-9, time.time() - t0):.0f} 段/秒", flush=True)
    out = []
    for r in rows:
        c = cache[r["file"]]
        ov = overlap(r.get("text", ""), c["reread"])
        flag = ov is not None and ov < a.thr and r.get("bad") != "1"
        out.append({"file": r["file"], "text": r.get("text", ""), "reread": c["reread"], "conf": c["conf"],
                    "overlap": "" if ov is None else f"{ov:.3f}", "flag": int(flag), "bad_before": r.get("bad", "0")})
    tmp = cache_p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, cache_p)
    return out, {"rows": rows, "secs": time.time() - t0, "read": len(todo)}


def sheet(d: Path, recs: list[dict], n: int, dst: Path) -> None:
    import random
    from make_check_sheet import _cjk_font
    font = _cjk_font(20)
    pick = [r for r in recs if r["flag"] == 1]
    random.Random(5).shuffle(pick)
    pick = pick[:n]
    if not pick:
        return
    ims, W = [], 0
    for r in pick:
        im = cv2.imdecode(np.fromfile(str(d / "seg_input" / r["file"]), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        if im.shape[0] > 70:
            im = cv2.resize(im, (int(im.shape[1] * 70 / im.shape[0]), 70))
        if im.shape[1] > 700:
            im = im[:, :700]
        im = cv2.copyMakeBorder(im, 3, 3, 6, 6, cv2.BORDER_CONSTANT, value=(215, 215, 215))
        ims.append((im, r))
        W = max(W, im.shape[1])
    TW = 760 if font else 0
    padded = [cv2.copyMakeBorder(m, 0, 0, 0, W - m.shape[1] + TW, cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for m, _ in ims]
    out = np.vstack(padded)
    if font:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        y = 0
        for i, (m, (_im, r)) in enumerate(zip(padded, ims), 1):
            dr.text((W + 6, y + 2), f"{i}. 标 {r['text'][:18]}", font=font, fill=(60, 60, 60))
            dr.text((W + 400, y + 2), f"读 {r['reread'][:16]}", font=font, fill=(190, 0, 0))
            y += m.shape[0]
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imencode(".png", out)[1].tofile(str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=THR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--sheet", type=int, default=0, help="每个目录出一张判中的核对图, 抽这么多条")
    ap.add_argument("--drop", action="store_true", help="判中的在 _segments.csv 里标 bad=1(先备份 .csv.bak4)")
    a = ap.parse_args()

    print("=" * 72)
    print("  RECHECK SEGMENTS  (ASCII only - safe to paste)")
    print("=" * 72)
    print(f"  判据: 标签里的字在重读结果里出现的比例 < {a.thr}  -> 图和标签对不上")
    summary = []
    with ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        for d in a.pairs:
            if not (d / "_segments.csv").exists():
                print(f"  {d}: 没有 _segments.csv, 先跑 split_segments")
                continue
            recs, info = run_dir(d, a, pool)
            live = [r for r in recs if r["bad_before"] != "1"]
            fl = [r for r in recs if r["flag"] == 1]
            print(f"  {d.name}: 段 {len(recs):,}  (这次重读 {info['read']:,}, {info['secs']:.0f} 秒)  "
                  f"判成对不上 {len(fl):,} = {len(fl) / max(1, len(live)):.2%} (按没标 bad 的算)")
            for t, c in Counter(r["text"] for r in fl).most_common(6):
                print(f"      {c:>5}x  标签 {t[:24]}")
            summary.append((d.name, len(live), len(fl)))
            if a.sheet:
                dst = d / "_recheck_flagged.png"
                sheet(d, recs, a.sheet, dst)
                print(f"    核对图 -> {dst}")
            if a.drop and fl:
                man = d / "_segments.csv"
                bak = man.with_suffix(".csv.bak4")
                if not bak.exists():
                    bak.write_bytes(man.read_bytes())
                flagged = {r["file"] for r in fl}
                rows = info["rows"]
                fields = list(rows[0].keys())
                if "bad" not in fields:
                    fields.append("bad")
                n = 0
                for r in rows:
                    if r["file"] in flagged:
                        r["bad"] = "1"
                        n += 1
                    elif not r.get("bad"):
                        r["bad"] = "0"
                with man.open("w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
                print(f"    ★ --drop: 标了 {n:,} 段 bad=1  (原件备份 {bak.name})")
    print("-" * 72)
    print(f"  {'目录':<32}{'段(没标bad)':>12}{'对不上':>9}{'占比':>8}")
    for name, n, k in summary:
        print(f"  {name:<32}{n:>12,}{k:>9,}{k / max(1, n):>8.2%}")
    print("=" * 72)


if __name__ == "__main__":
    main()
