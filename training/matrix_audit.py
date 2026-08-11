r"""一条命令把 localdet6 这一轮的产物从头到尾核一遍, 只出一份报告(只读)。

为什么要有这个
--------------
上一轮的核查被拆成了六七条命令分散在几段对话里, 结果是: 跑漏了一步(版式核对),
而那一步恰恰是本项目里抓到过最大错误的一步。**清单一旦要靠人记, 早晚会漏。**
所以固化成一个脚本: 该查的全在里面, 顺序固定, 缺什么大声报, 不截断。

查六件事:
  [1] 九格产物在不在        —— **应有清单写死在代码里**, 少一个就大声报, 不能靠 glob 有几个算几个
  [2] 每个 CSV 自证         —— 行数/源目录张数对账、定位率、切块数、聚合口径
  [3] 每个输入目录的身份     —— manifest 表头 + 文件名前缀反查造它的脚本 + **按像素判版式**
  [4] 改动区落在哪儿         —— 假图对源图求差, 看改动是不是真的在金额上
  [5] 训练测试是否同源       —— 裁块源图与测试集源图求交
  [6] 汇总判读表             —— 每一格能不能用, 不能用的写明原因

用法
----
  python training/matrix_audit.py
  python training/matrix_audit.py --skip-edit          # 跳过第4节(最慢的一节)
  python training/matrix_audit.py --probe D:\probe --src-root D:\download\TempFakeImages D:\download2\OtherImages

**只读**: 除了打印什么都不做。输出为一整份报告, 直接整段贴回来即可。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crop_src_audit import split_crop_name            # noqa: E402
from edit_region_audit import contain, diff_box, load_rgb  # noqa: E402
from engine_b_tamper import locate_amount             # noqa: E402
from locate_blue import is_blue_page, locate_amount_blue   # noqa: E402

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- 应有清单写死在这里。**少一格必须大声报** ----
# (教训: verify_workspace.py:44-45 —— 没有基准的检查等于没检查, 还会给出虚假的安心)
_CELLS: list[tuple[str, str]] = [
    ("万相 白", "wanwhite_ld6"),
    ("万相 蓝", "wanblue_ld6"),
    ("豆包 白", "seedwhite_ld6"),
    ("豆包 蓝", "seedblue_ld6"),
    ("千问 白", "qwenwhite_ld6"),
    ("千问 蓝", "qwenblue_ld6"),
    ("合成 白", "white_ld6"),
    ("合成 蓝", "blue_ld6"),
    ("真图池",  "genuine_20k_ld6"),
]

# 仓库已经记过的脏数据源(verify_workspace.py:37-38)
_DEPRECATED = {
    "api_local_seed": "训练测试同源, 已弃用 (verify_workspace.py:37)",
    "api_local_wan":  "训练测试同源, 已弃用 (verify_workspace.py:38)",
}

# 文件名前缀 -> 造它的脚本。**蓝图支持与否直接决定蓝图那几格的数字有没有意义。**
_GEN = [
    ("apilocal_", "gen_api_local_edit.py  (API局部改金额)", False),
    ("apifull_",  "gen_api_local_edit.py --save-full",      False),
    ("ailocal_",  "gen_local_ai_edit.py   (本地VAE局部改)", True),
    ("aivae_",    "gen_ai_fakes.py        (VAE整图)",       None),
    ("aii2i_",    "gen_ai_fakes.py        (img2img整图)",   None),
    ("api_",      "gen_api_fakes.py       (API整图重绘)",   None),
    ("crop_",     "localcrops 裁块",                        None),
]


def count_images(d: Path) -> int:
    n = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in _EXT:
                n += 1
    return n


def identify(d: Path) -> tuple[str, bool | None, list[str]]:
    """按文件名前缀反查造它的脚本。返回 (脚本, 是否支持蓝图定位, 样例文件名)。"""
    names: list[str] = []
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and os.path.splitext(e.name)[1].lower() in _EXT:
                    names.append(e.name)
                    if len(names) >= 400:
                        break
    except Exception:
        return "(读不了)", None, []
    hits = Counter()
    for nm in names:
        for pre, who, blue in _GEN:
            if nm.startswith(pre):
                hits[(who, blue)] += 1
                break
        else:
            hits[("(前缀不认识 -> 多半是真图池)", None)] += 1
    if not hits:
        return "(空目录)", None, []
    (who, blue), _ = hits.most_common(1)[0]
    return who, blue, names[:3]


def page_mix(d: Path, sample: int) -> tuple[int, int, int]:
    """按像素判版式(口径同 predict_tiled._is_blue_page)。返回 (蓝, 白, 读失败)。"""
    files: list[str] = []
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and os.path.splitext(e.name)[1].lower() in _EXT:
                    files.append(e.path)
                    if len(files) >= sample:
                        break
    except Exception:
        return 0, 0, 0
    blue = white = err = 0
    for p in files:
        a = load_rgb(Path(p))
        if a is None:
            err += 1
        elif is_blue_page(a):
            blue += 1
        else:
            white += 1
    return blue, white, err


def sec1_inventory(probe: Path) -> dict[str, Path | None]:
    print("=" * 104)
    print("[1] 九格产物清单 (应有清单写死在脚本里, 少一个就报)")
    print("=" * 104)
    found: dict[str, Path | None] = {}
    for label, name in _CELLS:
        sp = probe / name / "summary.csv"
        ok = sp.exists()
        found[name] = sp if ok else None
        print(f"  {'OK  ' if ok else '!!!!'} {label:8s} {name:20s} " +
              ("" if ok else "**没有产出 —— 这一格是空的, 不是待读数**"))
    miss = [n for n, v in found.items() if v is None]
    print(f"\n  应有 {len(_CELLS)} 格, 实有 {len(_CELLS) - len(miss)} 格" +
          (f", **缺 {len(miss)}: {', '.join(miss)}**" if miss else ", 全齐"))
    return found


def sec2_csv(found: dict[str, Path | None]) -> dict[str, Path]:
    print()
    print("=" * 104)
    print("[2] 每个 CSV 自证 (行数对账 / 定位率 / 切块数 / 聚合口径)")
    print("=" * 104)
    inputs: dict[str, Path] = {}
    for _label, name in _CELLS:
        sp = found.get(name)
        if sp is None:
            continue
        rows = list(csv.DictReader(open(sp, encoding="utf-8-sig")))
        if not rows:
            print(f"  !!!! {name:20s} 空文件")
            continue
        loc = sum(1 for r in rows if (r.get("roi_amount_located") or "").strip() == "1")
        ne = sum(1 for r in rows if (r.get("final_ai_score") or "") != (r.get("tile_max") or ""))
        nt = Counter((r.get("n_tiles") or "?") for r in rows)
        src = sorted({str(Path(r.get("image") or "").parent) for r in rows})
        total = sum(count_images(Path(s)) for s in src if Path(s).is_dir())
        for s in src:
            inputs[name] = Path(s)
        nts = " ".join(f"{k}块x{v}" for k, v in sorted(nt.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0))
        flag = "OK  " if (total == len(rows) and ne == 0) else "!!!!"
        print(f"  {flag} {name:20s} CSV={len(rows):6d} 源={total:6d} 差={total-len(rows):+5d} "
              f"定位={loc*100.0/len(rows):5.1f}% [{nts}] final≠max={ne}")
        print(f"       输入: {' ; '.join(src)}")
        if len(src) > 1:
            print("       !!!! 这个 CSV 混了多个输入目录")
    print()
    print("  判读: 差必须=0(否则打分失败的图不写行, 分母偷偷变小, 召回虚高, predict_tiled.py:228-229);")
    print("        final≠max 必须=0(否则混了不同 --agg 的批次);")
    print("        切块数只该有 4(定位成功 2x2) 和 18(退回 3x6) 两种。")
    return inputs


def sec3_identity(inputs: dict[str, Path], sample: int) -> dict[str, tuple[str, bool | None, str]]:
    print()
    print("=" * 104)
    print("[3] 每个输入目录的身份 (manifest 表头 / 造它的脚本 / **按像素判版式**)")
    print("=" * 104)
    out: dict[str, tuple[str, bool | None, str]] = {}
    seen: set[str] = set()
    for _label, name in _CELLS:
        d = inputs.get(name)
        if d is None or str(d) in seen:
            continue
        seen.add(str(d))
        who, blue_ok, samples = identify(d)
        mp = d / "manifest.csv"
        hdr = "(没有 manifest)"
        if mp.exists():
            try:
                with open(mp, encoding="utf-8-sig") as f:
                    hdr = (f.readline() or "").strip()
            except Exception as exc:  # noqa: BLE001
                hdr = f"(读不了: {exc})"
        b, w, err = page_mix(d, sample)
        ok = b + w
        if ok == 0:
            kind = "(判不了)"
        elif b > ok * 0.8:
            kind = "**蓝图为主**"
        elif w > ok * 0.8:
            kind = "**白图为主**"
        else:
            kind = "**混合!!**"
        out[name] = (who, blue_ok, kind)
        print(f"  {d}")
        print(f"      造它的脚本 : {who}" +
              ("" if blue_ok is None else ("   [有蓝图定位器]" if blue_ok else "   [**无蓝图定位器**]")))
        print(f"      manifest   : {hdr}")
        print(f"      样例文件名 : {' | '.join(samples) if samples else '(无)'}")
        print(f"      版式(抽{ok:4d}) : 蓝 {b*100//max(1,ok):3d}%  白 {w*100//max(1,ok):3d}%  {kind}" +
              (f"  (读失败 {err})" if err else ""))
        dep = _DEPRECATED.get(d.name)
        if dep:
            print(f"      !!!! {dep}")
    print()
    print("  判读: 目录名不是证据, 像素才是。名字带 blue 却判成白图为主(或反过来) = 这一格测的是跨版式。")
    print("        造样本的脚本若**无蓝图定位器**, 而这批又确实是蓝图, 那改动多半没做在金额上 -> 看第[4]节。")
    return out


def build_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            print(f"  源图池不存在, 跳过: {root}")
            continue
        n0 = len(index)
        for cur, _dirs, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in _EXT:
                    index.setdefault(f, Path(cur) / f)
        print(f"  索引 {root}: +{len(index)-n0} 张")
    return index


def sec4_edit(inputs: dict[str, Path], roots: list[Path], limit: int,
              full_thresh: float, hit: float) -> dict[str, str]:
    print()
    print("=" * 104)
    print("[4] 改动区落在哪儿 (假图对源图求差 -> 改动区外接框 -> 和两个金额定位器的框比)")
    print("=" * 104)
    verdict: dict[str, str] = {}
    targets = [(n, d) for _l, n in _CELLS for d in [inputs.get(n)]
               if d is not None and n != "genuine_20k_ld6" and (d / "manifest.csv").exists()]
    if not targets:
        print("  没有带 manifest 的假图集, 跳过。")
        return verdict
    index = build_index(roots)
    if not index:
        print("  !!!! 源图池索引为空, 这一节做不了。用 --src-root 指对源图池。")
        return verdict
    for name, d in targets:
        rows = list(csv.DictReader(open(d / "manifest.csv", encoding="utf-8-sig")))
        tally: Counter[str] = Counter()
        n = miss = 0
        for r in rows:
            if n >= limit:
                break
            fn = (r.get("file") or "").strip()
            sn = Path((r.get("src") or "").strip()).name
            if not fn or not sn:
                continue
            fp, sp = d / fn, index.get(sn)
            if not fp.exists() or sp is None:
                miss += 1
                continue
            fake, src = load_rgb(fp), load_rgb(sp)
            if fake is None or src is None:
                miss += 1
                continue
            n += 1
            blue = is_blue_page(src)
            box, ratio = diff_box(src, fake)
            if box is None:
                tally["没测到改动"] += 1
                continue
            if ratio > full_thresh:
                tally["整图重绘"] += 1
                continue
            lw, lb = locate_amount(src), locate_amount_blue(src)
            cw = contain(box, (lw[0], lw[1], lw[2], lw[3])) if lw else 0.0
            cb = contain(box, (lb[0], lb[1], lb[2], lb[3])) if lb else 0.0
            right, wrong = (cb, cw) if blue else (cw, cb)
            if right >= hit:
                tally["改在金额区(对)"] += 1
            elif wrong >= hit:
                tally["**错区域(另一个定位器的框)**"] += 1
            else:
                tally["**改在别处(红包卡/广告位)**"] += 1
        good = tally.get("改在金额区(对)", 0)
        share = good * 100.0 / max(1, n)
        verdict[name] = (f"改在金额区 {share:.0f}%" if n else "无法比对")
        print(f"  {name:20s} 比对 {n:4d} 张 (源图找不到 {miss})  -> " +
              "  ".join(f"{k} {v}" for k, v in tally.most_common()))
    print()
    print("  判读: '改在金额区' 占绝大多数 -> 这批是合格的改金额样本, 召回数有意义。")
    print("        '错区域'/'改在别处'     -> 造样本时定位错了, 这批测的不是改金额检测, 数不能用,")
    print("                                   而且同一条路径 --save-crops 存下的训练裁块也是错区域的。")
    print("        '整图重绘' 占多数       -> 这是线路A 的样本, 不该拿来测线路B。")
    return verdict


def sec5_leak(inputs: dict[str, Path], crops: Path, probe: Path) -> dict[str, str]:
    print()
    print("=" * 104)
    print("[5] 训练测试是否同源 (裁块源图 vs 测试集源图, 按源图文件名求交)")
    print("=" * 104)
    verdict: dict[str, str] = {}
    if not crops.is_dir():
        print(f"  !!!! 裁块目录不存在: {crops} —— 这一节做不了")
        return verdict
    owner: dict[str, str] = {}
    per_tag: Counter[str] = Counter()
    n_crop = n_bad = 0
    with os.scandir(crops) as it:
        for e in it:
            if not e.is_file() or os.path.splitext(e.name)[1].lower() not in _EXT:
                continue
            n_crop += 1
            parsed = split_crop_name(os.path.splitext(e.name)[0])
            if not parsed:
                n_bad += 1
                continue
            tag, idx = parsed
            owner[f"apilocal_{tag}_{idx}.jpg"] = tag
            per_tag[tag] += 1
    print(f"  裁块 {n_crop} 个 | 文件名可解析 {len(owner)} 个 | 解析不了 {n_bad} 个(VAE 那几组命名不同, 正常)")

    allsrc: set[str] = set()
    dup: dict[str, set[str]] = {}
    for mp in sorted(probe.glob("*/manifest.csv")):
        try:
            rows = list(csv.DictReader(open(mp, encoding="utf-8-sig")))
        except Exception:
            continue
        for r in rows:
            f = (r.get("file") or "").strip()
            s = Path((r.get("src") or "").strip()).name
            if not f or not s:
                continue
            dup.setdefault(f, set()).add(s)
            if f in owner:
                allsrc.add(s)
    print(f"  训练侧可追溯到的源图 {len(allsrc)} 张(去重)")
    collided = {f: v for f, v in dup.items() if len(v) > 1}
    if collided:
        print(f"  !!!! {len(collided)} 个生成图名对应多个源图 —— 同 tag 跑过两次, 裁块被成对覆盖")
        print("       (made 计数器每次从 0 重数; ai 和 nature 仍一样多, verify_workspace 那条断言照样报 OK)")

    print()
    for _label, name in _CELLS:
        d = inputs.get(name)
        if d is None or name == "genuine_20k_ld6":
            continue
        mp = d / "manifest.csv"
        if not mp.exists():
            verdict[name] = "无 manifest, 查不了"
            print(f"  ??   {name:20s} 没有 manifest.csv -> **不等于干净, 是查不了**")
            continue
        rows = list(csv.DictReader(open(mp, encoding="utf-8-sig")))
        src = [Path((r.get("src") or "").strip()).name for r in rows if (r.get("src") or "").strip()]
        ov = sorted({s for s in src if s in allsrc})
        verdict[name] = "干净" if not ov else f"**重叠 {len(ov)}**"
        print(f"  {'OK  ' if not ov else '!!!!'} {name:20s} 行 {len(rows):5d} 有src {len(src):5d} 重叠 {len(ov):5d}"
              + ("" if not ov else "  **这个召回数是记忆效应, 不能报**"))
        for s in ov[:10]:
            print(f"         {s}")
    print()
    print("  判读: 重叠必须 0。>0 = 测的是记忆效应。'无 manifest' 只是查不了, 报数时要标注'未验证同源'。")
    return verdict


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="localdet6 产物一次性核查(只读)")
    ap.add_argument("--probe", type=Path, default=Path(r"D:\probe"))
    ap.add_argument("--crops", type=Path, default=Path(r"D:\localcrops\ai"))
    ap.add_argument("--src-root", type=Path, nargs="+",
                    default=[Path(r"D:\download\TempFakeImages"), Path(r"D:\download2\OtherImages")])
    ap.add_argument("--sample", type=int, default=200, help="判版式抽多少张")
    ap.add_argument("--limit", type=int, default=40, help="第4节每个集合比对多少张")
    ap.add_argument("--full-thresh", type=float, default=0.5)
    ap.add_argument("--hit", type=float, default=0.5)
    ap.add_argument("--skip-edit", action="store_true", help="跳过第4节(最慢)")
    args = ap.parse_args()

    found = sec1_inventory(args.probe)
    inputs = sec2_csv(found)
    ident = sec3_identity(inputs, args.sample)
    edit = {} if args.skip_edit else sec4_edit(inputs, list(args.src_root), args.limit,
                                               args.full_thresh, args.hit)
    leak = sec5_leak(inputs, args.crops, args.probe)

    print()
    print("=" * 104)
    print("[6] 汇总: 每一格到底能不能用")
    print("=" * 104)
    print(f"  {'格':10s} {'产出':6s} {'版式':12s} {'造它的脚本':34s} {'改动落点':16s} {'同源':10s}")
    print("  " + "-" * 100)
    for label, name in _CELLS:
        if found.get(name) is None:
            print(f"  {label:10s} {'缺':6s} {'-':12s} {'-':34s} {'-':16s} {'-':10s}  **空格**")
            continue
        who, blue_ok, kind = ident.get(name, ("?", None, "?"))
        who_s = who.split("(")[0].strip() + ("" if blue_ok is not False else " [无蓝图定位器]")
        who_s = who_s[:34]
        d = inputs.get(name)
        dep = _DEPRECATED.get(d.name) if d else None
        note = "  **" + dep + "**" if dep else ""
        print(f"  {label:10s} {'有':6s} {kind:12s} {who_s:34s} "
              f"{edit.get(name, '(未跑)'):16s} {leak.get(name, '(未跑)'):10s}{note}")
    print()
    print("  一格要能进报告, 必须同时满足: 产出在 + 版式与名字一致 + 改动确实落在金额区 + 同源重叠为 0。")
    print("  任何一项不满足, 这一格的召回数就不能对外报, 要写明是哪一项卡住了。")


if __name__ == "__main__":
    main()
