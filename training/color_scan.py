r"""账单详情页正文字的颜色深度检查(经理说的 118) —— Python 参考实现 + 批量扫描。

现象
----
经理: "这个颜色的……直接分析深度就能搞定"。
支付宝的正文字(取值、账单管理那几行)是 #333333(灰度 51), 标签是 #999999(153), 页面底色 #F5F5F5(245)。
经理给的 118 假图(和 111/112/116 的假图是同一个造假工具, 1080x2400 苹果样式)正文字是**纯黑 #000000**,
标签 150, 底色 244 —— 颜色都差一点, 最明显的是正文字纯黑。
真图 JPEG 压过之后正文字的核心颜色还是稳稳的 50~51(118 真图就是 JPEG)。

注意: 苹果的状态栏和导航栏标题本来就是纯黑, 所以只看导航栏下面的正文。

量法
----
1. 导航栏下面(高度 9.5% 以下、96% 以上不要, 底部是苹果的横条), 不彩色(三通道最大减最小 < 30)的像素里,
   灰度 < 120 算深色字, 120~185 算灰色标签字, 按行切开。深色模式(中位灰度 < 128)不判。
2. 只要"左边灰标签 + 右边深色取值"的行: 标签在页面宽度 27% 以左、取值从 22% 往右, 两边都要有 >= 10 个像素,
   标签的核心颜色在 140~165。每行的"核心颜色" = 这一段像素里出现最多的灰度(笔画中间那种, 不受抗锯齿影响)。
3. 这些行要上下对齐(账单详情上面那张卡片): 至少 3 行, 标签左缘中位数在宽度 3%~10%, 取值左缘中位数在
   20%~34%, 每行和中位数差 <= 0.6% 宽度。对不齐的(收银台、支付结果页、银行短信、别的 App)不判,
   它们本来就用纯黑字。
4. 只判支付宝的页面(服务器 09-01~16 实测, 不加这两条会把 200 多张微信账单当成假图):
   卡片外面页面边上(宽度 0.4%~1.6%, 对齐那几行的高度)的颜色要是支付宝的灰底 238~250(#F5F5F5 = 245,
   造假工具 244/246); 微信账单、很多别的 App、支付宝 8 月以前的老版是整页白(255), 不判。
   标签核心色要 >= 148(支付宝 #999 = 153, 造假工具 150; 缩小压缩过的会变浅到 160 多); 抖音通知(147)这类页面在下限外。
5. 按行数算(每行一票): 取值核心 <= 12(纯黑)的行占 >= 60% 判可疑; 38~64(#333)的行占 >= 60% 判 Ok;
   其他判不了。纯黑只认 0 附近: 两种造假工具都是正好 0, 微信、支付宝话费充值页、银行短信是 21~25(黑色 90%)。
6. 要判可疑之前先看是不是拼音页(用现成的 pinyin_probe, C# 是 PinyinCheck), 拼音页不判:
   拼音字体有细体的变种, 字会压得很深, 拼音库里有几张支付结果页本来就是纯黑(本地实测 4 张误判, 都是拼音页)。

服务器验证(D:/download2/OtherImages 每 8 张抽 1 张, 101,285 张, 09-01~16)
--------
判可疑 66 张(0.07%), 逐张看过:
    34 张 111 或 112 也判了假(经理那个造假工具: 标签 150、底色 244、取值 0);
    32 张只有这一项判出来, 都是支付宝账单详情, 订单号是乱的(不是 2026MMDD2000400111 开头)、
    两个人两笔转账同一个订单号, 或者底色是造假工具的 244/246; 9 月 13~15 日有一批(底色 246, 标签 153)。
没加第 4 条和纯黑只认 0 附近之前判了 276 张, 多出来的 210 张是微信账单、话费充值页、银行短信、
抖音通知、骂人的文字截图, 都不是支付宝页面。
经理给的 111/112/116/118 真假图和 10-05 的两张假图全对; JPEG 压到 40、缩小到宽 600、放大照样判 2;
左右各裁掉 40 像素(页面边上的灰底被裁没了)判不了。本地拼音库 9,063 张 0 张可疑。

用法
----
    python -u training/color_scan.py --out E:/x/cs.csv --sheets E:/x/cs_sheets --copy-to E:/x/look D:/download2/OtherImages --every 8
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前, 见 pick_pinyin.py

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BLACK_MAX = 12           # 行核心颜色 <= 这个算纯黑。两种造假工具都是正好 0; 微信账单、支付宝话费充值页、
                         # 银行短信这些是 21~25(黑色 90% 不透明), 不能算(服务器 09-01~16 实测)
EDGE_LO, EDGE_HI = 238, 250   # 卡片外面页面边上的颜色: 支付宝是 #F5F5F5(245) 灰底上放白卡片, 造假工具 244/246;
                              # 微信账单和很多别的 App 是整页白(255), 不是支付宝的页面不判
LABEL_LO, LABEL_HI = 148, 165  # 标签颜色: 支付宝 #999(153), 造假工具 150; 抖音通知(147)这类别的页面在下限外。
                               # 上限放到 165(等于不限): 图被缩小 + 压缩以后标签字变浅, 造假图会到 156~163(10-05 经理转来的图)
GREY_LO, GREY_HI = 38, 64   # 行核心颜色在这个范围算 #333
SHARE = 0.60             # 占比 >= 这个就下结论


def _runs(on):
    segs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            a = i
            while i < n and on[i]:
                i += 1
            segs.append((a, i - 1))
        i += 1
    return segs


def measure_img(bgr, path=""):
    H, W = bgr.shape[:2]
    if W < 500 or H < 900:
        return None
    y0, y1 = int(0.095 * H), int(0.96 * H)
    x0, x1 = int(0.03 * W), int(0.97 * W)
    roi = bgr[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    chroma = roi.max(axis=2).astype(np.int16) - roi.min(axis=2)
    bg = int(np.bincount(g[:, :max(1, int(0.01 * W))].ravel(), minlength=256).argmax())   # 左边页边的底色
    if np.median(g) < 128:
        return {"path": path, "W": W, "H": H, "verdict": "CannotDetermine", "why": "深色模式"}
    dark = (g < 120) & (chroma < 30)
    label = (g >= 120) & (g <= 185) & (chroma < 30)             # 标签灰字(#999 一带)
    lx1 = int(0.27 * W) - x0                                     # 标签栏: 页面宽度 3%~27%
    vx0 = int(0.22 * W) - x0                                     # 取值栏: 22% 往右
    # ★ 只看"左边灰标签 + 右边深色取值"的行(账单详情上面那张卡片的样子)。
    #   别的页面(蓝底转账页、银行短信、订单说明、别的 App)本来就用纯黑字, 不在这里判(本地实测会误判)。
    cand = []           # (行顶, 标签核心色, 取值核心色, 标签左缘, 取值左缘)
    for a, b in _runs((dark | label).sum(1) >= 3):
        h = b - a + 1
        if h < 0.008 * H or h > 0.04 * H:
            continue
        lm = label[a:b + 1, :lx1]
        vm = dark[a:b + 1, vx0:]
        if lm.sum() < 10 or vm.sum() < 10:
            continue
        lcore = int(np.bincount(g[a:b + 1, :lx1][lm], minlength=256).argmax())
        vcore = int(np.bincount(g[a:b + 1, vx0:][vm], minlength=256).argmax())
        if not (140 <= lcore <= 165):
            continue
        ll = int(np.flatnonzero(lm.any(0))[0]) + x0
        vl = int(np.flatnonzero(vm.any(0))[0]) + vx0 + x0
        cand.append((a + y0, lcore, vcore, ll, vl, b + y0))
    # ★ 账单详情上面那张卡片: 标签都从左边 3%~10% 起、取值都从同一个 x 起(20%~34%), 上下对齐。
    #   收银台、支付结果页、订单说明这些别的页面对不齐, 就不判(它们本来就用纯黑字, 本地实测会误判)。
    pairs = []
    if len(cand) >= 3:
        llm = float(np.median([c[3] for c in cand]))
        vlm = float(np.median([c[4] for c in cand]))
        tol = 0.006 * W
        if 0.03 * W <= llm <= 0.10 * W and 0.20 * W <= vlm <= 0.34 * W:
            pairs = [c for c in cand if abs(c[3] - llm) <= tol and abs(c[4] - vlm) <= tol]
    rows_black = sum(1 for p in pairs if p[2] <= BLACK_MAX)
    rows_grey = sum(1 for p in pairs if GREY_LO <= p[2] <= GREY_HI)
    n = len(pairs)
    # 卡片左边外面(宽度 0.4%~1.6%)、对齐那几行高度上, 页面底色出现最多的灰度
    edge = ""
    if n:
        G = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        xa = int(0.004 * W)
        xb = max(xa + 1, int(0.016 * W))
        ev = np.concatenate([G[p[0]:p[5] + 1, xa:xb].ravel() for p in pairs])
        edge = int(np.bincount(ev, minlength=256).argmax())
    r = {"path": path, "W": W, "H": H, "rows": n, "rows_black": rows_black, "rows_grey": rows_grey,
         "rows_other": n - rows_black - rows_grey,
         "black_share": round(rows_black / n, 3) if n else "", "grey_share": round(rows_grey / n, 3) if n else "",
         "bg": bg, "edge": edge, "label_core": int(np.median([p[1] for p in pairs])) if n else "",
         "cores": " ".join(str(p[2]) for p in pairs[:30])}
    d = decide(r)
    if d["verdict"] == "Suspicious":          # 拼音检查比较贵, 只在要判可疑时才跑
        r["pinyin"] = int(_is_pinyin_page(bgr))
        d = decide(r, r["pinyin"])
    r.update(d)
    return r


def decide(r, pinyin=None):
    """按行数算(每行一票), 不按像素: 一大块黑色(按钮、图片)会把像素占比带偏(本地实测)。
    pinyin: 只在要判可疑时才会传进来(调用方先判拼音), 其余情况不用。"""
    if not r["rows"] or r["rows"] < 3:
        return {"verdict": "CannotDetermine", "why": "不是左标签右取值的账单详情页"}
    # ★ 只判支付宝的页面。微信账单也是左标签右取值, 字是黑色 90%(25)或纯黑, 不按支付宝的颜色判(服务器实测误判 200 多张)
    if not (EDGE_LO <= int(r["edge"]) <= EDGE_HI):
        return {"verdict": "CannotDetermine", "why": "页面边上不是支付宝的灰底(微信账单、别的 App)"}
    if not (LABEL_LO <= int(r["label_core"]) <= LABEL_HI):
        return {"verdict": "CannotDetermine", "why": "标签颜色不是支付宝的"}
    if float(r["black_share"]) >= SHARE:
        if pinyin:
            return {"verdict": "CannotDetermine", "why": "拼音页, 拼音字体的字本来就深, 不判"}
        return {"verdict": "Suspicious", "why": "取值字是纯黑, 支付宝是 #333"}
    if float(r["grey_share"]) >= SHARE:
        return {"verdict": "Ok", "why": ""}
    return {"verdict": "CannotDetermine", "why": "取值字颜色说不清"}


def _is_pinyin_page(bgr):
    """和 arrow_scan 一样用现成的 pinyin_probe(C# 那边是 PinyinCheck)。"""
    import arrow_scan
    return arrow_scan._is_pinyin_page(bgr)


def measure(p):
    try:
        buf = np.fromfile(p, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if bgr is None and buf.size > 2 and buf[0] == 0xFF and buf[1] == 0xD8:
            bgr = cv2.imdecode(np.concatenate([buf, np.array([0xFF, 0xD9], np.uint8)]), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"path": p, "verdict": "unreadable"}
        return measure_img(bgr, p)
    except Exception as e:  # noqa: BLE001
        return {"path": p, "verdict": "error", "why": repr(e)[:200]}


KEYS = ["path", "W", "H", "verdict", "black_share", "grey_share", "rows", "rows_black", "rows_grey", "rows_other", "label_core", "edge", "pinyin",
        "bg", "cores", "why"]


def tile(r, width=360):
    bgr = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    c = bgr[int(0.2 * H):int(0.45 * H), :]
    return cv2.resize(c, (width, max(1, int(c.shape[0] * width / W))), interpolation=cv2.INTER_AREA)


def write_sheets(rows, outdir, tag, cols=4, per=24):
    if not rows:
        return
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{tag}_index.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "no", "W", "H", "black_share", "grey_share", "path"])
        for s in range(0, len(rows), per):
            tiles = []
            for j, r in enumerate(rows[s:s + per]):
                try:
                    t = tile(r)
                except Exception:  # noqa: BLE001
                    t = np.full((200, 360, 3), 200, np.uint8)
                bar = np.full((20, t.shape[1], 3), 255, np.uint8)
                cv2.putText(bar, f"{s + j + 1} {r['W']}x{r['H']} blk {r['black_share']} g {r['grey_share']}", (3, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)
                tiles.append(np.vstack([bar, t]))
                w.writerow([f"{tag}_{s // per + 1:02d}.png", s + j + 1, r["W"], r["H"], r["black_share"],
                            r["grey_share"], r["path"]])
            hh = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0] + 4, 0, 4, cv2.BORDER_CONSTANT, value=(150, 150, 150))
                     for t in tiles]
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imencode(".png", grid)[1].tofile(os.path.join(outdir, f"{tag}_{s // per + 1:02d}.png"))


def report(rows, a):
    meas = [r for r in rows if r.get("verdict") in ("Ok", "Suspicious", "CannotDetermine")]
    c = collections.Counter(r["verdict"] for r in meas)
    print(f"\npages {len(meas):,}   Ok {c['Ok']:,}   Suspicious {c['Suspicious']:,}   CannotDetermine {c['CannotDetermine']:,}")
    for why, n in collections.Counter(r.get("why", "") for r in meas if r["verdict"] != "Ok").most_common():
        print(f"    {why or '-'}  {n:,}")
    vals = [float(r["black_share"]) for r in meas if r.get("black_share") not in ("", None)]
    if vals:
        h = np.histogram(vals, bins=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])[0]
        print("  black share histogram: " + "  ".join(
            f"{lo:.2f}+:{n}" for lo, n in zip([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], h)))
    print("\n  by resolution (pages, Suspicious, CannotDetermine-unclear):")
    by = collections.defaultdict(collections.Counter)
    for r in meas:
        by[f"{r['W']}x{r['H']}"][r["verdict"] + ("" if r["verdict"] != "CannotDetermine" else ":" + r.get("why", ""))] += 1
    for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:25]:
        print(f"    {k:<12}{sum(v.values()):>7,}  susp {v['Suspicious']:>5}  unclear {v['CannotDetermine:取值字颜色说不清']:>5}")
    days = collections.defaultdict(lambda: [0, 0])
    for r in meas:
        m = re.search(r"_(20\d{6})\d{6}\.", os.path.basename(r["path"]))
        d = days[m.group(1) if m else "?"]
        d[0] += r["verdict"] == "Suspicious"
        d[1] += 1
    print("\n  Suspicious by upload day: " + "  ".join(f"{k[4:]}:{v[0]}/{v[1]}" for k, v in sorted(days.items())))
    if a.sheets:
        sus = sorted([r for r in meas if r["verdict"] == "Suspicious"], key=lambda r: (r["W"], r["H"]))
        write_sheets(sus[:240], a.sheets, "suspicious")
        unc = [r for r in meas if r.get("why") == "取值字颜色说不清"]
        write_sheets(unc[:: max(1, len(unc) // 48)][:48], a.sheets, "unclear")
        print("sheets ->", a.sheets)
    if getattr(a, "copy_to", ""):
        # 整页原图拷出来人眼看: S_ 可疑(最多 400), P_ 拼音页不判的全部, U_ 说不清的抽 48, O_ 正常的抽 24
        import shutil
        os.makedirs(a.copy_to, exist_ok=True)
        key = lambda r: r["path"]
        sus = sorted([r for r in meas if r["verdict"] == "Suspicious"], key=key)
        pin = sorted([r for r in meas if str(r.get("pinyin", "")) == "1"], key=key)
        unc = sorted([r for r in meas if r.get("why") == "取值字颜色说不清"], key=key)
        ok = sorted([r for r in meas if r["verdict"] == "Ok"], key=key)
        picks = [(f"S_{i:03d}_{r['W']}x{r['H']}_", r) for i, r in enumerate(sus[:400], 1)]
        picks += [(f"P_{i:03d}_{r['W']}x{r['H']}_", r) for i, r in enumerate(pin, 1)]
        picks += [(f"U_{i:02d}_{r['W']}x{r['H']}_", r) for i, r in enumerate(unc[:: max(1, len(unc) // 48)][:48], 1)]
        picks += [(f"O_{i:02d}_{r['W']}x{r['H']}_", r) for i, r in enumerate(ok[:: max(1, len(ok) // 24)][:24], 1)]
        n = 0
        for pre, r in picks:
            try:
                shutil.copy2(r["path"], os.path.join(a.copy_to, pre + os.path.basename(r["path"])))
                n += 1
            except OSError:
                pass
        print(f"copied {n} images -> {a.copy_to}")


def main():
    ap = argparse.ArgumentParser(description="正文字颜色深度检查, 批量扫描")
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets", default="")
    ap.add_argument("--copy-to", default="", help="把可疑的、拼音页不判的、说不清的抽样、正常的抽样整页原图拷到这里")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--from-csv", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.from_csv:
        with open(a.out, encoding="utf-8-sig", newline="") as f:
            report(list(csv.DictReader(f)), a)
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8"):
        pass
    files = []
    for root in a.roots:
        if not os.path.isdir(root):
            raise SystemExit(f"not a folder: {root}")
        for dp, _d, fs in os.walk(root):
            files += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1].lower() in EXTS]
    files.sort()
    files = files[::max(1, a.every)]
    print(f"images to scan: {len(files):,}  (every {a.every})", flush=True)
    rows, bad, step = [], collections.Counter(), max(1000, len(files) // 50)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(measure, files, chunksize=32), 1):
            if r and r.get("verdict") in ("unreadable", "error"):
                bad[r["verdict"]] += 1
            elif r:
                rows.append(r)
            if i % step == 0 or i == len(files):
                print(f"  {i:,}/{len(files):,}  {i / max(time.time() - t0, 1e-9):.0f} img/s", flush=True)
    print(f"unreadable files {bad['unreadable']:,}   errors {bad['error']:,}", flush=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report(rows, a)
    print(f"done in {time.time() - t0:.0f}s  -> {a.out}")


if __name__ == "__main__":
    main()
