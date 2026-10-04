r"""状态栏信号格检查(经理说的 116) —— Python 参考实现 + 批量扫描。

现象
----
经理: "信号图标不对", 要"按照颜色像素处理"(他现在是按图片定位做的)。
苹果的信号格是系统按点数画的, 同一类屏幕上每一格的像素宽高、间距都是**固定值**, 一个像素都不差:
    1170 / 1179 / 1206 宽     单卡 宽 10 高 14 21 29 37 间距 6;  双卡上排 高 12 15 20 25, 下面一排 4 个灰点
    1290 / 1320 宽            单卡 宽 11 高 16 23 32 41 间距 7;  双卡上排 高 13 17 22 27
    (没信号时是 4 个一样高的点; 信号弱的格子是灰的, 形状不变)
假图(经理给的 116/111/112 三张 false.jpg 是同一个状态栏, 1080x2400):
    宽 8~9 高 13 20 28 35 间距 5~6 —— 高度的比例和苹果一样, 但格子细(宽/最高 0.23, 苹果 0.27)、
    间距大(间距/宽 0.75, 苹果 0.6), 是自己画的, 不是苹果的。
本地看过的苹果尺寸图里, 信号格不是标准值的, 几乎全出自早先别的检查挑出来的可疑图(负号、小数点等),
还有一张 112 确认过的假图 —— 和别的检查是独立的, 互相印证。

量法
----
1. 状态栏(高度 6.5% 以内、宽度 55% 往右), 比底色深 30 以上、不饱和的像素, 连通块。
2. 信号格: 底边对齐(+-2)、宽度相近(+-2)、间距均匀、从左到右不变矮的一排竖块, 取 4 个的那组;
   下面紧贴一排小方点(双卡)也记下来。
3. 签名 = 宽 | 4 格的高 | 3 个间距 | 下面的点数。
4. 苹果原尺寸截图: 签名必须是这个尺寸的标准签名之一, 不是就可疑。
5. 别的尺寸: 信号格和已知造假工具画的状态栏一样(位置、大小差不到 1 个像素)就可疑; 其余不判。
   ★ 试过"苹果版页面(标题居中) + 苹果截不出来的尺寸"就判可疑, 不行: 安卓上标题居中的页面很多
     (明细详情、交易详情、电子回单、银行 App), 安卓的信号格比例也有和苹果几乎一样的(宽 7 高 10 14 19 25)。
     这两样只记录(ios_layout、canvas_ok), 不参与判定。
   深色模式不判。

用法
----
    python -u training/signal_scan.py --out E:/x/sg.csv --sheets E:/x/sg_sheets D:/download2/OtherImages
    --every 8   每 8 张取 1 张(和前面几个扫描同一批)
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

# 苹果屏幕类别 -> 标准签名(宽|4 格高|3 个间距|下面点数)。先用本地看过的真图定, 服务器数据出来再补。
_A = {"10|14,21,29,37|6,6,6|0", "10|12,15,20,25|6,6,6|4", "10|11,11,11,11|6,6,6|0", "10|10,10,10,10|6,6,6|0"}
_B = {"11|16,23,32,41|7,7,7|0", "11|13,17,22,27|7,7,7|4", "11|12,12,12,12|7,7,7|0", "11|11,11,11,11|7,7,7|0"}
GENUINE = {
    (1170, 2532): _A, (1179, 2556): _A, (1206, 2622): _A,
    (1290, 2796): _B, (1320, 2868): _B, (1284, 2778): _A | _B,
    (1125, 2436): {"9|12,18,25,32|5,5,5|0"},
    (1242, 2688): {"10|13,19,27,35|5,5,5|0"},
    (828, 1792): {"7|9,13,18,23|3,3,3|0"},
}
# 苹果截图的原尺寸(宽 -> 高), 用来判断一张图能不能由苹果截图裁出来或等比缩放出来
IPHONE_SIZES = {1170: 2532, 1179: 2556, 1206: 2622, 1284: 2778, 1290: 2796, 1320: 2868, 1125: 2436,
                1242: 2688, 828: 1792, 1080: 2340, 750: 1334}


def _comps(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    return [(int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3]), int(st[i][4])) for i in range(1, n)]


def find_bars(bgr):
    """状态栏里的信号格。返回 dict 或 None(找不到), 深色模式返回 {'dark': 1}。"""
    H, W = bgr.shape[:2]
    y1, x0 = int(0.065 * H), int(0.55 * W)
    band = bgr[:y1, x0:]
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    # ★ 判"不是彩色"用 三通道最大减最小(绝对色度), 不用 HSV 饱和度: 接近黑色的像素饱和度很不稳,
    #   JPEG 里 (5,3,8) 这种黑点饱和度能到 95, 会把黑色信号格切成碎块(经理给的 116 假图就是这样)
    chroma = (band.max(axis=2).astype(np.int16) - band.min(axis=2)).astype(np.int16)
    bg = int(np.median(g))
    if bg < 128:
        return {"dark": 1}
    cs = [c for c in _comps((g < bg - 30) & (chroma < 40)) if c[4] >= 6]
    best = None
    for c in cs:
        bottom = c[1] + c[3]
        grp = sorted([d for d in cs if abs(d[1] + d[3] - bottom) <= 2 and abs(d[2] - c[2]) <= 2
                      and d[3] >= 0.8 * d[2] and d[2] >= 3], key=lambda d: d[0])
        # 从 c 开始往右取连着的, 间距均匀(相邻间距差 <= 2, 间距 < 1.5 宽)
        if c not in grp:
            continue
        i = grp.index(c)
        run = [c]
        for d in grp[i + 1:]:
            gap = d[0] - (run[-1][0] + run[-1][2])
            if gap < 0 or gap > 1.5 * c[2]:
                break
            if len(run) >= 2:
                g0 = run[1][0] - (run[0][0] + run[0][2])
                if abs(gap - g0) > 2:
                    break
            run.append(d)
            if len(run) == 4:
                break
        hs = [d[3] for d in run]
        if len(run) == 4 and all(hs[k] <= hs[k + 1] for k in range(3)):
            if best is None or run[0][0] < best[0][0]:
                best = run
    if not best:
        return None
    bars = best
    w = int(np.median([d[2] for d in bars]))
    hs = [d[3] for d in bars]
    gaps = [bars[k + 1][0] - (bars[k][0] + bars[k][2]) for k in range(3)]
    bottom = bars[0][1] + bars[0][3]
    left, right = bars[0][0], bars[3][0] + bars[3][2]
    dots = [d for d in cs if d[1] > bottom and d[1] - bottom <= 2 * w and abs(d[2] - d[3]) <= 2
            and left - 2 <= d[0] <= right + 2]
    widths = [d[2] for d in bars]
    sig = f"{w if max(widths) - min(widths) <= 0 else '~' + str(w)}|{','.join(map(str, hs))}|{','.join(map(str, gaps))}|{len(dots)}"
    return {"sig": sig, "w": w, "widths": " ".join(map(str, widths)), "heights": " ".join(map(str, hs)),
            "gaps": " ".join(map(str, gaps)), "dots": len(dots), "bar_left": x0 + left, "bar_bottom": bottom,
            "w_ratio": round(w / max(hs), 3), "gap_ratio": round(float(np.median(gaps)) / w, 3),
            "h_ratio": " ".join(f"{h / max(hs):.2f}" for h in hs)}


def ios_layout(bgr):
    """苹果版支付宝的页面: 导航栏标题居中; 安卓版标题靠左挨着返回箭头。
    看状态栏下面那一条(高度 3.5%~11%)的黑字: 中间一条(40%~60%)有字、左边(9%~30%)没字 -> 苹果。
    返回 1 / 0, 看不出来返回 ''。"""
    H, W = bgr.shape[:2]
    band = cv2.cvtColor(bgr[int(0.035 * H):int(0.11 * H)], cv2.COLOR_BGR2GRAY)
    if np.median(band) < 128:
        return ""
    dark = band < 110
    mid = int(dark[:, int(0.40 * W):int(0.60 * W)].sum())
    left = int(dark[:, int(0.09 * W):int(0.30 * W)].sum())
    if mid == 0 and left == 0:
        return ""
    return int(mid > 0 and left == 0)


def canvas_ok(W, H):
    """这个尺寸能不能是苹果截图裁出来(宽是苹果宽度、高不超过原图)或等比缩放出来(高宽比差 <= 0.5%)的。"""
    if W in IPHONE_SIZES and H <= IPHONE_SIZES[W]:
        return 1
    r = H / W
    return int(any(abs(r - h / w) <= 0.005 * (h / w) for w, h in IPHONE_SIZES.items()))


def measure_img(bgr, path=""):
    H, W = bgr.shape[:2]
    if W < 500 or H < 900:
        return None
    r = {"path": path, "W": W, "H": H}
    b = find_bars(bgr)
    if b and b.get("dark"):
        r.update(verdict="CannotDetermine", why="深色模式")
        return r
    if b:
        r.update(b)
    r["ios_layout"] = ios_layout(bgr)
    r["canvas_ok"] = canvas_ok(W, H)
    r.update(decide(W, H, b))
    return r


# 已知造假工具画的状态栏。同一个工具画出来的信号格位置、大小每张都一样(JPEG 压过会差 1 个像素)。
# 经理给的 116/111/112 三张 false.jpg, 和服务器上 111/112 判出来的 S_013、S_002、S_117、S_129 都是这一个:
# 1080x2400 上画的苹果状态栏, 格子细(宽/最高 0.23, 苹果 0.27)、间距大、第一格比别的宽一个像素。
FAKE_TEMPLATES = [
    {"W": 1080, "H": 2400, "left": 793, "bottom": 80, "heights": (13, 20, 28, 35), "widths": (8, 9), "gaps": (5, 6)},
]


def match_template(W, H, b):
    for t in FAKE_TEMPLATES:
        if (W, H) != (t["W"], t["H"]):
            continue
        if abs(b["bar_left"] - t["left"]) > 3 or abs(b["bar_bottom"] - t["bottom"]) > 2:
            continue
        hs = [int(v) for v in b["heights"].split()]
        ws = [int(v) for v in b["widths"].split()]
        gs = [int(v) for v in b["gaps"].split()]
        if (all(abs(h - th) <= 1 for h, th in zip(hs, t["heights"])) and all(w in t["widths"] for w in ws)
                and all(g in t["gaps"] for g in gs)):
            return True
    return False


def decide(W, H, b):
    """b = find_bars 的结果(没找到是 None)。
    标题居中(ios_layout)和画布尺寸(canvas_ok)只记录不判: 标题居中的安卓页、银行 App 页面很多,
    光凭这个会误判(本地实测, 明细详情 / 交易详情 / 电子回单 都是居中的)。"""
    sig = b["sig"] if b else ""
    if (W, H) in GENUINE:
        if not sig:
            return {"verdict": "CannotDetermine", "why": "苹果尺寸, 没找到信号格"}
        if sig in GENUINE[(W, H)]:
            return {"verdict": "Ok", "why": ""}
        return {"verdict": "Suspicious", "why": "苹果尺寸, 信号格不是这个屏幕的标准样子"}
    if b and match_template(W, H, b):
        return {"verdict": "Suspicious", "why": "状态栏和已知造假工具画的一样"}
    return {"verdict": "CannotDetermine", "why": "不是苹果原尺寸"}


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


KEYS = ["path", "W", "H", "verdict", "sig", "w", "widths", "heights", "gaps", "dots", "w_ratio", "gap_ratio",
        "h_ratio", "bar_left", "bar_bottom", "ios_layout", "canvas_ok", "why"]


def tile(r, width=420):
    bgr = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    c = bgr[:int(0.11 * H), :]
    return cv2.resize(c, (width, max(1, int(c.shape[0] * width / W))), interpolation=cv2.INTER_AREA)


def write_sheets(rows, outdir, tag, cols=3, per=30):
    if not rows:
        return
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{tag}_index.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "no", "W", "H", "sig", "ios_layout", "canvas_ok", "path"])
        for s in range(0, len(rows), per):
            tiles = []
            for j, r in enumerate(rows[s:s + per]):
                try:
                    t = tile(r)
                except Exception:  # noqa: BLE001
                    t = np.full((80, 420, 3), 200, np.uint8)
                bar = np.full((20, t.shape[1], 3), 255, np.uint8)
                cv2.putText(bar, f"{s + j + 1} {r['W']}x{r['H']} {r.get('sig', '')}", (3, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)
                tiles.append(np.vstack([bar, t]))
                w.writerow([f"{tag}_{s // per + 1:02d}.png", s + j + 1, r["W"], r["H"], r.get("sig", ""),
                            r.get("ios_layout", ""), r.get("canvas_ok", ""), r["path"]])
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
    # 苹果原尺寸: 每个尺寸的签名分布(这是定标准签名的依据)
    print("\n  iPhone native sizes: signature counts (top 8 per size)")
    by = collections.defaultdict(collections.Counter)
    for r in meas:
        k = (int(r["W"]), int(r["H"]))
        if k in GENUINE:
            by[k][r.get("sig") or r.get("why", "")] += 1
    for k in sorted(by, key=lambda k: -sum(by[k].values())):
        print(f"    {k[0]}x{k[1]}  pages {sum(by[k].values()):,}")
        for sig, n in by[k].most_common(8):
            mark = "" if sig in GENUINE[k] else "   <- not standard"
            print(f"        {n:>6,}  {sig}{mark}")
    # 别的尺寸: 同一个尺寸上一模一样的签名反复出现, 可能是造假工具的模板(要人眼看)
    other = [r for r in meas if (int(r["W"]), int(r["H"])) not in GENUINE and r.get("sig")]
    print(f"\n  non-native sizes with signal bars found: {len(other):,}   most repeated size|signature|bar_left|bottom:")
    cc = collections.Counter(f"{r['W']}x{r['H']}|{r['sig']}|{r.get('bar_left')}|{r.get('bar_bottom')}" for r in other)
    for k, n in cc.most_common(20):
        print(f"        {n:>6,}  {k}")
    ios_other = [r for r in meas if (int(r["W"]), int(r["H"])) not in GENUINE and str(r.get("ios_layout")) == "1"]
    print(f"  (title centred, non-native: {len(ios_other):,}, canvas no iPhone can produce: "
          f"{sum(1 for r in ios_other if str(r.get('canvas_ok')) == '0'):,} -- recorded only)")
    days = collections.defaultdict(lambda: [0, 0])
    for r in meas:
        m = re.search(r"_(20\d{6})\d{6}\.", os.path.basename(r["path"]))
        d = days[m.group(1) if m else "?"]
        d[0] += r["verdict"] == "Suspicious"
        d[1] += 1
    print("\n  Suspicious by upload day: " + "  ".join(f"{k[4:]}:{v[0]}/{v[1]}" for k, v in sorted(days.items())))
    if a.sheets:
        sus = sorted([r for r in meas if r["verdict"] == "Suspicious"], key=lambda r: (r["why"], r["W"], r.get("sig", "")))
        write_sheets(sus[:300], a.sheets, "suspicious")
        okn = [r for r in meas if r["verdict"] == "Ok"]
        write_sheets(okn[:: max(1, len(okn) // 30)][:30], a.sheets, "normal")
        print("sheets ->", a.sheets)


def main():
    ap = argparse.ArgumentParser(description="状态栏信号格检查, 批量扫描")
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets", default="")
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
