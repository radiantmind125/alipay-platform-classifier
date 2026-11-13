r"""账单详情页"处理进度"三个蓝色对勾和它右边文字的上下对齐检查(11-13 经理转来的"蓝色对勾要整齐") —— Python 参考实现 + 批量扫描。

现象
----
真图: 三个圆点(提交银行处理 / 银行处理中 / 银行入账成功)各自正对着右边那行字, 三个圆点相对各自文字的上下位置一模一样
     (同一张图里差不到 1 像素)。
假图: 圆点一列的间距和文字的间距对不上(11-13 三张假图都是圆点间距比文字间距大 3.6%), 越往下越偏,
     第三个"银行入账成功"那个圆点明显偏下 —— 就是经理图上标的"假图偏下"。
和 112(timeline_scan.py, 圆点左右缩进)是两回事: 11-13 的 fake.png 112 量出来缩进 11 像素, 正常, 放过去了; 这里抓得到。
以前 112 抓到的假图里也有一大半带这个毛病(本地 79 个假图编号里 53 个)。

量法
----
1. 找圆点: 和 112 完全一样(timeline_scan.find_circles / timeline), 同一列竖着排开 >= 2 个带白色对勾的蓝色圆点。
2. 每个圆点的上下中心: 圆点宽度 16% / 21% / 26% / 31% / 69% / 74% / 79% / 84% 这 8 列(避开中间那根连线),
   每列从上到下的灰度曲线取半高处的上下边(插值到小数像素), 上边平均、下边平均, 取中点。
3. 圆点右边的步骤文字: 圆点上下各放宽 0.8D(D = 圆点宽), 黑字(灰度 < 150 且不是彩色)按行切段,
   取离圆点中心最近、高 >= 0.3D 的一段; 量它的**下边**(整行墨量曲线 15% 处, 小数像素)。
   ★ 用下边不用中线: 拼音字体的页面拼音在汉字正上方, 有时连成一段有时断开, 上边和中线会跳; 下边不受拼音影响。
4. 文字下面那行灰色日期(如 11-12 04:06): 文字下方 1.8D 以内第一段高 >= 0.25D 的灰字, 量它的中线(25% 处上下边的中点)。
   三行日期通常是同一串数字, 数字没有拼音、上下一样高, 是最稳的参照。
5. 每个圆点: 圆点中心 - 文字下边、圆点中心 - 日期中线。各自在几个圆点之间的最大差 / D = 文字散度、日期散度。
   分数 = 两个散度里小的那个: 两种参照都说"圆点没对齐"才算数。
   ★ 为什么取小的: 手写字体真图(C_027)个别字的尾巴特别长, 文字下边会差出 0.17D, 但日期参照正常;
     有一张真图(C_052)日期行认偏了, 日期散度 0.21D, 但文字参照正常。假图是圆点自己挪了位置, 两种参照一起偏。
6. 分数 > 0.10 判可疑(Suspicious), 否则 Ok。(11-13 服务器扫完后从 0.08 提到 0.10, 原因见下面"服务器实测")

判不了(CannotDetermine)的几种, 都是宁可不判也不误判:
    圆点直径 < 20 像素(图被缩得太小, 手写字体真图缩到 40% 再压缩, 有一份到了 0.15);
    有圆点量不到圆心、右边的步骤文字或下面的日期;
    几个圆点的高矮差 > 10%(有东西压在圆点上, 比如截图标注的红框);
    步骤文字或日期的左缘前后差 > 0.25D(认到的不是同一种行);
    文字到日期的间距前后差 > 0.15D, 或日期行高矮差 > 25%(日期行认错了)。

本地实测(2026-11-13)
--------------------
真图时间轴页 822 张(拼音页 466、11 月白图 167、服务器 112 抽查里的正常页 ...): 分数最高 0.033。
    每张再过 6 种缩放(1 ~ 0.4) x 4 种 JPEG(不压 / 90 / 70 / 50), 19,363 份: 圆点直径 >= 20 的最高 0.045;
    直径 < 20 的里有 1 份手写字体到 0.15(所以直径 < 20 不判)。
假图: 11-13 三张 0.12 ~ 0.24(经理原图 fake.png 0.233); 112 以前的假图编号 79 个里 53 个 0.14 ~ 0.90,
    其余 26 个 <= 0.03(那批假图圆点没挪, 靠 112 抓); 所有缩放压缩下带这个毛病的假图最低 0.14。

服务器实测(2026-11-13, D:/download2/OtherImages 全部 217,166 张, 有时间轴 70,196 张, 量到 70,058 张)
------------------------------------------------------------------------------------------------
可疑 44 张, 逐张看过圆点都明显对不齐(圆点等间距排, 间距比文字间距大 3% ~ 11% 或小 7%); 其中 39 张 112 也判可疑,
    5 张 112 放过(都是安卓)。1179x2556 的 6 张和经理 11-11 的 fake.png 圆点偏移一模一样(同一个模板),
    那 6 张(11-01 ~ 11-04)112 还能抓, fake.png 已经把 112 那个缩进改对了, 这里照样抓。
0.08 ~ 0.10 之间 0 张, 所以阈值 0.08 和 0.10 在服务器上结果完全一样。
提到 0.10 的原因: 服务器上有一批安卓真图(1080 宽 D 45 / 720 宽 D 30, 9 月和 11 月都有)中间那个圆点固定差 1dp
    (3 像素 / 2 像素), 分数 0.066 ~ 0.067, 单看文字参照有到 0.081 的; 别的分辨率换算取整可能更高, 0.08 太贴。
    本地假图(原图和 1,738 份缩放压缩)和服务器 44 张没有一张落在 0.06 ~ 0.12, 提到 0.10 一张不丢。
另: 1220x2712 有 29 张(11-01 ~ 11-04)圆点间距比文字大 0.9%, 分数 0.063, 这里不判; 同尺寸真图 576 张都在
    0.997 ~ 1.003, 这 29 张 112 全判可疑, 是另一个假图模板, 靠 112 抓。

用法
----
    python -u training/step_align.py --out E:/x/sa.csv --sheets E:/x/sa_sheets --copy-to E:/x/sa_sus D:/download2/OtherImages
    --every 5   每 5 张取 1 张(按文件名排序后均匀取)
    --workers 8 进程数(默认 CPU 数 - 1)
    --from-csv  不扫图, 读 --out 的 csv 重新判(规则改了不用重扫)、出统计和对照图
C# 版是 demo/StepAlignCheck.cs, 两边逐步对应(整数列号、整数墨量和、同样的求和顺序, 结果逐位一样), 改一边要同步改另一边。
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前, 见 pick_pinyin.py

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline_scan as ts  # noqa: E402

cv2.setNumThreads(1)
EXTS = ts.EXTS

THRESHOLD = 0.10         # 分数 > 这个值判可疑。真图本地(直径 >= 20)最高 0.045; 服务器没判可疑的 7 万张最高 0.071; 假图最低 0.14
MIN_DIAMETER = 20        # 圆点直径小于这个像素数不判
CHORD_SPREAD = 0.10      # 圆点高矮差 / 中位数 > 这个值不判(真图最多 0.044)
LEFT_SPREAD = 0.25       # 步骤文字 / 日期左缘前后差 > 这么多个 D 不判(真图最多 0.083)
GAP_SPREAD = 0.15        # 文字下边到日期上边的间距前后差 > 这么多个 D 不判(真图 1 张 0.20 = 日期认偏, 其余 <= 0.11)
DATE_H_SPREAD = 0.25     # 日期行高矮差 / 中位数 > 这个值不判(真图最多 0.10)
COLS = (16, 21, 26, 31, 69, 74, 79, 84)   # 量圆点上下边的列: 圆点宽的百分之几处(避开中间的连线)
MARGIN = 6               # 圆点框上下各多看几个像素(形态学处理会削掉圆点边上一两个像素)
TEXT_FRAC = 0.15         # 文字下边: 墨量曲线从背景往上 15% 处
DATE_FRAC = 0.25         # 日期上下边: 25% 处


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _first_cross(p, thr):
    """曲线第一次 >= thr 的位置(小数像素, 和前一个点线性插值)。"""
    for i in range(len(p)):
        if p[i] >= thr:
            if i == 0:
                return 0.0
            return (i - 1) + (thr - p[i - 1]) / (p[i] - p[i - 1])
    return None


def _last_cross(p, thr):
    """曲线最后一次 >= thr 的位置(小数像素, 和后一个点线性插值)。"""
    n = len(p)
    for i in range(n - 1, -1, -1):
        if p[i] >= thr:
            if i == n - 1:
                return float(i)
            return i + (p[i] - thr) / (p[i] - p[i + 1])
    return None


def circle_edges(gray, c):
    """圆点的上下边(小数像素): COLS 那 8 列各自量半高处的上下边, 再各自平均。量得到的列 < 4 返回 None。"""
    x, y, w, h = c
    H, W = gray.shape[:2]
    y0, y1 = max(0, y - MARGIN), min(H, y + h + MARGIN)
    if y1 - y0 < 8:
        return None
    tops, bots = [], []
    for k in COLS:
        col = x + (w * k) // 100
        if col < 0 or col >= W:
            continue
        p = [255 - int(v) for v in gray[y0:y1, col]]
        bg = max(p[0], p[1], p[-2], p[-1])
        pk = max(p)
        if pk - bg < 50:
            continue
        thr = bg + 0.5 * (pk - bg)
        t, b = _first_cross(p, thr), _last_cross(p, thr)
        tops.append(t + y0)
        bots.append(b + y0)
    if len(tops) < 4:
        return None
    st = sb = 0.0
    for v in tops:
        st += v
    for v in bots:
        sb += v
    return st / len(tops), sb / len(bots)


def _profile(g, a, b, c0, c1):
    """行 [a-3, b+3] 在列 [c0, c1] 上的墨量(255 - 灰度 的整数和), 返回 (曲线, 起始行)。"""
    lo, hi = max(0, a - 3), min(g.shape[0], b + 4)
    blk = 255 - g[lo:hi, c0:c1 + 1].astype(np.int64)
    return [int(v) for v in blk.sum(1)], lo


def step_text(bgr, c, D):
    """圆点右边的步骤文字: 离圆点中心最近、高 >= 0.3D 的黑字行。返回 dict(下边 / 左缘 / 整数上下沿) 或 None。"""
    x, y, w, h = c
    H, W = bgr.shape[:2]
    x0, x1 = min(W - 1, int(x + w + 0.15 * D)), min(W, int(x + w + 6 * D))
    y0, y1 = max(0, int(y - 0.8 * D)), min(H, int(y + h + 0.8 * D))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    sub = bgr[y0:y1, x0:x1]
    dark, _ = ts._dark_grey(sub)
    cyr = y + h / 2.0 - y0
    best = None
    for a, b in ts._runs(dark.sum(1) >= 2):
        if b - a + 1 < 0.3 * D:
            continue
        d = abs((a + b) / 2.0 - cyr)
        if best is None or d < best[0]:
            best = (d, a, b)
    if best is None:
        return None
    _, a, b = best
    cols = np.flatnonzero(dark[a:b + 1].any(0))
    c0, c1 = int(cols[0]), int(cols[-1])
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    p, lo = _profile(g, a, b, c0, c1)
    bg, pk = min(p[0], p[-1]), max(p)
    if pk <= bg:
        return None
    bot = _last_cross(p, bg + TEXT_FRAC * (pk - bg))
    return {"bot": bot + lo + y0, "left": c0 + x0, "a": a + y0, "b": b + y0}


def date_row(bgr, t, D):
    """步骤文字下面那行灰色日期: 文字下方 1.8D 以内第一段高 >= 0.25D 的灰字。返回 dict(中线 / 左缘 / 上沿 / 行高) 或 None。"""
    H, W = bgr.shape[:2]
    x0, x1 = max(0, int(t["left"] - 0.3 * D)), min(W, int(t["left"] + 6 * D))
    y0, y1 = t["b"] + 1, min(H, int(t["b"] + 1.8 * D))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    sub = bgr[y0:y1, x0:x1]
    _, grey = ts._dark_grey(sub)
    pick = None
    for a, b in ts._runs(grey.sum(1) >= 2):
        if b - a + 1 >= 0.25 * D:
            pick = (a, b)
            break
    if pick is None or pick[0] < 2:          # 紧贴着文字: 分不清是日期还是文字的毛边
        return None
    a, b = pick
    cols = np.flatnonzero(grey[a:b + 1].any(0))
    c0, c1 = int(cols[0]), int(cols[-1])
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    p, lo = _profile(g, a, b, c0, c1)
    bg, pk = min(p[0], p[-1]), max(p)
    if pk <= bg:
        return None
    thr = bg + DATE_FRAC * (pk - bg)
    top, bot = _first_cross(p, thr), _last_cross(p, thr)
    return {"mid": (top + bot) / 2.0 + lo + y0, "left": c0 + x0, "a": a + y0, "h": b - a + 1}


def measure_img(bgr, path=""):
    """没有时间轴返回 None; 否则返回一行结果(verdict 是 Ok / Suspicious / CannotDetermine)。"""
    H, W = bgr.shape[:2]
    if W < 300 or H < 300:
        return None
    cs = ts.timeline(ts.find_circles(bgr))
    if not cs:
        return None
    D = float(np.median([c[2] for c in cs]))
    r = {"path": path, "W": W, "H": H, "n": len(cs), "D": D, "circle_left": int(np.median([c[0] for c in cs])),
         "circle_top": cs[0][1], "circle_bottom": cs[-1][1] + cs[-1][3]}
    m = []
    if D >= MIN_DIAMETER:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for c in cs:
            e = circle_edges(gray, c)
            t = step_text(bgr, c, D) if e else None
            d = date_row(bgr, t, D) if t else None
            if not (e and t and d):
                m.append(None)
                continue
            m.append({"top": e[0], "bot": e[1], "tbot": t["bot"], "tleft": t["left"], "tb": t["b"],
                      "dmid": d["mid"], "dleft": d["left"], "da": d["a"], "dh": d["h"]})
    r["m"] = json.dumps(m)
    r.update(decide(D, m))
    return r


def decide(D, m):
    """量好之后的判定。m = 每个圆点的量值(量不到的是 None)。单独拿出来是为了能拿扫描结果 csv 重新判(--from-csv)。"""
    out = {"verdict": "CannotDetermine", "score": "", "sp_text": "", "sp_date": "", "dy_text": "", "dy_date": ""}
    if D < MIN_DIAMETER:
        out["why"] = f"圆点直径小于 {MIN_DIAMETER} 像素, 图被缩小过"
        return out
    if len(m) < 2 or any(v is None for v in m):
        out["why"] = "有圆点量不到圆心、右边的步骤文字或下面的日期"
        return out
    chords = [v["bot"] - v["top"] for v in m]
    if (max(chords) - min(chords)) / _median(chords) > CHORD_SPREAD:
        out["why"] = "几个圆点高矮不一样(有东西压在圆点上)"
        return out
    for k in ("tleft", "dleft"):
        vals = [v[k] for v in m]
        if max(vals) - min(vals) > LEFT_SPREAD * D:
            out["why"] = "步骤文字或日期的左缘前后对不齐, 认到的不是同一种行"
            return out
    gaps = [v["da"] - v["tb"] for v in m]
    dhs = [v["dh"] for v in m]
    if max(gaps) - min(gaps) > GAP_SPREAD * D or (max(dhs) - min(dhs)) / _median(dhs) > DATE_H_SPREAD:
        out["why"] = "日期行认得不一致"
        return out
    cy = [(v["top"] + v["bot"]) / 2.0 for v in m]
    dt = [c - v["tbot"] for c, v in zip(cy, m)]
    dd = [c - v["dmid"] for c, v in zip(cy, m)]
    st, sd = (max(dt) - min(dt)) / D, (max(dd) - min(dd)) / D
    score = min(st, sd)
    out.update(score=round(score, 6), sp_text=round(st, 6), sp_date=round(sd, 6),
               dy_text=" ".join(f"{v:+.2f}" for v in dt), dy_date=" ".join(f"{v:+.2f}" for v in dd))
    if score > THRESHOLD:
        out["verdict"] = "Suspicious"
        out["why"] = f"圆点和右边文字上下没对齐: 几个圆点之间差 {score * D:.1f} 像素({score:.3f} 个直径)"
    else:
        out["verdict"] = "Ok"
        out["why"] = ""
    return out


def read_image(p):
    """读图; JPEG 被截断(没有结尾标记)解不出来时补上结尾标记再解一次。读不了返回 None。"""
    buf = np.fromfile(p, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if bgr is None and buf.size > 2 and buf[0] == 0xFF and buf[1] == 0xD8:
        bgr = cv2.imdecode(np.concatenate([buf, np.array([0xFF, 0xD9], np.uint8)]), cv2.IMREAD_COLOR)
    return bgr


def measure(p, with112=True):
    try:
        bgr = read_image(p)
        if bgr is None:
            return {"path": p, "verdict": "unreadable"}
        r = measure_img(bgr, p)
        if r and with112:
            # 顺带记下 112(左右缩进)的结论, 看这个检查抓到的有多少是 112 本来就抓得到的
            t = ts.measure_img(bgr, p) or {}
            r["v112"], r["off112"] = t.get("verdict", ""), t.get("offset", "")
        return r
    except Exception as e:  # noqa: BLE001
        return {"path": p, "verdict": "error", "why": repr(e)[:200]}


KEYS = ["path", "W", "H", "verdict", "score", "sp_text", "sp_date", "D", "n", "dy_text", "dy_date", "v112", "off112",
        "circle_left", "circle_top", "circle_bottom", "m", "why"]


def tile(r, width=300):
    """时间轴那一块(圆点 + 前两三个字 + 日期), 红线 = 每个圆点量到的中心。真图红线切每行字的同一个高度, 假图越往下越偏。"""
    bgr = read_image(r["path"])      # 截断的 JPEG 也要补结尾再解, 不然对照图里是灰格子(11-13 服务器 9 月那批)
    D, cl = float(r["D"]), int(r["circle_left"])
    H, W = bgr.shape[:2]
    y0, y1 = max(0, int(int(r["circle_top"]) - 0.8 * D)), min(H, int(int(r["circle_bottom"]) + 1.6 * D))
    x0, x1 = max(0, int(cl - 0.5 * D)), min(W, int(cl + 4.5 * D))
    c = bgr[y0:y1, x0:x1].copy()
    z = width / c.shape[1]
    c = cv2.resize(c, (width, max(1, int(c.shape[0] * z))), interpolation=cv2.INTER_AREA)
    for v in json.loads(r["m"]) if r.get("m") else []:
        if v:
            Y = int(round(((v["top"] + v["bot"]) / 2.0 - y0) * z))
            cv2.line(c, (0, Y), (c.shape[1] - 1, Y), (0, 0, 255), 1)
    return c


def write_sheets(rows, outdir, tag, cols=4, per=24):
    """对照图: 每张 24 格, 格子上写 序号 / 分数 / 宽; 序号对应 tag_index.csv。"""
    if not rows:
        return
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{tag}_index.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "no", "score", "W", "H", "v112", "path"])
        for s in range(0, len(rows), per):
            part = rows[s:s + per]
            tiles = []
            for j, r in enumerate(part):
                try:
                    t = tile(r)
                except Exception:  # noqa: BLE001
                    t = np.full((300, 300, 3), 200, np.uint8)
                bar = np.full((22, t.shape[1], 3), 255, np.uint8)
                cv2.putText(bar, f"{s + j + 1}  {float(r['score']):.3f}  {r['W']}  {r.get('v112', '')[:4]}", (3, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
                tiles.append(np.vstack([bar, t]))
                w.writerow([f"{tag}_{s // per + 1:02d}.png", s + j + 1, r["score"], r["W"], r["H"], r.get("v112", ""),
                            r["path"]])
            hh = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0] + 4, 0, 4, cv2.BORDER_CONSTANT, value=(150, 150, 150))
                     for t in tiles]
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imencode(".png", grid)[1].tofile(os.path.join(outdir, f"{tag}_{s // per + 1:02d}.png"))


def main():
    ap = argparse.ArgumentParser(description="处理进度圆点上下对齐检查, 批量扫描")
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--out", required=True, help="结果 csv(只记有时间轴的图)")
    ap.add_argument("--sheets", default="", help="出对照图的目录; 不给就不出")
    ap.add_argument("--copy-to", default="", help="把判可疑的原图拷到这个目录(最多 500 张), 拿回来逐张看")
    ap.add_argument("--every", type=int, default=1, help="每 N 张取 1 张")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--from-csv", action="store_true", help="不扫图, 读 --out 的 csv 按现在的规则重新判, 出统计和对照图")
    a = ap.parse_args()

    t0 = time.time()
    if a.from_csv:
        with open(a.out, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r.get("D", "") != "" and r.get("m", "") != "":
                r.update(decide(float(r["D"]), json.loads(r["m"])))
        report(rows, None, a)
        return
    if not a.roots:
        raise SystemExit("give image folders, or --from-csv")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)     # 先建好, 别扫了半小时才发现写不进去
    files = []
    for root in a.roots:
        if not os.path.isdir(root):
            raise SystemExit(f"not a folder: {root}")
        for dp, _d, fs in os.walk(root):
            files += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1].lower() in EXTS]
    files.sort()
    files = files[::max(1, a.every)]
    print(f"images to scan: {len(files):,}  (every {a.every})  listing took {time.time() - t0:.0f}s", flush=True)

    rows, step = [], max(1000, len(files) // 50)
    bad = {"unreadable": [], "error": []}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(measure, files, chunksize=32), 1):
            if r and r.get("verdict") in bad:
                bad[r["verdict"]].append(r)
            elif r:
                rows.append(r)
            if i % step == 0 or i == len(files):
                el = time.time() - t0
                ns = sum(1 for x in rows if x["verdict"] == "Suspicious")
                print(f"  {i:,}/{len(files):,}  timeline pages {len(rows):,}  suspicious {ns:,}  {i / max(el, 1e-9):.0f} img/s",
                      flush=True)
    print(f"unreadable files {len(bad['unreadable']):,}   errors {len(bad['error']):,}", flush=True)
    for r in bad["error"][:5]:
        print("  error:", r["path"], r["why"], flush=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, a.out)
    report(rows, len(files), a)
    print(f"done in {time.time() - t0:.0f}s  -> {a.out}")
    print("STEP_ALIGN_SCAN_DONE")


def report(rows, n_files, a):
    meas = [r for r in rows if r.get("verdict") in ("Ok", "Suspicious")]
    for r in meas:
        r["score"] = float(r["score"])
    sus = sorted([r for r in meas if r["verdict"] == "Suspicious"], key=lambda r: -r["score"])
    near = sorted([r for r in meas if r["verdict"] == "Ok" and r["score"] > 0.04], key=lambda r: -r["score"])
    print(f"\nscanned {n_files if n_files is not None else '?'}   timeline pages {len(rows):,}   measured {len(meas):,}")
    print(f"  SUSPICIOUS (score > {THRESHOLD})   {len(sus):,}")
    print(f"  OK_CLOSE (0.04 < score <= {THRESHOLD})   {len(near):,}")
    print(f"  OK   {len(meas) - len(sus) - len(near):,}")
    other = [r for r in rows if r.get("verdict") not in ("Ok", "Suspicious")]
    for why, n in Counter(r.get("why", "") for r in other).most_common():
        print(f"  NOT_MEASURED {n:,}  {why}")
    if meas:
        v = np.array([r["score"] for r in meas])
        print("  SCORE p50 {:.4f}  p99 {:.4f}  p99.9 {:.4f}  max {:.4f}".format(*np.percentile(v, [50, 99, 99.9]), v.max()))
    if sus:
        print("  suspicious by size:", ", ".join(f"{k} {n}" for k, n in Counter(f"{r['W']}x{r['H']}" for r in sus).most_common(12)))
        print("  suspicious also flagged by 112:", sum(1 for r in sus if r.get("v112") == "Suspicious"),
              " 112 Ok:", sum(1 for r in sus if r.get("v112") == "Ok"),
              " 112 cannot:", sum(1 for r in sus if r.get("v112") not in ("Suspicious", "Ok")))
        for r in sus[:15]:
            print(f"    {r['score']:.3f}  {r['W']}x{r['H']}  112 {r.get('v112', '')}  {os.path.basename(r['path'])}")
    if a.sheets:
        write_sheets(sus[:480], a.sheets, "suspicious")
        write_sheets(near[:96], a.sheets, "close")
        rest = [r for r in meas if r["verdict"] == "Ok" and r["score"] <= 0.04]
        write_sheets(rest[:: max(1, len(rest) // 24)][:24], a.sheets, "normal")
        print("sheets ->", a.sheets)
    if getattr(a, "copy_to", "") and sus:
        os.makedirs(a.copy_to, exist_ok=True)
        n = 0
        for i, r in enumerate(sus[:500], 1):
            try:
                shutil.copy2(r["path"], os.path.join(a.copy_to, f"S_{i:03d}_{r['score']:.3f}_{os.path.basename(r['path'])}"))
                n += 1
            except OSError as e:
                print("  copy failed:", r["path"], e)
        print(f"copied {n} suspicious originals -> {a.copy_to}")


if __name__ == "__main__":
    main()
