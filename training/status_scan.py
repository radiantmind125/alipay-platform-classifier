r"""支付宝结果页 / 账单详情页的"状态"不是成功的挑出来: 到账失败、资金原路返回、退款、交易关闭、提交成功(还没到账)、处理中、授权成功 ...

为什么(2026-10-15)
------------------
经理 10/14: "要把那种异常的挑出来 写代码试试吧"。
整页分类第一版在 1.8 万张随机图里看到: 蓝图库 5,949 张里有 8 张是"到账失败 资金将原路返回你的付款账户",
金额 1 万到 15 万, 却被归成了 支付成功(B3) —— 转账失败的截图被当成付款凭证传了上来。
第一版只认成功的标题(转账成功 / 到账成功 / 支付成功), 标题对不上就按下面的字段猜类别, 所以失败页混进了成功页。

怎么判(只读页头, 不跑整页分类, 快)
----------------------------------
蓝图(页头是支付宝蓝, 或者上半截蓝很多 = 通知条盖住了页头):
    1. 标题条: 和 page_type 一样 0~0.115 高度检测、识别, 长屏读不到再试 0~0.14; 还读不到就把框里上面那行拼音切掉再读
       (拼音页的汉字常读坏: 支付成功 -> 得勁); 通知条上的字不算
    2. 标题是 失败 / 退款 / 没到账 / 不是付款 的字(4 个字的失败标题允许错一个字) -> 那一类;
       标题是成功的(page_type.title_code, 认读错一个字的) -> OK; 汉字都读坏了看拼音(shibai yuanlu tuikuan tijiao ... chenggong)
    3. 标题读不到(通知条盖住): 再读到 0.34, 在 资金将原路返回 / 提交成功 / 处理中 这些字里找; 还找不到看有没有成功的标题
白图(页头白底): 读 0.05~0.40 那一段(金额下面那行状态): 先认 失败 / 退款(交易关闭 退款成功 ...), 再认 交易成功 一类,
    最后才认 处理中 一类 —— 而且先去掉处理进度时间轴的步骤名(提交银行处理 银行处理中 银行入账成功), 转账成功的页上也写着这几步
别的(深色、照片、页头是别的颜色): 不判, 记 SKIP

状态
----
    FAIL     失败、退回: 到账失败 转账失败 支付失败 原路返回 已退回 交易关闭 ...
    REFUND   退款: 退款成功 已退款 退款中 ...
    PENDING  还没到账: 提交成功 处理中 等待对方收款 预计..到账 ...
    NONPAY   不是付款: 授权成功 话费充值 充值成功 签约成功 ...
    OK       成功: 转账成功 到账成功 支付成功 付款成功 代付成功 交易成功 ...(代付成功 单独记在 kw 里)
    NONE     页头读到了字, 但没有任何状态字。★ 白底的账单详情大多本来就不写状态(1.8 万张里写了 交易成功 的只有 94 张),
             所以白底 NONE 是常态; 蓝底 NONE 多是拼音字体的标题读坏了(本地 300 张蓝图拼音页里 22 张)
    SKIP     不是蓝图也不是白图(深色、照片、页头别的颜色、图太小、读不出)

用法
----
    python training/status_scan.py --src E:\BlueImages D:\download2\BlueImages --out E:\_move_logs\status_blue --workers 20
    python training/status_scan.py --src E:\_move_logs\pagetype_census\page_type.csv --out E:\_move_logs\status_census --workers 20
★ --src 可以是目录(递归)、.txt 名单(一行一个路径)、.csv(有 path 列); 同一个 GWCZ 编号只算一次(和 page_type 一样)。
★ 输出 status.csv(每张一行)、summary.txt、sheets\<状态>_NN.png(联系表, 字画在图里)。
"""
from __future__ import annotations

import page_type as pt      # ★ 先 import: 它在 import numpy / cv2 之前把 OPENBLAS / OMP / MKL 设成 1 线程

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 顺序就是优先级: 一页上同时有 原路返回 和 处理中 的, 记 FAIL
STATUS_RE = [
    ("FAIL", re.compile(r"到账失败|转账失败|支付失败|付款失败|交易失败|提现失败|还款失败|充值失败|原路返回|原路退回|已退回|退回成功"
                        r"|已撤回|已撤销|交易关闭|已关闭")),
    ("REFUND", re.compile(r"退款成功|已退款|退款中|全额退款|部分退款|退款处理中|已全额退款")),
    ("PENDING", re.compile(r"提交成功|处理中|等待对方收款|等待对方领取|等待对方确认|待对方收款|待收款|待领取|等待付款|待支付"
                           r"|预计.{0,10}到账|正在处理")),
    ("NONPAY", re.compile(r"授权成功|话费充值|充值成功|签约成功|开通成功|绑定成功|解约成功")),
]
OK_WHITE = re.compile(r"交易成功|支付成功|付款成功|转账成功|到账成功|还款成功|收款成功|已到账|已收款|已完成")
STATUS_ORDER = ["FAIL", "REFUND", "PENDING", "NONPAY", "OK", "NONE", "SKIP"]
NAMES = {"FAIL": "失败退回", "REFUND": "退款", "PENDING": "还没到账", "NONPAY": "不是付款", "OK": "成功", "NONE": "没读到状态",
         "SKIP": "不判"}
WHITE_BAND = (0.05, 0.40)
# 账单详情(银行卡转账)的处理进度时间轴: 转账成功了也照样写着这几步, 不能当成"还在处理"
TIMELINE_STEPS = re.compile(r"处理进度|提交银行处理|银行处理中|银行入账成功|银行受理成功|银行已受理")
KEYS = ["path", "W", "H", "header", "status", "name", "kw", "title", "ocr", "secs", "src"]
FAIL_TITLES = ["到账失败", "转账失败", "支付失败", "付款失败", "交易失败", "提现失败", "还款失败"]
# 拼音页的拼音(小写、连在一起): 失败 shibai、原路 yuanlu、退款 tuikuan、提交成功 tijiaochenggong、处理中 chulizhong、
# 等待对方 dengdaiduifang、授权成功 shouquanchenggong、充值成功 chongzhichenggong
# ★ 不能只认 chongzhi: 成功页上常有 充值金+15 的奖励卡片(chongzhijin); tijiao 也只认 提交成功
PY_STATUS = [("FAIL", re.compile(r"shibai|yuanlu")), ("REFUND", re.compile(r"tuikuan")),
             ("PENDING", re.compile(r"tijiaoche|chulizhong|dengdaidui")),
             ("NONPAY", re.compile(r"shouquanche|chongzhiche"))]


def find_status(text: str):
    """在一段字(norm 以后)里找状态字, 返回 (状态, 关键字); 没有返回 ("", "")。"""
    for st, rx in STATUS_RE:
        m = rx.search(text)
        if m:
            return st, m.group()
    return "", ""


def text_status(t: str):
    """标题 / 页头的汉字: 先原样找状态字; 4 个字的失败标题允许错 1 个字(拼音字体常读错一个字, 同 page_type.has)。
    ★ 到账失败 和 到账成功 差两个字, 不会混。"""
    st, kw = find_status(t)
    if st:
        return st, kw
    for k in FAIL_TITLES:
        if pt.has(t, k):
            return "FAIL", k + "(错一个字)"
    return "", ""


def latin_status(lat: str):
    """拼音页: 汉字常读坏, 但上面那行拼音读得出来(dao zhang shi bai)。"""
    for st, rx in PY_STATUS:
        m = rx.search(lat)
        if m:
            return st, "拼音 " + m.group()
    if pt.pinyin_code(lat):
        return "OK", "拼音 chenggong"
    return "", ""


def scan(path: str) -> dict:
    out = {"path": path, "W": "", "H": "", "header": "", "status": "SKIP", "kw": "", "title": "", "ocr": ""}
    bgr = pt.decode(path)
    if bgr is None:
        out["kw"] = "读不出"
        return out
    H, W = bgr.shape[:2]
    out.update(W=W, H=H)
    if W < 300 or H < 400:
        out["kw"] = "图太小"
        return out
    small = cv2.resize(bgr, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    head = pt.header_kind(small)
    gs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dark = np.median(gs[int(gs.shape[0] * 0.095):int(gs.shape[0] * 0.96)]) < pt.DARK_MEDIAN
    btop = pt.blue_top(small)
    out["header"] = head
    if dark:
        out["kw"] = "深色"
        return out
    if head == "蓝底" or btop >= pt.BLUE_TOP:
        return _blue(out, bgr)
    if head == "白底":
        return _white(out, bgr)
    out["kw"] = "页头 " + head
    return out


def _title(bgr, y1: float, pinyin: bool):
    """读一条标题条(0~y1), 返回 (Reader, 标题区的字, 标题区的拼音)。"""
    rd = pt.Reader(bgr, pinyin)
    H, W = rd.H, rd.W
    in_title = lambda b: (b[1] + b[3]) / 2 < pt.TITLE_Y * H and b[0] <= 0.75 * W and not pt.is_banner(b)   # noqa: E731
    rd.detect(0.0, y1)
    rd.read(lambda b: True)
    return rd, rd.joined(in_title), rd.latin(in_title)


def _decide(t: str, lat: str):
    """标题区的字 + 拼音 -> (状态, 关键字); 定不了返回 ("", "")。不是成功的字优先于成功的字。"""
    st, kw = text_status(t)
    if st:
        return st, kw
    code, ok_kw = pt.title_code(t)
    if code:
        return "OK", ok_kw
    return latin_status(lat)


def _blue(out, bgr):
    H, W = bgr.shape[:2]
    not_banner = lambda b: not pt.is_banner(b)     # noqa: E731
    tries = [(pt.TITLE_Y, False)]
    if pt.title_strip(H, W) > pt.TITLE_Y:
        tries.append((pt.title_strip(H, W), False))     # 长屏: 0~0.14 的条再试一次(同 page_type._blue)
    tries.append((pt.title_strip(H, W), True))          # 拼音页: 框里切掉上面那行拼音再读(page_type.cut_pinyin)
    first = None
    for y1, py in tries:
        rd, t, lat = _title(bgr, y1, py)
        first = first or (rd, t)
        st, kw = _decide(t, lat)
        if st:
            out.update(status=st, kw=kw + ("(切拼音)" if py else ""), title=t[:20],
                       ocr=f"[{len(rd.text)}] " + rd.joined(not_banner)[:160])
            return out
    # 标题没读到(通知条盖住了, 或者标题字读坏了): 再读下面一段(到 0.34), 找状态字。
    # 先用不切拼音的 Reader(切拼音会把普通页的字从中间切开), 找不到再用切过拼音的那个
    for rdb, y0 in ((first[0], pt.TITLE_Y), (rd, pt.title_strip(H, W))):
        rdb.detect(y0, pt.BLUE_FRAC)
        rdb.read(lambda b: True)
        txt = rdb.joined(not_banner)
        st, kw = text_status(txt)
        if not st:
            st, kw = latin_status(rdb.latin(not_banner))
            if st == "OK":
                st, kw = "", ""                         # 下面一段的 chenggong 可能是别的(红包领取成功), 不算标题
        if st:
            out.update(status=st, kw=kw + "(标题下面)", title=first[1][:20], ocr=f"[{len(rdb.text)}] " + txt[:160])
            return out
        code, ok_kw = pt.title_code(txt)
        if code:
            out.update(status="OK", kw=ok_kw + "(标题下面)", title=first[1][:20], ocr=f"[{len(rdb.text)}] " + txt[:160])
            return out
    out.update(status="NONE", kw="", title=first[1][:20], ocr=f"[{len(first[0].text)}] " + first[0].joined(not_banner)[:160])
    return out


def _white(out, bgr):
    rd = pt.Reader(bgr)
    rd.detect(*WHITE_BAND)
    rd.read(lambda b: True)
    txt = rd.joined(lambda b: not pt.is_banner(b))
    out["ocr"] = f"[{len(rd.text)}] " + txt[:160]
    out["title"] = rd.first_cjk()[:20]
    st, kw = find_status(txt)
    if st in ("FAIL", "REFUND"):                    # 失败 / 退款 的字最先认
        out.update(status=st, kw=kw)
        return out
    m = OK_WHITE.search(txt)
    if m:                                           # 有 交易成功 一类就是成功, 不管时间轴上写着 银行处理中
        out.update(status="OK", kw=m.group())
        return out
    st, kw = find_status(TIMELINE_STEPS.sub("", txt))
    if st:
        out.update(status=st, kw=kw)
        return out
    out.update(status="NONE" if txt else "SKIP", kw="" if txt else "白底没读到字")
    return out


def work(task):
    path, src = task
    t0 = time.time()
    try:
        r = scan(path)
    except Exception as e:  # noqa: BLE001
        r = {"path": path, "status": "SKIP", "kw": f"出错 {type(e).__name__}"}
    r["name"] = NAMES.get(r["status"], "")
    r["secs"] = round(time.time() - t0, 3)
    r["src"] = src
    return r


def summary(rows, wall) -> list[str]:
    L = ["=" * 78, f"  页面状态 status_scan   {time.strftime('%Y-%m-%d %H:%M')}", "=" * 78]
    s = np.array([float(r.get("secs") or 0) for r in rows]) if rows else np.zeros(1)
    L.append(f"  图 {len(rows):,} 张   用时 {wall:.0f} 秒   {len(rows) / max(wall, 1e-9):.1f} 张/秒   每张(单进程) 平均 {s.mean():.2f} 秒")
    L.append("")
    L.append("  ---- 状态 x 页头 ----")
    heads = sorted({r.get("header", "") for r in rows})
    L.append("  " + f"{'':<16}" + "".join(f"{h or '-':>8}" for h in heads) + f"{'合计':>9}")
    for st in STATUS_ORDER:
        rs = [r for r in rows if r.get("status") == st]
        if rs:
            c = Counter(r.get("header", "") for r in rs)
            L.append(f"  {st:<8}{NAMES[st]:<8}" + "".join(f"{c[h]:>8,}" for h in heads) + f"{len(rs):>9,}")
    L.append("")
    L.append("  ---- 不是成功的: 关键字 ----")
    for st in STATUS_ORDER[:4]:
        c = Counter(r.get("kw", "") for r in rows if r.get("status") == st)
        if c:
            L.append(f"  {st} {NAMES[st]}: " + "  ".join(f"{k} {v:,}" for k, v in c.most_common(12)))
    c = Counter(r.get("kw", "") for r in rows if r.get("status") == "OK")
    L.append(f"  OK 成功: " + "  ".join(f"{k} {v:,}" for k, v in c.most_common(10)))
    c = Counter(r.get("kw", "") for r in rows if r.get("status") == "SKIP")
    L.append(f"  SKIP 不判: " + "  ".join(f"{k} {v:,}" for k, v in c.most_common(8)))
    L.append("  (白底 NONE = 页头没有状态字, 账单详情多数不写状态, 是常态; 蓝底 NONE = 标题没读出来, 多是拼音字体)")
    L.append("=" * 78)
    return L


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="页面状态: 挑出不是成功的(失败退回 / 退款 / 还没到账 / 不是付款)")
    ap.add_argument("--src", nargs="+", required=True, help="目录(递归)或名单(.txt 一行一个路径 / .csv 有 path 列)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="随机抽这么多张(0 = 全部)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--sheets", type=int, default=3, help="每个不是成功的状态最多出几张联系表(每张 12 格); 成功的只出 1 张抽查")
    a = ap.parse_args()

    todo = pt.collect(a.src)
    todo, dups = pt.dedupe_gwcz(todo)
    if a.n and a.n < len(todo):
        random.Random(a.seed).shuffle(todo)
        todo = todo[:a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"  图 {len(todo):,} 张   进程 {a.workers}   按 GWCZ 编号去重跳过 {dups:,} 张", flush=True)
    t0 = time.time()
    rows = []
    step = max(500, len(todo) // 40)
    with ProcessPoolExecutor(a.workers, initializer=pt._init) as pool:
        for i, r in enumerate(pool.map(work, todo, chunksize=8), 1):
            rows.append(r)
            if i % step == 0 or i == len(todo):
                el = time.time() - t0
                print(f"    {i:,}/{len(todo):,}   {i / max(el, 1e-9):.1f} 张/秒   "
                      + " ".join(f"{k}:{v:,}" for k, v in sorted(Counter(x['status'] for x in rows).items())), flush=True)
        wall = time.time() - t0
        tmp = a.out / "status.csv.tmp"
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, a.out / "status.csv")
        lines = summary(rows, wall)
        (a.out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        if a.sheets:
            sd = a.out / "sheets"
            sd.mkdir(exist_ok=True)
            jobs = []
            for st in STATUS_ORDER:
                rs = [r for r in rows if r.get("status") == st and r.get("W")]
                random.Random(a.seed).shuffle(rs)
                rs = rs[:12 * (a.sheets if st not in ("OK", "SKIP") else 1)]
                for j in range(0, len(rs), 12):
                    part = [dict(r, _i=i, cls=st, sub=r.get("kw", ""), reason=r.get("ocr", ""), pinyin="")
                            for i, r in enumerate(rs[j:j + 12], 1)]
                    jobs.append((part, str(sd / f"{st}_{j // 12 + 1:02d}.png")))
            for _dst in pool.map(pt.make_sheet, jobs):
                pass
            print(f"  联系表 {len(jobs)} 张 -> {sd}", flush=True)
    print(f"  -> {a.out / 'status.csv'}", flush=True)


if __name__ == "__main__":
    main()
