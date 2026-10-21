"""整页分类快速版(只看图, 不做 OCR): 规则 + 小 CNN(ONNX), 给整个图库按类型出清单。

page_type.py(规则 + OCR)一张 3 ~ 5 秒, 9 月 206 万张要 4 ~ 5 天; 这个只看图, 一张一百多毫秒。
每张图(前半段和 page_type.classify 一模一样, 用的就是 page_type / status_icon 里的函数):
  解码(截断的 JPEG 补 FFD9) -> 太小 U -> 1304 宽的凭证原图 W2 -> 翻拍照片 P -> 拼音标记 -> 页头异常图标(status_icon)
  -> 浮窗 X -> 蓝图异常图标 B4 -> 小 CNN(page_cnn.py 训的 head 视图, ONNX Runtime CPU 单线程)按门槛定类别, 把握不够记 U。
  CNN 每张都跑(规则定了的也跑, 记在 pred / p 里, 方便查规则和模型对不对得上)。
门槛(policy json): {"floor": 0.7, "thresholds": {"B1": 0.7, ...}}, 某类门槛 > 1 = 模型不判这一类。
  原始的 pred / p / top2 / p2 都留在 csv 里, 门槛改了只要重跑 lists, 不用重新扫图。

用法:
  python training/page_fast.py run   --src 目录或名单 [...] --onnx page_head.onnx --policy policy.json --out 输出目录
                                     [--workers 20] [--resume] [--limit N]
  python training/page_fast.py lists --csv 输出目录\\page_fast.csv --policy policy.json --out 清单目录
  python training/page_fast.py compare --csv page_fast.csv --labels page_type.csv [--set census]   (和 OCR 版的标签比)
  python training/page_fast.py folders --csv 蓝图库\\page_fast.csv 白图库\\page_fast.csv --policy policy.json
                                     [--lists 蓝图库清单目录 白图库清单目录] [--classes B2 W1] [--mode link|copy --copy-to 目录] [--dry-run]
                                     (每一类的原图放进一个文件夹; 默认硬链接, 不占硬盘)
输出: page_fast.csv(一张一行), summary.txt; lists: 每类一个 csv(GWCZ编号, 上传时间, 图库, 类别, 怎么定的, 把握, 拼音, 异常页) + summary.txt
★ 只读原图, 不改不挪(folders 只在别的文件夹里建硬链接或拷副本)。
"""
from __future__ import annotations

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import page_type as pt  # noqa: E402
import status_icon  # noqa: E402
from page_cnn_data import gid, head_view, split_of  # noqa: E402

CLASSES = ("B1", "B2", "B3", "B4", "W1", "W2", "W3", "W4", "D", "X")   # 和 page_cnn.CLASSES 一样的顺序(不 import torch)
BLUE_ABNORMAL = ("到账失败", "提交成功", "等待对方领取")
KEYS = ["path", "gid", "W", "H", "device", "header", "dark", "pinyin", "state", "rule", "pred", "p", "top2", "p2", "cls", "why", "ms"]
TS = re.compile(r"_(\d{8})(\d{6})\.")
_SESS = None
_POLICY = None
_STATE = None          # 路径 -> 异常页种类(status_icon 以前全库扫过的结果); None = 现场算


def decide(pred: str, p: float, policy: dict) -> str:
    th = policy["thresholds"].get(pred, 1.01)
    return pred if p >= max(policy.get("floor", 0.0), th) else "U"


def _init(onnx_path: str, policy: dict, state) -> None:
    global _SESS, _POLICY, _STATE
    import onnxruntime as ort
    cv2.setNumThreads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
    _SESS = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    _POLICY = policy
    _STATE = state


def load_state(files: list) -> dict:
    """status_icon 全库扫过的结果(icon.csv, 只写了不是成功页的那些): 判 Suspicious 的 路径 -> 种类。
    ★ 只有 status_icon.py 和那次扫描是同一版时才能这样用(9a3c862 以后没改过), 不然现场算(不给 --state-csv)。"""
    st = {}
    for fn in files:
        with open(fn, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("verdict") == "Suspicious":
                    st[r["path"]] = r["kind"]
    return st


def rules(bgr, out: dict) -> str:
    """只看图的规则(和 page_type.classify 前半段一样), 定得下来的返回类别, 定不下来返回 ""。"""
    H, W = bgr.shape[:2]
    if W == pt.VOUCHER_W and pt.VOUCHER_H[0] <= H <= pt.VOUCHER_H[1]:
        out["why"] = f"尺寸 {W}x{H}: 支付宝生成的转账凭证图"
        return "W2"
    cue = pt.size_cue(W, H)
    small = cv2.resize(bgr, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    head = pt.header_kind(small)
    gs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dark = int(np.median(gs[int(gs.shape[0] * 0.095):int(gs.shape[0] * 0.96)]) < pt.DARK_MEDIAN)
    btop = pt.blue_top(small)
    out.update(header=head, dark=dark)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mode = pt.mode_share(gray)
    qc = []

    def qf() -> float:
        # 平整度(16x16 小块标准差)一张要十几毫秒, 只在判照片的条件真要用到它时才算: 结果和 page_type 一模一样
        if not qc:
            qc.append(pt.flatness(gray))
        return qc[0]

    photo = ""
    if cue in ("横图", "相机尺寸"):
        if float((gray > pt.PHOTO_MIN_GRAY).mean()) >= 0.15:
            q2, mode2 = pt.flatness(gray, pt.PHOTO_MIN_GRAY), pt.mode_share(gray, pt.PHOTO_MIN_GRAY)
        else:
            q2, mode2 = qf(), mode
    if W * H >= pt.PHOTO_MP:
        photo = f"{W}x{H} 有 {W * H / 1e6:.1f} 百万像素"
    elif cue in ("横图", "相机尺寸") and (q2 >= pt.PHOTO_Q or mode2 < pt.PHOTO_MODE):
        photo = f"{cue}, 亮处平整度 {q2:.2f} 最多灰度占 {mode2:.3f}"
    elif cue == "长宽比怪" and (mode < pt.PHOTO_MODE or qf() >= pt.PHOTO_Q_ODD):
        photo = f"长宽比 {H / W:.2f}, 最多灰度占 {mode:.3f}"
    elif head != "蓝底" and mode < pt.PHOTO_MODE_FREE and qf() >= pt.PHOTO_Q_FREE:
        photo = f"底色有噪点(平整度 {qf():.2f}), 最多灰度只占 {mode:.3f}"
    if photo:
        out["why"] = "像照片: " + photo
        return "P"
    if (W, H) not in pt.IPHONE_RESOLUTIONS:
        out["pinyin"] = pt.pinyin_flag(gray, mode)
    if _STATE is not None:
        out["state"] = _STATE.get(out["path"], "")
    else:
        st = status_icon.check(bgr)
        if st["verdict"] == "Suspicious":
            out["state"] = st["kind"]
    if not dark and head != "蓝底" and btop >= pt.BLUE_TOP and pt.inset_window(small):
        out["why"] = "页头被盖住, 蓝色结果页是浮窗"
        return "X"
    is_blue = not dark and (head == "蓝底" or btop >= pt.BLUE_TOP_ROUTE)
    if is_blue and pt.inset_window(small):
        out["why"] = "蓝色结果页是浮窗(两边不是蓝的)"
        return "X"
    if is_blue and out["state"] in BLUE_ABNORMAL:      # 和 page_type 一样: 只有走蓝图那一路的才定 B4
        out["why"] = f"页头图标是 {out['state']}"
        return "B4"
    return ""


def work(path: str) -> dict:
    t0 = time.perf_counter()
    out = {k: "" for k in KEYS}
    out.update(path=path, gid=gid(path), cls="U")
    try:
        bgr = pt.decode(path)
        if bgr is None:
            out["why"] = "读不出"
            return out
        H, W = bgr.shape[:2]
        out.update(W=W, H=H, device=pt.DEVICE.get(pt.resolution_platform(W, H), "说不清"))
        if W < 300 or H < 400:
            out["why"] = "图太小"
            return out
        rl = rules(bgr, out)
        x = np.ascontiguousarray(head_view(bgr))[None]
        prob = _SESS.run(None, {"image": x})[0][0]
        o = np.argsort(-prob)
        out.update(pred=CLASSES[o[0]], p=f"{prob[o[0]]:.4f}", top2=CLASSES[o[1]], p2=f"{prob[o[1]]:.4f}")
        if rl:
            out.update(rule=rl, cls=rl)
        else:
            # ★ 按存下来的(4 位小数)定, 和 lists 重新定的结果一样
            out["cls"] = decide(out["pred"], float(out["p"]), _POLICY)
            if not out["why"]:
                out["why"] = f"模型 {out['pred']} {out['p']}" + ("" if out["cls"] != "U" else ", 把握不够")
    except Exception as e:  # noqa: BLE001
        out.update(cls="U", why="出错 " + repr(e)[:80])
    out["ms"] = int((time.perf_counter() - t0) * 1000)
    return out


def read_done(fcsv: Path) -> set:
    """接着跑: 先把文件截到最后一个完整的行(被硬关掉时最后一行常是半截, 甚至半个汉字), 再读; 列数不对就停。"""
    raw = fcsv.read_bytes()
    cut = raw.rfind(b"\n") + 1
    if cut < len(raw):
        with fcsv.open("r+b") as f:
            f.truncate(cut)
        print(f"  接着跑: 去掉最后半行 {len(raw) - cut} 字节", flush=True)
    done = set()
    with fcsv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if None in r or any(v is None for v in r.values()):
                print(f"  ★ {fcsv} 里有列数不对的行(在 {r.get('path')} 附近), 不接着跑, 先看看", flush=True)
                sys.exit(2)
            done.add(r["path"])
    return done


def run(a) -> None:
    policy = json.load(open(a.policy, encoding="utf-8"))
    state = load_state(a.state_csv) if a.state_csv else None
    todo = pt.collect(a.src)
    todo, dups = pt.dedupe_gwcz(todo)
    paths = [p for p, _t in todo]
    a.out.mkdir(parents=True, exist_ok=True)
    fcsv = a.out / "page_fast.csv"
    done = set()
    if fcsv.exists():
        if not a.resume:
            print(f"  ★ {fcsv} 已经有了, 要接着跑加 --resume, 要重跑先删掉", flush=True)
            sys.exit(2)
        done = read_done(fcsv)
        paths = [p for p in paths if p not in done]
    if a.limit:
        paths = paths[:a.limit]
    (a.out / "dups.txt").write_text(f"{dups}\n", encoding="utf-8")
    print(f"  图 {len(paths):,} 张(已做 {len(done):,}, 同一 GWCZ 重复跳过 {dups:,})  进程 {a.workers}  模型 {a.onnx}  "
          f"异常页标记 {'用 ' + str(len(state)) + ' 条以前扫的结果' if state is not None else '现场算'}", flush=True)
    new = not fcsv.exists()
    # ★ 一行一 flush(buffering=1): 半夜被关掉, 最多丢最后半行, 接着跑时截掉
    f = fcsv.open("w" if new else "a", encoding="utf-8-sig" if new else "utf-8", newline="", buffering=1)
    w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore", lineterminator="\n")
    if new:
        w.writeheader()
    t0, cnt = time.time(), Counter()
    step = max(2000, len(paths) // 50)
    try:
        # ★ ProcessPoolExecutor: 子进程死了会报错退出, 不会像 multiprocessing.Pool 那样一直等
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(a.onnx), policy, state)) as ex:
            for i, r in enumerate(ex.map(work, paths, chunksize=32), 1):
                w.writerow(r)
                cnt[r["cls"]] += 1
                if i % step == 0 or i == len(paths):
                    el = time.time() - t0
                    print(f"    {i:,}/{len(paths):,}  {i / max(el, 1e-9):.1f} 张/秒  "
                          + " ".join(f"{k}:{v}" for k, v in sorted(cnt.items())), flush=True)
    finally:
        f.close()
    print(f"  用时 {time.time() - t0:.0f} 秒 -> {fcsv}", flush=True)
    write_lists(fcsv, policy, a.out / "lists", dups)


def upload_time(path: str) -> str:
    m = TS.search(os.path.basename(path))
    if not m:
        return ""
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"


def iter_final(fcsv: Path, policy: dict, broken: list):
    """一行一行读 page_fast.csv, 按门槛(policy)定最后的类别(规则定的不看门槛)。列数不对的行跳过, 记在 broken[0]。
    lists 和 folders 都用这一个, 两边的类别保证一样。"""
    with fcsv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if None in r or any(v is None for v in r.values()):
                broken[0] += 1
                continue
            if not r["rule"] and r["pred"]:
                r["cls"] = decide(r["pred"], float(r["p"]), policy)
            yield r


def load_final(fcsv: Path, policy: dict) -> tuple:
    """返回 (全部行, 列数不对跳过的行数)。"""
    broken = [0]
    rows = list(iter_final(fcsv, policy, broken))
    return rows, broken[0]


def write_lists(fcsv: Path, policy: dict, out: Path, dups: int = -1) -> None:
    """按门槛(policy)重新定类别, 每类一个清单 + 汇总。规则定的不看门槛。
    ★ 先删掉 out 里旧的清单和汇总: 换了门槛重出时, 变成 0 张的类别不能留着旧文件。列数不对的行(半截)跳过、记数。"""
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.csv")) + [out / "summary.txt"]:
        if old.exists():
            old.unlink()
    if dups < 0 and (fcsv.parent / "dups.txt").exists():
        dups = int((fcsv.parent / "dups.txt").read_text(encoding="utf-8").strip() or 0)
    rows, broken = load_final(fcsv, policy)
    by = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    # 给人看的清单: 不带服务器路径(GWCZ 编号 + 图库就找得到), 文件小; 全部字段在 page_fast.csv 里
    head = ["GWCZ编号", "上传时间", "图库", "类别", "怎么定的", "把握", "拼音", "异常页"]
    for c, rs in by.items():
        with (out / f"{c}_{pt.NAMES.get(c, c)}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            for r in rs:
                how = "规则" if r["rule"] else ("模型" if r["pred"] and c != "U" else "")
                w.writerow([r["gid"], upload_time(r["path"]), Path(r["path"]).parent.name, c, how,
                            r["p"] if how == "模型" else "", "是" if str(r["pinyin"]) == "1" else "", r["state"]])
    n = len(rows)
    weak = set(policy.get("provisional", []))
    L = [f"一共 {n:,} 张" + (f"(同一 GWCZ 编号重复的 {dups:,} 张只分了第一张, 没算在里面)" if dups > 0 else "")
         + (f"; 列数不对跳过 {broken} 行" if broken else ""),
         f"门槛 {json.dumps(policy.get('thresholds', {}), ensure_ascii=False)}  下限 {policy.get('floor', 0)}", ""]
    L.append("  类别            张数     占比    规则定的   模型定的   拼音")
    for c in ("B1", "B2", "B3", "B4", "W1", "W2", "W3", "W4", "D", "X", "P", "U"):
        rs = by.get(c, [])
        L.append(f"  {c:2} {pt.NAMES.get(c, c):8} {len(rs):>10,} {len(rs) / max(n, 1):7.2%} {sum(1 for r in rs if r['rule']):>10,} "
                 f"{sum(1 for r in rs if not r['rule'] and r['pred'] and c != 'U'):>10,} {sum(1 for r in rs if str(r['pinyin']) == '1'):>7,}"
                 + ("   ★ 模型的门槛是暂定的, 还在人眼核" if c in weak else ""))
    u = Counter(r["why"].split(",")[0] if r["cls"] == "U" and not r["pred"] else (f"模型最像 {r['pred']}" if r["cls"] == "U" else "")
                for r in rows)
    L.append("")
    L.append("  记 U 的原因: " + "  ".join(f"{k} {v:,}" for k, v in u.most_common() if k))
    st = Counter(r["state"] for r in rows if r["state"])
    L.append("  异常页标记(status_icon): " + "  ".join(f"{k} {v:,}" for k, v in st.most_common()))
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def compare(a) -> None:
    """和 OCR 版(page_type)的标签比: 只看两边都有的图。--set census 时另外单算 page_cnn 分出来的 census test 那 30%。"""
    lab = {}
    with open(a.labels, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            lab[r["path"]] = r["cls"]
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["path"] in lab]
    L = []
    for name, rs in (("全部", rows), ("census test(模型没训过的那 30%)", [r for r in rows if split_of(r["gid"], "census") == "test"])):
        if a.set != "census" and name != "全部":
            continue
        judged = [r for r in rs if r["cls"] != "U"]
        err = [r for r in judged if r["cls"] != lab[r["path"]]]
        L.append(f"==== {name}: {len(rs):,} 张, 判出 {len(judged):,}({len(judged) / max(len(rs), 1):.1%}), "
                 f"和 OCR 版不一样 {len(err):,}({len(err) / max(len(judged), 1):.2%})")
        L.append("  不一样的(OCR 版 -> 快速版): " + "  ".join(f"{k[0]}>{k[1]} {v}" for k, v in
                                                       Counter((lab[r['path']], r['cls']) for r in err).most_common(20)))
        L.append("  快速版记 U 的(OCR 版是什么): " + "  ".join(f"{k} {v}" for k, v in
                                                     Counter(lab[r['path']] for r in rs if r['cls'] == 'U').most_common()))
        ms = [int(r["ms"]) for r in rs if r["ms"]]
        if ms:
            L.append(f"  每张单进程 平均 {np.mean(ms):.0f} 毫秒  中位 {np.median(ms):.0f}  p90 {np.percentile(ms, 90):.0f}")
        L.append("")
    bad_parity = False
    if a.preds:
        # 门槛是在训练缓存(缩略图存成 JPEG q95)上量的, 这里是原图现做的 head 视图 + ONNX: 同一张图两边的结果要对得上
        with open(a.preds, encoding="utf-8-sig", newline="") as f:
            ev = {r["path"]: r for r in csv.DictReader(f)}
        pol = json.load(open(a.policy, encoding="utf-8")) if a.policy else None
        both = [(r, ev[r["path"]]) for r in rows if r["path"] in ev and r["pred"]]
        if both:
            agree = sum(r["pred"] == e["pred"] for r, e in both) / len(both)
            dp = np.array([abs(float(r["p"]) - float(e["p"])) if r["pred"] == e["pred"] else 1.0 for r, e in both])
            L.append(f"==== 对拍: 现场(原图 + ONNX) vs 训练评估(缓存缩略图 + torch), 同一批 {len(both):,} 张")
            L.append(f"  最大那类一样 {agree:.2%};  把握差 中位 {np.median(dp):.4f}  p99 {np.percentile(dp, 99):.4f}")
            if pol:
                flip = sum(decide(r["pred"], float(r["p"]), pol) != decide(e["pred"], float(e["p"]), pol)
                           for r, e in both if not r["rule"])
                L.append(f"  按门槛的判定变了 {flip} 张(规则没定的那些)")
            if agree < 0.995:
                L.append("  ★ 对不上(< 99.5%), 先别跑全库")
                bad_parity = True
            L.append("")
    Path(a.csv).with_name("compare.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    with Path(a.csv).with_name("compare_diff.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "ocr", "fast", "rule", "pred", "p", "why"])
        for r in rows:
            if r["cls"] != "U" and r["cls"] != lab[r["path"]]:
                w.writerow([r["path"], lab[r["path"]], r["cls"], r["rule"], r["pred"], r["p"], r["why"]])
    if bad_parity:
        sys.exit(5)


def pick(a) -> None:
    """从 page_cnn 评估的 preds.csv 里挑要人看的图, 原图拷出来(只拷不动原图):
    规则定不下来、标签是 U / B0(OCR 版说不清)但模型把握 >= 0.9 判了一类的; 模型过了门槛还和标签不一样的(把握 >= 0.7)。"""
    import shutil
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.preds, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("val", "test", "test_py") and not r["rule"]]
    sel = []
    for r in rows:
        p = float(r["p"])
        if r["label"] in ("U", "B0") and p >= 0.9:
            sel.append((r, "标签说不清_模型判了"))
        elif r["label"] != r["pred"] and p >= 0.7 and r["label"] not in ("P",):
            sel.append((r, "和标签不一样"))
    with (a.out / "pick.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "why", "split", "label", "pred", "p", "top2", "p2", "path"])
        n = 0
        for r, why in sel:
            dst = f"{r['split']}_{r['label']}_to_{r['pred']}_{os.path.basename(r['path'])}"
            try:
                shutil.copy2(r["path"], a.out / dst)
                n += 1
            except OSError:
                dst = "拷不了"
            w.writerow([dst, why, r["split"], r["label"], r["pred"], r["p"], r["top2"], r["p2"], r["path"]])
    print(f"  挑出 {len(sel)} 张, 拷了 {n} 张 -> {a.out}", flush=True)


README_LINK = """9 月图库按类别放好的原图(和 9月图库分类清单 一样的分法)
- 蓝图库在 E 盘, 白图库在 D 盘(硬链接只能和原图在同一个盘): 盘符:\\{root}\\图库名\\类别\\
- 张数多的类别({over:,} 张以上)按上传日期再分一层
- 这里的每个文件都是原图的硬链接: 不占硬盘, 和原图是同一个文件
  * 删这里的文件没关系, 原图还在; 整个文件夹删掉就恢复原样
  * 不要在这里改图, 改了原图也跟着变
  * 看文件夹属性时大小会算上, 其实没多占硬盘
  * 原图删了以后, 只要这里还有链接, 硬盘空间就不会放出来; 要腾硬盘时这里也要一起删
- 清单数.csv: 每个文件夹几张
"""
README_COPY = """9 月图库按类别拷出来的原图(和 9月图库分类清单 一样的分法)
- 图库名\\类别\\, 张数多的类别({over:,} 张以上)按上传日期再分一层
- 这些是拷出来的副本, 改了删了都不影响原图
- 清单数.csv: 每个文件夹几张
"""


def folders(a) -> None:
    """按清单把每一类的原图放进一个文件夹, 给人一类一类地看、拷。类别和 lists 完全一样(同一个 iter_final + 同一个门槛)。
    link(默认): 硬链接, 不占硬盘, 原图不动。硬链接只能建在原图同一个盘上 -> 根目录 = 原图所在盘:\\{root-name}\\图库名\\类别\\[日期]\\
    copy: 拷一份到 --copy-to\\{root-name}\\... (先算空间够不够)。
    已经有的跳过: 中途停了再跑一遍就接着做。最后数每个类别文件夹的张数对不对, 每类抽 30 张核是不是同一个文件。
    ★ 省内存(两个图库 206 万行): 只留 (路径, 类别), 一类一类地做。"""
    import random
    import shutil
    policy = json.load(open(a.policy, encoding="utf-8"))
    if a.lists and len(a.lists) != len(a.csv):
        print("  ★ --lists 要和 --csv 一样多, 一一对应", flush=True)
        sys.exit(2)
    want = set(a.classes) if a.classes else None
    groups = defaultdict(list)
    for k, fcsv in enumerate(a.csv):
        broken, got = [0], Counter()
        for r in iter_final(fcsv, policy, broken):
            got[r["cls"]] += 1
            if want is None or r["cls"] in want:
                groups[(Path(r["path"]).parent.name, r["cls"])].append(r["path"])
        if broken[0]:
            print(f"  ★ {fcsv} 里有 {broken[0]} 行列数不对, 先看看", flush=True)
            sys.exit(2)
        if a.lists:
            # ★ 和交出去的清单对一遍: 每类张数一样才做(门槛文件或 page_fast.csv 不是那一版就停)
            lst = Counter()
            for fn in Path(a.lists[k]).glob("*.csv"):
                with fn.open(encoding="utf-8-sig", newline="") as f:
                    lst[fn.name.split("_")[0]] += sum(1 for _ in f) - 1
            if got != lst:
                print(f"  ★ 和清单 {a.lists[k]} 对不上: 这里 {dict(sorted(got.items()))} 清单 {dict(sorted(lst.items()))}", flush=True)
                sys.exit(4)
            print(f"  和清单对上了: {fcsv.parent.name} {sum(got.values()):,} 张", flush=True)
    if not groups:
        print("  ★ 没有要放的图(--classes 写错了?)", flush=True)
        sys.exit(2)
    info, roots = [], {}
    for (pool, c), ps in sorted(groups.items()):
        drives = {Path(p).drive for p in ps}
        if a.mode == "link" and len(drives) != 1:
            print(f"  ★ {pool} 的图不在同一个盘上: {sorted(drives)}", flush=True)
            sys.exit(2)
        base = Path(a.copy_to) if a.mode == "copy" else Path(drives.pop() + "\\")
        root = base / a.root_name
        roots[str(root)] = root
        info.append((pool, c, ps, len(ps) > a.by_day_over, str(root / pool / f"{c}_{pt.NAMES.get(c, c)}"), str(root)))
    total = sum(len(ps) for _, _, ps, _, _, _ in info)
    print(f"  要放 {total:,} 张, {a.mode}, 根目录 {', '.join(sorted(roots))}", flush=True)
    for pool, c, ps, byday, cf, root in info:
        names = {os.path.basename(p) for p in ps}
        if len(names) != len(ps):
            print(f"  ★ {pool} {c} 里有同名的图, 不做", flush=True)
            sys.exit(2)
        print(f"    {pool:14} {c:2} {pt.NAMES.get(c, c):8} {len(ps):>10,}" + ("  按日期分" if byday else ""), flush=True)
    if a.mode == "copy":
        need = sum(os.path.getsize(p) for _, _, ps, _, _, _ in info for p in ps if os.path.exists(p))
        free = shutil.disk_usage(a.copy_to).free
        print(f"  要拷 {need / 2**30:.1f} GB, {a.copy_to} 剩 {free / 2**30:.1f} GB", flush=True)
        if need * 1.05 > free:
            print("  ★ 空间不够, 不拷", flush=True)
            sys.exit(5)
    if a.dry_run:
        print("  (只看不做)", flush=True)
        return
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
        (root / "说明.txt").write_text((README_COPY if a.mode == "copy" else README_LINK).format(root=a.root_name, over=a.by_day_over),
                                     encoding="utf-8")
    made = have = missing = err = done = bad = 0
    errs, lines, smp = [], defaultdict(list), []
    random.seed(0)
    t0 = time.time()
    step = max(10000, total // 50)
    for pool, c, ps, byday, cf, root in info:
        miss_here, mk = 0, set()
        for src in ps:
            d = os.path.join(cf, upload_time(src)[:10] or "未知日期") if byday else cf
            if d not in mk:
                os.makedirs(d, exist_ok=True)
                mk.add(d)
            dst = os.path.join(d, os.path.basename(src))
            try:
                if a.mode == "copy":
                    if os.path.exists(dst):
                        have += 1
                    else:
                        shutil.copy2(src, dst)
                        made += 1
                else:
                    os.link(src, dst)
                    made += 1
            except FileExistsError:
                have += 1
            except FileNotFoundError:
                missing += 1
                miss_here += 1
                if len(errs) < 10:
                    errs.append(f"原图没了 {src}")
            except OSError as e:
                err += 1
                if len(errs) < 10:
                    errs.append(f"{src}: {e}")
            done += 1
            if done % step == 0 or done == total:
                el = time.time() - t0
                print(f"    {done:,}/{total:,}  {done / max(el, 1e-9):.0f} 张/秒  新建 {made:,} 已有 {have:,} 原图没了 {missing:,} 出错 {err:,}",
                      flush=True)
        # 核对这一类: 文件夹里(含日期子目录)的张数 = 要放的 - 原图没了的(多了少了都算对不上); 抽 30 张核是不是同一个文件
        on_disk = sum(len(fs) for _, _, fs in os.walk(cf))
        exp = len(ps) - miss_here
        ok = on_disk == exp
        bad += 0 if ok else 1
        print(f"    核对 {pool:14} {c:2} 文件夹里 {on_disk:>10,} 张  应有 {exp:>10,}" + ("" if ok else "  ★ 对不上"), flush=True)
        lines[root].append(f"{pool},{c},{os.path.basename(cf)},{on_disk},{'是' if byday else ''}")
        for src in random.sample(ps, min(30, len(ps))):
            d = os.path.join(cf, upload_time(src)[:10] or "未知日期") if byday else cf
            smp.append((src, os.path.join(d, os.path.basename(src))))
    for e in errs:
        print("    ★ " + e, flush=True)
    for k, root in roots.items():
        (root / "清单数.csv").write_text("\n".join(["图库,类别,文件夹,张数,按日期分"] + lines[k]) + "\n", encoding="utf-8-sig")
    chk = [(s, d) for s, d in smp if os.path.exists(s) and os.path.exists(d)]
    same = sum(1 for s, d in chk if (os.path.samefile(s, d) if a.mode == "link" else os.path.getsize(s) == os.path.getsize(d)))
    print(f"  抽 {len(chk)} 张核对: {'同一个文件' if a.mode == 'link' else '大小一样'} {same} 张", flush=True)
    print(f"  用时 {time.time() - t0:.0f} 秒  新建 {made:,}  已有 {have:,}  原图没了 {missing:,}  出错 {err:,}  文件夹对不上 {bad}", flush=True)
    if err or bad or same != len(chk):
        sys.exit(3)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="整页分类快速版(只看图)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--src", nargs="+", required=True)
    r.add_argument("--onnx", type=Path, required=True)
    r.add_argument("--policy", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    r.add_argument("--resume", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--state-csv", nargs="*", default=None, help="status_icon 以前全库扫的 icon.csv(同一版代码时省掉现场算)")
    li = sub.add_parser("lists")
    li.add_argument("--csv", type=Path, required=True)
    li.add_argument("--policy", type=Path, required=True)
    li.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--csv", type=Path, required=True)
    c.add_argument("--labels", type=Path, required=True)
    c.add_argument("--set", default="")
    c.add_argument("--preds", type=Path, default=None, help="page_cnn 评估的 preds.csv: 对拍现场推理和训练评估")
    c.add_argument("--policy", type=Path, default=None)
    pk = sub.add_parser("pick")
    pk.add_argument("--preds", type=Path, required=True, help="page_cnn 评估出的 preds.csv")
    pk.add_argument("--out", type=Path, required=True)
    fo = sub.add_parser("folders")
    fo.add_argument("--csv", type=Path, nargs="+", required=True, help="page_fast.csv(蓝图库、白图库各一个)")
    fo.add_argument("--policy", type=Path, required=True)
    fo.add_argument("--lists", type=Path, nargs="*", default=None, help="交出去的清单目录, 和 --csv 一一对应: 每类张数对上才做")
    fo.add_argument("--classes", nargs="*", default=None, help="只放这几类(默认全部)")
    fo.add_argument("--mode", choices=("link", "copy"), default="link")
    fo.add_argument("--copy-to", default="", help="copy 时放到哪(盘或目录)")
    fo.add_argument("--root-name", default="分图_9月")
    fo.add_argument("--by-day-over", type=int, default=50000, help="超过这么多张的类别按上传日期再分一层")
    fo.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "lists":
        write_lists(a.csv, json.load(open(a.policy, encoding="utf-8")), a.out)
    elif a.cmd == "pick":
        pick(a)
    elif a.cmd == "folders":
        if a.mode == "copy" and not a.copy_to:
            ap.error("--mode copy 要给 --copy-to")
        folders(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
