"""对 label/ 里的裁图跑 OCR, 把标签文字填进 _pairs.csv。

和裁图分成两步, 是因为 OCR 很慢(7205 张约 13.8 万行, 约 8 小时),
分开之后重跑 OCR 不用重裁, 调 OCR 参数也不用动裁图。

质量过滤
--------
读出来的东西不是每条都能当标签, 这几种直接标成不可用:

    空的                   什么都没读出来
    纯拉丁                 说明裁行没裁干净, 拼音漏进来了
    混着拼音               汉字里夹着一段拼音, 同上
    太短                   一两个字符, 多半是噪声

★ 剩下的才写进 text 列。`usable` 列记这一条能不能用, `kind` 列记判成了哪一类。

★★ 跑完**一定要人工核几十条**再往下走。这里量的是"读出来了没有",
   不是"读对了没有" —— 那个只有人眼能判。

跑这么久, 这两件必须有
----------------------
1. **中途存盘**。每 2000 行落一次盘, 用临时文件改名, 保证盘上那份永远是完整的。
   不然跑到第 7 小时断电, 8 小时白跑。

2. **一行出错不许带塌整趟**。13.8 万次 OCR, 只要有一张图让 ocr() 抛异常,
   整趟就没了。所以每一行单独兜住, 出错就记成不可用接着跑。

★ 断了用 --resume 接着跑, 已经填过的行直接跳过, 不重读。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# 纯拉丁(拼音漏进来了)。排除带 @ 和数字的 —— 回单上本来就有邮箱和卡号
LATIN_ONLY = re.compile(r"^[a-zA-ZÀ-ɏ\s'·]+$")

# ★★★★★ 只挡"整条都是拉丁"不够。实测漏过这两条:
#     黄某人（德）备注 tongzhi 到账
#     Sueuz uanxiangqing quan buzhangdan 立 ?
#   它们**混着**汉字, 所以 LATIN_ONLY 判不中, 照样被当成可用标签。
#   拿这种去训, 等于教模型把拼音也读出来 —— 正好是要它别做的事。
#
#   所以再加一条: 文本里只要有**连续 4 个以上拉丁字母**的片段就不要。
#   4 这个数是看着实测样本定的: `tongzhi` 7 个, `uanxiangqing` 12 个;
#   而回单上正常的英文片段(卡组织缩写之类)一般不超过 3 个字母,
#   邮箱另外用 @ 放行。
LATIN_RUN = re.compile(r"[a-zA-ZÀ-ɏ]{4,}")
MIN_LEN = 2

FIELDS = ["file", "source", "row", "label_y0", "label_y1",
          "input_y0", "input_y1", "has_pinyin", "text", "usable", "kind"]

CHECKPOINT = 2000        # 每这么多行落一次盘


def classify(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "空"
    if len(s) < MIN_LEN:
        return "太短"
    if LATIN_ONLY.match(s) and "@" not in s:
        return "纯拉丁(拼音漏进来了)"
    if "@" not in s and LATIN_RUN.search(s):
        return "混着拼音"
    return "可用"


# ★★★★★ 每个子进程自己留一份 RapidOCR。模型只在第一次调用时加载(一两秒),
#   之后这个进程里一直复用。不能在主进程建好传进来 —— onnxruntime 的 session
#   带着本地句柄, pickle 不过去。
_OCR = None


def _init_worker() -> None:
    """子进程起来时先把线程数按死成 1。

    ★★ onnxruntime 默认会自己开一堆 intra-op 线程。8 个进程 x 每个开 8 条线程
       = 64 条抢 8 个核, 抢得比单进程还慢。**并行要放在进程这一层**:
       一个进程读一张图, 线程数按死成 1。
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)


def _read_one(path_str: str) -> tuple[str, str]:
    """读一张标签裁图, 返回 (文字, 判成哪一类)。

    ★ 出错在这里就地兜住 —— 13.8 万次调用, 一次没兜住就是几个小时白跑。
      子进程里抛出去的异常会把整个进程池带塌, 比单进程更糟。
    """
    global _OCR
    try:
        p = Path(path_str)
        if not p.exists():
            return "", "图不在"
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "", "读不了"
        if _OCR is None:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        res, _ = _OCR(img)
        # 一行可能读出多段(左标签 + 右取值), 按 x 排好拼起来
        parts = []
        for box, t, _conf in (res or []):
            s = (t or "").strip()
            if s:
                parts.append((min(pt[0] for pt in box), s))
        parts.sort()
        text = " ".join(s for _, s in parts)
        return text, classify(text)
    except Exception as e:                     # noqa: BLE001
        return "", f"出错({type(e).__name__})"


def write_out(rows: list[dict], out: Path) -> None:
    """先写临时文件再改名 —— 保证盘上那份任何时候都是完整可读的。"""
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True,
                    help="make_pinyin_pairs.py 出的目录")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="接着上次跑, 已经填过的行跳过")
    ap.add_argument("--workers", type=int, default=1,
                    help="几个进程一起读。★★ 默认 1 是**量出来的**, 不是图省事 —— "
                         "onnxruntime 自己就会用多核, 多开进程反而更慢, 见下面注释")
    a = ap.parse_args()

    man = a.pairs / "_pairs.csv"
    if not man.exists():
        print(f"清单不在: {man}")
        return
    rows = list(csv.DictReader(man.open(encoding="utf-8-sig")))
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        print("清单是空的")
        return
    for r in rows:
        r.setdefault("text", "")
        r.setdefault("usable", "")
        r.setdefault("kind", "")

    out = a.pairs / "_pairs_labeled.csv"

    # ---------- 接着上次跑 ----------
    done = 0
    if a.resume and out.exists():
        old = {r["file"]: r for r in csv.DictReader(out.open(encoding="utf-8-sig"))}
        for r in rows:
            o = old.get(r["file"])
            if o and (o.get("kind") or "").strip():
                r["text"], r["usable"], r["kind"] = o.get("text", ""), o.get("usable", "0"), o["kind"]
                done += 1
        print(f"★ 接着上次跑: 已经填过 {done:,} 行, 这趟只读剩下的 {len(rows)-done:,} 行")
    elif out.exists():
        print(f"★★ {out.name} 已经在了。想接着上次跑就加 --resume,")
        print("   不加的话下面会从头读一遍, 把它覆盖掉。")

    todo = [r for r in rows if not (r.get("kind") or "").strip()]
    if not todo:
        print("没有要读的行了")
        write_out(rows, out)
        return

    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        print("没装 rapidocr_onnxruntime:  pip install rapidocr_onnxruntime")
        return

    label_dir = a.pairs / "label"
    paths = [str(label_dir / r["file"]) for r in todo]
    w = max(1, a.workers)
    print(f"要读 {len(todo):,} 条,  {w} 个进程")
    t0 = time.time()

    # ★★★★★ workers=1 时**不开进程池**, 就在本进程里顺着读。
    #   这不是图省事 —— onnxruntime 自己会把一次推理摊到多个核上,
    #   开进程池反而要把每个进程的线程数按死成 1(不按死就 N 进程 x N 线程互相抢)。
    #
    #   实测 8 核机器, 同样 150 行:
    #       workers=1  本进程跑, ORT 自己用多核     0.89 秒/行
    #       workers=2  2 进程 x 每进程 1 线程        0.81 秒/行   (在噪声范围内)
    #       workers=4  4 进程 x 每进程 1 线程        1.03 秒/行   <- 明显更慢
    #   重复跑的抖动约 6%, 所以 workers=2 和 1 其实没差, 4 是真慢。
    #
    # ★ 我一开始以为"单进程"就是慢的原因, 加了进程池。**量完发现不是** ——
    #   OCR 从来就没卡在单核上。要真想快, 得靠 GPU, 不是靠多开进程。
    #   进程池留着, 但默认 1。
    def emit(i: int, text: str, kind: str) -> None:
        r = todo[i - 1]
        r["text"] = text
        r["kind"] = kind
        r["usable"] = "1" if kind == "可用" else "0"
        if i % 500 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / max(el, 1e-6)
            left = (len(todo) - i) / max(rate, 1e-6)
            print(f"  {i:,}/{len(todo):,}   {rate:.1f} 行/秒   "
                  f"还要约 {left/3600:.1f} 小时", flush=True)
        if i % CHECKPOINT == 0:
            write_out(rows, out)

    if w == 1:
        for i, p in enumerate(paths, 1):
            emit(i, *_read_one(p))
    else:
        # ex.map 按**送进去的顺序**把结果吐回来, 所以直接和 todo 对得上号
        with ProcessPoolExecutor(max_workers=w, initializer=_init_worker) as ex:
            for i, (text, kind) in enumerate(ex.map(_read_one, paths, chunksize=16), 1):
                emit(i, text, kind)

    write_out(rows, out)

    tally: dict[str, int] = {}
    for r in rows:
        k = r.get("kind") or "空"
        tally[k] = tally.get(k, 0) + 1
    usable = sum(1 for r in rows if r.get("usable") == "1")
    py_usable = sum(1 for r in rows
                    if r.get("usable") == "1" and r.get("has_pinyin") == "1")

    print()
    print("=" * 60)
    print(f"读完 {len(rows):,} 条  (这趟读了 {len(todo):,} 条, 用时 {(time.time()-t0)/3600:.1f} 小时)")
    print("=" * 60)
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v:6,d}  ({v/len(rows):5.1%})")
    print()
    print(f"★ 可用 {usable:,} 条 ({usable/len(rows):.1%})")
    print(f"  其中上面有拼音的 {py_usable:,} 条 —— 这些才是真正要学的样本")
    print()
    print(f"填好的清单 -> {out}")
    print()
    print("★★ 下一步**必须**人工核几十条: 打开 input/ 的图, 对着 text 看读对了没有。")
    print("   这里只知道 读出来了, 不知道 读对了。")


if __name__ == "__main__":
    main()
