"""把图库里**带拼音的图**全部挑出来。

为什么要这个
------------
经理 9/14 的原话是三句:
    拼音的不行ocr识别出来很多都是不全的
    估计要单独训练一个
    先把拼音的图全部挑出来 至少要一万张吧

第三句一直没做。擦图那个脚本里已经有现成的判据, 这里把它当**筛子**用,
扫全库, 出一份名单。

★★★ 顺带能回答一个要紧的问题: 库里到底有没有一万张拼音图。
    本机拿 2504 张真实库样本量过, 阈值 0.06 之上**一张都没有**。
    召回按 75% 折, 这 2504 张里真实的拼音图上限也就三四张, 合 0.16% 以下。
    照这个比例要凑一万张, 得有 **六百万张以上**的原图。
    ★ 这只是一个两个月的切片, 不代表全库 —— 但脚本跑一遍全库就知道了,
      而且这个数得在开始训练之前先弄清楚。

判据和精度
----------
分数 = 判为注音的连通块 / 总连通块, 是个比例, 不随图的大小变。

解码尺寸(40 张拼音图上量的):

    原图        召回 90%   每核每秒  30 张
    缩一半      召回 72%   每核每秒  69 张
    缩四分之一  召回  0%   每核每秒 123 张

★★ 缩四分之一直接把信号毁掉了 —— 拼音笔画太细, 降采样之后整个没了,
   所有图的分数都是 0(那一档看着"召回 100%"只是因为全都过阈值了, 是假的)。
   所以**必须原图解码**, 慢就慢。

★★★ 阈值 0.06 是拿 **2504 张真实库样本**标定的, 误判 0 张, 召回 75%。
   一开始用 61 张对照定的 0.01 是**错的**, 详见下面 DEFAULT_THRESHOLD 那段。

★ 召回 75% 的意思是: 四张里挑得出三张。
  所以**库里真实的拼音图比这个脚本报的数多三成左右**。

怎么用
------
    # 扫全库, 出名单(可以中断, 再跑接着来)
    python pick_pinyin.py --root 图库目录 --out 名单.jsonl --workers 8

    # 看结果, 不重跑
    python pick_pinyin.py --out 名单.jsonl --report

    # 把挑出来的复制走(默认只挑分数最高的, 够数就停)
    python pick_pinyin.py --out 名单.jsonl --copy-to 目标目录 --limit 10000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import EXTS, annotation_labels, local_background, text_mask  # noqa: E402

# ★★★★★ 阈值。**这个数被实测推翻过一次, 过程记在这里。**
#
#   一开始拿 61 张非拼音图当对照, 量出来 95 分位是 0.0000, 于是取了 0.01,
#   说"只要大于 0 就没有误判"。**错的。**
#
#   换成真实库抽样(2504 张)一跑, 0.01 挑出 53 张, 打开一看
#   **一张拼音都没有** —— 全是普通回单、一张麻将游戏截图、一张聊天记录。
#
#   为什么会误判: 判据是"小块紧贴在大块正上方, 而且同一高度上凑得成一排"。
#   回单上天然就有这种东西 ——
#       收款方姓名  **浩          打码的星号就是一排小块
#       付款方账号  150 **** **02
#       收款方账号  rya***@outlook.com
#       支付奖励    橙色小胶囊里的小字
#   这些和拼音在形状上**没有区别**, 光看连通块分不开。
#
#   ★★ 教训: 61 张对照太少也太单一。真实库里的花样多得多。
#
#   重新用 2504 张真实库样本标定:
#       阈值    误判      召回
#       0.01    53 张     90%
#       0.02     5 张     88%
#       0.04     1 张     75%
#       0.06     0 张     75%    <- 取这个
#       0.10     0 张     68%
#
#   ★ 真拼音整页都是拼音, 分数中位 0.1495; 上面那些误判最高才 0.0527。
#     0.06 正好在两者之间, 误判归零而召回只从 90% 掉到 75%。
DEFAULT_THRESHOLD = 0.06

# ★ 太小的图不是回单截图, 直接跳过, 省解码时间
MIN_SIDE = 200


def score_one(path_str: str) -> dict:
    """返回一张图的拼音分数。**这个函数在子进程里跑, 不能抛异常出来。**"""
    p = Path(path_str)
    rec = {"path": path_str, "score": None, "err": None}
    try:
        buf = np.fromfile(str(p), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rec["err"] = "decode"
            return rec
        h, w = img.shape
        if min(h, w) < MIN_SIDE:
            rec["err"] = "too_small"
            return rec
        mask = text_mask(img, local_background(img))
        _, kept, total, _ = annotation_labels(mask)
        rec["score"] = (len(kept) / total) if total else 0.0
        rec["wh"] = [int(w), int(h)]
    except Exception as e:                      # noqa: BLE001
        rec["err"] = type(e).__name__
    return rec


def load_done(out: Path) -> set[str]:
    """已经量过的跳过 —— 扫全库要几个小时, 必须能中断续跑。"""
    done: set[str] = set()
    if not out.exists():
        return done
    with out.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["path"])
            except Exception:                   # noqa: BLE001
                continue                        # 半行(上次被 Ctrl-C 砍断)直接丢掉
    return done


def iter_images(root: Path):
    for dirpath, _, names in os.walk(root):
        for n in names:
            if Path(n).suffix.lower() in EXTS:
                yield str(Path(dirpath) / n)


def read_all(out: Path) -> list[dict]:
    rows = []
    with out.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:                   # noqa: BLE001
                continue
    return rows


def report(out: Path, threshold: float) -> None:
    rows = read_all(out)
    if not rows:
        print(f"{out} 里没有记录")
        return
    ok = [r for r in rows if r.get("score") is not None]
    bad = [r for r in rows if r.get("score") is None]
    hits = [r for r in ok if r["score"] >= threshold]

    print("=" * 66)
    print(f"扫过 {len(rows)} 张    量到分数的 {len(ok)}    读不了/太小 {len(bad)}")
    print("=" * 66)
    if bad:
        why: dict[str, int] = {}
        for r in bad:
            why[r.get("err") or "?"] = why.get(r.get("err") or "?", 0) + 1
        print("  跳过的原因: " + ",  ".join(f"{k} {v}" for k, v in sorted(why.items())))
        print()

    pct = len(hits) / len(ok) if ok else 0
    print(f"★ 阈值 {threshold:.3f} 之上的: {len(hits)} 张  占量到的 {pct:.2%}")
    print()

    if ok:
        a = np.array([r["score"] for r in ok])
        print("  分数分布:")
        # ★ 分档边界必须**带上阈值本身**, 否则会出现一档横跨阈值, 标注就是错的
        edges = sorted({0.0, 0.001, 0.01, round(threshold, 4), 0.15, 1.01})
        for lo, hi in zip(edges, edges[1:]):
            n = int(((a >= lo) & (a < hi)).sum())
            tag = "  <- 阈值以下" if hi <= threshold else ""
            print(f"    {lo:.3f} - {hi:.3f}   {n:7d}{tag}")
        print()

    # ★★★ 经理要一万张。够不够, 这里直接说清楚。
    print("=" * 66)
    if len(hits) >= 10000:
        print(f"★★★ 够一万张(挑出 {len(hits)} 张)。")
    else:
        print(f"★★★ **不够一万张** —— 只挑出 {len(hits)} 张。")
        # 召回 75%, 所以库里真实的比挑出来的多约三成
        if pct > 0 and pct / 0.75 > 1.0:
            # ★★ 已经筛过的目录: 按召回折回去会超过 100%(实测量到过 129%),
            #   而且 10000/1.0 = 10000 这种推算只是把目标数原样念一遍, 没有信息。
            #   这种情况下真正有用的是**还差多少张**。
            print(f"    挑出来占 {pct:.3%} —— 这么高不像随机库样本, "
                  "多半是对着一个**已经筛过的**目录在跑。")
            print(f"    ★ 那推算占比就没意义了, 有意义的是: **还差 "
                  f"{10000 - len(hits):,} 张**。")
        elif pct > 0:
            true_pct = pct / 0.75
            need = int(10000 / true_pct)
            print(f"    挑出来占 {pct:.3%}, 按召回 75% 折回去, 真实占比约 {true_pct:.3%}。")
            print(f"    要凑一万张得有大约 {need:,} 张原图, 现在扫了 {len(ok):,} 张。")
        else:
            # 一张都没挑出来: 给个上界, 别说"库里没有"
            if ok:
                ub = 3.0 / 0.75 / len(ok)       # 零观测的 95% 上界 ~3/n, 再按召回折
                print(f"    ★ 一张都没挑出来。{len(ok):,} 张里真实占比的 95% 上界约 {ub:.3%},")
                print(f"      也就是最多 {int(ub*len(ok))+1} 张。要凑一万张得有 "
                      f"{int(10000/ub):,} 张以上的原图。")
        print("    这条要跟经理说 —— 一万张这个前提可能得改。")
    print("=" * 66)


def make_band_sheet(out: Path, sheet: Path, lo: float, hi: float,
                    n: int = 12, crop: tuple[float, float] = (0.25, 0.55)) -> None:
    """只贴**某个分数区间**的图, 用来找"拼音到底从几分开始出现"。

    ★★★★★ 为什么必须有这个: `make_sheet` 贴的是分数最高的那些, 那些**当然**有拼音。
       拿它去确认阈值定得对不对, 是自己骗自己 —— 阈值附近那一段根本没看到。

    2026-09-16 实测栽在这上面: 蓝图库按 0.06 挑出 8160 张, 贴出来的 12 张里
    只有 3 张真有拼音, 0.069 到 0.150 那一段**一张都没有**。
    阈值 0.06 是拿**白图**标定的, 蓝图上那些小徽章(充值金+15 / 学分+50 / 里程币+20)
    和促销卡片, 形状上同样是"小字压在大字上面一排", 判据分不开。

    crop 是裁哪一段页面。蓝图和白图版面不一样, 默认那个窗口是按白图调的。
    """
    rows = [r for r in read_all(out)
            if r.get("score") is not None and lo <= r["score"] < hi]
    if not rows:
        print(f"{lo:.3f} 到 {hi:.3f} 这一段里没有图")
        return
    rows.sort(key=lambda r: -r["score"])
    step = max(1, len(rows) // n)
    picked = rows[::step][:n]

    cells = []
    for r in picked:
        src = Path(r["path"])
        if not src.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h = img.shape[0]
        c = img[int(h * crop[0]): int(h * crop[1]), :]
        if c.size == 0:
            continue
        c = cv2.resize(c, (520, max(1, int(520 * c.shape[0] / c.shape[1]))))
        cv2.putText(c, f"{r['score']:.3f}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cells.append(c)
    if not cells:
        print("图都读不出来(路径可能变了)")
        return

    hh = max(x.shape[0] for x in cells)
    cells = [cv2.copyMakeBorder(x, 0, hh - x.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for x in cells]
    per = 3
    usable = len(cells) - len(cells) % per
    if usable == 0:
        usable, per = len(cells), len(cells)
    grid = np.vstack([np.hstack(cells[i:i + per]) for i in range(0, usable, per)])
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", grid)[1].tofile(str(sheet))
    print(f"{lo:.3f} 到 {hi:.3f} 这一段共 {len(rows)} 张, 抽了 {usable} 张 -> {sheet}")
    print("★★ 打开看: 这一段里**有几张真有拼音**。一张都没有就说明阈值还得往上抬。")


def make_sheet(out: Path, sheet: Path, threshold: float, n: int = 12) -> None:
    """把分数**跨着挡次**抽几张贴成一张图, 好让人肉眼核。

    ★★★★★ 这一步不是可选的装饰。阈值 0.01 那次, 分数表看着完全正常
       (挑出 2.12%, 听着很合理), **是打开图看才发现一张拼音都没有的**。
       光看数字永远发现不了这种错, 所以把它做进工具里。

    抽法是**按分数分层抽**, 不是只抽最高的 —— 要看的正是阈值附近那些。
    """
    rows = [r for r in read_all(out) if r.get("score") is not None]
    if not rows:
        print("没有可看的记录")
        return
    rows.sort(key=lambda r: -r["score"])
    hits = [r for r in rows if r["score"] >= threshold]
    picked = []
    if hits:
        step = max(1, len(hits) // n)
        picked = hits[::step][:n]
    if len(picked) < n:      # 过阈值的不够就补几张阈值底下的, 好看清楚边界在哪
        below = [r for r in rows if r["score"] < threshold][: n - len(picked)]
        picked += below
    if not picked:
        print("没有可看的记录")
        return

    cells = []
    for r in picked:
        src = Path(r["path"])
        if not src.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h = img.shape[0]
        crop = img[int(h * 0.25): int(h * 0.55), :]        # 正文字段那一段
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (520, max(1, int(520 * crop.shape[0] / crop.shape[1]))))
        col = (0, 0, 255) if r["score"] >= threshold else (128, 128, 128)
        cv2.putText(crop, f"{r['score']:.3f}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cells.append(crop)
    if not cells:
        print("图都读不出来(路径可能变了)")
        return

    hh = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in cells]
    per = 3
    usable = len(cells) - len(cells) % per
    if usable == 0:
        usable, per = len(cells), len(cells)
    grid = np.vstack([np.hstack(cells[i:i + per]) for i in range(0, usable, per)])
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", grid)[1].tofile(str(sheet))
    print(f"贴了 {usable} 张 -> {sheet}")
    print("★ 红字是过阈值的, 灰字是没过的。")
    print("★★ 打开看一眼: 红字那些**汉字头上到底有没有拼音**。")
    print("   没有的话阈值就还是松的, 别拿这批去训练。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None, help="图库根目录(递归扫)")
    ap.add_argument("--out", type=Path, required=True, help="名单 jsonl(可续跑)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--report", action="store_true", help="只看已有名单, 不重扫")
    ap.add_argument("--sheet", type=Path, default=None,
                    help="★ 抽几张贴成一张图肉眼核 —— 光看数字发现不了误判")
    ap.add_argument("--band", type=str, default=None,
                    help="★★ 只贴某个分数区间, 例 --band 0.06,0.10。"
                         "找阈值必须用这个 —— 只贴最高分那些是自己骗自己")
    ap.add_argument("--crop", type=str, default="0.25,0.55",
                    help="贴图时裁页面的哪一段(蓝图白图版面不一样)")
    ap.add_argument("--copy-to", type=Path, default=None, help="把挑出来的复制到这里")
    ap.add_argument("--limit", type=int, default=0, help="最多复制多少张(0 = 不限)")
    ap.add_argument("--copy-clean", action="store_true",
                    help="★ 反过来复制**没有拼音**的那些(分数最低的), 用来做对照组。"
                         "对照组必须和拼音组来自**同一个库**, 否则比的是两批不同的图。")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--skip-names-from", type=Path, default=None,
                    help="这个目录里已经有同名文件的就不扫(比如老图挪去的 E:\\BlueImages)。"
                         "新传的一批里要是混着老图, 省得把老图再扫一遍")
    a = ap.parse_args()

    if a.copy_to:
        if a.copy_clean:
            # ★★ 对照组: 取分数**最低**的, 也就是最确定没有拼音的。
            #   不是随便抽 —— 随便抽会混进阈值附近那些拿不准的。
            rows = [r for r in read_all(a.out)
                    if r.get("score") is not None and r["score"] < a.threshold]
            rows.sort(key=lambda r: r["score"])
        else:
            rows = [r for r in read_all(a.out)
                    if r.get("score") is not None and r["score"] >= a.threshold]
            rows.sort(key=lambda r: -r["score"])    # 分数高的优先, 最像拼音的先要
        if a.limit:
            rows = rows[: a.limit]
        a.copy_to.mkdir(parents=True, exist_ok=True)
        n = 0
        copied: list[tuple[str, float, str]] = []
        for r in rows:
            src = Path(r["path"])
            if not src.exists():
                continue
            dst = a.copy_to / src.name
            i = 1
            while dst.exists():                 # 不同目录可能重名, 不能覆盖
                dst = a.copy_to / f"{src.stem}__{i}{src.suffix}"
                i += 1
            shutil.copy2(src, dst)
            copied.append((dst.name, r["score"], str(src)))
            n += 1
        what = "**没有拼音的**(对照组)" if a.copy_clean else "挑出来的"
        print(f"复制了 {n} 张{what} -> {a.copy_to}")
        if rows:
            print(f"  分数范围 {rows[0]['score']:.4f} 到 {rows[min(n, len(rows))-1]['score']:.4f}")

        # ★★ 一起写一份清单。没有它, 归集完就只剩一堆文件名,
        #   分数和来源都丢了, 以后想按分数段抽查都没法抽。
        #   追加模式: 两个库可以往同一个目录归集, 清单接着写。
        man = a.copy_to / "_manifest.csv"
        new = not man.exists()
        with man.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["file", "score", "source"])
            w.writerows(copied)
        total = sum(1 for _ in man.open(encoding="utf-8-sig")) - 1
        print(f"  清单 -> {man}   (累计 {total} 行)")
        return

    if a.sheet:
        try:
            c1, c2 = (float(x) for x in a.crop.split(","))
        except Exception:                        # noqa: BLE001
            print("--crop 格式应当是 0.25,0.55")
            return
        if a.band:
            try:
                lo, hi = (float(x) for x in a.band.split(","))
            except Exception:                    # noqa: BLE001
                print("--band 格式应当是 0.06,0.10")
                return
            make_band_sheet(a.out, a.sheet, lo, hi, crop=(c1, c2))
        else:
            report(a.out, a.threshold)
            print()
            make_sheet(a.out, a.sheet, a.threshold)
        return

    if a.report:
        report(a.out, a.threshold)
        return

    if not a.root:
        print("要扫库就得给 --root, 只看结果就加 --report")
        return
    if not a.root.exists():
        print(f"目录不在: {a.root}")
        return

    done = load_done(a.out)
    if done:
        print(f"★ 上次已经量过 {len(done):,} 张, 这次跳过它们。")
    print("正在列文件...")
    todo = [p for p in iter_images(a.root) if p not in done]
    if a.skip_names_from:
        if not a.skip_names_from.exists():
            print(f"--skip-names-from 目录不在: {a.skip_names_from}")
            return
        seen = {os.path.basename(p) for p in iter_images(a.skip_names_from)}
        before = len(todo)
        todo = [p for p in todo if os.path.basename(p) not in seen]
        print(f"★ {a.skip_names_from} 里已有同名的 {before - len(todo):,} 张, 不扫")
    print(f"这次要量 {len(todo):,} 张,  {a.workers} 个进程")
    if not todo:
        print("没有新的要量。")
        report(a.out, a.threshold)
        return

    est = len(todo) / (30.0 * a.workers)
    print(f"★ 按每核每秒 30 张估, 大约要 {est/60:.0f} 分钟"
          f"({est/3600:.1f} 小时)。中途 Ctrl-C 可以, 再跑接着来。")
    print()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = hits = 0
    # ★ 边算边写, 不在内存里攒 —— 一是能续跑, 二是全库放不下
    with a.out.open("a", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rec in ex.map(score_one, todo, chunksize=a.chunk):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if rec.get("score") is not None and rec["score"] >= a.threshold:
                hits += 1
            if n % 2000 == 0:
                f.flush()
                el = time.time() - t0
                rate = n / el if el else 0
                left = (len(todo) - n) / rate if rate else 0
                print(f"  {n:,}/{len(todo):,}   挑出 {hits:,}   "
                      f"{rate:.0f} 张/秒   还要 {left/60:.0f} 分钟")
    print(f"\n量完 {n:,} 张, 用了 {(time.time()-t0)/60:.1f} 分钟\n")
    report(a.out, a.threshold)


if __name__ == "__main__":
    main()
