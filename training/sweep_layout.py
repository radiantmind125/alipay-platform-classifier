r"""端到端比较**版面开关**的几种设定 —— 拿每页真正读出来的东西逐页配对比, 不看净变化。

为什么
------
改版面(找拼音带、切行)的任何开关, 都会让喂给模型的图跟着变, 而模型是在
现状的几何上训出来的。好不好只能**真读一遍**才知道。这一程替身指标已经骗过
好几次(声调、块数倍数、宽/字、'有拼音的行'), 所以这里:

- 每个设定都走一遍**和上线完全一样**的 read_image
- **逐页配对**和现状比: 多少页变好、多少页变差, 不只看平均数
  (阈值 52 那次, 蓝图行的进出约 24%, 净变化只剩一点, 光看净数会被骗)
- 逐字段数**多出来几页、丢了几页**, 丢得明显多的点名
  (底边留白 8 那次就是字段/张净涨, 却悄悄丢了 计入收支)
- 单独看蓝图顶部那几个串(转账成功 收款方 付款方式 账户余额 回首页)
- '版面为空'(整页一段都没切出来)和'读不出字段'分开数

能试的开关(--variants, 逗号隔开)
--------------------------------
    thr<N>   版面掩膜阈值 = N                 例: thr52
    lr<K>    版面用 local_ratio, LOCAL_BIG_MIN = K/10   例: lr13 lr11 lr10
    xa       拼音带带横向范围(LAYOUT_XAWARE)
    用 + 组合, 例: lr10+xa
现状(什么都不改)总是自动放第一个当对照。

用法
----
    python sweep_layout.py --src D:\download2\pinyin_hits_blue ^
        --onnx D:\alipay-ai-data\rec_pinyin.onnx ^
        --pairs D:\alipay-ai-data\pinyin-pairs-blue2 --n 400 --variants lr13,lr10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import erase_pinyin  # noqa: E402
import pinyin_ocr  # noqa: E402
from erase_pinyin import DIFF_THRESHOLD, EXTS  # noqa: E402
from pinyin_ocr import Recognizer, read_image  # noqa: E402

HAN = re.compile(r"[一-鿿]")
# 和 e2e_bench 同一份 15 个字段名, 数和之前报过的可比
FIELDS_15 = ["账单详情", "全部账单", "创建时间", "付款方式", "订单号", "转账备注",
             "收款方", "账户余额", "计入收支", "转账成功", "交易成功", "对方账户",
             "商家名称", "支付时间", "交易号"]
# 蓝图顶部那一片的串。回首页不在 15 个字段里, 但它是漏拼音最多的串之一
HEADER = ["转账成功", "收款方", "付款方式", "账户余额", "回首页"]

# 可以临时改的开关: 名字 -> (模块, 属性)。都是在函数调用时才读的模块全局量
KNOBS = {
    "LAYOUT_MASK_THR": (pinyin_ocr, "LAYOUT_MASK_THR"),
    "LAYOUT_LOCAL_RATIO": (pinyin_ocr, "LAYOUT_LOCAL_RATIO"),
    "LOCAL_BIG_MIN": (erase_pinyin, "LOCAL_BIG_MIN"),
    "LAYOUT_XAWARE": (pinyin_ocr, "LAYOUT_XAWARE"),
}


def _one(part: str) -> dict:
    m = re.fullmatch(r"thr(\d+)", part)
    if m:
        return {"LAYOUT_MASK_THR": int(m.group(1))}
    m = re.fullmatch(r"lr(\d+)", part)
    if m:
        return {"LAYOUT_LOCAL_RATIO": True, "LOCAL_BIG_MIN": int(m.group(1)) / 10}
    if part == "xa":
        return {"LAYOUT_XAWARE": True}
    raise SystemExit(f"不认识的设定: {part!r}  "
                     f"(支持 thr<阈值>, lr<LOCAL_BIG_MIN x 10>, xa, 可以用 + 组合如 lr10+xa)")


def parse_variant(tok: str) -> tuple[str, dict]:
    tok = tok.strip()
    knobs: dict = {}
    for part in tok.split("+"):
        if not part:
            raise SystemExit(f"不认识的设定: {tok!r}")
        knobs.update(_one(part))
    return tok, knobs


def current_label() -> str:
    thr = pinyin_ocr.LAYOUT_MASK_THR
    thr = DIFF_THRESHOLD if thr is None else thr
    lr = "开" if pinyin_ocr.LAYOUT_LOCAL_RATIO else "关"
    xa = ",xa开" if pinyin_ocr.LAYOUT_XAWARE else ""
    return f"现状(thr{thr},lr{lr}{xa})"


def load_ceiling(pairs: Path | None) -> dict[str, str]:
    """同一张图每行 label 裁图(拼音在框外)读出来的字拼起来。没给就是空。"""
    if not pairs:
        return {}
    man = pairs / "_pairs_labeled.csv"
    if not man.exists():
        print(f"  (天花板清单不在: {man}, 不算天花板)")
        return {}
    agg: dict[str, list[str]] = defaultdict(list)
    for r in csv.DictReader(man.open(encoding="utf-8-sig")):
        if (r.get("text") or "").strip():
            agg[Path(r["source"]).name].append(r["text"])
    return {k: " ".join(v) for k, v in agg.items()}


def core(files: list[Path], rec: Recognizer, variants: list[tuple[str, dict]],
         ceil_by_src: dict[str, str] | None = None) -> dict:
    """每个设定把这批图全读一遍, 返回逐页结果。

    ★ 先记下所有开关**跑之前的值**, 每个设定跑之前先还原再套用,
      全部跑完(包括中途出错)再还原 —— 不写死 None, 默认值哪天改了也不会错。
    """
    ceil_by_src = ceil_by_src or {}
    prev = {k: getattr(mod, attr) for k, (mod, attr) in KNOBS.items()}
    out: dict = {"files": [p.name for p in files], "variants": {}}
    try:
        for label, knobs in variants:
            for k, (mod, attr) in KNOBS.items():
                setattr(mod, attr, prev[k])
            for k, v in knobs.items():
                mod, attr = KNOBS[k]
                setattr(mod, attr, v)
            pages = []
            t0 = time.time()
            for p in files:
                segs = read_image(p, rec)
                txt = " ".join(s["text"] for s in segs)
                rec_ = {
                    "fields": [f for f in FIELDS_15 if f in txt],
                    "header": [h for h in HEADER if h in txt],
                    "han": len(HAN.findall(txt)),
                    "segs": len(segs),
                    "blank_segs": sum(1 for s in segs if not (s["text"] or "").strip()),
                    "pin_segs": sum(1 for s in segs if s.get("has_pinyin")),
                }
                tc = ceil_by_src.get(p.name)
                if tc is not None:
                    rec_["c_fields"] = sum(1 for f in FIELDS_15 if f in tc)
                    rec_["c_han"] = len(HAN.findall(tc))
                pages.append(rec_)
            sec = time.time() - t0
            out["variants"][label] = {"knobs": knobs, "pages": pages, "sec": sec}
            mean_f = float(np.mean([len(r["fields"]) for r in pages])) if pages else 0.0
            print(f"    {label:<18} 跑完  {mean_f:.3f} 字段/张  {sec:.0f} 秒", flush=True)
    finally:
        for k, (mod, attr) in KNOBS.items():
            setattr(mod, attr, prev[k])
    return out


def _sig(up: int, down: int) -> str:
    """配对符号检验的粗判: |多-少| >= 2*sqrt(多+少) 大约相当于 p<0.05。"""
    n = up + down
    if n == 0:
        return "没变化"
    return "显著" if abs(up - down) >= 2 * math.sqrt(n) else "噪声以内"


def report(res: dict, title: str, n_bad: int = 0) -> None:
    labels = list(res["variants"].keys())
    base_lab = labels[0]
    base = res["variants"][base_lab]["pages"]
    n = len(base)
    print()
    print("=" * 84)
    print(f"  SWEEP LAYOUT  (ASCII only - safe to paste)   {title}")
    print("=" * 84)
    print(f"  {n} 张图, 每个设定都走一遍和上线完全一样的 read_image"
          + (f"   (另有 {n_bad} 张坏图跳过)" if n_bad else ""))
    print()
    hd = (f"  {'设定':<19}{'字段/张':>8}{'变化':>8}{'汉字/张':>8}{'读不出':>7}"
          f"{'版面为空':>8}{'段/张':>7}{'空白段':>7}{'带拼音段':>9}"
          f"{'字段比天花板':>12}{'汉字比天花板':>12}{'秒':>6}")
    print(hd)
    print("  " + "-" * 110)
    bf = float(np.mean([len(r["fields"]) for r in base])) if n else 0.0
    for lab in labels:
        pg = res["variants"][lab]["pages"]
        f = float(np.mean([len(r["fields"]) for r in pg])) if n else 0.0
        d = "" if lab == base_lab else f"{f - bf:+.3f}"
        han = sum(r["han"] for r in pg) / max(1, n)
        zero = sum(1 for r in pg if not r["fields"])
        empty = sum(1 for r in pg if r["segs"] == 0)
        segs = sum(r["segs"] for r in pg)
        blank = sum(r["blank_segs"] for r in pg) / max(1, n)
        pin = sum(r["pin_segs"] for r in pg) / max(1, segs)
        cov = [r for r in pg if "c_fields" in r]
        rf = (sum(len(r["fields"]) for r in cov) / sum(r["c_fields"] for r in cov)
              if cov and sum(r["c_fields"] for r in cov) else None)
        rc = (sum(r["han"] for r in cov) / sum(r["c_han"] for r in cov)
              if cov and sum(r["c_han"] for r in cov) else None)
        print(f"  {lab:<19}{f:>8.3f}{d:>8}{han:>8.1f}{zero:>7}{empty:>8}"
              f"{segs / max(1, n):>7.1f}{blank:>7.2f}{pin:>8.1%}"
              f"{(f'{rf:.3f}' if rf is not None else '-'):>12}"
              f"{(f'{rc:.3f}' if rc is not None else '-'):>12}"
              f"{res['variants'][lab]['sec']:>6.0f}")
    cov_n = sum(1 for r in base if "c_fields" in r)
    print()
    if cov_n:
        print(f"  (天花板两列只在 {cov_n} 张有配对清单的图上算)")
    print("  '读不出' = 一个字段都没读到;  '版面为空' = 整页一段都没切出来(两件事分开数)")
    print("  '带拼音段'是替身指标, 只看方向")
    print()

    for lab in labels[1:]:
        pg = res["variants"][lab]["pages"]
        print("-" * 84)
        print(f"  {lab}  对  {base_lab}  逐页配对")
        up = sum(1 for a, b in zip(base, pg) if len(b["fields"]) > len(a["fields"]))
        down = sum(1 for a, b in zip(base, pg) if len(b["fields"]) < len(a["fields"]))
        print(f"    字段数  变多 {up} 页, 变少 {down} 页, 不变 {n - up - down} 页"
              f"   -> {_sig(up, down)}")
        hu = sum(1 for a, b in zip(base, pg) for h in HEADER
                 if h in b["header"] and h not in a["header"])
        hd_ = sum(1 for a, b in zip(base, pg) for h in HEADER
                  if h in a["header"] and h not in b["header"])
        print(f"    蓝图顶部串(转账成功 收款方 付款方式 账户余额 回首页)"
              f"  多读出 {hu} 次, 少读出 {hd_} 次   -> {_sig(hu, hd_)}")
        flagged = []
        lines = []
        for fd in FIELDS_15 + ["回首页"]:
            key = "header" if fd == "回首页" else "fields"
            g = sum(1 for a, b in zip(base, pg) if fd in b[key] and fd not in a[key])
            ls = sum(1 for a, b in zip(base, pg) if fd in a[key] and fd not in b[key])
            if g or ls:
                lines.append(f"      {fd:<8} 多 {g:>3} 页  丢 {ls:>3} 页")
            # ★ 丢的比多的多出 max(5, 2*sqrt(总变动)) 页就点名
            if ls - g >= max(5, 2 * math.sqrt(g + ls)):
                flagged.append(f"{fd}(多{g} 丢{ls})")
        if lines:
            print("    逐字段(只列有变动的):")
            print("\n".join(lines))
        print("    ★★★★★ 明显丢了的字段: " + ("  ".join(flagged) if flagged else "没有"))
    print("-" * 84)
    print()
    print("  ★★ 选设定要**白蓝一起看**: 蓝图配对显著变好、白图没有字段被点名、")
    print("     '版面为空'和'空白段'不涨, 才算数。本机老模型的数只当预筛, 以服务器为准。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=17, help="和 e2e_bench 一致")
    ap.add_argument("--variants", default="lr13,lr10",
                    help="逗号隔开, 例 thr52,lr13,lr10。现状总会自动放第一个当对照")
    ap.add_argument("--dump", type=Path, default=None, help="把逐页结果存成 json")
    a = ap.parse_args()

    # ★ 抽样和 e2e_bench 一字不差, 这样'现状'那一行应当复现之前报过的数
    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in EXTS)
    random.seed(a.seed)
    files = random.sample(files, min(a.n, len(files)))
    ok, n_bad = [], 0
    for p in files:
        if cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) is None:
            print(f"    坏图跳过: {p.name}")
            n_bad += 1
            continue
        ok.append(p)
    if not ok:
        print("  一张能读的图都没有, 不跑了。")
        return

    variants: list[tuple[str, dict]] = [(current_label(), {})]
    seen = {variants[0][0]}
    for tok in a.variants.split(","):
        if not tok.strip():
            continue
        v = parse_variant(tok)
        if v[0] not in seen:
            variants.append(v)
            seen.add(v[0])
    rec = Recognizer(None, a.onnx)
    print(f"  {a.src.name}: {len(ok)} 张, 设定 {[v[0] for v in variants]}")
    res = core(ok, rec, variants, load_ceiling(a.pairs))
    report(res, a.src.name, n_bad)
    if a.dump:
        a.dump.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"  逐页结果存到 {a.dump}")


if __name__ == "__main__":
    main()
