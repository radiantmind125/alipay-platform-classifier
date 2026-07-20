r"""漂移监控(零标签、只读):把跨信号审计做成"随时间比对"的持续预警。

用同一个池子按时间切成 参考窗(较早) vs 当前窗(较近),比这几项有没有漂:
  - Tier-0 覆盖率  —— 掉了多半是**出新机型/新分辨率**(表没命中,静默劣化)
  - 不确定率       —— 涨了说明模型在新数据上没把握
  - 类别(苹果占比) —— 突变可能是攻击或人群变化
  - 冲突率         —— 分辨率↔状态栏矛盾比例(需 --crosscheck 跑过才有);涨=更多缩放伪造/模型退化
  - PSI(短边/长宽比) —— 便宜特征的分布漂移量(<0.1 稳,0.1–0.25 关注,>0.25 告警)

**注意:变≠错。** 无标签下良性漂移(新机型)和攻击分不开,这里只当**触发器**:报警 → 去补采
真值锚点 / 重跑红队 / 看个案,不直接下"错了多少"的结论。

用法:
  python scripts\drift_monitor.py --results runs\pool_full\final_device_v2.jsonl --inspect runs\pool_full\inspect.jsonl
  # 可选:--split-date 20260705 明确切点;默认按时间中位数对半切
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

def _ts(fname: str) -> str | None:
    # 取**最后**一段 14 位数字 = 日期时间戳(前面那串长数字是 voucher ID,不能用)
    ms = re.findall(r"\d{14}", fname)
    return ms[-1] if ms else None


def _load(p: Path) -> dict:
    return {json.loads(l)["file"]: json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def _psi(ref: list[float], cur: list[float], nbins: int = 10) -> float | None:
    r, c = np.asarray(ref, float), np.asarray(cur, float)
    if len(r) < nbins or len(c) < 1:
        return None
    edges = np.unique(np.quantile(r, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    rh = np.clip(np.histogram(r, edges)[0] / len(r), 1e-6, None)
    ch = np.clip(np.histogram(c, edges)[0] / len(c), 1e-6, None)
    return float(np.sum((ch - rh) * np.log(ch / rh)))


def _metrics(files: list[str], res: dict, insp: dict) -> dict:
    n = len(files)
    dev = [res[f].get("device") for f in files]
    tier0 = sum(1 for f in files if res[f].get("source") == "resolution")
    unc = sum(1 for d in dev if d == "uncertain")
    ios = sum(1 for d in dev if d == "ios")
    andr = sum(1 for d in dev if d == "android")
    conf_present = any("device_prior_conflict" in res[f] for f in files)
    conflict = sum(1 for f in files if res[f].get("device_prior_conflict"))
    shorts, aspects = [], []
    for f in files:
        r = insp.get(f, {})
        w, h = r.get("width", 0), r.get("height", 0)
        if w and h:
            shorts.append(min(w, h))
            aspects.append(max(w, h) / min(w, h))
    return {"n": n, "tier0": tier0 / n if n else 0, "uncertain": unc / n if n else 0,
            "ios_frac": ios / (ios + andr) if (ios + andr) else 0,
            "conflict": conflict / n if n else 0, "conf_present": conf_present,
            "shorts": shorts, "aspects": aspects}


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--inspect", type=Path, required=True)
    ap.add_argument("--split-date", default="", help="YYYYMMDD;不给则按时间中位数对半切")
    args = ap.parse_args(argv)

    res = _load(args.results)
    insp = _load(args.inspect)
    dated = [(f, _ts(f)) for f in res if _ts(f)]
    dated.sort(key=lambda x: x[1])
    if len(dated) < 20:
        print("样本太少或文件名无时间戳,无法做漂移比对")
        return
    if args.split_date:
        cut = args.split_date + "000000"
    else:
        cut = dated[len(dated) // 2][1]
    ref = [f for f, t in dated if t < cut]
    cur = [f for f, t in dated if t >= cut]
    if not ref or not cur:
        print(f"切点 {cut} 使某一窗为空,换 --split-date")
        return

    m0, m1 = _metrics(ref, res, insp), _metrics(cur, res, insp)
    print(f"漂移监控  参考窗={m0['n']}(<{cut[:8]}) vs 当前窗={m1['n']}(>={cut[:8]})\n")

    alarms: list[str] = []

    def line(name, a, b, *, pct=True, warn=None, higher_bad=True):
        d = b - a
        fa, fb, fd = (f"{a:.1%}", f"{b:.1%}", f"{d:+.1%}") if pct else (f"{a:.3f}", f"{b:.3f}", f"{d:+.3f}")
        tag = ""
        if warn is not None and ((higher_bad and d >= warn) or (not higher_bad and d <= -warn)):
            tag = "  ← 告警"
            alarms.append(name)
        print(f"  {name:14} {fa:>8} -> {fb:>8}  ({fd}){tag}")

    line("Tier-0覆盖率", m0["tier0"], m1["tier0"], warn=0.03, higher_bad=False)  # 掉=坏
    line("不确定率", m0["uncertain"], m1["uncertain"], warn=0.02, higher_bad=True)
    line("苹果占比", m0["ios_frac"], m1["ios_frac"], warn=0.05, higher_bad=True)
    if m0["conf_present"] or m1["conf_present"]:
        line("冲突率", m0["conflict"], m1["conflict"], warn=0.01, higher_bad=True)
    else:
        print(f"  {'冲突率':14} (未启用 --crosscheck,无 device_prior_conflict 字段)")

    for name, key in (("PSI·短边", "shorts"), ("PSI·长宽比", "aspects")):
        v = _psi(m0[key], m1[key])
        if v is None:
            continue
        sev = "稳定" if v < 0.1 else ("关注" if v < 0.25 else "告警")
        if v >= 0.25:
            alarms.append(name)
        print(f"  {name:14} {v:.3f}  [{sev}]")

    print()
    if alarms:
        print(f"总体:⚠ 关注/告警 —— {', '.join(alarms)}")
        print("  建议:去补采真值锚点核实、重跑红队、抽这批漂移样本看个案(看个案≠标全量,不违约束)。")
    else:
        print("总体:✓ 稳定 —— 各项在阈值内。")


if __name__ == "__main__":
    main()
