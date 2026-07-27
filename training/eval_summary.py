r"""读 predict_all_models.py 产出的 summary.csv,算分布 + 特异性/召回。

命令行用法:
  python training/eval_summary.py D:\ssp_test\gen_clean_out\summary.csv --kind genuine
  python training/eval_summary.py D:\ssp_test\realfake2_out\summary.csv  --kind fake

也被 run_eval_ab.py 直接 import summarize() 复用。
--kind genuine -> 报"特异性"(Pass 率 = 1 - 误杀率),真图应尽量 Pass。
--kind fake    -> 报"召回"(Reject 率),假图应尽量 Reject。
阈值与 predict 一致:Reject>=0.90 / ManualReview 0.60-0.90 / Pass<0.60。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def _load(csv_path) -> tuple[list[float], Counter, int]:
    """返回 (有效分数升序, 决策计数, 剔除的评分失败张数)。"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    if not rows:
        return [], Counter(), 0
    # 剔除"评分失败"的图:predict 里某图在模型上抛异常时 scores 为空({}),final_ai_score 保持 0.0,
    # 直接当分统计会把这些图误算成"正确 Pass"(真图特异性虚高)/"漏检"(假图召回虚低)。
    has_sj = any("scores_json" in r for r in rows)

    def _scored(r) -> bool:
        if r.get("final_ai_score") in (None, ""):
            return False
        if has_sj:
            return (r.get("scores_json") or "").strip() not in ("", "{}")
        return True

    good = [r for r in rows if _scored(r)]
    dropped = len(rows) - len(good)
    scores = sorted(float(r["final_ai_score"]) for r in good)
    dec = Counter((r.get("decision") or "").strip() for r in good)
    return scores, dec, dropped


def summarize(csv_path, kind: str = "auto", reject: float = 0.90, review: float = 0.60) -> str:
    """算好一段可读文本返回(命令行和 orchestrator 共用)。"""
    p = Path(csv_path)
    if not p.exists():
        return f"[缺文件] {csv_path} 不存在 —— 上游 predict 没写出来(--input 空/错 或 --model_root 没找到 .pth)。"
    scores, dec, dropped = _load(csv_path)
    n = len(scores)
    lines = [f"文件: {csv_path}"]
    if dropped:
        lines.append(f"注意: 剔除 {dropped} 张评分失败/未评分的图(不计入统计)。")
    if n == 0:
        lines.append("没有可用的 final_ai_score(空表或全部评分失败)。")
        return "\n".join(lines)

    n_reject = sum(1 for x in scores if x >= reject)
    n_review = sum(1 for x in scores if review <= x < reject)
    n_pass = sum(1 for x in scores if x < review)
    lines.append(f"张数: {n}")
    lines.append(f"ai_score  min={scores[0]:.3f}  median={scores[n // 2]:.3f}  "
                 f"mean={sum(scores) / n:.3f}  max={scores[-1]:.3f}")
    lines.append(f"决策(来自 decision 列): {dict(dec)}")
    lines.append(f"按阈值: Reject>={reject:.2f}: {n_reject}/{n}  |  "
                 f"Review[{review:.2f},{reject:.2f}): {n_review}/{n}  |  Pass<{review:.2f}: {n_pass}/{n}")
    if kind in ("genuine", "auto"):
        lines.append(f"[真图特异性] Pass 率={n_pass / n * 100:.1f}%  |  误杀(Reject)率={n_reject / n * 100:.1f}%  "
                     f"|  另有 {n_review} 张进人工复核({n_review / n * 100:.1f}%)")
    if kind in ("fake", "auto"):
        lines.append(f"[假图召回] Reject 率={n_reject / n * 100:.1f}%  |  漏检(Pass)率={n_pass / n * 100:.1f}%  "
                     f"|  另有 {n_review} 张进人工复核({n_review / n * 100:.1f}%)")
    return "\n".join(lines)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="汇总 SSP summary.csv")
    ap.add_argument("csv_path", help="predict 产出的 summary.csv 路径")
    ap.add_argument("--kind", choices=["genuine", "fake", "auto"], default="auto")
    ap.add_argument("--reject", type=float, default=0.90)
    ap.add_argument("--review", type=float, default=0.60)
    args = ap.parse_args()
    print(summarize(args.csv_path, args.kind, args.reject, args.review))


if __name__ == "__main__":
    main()
