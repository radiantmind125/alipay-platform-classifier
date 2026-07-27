r"""读 predict_all_models.py 产出的 summary.csv,打印分布 + 决策统计。

不用把 csv 贴回聊天,服务器本地直接算。
用法:
  python training/eval_summary.py D:\ssp_test\gen_clean_out\summary.csv --kind genuine
  python training/eval_summary.py D:\ssp_test\realfake2_out\summary.csv  --kind fake

--kind genuine -> 报"特异性"(Pass 率 = 1 - 误杀率),真图应尽量 Pass。
--kind fake    -> 报"召回"(Reject 率),假图应尽量 Reject。
阈值与 predict 一致:Reject>=0.90 / ManualReview 0.60-0.90 / Pass<0.60。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="汇总 SSP summary.csv")
    ap.add_argument("csv_path", help="predict 产出的 summary.csv 路径")
    ap.add_argument("--kind", choices=["genuine", "fake", "auto"], default="auto",
                    help="genuine 报特异性 / fake 报召回 / auto 两个都报")
    ap.add_argument("--reject", type=float, default=0.90)
    ap.add_argument("--review", type=float, default=0.60)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv_path, encoding="utf-8-sig")))
    if not rows:
        print("空文件或读不到行:", args.csv_path)
        return

    scores = sorted(float(r["final_ai_score"]) for r in rows if r.get("final_ai_score") not in (None, ""))
    dec = Counter((r.get("decision") or "").strip() for r in rows)
    n = len(scores)
    if n == 0:
        print("没有可用的 final_ai_score 列。字段:", list(rows[0].keys()))
        return

    n_reject = sum(1 for x in scores if x >= args.reject)
    n_review = sum(1 for x in scores if args.review <= x < args.reject)
    n_pass = sum(1 for x in scores if x < args.review)
    median = scores[n // 2]
    mean = sum(scores) / n

    print(f"文件: {args.csv_path}")
    print(f"张数: {n}")
    print(f"ai_score  min={scores[0]:.3f}  median={median:.3f}  mean={mean:.3f}  max={scores[-1]:.3f}")
    print(f"决策(来自 decision 列): {dict(dec)}")
    print(f"按阈值: Reject>={args.reject:.2f}: {n_reject}/{n}  |  "
          f"Review[{args.review:.2f},{args.reject:.2f}): {n_review}/{n}  |  "
          f"Pass<{args.review:.2f}: {n_pass}/{n}")

    if args.kind in ("genuine", "auto"):
        spec = n_pass / n
        killed = n_reject / n
        print(f"[真图特异性] Pass 率={spec*100:.1f}%  |  误杀(Reject)率={killed*100:.1f}%  "
              f"|  另有 {n_review} 张进人工复核({n_review/n*100:.1f}%)")
    if args.kind in ("fake", "auto"):
        recall = n_reject / n
        miss = n_pass / n
        print(f"[假图召回] Reject 率={recall*100:.1f}%  |  漏检(Pass)率={miss*100:.1f}%  "
              f"|  另有 {n_review} 张进人工复核({n_review/n*100:.1f}%)")


if __name__ == "__main__":
    main()
