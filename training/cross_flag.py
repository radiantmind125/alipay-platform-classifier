r"""用**两条线交叉验证**从"真图"池里挖出漏网的假图 —— 不用逐张人工看。

依据(实测): 线路A(整图AI, v7)和线路B(局部改金额, localdet3)是**两个独立训练、架构和训练数据都不同**的模型。
标定自动拒阈值时发现: 两条线各自分数最高的那批图里**有 6 张是同一批文件**。
模型各自的癖好不会跨模型复现 —— **两条线同时点名同一张"真图", 基本就是这张图真的有问题。**
人工抽查也印证了: 那批里 14/20 查实是假图(11 张带「豆包AI生成」水印)。

所以: 交叉命中 = 高置信假图候选, 可以批量清池子, 不必一张张看。

脚本还会算一个**巧合基准**: 若两条线互相独立, 交叉命中数应该约等于 (A命中率 x B命中率 x 总数)。
实际值远大于这个基准, 才说明是真信号而不是碰巧。

用法:
  python training/cross_flag.py --a D:\probe\genuine_20k_v7\summary.csv \
      --b D:\probe\genuine_20k_ld3\summary.csv --out D:\probe\cross_flagged.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load(p: Path, col: str, require_located: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if require_located and (r.get("roi_amount_located") or "1").strip() != "1":
            continue
        name = (r.get("image_name") or "").strip() or Path((r.get("image") or "")).name
        v = (r.get(col) or "").strip()
        if not name or not v:
            continue
        try:
            out[name] = float(v)
        except ValueError:
            pass
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="两条线交叉命中 = 高置信假图(批量清池子)")
    ap.add_argument("--a", type=Path, required=True, help="线路A(v7 整图)的 summary.csv")
    ap.add_argument("--b", type=Path, required=True, help="线路B(localdet3 局部)的 summary.csv")
    ap.add_argument("--a-col", default="final_ai_score")
    ap.add_argument("--b-col", default="tile_top3")
    ap.add_argument("--a-thr", type=float, default=0.90, help="线路A 算高分的门槛")
    ap.add_argument("--b-thr", type=float, default=0.90, help="线路B 算高分的门槛")
    ap.add_argument("--out", type=Path, default=None, help="把交叉命中的文件名写到这里(可直接喂 --exclude)")
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    A = _load(args.a, args.a_col, False)
    B = _load(args.b, args.b_col, True)      # 线路B 只在金额定位成功时才有意义
    both_seen = set(A) & set(B)
    if not both_seen:
        raise SystemExit("两个 summary 没有共同的图片(检查 image_name 列)")

    a_hi = {n for n in both_seen if A[n] >= args.a_thr}
    b_hi = {n for n in both_seen if B[n] >= args.b_thr}
    cross = a_hi & b_hi

    n = len(both_seen)
    exp = len(a_hi) * len(b_hi) / n if n else 0      # 若两条线独立, 交叉命中的期望个数
    print(f"两条线都评过的图: {n} 张")
    print(f"  线路A >= {args.a_thr}: {len(a_hi)} 张 ({len(a_hi)/n*100:.2f}%)")
    print(f"  线路B >= {args.b_thr}: {len(b_hi)} 张 ({len(b_hi)/n*100:.2f}%)")
    print(f"  **两条线都命中: {len(cross)} 张**")
    print(f"  若两条线互相独立, 巧合应只有约 {exp:.1f} 张 -> 实际是它的 {(len(cross)/exp if exp else 0):.0f} 倍")
    if exp and len(cross) > 3 * exp:
        print("  => 远超巧合基准, 说明这是真信号: 这批基本可以当高置信假图处理。")
    else:
        print("  => 没有明显超过巧合基准, 交叉命中的证据力不强, 建议还是人工抽查。")

    ranked = sorted(cross, key=lambda x: -(A[x] + B[x]))
    print(f"\n交叉命中(按两条线分数之和排序, 显示前 {min(args.show, len(ranked))} 张):")
    print("  线路A    线路B   文件名")
    for nm in ranked[:args.show]:
        print(f"  {A[nm]:.4f}  {B[nm]:.4f}  {nm}")

    if args.out:
        args.out.write_text("\n".join(ranked) + "\n", encoding="utf-8")
        print(f"\n已写出 {len(ranked)} 个文件名 -> {args.out}")
        print("用法: 人工抽查确认几张后, 直接当 autoreject_threshold.py 的 --exclude 用。")
    print("\n提醒: 这是**候选**不是判决。抽查几张确认规则可靠, 再整批当假图处理。")


if __name__ == "__main__":
    main()
