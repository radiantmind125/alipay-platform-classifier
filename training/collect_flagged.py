r"""从 predict 的 summary.csv 里挑出被判"假"的图(误杀候选),按分数命名拷进一个文件夹,供人工逐张核查。

为什么要看:真图测试集里被判 Reject 的,可能是**真误杀**,也可能是**图库里混进来、没标注的真假图**
(比如去了 EXIF 的翻拍)。逐张看一眼就能把两者分开 —— 真误杀该降阈值/加增广,混进来的假图反而是白捡的真样本。

用法:
  # 默认挑 final_ai_score>=0.90(Reject 那批)
  python training/collect_flagged.py D:\ssp_test\gen10k_out\summary.csv --out D:\ssp_test\gen10k_flagged
  # 想连 0.60-0.90 的边缘复核区一起看:
  python training/collect_flagged.py D:\ssp_test\gen10k_out\summary.csv --out D:\ssp_test\gen10k_flagged --min 0.60
  # 只要分最高的前 50 张:--limit 50
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="挑出被判假的图(误杀候选)拷出来核查")
    ap.add_argument("csv_path", help="predict 产出的 summary.csv")
    ap.add_argument("--out", type=Path, required=True, help="误杀候选拷到这里")
    ap.add_argument("--min", type=float, default=0.90, help="分数下限(含),默认 0.90=Reject")
    ap.add_argument("--max", type=float, default=1.01, help="分数上限(含)")
    ap.add_argument("--limit", type=int, default=0, help="只取分最高的前 N 张;0=全部")
    ap.add_argument("--force", action="store_true", help="输出目录非空也清空重建")
    args = ap.parse_args()

    p = Path(args.csv_path)
    if not p.exists():
        sys.exit(f"找不到 {args.csv_path}(上游 predict 没写出来?先确认 summary.csv 存在)。")

    rows = list(csv.DictReader(open(args.csv_path, encoding="utf-8-sig")))
    picked = []
    for r in rows:
        v = r.get("final_ai_score")
        if v in (None, ""):
            continue
        try:
            s = float(v)
        except ValueError:
            continue
        if args.min <= s <= args.max:
            picked.append((s, r.get("image") or r.get("image_name") or "", (r.get("decision") or "").strip()))
    picked.sort(key=lambda t: t[0], reverse=True)
    if args.limit > 0:
        picked = picked[:args.limit]

    # 输出目录:拒删磁盘根;非空需 --force
    d = args.out
    if len(d.parts) < 2:
        sys.exit(f"拒绝: 输出目录 {d} 太浅/像磁盘根, 不删。")
    if d.exists() and any(d.iterdir()) and not args.force:
        sys.exit(f"输出目录已存在且非空: {d}。加 --force 才清空重建(先确认是核查目录)。")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

    copied = missing = 0
    listing = []
    for s, img, dec in picked:
        listing.append((s, dec, img))
        src = Path(img)
        if src.exists():
            # 分数前缀:文件名排序即按分数聚拢
            shutil.copy2(src, d / f"{s:.3f}_{src.name}")
            copied += 1
        else:
            missing += 1

    # 写一份清单
    with open(d / "flagged_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["final_ai_score", "decision", "image"])
        for s, dec, img in listing:
            w.writerow([f"{s:.6f}", dec, img])

    print(f"命中 {len(picked)} 张(分数 {args.min}~{args.max})。拷出 {copied} 张 -> {d}"
          + (f";另有 {missing} 张源文件找不到" if missing else ""))
    print("清单: " + str(d / "flagged_list.csv"))
    if listing:
        print("分数最高的几张:")
        for s, dec, img in listing[:15]:
            print(f"  {s:.3f}  {dec}  {Path(img).name}")
    print("请逐张看:真误杀?还是图库里混进来没标注的真假图(翻拍/生成)?后者是白捡的真样本,可加进召回集。")


if __name__ == "__main__":
    main()
