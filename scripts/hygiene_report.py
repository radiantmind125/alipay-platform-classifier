"""整池数据体检报告：分辨率分布、分桶、翻拍占比、去重比、时间跨度、iOS/安卓基线比例。

可在服务器全量池重复运行，作为漂移监控的基线。只用标准库 + 本项目 numpy 模块。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.bootstrap import resolution_platform  # noqa: E402
from alipay_platform.grouping import cluster_by_dhash, parse_timestamp  # noqa: E402
from alipay_platform.photo_detector import photo_verdict_from_meta  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=Path, default=Path("runs/pool_20260701/inspect.jsonl"))
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in args.inspect.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if "error" not in r]
    err = [r for r in recs if "error" in r]
    n = len(ok)
    print(f"== 数据体检（{args.inspect.name}）==")
    print(f"可解析 {n} 张；解析失败/损坏 {len(err)} 张")

    # 分桶。
    buckets = Counter(resolution_platform(r["width"], r["height"]) for r in ok)
    print("\n分辨率自举分桶：")
    for k in ("ios", "android", "abstain"):
        print(f"  {k:<9} {buckets[k]:<6} = {buckets[k]/n*100:.1f}%")
    # 估计真实 iOS:安卓（模棱两可按经验~9成安卓折算，仅作基线）。
    est_ios = buckets["ios"] + int(buckets["abstain"] * 0.1)
    est_and = buckets["android"] + int(buckets["abstain"] * 0.9)
    print(f"  基线估计 iOS≈{est_ios/n*100:.0f}%  安卓≈{est_and/n*100:.0f}%（模棱两可按~9成安卓折算）")

    # 翻拍/非截图。
    photo = sum(1 for r in ok if photo_verdict_from_meta(r["width"], r["height"], has_capture_tags=bool(r.get("has_capture_tags"))).is_photo)
    print(f"\n翻拍/非截图（元数据）：{photo} = {photo/n*100:.2f}%")

    # 去重比（阈值 1 = 几乎一模一样的图；模板相似不算重复）。
    items = [(r["file"], int(r["dhash"], 16)) for r in ok if r.get("dhash")]
    uniq = len(set(cluster_by_dhash(items, threshold=1).values()))
    templ = len(set(cluster_by_dhash(items, threshold=6).values()))
    print(f"近重复族(阈值1)：{uniq}（真去重比 {len(items)/max(1,uniq):.2f}x）；"
          f"模板族(阈值6)：{templ}（模板高度雷同，故切分按族防泄漏）")

    # 分辨率 Top。
    res = Counter((r["width"], r["height"]) for r in ok)
    print("\n分辨率 Top10：")
    for (w, h), c in res.most_common(10):
        tag = resolution_platform(w, h)
        print(f"  {w}x{h:<6} {c:<5} [{tag}]")

    # 时间跨度。
    tss = sorted(t for t in (parse_timestamp(r["file"]) for r in ok) if t)
    if tss:
        print(f"\n时间跨度：{tss[0]} ~ {tss[-1]}（可用于时间切分）")


if __name__ == "__main__":
    main()
