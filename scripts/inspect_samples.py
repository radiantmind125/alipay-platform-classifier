"""对真实样本做“真跑”检查：元数据投票、银标决定、几何裁剪、若干便宜特征。

只依赖 numpy + Pillow，因此在没有 torch / OpenCV 的机器上也能真实运行。用途：拿到真实的
安卓 / 苹果支付宝转账成功截图后，一条命令验证：
  1. 元数据标注函数在真图上是否如预期（iPhone 分辨率白名单命中率、Display-P3、EXIF）；
  2. 几何状态栏裁剪是否真的框住了状态栏（会把裁剪写盘供肉眼核对）；
  3. 若干只用 numpy 就能算的便宜特征在安卓 vs 苹果之间是否有区分度（探索性）。

约定：把图片放到 data/samples/ 下，可按 ios/ 与 android/ 子目录组织（子目录名即已知标签）；
放在 unknown/ 或直接平铺则标签视为未知。命令：
    python scripts/inspect_samples.py --input data/samples --output runs/inspect_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# 允许从仓库根直接运行脚本。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alipay_platform.cache_regions import iter_image_paths, load_upright_rgb  # noqa: E402
from alipay_platform.hashing import dhash  # noqa: E402
from alipay_platform.labeling import aggregate  # noqa: E402
from alipay_platform.metadata_seed import metadata_votes, read_metadata_facts  # noqa: E402
from alipay_platform.regions import status_bar_strip  # noqa: E402

# 目录名 -> 已知标签。
_LABEL_DIRS = {
    "ios": "ios",
    "苹果": "ios",
    "apple": "ios",
    "iphone": "ios",
    "android": "android",
    "安卓": "android",
}


def infer_label(path: Path, root: Path) -> str:
    """从样本相对 root 的任一层父目录名推断已知标签，推断不出则 'unknown'。"""
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    for part in parts:
        if part in _LABEL_DIRS:
            return _LABEL_DIRS[part]
    return "unknown"


def status_bar_features(strip_rgb: np.ndarray) -> dict[str, float]:
    """只用 numpy 的几个便宜描述子（探索性，用来看安卓 vs 苹果是否有区分度）。"""
    gray = strip_rgb.mean(axis=2)
    # 横向梯度能量，按左/中/右三等分统计——右侧图标越密（安卓双卡 + 网络文字）能量越高。
    edges = np.abs(np.diff(gray, axis=1))
    width = edges.shape[1]
    third = max(1, width // 3)
    left = float(edges[:, :third].mean())
    mid = float(edges[:, third : 2 * third].mean())
    right = float(edges[:, 2 * third :].mean())
    total = left + mid + right + 1e-6
    return {
        "luma_mean": round(float(gray.mean()), 2),        # 深色 iOS 状态栏 vs 浅色
        "edge_left": round(left, 3),
        "edge_mid": round(mid, 3),
        "edge_right": round(right, 3),
        "edge_right_ratio": round(right / total, 3),       # 右侧图标密度占比
        "strip_h": int(strip_rgb.shape[0]),
        "strip_w": int(strip_rgb.shape[1]),
    }


def _save(strip_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip_rgb).save(path)


def _shard_of(relative_path: Path, shard_count: int) -> int:
    """按相对路径稳定哈希分片，便于多进程并行。"""
    import hashlib

    digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def inspect(input_dir: Path, output_dir: Path, strip_fraction: float,
            shard_index: int = 0, shard_count: int = 1) -> list[dict]:
    records: list[dict] = []
    for source in iter_image_paths(input_dir):
        if shard_count > 1 and _shard_of(source.relative_to(input_dir), shard_count) != shard_index:
            continue
        label = infer_label(source, input_dir)
        stem = source.relative_to(input_dir).with_suffix("").as_posix().replace("/", "__")
        try:
            facts = read_metadata_facts(source)
            votes = metadata_votes(facts)
            decision = aggregate(votes)
            image = load_upright_rgb(source)
            strip = status_bar_strip(image, fraction=strip_fraction)
            _save(strip.image, output_dir / "status_bar" / f"{stem}.png")
            record = {
                "file": source.relative_to(input_dir).as_posix(),
                "known_label": label,
                "width": facts.width,
                "height": facts.height,
                "make": facts.make,
                "has_capture_tags": facts.has_capture_tags,
                "has_icc": facts.icc_profile is not None,
                "resolution_vote": votes[0].label,
                "icc_vote": votes[1].label,
                "exif_vote": votes[2].label,
                "silver": decision.outcome,
                "silver_label": decision.label,
                "dhash": f"{dhash(Image.fromarray(image)):016x}",
                "features": status_bar_features(strip.image),
            }
        except Exception as error:  # noqa: BLE001 - 记录并继续
            record = {"file": source.relative_to(input_dir).as_posix(), "error": f"{type(error).__name__}: {error}"}
        records.append(record)
    return records


def _print_pool_summary(ok: list[dict]) -> None:
    """图片池分布概览——不需要标签，直接从真实原始池估计关键分布。

    这些数字正好回答计划里的开放问题：池子的真实 iOS:安卓 比例、翻拍 vs 直接截图的占比、
    元数据是否被保留。
    """
    n = len(ok)
    if not n:
        return
    print("\n== 图片池分布概览（无标签也适用）==")
    print(f"  成功解析 {n} 张")

    # 尺寸分布（按短边,长边归一，忽略横竖）。
    dims: dict[tuple[int, int], int] = defaultdict(int)
    for r in ok:
        dims[(min(r["width"], r["height"]), max(r["width"], r["height"]))] += 1
    print("  尺寸 Top（短边x长边）：")
    for (short, long), count in sorted(dims.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {short}x{long:<6} x {count}")

    whitelist = sum(1 for r in ok if r["resolution_vote"] == "ios")
    p3 = sum(1 for r in ok if r["icc_vote"] == "ios")
    photo = sum(1 for r in ok if r["has_capture_tags"])
    print(f"  命中 iPhone 分辨率白名单：{whitelist}/{n}（≈ iOS 占比下界；改过尺寸的 iOS 不会命中，故偏低）")
    print(f"  带 Display-P3 色彩标：{p3}/{n}")
    print(f"  疑似翻拍（含相机 EXIF）：{photo}/{n}；直接截图：{n - photo}/{n}")

    silver: dict[str, int] = defaultdict(int)
    for r in ok:
        silver[r["silver"]] += 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(silver.items()))
    print(f"  银标结果：{parts}")
    print("  说明：元数据本身无法投“安卓”，所以 accept:android 恒为 0——安卓要靠状态栏/对勾像素判定。")


def _print_report(records: list[dict]) -> None:
    ok = [r for r in records if "error" not in r]
    bad = [r for r in records if "error" in r]
    print(f"\n共 {len(records)} 张，成功 {len(ok)}，失败 {len(bad)}\n")
    header = f"{'文件':<28}{'标签':<8}{'尺寸':<12}{'分辨率投票':<10}{'ICC':<6}{'EXIF':<7}{'银标':<20}{'右侧密度':<8}"
    print(header)
    print("-" * len(header))
    for r in ok:
        size = f"{r['width']}x{r['height']}"
        feats = r.get("features", {})
        print(
            f"{r['file'][:27]:<28}{r['known_label']:<8}{size:<12}{r['resolution_vote']:<10}"
            f"{r['icc_vote']:<6}{r['exif_vote']:<7}{r['silver']:<20}{feats.get('edge_right_ratio', ''):<8}"
        )
    for r in bad:
        print(f"[失败] {r['file']}: {r['error']}")

    _print_pool_summary(ok)

    # 若有已知标签，给出元数据标注函数的真实命中情况。
    labeled = [r for r in ok if r["known_label"] in ("ios", "android")]
    if labeled:
        print("\n== 元数据标注函数在已知标签上的表现 ==")
        by_label: dict[str, list[dict]] = defaultdict(list)
        for r in labeled:
            by_label[r["known_label"]].append(r)
        for lbl, rows in sorted(by_label.items()):
            res_hit = sum(1 for r in rows if r["resolution_vote"] == "ios")
            icc_hit = sum(1 for r in rows if r["icc_vote"] == "ios")
            exif_hit = sum(1 for r in rows if r["exif_vote"] == "ios")
            avg_right = np.mean([r["features"]["edge_right_ratio"] for r in rows]) if rows else 0.0
            avg_luma = np.mean([r["features"]["luma_mean"] for r in rows]) if rows else 0.0
            print(
                f"  {lbl:<8} n={len(rows):<3} 分辨率投苹果={res_hit:<3} P3={icc_hit:<3} EXIF苹果={exif_hit:<3}"
                f" 右侧密度均值={avg_right:.3f} 亮度均值={avg_luma:.1f}"
            )
        ios_rows = by_label.get("ios", [])
        android_rows = by_label.get("android", [])
        if ios_rows:
            recall = sum(1 for r in ios_rows if r["resolution_vote"] == "ios") / len(ios_rows)
            print(f"\n  分辨率白名单对 iOS 的召回：{recall:.0%}（低召回是设计如此，只求高精度）")
        if android_rows:
            false_hits = sum(1 for r in android_rows if r["resolution_vote"] == "ios")
            print(f"  分辨率白名单在安卓上的误命中：{false_hits}/{len(android_rows)}（应为 0，否则要收紧白名单）")


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="对真实样本做真跑检查（元数据 + 几何裁剪 + 便宜特征）")
    parser.add_argument("--input", type=Path, default=Path("data/samples"))
    parser.add_argument("--output", type=Path, default=Path("runs/inspect_v1"))
    parser.add_argument("--strip-fraction", type=float, default=0.06, help="状态栏条占图高比例")
    parser.add_argument("--shard-index", type=int, default=0, help="本进程处理的分片号 [0, shard-count)")
    parser.add_argument("--shard-count", type=int, default=1, help="总分片数（多开几个进程并行）")
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"输入目录不存在：{args.input}")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index 必须在 [0, --shard-count) 内")
    records = inspect(args.input, args.output, args.strip_fraction, args.shard_index, args.shard_count)
    if not records:
        raise SystemExit(f"{args.input} 下（本分片）没有图片。检查路径或 --shard-count。")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.shard_count > 1:
        # 各分片写各自的 jsonl；跑完把所有分片合并成 inspect.jsonl（见 RUNBOOK）。状态栏条同写一个目录，文件名唯一无冲突。
        manifest = args.output / f"inspect.shard-{args.shard_index:03d}-of-{args.shard_count:03d}.jsonl"
    else:
        manifest = args.output / "inspect.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.shard_count > 1:
        ok = sum(1 for r in records if "error" not in r)
        print(f"分片 {args.shard_index+1}/{args.shard_count} 完成：{len(records)} 条（成功 {ok}）-> {manifest}")
        print("所有分片跑完后，合并：Get-Content <out>\\inspect.shard-*.jsonl | Set-Content <out>\\inspect.jsonl")
    else:
        _print_report(records)
        print(f"\n明细已写入 {manifest}，状态栏裁剪在 {args.output / 'status_bar'}/ 下（可肉眼核对）")


if __name__ == "__main__":
    main()
