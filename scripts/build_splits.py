"""大规模防泄漏切分：按“近重复族”整组分配 train/val/test，并留出时间测试集与对抗测试集。

- 组内绝不跨切分（GroupKFold 思路），杜绝近重复模板泄漏。
- val = 人工金标；对抗测试 = 金标里“模棱两可（缩放 iPhone / 非常规安卓）”子集（真正头号指标）。
- 时间测试 = 最近 N% 时间窗的自举标注图（从 train 中留出）。
- 纯模棱两可且无金标的 -> predict（留给门控自训练打伪标签）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.bootstrap import resolution_platform  # noqa: E402
from alipay_platform.grouping import cluster_by_dhash, parse_timestamp, temporal_cutoff  # noqa: E402

LABEL_ID = {"android": 0, "ios": 1}


def _crop_path(crop_dir: Path, file: str) -> Path:
    return crop_dir / f"{Path(file).with_suffix('').as_posix().replace('/', '__')}.png"


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=Path, default=Path("runs/pool_20260701/inspect.jsonl"))
    ap.add_argument("--bootstrap", type=Path, default=Path("runs/pool_20260701/bootstrap_labels.jsonl"))
    ap.add_argument("--gold", type=Path, default=Path("gold/gold_seed_v2.json"))
    ap.add_argument("--crops", type=Path, default=Path("runs/pool_20260701/status_bar"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/data"))
    ap.add_argument("--temporal-holdout", type=float, default=0.15)
    # 注意：支付宝成功页是近乎同一模板，全局 dHash 会按“模板”而非“交易”聚类；阈值 6 会把
    # 几千张并成一坨。真正的近重复只在阈值 0~1。真·交易族分组应改用 mate 的 OCR 字段
    #（金额+时间+收款人）。这里用 1 只去“几乎一模一样”的重复图。
    ap.add_argument("--dhash-threshold", type=int, default=1)
    args = ap.parse_args(argv)

    recs = {json.loads(l)["file"]: json.loads(l) for l in args.inspect.read_text(encoding="utf-8").splitlines() if l.strip()}
    boot = {json.loads(l)["file"]: json.loads(l)["label"] for l in args.bootstrap.read_text(encoding="utf-8").splitlines() if l.strip()}
    gold = {g["file"]: g["platform"] for g in json.loads(args.gold.read_text(encoding="utf-8"))["labels"]}

    files = [f for f, r in recs.items() if r.get("dhash")]
    items = [(f, int(recs[f]["dhash"], 16)) for f in files]
    group = cluster_by_dhash(items, threshold=args.dhash_threshold)

    ts = {f: parse_timestamp(f) for f in files}
    labeled_ts = [ts[f] for f in files if boot.get(f) in LABEL_ID and ts[f]]
    cutoff = temporal_cutoff(labeled_ts, args.temporal_holdout)

    def is_ambiguous(f: str) -> bool:
        r = recs[f]
        return resolution_platform(r["width"], r["height"]) == "abstain"

    # 组内属性汇总。
    g_has_gold: dict[int, bool] = defaultdict(bool)
    g_max_ts: dict[int, str] = defaultdict(str)
    for f in files:
        gid = group[f]
        if f in gold:
            g_has_gold[gid] = True
        if ts[f] and ts[f] > g_max_ts[gid]:
            g_max_ts[gid] = ts[f]

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test_temporal": [], "test_adversarial": [], "predict": []}
    for f in files:
        gid = group[f]
        crop = _crop_path(args.crops, f)
        if not crop.exists():
            continue
        row = {"strip": crop.as_posix(), "file": f}
        if g_has_gold[gid]:
            if f in gold:
                row["label"] = LABEL_ID[gold[f]]
                row["platform"] = gold[f]
                splits["val"].append(row)
                if is_ambiguous(f):
                    splits["test_adversarial"].append(row)   # 头号指标子集
            continue
        lab = boot.get(f)
        if lab in LABEL_ID:
            row["label"] = LABEL_ID[lab]
            row["platform"] = lab
            if cutoff and g_max_ts[gid] >= cutoff:
                splits["test_temporal"].append(row)
            else:
                splits["train"].append(row)
        elif lab == "abstain":
            splits["predict"].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 泄漏校验：同一 group 不得跨 train/test_temporal；同一文件不得进多个切分。
    file_splits: dict[str, list[str]] = defaultdict(list)
    for name, rows in splits.items():
        for r in rows:
            if name != "test_adversarial":  # 对抗集是 val 的子集，允许重叠
                file_splits[r["file"]].append(name)
    dup = {f: s for f, s in file_splits.items() if len(s) > 1}
    train_groups = {group[r["file"]] for r in splits["train"]}
    temporal_groups = {group[r["file"]] for r in splits["test_temporal"]}
    cross = train_groups & temporal_groups

    def cnt(rows):
        c = {"android": 0, "ios": 0}
        for r in rows:
            if r.get("platform") in c:
                c[r["platform"]] += 1
        return c

    total_groups = len(set(group.values()))
    print(f"总文件 {len(files)}，近重复族 {total_groups}（去重比 {len(files)/max(1,total_groups):.2f}x）")
    print(f"时间切点(最近 {args.temporal_holdout:.0%}): {cutoff}")
    for name in ("train", "val", "test_temporal", "test_adversarial", "predict"):
        print(f"  {name:<16} {len(splits[name]):<6} {cnt(splits[name]) if name!='predict' else ''}")
    print(f"\n泄漏校验：跨切分文件 {len(dup)}（应为 0）；train∩temporal 组交集 {len(cross)}（应为 0）")
    print(f"清单写入 {args.out_dir}/")


if __name__ == "__main__":
    main()
