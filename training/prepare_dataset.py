"""从自举标签 + 缓存的状态栏条，构建“训练/验证”清单，供 GPU 上的 tiny-CNN 训练用。

原则（大规模防泄漏）：
- 训练集 = 分辨率自举给出的 ios/android（高精度、免费、可达数万级）。
- 验证/测试集 = 人工金标（gold-only），绝不混入银标。
- 用 dHash 去重，并剔除任何与金标近重复的训练图，防止验证泄漏。
- 标签仅来自分辨率（弱标注），进入 CNN 的只有“状态栏条像素”，分辨率不作特征。

只用 numpy + PIL，可在任意机器上先跑好清单，再拷到 GPU 服务器训练。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.hashing import hamming_distance  # noqa: E402

LABEL_ID = {"android": 0, "ios": 1}


def _crop_path(crop_dir: Path, file: str) -> Path:
    stem = Path(file).with_suffix("").as_posix().replace("/", "__")
    return crop_dir / f"{stem}.png"


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=Path, default=Path("runs/pool_20260701/bootstrap_labels.jsonl"))
    ap.add_argument("--inspect", type=Path, default=Path("runs/pool_20260701/inspect.jsonl"), help="提供每图 dhash")
    ap.add_argument("--gold", type=Path, default=Path("gold/gold_seed_v2.json"))
    ap.add_argument("--crops", type=Path, default=Path("runs/pool_20260701/status_bar"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/data"))
    ap.add_argument("--dhash-threshold", type=int, default=6, help="与金标 dHash 汉明距离<=此值即视为近重复并剔除")
    args = ap.parse_args(argv)

    boot = [json.loads(l) for l in args.bootstrap.read_text(encoding="utf-8").splitlines() if l.strip()]
    dhash = {json.loads(l)["file"]: json.loads(l).get("dhash") for l in args.inspect.read_text(encoding="utf-8").splitlines() if l.strip()}
    gold = json.loads(args.gold.read_text(encoding="utf-8"))["labels"]
    gold_files = {g["file"] for g in gold}
    gold_hashes = [int(dhash[g["file"]], 16) for g in gold if g["file"] in dhash and dhash.get(g["file"])]

    def near_gold(file: str) -> bool:
        h = dhash.get(file)
        if not h:
            return False
        hv = int(h, 16)
        return any(hamming_distance(hv, gh) <= args.dhash_threshold for gh in gold_hashes)

    # 训练集：分辨率给出 ios/android，且不是金标、不与金标近重复；再按 dHash 去重。
    seen_hash: set[str] = set()
    train: list[dict] = []
    dropped_leak = dropped_dup = missing_crop = 0
    for r in boot:
        f, lab = r["file"], r["label"]
        if lab not in LABEL_ID or f in gold_files:
            continue
        if near_gold(f):
            dropped_leak += 1
            continue
        h = dhash.get(f)
        if h and h in seen_hash:
            dropped_dup += 1
            continue
        if h:
            seen_hash.add(h)
        crop = _crop_path(args.crops, f)
        if not crop.exists():
            missing_crop += 1
            continue
        train.append({"strip": crop.as_posix(), "label": LABEL_ID[lab], "platform": lab, "file": f})

    # 验证/测试集：金标。
    val: list[dict] = []
    for g in gold:
        if g["platform"] not in LABEL_ID:
            continue
        crop = _crop_path(args.crops, g["file"])
        if not crop.exists():
            continue
        val.append({"strip": crop.as_posix(), "label": LABEL_ID[g["platform"]], "platform": g["platform"], "file": g["file"]})

    # 模棱两可（分辨率弃权）——留作自训练/伪标签的预测集。
    predict = [{"file": r["file"], "strip": _crop_path(args.crops, r["file"]).as_posix()}
               for r in boot if r["label"] == "abstain" and _crop_path(args.crops, r["file"]).exists()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("predict", predict)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def counts(rows):
        c = {"android": 0, "ios": 0}
        for r in rows:
            if r.get("platform") in c:
                c[r["platform"]] += 1
        return c

    print(f"train: {len(train)}  {counts(train)}  (去重丢弃 {dropped_dup}，防泄漏丢弃 {dropped_leak}，缺条 {missing_crop})")
    print(f"val(gold): {len(val)}  {counts(val)}")
    print(f"predict(ambiguous): {len(predict)}")
    print(f"清单写入 {args.out_dir}/（train.jsonl / val.jsonl / predict.jsonl）")
    print("说明：大规模时验证集应扩到数百张金标；train 可达数万级免费标签。")


if __name__ == "__main__":
    main()
