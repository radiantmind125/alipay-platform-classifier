"""收尾：把 Tier-0(分辨率) + Tier-1(CNN/分类器) 合并成每图最终设备标签，并叠加欺诈融合评分。

输入：
  --bootstrap  bootstrap_labels.jsonl（ios/android/abstain）
  --cnn        对 abstain 的分类结果 {file, device, conf, p_ios}（上线端 infer_cpu 的产物；
               本机可用之前 train_strip_classifier 产的 device_predictions.jsonl 演示）
  --inspect    inspect.jsonl（提供 EXIF/尺寸，用于翻拍与 EXIF 不一致信号）
  --checkmark  可选，mate 检测器给的 {file, has_checkmark}（“没有勾”信号）
产出：final_device.jsonl {file, device, source, confidence, fraud_score, fraud_verdict, fraud_reasons}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.bootstrap import resolution_platform  # noqa: E402
from alipay_platform.fusion import (  # noqa: E402
    FraudSignals,
    device_prior_conflict,
    fraud_score,
    merge_device,
)
from alipay_platform.photo_detector import photo_verdict_from_meta  # noqa: E402


def _load_jsonl(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    return {json.loads(l)["file"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=Path, default=Path("runs/pool_20260701/bootstrap_labels.jsonl"))
    ap.add_argument("--cnn", type=Path, default=Path("runs/pool_20260701/device_predictions.jsonl"))
    ap.add_argument("--inspect", type=Path, default=Path("runs/pool_20260701/inspect.jsonl"))
    ap.add_argument("--checkmark", type=Path, help="可选：mate 检测器 {file, has_checkmark}")
    ap.add_argument("--out", type=Path, default=Path("runs/pool_20260701/final_device.jsonl"))
    args = ap.parse_args(argv)

    boot = {json.loads(l)["file"]: json.loads(l)["label"] for l in args.bootstrap.read_text(encoding="utf-8").splitlines() if l.strip()}
    cnn = _load_jsonl(args.cnn)
    meta = _load_jsonl(args.inspect)
    check = _load_jsonl(args.checkmark)

    dev_counts: Counter[str] = Counter()
    fraud_counts: Counter[str] = Counter()
    rows: list[dict] = []
    for f, lab in boot.items():
        m = merge_device(lab, cnn.get(f))
        dev_counts[m["device"]] += 1

        r = meta.get(f, {})
        w, h = r.get("width", 0), r.get("height", 0)
        res_plat = resolution_platform(w, h) if w and h else "abstain"
        is_photo = photo_verdict_from_meta(w, h, has_capture_tags=bool(r.get("has_capture_tags"))).is_photo if w and h else False
        exif_mismatch = bool(r.get("has_capture_tags")) and res_plat == "ios"   # 声称 iPhone 却有相机字段
        conflict = False
        c = cnn.get(f)
        if c is not None:
            conflict = device_prior_conflict(res_plat, c.get("device", ""), float(c.get("conf", 0.0)))
        no_check = (f in check) and (check[f].get("has_checkmark") is False)

        fr = fraud_score(FraudSignals(no_checkmark=no_check, device_prior_conflict=conflict,
                                      photo_of_screen=is_photo, exif_device_mismatch=exif_mismatch))
        fraud_counts[fr.verdict] += 1
        rows.append({"file": f, "device": m["device"], "source": m["source"], "confidence": m["confidence"],
                     "fraud_score": fr.score, "fraud_verdict": fr.verdict, "fraud_reasons": fr.reasons})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fo:
        for row in rows:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"最终设备标签 {n} 张：")
    for k in ("ios", "android", "uncertain", "unknown"):
        if dev_counts[k]:
            print(f"  {k:<10} {dev_counts[k]:<6} = {dev_counts[k]/n*100:.1f}%")
    print("欺诈融合裁决：")
    for k in ("pass", "review", "reject"):
        print(f"  {k:<8} {fraud_counts[k]:<6} = {fraud_counts[k]/n*100:.2f}%")
    print(f"\n写入 {args.out}")
    if not check:
        print("提示：未提供 --checkmark（mate 检测器的‘有无勾’），‘没有勾=假图’信号本次未参与。")


if __name__ == "__main__":
    main()
