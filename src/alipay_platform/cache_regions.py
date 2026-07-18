"""对原始图片池做一次性的区域缓存。

只付一次区域定位的代价，把每张图下游标注和分类都要用的小裁剪缓存下来：
  - 整宽状态栏条（需求里的区域一）
  - 时钟子裁剪（SF Pro vs Roboto 特征的输入）
  - 成功对勾裁剪（需求里的区域二）
外加零解码的元数据（宽高 / EXIF / ICC）、元数据银标决定，以及用于按交易分组的 dHash。

CPU 纪律：
  * 这里绝不构造 PaddleOCR——平台判别是视觉信号。
  * 状态栏从 EXIF 摆正后的“原图全分辨率”数组上按几何位置裁剪，绝不从纠正/缩放/JPEG 图上
    裁（那会抹掉“像素差”）。
  * 检测器是可选的，只当作缩小输入尺寸后的“粗定位候选”。没有它时，状态栏仍由几何位置得到，
    对勾则留空（后续用模板定位补上）。

torch / torchvision / 同项目 ``transfer_receipt_ai`` 只在提供了 ``--detector`` 时才惰性
导入，因此几何 + 元数据这条路只需要 numpy + Pillow。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from .hashing import dhash
from .labeling import aggregate
from .metadata_seed import metadata_votes, read_metadata_facts
from .regions import (
    Crop,
    checkmark_crop_from_status_box,
    clock_crop_from_time_box,
    status_bar_strip,
    status_bar_strip_from_time_box,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_paths(root: Path) -> Iterable[Path]:
    """递归枚举支持的图片（本地实现，等价于检测器工程里的同名函数）。"""
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def load_upright_rgb(path: Path) -> np.ndarray:
    """EXIF 摆正后的原图全分辨率 RGB 数组（只用 Pillow，不用 OpenCV）。"""
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def _selection_key(relative_path: Path) -> bytes:
    return hashlib.sha256(relative_path.as_posix().encode("utf-8")).digest()


def _shard_for(relative_path: Path, shard_count: int) -> int:
    return int.from_bytes(_selection_key(relative_path)[:8], "big") % shard_count


def _save_crop(crop: Crop | None, path: Path) -> str | None:
    if crop is None or crop.image.size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    # 用 PNG：无损，绝不会把刚分离出来的抗锯齿再抹一遍。
    Image.fromarray(crop.image).save(path)
    return path.as_posix()


class _Proposer:
    """可选的、基于检测器的区域候选（需要 torch/torchvision）。"""

    def __init__(self, checkpoint: Path, *, device: str, min_size: int, max_size: int) -> None:
        # 惰性导入，让几何这条路不需要 torch。
        from transfer_receipt_ai.model import LRCNNConfig, LRCNNPredictor  # type: ignore

        self._predict = LRCNNPredictor(
            checkpoint,
            device=device,
            score_threshold=0.30,
            model_config=LRCNNConfig(min_size=min_size, max_size=max_size),
        )

    def boxes(self, image_rgb: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
        """返回每个类别的最佳框，坐标在原图数组坐标系下。

        这里刻意喂未纠正的摆正原图：Faster R-CNN 内部会自己缩放并把框映射回输入坐标，所以
        不需要单应矩阵，也彻底跳过了有风险的屏幕四边形透视纠正。
        """
        result: dict[str, tuple[float, float, float, float]] = {}
        for detection in self._predict.predict(image_rgb):
            result[detection.label] = detection.bbox_xyxy
        return result


@dataclass
class CacheStats:
    written: int = 0
    skipped: int = 0
    failed: int = 0


def process_one(
    source_path: Path,
    relative_path: Path,
    crop_dir: Path,
    proposer: _Proposer | None,
) -> dict[str, object]:
    """构造一条清单记录（顺带把裁剪写盘）。"""
    facts = read_metadata_facts(source_path)
    votes = metadata_votes(facts)
    decision = aggregate(votes)

    image = load_upright_rgb(source_path)
    stem = relative_path.with_suffix("").as_posix().replace("/", "__")

    boxes = proposer.boxes(image) if proposer is not None else {}

    time_box = boxes.get("time")
    if time_box is not None:
        status_crop = status_bar_strip_from_time_box(image, time_box)
        clock_crop = clock_crop_from_time_box(image, time_box)
    else:
        status_crop = status_bar_strip(image)
        clock_crop = None

    transfer_box = boxes.get("transfer_status")
    check_crop = checkmark_crop_from_status_box(image, transfer_box) if transfer_box is not None else None

    record = {
        "source": source_path.resolve().as_posix(),
        "relative_path": relative_path.as_posix(),
        "width": facts.width,
        "height": facts.height,
        "dhash": f"{dhash(Image.fromarray(image)):016x}",
        "metadata": {
            "make": facts.make,
            "has_capture_tags": facts.has_capture_tags,
            "has_icc": facts.icc_profile is not None,
        },
        "metadata_votes": [{"label": v.label, "confidence": v.confidence, "reason": v.reason} for v in votes],
        "silver": {"outcome": decision.outcome, "label": decision.label, "reasons": decision.reasons},
        "regions": {
            "status_bar": _save_crop(status_crop, crop_dir / "status_bar" / f"{stem}.png"),
            "clock": _save_crop(clock_crop, crop_dir / "clock" / f"{stem}.png"),
            "checkmark": _save_crop(check_crop, crop_dir / "checkmark" / f"{stem}.png"),
        },
        "detector_used": proposer is not None,
        "detector_boxes": {label: [round(v, 2) for v in box] for label, box in boxes.items()},
    }
    return record


def run(
    *,
    input_path: Path,
    output_dir: Path,
    detector: Path | None = None,
    device: str = "cpu",
    proposer_min_size: int = 512,
    proposer_max_size: int = 1024,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
    skip_existing: bool = True,
) -> CacheStats:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_count 必须为正，shard_index 必须在 [0, shard_count) 内")
    root = input_path.parent if input_path.is_file() else input_path
    all_paths = list(iter_image_paths(input_path))
    selected = sorted(
        (p for p in all_paths if _shard_for(p.relative_to(root), shard_count) == shard_index),
        key=lambda p: _selection_key(p.relative_to(root)),
    )
    if limit is not None:
        selected = selected[:limit]

    proposer = None
    if detector is not None:
        proposer = _Proposer(detector, device=device, min_size=proposer_min_size, max_size=proposer_max_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "crops"
    suffix = "" if shard_count == 1 else f".shard-{shard_index:03d}-of-{shard_count:03d}"
    manifest_path = output_dir / f"region_cache{suffix}.jsonl"
    errors_path = output_dir / f"region_cache_errors{suffix}.jsonl"

    done: set[str] = set()
    if skip_existing and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["relative_path"])
                except (json.JSONDecodeError, KeyError):
                    continue

    stats = CacheStats()
    with manifest_path.open("a", encoding="utf-8") as manifest, errors_path.open("a", encoding="utf-8") as errors:
        for source_path in selected:
            relative_path = source_path.relative_to(root)
            if relative_path.as_posix() in done:
                stats.skipped += 1
                continue
            try:
                record = process_one(source_path, relative_path, crop_dir, proposer)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                stats.written += 1
            except Exception as error:  # noqa: BLE001 - 记录错误并继续处理其余图片
                stats.failed += 1
                errors.write(
                    json.dumps(
                        {"source": source_path.resolve().as_posix(), "error": type(error).__name__, "message": str(error)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors.flush()
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为平台分类器缓存状态栏 / 时钟 / 对勾裁剪及元数据")
    parser.add_argument("--input", type=Path, required=True, help="原始图片或目录（如 D:\\download\\raw_images）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, help="可选的 LRCNN 权重，作为区域粗定位候选")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--proposer-min-size", type=int, default=512)
    parser.add_argument("--proposer-max-size", type=int, default=1024)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, help="每个分片最多处理这么多张（试跑用）")
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    # 让中文输出在任意控制台编码下都不报错。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    stats = run(
        input_path=args.input,
        output_dir=args.output,
        detector=args.detector,
        device=args.device,
        proposer_min_size=args.proposer_min_size,
        proposer_max_size=args.proposer_max_size,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        limit=args.limit,
        skip_existing=not args.no_skip_existing,
    )
    print(f"区域缓存完成：写入={stats.written}，跳过={stats.skipped}，失败={stats.failed}")


if __name__ == "__main__":  # pragma: no cover
    main()
