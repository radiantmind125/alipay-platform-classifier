"""几何 + 元数据缓存路径的端到端测试（无检测器、无 torch）。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from alipay_platform.cache_regions import run


def _write_iphone_png(path: Path) -> None:
    # iPhone 13 渲染尺寸 => 分辨率标注函数投“苹果”。
    arr = np.zeros((2532, 1170, 3), dtype=np.uint8)
    arr[:120, :, :] = 200  # 顶部一条浅色“状态栏”带
    Image.fromarray(arr).save(path)


def test_geometry_only_cache_produces_manifest_and_crops() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_iphone_png(raw / "txn_001.png")
        out = tmp_path / "cache"

        stats = run(input_path=raw, output_dir=out, detector=None)
        assert stats.written == 1 and stats.failed == 0

        manifest = out / "region_cache.jsonl"
        assert manifest.exists()
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(records) == 1
        record = records[0]

        # 元数据把它标成苹果（只有分辨率标注函数触发 -> 不足以采纳银标，这是对的：
        # 单个标注函数不能独自制造标签）。
        assert record["width"] == 1170 and record["height"] == 2532
        assert record["silver"]["outcome"] == "review:insufficient"
        assert any(v["label"] == "ios" for v in record["metadata_votes"])

        # 即便没有检测器，状态栏裁剪也能由几何位置写出。
        status_bar = record["regions"]["status_bar"]
        assert status_bar is not None and Path(status_bar).exists()
        # 没有检测器 => 暂时没有对勾裁剪。
        assert record["regions"]["checkmark"] is None
        assert record["detector_used"] is False


def test_cache_is_resumable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_iphone_png(raw / "txn_001.png")
        out = tmp_path / "cache"

        first = run(input_path=raw, output_dir=out, detector=None)
        assert first.written == 1
        # 重复运行会跳过已缓存的图，而不是重复写入。
        second = run(input_path=raw, output_dir=out, detector=None)
        assert second.written == 0 and second.skipped == 1
