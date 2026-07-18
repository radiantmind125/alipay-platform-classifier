"""分组/切分工具的测试。"""

from __future__ import annotations

from alipay_platform.grouping import (
    UnionFind,
    cluster_by_dhash,
    parse_timestamp,
    temporal_cutoff,
)


def test_parse_timestamp() -> None:
    assert parse_timestamp("voucher_abc123_20260701000133.jpg") == "20260701000133"
    assert parse_timestamp("no_timestamp.png") is None


def test_union_find() -> None:
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


def test_cluster_exact_duplicates_group_together() -> None:
    items = [("a", 0xFFFF0000FFFF0000), ("b", 0xFFFF0000FFFF0000), ("c", 0x0000000000000000)]
    g = cluster_by_dhash(items, threshold=6)
    assert g["a"] == g["b"]
    assert g["a"] != g["c"]


def test_cluster_near_duplicates_group() -> None:
    base = 0x0F0F0F0F0F0F0F0F
    near = base ^ 0b111  # 相差 3 位
    far = ~base & ((1 << 64) - 1)
    g = cluster_by_dhash([("a", base), ("b", near), ("c", far)], threshold=6)
    assert g["a"] == g["b"]
    assert g["a"] != g["c"]


def test_temporal_cutoff() -> None:
    ts = ["20260701000000", "20260702000000", "20260703000000", "20260704000000"]
    cut = temporal_cutoff(ts, holdout_frac=0.25)
    # 25% 最近 => 切点使最后 1 张进测试集
    assert cut == "20260704000000"
    assert temporal_cutoff([], 0.25) is None
