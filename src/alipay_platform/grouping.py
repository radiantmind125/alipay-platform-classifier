"""按“交易族/近重复”分组与切分的工具，防止大规模下的数据泄漏。

- 近重复用 dHash + LSH 分桶（避免 O(n^2)，可扩到十万级）后并查集合并。
- 交易族键：文件名里的订单号/时间戳，或（有 mate 的 OCR 时）金额+时间+收款人。
- 时间戳从文件名解析，用于“最近 N 周时间测试集”。

只用标准库；可在 numpy 环境先跑好分组/切分，再拷到 GPU 训练。
"""

from __future__ import annotations

import re
from collections import defaultdict

_TS = re.compile(r"_(\d{14})(?:\D|$)")   # s3_voucher_..._YYYYMMDDHHMMSS.ext


def parse_timestamp(filename: str) -> str | None:
    """从文件名解析 14 位时间戳（YYYYMMDDHHMMSS），失败返回 None。"""
    m = _TS.search(filename)
    return m.group(1) if m else None


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def cluster_by_dhash(items: list[tuple[str, int]], *, threshold: int = 6, bands: int = 4, max_bucket: int = 400) -> dict[str, int]:
    """把 (id, dhash_int) 按近重复聚成族，返回 id -> 族号。

    LSH：把 64 位哈希切成 ``bands`` 段，同段值相同的进同一候选桶，桶内再按汉明距离合并。
    ``max_bucket`` 跳过异常大的桶（多为巧合共享某段值），靠其它段兜底，避免退化成 O(k^2)。

    警告：当页面高度雷同（如支付宝成功页同一模板），64 位 dHash 主要反映“布局”，阈值放大
    会把几千张模板相似图并成一坨。做“近重复去重”时阈值取 0~1（只并几乎一模一样的图）；
    真正的“交易族”分组应改用 OCR 字段（金额+时间+收款人+订单号），dHash 不胜任。
    """
    n = len(items)
    uf = UnionFind(n)
    band_bits = 64 // bands
    mask = (1 << band_bits) - 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (_id, h) in enumerate(items):
        for b in range(bands):
            buckets[(b, (h >> (b * band_bits)) & mask)].append(idx)
    for idxs in buckets.values():
        if len(idxs) < 2 or len(idxs) > max_bucket:
            continue
        for a in range(len(idxs)):
            ha = items[idxs[a]][1]
            for c in range(a + 1, len(idxs)):
                if _hamming(ha, items[idxs[c]][1]) <= threshold:
                    uf.union(idxs[a], idxs[c])
    roots: dict[int, int] = {}
    out: dict[str, int] = {}
    for idx, (id_, _h) in enumerate(items):
        r = uf.find(idx)
        out[id_] = roots.setdefault(r, len(roots))
    return out


def temporal_cutoff(timestamps: list[str], holdout_frac: float) -> str | None:
    """返回时间分位切点：时间戳 >= 该值的作为最近 N% 时间测试集。"""
    valid = sorted(t for t in timestamps if t)
    if not valid:
        return None
    k = int(len(valid) * (1.0 - holdout_frac))
    k = min(max(k, 0), len(valid) - 1)
    return valid[k]
