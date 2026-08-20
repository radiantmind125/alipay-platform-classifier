r"""**确定性取块** —— 线路A 唯一还没法跨语言复现的那一步。

问题
----
线路A 的分数 = 取块 + 模型前向。ONNX 只带得走模型那一半。
取块现在走 torchvision 的 `RandomCrop`, 用的是 PyTorch 的梅森旋转随机数 ——
**别的语言不可能逐位复现**。取的块不一样, 分数就不一样, 标好的阈值也就不作数了。
(实测取块不同的影响: 阈值附近标准差 0.0305, **判定不一致 46.9%** —— 是量级差异, 不是舍入。)

做法
----
把随机数换成一个**三行就能写完、任何语言结果都一样**的发生器, 其余一律照抄原版:

    种子    state = crc32(整个文件字节);  若为 0 则取 1   (xorshift32 不能从 0 开始)
    发生器  state ^= state<<13;  state ^= state>>17;  state ^= state<<5    (全程 uint32)
    每块    y = next() % (H-32+1);   x = next() % (W-32+1)      **先 y 后 x**
    每轮    取 64 块, 算纹理能量, **取能量最小的那一块; 并列取下标最小的**
    整图    上面重复 16 轮 -> 16 块 -> 各自过模型 -> 求平均

为什么这几条必须一字不差
------------------------
- **先 y 后 x**: 原版 `RandomCrop.get_params` 就是先 `randint(0, h-th+1)` 再 `randint(0, w-tw+1)`。
  反过来两边取到的块完全不同。
- **并列取下标最小**: 原版是 `patch_list.sort(key=compute)` 再取 `[0]`, 而 Python 的 sort
  **是稳定的** —— 能量相同时保持插入顺序。收据图有大片纯色, **能量并列 0 是常态**,
  这一条不写死, 两边就会选到不同的块。
- **能量公式**: 与原版 `compute()` 逐字一致(水平 + 垂直 + 两条对角线的相邻像素绝对差之和,
  int64 累加)。

取模偏差可以忽略: 值域约 2400, 而 2^32 是它的一百多万倍。

★ 换了发生器 = 换了一次抽样 -> **每张图的分数都会变**, 必须在整池上重跑核对。
  预期不会有系统性平移(当初"加种子"那次实测中位差 0.000000、正负均衡、每万数没动),
  但**预期不能代替实测**。

自检
----
  python training/patch_select.py --selftest
"""

from __future__ import annotations

import numpy as np

PATCH_SIZE = 32
TRAINSIZE = 256
REPEAT = 16
_M32 = 0xFFFFFFFF


def xorshift32(state: int) -> int:
    """三行发生器。全程 uint32, 所以每一步都要掩回 32 位。"""
    state ^= (state << 13) & _M32
    state ^= state >> 17
    state ^= (state << 5) & _M32
    return state & _M32


def energy(patch: np.ndarray) -> int:
    """纹理能量, 与 SSP `utils/patch.py` 的 `compute()` 逐字一致。

    int64 累加 —— 32x32x3 的绝对差之和最大约 32*32*3*255*4 ≈ 1.25e7, int32 也够,
    但原版用的是 int64, 这里保持一致, 免得将来有人改块大小时溢出。
    """
    a = patch.astype(np.int64)
    return int(np.abs(a[:, :-1] - a[:, 1:]).sum()
               + np.abs(a[:-1, :] - a[1:, :]).sum()
               + np.abs(a[:-1, :-1] - a[1:, 1:]).sum()
               + np.abs(a[1:, :-1] - a[:-1, 1:]).sum())


def select_positions(h: int, w: int, seed: int,
                     patch_size: int = PATCH_SIZE, num_patch: int = 64,
                     repeat: int = REPEAT) -> list[list[tuple[int, int]]]:
    """只算坐标, 不碰像素 —— 这样 .NET 那边可以**单独核对坐标**再核对分数。

    分阶段核对很重要: 出了偏差能立刻知道是**取块错了**还是**模型喂错了**,
    而不是只看到最后一个数不对。
    """
    state = (seed & _M32) or 1
    ry, rx = h - patch_size + 1, w - patch_size + 1
    if ry < 1 or rx < 1:
        raise ValueError(f"图太小: {w}x{h} 放不下 {patch_size}x{patch_size} 的块")
    out = []
    for _ in range(repeat):
        pos = []
        for _ in range(num_patch):
            state = xorshift32(state); y = state % ry      # ★ 先 y
            state = xorshift32(state); x = state % rx      # ★ 后 x
            pos.append((y, x))
        out.append(pos)
    return out


def select_patches(rgb: np.ndarray, seed: int,
                   patch_size: int = PATCH_SIZE, num_patch: int = 64,
                   repeat: int = REPEAT) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """返回 (patches, chosen)。patches 形状 (repeat, 32, 32, 3) uint8, 直接能喂 ONNX。"""
    h, w = rgb.shape[:2]
    rounds = select_positions(h, w, seed, patch_size, num_patch, repeat)
    patches, chosen = [], []
    for pos in rounds:
        best_e = best = best_yx = None
        for (y, x) in pos:
            p = rgb[y:y + patch_size, x:x + patch_size]
            e = energy(p)
            # ★ **严格小于** —— 并列时保留先出现的那个, 等价于原版"稳定排序后取 [0]"
            if best_e is None or e < best_e:
                best_e, best, best_yx = e, p, (y, x)
        patches.append(best)
        chosen.append(best_yx)
    return np.stack(patches).astype(np.uint8), chosen


def _selftest() -> int:
    import sys
    bad = 0

    def chk(ok: bool, msg: str) -> None:
        nonlocal bad
        bad += (not ok)
        print(f"  {'OK ' if ok else '**红**'} {msg}")

    # 1. 发生器: 固定种子的前几个值(给 .NET 逐位核对用的测试向量)
    s, seq = 12345, []
    for _ in range(5):
        s = xorshift32(s); seq.append(s)
    chk(all(0 <= v <= _M32 for v in seq), f"xorshift32 值域正常, seed=12345 前 5 个: {seq}")
    chk(xorshift32(1) == 270369, f"xorshift32(1) == 270369 (实得 {xorshift32(1)})")

    # 2. 能量公式必须和 SSP 原版 compute() 完全一致
    rng = np.random.default_rng(7)
    p = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    a = p.astype(np.int64)
    ref = int(np.sum(np.abs(a[:, :-1, :] - a[:, 1:, :]))
              + np.sum(np.abs(a[:-1, :, :] - a[1:, :, :]))
              + np.sum(np.abs(a[:-1, :-1, :] - a[1:, 1:, :]))
              + np.sum(np.abs(a[1:, :-1, :] - a[:-1, 1:, :])))
    chk(energy(p) == ref, f"能量公式与原版 compute() 一致 ({energy(p)})")
    chk(energy(np.full((32, 32, 3), 200, np.uint8)) == 0, "纯色块能量为 0")

    # 3. 确定性: 同图同种子必须完全一样
    img = rng.integers(0, 256, (300, 200, 3), dtype=np.uint8)
    a1, c1 = select_patches(img, 42)
    a2, c2 = select_patches(img, 42)
    chk(np.array_equal(a1, a2) and c1 == c2, "同种子两次结果完全相同")
    a3, c3 = select_patches(img, 43)
    chk(c1 != c3, "换种子取到的块不同(否则种子没起作用)")

    # 4. 并列取下标最小 —— 收据图大片纯色, 这是常态, 必须钉死
    flat = np.full((200, 200, 3), 255, np.uint8)
    flat[100:140, 100:140] = 0                       # 只有一小块有纹理
    pos = select_positions(200, 200, 99, repeat=1)[0]
    _, ch = select_patches(flat, 99, repeat=1)
    first_zero = next((yx for yx in pos
                       if energy(flat[yx[0]:yx[0] + 32, yx[1]:yx[1] + 32]) == 0), None)
    chk(first_zero is not None and ch[0] == first_zero,
        f"能量并列时取**先出现**的那个 (选中 {ch[0]}, 第一个零能量块 {first_zero})")

    # 5. 坐标必须落在合法范围内
    h, w = img.shape[:2]
    allpos = [yx for r in select_positions(h, w, 5) for yx in r]
    chk(all(0 <= y <= h - 32 and 0 <= x <= w - 32 for y, x in allpos),
        f"{len(allpos)} 个坐标全部合法")
    chk(len(allpos) == 16 * 64, f"坐标个数 = 16 轮 x 64 块 = {len(allpos)}")

    print(f"\n  {'全部通过' if not bad else f'**{bad} 条红**'}")
    return bad


if __name__ == "__main__":
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="确定性取块(线路A 跨语言复现的那一步)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
