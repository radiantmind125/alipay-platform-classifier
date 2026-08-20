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
    每轮    取 64 块, 算纹理能量, **取能量最大的那一块; 并列取下标最大的**
    整图    上面重复 16 轮 -> 16 块 -> 各自过模型 -> 求平均

★★ 方向: 取**最大**, 不是最小 —— 这一条是吃了大亏才查清的
--------------------------------------------------------
第一版按"取最小"写, 结果 597 张图的分数**中位涨了 +0.1615, 97.7% 的图都涨**。
查下去发现: **服务器上跑的 `utils/patch.py` 和另一份副本不一样**。

    另一份副本:  new_img = patch_list[0]     # 升序排完取第一个 = 能量**最小**
    服务器实跑:  new_img = patch_list[-1]    # 升序排完取最后一个 = 能量**最大**

行为实验坐实了这一点: 同一个种子下, `patch_img` 返回的块在它自己 64 个候选里
**排第 64/64 位**, 三张图无一例外 —— 即便候选里有 6~34 个能量为 0 的纯色块, 它一个都不选。

**所以 SSP 这一步取的是"整张图里最有纹理的那一块", 不是"最平的那一块"。**

★ 由此带来的运维风险(必须写进 DEPLOY_SPEC)
--------------------------------------------
打分行为依赖于**第三方仓库里一个被改过的文件**, 而那个仓库**不在我们的 git 管辖内**。
一旦有人重新 clone 或 reset SSP 仓库, `[-1]` 变回 `[0]`, **所有分数和阈值当场作废**,
而且**不会报错** —— 只会安静地给出另一套数。

为什么这几条必须一字不差
------------------------
- **先 y 后 x**: 原版 `RandomCrop.get_params` 就是先 `randint(0, h-th+1)` 再 `randint(0, w-tw+1)`。
  反过来两边取到的块完全不同。
- **并列取下标最大**: 原版是升序稳定排序后取 `[-1]` —— 并列的最大值里保留**最后出现**的那个。
  所以判断要用 `>=` 一路覆盖, 用 `>` 会停在第一个, 两边就会选到不同的块。
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
            # ★★ 取**能量最大**的那一块, 不是最小 —— 见文件头"方向"一节。
            #   `>=` 而不是 `>`: 原版是**升序稳定排序后取 `[-1]`**, 并列的最大值里
            #   保留的是**最后出现**的那个; 用 `>=` 一路覆盖才等价。
            if best_e is None or e >= best_e:
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

    # 4. ★★ 方向: 必须取能量**最大**的块。第一版写成取最小, 导致分数系统性偏高 0.16
    flat = np.full((200, 200, 3), 255, np.uint8)
    flat[100:140, 100:140] = 0                       # 只有一小块有纹理
    pos = select_positions(200, 200, 99, repeat=1)[0]
    _, ch = select_patches(flat, 99, repeat=1)
    es = [energy(flat[y:y + 32, x:x + 32]) for y, x in pos]
    chk(energy(flat[ch[0][0]:ch[0][0] + 32, ch[0][1]:ch[0][1] + 32]) == max(es),
        f"取的是候选里能量**最大**的 (选中能量 "
        f"{energy(flat[ch[0][0]:ch[0][0]+32, ch[0][1]:ch[0][1]+32]):,}, 候选最大 {max(es):,})")
    # 并列取**最后出现**的那个(等价于升序稳定排序后取 [-1])
    last_max = [yx for yx, e in zip(pos, es) if e == max(es)][-1]
    chk(ch[0] == last_max, f"并列时取**最后出现**的 (选中 {ch[0]}, 最后一个最大值 {last_max})")

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
