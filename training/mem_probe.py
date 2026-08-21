r"""单进程内存占用与线程数实测(给"多服务器多进程"的容量规划用)。

为什么要专门写这个
------------------
经理问的是"单进程运行时占多少内存", 但这里面有两件事必须**分开报**:

1. **能带到 .NET 那边去的**: 两个 ONNX 的权重 + onnxruntime 自己的开销 + 每张图的临时缓冲。
2. **带不过去的**: Python 解释器 + numpy/opencv/Pillow 的常驻开销。
   .NET 那边**没有这一块**, 所以直接把 Python 进程的 RSS 报上去会**高估**。

所以下面按阶段量, 让这两块能拆开看。

★★ 还有一件他没问但更要紧的事: **onnxruntime 默认按 CPU 核数开线程。**
本仓库里所有 `InferenceSession(...)` 都没传 `SessionOptions`, 也就是**每个进程都以为自己独占整台机器**。
多进程部署时 N 个进程 x M 个核 = 严重超订, 每张图的延迟会比单进程实测**明显变差**。
所以这个脚本同时量**每张图的耗时**, 并要求显式指定 `--threads`, 跑两遍就能看出差多少。

用法
----
  # 先按"多进程该用的配置"跑(每进程 1 线程)
  python training/mem_probe.py --a-onnx E:\SSP_Work\onnx\aigen_v7.onnx ^
      --b-onnx E:\SSP_Work\onnx\localdet9.onnx ^
      --input E:\SSP_Work\probe\genuine_20k_v7 --n 200 --threads 1

  # 再按"独占整机"的配置跑, 对比延迟
  ... --threads 0        (0 = 交给 onnxruntime 自己决定, 也就是现在的默认行为)
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
import zlib
from ctypes import wintypes
from pathlib import Path

_HERE = Path(__file__).resolve().parent


# ---------- Windows 进程内存(不依赖 psutil, 免得还要装东西) ----------
class _MEMCOUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _mem() -> tuple[float, float, float]:
    """返回 (工作集 MB, 私有字节 MB, 峰值工作集 MB)。

    **私有字节**才是"这个进程真正要独占多少" —— 多进程部署按这个数乘进程数估。
    工作集会被操作系统按内存压力回收, 单看它会低估。
    """
    c = _MEMCOUNTERS()
    c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return (c.WorkingSetSize / 1048576.0,
            c.PrivateUsage / 1048576.0,
            c.PeakWorkingSetSize / 1048576.0)


_STAGES: list[tuple[str, float, float, float]] = []


def mark(name: str) -> None:
    ws, pv, pk = _mem()
    _STAGES.append((name, ws, pv, pk))
    print(f"  {name:<28s} 工作集 {ws:8.1f} MB   私有 {pv:8.1f} MB", flush=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="单进程内存占用与线程数实测")
    ap.add_argument("--a-onnx", type=Path, required=True)
    ap.add_argument("--b-onnx", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="一张图, 或一个目录")
    ap.add_argument("--n", type=int, default=200, help="跑多少张(看稳态和有没有一直涨)")
    ap.add_argument("--threads", type=int, default=1,
                    help="onnxruntime 的 intra_op 线程数。**多进程部署应当用 1**; "
                         "传 0 = 不设置, 也就是现在代码里的默认行为(按核数开)")
    # ★ 这两个必须和 patch_select.py 的 PATCH_SIZE / TRAINSIZE 一致(32 / 256)。
    #   写成 128 的话每轮候选块从 64 掉到 16, **不会报错**, 只会让耗时看着比真实值好四倍。
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--trainsize", type=int, default=256)
    args = ap.parse_args()

    print(f"\n=== 阶段内存 (threads={args.threads}) ===", flush=True)
    mark("0 起步(只有标准库)")

    import numpy as np                                          # noqa: E402
    import onnxruntime as ort                                   # noqa: E402
    from PIL import Image                                       # noqa: E402
    sys.path.insert(0, str(_HERE))
    from patch_select import select_patches                     # noqa: E402
    from predict_tiled import _richest_patch, _tiles            # noqa: E402
    from locate_blue import locate_amount_auto                  # noqa: E402
    mark("1 导入完(numpy/cv2/PIL/ort)")

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    sa = ort.InferenceSession(str(args.a_onnx), so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(str(args.b_onnx), so, providers=["CPUExecutionProvider"])
    mark("2 两个 ONNX 都加载完")

    a_mb = args.a_onnx.stat().st_size / 1048576.0
    b_mb = args.b_onnx.stat().st_size / 1048576.0

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if args.input.is_dir():
        imgs = [p for p in sorted(args.input.rglob("*")) if p.suffix.lower() in exts][: args.n]
    else:
        imgs = [args.input] * args.n
    if not imgs:
        raise SystemExit(f"!! {args.input} 下没找到图")

    npatch = (args.trainsize // args.patch_size) ** 2

    def score_one(p: Path) -> None:
        raw = p.read_bytes()
        seed = zlib.crc32(raw) & 0x7FFFFFFF or 1
        rgb = np.asarray(Image.open(p).convert("RGB"))
        h, w = rgb.shape[:2]
        # 线路A
        pa, _ = select_patches(rgb, seed, patch_size=args.patch_size,
                               num_patch=npatch, repeat=16)
        sa.run(["ai_score"], {"patches": np.stack([np.asarray(x, np.uint8) for x in pa])})
        # 线路B
        loc, _page = locate_amount_auto(rgb)
        arr, cols, rows = rgb, 3, 6
        if loc:
            bx0, by0, bx1, by1 = loc[0], loc[1], loc[2], loc[3]
            pad = 8
            arr = rgb[max(0, by0 - pad):min(h, by1 + pad), max(0, bx0 - pad):min(w, bx1 + pad)]
            cols, rows = 2, 2
        else:
            arr = rgb[: max(32, int(h * 0.6))]
        tiles = _tiles(arr.shape[1], arr.shape[0], cols, rows, 0.15)
        tp = [np.asarray(_richest_patch(arr[y0:y1, x0:x1], args.patch_size), np.uint8)
              for (x0, y0, x1, y1) in tiles]
        sb.run(["ai_score"], {"patches": np.stack(tp)})

    t0 = time.perf_counter()
    score_one(imgs[0])
    first = time.perf_counter() - t0
    mark("3 跑完第 1 张")

    times = []
    for p in imgs[1:]:
        t = time.perf_counter()
        try:
            score_one(p)
        except Exception:
            continue                      # 坏图不算进耗时, 内存照样看得到
        times.append(time.perf_counter() - t)
    mark(f"4 跑完 {len(times) + 1} 张(稳态)")

    ws, pv, pk = _mem()
    times.sort()
    med = times[len(times) // 2] if times else first

    print(f"\n=== 结论 (threads={args.threads}) ===")
    print(f"两个 ONNX 文件合计         {a_mb + b_mb:8.1f} MB (磁盘上)")
    imp = _STAGES[1][2] - _STAGES[0][2]
    ses = _STAGES[2][2] - _STAGES[1][2]
    run = pv - _STAGES[2][2]
    print(f"Python + numpy/cv2/PIL/ort {imp:8.1f} MB  <- ★ .NET 那边**没有这一块**")
    print(f"两个 ONNX 会话             {ses:8.1f} MB  <- 这一块 .NET 也要付")
    print(f"跑图带来的增长             {run:8.1f} MB  <- 缓冲 + onnxruntime 内存池")
    print(f"单进程稳态(私有字节)       {pv:8.1f} MB")
    print(f"峰值工作集                 {pk:8.1f} MB")
    print(f"\n第 1 张 {first:.3f} 秒(含预热), 之后中位 {med:.3f} 秒/张, 共 {len(times) + 1} 张")
    print(f"折算单进程吞吐             {3600.0 / med:8.0f} 张/小时"
          f"  ({86400.0 / med / 10000.0:.1f} 万/天)")

    growth = pv - _STAGES[3][2]
    print(f"\n第 1 张之后又涨了 {growth:.1f} MB "
          f"({'正常, 是内存池预热' if growth < 60 else '★ 涨得偏多, 值得再跑 1000 张看是不是一直涨'})")
    print("\n★ 多进程估算: 每台机器能开几个进程 ≈ (可用内存 - 系统占用) / 上面那个"
          "\"单进程稳态\"; 总吞吐 ≈ 进程数 x 上面那个\"单进程吞吐\"。")
    print("★ 但前提是 **每个进程都设成 threads=1**。不设的话每个进程都按核数开线程, "
          "N 个进程互相抢核, 每张图的耗时会比这里量到的差很多。")


if __name__ == "__main__":
    main()
