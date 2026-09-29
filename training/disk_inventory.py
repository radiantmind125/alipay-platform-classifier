r"""盘上有什么: 每个目录多少文件、多大、最近什么时候动过, 是不是 git 仓库 / Python 虚拟环境 / 有模型。只读。

为什么要这个
------------
经理要把我们的东西从 D 盘挪到 E 盘。D 盘上混着三类东西:
    我们自己的      alipay-ai-data、probe、localcrops、ssp_*、download 底下的实验输出 ...
    经理下载的      download\TempFakeImages、download2\BlueImages / OtherImages ...
    大家共用的      Hx.AI.py(线上仓库)、推理用的虚拟环境 —— 挪了可能把正在跑的服务弄坏
挪之前得先看清楚每个目录是什么、多大、E 盘放不放得下。

★ 虚拟环境(有 pyvenv.cfg)挪了会坏: Scripts 里的 exe 启动器写死了绝对路径。要在新盘上重建, 不能直接拷。
★ 硬链接: 目录里的文件要是硬链接(比如 pinyin_hits 当初是硬链接归集的), 拷到别的盘会变成实打实的一份,
  占的空间比在 D 盘上看着的多。抽前 200 个文件看链接数。
★ 输出只用 ASCII, 服务器控制台是 GBK, 中文贴回来是乱码。

用法
----
    python training\disk_inventory.py
    python training\disk_inventory.py --root D:\ --expand D:\download D:\download2 D:\alipay-ai-data
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

SKIP = {"$recycle.bin", "system volume information", "recovery", "config.msi"}
MODEL_EXT = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".pdparams"}


def walk_stats(root: Path, nlink_probe: int = 200) -> dict:
    """递归统计一个目录。碰到没权限的子目录跳过并计数, 不中断。"""
    n = size = 0
    newest = oldest = None
    exts = Counter()
    flags = set()
    denied = 0
    probed = linked = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            denied += 1
            continue
        with it:
            for e in it:
                try:
                    # ★ 目录联接(junction)在 Python 3.12 里 is_dir(follow_symlinks=False) 也是 True,
                    #   跟进去会把别处的东西算进来, 指向上层的还会绕圈。不跟, 记个数
                    if getattr(e, "is_junction", lambda: False)():
                        flags.add("junction")
                        continue
                    if e.is_dir(follow_symlinks=False):
                        low = e.name.lower()
                        if low == ".git":
                            flags.add("git")
                            continue                # 不往 .git 里数
                        stack.append(Path(e.path))
                        continue
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    denied += 1
                    continue
                n += 1
                size += st.st_size
                mt = st.st_mtime
                newest = mt if newest is None or mt > newest else newest
                oldest = mt if oldest is None or mt < oldest else oldest
                ext = os.path.splitext(e.name)[1].lower()
                exts[ext or "(none)"] += 1
                if e.name.lower() == "pyvenv.cfg":
                    flags.add("VENV")
                if ext in MODEL_EXT:
                    flags.add("model")
                if probed < nlink_probe:
                    try:
                        if os.stat(e.path).st_nlink > 1:   # DirEntry.stat 在 Windows 上不给链接数
                            linked += 1
                    except OSError:
                        pass
                    probed += 1
    if linked:
        flags.add(f"hardlink {linked}/{probed}")
    return {"n": n, "size": size, "newest": newest, "oldest": oldest,
            "exts": exts, "flags": flags, "denied": denied}


def fmt_t(t) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d") if t else "-"


def line(name: str, s: dict) -> str:
    top = ",".join(f"{k}:{v}" for k, v in s["exts"].most_common(3))
    fl = " ".join(sorted(s["flags"]))
    den = f" denied:{s['denied']}" if s["denied"] else ""
    return (f"  {name:<44}{s['n']:>10,}{s['size'] / 2**30:>9.1f}  {fmt_t(s['oldest'])} {fmt_t(s['newest'])}"
            f"  {fl:<22} {top}{den}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("D:/"))
    ap.add_argument("--expand", type=Path, nargs="*",
                    default=[Path("D:/download"), Path("D:/download2"), Path("D:/alipay-ai-data")],
                    help="这些目录再往下拆一层, 各子目录分开列")
    ap.add_argument("--drives", nargs="*", default=["D:/", "E:/"], help="看剩余空间的盘")
    a = ap.parse_args()

    print("=" * 110)
    print("  DISK INVENTORY  (ASCII only - safe to paste)   read-only")
    print("=" * 110)
    for dr in a.drives:
        try:
            u = shutil.disk_usage(dr)
            print(f"  drive {dr:<4} total {u.total / 2**30:8.0f} GB   used {u.used / 2**30:8.0f} GB   "
                  f"free {u.free / 2**30:8.0f} GB")
        except OSError:
            print(f"  drive {dr:<4} not found")
    print()
    head = (f"  {'folder':<44}{'files':>10}{'GB':>9}  {'oldest':<10} {'newest':<10}  {'flags':<22} top extensions")
    expand = {os.path.normcase(os.path.abspath(p)) for p in a.expand}

    def show(parent: Path, title: str) -> None:
        print(f"  (scanning {parent} ... can take a few minutes on big folders)", flush=True)
        print(f"  ---- {title} ----")
        print(head)
        try:
            entries = sorted(os.scandir(parent), key=lambda e: e.name.lower())
        except OSError as ex:
            print(f"  cannot list {parent}: {ex}")
            return
        loose_n = loose_size = 0
        rows = []
        for e in entries:
            if e.name.lower() in SKIP:
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    rows.append((e.name, walk_stats(Path(e.path))))
                else:
                    loose_n += 1
                    loose_size += e.stat().st_size
            except OSError:
                continue
        for name, s in sorted(rows, key=lambda r: -r[1]["size"]):
            mark = "  (expanded below)" if os.path.normcase(os.path.abspath(os.path.join(parent, name))) in expand else ""
            print(line(name, s) + mark)
        if loose_n:
            print(f"  {'(loose files at this level)':<44}{loose_n:>10,}{loose_size / 2**30:>9.1f}")
        print()

    show(a.root, f"top level of {a.root}")
    for p in a.expand:
        if p.exists():
            show(p, f"inside {p}")
        else:
            print(f"  ---- {p} not found ----\n")
    print("  flags: VENV = Python virtual environment (recreate on the new drive, do not copy)")
    print("         git = repository, model = has .onnx/.pt/..., hardlink = sampled files with more than one link")
    print("=" * 110)


if __name__ == "__main__":
    sys.exit(main())
