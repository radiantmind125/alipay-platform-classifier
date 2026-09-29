r"""已经处理过的图的文件名 —— 从目录、扫描名单(.jsonl)或名单文本(.txt)里读。

为什么要这个
------------
经理往 D:\download2\BlueImages、OtherImages 里传图, 会把上一批删掉再传新的。
上一批里挑出来的拼音图我们拷走了, 但**没挑中的原图跟着删了**, 只剩扫描名单
(pick_pinyin 的 --out, 每张扫过的图一行)。新传的这批要是又带着上一批的图,
光拿老图库(E 盘)比是认不出来的 —— 得把上一批的扫描名单也算进"见过的"。

★ 按文件名认, 不按路径: 同一张图挪过目录、换过盘, 文件名不变(凭证号 + 时间戳)。

支持三种来源, 可以混着给:
    目录     里面(含子目录)所有图片的文件名
    .jsonl   每行一个 JSON, 取 path 字段的文件名(pick_pinyin 的扫描名单)
    .txt     每行一个文件名或路径
"""
from __future__ import annotations

import json
import os
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _base(p: str) -> str:
    # 名单里存的是 Windows 路径, 在别的系统上 os.path.basename 认不出反斜杠
    return os.path.basename(p.replace("\\", "/"))


def load(paths) -> tuple[set[str], list[str], dict[str, str]]:
    """返回 (文件名集合, 每个来源读到多少的说明, 文件名 -> 现在还在的文件路径)。

    第三样只有目录来源才有(名单里的原图可能已经删了), 拿来抽样逐字节比。
    来源不存在就报出来, 不静默跳过 —— 少算了见过的, 就会把老图当新图再扫一遍。
    """
    names: set[str] = set()
    notes: list[str] = []
    where: dict[str, str] = {}
    for p in paths or []:
        p = Path(p)
        before = len(names)
        if p.is_dir():
            n = 0
            for dirpath, _d, files in os.walk(p):
                for f in files:
                    if os.path.splitext(f)[1].lower() in EXTS:
                        names.add(f)
                        where.setdefault(f, os.path.join(dirpath, f))
                        n += 1
            notes.append(f"{p}  目录, 图片 {n:,} 张")
        elif p.is_file() and p.suffix.lower() == ".jsonl":
            n = bad = 0
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:           # noqa: BLE001
                        bad += 1                # 半行(上次被砍断的)
                        continue
                    if r.get("path"):
                        names.add(_base(r["path"]))
                        n += 1
            notes.append(f"{p}  扫描名单, {n:,} 条" + (f"(坏行 {bad})" if bad else ""))
        elif p.is_file():
            n = 0
            with p.open(encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        names.add(_base(line))
                        n += 1
            notes.append(f"{p}  名单, {n:,} 行")
        else:
            raise FileNotFoundError(f"见过的来源不在: {p}")
        notes[-1] += f", 新增文件名 {len(names) - before:,} 个"
    return names, notes, where
