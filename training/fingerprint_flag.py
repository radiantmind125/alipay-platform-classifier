r"""用**文件编码指纹**从"真图"池里挖假图 —— 不看画面, 不用模型, 秒级跑完。

为什么要有这个(它补的是 cross_flag.py 的短板)
---------------------------------------------
`cross_flag.py` 用"两条线都打高分"筛假图, **精确率高但召回只有约 50%**:
线路B 必须先**定位到金额区**才出信号, 而诈骗收款页、聊天转账截图这类根本没有标准金额区,
线路B 压根评不了 -> 这类假图两条线规则永远抓不到。

这个脚本走**完全不同的一条路**: 只看"**这个文件是被什么东西写出来的**"。
手机系统截图由固定的编码器写出, PNG 块序列 / EXIF 标签 / 分辨率 都有稳定指纹;
AI 生成或用工具去水印另存的图, 指纹必然不同。**跟画面内容无关, 所以模型看不见的它也能抓。**

实测依据(2026-08-07, 在 124,151 张图池上跑出来的)
-----------------------------------------------
- 全池 333 张 EXIF 带 ImageWidth 的 PNG, 干净地分成两群, **零重叠**:
  * 310 张写**真实尺寸**, 其中 309 张带设备 Software 串(`Android Flyme 12.6.0.0A` 这种);
  * **23 张写 ImageWidth=0 / ImageHeight=0 / Orientation=0(非法值, 合法是 1..8), 且一张都不带 Software 串。**
- 这 23 张用了 16 个分辨率, **14 个在 12.4 万张里独此一家**; **0/23 落在标准机型分辨率上**
  (对照: 正常的 App 重存家族有 44.2% 落在标准分辨率)。
  唯二重合的 1373x3051 / 1408x3104, 重合对象**全部是已查实的假图**。
- 全池 47,944 张带 EXIF DateTime 的 PNG 里, **eXIf 长度 150 字节的只有 4 张**,
  **逐字节 diff: 150 字节里 136 字节完全相同**, 变的 14 字节全是时间数字。
  = 某个工具往每张图盖同一个模板, 只换时间。

用法
----
  python training/fingerprint_flag.py --input D:\probe\genuine_20k --out D:\probe\fp_flagged.txt
  # 也可以直接扫训练集的 nature 类:
  python training/fingerprint_flag.py --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\train\nature \
      --out D:\probe\trainnat_fp_bad.txt

四条规则
--------
  R1 零填充EXIF   ImageWidth==0 且没有 Software 串   -> 精确率很高, 默认计入
  R2 模板EXIF     eXIf 长度 == 150 字节              -> 精确率极高, 默认计入
  R3 孤儿分辨率   分辨率全池只出现 1 次且非标准      -> --with-r3 才计入
  R4 非标准分辨率 分辨率不在标准机型尺寸集合里       -> --with-r4 才计入, **召回最高**

**R1/R2 和 R4 抓的是不同的团伙, 必须一起用**:
- R1/R2 只命中"零填充EXIF"那个工具家族(全池 23 张), **抓不到豆包那批** —— 豆包那批根本不写 EXIF;
- R4 反过来: 实测**已查实假图用到的 14 种分辨率里 13 种是非标准的**, 而 R4 在全池只命中
  2,095 张(1.69%) -> 这是目前召回最高的一条无模型规则。

输出的文件名列表可以直接喂给 autoreject_threshold.py 的 --exclude。
**注意: 这是候选不是判决。** R1/R2 可以直接用; R3/R4 建议和 cross_flag.py 的结果取交集, 或人工抽查后再用。
"""

from __future__ import annotations

import argparse
import collections
import struct
import sys
from pathlib import Path

# 标准机型分辨率的判定: 在本次扫描的池子里出现次数 >= 这个数, 就算"某个真实机型的屏幕尺寸"
_STD_MIN_COUNT = 50

# EXIF 标签号
_TAG_IMAGE_WIDTH = 0x0100
_TAG_IMAGE_LENGTH = 0x0101
_TAG_SOFTWARE = 0x0131
_TAG_EXIF_IFD = 0x8769


def _parse_exif(blob: bytes) -> dict:
    """解 TIFF/EXIF, 返回 {标签号: 值}。只取我们要用的几个。"""
    out: dict = {}
    try:
        if blob[:2] == b"MM":
            end = ">"
        elif blob[:2] == b"II":
            end = "<"
        else:
            return out
        stack = [struct.unpack(end + "I", blob[4:8])[0]]
        seen = set()
        while stack:
            off = stack.pop()
            if off in seen or off + 2 > len(blob):
                continue
            seen.add(off)
            n = struct.unpack(end + "H", blob[off:off + 2])[0]
            if n > 200:                     # 明显坏掉的 IFD, 别越读越乱
                continue
            for i in range(n):
                e = off + 2 + i * 12
                if e + 12 > len(blob):
                    break
                tag, typ, cnt = struct.unpack(end + "HHI", blob[e:e + 8])
                raw = blob[e + 8:e + 12]
                if tag == _TAG_EXIF_IFD:
                    stack.append(struct.unpack(end + "I", raw)[0])
                elif tag in (_TAG_IMAGE_WIDTH, _TAG_IMAGE_LENGTH):
                    out[tag] = (struct.unpack(end + "I", raw)[0] if typ == 4
                                else struct.unpack(end + "H", raw[:2])[0])
                elif tag == _TAG_SOFTWARE:
                    p = struct.unpack(end + "I", raw)[0]
                    s = raw[:cnt] if cnt <= 4 else blob[p:p + cnt]
                    out[tag] = s.decode("latin1", "replace").strip("\x00").strip()
    except Exception:
        pass
    return out


def fingerprint(path: Path) -> dict | None:
    """读文件头部(不解码像素), 返回这张图的编码指纹。"""
    try:
        with open(path, "rb") as f:
            d = f.read(300000)          # eXIf / 各种块头都在最前面
    except Exception:
        return None

    if d[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", d[16:24])
        seq: list[str] = []
        exif = b""
        i = 8
        while i < len(d) - 8:
            ln = struct.unpack(">I", d[i:i + 4])[0]
            typ = d[i + 4:i + 8].decode("latin1", "replace")
            if typ == "IDAT":
                seq.append("IDAT*")
                break
            seq.append(typ)
            if typ == "eXIf":
                exif = d[i + 8:i + 8 + ln]
            if typ == "IEND":
                break
            i += 12 + ln
        return dict(fmt="png", w=w, h=h, chunks=",".join(seq),
                    exif_len=len(exif), exif=_parse_exif(exif) if exif else {})

    if d[:2] == b"\xff\xd8":
        exif = b""
        w = h = 0
        i = 2
        while i < len(d) - 4:
            if d[i] != 0xFF:
                break
            m = d[i + 1]
            if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            ln = struct.unpack(">H", d[i + 2:i + 4])[0]
            if m == 0xE1 and d[i + 4:i + 10] == b"Exif\x00\x00":
                exif = d[i + 10:i + 2 + ln]
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", d[i + 5:i + 9])
            if m == 0xDA:
                break
            i += 2 + ln
        return dict(fmt="jpg", w=w, h=h, chunks="", exif_len=len(exif),
                    exif=_parse_exif(exif) if exif else {})
    return None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="用文件编码指纹批量筛可疑假图(不看画面, 不用模型)")
    ap.add_argument("--input", type=Path, required=True, help="要扫的图片目录")
    ap.add_argument("--out", type=Path, default=None, help="把命中的文件名写到这里(可直接喂 --exclude)")
    ap.add_argument("--template-exif-len", type=int, default=150,
                    help="R2: 把这个字节长度的 eXIf 当'工具模板'。实测本池是 150")
    ap.add_argument("--with-r3", action="store_true",
                    help="把 R3(孤儿分辨率)也算进命中。默认只报数不命中, 因为它单独用精确率不够")
    ap.add_argument("--with-r4", action="store_true",
                    help="把 R4(非标准分辨率)也算进命中。召回最高但精确率中等, 建议和模型分数取交集")
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    files = [p for p in args.input.rglob("*")
             if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
    if not files:
        raise SystemExit(f"{args.input} 下没找到图片")
    print(f"扫描 {len(files):,} 张 ...")

    fps: dict[Path, dict] = {}
    for p in files:
        fp = fingerprint(p)
        if fp:
            fps[p] = fp
    print(f"读出指纹 {len(fps):,} 张")

    res_count = collections.Counter((fp["w"], fp["h"]) for fp in fps.values() if fp["w"])
    std = {wh for wh, c in res_count.items() if c >= _STD_MIN_COUNT}
    print(f"分辨率 {len(res_count)} 种, 其中出现 >= {_STD_MIN_COUNT} 次的'标准机型尺寸' {len(std)} 种")

    r1, r2, r3, r4 = [], [], [], []
    for p, fp in fps.items():
        ex = fp["exif"]
        # R1: EXIF 里有尺寸标签但填 0, 且没有设备 Software 串 -> 不是任何一条真机截图流水线写的
        if _TAG_IMAGE_WIDTH in ex and ex.get(_TAG_IMAGE_WIDTH) == 0 and not ex.get(_TAG_SOFTWARE):
            r1.append(p)
        # R2: eXIf 长度正好等于那个工具模板的长度
        if fp["fmt"] == "png" and fp["exif_len"] == args.template_exif_len:
            r2.append(p)
        # R3: 分辨率在本池里独一无二, 且不是标准机型尺寸
        if fp["w"] and res_count[(fp["w"], fp["h"])] == 1 and (fp["w"], fp["h"]) not in std:
            r3.append(p)
        # R4: 分辨率不在标准机型尺寸集合里(R3 的放宽版, 召回高得多)
        if fp["w"] and (fp["w"], fp["h"]) not in std:
            r4.append(p)

    print()
    print("=" * 78)
    print(f"R1 零填充EXIF(ImageWidth=0 且无 Software) : {len(r1):6d} 张   ← 精确率很高, 直接用")
    print(f"R2 模板EXIF(eXIf 长度 == {args.template_exif_len:3d} 字节)      : {len(r2):6d} 张   ← 精确率极高, 直接用")
    print(f"R3 孤儿分辨率(只出现1次且非标准)          : {len(r3):6d} 张   "
          f"← {'已计入' if args.with_r3 else '仅报数(单独用精确率不够)'}")
    print(f"R4 非标准分辨率(R3 放宽版, 召回最高)      : {len(r4):6d} 张   "
          f"← {'已计入' if args.with_r4 else '仅报数(建议和模型分数取交集)'}")
    print("=" * 78)
    print("实测参考(2026-08-07, 12.4万张池): R4 只命中全池 1.69%(2,095张), 而**已查实假图用到的")
    print("14 种分辨率里有 13 种是非标准的** -> R4 是目前召回最高的一条无模型规则。")
    print("R1/R2 只抓'零填充EXIF'那个工具家族(23张), **抓不到豆包那批**(它们根本不写EXIF),")
    print("所以 R1/R2 和 R4 是互补的, 别只用一个。")

    s1, s2, s3, s4 = set(r1), set(r2), set(r3), set(r4)
    hit = s1 | s2 | (s3 if args.with_r3 else set()) | (s4 if args.with_r4 else set())
    print(f"\n合计命中 {len(hit)} 张 ({len(hit)/max(1,len(fps))*100:.3f}%)")

    # 命中的图落在标准分辨率上的比例 —— 这是**规则可信度的自检**。
    # 实测: 假图组 0% 落标准尺寸, 正常图 44%+。若这里明显偏高, 说明规则在这个池子上不成立。
    if hit:
        on_std = sum(1 for p in hit if (fps[p]["w"], fps[p]["h"]) in std)
        print(f"自检: 命中的图里只有 {on_std}/{len(hit)} ({on_std/len(hit)*100:.1f}%) 落在标准机型尺寸上。")
        print("      (实测参考: 真图约 44%+ 落在标准尺寸, 假图 0%。这个数越低越说明规则抓对了。)")

    print(f"\n命中明细(前 {min(args.show, len(hit))} 张):")
    print("  规则      尺寸        eXIf  文件名")
    for p in sorted(hit, key=lambda x: (fps[x]["w"], fps[x]["h"]))[:args.show]:
        fp = fps[p]
        tags = "".join(t for t, s in (("1", p in s1), ("2", p in s2), ("3", p in s3), ("4", p in s4)) if s)
        print(f"  R{tags:<8s} {fp['w']}x{fp['h']:<7d} {fp['exif_len']:5d}  {p.name}")

    if args.out:
        args.out.write_text("\n".join(sorted(p.name for p in hit)) + "\n", encoding="utf-8")
        print(f"\n已写出 {len(hit)} 个文件名 -> {args.out}")
        print("用法: 抽查几张确认后, 直接当 autoreject_threshold.py 的 --exclude 用。")

    print("\n提醒: 这是**候选**不是判决。R1/R2 实测精确率很高可以直接用;")
    print("      R3 建议和 cross_flag.py 的结果取交集, 或者人工抽查后再用。")


if __name__ == "__main__":
    main()
