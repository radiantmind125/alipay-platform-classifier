r"""用**文件编码指纹**从"真图"池里挖可疑图 —— 不看画面, 不用模型, 秒级跑完。

为什么要有这个(它补的是 cross_flag.py 的短板)
---------------------------------------------
`cross_flag.py` 用"两条线都打高分"筛假图, **精确率高但召回只有约 50%**:
线路B 必须先**定位到金额区**才出信号, 而诈骗收款页、聊天转账截图这类根本没有标准金额区,
线路B 压根评不了 -> 这类假图两条线规则永远抓不到。

这个脚本走**完全不同的一条路**: 只看"**这个文件是被什么东西写出来的**"。
手机系统截图由固定的编码器写出, PNG 块序列 / EXIF 标签 / JPEG 量化表 / 分辨率 都有稳定指纹;
AI 生成或用修图 App 另存的图, 指纹必然不同。**跟画面内容无关, 所以模型看不见的它也能抓。**

★★ 先读这一段: 输出**不能**直接当 --exclude 用 ★★
------------------------------------------------
2026-08-07 实测踩过的坑: 在 20,000 张真图池上跑 `--with-r4`, 命中 **1600 张(8%)**,
而明细里全是 `147x320 / 540x1200 / 573x1280 / 576x1280` 这种**小图** ——
那是**被微信/QQ 转发压缩过的真截图**(长边被压到 1280), **不是假图**。
把它们当假图剔掉, 会**人为压低测出来的误杀率**, 于是标定出一条**过于激进的自动拒阈值**,
上线后真的去误杀用户。**这是会赔钱的错误方向, 比漏检严重。**

所以本脚本的定位是: **产出"待核查候选"**, 必须用 `--summary` 拿模型分数验证过、
或者人工抽查过, 才能进 --exclude。R1/R2 精确率高可以直接用; R3/R4/R5 不行。

四类规则
--------
  R1 零填充EXIF   ImageWidth==0 且没有 Software 串        -> 精确率很高, 默认计入
  R2 模板EXIF     eXIf 长度 == 150 字节                   -> 精确率极高, 默认计入
  R3 孤儿分辨率   分辨率在本次扫描里只出现 1 次且非标准   -> --with-r3
  R4 非标准分辨率 分辨率不在标准机型尺寸集合里            -> --with-r4 (召回最高, 误报也最多)
  R5 修图App痕迹  JPEG 的量化表/色度采样不是手机截图那一套 -> --with-r5

实测依据(2026-08-07)
--------------------
- 全池 333 张 EXIF 带 ImageWidth 的 PNG, 干净地分成两群, **零重叠**:
  310 张写**真实尺寸**且 309 张带设备 Software 串(`Android Flyme 12.6.0.0A` 这种);
  **23 张写 ImageWidth=0 / Orientation=0(非法值), 且一张都不带 Software 串。**
- 全池 47,944 张带 EXIF DateTime 的 PNG 里 **eXIf 长度 150 字节的只有 4 张**,
  逐字节 diff **150 字节里 136 字节完全相同**, 变的 14 字节全是时间数字 = 工具盖的模板。
- `.jpg` 55,481 张: **96.8% 共用同一张量化表**, 56.8% 带设备串 = 手机截图流水线;
  `.jpeg` 611 张: **24 种量化表**(top1 才 24%), 只有 4.75% 带 Software 串, 且那些串是
  **`Xingtu iOS 14.6.0`(醒图) / `美图秀秀-iOS-12.12.1`** 这种**修图 App**。
  线路B 上 `.jpeg` 中位数 **0.2700** 而 `.png`/`.jpg` 都是 **0.0000** -> R5 由此而来。
  **注意: "过了修图App" != "是假图"** —— 用户合法裁个图也会这样。R5 是风险信号不是判决。

用法
----
  # 只看统计, 不写文件(建议先这么跑一次)
  python training/fingerprint_flag.py --input D:\probe\genuine_20k

  # 用模型分数验证规则到底准不准(**关键一步**)
  python training/fingerprint_flag.py --input D:\probe\genuine_20k \
      --summary D:\probe\genuine_20k_ld3\summary.csv --col tile_top3 --with-r4 --with-r5

  # 验证过之后再写出候选名单
  python training/fingerprint_flag.py --input D:\probe\genuine_20k --out D:\probe\fp_flagged.txt
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import struct
import sys
from pathlib import Path

# "标准机型分辨率"的判定用**占比**而不是绝对张数 —— 否则同一条规则在 2 万张和 12 万张的
# 目录上松紧完全不同(踩过: 绝对值 50 在 2 万张上等于 0.25%, 在 12 万张上只有 0.04%)。
_STD_MIN_SHARE = 0.0015          # 占扫描总量 0.15% 以上, 才算某个真实机型的屏幕尺寸
_STD_MIN_ABS = 20                # 同时要求的绝对下限, 防止小目录上人人都是"标准"
_MIN_REF = 2000                  # 学"什么算正常"至少要这么多张; 不够就明确说结果不可信
_MIN_REF_JPG = 500               # R5 学量化表基线至少要这么多张 JPEG

# EXIF 标签号
_TAG_IMAGE_WIDTH = 0x0100
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
                elif tag == _TAG_IMAGE_WIDTH:
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
        ex = _parse_exif(exif) if exif else {}
        return dict(fmt="png", w=w, h=h, chunks=",".join(seq), exif_len=len(exif),
                    exif=ex, qtab="", chroma="", soft=ex.get(_TAG_SOFTWARE, ""))

    if d[:2] == b"\xff\xd8":
        exif = b""
        w = h = 0
        qt: list[bytes] = []
        chroma = ""
        i = 2
        while i < len(d) - 4:
            if d[i] != 0xFF:
                break
            m = d[i + 1]
            if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            ln = struct.unpack(">H", d[i + 2:i + 4])[0]
            if m == 0xDB:                                   # DQT 量化表 = 编码器指纹
                qt.append(d[i + 4:i + 2 + ln])
            elif m == 0xE1 and d[i + 4:i + 10] == b"Exif\x00\x00":
                exif = d[i + 10:i + 2 + ln]
            elif m in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", d[i + 5:i + 9])
                nc = d[i + 9]
                chroma = ",".join(f"{d[i + 10 + 3 * c + 1]:02x}" for c in range(min(nc, 3)))
            elif m == 0xDA:
                break
            i += 2 + ln
        ex = _parse_exif(exif) if exif else {}
        return dict(fmt="jpg", w=w, h=h, chunks="", exif_len=len(exif), exif=ex,
                    qtab=hashlib.sha1(b"".join(qt)).hexdigest()[:10] if qt else "",
                    chroma=chroma, soft=ex.get(_TAG_SOFTWARE, ""))
    return None


def fingerprint_reference(d: Path):
    """扫一个大目录当"什么算正常"的参照集(只读文件头, 很快)。"""
    out = {}
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            fp = fingerprint(p)
            if fp:
                out[p] = fp
    return (out,)


def _validate(tag: str, hit: set, fps: dict, scores: dict, col: str) -> None:
    """★用模型分数验证规则: 命中的图分数**确实更高**, 这条规则才站得住。"""
    a = [scores[p.name] for p in hit if p.name in scores]
    b = [scores[p.name] for p in fps if p not in hit and p.name in scores]
    if len(a) < 5 or len(b) < 5:
        print(f"  {tag:24s} 样本太少, 没法验证 (命中 {len(a)} / 未命中 {len(b)})")
        return
    a.sort()
    b.sort()
    med_a, med_b = a[len(a) // 2], b[len(b) // 2]
    hi_a = sum(1 for x in a if x >= 0.9) / len(a)
    hi_b = sum(1 for x in b if x >= 0.9) / len(b)
    lift = (hi_a / hi_b) if hi_b > 0 else float("inf")
    verdict = ("规则成立" if hi_a > 3 * hi_b and hi_b >= 0
               else "★可疑: 命中组并不比未命中组分高, 别拿它去 --exclude")
    print(f"  {tag:24s} 命中{len(a):6d}张 中位数{med_a:.4f} >=0.9占比{hi_a*100:6.2f}%  |  "
          f"其余{len(b):6d}张 中位数{med_b:.4f} >=0.9占比{hi_b*100:6.2f}%  |  "
          f"提升{lift:6.1f}x  {verdict}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="用文件编码指纹批量筛可疑图(不看画面, 不用模型)")
    ap.add_argument("--input", type=Path, required=True, help="要扫的图片目录")
    ap.add_argument("--out", type=Path, default=None, help="把命中的文件名写到这里")
    ap.add_argument("--template-exif-len", type=int, default=150,
                    help="R2: 把这个字节长度的 eXIf 当'工具模板'。实测本池是 150")
    ap.add_argument("--min-height", type=int, default=2000,
                    help="R3/R4 的尺寸下限。低于它的基本是**被转发压缩过的真截图**(长边1280那批), "
                         "不是假图 —— 已查实假图的高度最小 2246")
    ap.add_argument("--with-r3", action="store_true", help="把 R3(孤儿分辨率)计入命中")
    ap.add_argument("--with-r4", action="store_true", help="把 R4(非标准分辨率)计入命中")
    ap.add_argument("--with-r5", action="store_true", help="把 R5(修图App痕迹)计入命中")
    ap.add_argument("--ref-dir", type=Path, default=None,
                    help="从这个大目录学'什么算正常的分辨率/量化表'。不给就用 --input 自己, "
                         "但 --input 太小时基线学不出来(会有告警)")
    ap.add_argument("--summary", type=Path, default=None,
                    help="★模型打分的 summary.csv, 用来验证每条规则到底准不准")
    ap.add_argument("--col", default="tile_top3", help="--summary 里用哪一列分数")
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
    n = len(fps)
    print(f"读出指纹 {n:,} 张")

    # ★R3/R4/R5 的"什么算正常"是**从被扫的这批图里学出来的**, 所以这批图太少就学不出东西。
    #  (踩过: 在 4 张图的目录上跑, "标准分辨率"学出 0 种, 于是 4 张全部命中 R4, 毫无意义。)
    ref = fingerprint_reference(args.ref_dir) if args.ref_dir else None
    src = ref[0] if ref else fps
    n_ref = len(src)
    if args.ref_dir:
        print(f"参照集: {args.ref_dir} 的 {n_ref:,} 张")
    if n_ref < _MIN_REF:
        print(f"⚠ 参照样本只有 {n_ref:,} 张(< {_MIN_REF:,}), **R3/R4/R5 学不出可靠的'正常'基线**, "
              f"结果只能当参考。想认真跑就加 --ref-dir 指一个大目录。")

    res_count = collections.Counter((fp["w"], fp["h"]) for fp in src.values() if fp["w"])
    need = max(_STD_MIN_ABS, int(n_ref * _STD_MIN_SHARE))
    std = {wh for wh, c in res_count.items() if c >= need}
    print(f"分辨率 {len(res_count)} 种; '标准机型尺寸'门槛 = 出现 >= {need} 次"
          f"(占比 {_STD_MIN_SHARE*100:.2f}% 或 {_STD_MIN_ABS} 张取大) -> {len(std)} 种")

    # 手机截图流水线的量化表: 占 JPEG 多数的那几张。剩下的就是别的东西写的。
    jq = collections.Counter(fp["qtab"] for fp in src.values() if fp["fmt"] == "jpg" and fp["qtab"])
    native_q, acc = set(), 0
    for q, c in jq.most_common():
        native_q.add(q)
        acc += c
        if acc >= 0.90 * sum(jq.values()):
            break
    r5_ok = sum(jq.values()) >= _MIN_REF_JPG
    if jq:
        print(f"JPEG 量化表 {len(jq)} 种; 覆盖 90% 的'手机原生'表有 {len(native_q)} 种"
              + ("" if r5_ok else f"  ⚠ JPEG 只有 {sum(jq.values())} 张(< {_MIN_REF_JPG}), R5 不可信, 已禁用"))

    r1, r2, r3, r4, r5 = [], [], [], [], []
    for p, fp in fps.items():
        ex, wh = fp["exif"], (fp["w"], fp["h"])
        big = fp["h"] >= args.min_height
        if _TAG_IMAGE_WIDTH in ex and ex.get(_TAG_IMAGE_WIDTH) == 0 and not ex.get(_TAG_SOFTWARE):
            r1.append(p)                                    # R1 零填充 EXIF
        if fp["fmt"] == "png" and fp["exif_len"] == args.template_exif_len:
            r2.append(p)                                    # R2 模板 EXIF
        if fp["w"] and big and wh not in std and res_count[wh] == 1:
            r3.append(p)                                    # R3 孤儿分辨率
        if fp["w"] and big and wh not in std:
            r4.append(p)                                    # R4 非标准分辨率
        if r5_ok and fp["fmt"] == "jpg" and fp["qtab"] and not fp["soft"].startswith("Android"):
            if fp["qtab"] not in native_q or fp["chroma"] == "11,11,11":
                r5.append(p)                                # R5 修图 App 痕迹

    s1, s2, s3, s4, s5 = map(set, (r1, r2, r3, r4, r5))
    print()
    print("=" * 84)
    print(f"R1 零填充EXIF(ImageWidth=0 且无 Software)  : {len(r1):6d} 张  ← 精确率很高, 默认计入")
    print(f"R2 模板EXIF(eXIf == {args.template_exif_len:3d} 字节)          : {len(r2):6d} 张  ← 精确率极高, 默认计入")
    print(f"R3 孤儿分辨率(高>={args.min_height} 且只出现1次)   : {len(r3):6d} 张  "
          f"← {'已计入' if args.with_r3 else '仅报数'}")
    print(f"R4 非标准分辨率(高>={args.min_height})            : {len(r4):6d} 张  "
          f"← {'已计入' if args.with_r4 else '仅报数'}")
    print(f"R5 修图App痕迹(量化表/色度非手机原生)      : {len(r5):6d} 张  "
          f"← {'已计入' if args.with_r5 else '仅报数'}")
    print("=" * 84)

    hit = (s1 | s2 | (s3 if args.with_r3 else set())
           | (s4 if args.with_r4 else set()) | (s5 if args.with_r5 else set()))
    print(f"合计命中 {len(hit)} 张 ({len(hit)/max(1,n)*100:.3f}%)")

    if args.summary and args.summary.exists():
        scores: dict[str, float] = {}
        for row in csv.DictReader(open(args.summary, encoding="utf-8-sig")):
            nm = (row.get("image_name") or "").strip() or Path(row.get("image") or "").name
            v = (row.get(args.col) or "").strip()
            if nm and v:
                try:
                    scores[nm] = float(v)
                except ValueError:
                    pass
        print(f"\n★用模型分数验证各条规则 (列 {args.col}, 读到 {len(scores):,} 个分数)")
        print("  规则                     命中组                                 未命中组")
        for tag, s in (("R1 零填充EXIF", s1), ("R2 模板EXIF", s2), ("R3 孤儿分辨率", s3),
                       ("R4 非标准分辨率", s4), ("R5 修图App痕迹", s5)):
            if s:
                _validate(tag, s, fps, scores, args.col)
        print("  解读: '提升' 是命中组高分率 / 未命中组高分率。**低于 3 倍的规则别拿去 --exclude**,")
        print("        那说明它抓的多半是正常图(比如被转发压缩过的真截图), 剔掉会把误杀率算低。")
    elif args.summary:
        print(f"\n(--summary 指向的文件不存在: {args.summary})")

    print(f"\n命中明细(前 {min(args.show, len(hit))} 张):")
    print("  规则       尺寸        eXIf  量化表      Software")
    for p in sorted(hit, key=lambda x: (fps[x]["w"], fps[x]["h"]))[:args.show]:
        fp = fps[p]
        tags = "".join(t for t, s in (("1", p in s1), ("2", p in s2), ("3", p in s3),
                                      ("4", p in s4), ("5", p in s5)) if s)
        print(f"  R{tags:<9s} {fp['w']}x{fp['h']:<7d} {fp['exif_len']:5d}  "
              f"{fp['qtab'] or '-':10s}  {fp['soft'][:28]:28s} {p.name}")

    if args.out:
        args.out.write_text("\n".join(sorted(p.name for p in hit)) + "\n", encoding="utf-8")
        print(f"\n已写出 {len(hit)} 个文件名 -> {args.out}")

    print()
    print("★ 再强调一次: 这是**候选**不是判决。")
    print("  R1/R2 实测精确率高, 可以直接进 --exclude。")
    print("  R3/R4/R5 **必须先用 --summary 验证过或人工抽查过**再用 —— ")
    print("  否则你会把'被转发压缩过的真截图'当假图剔掉, 于是误杀率被算低, 阈值定得过激进, 上线赔钱。")


if __name__ == "__main__":
    main()
