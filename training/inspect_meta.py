r"""把图里**带溯源信息的那些**的元数据挖出来看, 找我们还不认识的生成器(只读源图)。

为什么要挖
----------
验收池(88,645 张真实进件)里线路C 报了:

- **铁证 2 张** —— 打开一看, C2PA 清单里明写 `gpt-image` `version 2.0`
  `digitalSourceType=.../trainedAlgorithmicMedia` `c2pa.watermarked`, 生成于 2026-08-09。
  **真实骗子已经在用 gpt-image-2 造支付宝收据了。**
- **软标记 97 张**, 其中 **C2PA凭证 85 张** —— 有 C2PA 容器**但没有生成声明**。

★ **手机截图本来不该带 C2PA。** 那 85 张里到底装的什么, 我们从没看过。
  如果里面有生成器名字而我们的正则不认识, 那就是**白白丢掉的铁证**。
  这个脚本就是去看那 85 张(以及任何带溯源信息的图)里到底写了什么。

判读
----
- 挖出**新的生成器名字** -> 加进 `watermark_scan._AIGC_HARD`, 线路C 覆盖面直接变大
- 全是相机/修图 App 的签名 -> 确认那 85 张是正常的, 维持"软标记只统计不判定"

用法
----
  python training/inspect_meta.py --input D:\probe\gen100k_imgs --out D:\probe\meta_dump

  # 只想快速看看, 先扫一部分
  python training/inspect_meta.py --input D:\probe\gen100k_imgs --limit 20000 --out D:\probe\meta_dump

**只读源图**: 只往 --out 写。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 只要带上这些痕迹之一, 就值得把里面的字符串挖出来看
_INTERESTING = re.compile(
    rb"c2pa|jumd|caBX|digitalSourceType|claim_generator|softwareAgent|xmp|XMP|"
    rb"Adobe|Photoshop|GIMP|Midjourney|Stable ?Diffusion|DALL|OpenAI|gpt-image|"
    rb"firefly|imagen|gemini|flux|comfy|automatic1111|novelai|"
    rb"AIGC|tc260|doubao|jimeng|hunyuan|wanx|qwen|kolors|seedream|liblib|meitu|retouch",
    re.I)

# 从这些键后面把值捞出来
_KEYVAL = [
    ("claim_generator", re.compile(rb"claim_generator(?:_info)?[\"':\s]{0,4}([\x20-\x7e]{3,80})")),
    ("softwareAgent",   re.compile(rb"softwareAgent[\"':\s]{0,4}([\x20-\x7e]{3,80})")),
    ("digitalSource",   re.compile(rb"digitalSourceType[\"':\s]{0,6}([\x20-\x7e]{3,90})")),
    ("Software",        re.compile(rb"Software[\"':\s\x00]{0,4}([\x20-\x7e]{3,60})")),
    ("CreatorTool",     re.compile(rb"CreatorTool[\"':\s>]{0,4}([\x20-\x7e]{3,60})")),
    ("Model",           re.compile(rb"\bModel[\"':\s\x00]{0,4}([\x20-\x7e]{3,40})")),
    ("Make",            re.compile(rb"\bMake[\"':\s\x00]{0,4}([\x20-\x7e]{3,40})")),
]

_GEN_HINT = re.compile(rb"gpt-image|dall|openai|midjourney|stable ?diffusion|firefly|imagen|"
                       rb"gemini|flux|comfy|novelai|doubao|jimeng|hunyuan|wanx|qwen|kolors|"
                       rb"seedream|trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia", re.I)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="挖带溯源信息的图的元数据, 找没见过的生成器(只读源图)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=65536,
                    help="每张只读文件头这么多字节。C2PA/XMP 都在很靠前的位置"
                         "(实测那两张铁证在偏移 2968), 读 64KB 足够, 比读 400KB 快好几倍")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = [p for p in sorted(args.input.rglob("*")) if p.suffix.lower() in _EXTS and p.is_file()]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"{args.input} 底下没找到图")
    print(f"{len(files):,} 张, 每张只读前 {args.cap // 1024} KB", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    hit_rows = []
    vals: dict[str, Counter] = {k: Counter() for k, _ in _KEYVAL}
    tokens: Counter = Counter()
    n_interesting = n_genhint = 0
    t = time.time()
    for i, p in enumerate(files, 1):
        try:
            with open(p, "rb") as fh:
                d = fh.read(args.cap)
        except OSError:
            continue
        if not _INTERESTING.search(d):
            continue
        n_interesting += 1
        row = {"image_name": p.name}
        for key, pat in _KEYVAL:
            m = pat.search(d)
            if m:
                v = m.group(1).decode("utf-8", "replace").strip().strip('"\'<>,')
                v = re.sub(r"[^\x20-\x7e]", "", v)[:70]
                if v:
                    vals[key][v] += 1
                    row[key] = v
        # ★ 发现新生成器主要靠这个, 而不是靠上面那套 key=value:
        #   C2PA 是 CBOR 编码的, 字符串前面有长度前缀字节(所以原文看到的是 "eigpt-image" 而不是
        #   "gpt-image")。想精确解析反而容易漏, 不如**把标记附近所有可读片段都捞出来**统计,
        #   没见过的生成器名字自然会浮出来。
        for m in _INTERESTING.finditer(d):
            w = d[max(0, m.start() - 160): m.end() + 240]
            for tok in re.findall(rb"[\x20-\x7e]{5,60}", w):
                s = tok.decode("ascii", "replace").strip()
                for piece in re.split(r"[\s,;/\\\"'<>{}\[\]()]+", s):
                    piece = piece.strip(".:-_=")
                    if len(piece) >= 5 and not piece.isdigit():
                        tokens[piece] += 1
        g = _GEN_HINT.search(d)
        if g:
            n_genhint += 1
            row["生成器线索"] = g.group(0).decode("utf-8", "replace")
        hit_rows.append(row)
        if i % 20000 == 0:
            print(f"  已扫 {i:,}/{len(files):,}  命中 {n_interesting:,}  (用时 {time.time()-t:.0f}s)",
                  flush=True)

    sp = args.out / "meta_hits.csv"
    cols = ["image_name"] + [k for k, _ in _KEYVAL] + ["生成器线索"]
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(hit_rows)

    print(f"\n{'='*60}")
    print(f"扫了 {len(files):,} 张 | 带溯源痕迹的 **{n_interesting:,}** 张 | "
          f"其中带**生成器线索**的 **{n_genhint:,}** 张  (用时 {time.time()-t:.0f}s)")
    for key, _ in _KEYVAL:
        c = vals[key]
        if not c:
            continue
        print(f"\n[{key}] 共 {sum(c.values()):,} 处, 不同取值 {len(c)} 种:")
        for v, n in c.most_common(12):
            mark = "  <-- **生成器**" if _GEN_HINT.search(v.encode()) else ""
            print(f"   {n:>6,d}  {v}{mark}")
    # 已认识的关键词单独排除, 剩下的按出现次数排 —— **没见过的生成器往往就藏在这里**
    known = re.compile(r"c2pa|jumd|caBX|xmp|adobe|iptc|newscodes|digitalsourcetype|http|urn:|"
                       r"uuid|sha256|jpeg|image/|action|when|version|generator|softwareAgent|"
                       r"assertion|claim|ingredient|manifest|instance|thumbnail|created|hash", re.I)
    gen = [(t, n) for t, n in tokens.items() if _GEN_HINT.search(t.encode("ascii", "replace"))]
    other = [(t, n) for t, n in tokens.items()
             if not known.search(t) and not _GEN_HINT.search(t.encode("ascii", "replace"))]
    if gen:
        print(f"\n★★ **生成器名字**(共 {len(gen)} 种):")
        for t, n in sorted(gen, key=lambda x: -x[1])[:20]:
            print(f"   {n:>6,d}  {t}")
    if other:
        print(f"\n其余可读片段(排掉已知关键词后, 前 25 种) —— **没见过的生成器可能就在这里**:")
        for t, n in sorted(other, key=lambda x: -x[1])[:25]:
            print(f"   {n:>6,d}  {t}")

    print(f"\n-> {sp}")
    print("\n判读: 出现我们没见过的生成器名字 -> 加进 watermark_scan._AIGC_HARD, 线路C 覆盖面直接变大;")
    print("      全是相机/修图 App -> 确认软标记那批是正常的, 维持'只统计不判定'。")


if __name__ == "__main__":
    main()
