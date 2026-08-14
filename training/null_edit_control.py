r"""**空编辑对照组**: 走完整条造样本管线, 但**一个像素都不改**(只读源图)。

为什么这是最要紧的一个对照
--------------------------
我们所有假图格子, 都共用**同一条管线**:

    定位金额 -> 裁下来 -> JPEG q95 编码 -> 交给模型重绘 -> 缩回原尺寸 -> 羽化贴回 -> 存 jpg q95

线路B 在这些格子上召回很高(万相 93.5%, gpt-image-2 89.3%), 而真图误杀很低。
**但有两种解释都能吻合这个现象**:

1. **它在认"这块被模型生成过"** —— 那么召回数字就是我们说的意思, 而且跨厂商泛化成立;
2. **它在认"这块被我们的管线动过"**(羽化缝 + 两次 JPEG 重编码) ——
   那么任何生成器都会得到差不多的高分(**实测确实如此**), 真图从没被合成过所以分数低(**也吻合**),
   而"跨厂商泛化"就是个**假象**: 我们量的是自己的管线, 不是模型的能力。

**第二种如果成立, DEPLOY_SPEC 里每一个召回数字都要重写** ——
真实骗子的合成方式和我们不一样, 数字就不迁移。

这个脚本把管线里**除了生成之外的每一步都原样做一遍**:
裁下来、JPEG q95 编码、解码、缩回原尺寸、羽化贴回、存 jpg q95。
**贴回去的就是原来那块像素。**

判读
----
- **分数接近 0 / 召回接近 0** -> 信号来自**生成**, 现有所有数字站得住
- **分数很高 / 召回很高**     -> 我们一直在量**自己的管线**, 召回数字必须重新表述

★ 这一步不花 API 钱, 纯本地。**它决定了要发出去的那些数字是不是真的那个意思。**

用法
----
  python training/null_edit_control.py --src-root D:\download2\OtherImages ^
      --out D:\probe\nulledit --n 150 --exclude-src D:\probe --seed 61

**只读源图**: 只往 --out 写。
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_api_local_edit import _collect, _feather_paste          # noqa: E402
from locate_blue import locate_amount_auto                        # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="空编辑对照组: 走完整管线但不改像素(只读源图)")
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--send-pad", type=float, default=0.5,
                    help="和 gen_api_local_edit 的同名参数保持一致, 否则裁块范围就不是同一个了")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--exclude-src", type=Path, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="nulledit")
    args = ap.parse_args()

    from gen_api_local_edit import _used_srcs
    excl = _used_srcs(list(args.exclude_src)) if args.exclude_src else set()
    srcs = _collect(args.src_root, args.n * 3, args.seed, excl)
    if not srcs:
        raise SystemExit("没采到真图源")
    print(f"真图源 {len(srcs)} 张 | 目标造 {args.n} 张空编辑对照", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    mf = open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig")
    mw = csv.writer(mf); mw.writerow(["file", "model", "src"])

    made = skipped = 0
    page_n = {"blue": 0, "white": 0}
    for sp in srcs:
        if made >= args.n:
            break
        try:
            src_im = ImageOps.exif_transpose(Image.open(sp)).convert("RGB")
            src = np.asarray(src_im)
            loc_s, page_s = locate_amount_auto(src)
            if not loc_s:
                skipped += 1
                continue
            sx0, sy0, sx1, sy1 = loc_s[0], loc_s[1], loc_s[2], loc_s[3]
            H, W = src.shape[:2]

            # ---- 以下每一步都和 gen_api_local_edit 的 --send crop 分支逐行对齐 ----
            pad = int(max(8, (sy1 - sy0) * args.send_pad))
            cx0, cy0 = max(0, sx0 - pad), max(0, sy0 - pad)
            cx1, cy1 = min(W, sx1 + pad), min(H, sy1 + pad)

            buf = io.BytesIO()                                   # 发出去那一步的 JPEG 编码
            Image.fromarray(src[cy0:cy1, cx0:cx1]).save(buf, "JPEG", quality=95)

            # **唯一的区别在这里**: 不交给模型, 直接把刚编码的那块解回来
            gen_im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

            gen = np.asarray(gen_im.resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS))
            ox, oy = sx0 - cx0, sy0 - cy0
            piece = gen[oy:oy + (sy1 - sy0), ox:ox + (sx1 - sx0)]
            out = _feather_paste(src, piece, (sx0, sy0, sx1, sy1))

            fn = f"nulledit_{args.tag}_{made:05d}.jpg"
            Image.fromarray(out).save(args.out / fn, quality=args.quality)
            mw.writerow([fn, "NULL-EDIT(未经任何模型)", sp.name]); mf.flush()
            made += 1
            page_n[page_s] = page_n.get(page_s, 0) + 1
            if made % 25 == 0:
                print(f"  已造 {made}/{args.n} 张...", flush=True)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  跳过 {sp.name}: {type(exc).__name__}: {str(exc)[:110]}", flush=True)
    mf.close()

    print(f"\n完成: {made} 张 -> {args.out}(跳过 {skipped} 张)")
    print(f"版式: 蓝图 {page_n.get('blue', 0)} 张, 白图 {page_n.get('white', 0)} 张")
    print()
    print("下一步: 用线路B 打一遍分, 和真正的假图格子放同一张表里比")
    print("  预期 **接近 0** —— 若不接近 0, 说明线路B 认的是我们的管线痕迹(羽化缝 + 两次 JPEG),")
    print("  而不是'这块被生成过'。那样的话 DEPLOY_SPEC 里每一个召回数字都要重新表述。")


if __name__ == "__main__":
    main()
