r"""**单张打分, 返回 JSON** —— 给 .NET 那边对接用的参照实现(只读源图)。

两个用途, 都很实在
------------------
1. 若最终决定"打分留在 Python", 这个就是**产物本身**;
2. 若最终决定"用 .NET 调 ONNX", 这个就是**对照物** —— 他那边算出来的每一项都能和这里逐个对。

所以这里**不重写任何打分逻辑**: 取块用 `patch_select`, 定位和切块直接 import `predict_tiled`
和 `locate_blue` 里那几个函数, 模型走 ONNX。和 `ssp_decide` 是同一个原则 ——
**一旦重写, 分数就可能有细微出入, 而出入不会报错, 只会让验收对不上。**

为什么用 ONNX 而不是 torch
--------------------------
.NET 那边跑的是 ONNX。参照实现也走 ONNX, 两边才是**同一条路**;
用 torch 的话即便结果一样, 也少了一层可比性。

输出
----
每张图一行 JSON(多张就是 NDJSON), 字段见下面 `--help` 或直接跑一次看。
**`ai_score` 不是概率** —— 它的含义是分位, 详见 `rarity` 字段。

用法
----
  python training/ssp_score_one.py --a-onnx E:\SSP_Work\onnx\aigen_v7.onnx ^
      --b-onnx E:\SSP_Work\onnx\localdet9.onnx --input <一张图或一个目录>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 分数 -> "多少张真图里才出现一张"。★ **这不是概率**: 0.80 不代表 80% 是假图,
# 它的含义是**分位** —— 9.8 万张真图里只有约 20 张能到 0.80。
# 依据: 标定池上按误杀预算反推的分数线(见 DEPLOY_SPEC)。
_RARITY = [
    (0.9317, "很罕见: 约每 1 万张真图才出现 1 张"),
    (0.8031, "罕见: 约每 5000 张真图出现 1 张"),
    (0.6497, "偏高: 约每 1000 张真图出现 1 张"),
    (0.0, "常见: 比每 1000 张 1 张更常见, 属正常范围"),
]

# 能从元数据里**明确认出厂商**的几个。认不出就返回 null, 不猜。
_VENDOR = [(b"gpt-image", "OpenAI gpt-image"), (b"doubao", "豆包"), (b"jimeng", "即梦"),
           (b"aweme", "抖音")]


def _rarity(score: float) -> str:
    for thr, label in _RARITY:
        if score >= thr:
            return label
    return _RARITY[-1][1]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="单张打分返回 JSON(参照实现, 只读源图)")
    ap.add_argument("--a-onnx", type=Path, required=True, help="线路A 的 ONNX")
    ap.add_argument("--b-onnx", type=Path, required=True, help="线路B 的 ONNX")
    ap.add_argument("--input", type=Path, required=True, help="一张图, 或一个目录")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--meta-cap", type=int, default=400_000, help="线路C 只读文件头这么多字节")
    ap.add_argument("--pretty", action="store_true", help="多行缩进(默认每张一行, 方便管道)")
    ap.add_argument("--no-summary", action="store_true",
                    help="不要末尾那份汇总。默认**会**打一份到 stderr —— "
                         "stdout 保持干净的 NDJSON, 重定向到文件时汇总照样看得见")
    args = ap.parse_args()

    import importlib.util
    miss = [m for m in ("onnxruntime", "cv2") if not importlib.util.find_spec(m)]
    bad = [f"缺 Python 包 {miss}" for _ in ([1] if miss else [])]
    for p, n in ((args.a_onnx, "--a-onnx"), (args.b_onnx, "--b-onnx")):
        if not p.is_file():
            bad.append(f"{n} 不存在: {p}")
    if not args.input.exists():
        bad.append(f"--input 不存在: {args.input}")
    if bad:
        raise SystemExit("跑不了, 先解决这些:\n  " + "\n  ".join(bad))

    import onnxruntime as ort                                     # noqa: E402
    from PIL import Image                                         # noqa: E402
    from ssp_decide import decide, load_config                     # noqa: E402
    from patch_select import select_patches                        # noqa: E402
    # ★ 直接用线路B 那套现成函数, 一行都不重写
    from predict_tiled import _is_blue_page, _richest_patch, _tiles  # noqa: E402
    from locate_blue import locate_amount_auto                      # noqa: E402
    import ast as _ast
    _ws = _ast.parse((_HERE / "watermark_scan.py").read_text(encoding="utf-8"))
    _keep = [n for n in _ws.body if isinstance(n, _ast.Assign)
             and getattr(n.targets[0], "id", "") in ("_AIGC_HARD", "_AIGC_SOFT")]
    _ns: dict = {"re": __import__("re")}
    exec(compile(_ast.Module(body=_keep, type_ignores=[]), "<ws>", "exec"), _ns)
    HARD, SOFT = _ns["_AIGC_HARD"], _ns["_AIGC_SOFT"]

    cfg = load_config(args.config or (_HERE / "ssp_config.json"))
    la, lb = cfg["line_a"], cfg["line_b"]
    sa = ort.InferenceSession(str(args.a_onnx), providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(str(args.b_onnx), providers=["CPUExecutionProvider"])

    imgs = ([args.input] if args.input.is_file()
            else [p for p in sorted(args.input.rglob("*")) if p.suffix.lower() in _EXTS])
    if not imgs:
        raise SystemExit(f"{args.input} 底下没找到图")

    # ★ 汇总自己算, 不指望调用方去解析输出文件。
    #   起因: 用 PowerShell 的 `>` 存下来再解析, 结果 5.1 的 `>` 默认写 **UTF-16 LE**,
    #   Python 按 UTF-8 读第一个字节 0xff 就炸。绕开这类坑最省事的办法是**不产生中间文件**。
    from collections import Counter
    stat = {"n": 0, "located": 0, "err": 0, "dec": Counter(), "gen": Counter()}

    for p in imgs:
        out: dict = {"image": p.name}
        try:
            raw = p.read_bytes()
            out["sha256"] = hashlib.sha256(raw).hexdigest()
            rgb = np.asarray(Image.open(p).convert("RGB"))
            h, w = rgb.shape[:2]
            out["width"], out["height"] = int(w), int(h)

            # ---- 线路A ----
            seed = zlib.crc32(raw) & 0x7FFFFFFF
            patches, _ = select_patches(rgb, seed)
            a_scores = sa.run(["ai_score"], {"patches": patches})[0]
            a = float(a_scores.mean())
            out["line_a"] = {"score": round(a, 6), "seed": int(seed), "patches": int(len(patches))}

            # ---- 线路B ----
            loc, page = locate_amount_auto(rgb)
            b_val, located = None, bool(loc)
            arr = rgb
            if loc:
                x0, y0, x1, y1 = loc[0], loc[1], loc[2], loc[3]
                arr = rgb[y0:y1, x0:x1]                      # amount-pad 0 -> 不外扩
            elif 0 < 0.6 < 1.0:
                arr = rgb[:max(32, int(h * 0.6))]            # roi-top 0.6
            th, tw = arr.shape[:2]
            tiles = [_richest_patch(arr[ty0:ty1, tx0:tx1], 32)
                     for (tx0, ty0, tx1, ty1) in _tiles(tw, th, 3, 6, 0.15)]
            if tiles:
                t = np.stack([np.asarray(x, np.uint8) for x in tiles])
                ts = sb.run(["ai_score"], {"patches": t})[0]
                b_val = float(np.sort(ts)[-3:].mean())       # agg top3
            out["line_b"] = {"score": None if b_val is None else round(b_val, 6),
                             "located": located, "page": page, "tiles": len(tiles)}

            # ---- 线路C ----
            head = raw[:args.meta_cap]
            hard = sorted(k for k, pat in HARD.items() if pat.search(head))
            soft = sorted(k for k, pat in SOFT.items() if pat.search(head))
            vendor = next((v for tok, v in _VENDOR if tok.lower() in head.lower()), None)
            out["line_c"] = {"hard": hard, "soft": soft}
            # ★ 认不出厂商就是 null, **不猜**。带元数据的图约占两千分之一,
            #   剩下的模型只判"是不是 AI", 判不了"是哪家"。
            out["generator"] = vendor if hard else None

            # ---- 判定 ----
            d, why = decide((a, True), (b_val, located) if b_val is not None else None,
                            cfg, p.suffix.lower(), set(hard) or None)
            out["decision"], out["reason"] = d, why
            # ★ 给"这个分有多不寻常"一个人能读的口径。**不要当成概率用。**
            out["rarity"] = _rarity(a)
            out["thresholds"] = {"a_strict": la["strict"], "a_review": la["review"],
                                 "b_strict": lb["strict"], "b_review": lb["review"]}
        except Exception as exc:                              # noqa: BLE001
            # ★ 打不开/打分失败 -> **人工复核**, 绝不放行。这是踩过的坑: 失败被当成 0 分放行了。
            out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            out["decision"], out["reason"] = "人工复核", "打分失败, 无法判定"
            stat["err"] += 1
        stat["n"] += 1
        stat["dec"][out["decision"]] += 1
        stat["located"] += int(bool(out.get("line_b", {}).get("located")))
        if out.get("generator"):
            stat["gen"][out["generator"]] += 1
        print(json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None), flush=True)

    if not args.no_summary and stat["n"]:
        n = stat["n"]
        w = sys.stderr                                  # ★ 走 stderr, stdout 留给 NDJSON
        print(f"\n{'=' * 46}", file=w)
        print(f"共 {n} 张 | 金额定位成功 {stat['located']} = **{stat['located'] / n * 100:.1f}%**"
              f"  (标定池是 98.0%, 差很多就要查定位器)", file=w)
        for k in ("自动拒", "人工复核", "放行"):
            c = stat["dec"].get(k, 0)
            print(f"  {k:6s}{c:>7,d}{c / n * 100:>8.2f}%{c / n * 1e4:>9.1f}/万", file=w)
        if stat["err"]:
            print(f"  !!!! 打分失败 {stat['err']} 张 —— 已按'无法判定'进人工复核, 没有放行", file=w)
        if stat["gen"]:
            print(f"  元数据认出厂商: {dict(stat['gen'])}", file=w)
        print("★ 这份汇总走 stderr, 所以 stdout 重定向成文件时它照样显示在屏幕上。", file=w)


if __name__ == "__main__":
    main()
