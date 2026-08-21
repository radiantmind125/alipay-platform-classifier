r"""线路C 到底贡献了多少检出率 —— **开和关各报一个数**。

为什么必须分开报
----------------
万相 / 千问 / gpt-image 都是**真实服务**, 生成时会把 AIGC 标记写进文件。
线路C 读那个标记, **一抓一个准**。所以把线路C 算进去, 检出率会漂亮得离谱 ——
**但真实作案的图截个图、过一遍平台管线, 标记就没了。**

所以有意义的那个数是**关掉线路C** 的; 开着线路C 的那个只是**天花板**。
(这一点在反向对照那次已经验证过: 把线路C 整条拿掉, 那张 gpt-image 假图
 照样被线路A 和线路B 各自独立判成自动拒。)

怎么算的
--------
**只打一遍分**(调 `ssp_score_one.py`, 就是交付出去的那份参照实现),
拿它每张图的 JSON, 然后用**同一个 `decide()`** 算两遍判定:
一遍带 `line_c.hard`, 一遍把它清空。

★ 特意**不重写打分逻辑** —— 线路B 的 pad 和网格已经因为"另写一份"错过两次了。

样本怎么选
----------
`--sets` 默认只挑**留出生成器**(`heldout_*`, 训练时没见过)和**空编辑对照**。
**训练用过的目录一律不进默认清单**(`*_train*` / `lew2_*` / `ai_fakes_v4`) ——
拿训过的生成器报召回是**乐观的**, 这条规矩 `AIGEN_RETRAIN_RUNBOOK` 早就写死了。

用法
----
  python training/linec_contribution.py --a-onnx E:\SSP_Work\onnx\aigen_v7.onnx ^
      --b-onnx E:\SSP_Work\onnx\localdet9.onnx --root E:\SSP_Work\probe
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# (目录名, 说明, 类别)  类别: 未训 / 对照 / 训过
DEFAULT_SETS = [
    ("heldout_gpt2",      "gpt-image-2 白图(跨厂商)",      "未训"),
    ("heldout_gpt2blue",  "gpt-image-2 蓝图(跨厂商)",      "未训"),
    ("heldout_qwen26",    "千问2026 白图(跨版本)",          "未训"),
    ("heldout_q26b",      "千问2026 蓝图(跨版本)",          "未训"),
    ("nulledit",          "空编辑对照 白(未经任何模型)",     "对照"),
    ("nulledit_blue",     "空编辑对照 蓝(未经任何模型)",     "对照"),
]


def score_dir(py: str, a_onnx: Path, b_onnx: Path, d: Path) -> list[dict]:
    """调交付出去的那份参照实现打一遍分, 收 NDJSON。

    走 subprocess 直接拿 bytes 再按 utf-8 解 —— **不经过 PowerShell 重定向**,
    那玩意儿在 5.1 上写的是 UTF-16, 读回来第一个字节就炸。
    """
    cmd = [py, str(_HERE / "ssp_score_one.py"),
           "--a-onnx", str(a_onnx), "--b-onnx", str(b_onnx),
           "--input", str(d), "--no-summary"]
    r = subprocess.run(cmd, capture_output=True)
    out = r.stdout.decode("utf-8", errors="replace")
    recs = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue                      # 配置自动生成那行提示也走 stdout, 跳过
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    if not recs:
        err = r.stderr.decode("utf-8", errors="replace")[-600:]
        print(f"  !! {d.name} 一条都没解出来。stderr 尾部:\n{err}", flush=True)
    return recs


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="线路C 贡献度: 开和关各一个数")
    ap.add_argument("--a-onnx", type=Path, required=True)
    ap.add_argument("--b-onnx", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="probe 根目录")
    ap.add_argument("--sets", nargs="*", default=None,
                    help="覆盖默认清单(目录名)。默认只跑留出生成器和空编辑对照")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(_HERE))
    from ssp_decide import decide, load_config          # noqa: E402
    cfg = load_config(args.config or (_HERE / "ssp_config.json"))

    sets = ([(s, s, "自选") for s in args.sets] if args.sets else DEFAULT_SETS)

    print("\n只打一遍分, 然后用同一个 decide() 算两遍判定(带线路C / 不带线路C)\n", flush=True)
    rows = []
    for name, desc, kind in sets:
        d = args.root / name
        if not d.is_dir():
            print(f"  跳过 {name}: 目录不存在", flush=True)
            continue
        print(f"  打分 {name} ...", flush=True)
        recs = score_dir(sys.executable, args.a_onnx, args.b_onnx, d)
        if not recs:
            continue

        tally = {True: {"自动拒": 0, "人工复核": 0, "放行": 0},
                 False: {"自动拒": 0, "人工复核": 0, "放行": 0}}
        c_hit = 0
        for r in recs:
            a = r.get("line_a") or {}
            b = r.get("line_b") or {}
            c = r.get("line_c") or {}
            hard = set(c.get("hard") or [])
            if hard:
                c_hit += 1
            ext = "." + str(r.get("image", "")).rsplit(".", 1)[-1].lower()
            at = (float(a["score"]), True) if a.get("score") is not None else None
            bt = ((float(b["score"]), bool(b.get("located")))
                  if b.get("score") is not None else None)
            for with_c in (True, False):
                dd, _ = decide(at, bt, cfg, ext, hard if with_c else None)
                tally[with_c][dd] = tally[with_c].get(dd, 0) + 1

        n = len(recs)
        rows.append((name, desc, kind, n, c_hit, tally))

    if not rows:
        raise SystemExit("!! 一个集合都没跑成")

    def pct(x: int, n: int) -> str:
        return f"{100.0 * x / n:5.1f}%" if n else "  -  "

    print("\n" + "=" * 100)
    print(f"{'集合':<18}{'类别':<6}{'n':>5}  "
          f"{'自动拒(带C)':>11}{'自动拒(无C)':>11}  "
          f"{'命中(带C)':>10}{'命中(无C)':>10}  {'C 独有':>8}")
    print("-" * 100)
    for name, desc, kind, n, c_hit, t in rows:
        hard_c = t[True]["自动拒"]
        hard_n = t[False]["自动拒"]
        hit_c = n - t[True]["放行"]
        hit_n = n - t[False]["放行"]
        print(f"{name:<18}{kind:<6}{n:>5}  "
              f"{pct(hard_c, n):>11}{pct(hard_n, n):>11}  "
              f"{pct(hit_c, n):>10}{pct(hit_n, n):>10}  "
              f"{pct(hit_c - hit_n, n):>8}")
        print(f"    {desc}   元数据命中 {c_hit}/{n} = {pct(c_hit, n)}")
    print("=" * 100)

    print("\n判读:")
    print("  ★ **'无C' 那两列才是有参考价值的数** —— 真实作案的图截个图元数据就没了。")
    print("  ★ 'C 独有' = 只有线路C 抓到、两条模型线都没抓到的比例。**这一块在真实环境里会消失。**")
    print("  ★ 空编辑对照那两行**应该接近 0** —— 它们没经过任何生成模型。")
    print("     如果不是 0, 说明模型在抓'这块被动过'而不是'这块被生成过', 结论要重写。")
    print("\n  注: 只有**留出生成器**(训练时没见过)的数能代表'对新 AI 的识别率'。")
    print("     训过的生成器会偏乐观, 默认清单里没放。")


if __name__ == "__main__":
    main()
