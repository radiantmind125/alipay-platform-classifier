r"""**上线第一版的唯一入口**: 一批图进来, 出"自动拒 / 人工复核 / 放行"。

这个文件存在的理由
------------------
到目前为止我们只有**研究脚本**和一堆量出来的数字, **没有一个能直接跑的东西**。
判定规则(两条线取并集)一直只活在 `combined_threshold.py` 的**分析代码**里,
`.jpeg` 免自动拒那条**压根没实现过**, 阈值散在命令行历史里。
这个脚本把量好的那套配置变成**一个可调用的产物**。

为什么是"编排"而不是"重写打分"
------------------------------
线路A 的打分在 SSP 仓库的 `predict_all_models.py` 里, 线路B 在 `predict_tiled.py` 里。
**这里一行打分逻辑都不重写** —— 直接调这两个脚本。
理由很实在: DEPLOY_SPEC 里每一个数字都是那两个脚本算出来的。只要重写, 分数就可能有细微出入,
而**出入不会报错, 只会让验收对不上, 然后花几天查**。编排的话, 验收是**天然**通过的。

判定规则(与 `combined_threshold.py` 逐字一致)
--------------------------------------------
    命中 = (A分 >= thrA) 或 (B定位成功 且 B分 >= thrB)

三种结果, 严格档的阈值比复核档高, 所以严格集合**完整包含**在复核集合里:

    过严格档 且 不是 .jpeg  -> 自动拒
    过复核档(或是 .jpeg 过了严格档) -> 人工复核
    都没过                  -> 放行

**`.jpeg` 永不自动拒**: `.jpg` 和 `.jpeg` 是同一种文件格式, 区别只在文件名,
所以"给 .jpeg 放宽阈值"会被改名破解; 而"只免自动拒、复核线照常"改名进来也只是被人看,
不构成绕过。实测这样代价只有 4.2/万 的人工量, 却换来 5 个覆盖点。

**两条线都拿不到分数的图 -> 送人工**, 不放行。无法判定不等于没问题。

用法
----
  # 1) 用已经算好的分数(验收/回归用, 秒级)
  python training/ssp_decide.py --a-csv D:\probe\gen100k_v7.csv --b-csv D:\probe\gen100k_b.csv \
      --out D:\probe\decide_check

  # 2) 从图片目录现打分(线上批量用; 会依次调用两条线的打分脚本)
  python training/ssp_decide.py --input D:\incoming --out D:\probe\decide_今天 \
      --ssp-repo D:\SSP --score

  # 3) 只想看配置
  python training/ssp_decide.py --print-config

配置在 `ssp_config.json`(不存在会自动写一份默认的)。**改阈值改这个文件, 不要改代码。**
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# 每个数字的出处都写在这里 —— 将来有人问"这 0.8031 哪来的", 不用翻聊天记录
DEFAULT_CONFIG = {
    "_说明": "SSP 上线第一版配置。阈值全部在 98,025 张真图上标定(排除名单 exclude_v4)。",
    "line_a": {
        "model": r"D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth",
        "score_col": "final_ai_score",
        "strict": 0.8031,
        "review": 0.6497,
        "_说明": "整图 AI 生成。阈值 = 1/5000 与 1/1000 预算下的第 k+1 高真图分数。",
    },
    "line_b": {
        "model": r"D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet9\Net_epoch_best.pth",
        "score_col": "tile_top3",
        "strict": 0.9811,
        "review": 0.7549,
        "flags": ["--roi-amount", "--amount-pad", "0", "--blue-locator",
                  "--agg", "top3", "--roi-top", "0.6"],
        "_说明": "局部改金额。**只在金额定位成功时出信号**, 定不到 = 无意见, 不是判真。",
    },
    "never_auto_reject_ext": [".jpeg"],
    "_预期指标": {
        "自动拒_每万": 2.7, "人工复核_每万": 18.4,
        "改金额覆盖_自动拒": 76.9, "改金额覆盖_复核": 95.8,
        "整图覆盖_万相": 99.4, "整图覆盖_千问": 100.0,
        "_说明": "验收时应当复现这几个数; 对不上说明装配有问题, 不是模型有问题。",
    },
    "_线上要盯的": {
        "定位率": 97.9, "蓝图占比": 42.0, "png占比": 49.7, "jpg占比": 49.9, "jpeg占比": 0.41,
        "_说明": "标定池的分布。**线上分布若与此明显不同, 阈值就是错的**, 要重标。",
    },
}


def load_config(path: Path) -> dict:
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"(没找到配置, 已写一份默认的到 {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scores(p: Path, col: str, need_located: bool) -> dict[str, tuple[float, bool]]:
    """读一条线的 summary.csv -> {文件名: (分数, 是否定位成功)}。"""
    out: dict[str, tuple[float, bool]] = {}
    dup = 0
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            nm = (r.get("image_name") or "").strip() or Path(r.get("image") or "").name
            v = (r.get(col) or "").strip()
            if not nm or not v:
                continue
            try:
                s = float(v)
            except ValueError:
                continue
            loc = True
            if need_located:
                loc = (r.get("roi_amount_located") or "1").strip() == "1"
            if nm in out:
                dup += 1
            out[nm] = (s, loc)
    if dup:
        # 同名会让两条线对错行。gen_local_ai_edit 按 <标记>_<编号> 命名, 不同批次同标记就会撞。
        print(f"  !!!! {p.name} 里有 {dup} 个重复文件名, 后面的覆盖了前面的 —— 两条线可能对错行, 先查清楚")
    return out


def decide(a: tuple[float, bool] | None, b: tuple[float, bool] | None,
           cfg: dict, ext: str) -> tuple[str, str]:
    """返回 (判定, 原因)。规则与 combined_threshold 一致。"""
    if a is None and b is None:
        return "人工复核", "两条线都没有分数"          # 无法判定 != 没问题
    la, lb = cfg["line_a"], cfg["line_b"]
    a_s = a[0] if a else None
    b_s, b_loc = (b[0], b[1]) if b else (None, False)

    hit_strict = (a_s is not None and a_s >= la["strict"]) or (b_loc and b_s is not None and b_s >= lb["strict"])
    hit_review = (a_s is not None and a_s >= la["review"]) or (b_loc and b_s is not None and b_s >= lb["review"])

    why = []
    if a_s is not None and a_s >= la["review"]:
        why.append(f"A={a_s:.4f}")
    if b_loc and b_s is not None and b_s >= lb["review"]:
        why.append(f"B={b_s:.4f}")
    reason = "+".join(why) or "两条线都低于复核线"

    if hit_strict and ext in cfg.get("never_auto_reject_ext", []):
        return "人工复核", reason + f" (但 {ext} 免自动拒)"
    if hit_strict:
        return "自动拒", reason
    if hit_review:
        return "人工复核", reason
    return "放行", reason


def _run_scoring(args, cfg: dict) -> tuple[Path, Path]:
    """现打分: 依次调两条线的脚本。**一行打分逻辑都不在这里。**"""
    out = Path(args.out)
    a_dir, b_dir = out / "_lineA", out / "_lineB"
    la, lb = cfg["line_a"], cfg["line_b"]

    t = time.time()
    print(f"\n[线路A] {la['model']}", flush=True)
    r = subprocess.run([sys.executable, "predict_all_models.py", "--model_root", la["model"],
                        "--input", str(args.input), "--output_dir", str(a_dir),
                        "--device", args.device], cwd=str(args.ssp_repo))
    if r.returncode != 0:
        raise SystemExit(f"线路A 打分失败(退出码 {r.returncode}) —— 先单独跑通它再回来")
    ta = time.time() - t

    t = time.time()
    print(f"\n[线路B] {lb['model']}", flush=True)
    r = subprocess.run([sys.executable, str(_HERE / "predict_tiled.py"),
                        "--ssp-repo", str(args.ssp_repo), "--model", lb["model"],
                        "--input", str(args.input), "--output_dir", str(b_dir),
                        *lb["flags"], "--device", args.device])
    if r.returncode != 0:
        raise SystemExit(f"线路B 打分失败(退出码 {r.returncode})")
    tb = time.time() - t
    print(f"\n打分耗时: 线路A {ta:.0f}s | 线路B {tb:.0f}s | 合计 {ta + tb:.0f}s")
    return a_dir / "summary.csv", b_dir / "summary.csv"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="SSP 判定入口: 两条线取并集, 出三种结果")
    ap.add_argument("--config", type=Path, default=_HERE / "ssp_config.json")
    ap.add_argument("--print-config", action="store_true", help="打印当前配置后退出")
    ap.add_argument("--a-csv", type=Path, default=None, help="线路A 已算好的 summary.csv")
    ap.add_argument("--b-csv", type=Path, default=None, help="线路B 已算好的 summary.csv")
    ap.add_argument("--input", type=Path, default=None, help="图片目录(配 --score 现打分)")
    ap.add_argument("--score", action="store_true", help="从 --input 现打分, 而不是用现成 CSV")
    ap.add_argument("--ssp-repo", type=Path, default=Path(r"D:\SSP"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=False, help="输出目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.print_config:
        print(json.dumps(cfg, ensure_ascii=False, indent=2)); return
    if not args.out:
        raise SystemExit("要给 --out")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.score:
        if not args.input:
            raise SystemExit("--score 要配 --input")
        a_csv, b_csv = _run_scoring(args, cfg)
    else:
        a_csv, b_csv = args.a_csv, args.b_csv
        if not (a_csv and b_csv):
            raise SystemExit("要么给 --a-csv 和 --b-csv, 要么给 --input 加 --score")

    A = _read_scores(Path(a_csv), cfg["line_a"]["score_col"], False)
    B = _read_scores(Path(b_csv), cfg["line_b"]["score_col"], True)
    names = sorted(set(A) | set(B))
    if not names:
        raise SystemExit("两个 CSV 都没读到分数")
    print(f"\n线路A {len(A):,} 张 | 线路B {len(B):,} 张 | 合计 {len(names):,} 张")
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    if only_a or only_b:
        print(f"  (只有A 的 {only_a} 张, 只有B 的 {only_b} 张 —— 正常情况下B 会少一些, 因为定位不到就不出信号)")

    rows, dec_n, ext_n = [], Counter(), Counter()
    n_loc = 0
    for nm in names:
        a, b = A.get(nm), B.get(nm)
        ext = Path(nm).suffix.lower()
        d, why = decide(a, b, cfg, ext)
        dec_n[d] += 1
        ext_n[ext] += 1
        n_loc += int(bool(b and b[1]))
        rows.append({"image_name": nm, "decision": d,
                     "a_score": f"{a[0]:.6f}" if a else "",
                     "b_score": f"{b[0]:.6f}" if b else "",
                     "b_located": int(bool(b and b[1])), "ext": ext, "reason": why})

    sp = args.out / "decisions.csv"
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {sp}")

    n = len(rows)
    print(f"\n{'判定':10s}{'张数':>9s}{'占比':>9s}{'每万':>9s}")
    print("-" * 38)
    for d in ("自动拒", "人工复核", "放行"):
        c = dec_n.get(d, 0)
        print(f"{d:10s}{c:>9,d}{c / n * 100:>8.2f}%{c / n * 1e4:>9.1f}")

    exp = cfg.get("_线上要盯的", {})
    print(f"\n分布自检(和标定池比, 差得多就说明阈值不适用):")
    print(f"{'指标':14s}{'这一批':>10s}{'标定池':>10s}")
    print("-" * 36)
    loc_rate = n_loc / max(1, len(B)) * 100 if B else 0.0
    print(f"{'定位率':14s}{loc_rate:>9.1f}%{exp.get('定位率', 0):>9.1f}%")
    for e, key in ((".png", "png占比"), (".jpg", "jpg占比"), (".jpeg", "jpeg占比")):
        print(f"{e:14s}{ext_n.get(e, 0) / n * 100:>9.2f}%{exp.get(key, 0):>9.2f}%")
    other = n - sum(ext_n.get(e, 0) for e in (".png", ".jpg", ".jpeg"))
    if other:
        print(f"{'其它扩展名':14s}{other / n * 100:>9.2f}%{'—':>10s}")
    print()
    print("★ 这几个分布是**线上唯一能自己发现阈值失效的手段** ——")
    print("  所有误杀/覆盖数字都建立在'进件长得像标定池'之上, 差得多就要重标, 不是调代码。")


if __name__ == "__main__":
    main()
