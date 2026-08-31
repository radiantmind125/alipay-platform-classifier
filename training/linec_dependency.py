r"""线路C 关掉的话, 那些**真实假图**还抓得住吗?

为什么要问这个
--------------
经理 2026-08-30 在群里说了两件事:

  1. "你加入元数据判断和我现有的程序判断有冲突, 我想下怎么解决"
  2. "目前观察SSP的作用很小"

第 1 条最省事的解法是**把线路C 关掉**(`--no-line-c`, 或配置里 `line_c.enabled=false`)。
但在关掉之前必须先回答一个问题: **线路C 现在到底在独扛多少?**

现有的证据只有**一张图** —— 那次负向对照, 把一张已知的 gpt-image 假图的元数据整条拿掉,
它仍然被线路A 和线路B **各自独立**判成自动拒。
**一张图推广不到一批上去。** 万一那批里大部分只有线路C 抓得到,
那"关掉线路C"就等于**把 SSP 在真实流量里的战果送掉大半** —— 而且是当着经理的面主动送掉。

★★ **先说清"多少张"这件事, 因为仓库里这个数是错的(2026-08-31 复核发现)**

`DEPLOY_SPEC:396` / `ssp_decide.py:111` / `ssp_config.json:33` 都写着
"10 万池里命中 **38** 张", 而且写在**线路C(纯元数据, 不用模型)**那一节下面。
**但 38 是 `watermark_scan` 的"水印 ∪ AIGC元数据"并集** ——
见 `watermark_scan.py:578` `names = sorted(wm_names | {...gen_hits})` 与 :581 的
"水印 N + 只有EXIF认的 M"; commit `1672afa` 原文也是"认出 38 张(**带水印或 AIGC 元数据**)"。
`DEPLOY_SPEC:63`「水印 / AIGC 元数据 | watermark_scan 认出的 38 张」才是正确写法。

-> **仓库里从来没有记录过"纯元数据"在 10 万池上单独命中多少张**,
   而且**不能用"38 减去水印命中"反推**。
   (另有 `linec_contribution.py:27` 记的是"靠元数据翻出过 **7** 张", 口径又不一样。)

**所以: 跑这个脚本时不要喂 `wm_hits.txt` 那种并集名单**(会把水印命中也算进线路C 的账),
**要用 `--scan` 现场只跑 `aigc_metadata()`** —— 那条路是纯元数据, 干净。
本机 12.4 万张实扫的参考值: **铁证 81 张 = 6.5/万**(豆包占 79%)。

它做什么
--------
`DEPLOY_SPEC` §标定池: 线路A 的 summary 有 **99,990 行**, 排除名单命中后才剩 98,013 张。
也就是说**那些铁证图的模型分数本来就在 CSV 里躺着**, 不用重新打分, 不用 GPU。

★ **但要注意用哪一份 CSV**(2026-08-31 复核发现的坑):
`accept_seeded\_lineA\summary.csv` 是**重搭池**跑出来的, 只有 **90,377 行** ——
原始四个目录已被删, 按文件名只找回 90.4%(`DEPLOY_SPEC:581-582`)。
拿它当分母, 会有约 9.6% 的铁证图**查不到分**, 被算成"两条线都没分数"->判人工复核,
**方向是低估关掉线路C 的代价**。下面会把"有多少张查不到分"当成硬自检打出来。

  1. 拿到线路C 铁证名单(**建议 `--scan` 现场扫**; 用现成名单务必确认它不是并集)
  2. 去线路A / 线路B 的 summary 里查它们的分
  3. **调 `ssp_decide.decide()` 本人**, 但 `c_hard=None` —— 也就是"假装线路C 不存在"
  4. 数一下: 有多少张会**放行**

★ 第 3 步故意 import 生产代码而不是照抄判定规则。
  这个项目已经栽过不止一次"两处实现慢慢长歪"的跟头(见 DEPLOY_SPEC 各处更正)。
  并集规则、`.jpeg` 免拒、线路B 要先定位到 —— 照抄一遍就是又开一个走样的口子。

怎么读结果
----------
- **放行 0 张** -> 线路C 是纯冗余, 关掉不掉战果, 可以放心答应经理
- **放行 一大半** -> 线路C 在独扛, 关掉要付真实代价, 得回去和经理谈别的解法
  (比如把 `on_hard` 从"自动拒"改成"人工复核", 冲突就没那么硬了)

用法
----
  # 有现成的线路C 名单
  python training/linec_dependency.py --hits D:\probe\wm_hits.txt ^
      --a-csv E:\SSP_Work\probe\accept_seeded\_lineA\summary.csv ^
      --b-csv E:\SSP_Work\probe\accept_seeded\_lineB\summary.csv

  # 没有名单, 现场扫(只读文件头, 不解码像素)
  python training/linec_dependency.py --scan D:\download2\BlueImages ^
      --a-csv ... --b-csv ...

**只读**: 不写任何东西到源目录, 只往 --out 写一份明细 CSV。
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ssp_decide import DEFAULT_CONFIG, decide, load_config, _read_scores  # noqa: E402

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _norm(s: str) -> str:
    """统一成纯文件名 —— 名单里有的带路径有的不带, 不统一就对不上。"""
    return os.path.basename(s.strip().replace("\\", "/"))


def _scan_hard(root: Path, limit: int = 0) -> dict[str, set[str]]:
    """现场扫铁证。只读文件头, 不解码像素(见 watermark_scan.aigc_metadata)。"""
    from watermark_scan import aigc_metadata

    hits: dict[str, set[str]] = {}
    seen = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() not in _EXTS:
                continue
            seen += 1
            if limit and seen > limit:
                break
            hard, _soft, _d = aigc_metadata(Path(dp) / fn)
            if hard:
                hits[fn] = hard
            if seen % 20000 == 0:
                print(f"    已扫 {seen:,} 张, 铁证 {len(hits)}", flush=True)
        if limit and seen > limit:
            break
    print(f"  扫完 {seen:,} 张, 铁证命中 {len(hits)} 张", flush=True)
    return hits


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="量一下: 关掉线路C 之后, 真实假图还抓得住多少")
    ap.add_argument("--hits", type=Path, default=None, help="现成的线路C 铁证名单(一行一个文件名)")
    ap.add_argument("--scan", type=Path, default=None, help="没有名单就现场扫这个目录")
    ap.add_argument("--limit", type=int, default=0, help="现场扫时最多看多少张")
    ap.add_argument("--a-csv", type=Path, required=True, help="线路A summary.csv")
    ap.add_argument("--b-csv", type=Path, default=None, help="线路B summary.csv")
    ap.add_argument("--config", type=Path, default=None, help="ssp_config.json; 不给就用内置默认")
    ap.add_argument("--out", type=Path, default=None, help="明细写这里")
    args = ap.parse_args()

    if not args.hits and not args.scan:
        raise SystemExit("!! --hits 和 --scan 至少要给一个")

    # ★ 不给 --config 就直接用内置默认。**不能**拿一个假路径去调 load_config ——
    #   它在文件不存在时会**把默认配置写到那个路径去**, 等于在当前目录拉一坨垃圾文件。
    cfg = load_config(args.config) if args.config else copy.deepcopy(DEFAULT_CONFIG)

    # ---- 1. 铁证名单 ----
    if args.hits:
        if not args.hits.is_file():
            raise SystemExit(f"!! 名单不存在: {args.hits}")
        names = [_norm(x) for x in args.hits.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        hits = {n: {"(名单)"} for n in names}
        print(f"线路C 铁证名单 {len(hits)} 张  <- {args.hits}")
    else:
        print(f"现场扫铁证: {args.scan}", flush=True)
        hits = _scan_hard(args.scan, args.limit)

    if not hits:
        raise SystemExit("!! 一张铁证都没有, 没什么可算的")

    # ---- 2. 两条模型线的分 ----
    la, lb = cfg["line_a"], cfg["line_b"]
    a_map = _read_scores(args.a_csv, la["score_col"], False)
    print(f"线路A summary {len(a_map):,} 行  <- {args.a_csv}")
    b_map: dict[str, tuple[float, bool]] = {}
    if args.b_csv and args.b_csv.is_file():
        b_map = _read_scores(args.b_csv, lb["score_col"], True)
        print(f"线路B summary {len(b_map):,} 行  <- {args.b_csv}")
    else:
        print("!! 没给线路B summary —— 结果只反映线路A, 会**低估**关掉线路C 之后的覆盖")

    # ---- 3. 假装线路C 不存在, 重新判一遍 ----
    rows = []
    tally: Counter[str] = Counter()
    missing = 0
    for name in sorted(hits):
        a = a_map.get(name)
        b = b_map.get(name)
        if a is None and b is None:
            missing += 1
        ext = os.path.splitext(name)[1].lower()
        # ★ c_hard=None = "线路C 关掉了"。判定规则完全走生产代码, 这里不重写。
        verdict, why = decide(a, b, cfg, ext, c_hard=None)
        tally[verdict] += 1
        rows.append({
            "image_name": name,
            "线路C铁证": "+".join(sorted(hits[name])),
            "线路A分": f"{a[0]:.6f}" if a else "",
            "线路B分": f"{b[0]:.6f}" if b else "",
            "B定位到": "1" if (b and b[1]) else "0",
            "关掉线路C后的判定": verdict,
            "依据": why,
        })

    n = len(rows)
    print(f"\n{'='*66}")
    print(f"把线路C 关掉之后, 这 {n} 张**真实假图**会怎么判")
    print(f"{'='*66}")
    for v in ("自动拒", "人工复核", "放行"):
        c = tally.get(v, 0)
        print(f"  {v:<6} {c:>4} 张   {100.0*c/n:>5.1f}%")
    other = {k: v for k, v in tally.items() if k not in ("自动拒", "人工复核", "放行")}
    for k, v in other.items():
        print(f"  {k:<6} {v:>4} 张   {100.0*v/n:>5.1f}%")
    # ★ 硬自检: 查不到分的比例。**这个数大, 结论就不能要。**
    if missing:
        pct = 100.0 * missing / n
        print(f"\n  !!!! **{missing} 张({pct:.1f}%)在两个 summary 里都查不到分**")
        print(f"       它们被算成'两条线都没有分数' -> 判**人工复核**, **不是真的被模型抓到**。")
        print(f"       方向: **低估**关掉线路C 的代价(该放行的被算成了复核)。")
        if pct > 5.0:
            print(f"       ★★ 超过 5% 了 —— 多半是**用错了 CSV**: "
                  f"`accept_seeded` 那份是重搭池, 只有 90,377 行(原池 99,990, 只找回 90.4%)。")
            print(f"       换 99,990 行那份(gen100k_v7.csv)再跑一遍; "
                  f"或者在报告里**把这个比例一起写上**, 别只报放行张数。")

    caught = tally.get("自动拒", 0)
    let_go = tally.get("放行", 0)
    print(f"\n{'-'*66}")
    if let_go == 0:
        print("-> **关掉线路C 一张都不会放行。** 线路C 在这批上是冗余的,")
        print("   可以答应经理关掉, 不必担心丢战果。")
    else:
        print(f"-> **关掉线路C 会放行 {let_go} 张({100.0*let_go/n:.0f}%)。**")
        print("   线路C 不是冗余, 它在独扛这部分。直接关掉要付真实代价。")
        print("   替代解法: 把 line_c.on_hard 从「自动拒」改成「人工复核」——")
        print("   冲突的是**判定结果**的话, 降一档就不打架了, 证据也没丢。")
    print(f"   (其中模型自己就能直接自动拒的: {caught} 张, {100.0*caught/n:.0f}%)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
