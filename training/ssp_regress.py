r"""回归测试: 把当前行为冻住, 以后任何改动**秒级**验证有没有改坏(只读源图)。

为什么值得单独做一个
--------------------
这一周在这条线上抓到的 bug, 全是**"输出看着正常, 内容是错的"**那一类:
定位率分母搞错(96.0% vs 真值 98.0%)、`--model_root` 给成文件、两条线都失败的图凭空消失、
`_find_url` 把 URL 截断……**每一个都不会报错**, 只会安静地给出一个像模像样的错数。
而唯一能在几秒内发现它们的办法, 就是**跟一份冻住的期望值逐项比**。

三层
----
1. **判定逻辑层**(不用图不用模型, 毫秒级): 给定分数组合, 判定必须是这个。
   `.jpeg` 免自动拒、定位失败不出信号、两条线都没分送人工、线路C 铁证优先 —— 全在这里钉死。
2. **读表层**(直接喂 CSV, 毫秒级): 验从 summary.csv 到分数这一段。
   ★ 这一层是**吃了亏才补的**: 第 1 层用 `a=None` 断言过"两条线都没分 -> 人工复核",
   测试一直是绿的, 但线上 `a` **永远不可能是 None** —— 线路A 打分失败时 CSV 里写的是 0.0,
   打不开的图于是被当成"0 分真图"放行, 那道保险从来没触发过。
   **用生产产生不了的输入去断言安全性质, 等于没测。**
3. **端到端层**(要图要模型): 固定一批图重打一遍, 分数和判定都必须和基线一致。

★ 真图不能进仓库(仓库是公开的), 所以基线里**只存文件名和分数**, 不存图。

用法
----
  # 第一次: 冻基线(挑一个不会被删的目录)
  python training/ssp_regress.py --make --input D:\probe\regress_imgs --baseline D:\probe\regress.json

  # 以后每次改完
  python training/ssp_regress.py --check --input D:\probe\regress_imgs --baseline D:\probe\regress.json

  # 只跑判定逻辑那一层(不用图, 随时可跑)
  python training/ssp_regress.py --logic-only

★ **冻完基线立刻 --check 一次。** 这一步不是走过场:
  线路A 走的是官方 SSP 那条**随机取块**的路(`predict_tiled` 的文档里明说"原版随机取块 同图两次分数会不同",
  它自己改成确定性正是为了修这个)。所以第一次 check 若对不上, **那不是回归, 是这条管线本身不确定** ——
  同一张图两次跑可能一次拒一次放, 这对"会自动拒用户"的系统是必须知道的事。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---- 第 1 层: 判定逻辑(不用图, 不用模型) ----
# 每条都对应一个真实踩过或必须守住的行为。改判定规则时**先看这张表会不会红**。
_LOGIC_CASES = [
    # (A分, B分, B定位, 扩展名, 线路C铁证, 期望判定, 这条守的是什么)
    (0.10, 0.10, True, ".jpg", None, "放行", "两条线都低于复核线"),
    (0.70, 0.10, True, ".jpg", None, "人工复核", "只有线路A 过复核线"),
    (0.10, 0.80, True, ".jpg", None, "人工复核", "只有线路B 过复核线"),
    (0.10, 0.99, False, ".jpg", None, "放行", "★B 分很高但没定位到 -> 不出信号, 不是判假"),
    (0.90, 0.10, True, ".jpg", None, "自动拒", "线路A 过严格线"),
    (0.10, 0.99, True, ".jpg", None, "自动拒", "线路B 过严格线且定位成功"),
    (0.99, 0.99, True, ".jpeg", None, "人工复核", "★.jpeg 过严格线也只送人工(改名绕不过去)"),
    (0.10, 0.10, True, ".jpeg", None, "放行", ".jpeg 分数低照常放行, 不白占人工"),
    (0.90, None, None, ".png", None, "自动拒", "只有线路A 有分"),
    (None, 0.99, True, ".png", None, "自动拒", "只有线路B 有分"),
    (None, None, None, ".png", None, "人工复核", "★两条线都没分 -> 送人工, 不放行"),
    (0.8031, 0.0, True, ".png", None, "自动拒", "边界: 正好等于线路A 严格线"),
    (0.8030, 0.0, True, ".png", None, "人工复核", "边界: 差一点点"),
    (0.6497, 0.0, True, ".png", None, "人工复核", "边界: 正好等于线路A 复核线"),
    (0.6496, 0.0, True, ".png", None, "放行", "边界: 差一点点"),
    (0.10, 0.10, True, ".png", ["TC260国标"], "自动拒", "★线路C 铁证: 模型都低分也拒"),
    (0.10, 0.10, True, ".jpeg", ["doubao"], "自动拒", "★.jpeg 豁免不适用于线路C(元数据改名去不掉)"),
    (None, None, None, ".png", ["C2PA生成"], "自动拒", "★两条线没分但元数据是铁证"),
]

_FIELDS = ("a_score", "b_score", "b_located", "decision")


def _thr(cfg: dict) -> dict:
    """基线里记下当时的阈值 —— 阈值一改, 判定当然会变, 那不叫回归。"""
    return {"A严": cfg["line_a"]["strict"], "A复": cfg["line_a"]["review"],
            "B严": cfg["line_b"]["strict"], "B复": cfg["line_b"]["review"]}


def run_logic(cfg: dict) -> int:
    from ssp_decide import decide
    bad = 0
    print(f"[判定逻辑] {len(_LOGIC_CASES)} 条")
    for a, b, loc, ext, ch, want, note in _LOGIC_CASES:
        A = (a, True) if a is not None else None
        B = (b, loc) if b is not None else None
        got, why = decide(A, B, cfg, ext, set(ch) if ch else None)
        if got != want:
            bad += 1
            print(f"  红 {ext:6s} A={a} B={b} 定位={loc} C={ch}")
            print(f"     期望 {want}, 实得 {got} ({why})   —— {note}")
    print(f"  {len(_LOGIC_CASES) - bad}/{len(_LOGIC_CASES)} 通过" + ("" if not bad else "  ★有红的"))
    return bad


def run_csv(cfg: dict) -> int:
    """第 1.5 层: 直接喂 CSV, 验**读表**这一段。

    ★ 这一层是补上来的, 因为第 1 层漏掉了一个 critical 级的洞:
      第 1 层用 `a=None` 断言"两条线都没分 -> 人工复核", **测试是绿的**,
      但线上 `a` 永远不可能是 None —— 线路A 打分失败时 summary.csv 里写的是 **0.0**,
      于是打不开的图被当成"0 分真图"**放行**, 那道保险从来没触发过。
      **用生产根本产生不了的输入去断言一条安全性质, 等于没测。**
    """
    import tempfile
    from ssp_decide import _read_scores, decide
    bad = 0
    cases = [
        # (CSV 内容, 期望这张图在不在结果里, 说明)
        ("image_name,final_ai_score,scores_json\nx.png,0.0,{}\n", False,
         "★ 打分失败(scores_json 为 {}) -> 必须当作没有分数"),
        ('image_name,final_ai_score,scores_json\nx.png,0.0,"{""Net_epoch_best"": 0.0}"\n', True,
         "真的是 0 分(scores_json 有内容) -> 正常保留"),
        ("image_name,final_ai_score\nx.png,0.0\n", True,
         "没有 scores_json 这一列(线路B 就是这样) -> 防护不能误伤"),
        ("image_name,final_ai_score,scores_json\nx.png,0.85,{}\n", False,
         "★ 高分但打分失败 -> 照样不算数, 不能拿它去自动拒"),
    ]
    print(f"[读表] {len(cases)} 条")
    with tempfile.TemporaryDirectory() as td:
        for i, (body, want_in, note) in enumerate(cases):
            p = Path(td) / f"c{i}.csv"
            p.write_text(body, encoding="utf-8")
            got = _read_scores(p, "final_ai_score", False)
            if ("x.png" in got) != want_in:
                bad += 1
                print(f"  红 期望{'保留' if want_in else '剔除'}, 实得{'保留' if 'x.png' in got else '剔除'}"
                      f"   —— {note}")
    # 端到端串一遍: 打分失败的图**必须**落到人工复核, 绝不能放行
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.csv"
        p.write_text("image_name,final_ai_score,scores_json\nx.png,0.0,{}\n", encoding="utf-8")
        a = _read_scores(p, "final_ai_score", False).get("x.png")
        d, why = decide(a, None, cfg, ".png", None)
        if d != "人工复核":
            bad += 1
            print(f"  红 打分失败的图判成了 {d}({why}), 应该是 人工复核")
    print(f"  {len(cases) + 1 - bad}/{len(cases) + 1} 通过" + ("" if not bad else "  ★有红的"))
    return bad


def run_meta() -> int:
    """第 1.6 层: 线路C 的元数据正则(不读图, 直接喂字节)。

    ★ 也是**吃了亏才补的**: 国标 AIGC 标签有三种载体, 早先只认了 XMP 命名空间那一种,
      PNG 的 `tEXtAIGC` 块和 JPEG EXIF 注释里的 `{"AIGC": ...}` **全漏了**。
      下面前三条就是从真实漏网文件里原样抄来的字节。
      后两条是**不能误报**的对照 —— 正则放宽最怕的就是把真截图也拖下水。
    """
    import ast as _a
    ws = _a.parse((_HERE / "watermark_scan.py").read_text(encoding="utf-8"))
    keep = [n for n in ws.body if isinstance(n, _a.Assign)
            and getattr(n.targets[0], "id", "") in ("_AIGC_HARD", "_AIGC_SOFT")]
    ns: dict = {"re": __import__("re")}
    exec(compile(_a.Module(body=keep, type_ignores=[]), "<ws>", "exec"), ns)
    HARD = ns["_AIGC_HARD"]

    cases = [
        (True, "PNG tEXt 块(实测漏网原文)",
         b'\x89PNG\x00\x00IHDR\x00\x00tEXtAIGC\x00{"ContentPropagator":"001191440101MA9Y9T4H7A00000"'
         b',"Label":"1","ProduceID":"u:100834564696/task:e5118399e50d44ca9b92b3205a6fe4b8"}'),
        (True, "JPEG EXIF 注释(实测漏网原文)",
         b'\xff\xd8Exif\x00\x00MM\x00*ASCII\x00{"AIGC": {"Label":"1","ContentProducer":'
         b'"001191440300708461136T1WFUO","ReservedCode1":"{\\"Type\\":\\" TC260PG\\",\\"AlgID\\":\\"sm3\\"'),
        (True, "XMP 命名空间(原有写法, 不能改坏)",
         b'<rdf:RDF xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/"><TC260:AIGC>1</TC260:AIGC>'),
        (False, "★真截图的 XMP/EXIF(不能误报)",
         b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description'
         b' xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"'
         b'><exif:UserComment><rdf:Alt><rdf:li>Screenshot</rdf:li></rdf:Alt></exif:UserComment>mntrRGB'),
        (False, "★相机照片的 EXIF(不能误报)",
         b'\xff\xd8Exif\x00\x00MM\x00*Make\x00HUAWEI\x00Model\x00P60\x00Software\x00HarmonyOS'),
        # ★ 抖音那个块**故意不当铁证**(证据只有 n=2), 这条钉住"它不能变成自动拒"
        (False, "★抖音 aigc 块只算软标记, **不能**升级成铁证",
         b'{"data":{"product":"aweme","aigc_info":"{\\"aigc_type\\":1,\\"is_sticker_aigc\\":0}"}}'),
    ]
    bad = 0
    print(f"[线路C 元数据] {len(cases)} 条")
    for want, note, data in cases:
        hit = sorted(k for k, p in HARD.items() if p.search(data))
        if bool(hit) != want:
            bad += 1
            print(f"  红 期望{'命中' if want else '**不**命中'}, 实得 {hit or '无'}   —— {note}")
    # 抖音块必须被**软标记**认出来(只统计不判定), 否则加了等于没加
    SOFT = ns["_AIGC_SOFT"]
    dy = b'{"product":"aweme","aigc_info":"{\\"aigc_type\\":1}"}'
    if not any(p.search(dy) for p in SOFT.values()):
        bad += 1
        print("  红 抖音 aigc 块**连软标记都没认出来** —— 那这条规则等于没加")
    print(f"  {len(cases) - bad}/{len(cases)} 通过" + ("" if not bad else "  ★有红的"))
    return bad


def score_dir(args, out: Path) -> dict[str, dict]:
    """跑一遍真正的入口(不是重写), 拿回每张图的分数和判定。"""
    cmd = [sys.executable, str(_HERE / "ssp_decide.py"), "--input", str(args.input), "--score",
           "--ssp-repo", str(args.ssp_repo), "--device", args.device, "--out", str(out)]
    if args.config:
        cmd += ["--config", str(args.config)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"ssp_decide 跑失败(退出码 {r.returncode})")
    got: dict[str, dict] = {}
    with open(out / "decisions.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            got[row["image_name"]] = {k: row.get(k, "") for k in _FIELDS}
    return got


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    sys.path.insert(0, str(_HERE))
    ap = argparse.ArgumentParser(description="回归测试: 判定逻辑 + 端到端分数(只读源图)")
    ap.add_argument("--make", action="store_true", help="冻基线")
    ap.add_argument("--check", action="store_true", help="跟基线比")
    ap.add_argument("--logic-only", action="store_true", help="只跑判定逻辑那一层, 不用图")
    ap.add_argument("--input", type=Path, default=None, help="固定的那批图(挑个不会被删的目录)")
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--ssp-repo", type=Path,
                    default=Path(__import__("ssp_decide")._drive_fallback(r"D:\SSP")))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--device", default=None,
                    help="不给就用配置里的 device(和 ssp_decide 同一套规则)。"
                         "基线在哪个设备上冻的核对时就该用哪个 —— cuda 和 cpu 的浮点结果本来就有微小差异")
    args = ap.parse_args()

    from ssp_decide import load_config
    cfg = load_config(args.config or (_HERE / "ssp_config.json"))
    if args.device is None:                      # 和 ssp_decide 同一套解析规则, 免得两边不一致
        args.device = cfg.get("device", "cpu")

    bad = run_logic(cfg) + run_csv(cfg) + run_meta()
    if args.logic_only:
        raise SystemExit(1 if bad else 0)
    if not (args.input and args.baseline):
        raise SystemExit("端到端那一层要给 --input 和 --baseline(或者用 --logic-only)")

    with tempfile.TemporaryDirectory() as td:
        got = score_dir(args, Path(td))

    if args.make:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(
            {"_说明": "SSP 回归基线。只存文件名和分数, **不存图**(仓库是公开的)。",
             "device": args.device, "阈值": _thr(cfg),
             "cases": got}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[端到端] 已冻 {len(got)} 张 -> {args.baseline}")
        print("★ **现在立刻再跑一次 --check。** 对不上就说明这条管线本身不确定(线路A 是随机取块的),")
        print("  那是必须知道的事实, 不是测试写错了。")
        raise SystemExit(1 if bad else 0)

    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    exp = base.get("cases", {})
    # 差异有三个来源, 先把**不是代码改坏**的两个排掉, 免得白查一遍
    if base.get("阈值") != _thr(cfg):
        print("\n  !!!! **配置里的阈值和基线不一样了** —— 判定的差异多半来自这个, 不是代码改坏了")
        print(f"       基线 {base.get('阈值')}\n       现在 {_thr(cfg)}")
    if base.get("device") and base["device"] != args.device:
        print(f"\n  !!!! **设备不一样**: 基线在 {base['device']} 上冻的, 这次是 {args.device} ——")
        print("       cuda 和 cpu 的浮点结果本来就有微小差异, 分数末位对不上属正常, 看判定有没有变")

    miss = sorted(set(exp) - set(got))
    extra = sorted(set(got) - set(exp))
    diffs, maxd = [], 0.0
    for nm in sorted(set(exp) & set(got)):
        for k in _FIELDS:
            e, g = str(exp[nm].get(k, "")), str(got[nm].get(k, ""))
            if e == g:
                continue
            if k in ("a_score", "b_score"):
                try:
                    maxd = max(maxd, abs(float(e or 0) - float(g or 0)))
                except ValueError:
                    pass
            diffs.append((nm, k, e, g))

    print(f"\n[端到端] 基线 {len(exp)} 张 | 这次 {len(got)} 张")
    if miss:
        print(f"  !!!! 基线里有但这次没出结果的 {len(miss)} 张: {miss[:3]}")
    if extra:
        print(f"  (这次多出来 {len(extra)} 张, 基线里没有: {extra[:3]})")
    if not diffs:
        print("  全部一致 —— 分数和判定都没变")
    else:
        dec = [d for d in diffs if d[1] == "decision"]
        print(f"  !!!! {len(diffs)} 处不一致, 其中**判定变了的 {len(dec)} 张**; 分数最大偏差 {maxd:.6f}")
        for nm, k, e, g in diffs[:10]:
            print(f"       {nm[:44]:46s} {k:10s} 基线 {e}  ->  这次 {g}")
        if dec:
            print("\n  ★ 判定变了才是真问题。只有分数微动而判定没变, 多半是浮点/随机性, 看最大偏差有多大。")
    raise SystemExit(1 if (bad or diffs or miss) else 0)


if __name__ == "__main__":
    main()
