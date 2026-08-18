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
    "_说明": "SSP 上线第一版配置。四个阈值都在 **97,610 张真图**上标定(排除名单 exclude_v4.txt, "
             "剔了 4,738 张已查实假图/翻拍/非收据/.jpeg), 标定日期 2026-08-15。",
    "line_a": {
        # ★ 这里是**目录**不是 .pth —— `predict_all_models.py --model_root` 要的是"模型根目录",
        #   它自己在目录里按 --model_pattern 找权重(所以脚本叫 all_models: 根目录下有几个就跑几个)。
        #   给成 .pth 全路径会让它去**文件里面**找文件, 报"未找到模型文件"而路径看着完全正确。
        #   线路B 的 `predict_tiled --model` 反过来要的是 .pth 全路径, 两边不一样, 别改混。
        "model_root": r"D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7",
        "model_pattern": "Net_epoch_best.pth",
        "score_col": "final_ai_score",
        "strict": 0.8031,
        "review": 0.6497,
        "_说明": "整图 AI 生成。阈值 = 1/5000 与 1/1000 预算下的第 k+1 高真图分数。"
                 "**根目录下只能有这一个权重**, 多了就会多打几遍分, 列名也会变。",
    },
    "line_b": {
        # 这里是 **.pth 全路径**(和线路A 的 model_root 不同, 见上)
        "model": r"D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet9\Net_epoch_best.pth",
        "score_col": "tile_top3",
        "strict": 0.9811,
        "review": 0.7549,
        "flags": ["--roi-amount", "--amount-pad", "0", "--blue-locator",
                  "--agg", "top3", "--roi-top", "0.6"],
        "_说明": "局部改金额。**只在金额定位成功时出信号**, 定不到 = 无意见, 不是判真。",
    },
    "line_c": {
        "enabled": True,
        "on_hard": "自动拒",
        "_说明": "AIGC 元数据(不用模型, 每张约 1 毫秒)。**铁证**标记 = TC260 国标 / doubao / jimeng / "
                 "C2PA 的 trainedAlgorithmicMedia —— 都是生成器自己写进文件的, 裁掉水印也去不掉。"
                 "10 万池里命中 38 张, 逐张开图看过**全是假图**; 这 38 张本来就在排除名单里, "
                 "所以接上它**不改变 2.7/万**。它的价值是**覆盖我们没训过的生成器**, 包括国外的。"
                 "把 on_hard 改成 人工复核 可以更保守。"
                 "**注意 .jpeg 免自动拒对线路C 不适用**: 那条豁免是防"
                 "「改文件名绕开阈值」, 而元数据是生成器写进文件里的, 改名去不掉, "
                 "没有可绕的余地 —— 豁免它只会白丢覆盖。",
        "_软标记不参与判定": "retouch/醒图、meitu/美图、只有 C2PA 容器没有生成声明 —— "
                             "这些只说明**过了某个 App 或带了溯源信息**, 用户合法裁个图也会这样。"
                             "**只统计不判定**(DEPLOY_SPEC 原话: 单独报, 不进排除名单)。"
                             "命中率我们从没量过, 贸然送人工可能把复核队列撑爆, 所以先只看数。",
    },
    "never_auto_reject_ext": [".jpeg"],
    # ★ 线上是 **CPU 服务器**(这条从入职第一天就定了, 见 session §1)。GPU 只用于开发和训练。
    #   所以默认 cpu, 别在配置里改成 cuda 然后拿 GPU 的速度去估线上容量。
    "device": "cpu",
    "_算力提醒": "每张图约 20 次 ResNet-50 前向(线路A 16 个 patch + 线路B 4 个块, "
                 "SSPNet 会把每个 32x32 块放大到 256x256 再过骨干网)。**这在 CPU 上不便宜。** "
                 "注意: 线路A 的 --repeat 不能调小 —— 它会改变 final_ai_score, 0.8031 就不再等于 1/5000。",
    "_预期指标": {
        "自动拒_每万": 2.7, "人工复核_每万": 18.4,
        "改金额覆盖_自动拒": 76.9, "改金额覆盖_复核": 95.8,
        "整图覆盖_万相": 99.4, "整图覆盖_千问": 100.0,
        "_说明": "验收时应当复现这几个数; 对不上说明装配有问题, 不是模型有问题。",
    },
    "_线上要盯的": {
        "定位率": 98.0, "蓝图占比": 42.0, "png占比": 49.6, "jpg占比": 50.0, "jpeg占比": 0.41,
        "_说明": "标定池的分布, 口径是**含 .jpeg 的 98,013 张**(线上进件里 .jpeg 是存在的, "
                 "只是不参与定线)。**线上分布若与此明显不同, 阈值就是错的**, 要重标, 不是改代码。",
    },
}


def load_config(path: Path) -> dict:
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"(没找到配置, 已写一份默认的到 {path})")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # 配置文件是**第一次运行时自动落盘**的, 之后代码里加了新键它也不会自己更新。
    # 缺键不会报错, 只会安静地走 .get() 的兜底值 —— 比如少了 never_auto_reject_ext,
    # .jpeg 免自动拒就**无声失效**了。所以这里明说一声。
    miss = [k for k in DEFAULT_CONFIG if not k.startswith("_") and k not in cfg]
    sub = [f"{k}.{s}" for k, v in DEFAULT_CONFIG.items()
           if isinstance(v, dict) and isinstance(cfg.get(k), dict)
           for s in v if not s.startswith("_") and s not in cfg[k]]
    if miss or sub:
        print(f"  !!!! 配置比代码里的默认值**少了这些键**: {miss + sub}")
        print(f"       它们会走兜底值, 不会报错。想用新默认值就删掉 {path.name} 让它重新生成。")
    return cfg


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
            # ★ 线路A 打分失败时**不是空格子**: predict_all_models 先给每张图预置 final_ai_score=0.0,
            #   异常分支只往 detail.csv 写一行, summary.csv 里那张图照样是 0.0。
            #   不挡掉的话, 一张打不开的图会以"0 分"混进来, 被当成**证据确凿的真图放行** ——
            #   而"两条线都没分 -> 人工复核"那道保险**永远不会触发**。
            #   失败的判别特征: scores_json 是 {}(正常打分即使真是 0 分也会有 {"模型名": 0.0})。
            #   eval_summary.py 和 autoreject_threshold.py 早就这么挡了, 标定没受影响,
            #   漏的只有这条**真正上线的路径**。
            if "scores_json" in r and (r.get("scores_json") or "").strip() in ("", "{}"):
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
           cfg: dict, ext: str, c_hard: set[str] | None = None) -> tuple[str, str]:
    """返回 (判定, 原因)。两条模型线的规则与 combined_threshold 一致; 线路C 是额外的一条"或"。"""
    # 线路C: 生成器自己写在文件里的铁证。**不看模型分数**, 所以模型漏了它也能抓到。
    # 放在最前面: 它是最硬的证据, 而且 1 毫秒就能判完。
    if c_hard:
        lc = cfg.get("line_c", {})
        if lc.get("enabled", True):
            return lc.get("on_hard", "自动拒"), "线路C " + "+".join(sorted(c_hard))
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


def line_a_root(cfg: dict) -> str:
    """线路A 的模型**根目录**。

    容错: 配置里若不小心写成了 `.pth` 全路径, 自动取它所在目录 ——
    这正是把整个上线卡住的那个坑(`--model_root` 要目录, 给文件会报"未找到模型文件",
    而报错里回显的路径看着完全正确, 极难看出问题)。
    """
    v = cfg["line_a"].get("model_root") or cfg["line_a"].get("model") or ""
    p = Path(v)
    return str(p.parent if p.suffix.lower() == ".pth" else p)


def preflight(args, cfg: dict) -> None:
    """动手之前先把要用的东西全查一遍, **一次性把缺的都列出来**。

    起因: 配置里线路A 的模型路径写错了, 结果跑到 subprocess 深处才报
    "未找到模型文件" —— 而那时候已经建了输出目录、打印了一堆东西。
    路径这种事应该在第一秒就拦住, 而且要**一次说全**, 不要修一个再暴露下一个。
    """
    bad: list[str] = []
    repo = Path(args.ssp_repo)
    if not repo.is_dir():
        bad.append(f"SSP 仓库目录不存在: {repo}")
    elif not (repo / "predict_all_models.py").exists():
        bad.append(f"{repo} 里没有 predict_all_models.py(--ssp-repo 指对了吗)")
    if not (_HERE / "predict_tiled.py").exists():
        bad.append(f"同目录下没有 predict_tiled.py: {_HERE}")
    # 线路A 要的是**目录**, 线路B 要的是 **.pth 文件** —— 两边分别查, 别拿一套规则套两边
    root = Path(line_a_root(cfg))
    pat = cfg["line_a"].get("model_pattern", "Net_epoch_best.pth")
    if not root.is_dir():
        bad.append(f"线路A 的 model_root 不是目录: {root}")
    else:
        hits = sorted(root.glob(pat))
        if not hits:
            bad.append(f"线路A 目录里没有 {pat}: {root}")
        elif len(hits) > 1:
            bad.append(f"线路A 目录里有 {len(hits)} 个 {pat} —— 会打多遍分, 列名也会变: {root}")
        if not root.is_dir() or not hits:
            par = root.parent
            if par.is_dir():                     # 顺手把同级有哪些快照列出来, 省得再去翻
                subs = sorted(p.name for p in par.iterdir() if p.is_dir())[:20]
                if subs:
                    bad.append(f"    {par} 下现有: {', '.join(subs)}")
    mb = Path(cfg["line_b"]["model"])
    if not mb.is_file():
        bad.append(f"线路B 的 model 不是文件(这一条要 .pth 全路径): {mb}")
    if bad:
        raise SystemExit("跑不了, 先解决这些:\n  " + "\n  ".join(bad)
                         + f"\n\n改 {args.config} 里的 model 字段, 不要改代码。")


def _run_scoring(args, cfg: dict) -> tuple[Path, Path]:
    """现打分: 依次调两条线的脚本。**一行打分逻辑都不在这里。**"""
    out = Path(args.out)
    a_dir, b_dir = out / "_lineA", out / "_lineB"
    la, lb = cfg["line_a"], cfg["line_b"]

    t = time.time()
    a_root = line_a_root(cfg)
    print(f"\n[线路A] model_root={a_root}  pattern={la.get('model_pattern', 'Net_epoch_best.pth')}", flush=True)
    r = subprocess.run([sys.executable, "predict_all_models.py", "--model_root", a_root,
                        "--model_pattern", la.get("model_pattern", "Net_epoch_best.pth"),
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
    n_in = sum(1 for p in Path(args.input).rglob("*")
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    tot = ta + tb
    print(f"\n打分耗时({args.device}): 线路A {ta:.0f}s | 线路B {tb:.0f}s | 合计 {tot:.0f}s")
    if n_in:
        per = tot / n_in
        print(f"  每张 {per:.2f}s  ({n_in / max(tot, 1e-9):.1f} 张/秒)  "
              f"线路A 占 {ta / max(tot, 1e-9) * 100:.0f}%")
        print(f"  单进程一天(24h)约 {int(86400 / per):,} 张 —— "
              f"**拿这个数去对进件量**, 不够就上多进程或换 ONNX Runtime, "
              f"**不要动 --repeat 或切块数(那会改分数, 阈值就作废)**。")
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
    ap.add_argument("--device", default=None,
                    help="不给就用配置里的 device(默认 **cpu**, 因为线上是 CPU 服务器)。"
                         "只有开发/调试才该显式给 cuda。")
    ap.add_argument("--no-line-c", action="store_true",
                    help="不跑线路C。**只在纯核对阈值的时候用** —— 线路C 要逐张读文件头, "
                         "10 万张就是几十 GB 磁盘读; 而验收用的那个池子里那 38 张铁证本来就在排除名单里, "
                         "跑不跑都是 26/180。**线上不要加这个**, 那等于白扔一条免费的覆盖。")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="**验收专用**: 跳过名单里的文件名。标定时把已查实的假图和翻拍剔出了真图池, "
                         "要复现 2.7/万 那些数就得用同一个池子。**线上不要传这个** —— "
                         "线上没有'已知是假图'的名单, 每张都得判。")
    ap.add_argument("--out", type=Path, required=False, help="输出目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is None:
        args.device = cfg.get("device", "cpu")
    if args.device != "cpu" and args.score:
        print(f"  !!!! 正在用 {args.device} 打分 —— **线上是 CPU 服务器**, "
              f"这个速度不能拿去估线上容量。")
    if args.print_config:
        print(json.dumps(cfg, ensure_ascii=False, indent=2)); return
    if not args.out:
        raise SystemExit("要给 --out")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.score:
        if not args.input:
            raise SystemExit("--score 要配 --input")
        if not Path(args.input).is_dir():
            raise SystemExit(f"--input 不是目录: {args.input}")
        preflight(args, cfg)                     # 路径问题在第一秒拦住, 别跑到一半才报
        a_csv, b_csv = _run_scoring(args, cfg)
    else:
        a_csv, b_csv = args.a_csv, args.b_csv
        if not (a_csv and b_csv):
            raise SystemExit("要么给 --a-csv 和 --b-csv, 要么给 --input 加 --score")

    A = _read_scores(Path(a_csv), cfg["line_a"]["score_col"], False)
    B = _read_scores(Path(b_csv), cfg["line_b"]["score_col"], True)
    names_set = set(A) | set(B)

    # ★ 一张都不能丢: 两条线**都**没处理成功的图(比如文件损坏), 不会出现在任何一个 CSV 里,
    #   于是就从结果里**凭空消失**了。批量分析时无所谓, 线上"提交了一张但没有任何判定"
    #   比"被判去人工"糟得多。所以现打分时按输入目录点名, 缺的补进来走"两条线都没分"->人工复核。
    # 给了 --input 就以目录为准点名, 不限于 --score: 用现成 CSV 时同样可能有图两条线都没分。
    if args.input and Path(args.input).is_dir():
        _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        seen = {p.name for p in Path(args.input).rglob("*")
                if p.suffix.lower() in _EXTS and p.is_file()}
        missing = seen - names_set
        if missing:
            print(f"  !!!! 输入里有 {len(missing)} 张**两条线都没打出分**(多半是文件损坏), "
                  f"已按'无法判定'送人工, 不会丢:")
            for nm in sorted(missing)[:5]:
                print(f"         {nm}")
        names_set |= seen
    names = sorted(names_set)
    if args.exclude and args.exclude.exists():
        drop = {ln.strip().lstrip("﻿")
                for ln in args.exclude.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        before = len(names)
        names = [n for n in names if n not in drop]
        print(f"(验收模式: 按名单跳过 {before - len(names)} 张; **线上不该用这个开关**)")
    if not names:
        raise SystemExit("两个 CSV 都没读到分数")
    print(f"\n线路A {len(A):,} 张 | 线路B {len(B):,} 张 | 合计 {len(names):,} 张")
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    if only_a or only_b:
        print(f"  (只有A 的 {only_a} 张, 只有B 的 {only_b} 张 —— 正常情况下B 会少一些, 因为定位不到就不出信号)")

    # ---- 线路C: 元数据里的 AIGC 铁证(不用模型, 每张约 1 毫秒) ----
    # 要读原文件, 所以先把文件名映射到真实路径。**线路A 的 CSV 里那一列指向临时硬链接目录, 早删了**,
    # 所以优先用 --input, 其次用线路B 的 CSV(它指向原始下载目录)。
    c_hard: dict[str, set[str]] = {}
    c_soft: Counter = Counter()
    c_readable = 0
    lc = cfg.get("line_c", {})
    if args.no_line_c:
        print("\n(--no-line-c: 跳过线路C —— 只有纯核对阈值时才该这样)")
    elif lc.get("enabled", True):
        _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        path_of: dict[str, Path] = {}
        if args.input and Path(args.input).is_dir():
            # ★ 必须 sorted: rglob 的顺序由文件系统决定, 不排序的话**同名文件挑中哪一个都不一定**,
            #   而线路C 一旦命中铁证就直接自动拒(不看分数) —— 判定会随目录枚举顺序变。
            dup_c = 0
            for p in sorted(Path(args.input).rglob("*")):
                if p.suffix.lower() in _EXTS and p.is_file():
                    if p.name in path_of:
                        dup_c += 1
                        continue
                    path_of[p.name] = p
            if dup_c:
                print(f"  !!!! 输入目录里有 {dup_c} 个重复文件名 —— 线路C 只能读其中一个, "
                      f"**可能不是当初打分的那一张**。两条线的分数本来也会对错行, 先查清楚。")
        with open(b_csv, encoding="utf-8-sig") as fh:      # 线路B 的路径是活的
            for r in csv.DictReader(fh):
                nm = (r.get("image_name") or "").strip()
                p = (r.get("image") or "").strip()
                if nm and p and nm not in path_of:
                    path_of[nm] = Path(p)
        try:
            from watermark_scan import aigc_metadata
        except Exception as exc:  # noqa: BLE001
            print(f"  !!!! 线路C 载入失败({exc}), 这一轮不出信号")
            aigc_metadata = None  # type: ignore[assignment]
        if aigc_metadata:
            t = time.time()
            print(f"\n线路C: 逐张读文件头检查 AIGC 标记({len(names):,} 张)...", flush=True)
            for i, nm in enumerate(names, 1):
                p = path_of.get(nm)
                if not p or not p.exists():
                    continue
                c_readable += 1
                hard, soft, _ = aigc_metadata(p)
                if hard:
                    c_hard[nm] = hard
                for s in soft:
                    c_soft[s] += 1
                if i % 20000 == 0:               # 10 万张全程无输出会被当成卡死
                    print(f"  已查 {i:,}/{len(names):,}  (用时 {time.time() - t:.0f}s)", flush=True)
            print(f"\n线路C: 读到 {c_readable:,}/{len(names):,} 张的原文件, "
                  f"铁证 {len(c_hard)} 张, 软标记 {sum(c_soft.values())} 处, 用时 {time.time() - t:.0f}s")
            if c_readable == 0:
                # 读不到文件时命中数必然是 0, 那和"干净"长得一模一样 —— 必须说破
                print("  !!!! **一个原文件都没读到, 所以线路C 这一轮等于没跑**(不是'没有 AIGC 图')。"
                      "多半是 CSV 里的路径已失效, 给 --input 指到真实图片目录即可。")
            elif c_readable < len(names) * 0.9:
                print(f"  !!!! 有 {len(names) - c_readable:,} 张读不到原文件, 线路C 对这些图没出信号")
            if c_soft:
                print(f"  (软标记只统计不判定: {dict(c_soft.most_common())} —— "
                      f"过修图 App 不等于生成, 用户合法裁图也会这样)")

    rows, dec_n, ext_n = [], Counter(), Counter()
    n_loc = 0
    for nm in names:
        a, b = A.get(nm), B.get(nm)
        ext = Path(nm).suffix.lower()
        d, why = decide(a, b, cfg, ext, c_hard.get(nm))
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
    # 分母必须是**这一批实际参与判定的张数**, 不是 B 的 CSV 总行数。
    # 原来写成 len(B) —— 验收时 names 被排除名单筛到 98,013 而 len(B) 还是 99,990,
    # 于是定位率印成 96.0% 而真值是 98.0%。判定一点没错, 但这个数是**线上用来发现漂移的标尺**,
    # 系统性偏低 2 个点就会在上线第一天报一个自信的假警报。
    loc_rate = n_loc / max(1, len(names)) * 100
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
