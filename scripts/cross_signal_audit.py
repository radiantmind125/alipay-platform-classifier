r"""跨信号审计:没有人工真值时,诚实评估"无人工能验证到什么程度"。

思路:分辨率规则(覆盖集)精度很高(~99%),可当"代理真值"来**标定**其它弱信号
(ICC/EXIF…)到底可不可信;再看 CNN 真正的战场(分辨率弃权集)上,还有多少
**独立**信号能验证它。只读分析,不动模型、不需 GPU、不需人工。

产出:
  - 终端:各信号覆盖率、用覆盖集标定的信号可靠度、CNN 战场上的独立验证覆盖、诚实结论
  - --out-dir 下:disagreements.jsonl(CNN 与某元数据信号冲突的样本 = 无人工"可疑/难例"集)

用法:
  python scripts\cross_signal_audit.py --inspect runs\pool_full\inspect.jsonl --results runs\pool_full\final_device_v2.jsonl --out-dir runs\pool_full\audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.bootstrap import resolution_platform  # noqa: E402

VOTE = ("ios", "android")


def _load(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "file" in r:
            out[r["file"]] = r
    return out


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    insp = _load(args.inspect)
    res = _load(args.results)
    files = [f for f in insp if f in res]

    # ---- 各信号(labeling function):每张图给 ios / android / abstain ----
    def lf_res(f: str) -> str:
        r = insp[f]
        try:
            return resolution_platform(int(r.get("width", 0)), int(r.get("height", 0)))
        except Exception:
            return "abstain"

    def lf_meta(field: str):
        def fn(f: str) -> str:
            v = insp[f].get(field, "abstain")
            return v if v in VOTE else "abstain"
        return fn

    def lf_cnn(f: str) -> str:
        r = res[f]
        # 只有走了 CNN 的行才算 CNN 的预测(source=cnn 就是分辨率弃权、由模型判的那批)
        return r["device"] if r.get("source") == "cnn" and r.get("device") in VOTE else "abstain"

    LFS = [("分辨率", lf_res), ("ICC", lf_meta("icc_vote")), ("EXIF", lf_meta("exif_vote")), ("CNN", lf_cnn)]

    n = len(files)
    print(f"样本:inspect∩results = {n}\n")

    # ---- 1) 覆盖率 ----
    print("【1】各信号覆盖率(它敢投 ios/android 的比例)")
    for name, fn in LFS:
        c = {"ios": 0, "android": 0, "abstain": 0}
        for f in files:
            c[fn(f)] = c.get(fn(f), 0) + 1
        cov = c["ios"] + c["android"]
        print(f"  {name:6} 投票 {cov:6}/{n} = {cov/n:5.1%}   (ios={c['ios']}, android={c['android']}, 弃权={c['abstain']})")

    # ---- 2) 覆盖集当代理真值 ----
    covered = [f for f in files if lf_res(f) in VOTE]
    abstain = [f for f in files if lf_res(f) == "abstain"]
    ci = sum(1 for f in covered if lf_res(f) == "ios")
    print(f"\n【2】分辨率覆盖集(代理真值)= {len(covered)}/{n} = {len(covered)/n:.1%}  (ios={ci}, android={len(covered)-ci})")
    print(f"     分辨率弃权集(CNN 的真正战场)= {len(abstain)}/{n} = {len(abstain)/n:.1%}")

    # ---- 3) 用覆盖集标定元数据信号是否可信 ----
    print("\n【3】用覆盖集标定元数据信号(它投票时,和'代理真值'对得上吗)")
    for name, fn in LFS:
        if name in ("分辨率", "CNN"):
            continue
        have = [f for f in covered if fn(f) in VOTE]
        if not have:
            print(f"  {name:6} 在覆盖集内无票 → 无法标定,基本没信号")
            continue
        agree = sum(1 for f in have if fn(f) == lf_res(f))
        # 该信号投 ios 时,真值其实是 android 的比例(误报)
        false_ios = sum(1 for f in have if fn(f) == "ios" and lf_res(f) == "android")
        verdict = "可信" if agree / len(have) >= 0.9 else ("基本不可信" if agree / len(have) <= 0.6 else "存疑")
        print(f"  {name:6} 覆盖集内有票 {len(have):5},与真值一致 {agree}/{len(have)} = {agree/len(have):5.1%} → {verdict}"
              + (f"（投ios却其实是android 的 {false_ios} 个 = 假iOS)" if false_ios else ""))

    # ---- 4) CNN 战场上的独立验证覆盖 ----
    print("\n【4】CNN 战场(分辨率弃权集)上,有多少'独立信号'能验证 CNN")
    dis_rows = []
    for name, fn in LFS:
        if name in ("分辨率", "CNN"):
            continue
        have = [f for f in abstain if fn(f) in VOTE and lf_cnn(f) in VOTE]
        if not have:
            print(f"  {name:6} 在弃权集内无票 → 对 CNN 零独立验证")
            continue
        agree = sum(1 for f in have if fn(f) == lf_cnn(f))
        print(f"  {name:6} 能验到 {len(have):5}/{len(abstain)} = {len(have)/max(1,len(abstain)):5.1%},与 CNN 一致 {agree}/{len(have)} = {agree/len(have):5.1%}")
        for f in have:
            if fn(f) != lf_cnn(f):
                dis_rows.append({"file": f, "signal": name, "signal_vote": fn(f), "cnn": lf_cnn(f),
                                 "confidence": res[f].get("confidence")})

    # ---- 5) 冲突集(无人工的可疑/难例) ----
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / "disagreements.jsonl"
        with out.open("w", encoding="utf-8") as w:
            for r in dis_rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n【5】CNN 与元数据冲突样本 = {len(dis_rows)} 条 → {out}(可当无人工的可疑/难例集)")

    # ---- 6) 诚实结论 ----
    print("\n【6】诚实结论")
    val_cov = 0
    for name, fn in LFS:
        if name in ("分辨率", "CNN"):
            continue
        val_cov += sum(1 for f in abstain if fn(f) in VOTE and lf_cnn(f) in VOTE)
    print(f"  - CNN 战场共 {len(abstain)} 张,其中被任一元数据信号独立验证到的约 {val_cov} 张"
          f"(≈{val_cov/max(1,len(abstain)):.1%});其余是无人工验证盲区。")
    print("  - 若上面元数据信号被判'基本不可信/无票',说明当前元数据不足以无人工评估 CNN;")
    print("    真正能突破的路子:①编码器指纹(需先验证转发重压缩后是否残留)②增强一致性(不需独立信号)③少量已知机型锚点(采集,非复核)。")


if __name__ == "__main__":
    main()
