"""读诊断跑出来的结果 JSON, 判"订单号读不全"到底是哪种病。

用法(两组都要给, 只给一组没法下结论):

    python analyze_voucher.py --pinyin 白图组输出目录 --control 对照组输出目录 \
                             --field 订单号字段名 --max-len 长度上限

为什么一定要对照组
------------------
订单号实测有 28 位和 32 位两种, **都超过识别流程的单行长度上限**。
要是那个上限在推理时生效, 那每一张都会被截, 跟拼音没关系 —— 只有对照组
能把这种情况和"拼音专有的毛病"分开。

★ 上限具体多少、字段具体叫什么, 都用命令行参数传, 脚本里不写死。

★ 这个脚本**不改任何东西**, 只读 JSON。
★ 没见过真的输出文件, 所以对字段名是**防御性**的: 找不到就把实际有的键打出来,
  不猜、不硬套。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

# 订单号字段叫什么, 用 --field 传进来(不同版本的输出可能不一样)。
# 这里只放几个通用写法当兜底。
FALLBACK_KEYS = ("order_no", "order_number", "订单号")
TYPE_KEYS = ("voucher_type", "document_type", "类型")

# 批处理产物, 不是单张结果, 要排掉
SKIP_PREFIX = ("batch", "worker", "inference", "summary", "manifest", "sha256sums")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """小样本比例的置信区间。别直接拿 k/n 当真值。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def pick(d: dict, keys) -> tuple[str | None, object]:
    for k in keys:
        if k in d:
            return k, d[k]
    return None, None


def load_dir(root: Path, field: str) -> tuple[list[dict], Counter, set]:
    """返回(逐张记录, 出问题的计数, 见到过的顶层键)"""
    rows: list[dict] = []
    issues: Counter = Counter()
    keys_seen: set = set()
    keys = (field,) + FALLBACK_KEYS

    for p in sorted(root.rglob("*.json")):
        if p.name.lower().startswith(SKIP_PREFIX):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception as e:
            issues[f"读不了({e.__class__.__name__})"] += 1
            continue
        if not isinstance(obj, dict):
            issues["不是对象"] += 1
            continue

        keys_seen.update(obj.keys())
        vkey, v = pick(obj, keys)
        if vkey is None:
            issues["没有订单号字段"] += 1
            rows.append({"file": p.name, "len": None, "missing_key": True})
            continue

        _, vtype = pick(obj, TYPE_KEYS)
        if v is None:
            rows.append({"file": p.name, "len": -1, "value": None, "type": vtype})
        else:
            s = str(v).strip()
            rows.append({"file": p.name, "len": len(s), "value": s, "type": vtype})
    return rows, issues, keys_seen


def summarize(name: str, rows: list[dict], issues: Counter, keys_seen: set,
              max_len: int) -> dict:
    print(f"\n{'=' * 58}")
    print(f"{name}   共 {len(rows)} 张")
    print("=" * 58)

    if issues:
        print("  读取问题:")
        for k, n in issues.most_common():
            print(f"    {k}: {n}")

    if not rows:
        print("  没有可用记录")
        return {"n": 0}

    if any(r.get("missing_key") for r in rows):
        print(f"\n  ★ 有记录里找不到订单号字段。实际见到的顶层键:")
        print(f"    {sorted(keys_seen)}")
        print("    —— 字段名对不上, 先别往下读结论。")

    lens = [r["len"] for r in rows if r.get("len") is not None]
    n = len(lens)
    if n == 0:
        return {"n": 0}

    nulls = sum(1 for x in lens if x == -1)
    good = [x for x in lens if x >= 0]

    print(f"\n  长度分布:")
    for L, c in sorted(Counter(lens).items()):
        tag = "  <- null" if L == -1 else ("  <- 恰好卡在上限, 可疑" if L == max_len else "")
        bar = "#" * min(40, c)
        print(f"    {L:>4}  {c:>4}  {bar}{tag}")

    lo, hi = wilson(nulls, n)
    print(f"\n  null 率 {nulls}/{n} = {nulls/n:.1%}   95%区间 [{lo:.1%}, {hi:.1%}]")

    if good:
        exact25 = sum(1 for x in good if x == max_len)
        print(f"  非 null 的 {len(good)} 张: 长度 {min(good)} ~ {max(good)}, "
              f"恰好卡在上限({max_len})的有 {exact25} 张")
    return {"n": n, "nulls": nulls, "null_rate": nulls / n,
            "lens": lens, "good": good,
            "exact25": sum(1 for x in good if x == max_len) if good else 0}


def verdict(py: dict, ct: dict, max_len: int) -> None:
    print(f"\n{'=' * 58}")
    print("结论")
    print("=" * 58)

    if not py.get("n") or not ct.get("n"):
        print("  两组里有一组是空的, 下不了结论。")
        return

    p_null, c_null = py["null_rate"], ct["null_rate"]
    p25 = py["exact25"] / len(py["good"]) if py["good"] else 0
    c25 = ct["exact25"] / len(ct["good"]) if ct["good"] else 0

    plo, phi = wilson(py["nulls"], py["n"])
    clo, chi = wilson(ct["nulls"], ct["n"])
    separated = plo > chi          # 区间不重叠才算真有差别

    print(f"  白图组 null {p_null:.1%} [{plo:.1%}, {phi:.1%}]")
    print(f"  对照组 null {c_null:.1%} [{clo:.1%}, {chi:.1%}]")

    if p25 > 0.3 or c25 > 0.3:
        print(f"\n  ★★ 大量长度**恰好卡在上限 {max_len}** ——")
        print("     说明那个单行长度上限在推理时是生效的, 订单号实测 28/32 位, 被截了。")
        if c25 > 0.3:
            print("     对照组**也是**这样, 所以**跟拼音无关**。")
            print("     → 改一个参数的事。归集的那批图原地不动, 别开训。")
        else:
            print("     只有拼音组这样, 要再看为什么拼音会触发。")
        return

    if not separated:
        print("\n  ★★ 两组的 null 率**区间重叠**, 没有统计上的差别。")
        print("     → 订单号读不出**不是拼音特有的问题**。")
        print("     很可能是订单号这个字段本身就没做完(跨行合并 flux 答应过但仓库里找不到),")
        print("     或者别名门的三要素判定没过。")
        print("     → 先回去问清楚, 别拿归集的那批图开训。")
        return

    print("\n  ★★ 白图组的 null 率**显著高于**对照组, 是拼音特有的。")
    print("     下一步分两条:")
    print("       a) 值读出来了但短 -> 检测框把字裁掉了, 调检测框的扩张参数(旁路副本上试)")
    print("       b) 值直接是 null  -> 多半是标签'订单号'被它头上的拼音污染,")
    print("          导致别名门匹配不上。这种**重训 rec 治不好**,")
    print("          要么让标签匹配容忍拼音, 要么按位置取值(TextRegion 那两个谓词)。")

    if py["good"]:
        short = [x for x in py["good"] if 0 < x < max_len]
        print(f"\n     白图组非 null 里, 长度不足 {max_len} 的有 {len(short)} 张")
        print(f"     -> {'偏向 a' if len(short) > len(py['good']) * 0.3 else '偏向 b'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinyin", required=True, type=Path, help="白图拼音组的输出目录")
    ap.add_argument("--control", required=True, type=Path, help="对照组的输出目录")
    ap.add_argument("--dump-csv", type=Path, default=None)
    ap.add_argument("--field", required=True,
                    help="结果 JSON 里订单号那个字段叫什么, 按实际输出填")
    ap.add_argument("--max-len", type=int, required=True,
                    help="识别流程的单行长度上限, 按实际配置填")
    a = ap.parse_args()

    for d in (a.pinyin, a.control):
        if not d.exists():
            print(f"目录不在: {d}")
            return

    pr, pi, pk = load_dir(a.pinyin, a.field)
    cr, ci, ck = load_dir(a.control, a.field)
    py = summarize("白图拼音组", pr, pi, pk, a.max_len)
    ct = summarize("对照组", cr, ci, ck, a.max_len)
    verdict(py, ct, a.max_len)

    if a.dump_csv:
        import csv
        with a.dump_csv.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["组", "文件", "长度", "值"])
            for grp, rows in (("拼音", pr), ("对照", cr)):
                for r in rows:
                    w.writerow([grp, r.get("file", ""), r.get("len", ""), r.get("value", "")])
        print(f"\n明细 -> {a.dump_csv}")


if __name__ == "__main__":
    main()
