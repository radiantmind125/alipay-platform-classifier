"""把两条路的结果按**字段**各取更好的那个, 合成一份。

为什么要这样
------------
实测两条路各治一簇:

    字段               原图   擦完   切片   天花板
    transfer_status      1     19     56     94     <- 切片好
    payment_method       4     36     50     91     <- 切片好
    transfer_note        2     10     31     44     <- 切片好
    amount               3      8      0     46     <- 擦图好(切片归零)
    voucher_type         1      8      1     39     <- 擦图好
    transfer_time        0      7      0     42     <- 擦图好
    voucher_number       0      7      0     41     <- 擦图好

后面那一簇要"账单分类+订单号+创建时间"三要素同时出现才启用, 而切片把整页拆了,
没有任何一块同时有三样, 所以全归零。

★ 所以同一批图跑两遍, 每个字段取各自更好的那一路。

★★ 谁更好是**按实测数据自动定的**, 不写死。但这就有个问题 ——
   在**同一批数据**上挑路线再在**同一批数据**上报成绩, 是自己考自己。
   所以脚本会明确警告, 并支持 --route-from 用**另一批**数据定路线。

    # 先在一批上定路线, 再在另一批上验
    python route_fields.py --run 擦完=A目录 --run 切片=B目录 --out 合并目录
    python route_fields.py --run 擦完=C目录 --run 切片=D目录 --out 目录2 --route-from 路线.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

SKIP_PREFIX = ("batch", "worker", "inference", "summary", "manifest",
               "sha256sums", "_tiles")
SOURCE_KEYS = ("_source_image", "input_image", "source", "image", "image_path")
META_PREFIX = ("_",)


# ★★ 各条路给文件加的后缀。要**剥掉**才能把两路的同一张图对上。
#   实测踩过: 擦图输出 `xxx_clean.png`, 切片合并后是 `xxx` ——
#   两边的键永远配不上, 于是取了并集(100+100=200 张), 每个字段只剩一半的值,
#   分流成绩整整低一倍。
SUFFIXES = ("_clean", "_erased", "_merged")


def norm_key(name: str) -> str:
    """把各条路加的后缀剥掉, 让同一张原图在两路里得到同一个键。"""
    k = Path(str(name)).stem
    changed = True
    while changed:                    # 后缀可能叠加
        changed = False
        for s in SUFFIXES:
            if k.endswith(s):
                k = k[: -len(s)]
                changed = True
    return k


def load_dir(d: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(d.rglob("*.json")):
        if p.name.lower().startswith(SKIP_PREFIX):
            continue
        try:
            o = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        src = next((o[k] for k in SOURCE_KEYS if o.get(k)), None)
        out[norm_key(src if src else p.stem)] = o
    return out


def filled(v) -> bool:
    return v is not None and str(v).strip() != ""


def mcnemar_p(b: int, c: int) -> float:
    """配对数据的精确检验(两条路跑的是**同一批图**, 所以必须用配对检验)。

    b = 第一路读出来了而第二路没有的张数
    c = 反过来的张数
    两路一样好的话, 这 b+c 张里落到哪边应当是五五开。

    ★ 为什么要这个: 路线是从当前这批数据挑的, 不做检验就是自己考自己。
      差距大的字段(56% vs 19%)换一批也一样, 差距小的(86% vs 85%)纯属噪声,
      把它们分开才知道哪些结论经得起换一批数据。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_counts(objs_a: dict, objs_b: dict, field: str) -> tuple[int, int]:
    """同一批图上, A 有 B 无 / B 有 A 无 各多少张。"""
    b = c = 0
    for k in objs_a:
        if k not in objs_b:
            continue
        fa = filled(objs_a[k].get(field))
        fb = filled(objs_b[k].get(field))
        if fa and not fb:
            b += 1
        elif fb and not fa:
            c += 1
    return b, c


def fill_rates(objs: dict[str, dict]) -> dict[str, float]:
    seen: Counter = Counter()
    hit: Counter = Counter()
    for o in objs.values():
        for k, v in o.items():
            if k.startswith(META_PREFIX) or k.endswith("__conflict"):
                continue
            seen[k] += 1
            if filled(v):
                hit[k] += 1
    return {k: (hit[k] / seen[k] if seen[k] else 0.0) for k in seen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="名字=目录, 至少给两个")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--route-from", type=Path, default=None,
                    help="用另一批数据定好的路线(json), 不传就从当前数据定")
    ap.add_argument("--save-route", type=Path, default=None,
                    help="把定出来的路线存下来, 给下一批用")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        if "=" not in spec:
            print(f"格式应当是 名字=目录: {spec}")
            return
        name, path = spec.split("=", 1)
        d = Path(path)
        if not d.exists():
            print(f"目录不在: {d}")
            return
        objs = load_dir(d)
        if not objs:
            print(f"{d} 里没读到结果")
            return
        runs.append((name, objs))
    if len(runs) < 2:
        print("至少要两条路才有得挑")
        return

    # ★★★ 先核两路的键对不对得上。对不上就没得合, 算出来的数是假的。
    keysets = [set(objs) for _, objs in runs]
    common = set.intersection(*keysets)
    union = set.union(*keysets)
    print("=" * 72)
    print(f"两路各自的图数: " + ",  ".join(f"{n} {len(s)}" for (n, _), s in zip(runs, keysets)))
    print(f"两路都有的: {len(common)}    合起来: {len(union)}")
    if not common:
        print()
        print("★★★ 两路**一张都对不上** —— 键的命名不一致, 没法合。")
        for (n, _), s in zip(runs, keysets):
            print(f"    {n} 的键例如: {sorted(s)[:2]}")
        print("    (各条路给文件加的后缀要在 SUFFIXES 里列出来才能剥掉)")
        return
    if len(common) < 0.8 * len(union):
        print(f"★★ 只有 {len(common)/len(union):.0%} 的图两路都有 —— 差得有点多, 结果要打折看。")
    print("=" * 72)
    print()

    # 只在两路都有的图上比, 否则一路缺的会被算成"没读出来"
    runs = [(n, {k: v for k, v in objs.items() if k in common}) for n, objs in runs]
    rates = {name: fill_rates(objs) for name, objs in runs}
    all_fields = sorted({k for r in rates.values() for k in r})

    if a.route_from:
        route = json.loads(a.route_from.read_text(encoding="utf-8"))
        print(f"★ 路线来自 {a.route_from}(**另一批**数据定的, 这次是真验证)")
    else:
        route = {}
        for k in all_fields:
            best = max(runs, key=lambda t: rates[t[0]].get(k, 0.0))[0]
            route[k] = best
        print("★★ 路线是**从当前这批数据**定的 —— 在同一批上挑路线又在同一批上报成绩,")
        print("   是自己考自己, **成绩会偏高**。")
        print("   要真验证: 加 --save-route 存下来, 再拿**另一批**图 --route-from 跑一遍。")
    print()

    print("=" * 84)
    print(f"{'字段':<22}" + "".join(f"{n:>11}" for n, _ in runs)
          + f"{'选谁':>10}{'这个选择稳不稳':>16}")
    print("=" * 84)
    shaky = []
    for k in sorted(all_fields, key=lambda k: -max(rates[n].get(k, 0) for n, _ in runs)):
        row = f"{k:<22}"
        for n, _ in runs:
            row += f"{rates[n].get(k, 0):>10.0%} "
        row += f"{route.get(k, runs[0][0]):>10}"
        if len(runs) == 2:
            b, c = paired_counts(runs[0][1], runs[1][1], k)
            pv = mcnemar_p(b, c)
            if b + c == 0:
                tag = "两路完全一样"
            elif pv < 0.01:
                tag = f"★ 稳 p={pv:.0e}"
            elif pv < 0.05:
                tag = f"较稳 p={pv:.2f}"
            else:
                tag = f"★★拿不准 p={pv:.2f}"
                shaky.append(k)
            row += f"{tag:>16}"
        print(row)
    print()
    if shaky:
        print(f"★★ 这几个字段两路差距在噪声范围内, 选哪路都行, 换一批可能反过来:")
        print(f"   {', '.join(shaky)}")
        print("   要紧的业务字段不在里面的话, 路线就是可信的。")
        print()

    # 合成: 每个字段从它该走的那条路取
    by_name = dict(runs)
    keys = set()
    for _, objs in runs:
        keys |= set(objs)
    a.out.mkdir(parents=True, exist_ok=True)
    combined: Counter = Counter()
    total = 0
    for key in sorted(keys):
        merged = {}
        for f in all_fields:
            src = by_name.get(route.get(f, runs[0][0]), {})
            o = src.get(key)
            merged[f] = o.get(f) if o else None
        merged["_source_image"] = key
        merged["_routed"] = {f: route.get(f) for f in all_fields}
        (a.out / f"{key}.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        total += 1
        for f in all_fields:
            if filled(merged[f]):
                combined[f] += 1

    print("=" * 72)
    print(f"分流之后 ({total} 张)")
    print("=" * 72)
    for f in sorted(all_fields, key=lambda f: -combined[f]):
        best_single = max(rates[n].get(f, 0) for n, _ in runs)
        got = combined[f] / total if total else 0
        mark = "" if abs(got - best_single) < 1e-9 else "  <- 和单路最好值不一致, 查一下"
        print(f"  {f:<22} {got:>6.0%}{mark}")
    print()
    if a.save_route:
        a.save_route.write_text(json.dumps(route, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"路线已存 -> {a.save_route}")
        print("★ 下一批用 --route-from 带上它, 那才是没有自己考自己的成绩。")


if __name__ == "__main__":
    main()
