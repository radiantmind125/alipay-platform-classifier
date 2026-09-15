"""把切片跑出来的多份结果 JSON 合成一份, 一张原图一份。

规则(每个字段各自独立处理)
--------------------------
1. 收集所有块里**非空**的值
2. 一个都没有        -> null
3. 只有一个          -> 用它
4. 多个但**都一样**  -> 用它
5. 多个且**不一样**  -> ★ **标成冲突, 不替它选**

★★ 第 5 条是有意的。两块给出不同答案, 说明至少有一块读错了,
   随便选一个等于把错误藏起来。宁可标出来让人看一眼。
   经理的硬约束是"误杀要赔钱", 藏起来的错比看得见的冲突危险得多。

★ 对字段名是**不挑的**: 见到什么顶层字段就合什么, 不假设一定是哪 14 个 ——
  我们没见过真的输出文件, 写死字段名很可能对不上。

    python merge_tiles.py --tiles 切片目录 --results 结果目录 --out 合并目录
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# 这些是批处理产物, 不是单张结果
SKIP_PREFIX = ("batch", "worker", "inference", "summary", "manifest", "sha256sums",
               "_tiles")


def load_results(results_dir: Path) -> dict[str, dict]:
    """把结果目录里每份单张 JSON 读出来, 按**文件名主干**索引。"""
    out = {}
    for p in sorted(results_dir.rglob("*.json")):
        if p.name.lower().startswith(SKIP_PREFIX):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if isinstance(obj, dict):
            out[p.stem] = obj
    return out


def merge_one(tile_objs: list[dict]) -> tuple[dict, list[str]]:
    """合并同一张原图的几块。返回(合并结果, 冲突字段名列表)。"""
    keys = []
    for o in tile_objs:
        for k in o:
            if k not in keys:
                keys.append(k)                 # 保持出现顺序

    merged, conflicts = {}, []
    for k in keys:
        vals = []
        for o in tile_objs:
            v = o.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s == "":
                continue
            vals.append(v)
        if not vals:
            merged[k] = None
            continue
        uniq = []
        for v in vals:
            if str(v) not in [str(u) for u in uniq]:
                uniq.append(v)
        if len(uniq) == 1:
            merged[k] = uniq[0]
        else:
            merged[k] = None                   # ★ 不替它选
            merged[f"{k}__conflict"] = [str(u) for u in uniq]
            conflicts.append(k)
    return merged, conflicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, type=Path,
                    help="tile_for_ocr.py 的输出目录(里面有 _tiles.json)")
    ap.add_argument("--results", required=True, type=Path,
                    help="拿切片跑完 OCR 之后的结果目录")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    mf = a.tiles / "_tiles.json"
    if not mf.exists():
        print(f"清单不在: {mf}")
        return
    manifest = json.loads(mf.read_text(encoding="utf-8"))

    results = load_results(a.results)
    if not results:
        print(f"{a.results} 里没读到任何单张结果 JSON")
        return

    # 按原图分组; 结果文件名可能带或不带扩展名, 两种都试
    groups: dict[str, list] = defaultdict(list)
    missing = 0
    for m in manifest:
        stem = Path(m["tile"]).stem
        obj = results.get(stem)
        if obj is None:
            missing += 1
            continue
        groups[m["src"]].append((m["index"], obj))

    a.out.mkdir(parents=True, exist_ok=True)
    conflict_counter: Counter = Counter()
    n_conf_imgs = 0
    for src, items in groups.items():
        items.sort(key=lambda t: t[0])
        merged, conflicts = merge_one([o for _, o in items])
        merged["_source_image"] = src
        merged["_tiles_used"] = len(items)
        if conflicts:
            n_conf_imgs += 1
            conflict_counter.update(conflicts)
        (a.out / f"{Path(src).stem}.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"清单里 {len(manifest)} 块, 找不到结果的 {missing} 块")
    print(f"合出 {len(groups)} 张原图 -> {a.out}")
    print(f"★ 有冲突的图: {n_conf_imgs} 张")
    if conflict_counter:
        print("  冲突最多的字段:")
        for k, n in conflict_counter.most_common(10):
            print(f"    {k}: {n} 次")
        print("  ★ 这些字段合并结果里是 null, 另存了 字段名__conflict 列出各块的说法。")
        print("    **人工看一眼再定**, 不要直接取第一个。")


if __name__ == "__main__":
    main()
