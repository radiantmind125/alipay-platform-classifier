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


# 结果 JSON 里记原图路径的字段可能叫什么。按顺序试。
SOURCE_KEYS = ("input_image", "source", "image", "image_path", "file", "input")


def result_key(obj: dict, path: Path) -> str:
    """这份结果对应的是哪张输入图 —— 返回那张图的文件名主干。

    ★★ **不能按结果文件的文件名来配。** 实测批跑脚本把每张结果写成
       workers\\worker-NN\\results\\input-list\\<sha256>.json,
       文件名是**哈希**, 和输入图的名字毫无关系。
       按文件名配的话一个都配不上(实测 293 块全部配不上)。
       真正的对应关系在结果内部记着原图路径的那个字段里。
    """
    for k in SOURCE_KEYS:
        v = obj.get(k)
        if v:
            return Path(str(v)).stem
    return path.stem                      # 实在找不到就退回文件名


def load_results(results_dir: Path) -> tuple[dict[str, dict], int, set]:
    """读出所有单张结果。返回(按输入图主干索引的结果, 没认出来源的份数, 见过的键)。"""
    out: dict[str, dict] = {}
    unknown = 0
    keys_seen: set = set()
    for p in sorted(results_dir.rglob("*.json")):
        if p.name.lower().startswith(SKIP_PREFIX):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue                      # 清单是个数组, 不是单张结果
        keys_seen.update(obj.keys())
        if not any(obj.get(k) for k in SOURCE_KEYS):
            unknown += 1
        out[result_key(obj, p)] = obj
    return out, unknown, keys_seen


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

    results, unknown, keys_seen = load_results(a.results)
    if unknown:
        print(f'★ 有 {unknown} 份结果里找不到记原图路径的字段。')
        print(f'  结果里实际出现过的键: {sorted(keys_seen)}')
        print('  —— 把其中一份结果发我, 我把字段名加进 SOURCE_KEYS。')
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
    if missing == len(manifest) and manifest:
        print("★★ 一块都没配上 —— 多半是结果里记原图路径的字段名和预期不一样。")
        print(f"   清单里的块名例如: {Path(manifest[0]['tile']).stem}")
        ks = sorted(results.keys())[:3]
        print(f"   结果那边认出来的键例如: {ks}")
        print("   两边对不上就把一份结果 JSON 发我。")
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
