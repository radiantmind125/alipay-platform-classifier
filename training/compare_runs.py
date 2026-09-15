"""比较几种跑法的 OCR 结果, 判哪一种更好 —— **不需要人工标注真值**。

怎么在没有真值的情况下判好坏
----------------------------
★★★ 核心指标: **输出里有没有混进拼音字母**。

拼音坏事的方式是检测把"拼音行 + 汉字行"框成一块, 识别就把拼音字母和汉字
一起吐出来 —— `dìngdānhào订单号` 这种。

而**带声调的元音**(ā á à ē é è ī í ǐ ì ō ó ǒ ò ū ú)在正常回单里
**根本不可能出现**: 金额、姓名、账号、邮箱、订单号, 没有一个会带声调符号。
所以输出里只要出现这些字符, **一定是拼音漏进来了**, 不会误判。

这就有了一个干净的、不用真值的判据:

    污染率 = 有拼音字母混进来的字段数 / 非空字段总数

    污染率降下去 = 这个法子有效

另外几个辅助指标:
    非空率     字段读出来的比例(读不出来就是 null)
    值长度     被截断的话会偏短
    一致率     两种跑法给出同一个值的比例

    python compare_runs.py --run 原图=目录A --run 擦完=目录B --run 切片=目录C
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SKIP_PREFIX = ("batch", "worker", "inference", "summary", "manifest",
               "sha256sums", "_tiles")

# 字符集里**确实存在**的带声调元音(服务器上量过: 24 个里有这 16 个)。
# 识别模型只可能吐出这些, 缺的那 8 个(ǎ ě ǔ ù ǖ ǘ ǚ ǜ)它吐不出来。
TONED = "āáàēéèīíǐìōóǒòūú"
TONED_RE = re.compile(f"[{TONED}]")

# 内部记账字段, 不参与统计
META_PREFIX = ("_", "input_image", "device")

# 结果里记原图路径的字段可能叫什么。批跑脚本把结果写成 <sha256>.json,
# 文件名是哈希和输入图无关, 所以**必须**从结果内部找来源。
SOURCE_KEYS = ("_source_image", "input_image", "source", "image", "image_path")


def load_dir(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.rglob("*.json")):
        if p.name.lower().startswith(SKIP_PREFIX):
            continue
        try:
            o = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if isinstance(o, dict):
            # ★★ 这里**不能**对 p.stem 再取一次 stem。
            #   我们的文件名里带点(r0.252_f0.69_s3_voucher_...),
            #   Path.stem 会把最后一段当扩展名剥掉:
            #       "r0.252_f0.69_s3_x"  --stem-->  "r0.252_f0"
            #   于是好几个不同的文件塌成同一个键, 后面的把前面的覆盖掉,
            #   统计出来的图数会**少**。实测 12 张被数成 8 张。
            #   _source_image 是带扩展名的原图名, 那个才需要剥一次。
            src = next((o[k] for k in SOURCE_KEYS if o.get(k)), None)
            key = Path(str(src)).stem if src else p.stem
            if key in out:
                # 同名冲突要报出来, 不能默默覆盖
                out[f"{key}#{len(out)}"] = o
            else:
                out[key] = o
    return out


def field_items(obj: dict):
    for k, v in obj.items():
        if k.startswith(META_PREFIX) or k.endswith("__conflict"):
            continue
        if v is None:
            yield k, None
            continue
        s = str(v).strip()
        yield k, (s if s else None)


def analyse(name: str, objs: dict[str, dict]) -> dict:
    total = filled = polluted = 0
    lens = []
    polluted_examples = []
    per_field_pollution: Counter = Counter()
    per_field_filled: Counter = Counter()      # ★ 逐字段非空张数
    per_field_seen: Counter = Counter()

    for stem, o in objs.items():
        for k, v in field_items(o):
            total += 1
            per_field_seen[k] += 1
            if v is None:
                continue
            filled += 1
            per_field_filled[k] += 1
            lens.append(len(v))
            if TONED_RE.search(v):
                polluted += 1
                per_field_pollution[k] += 1
                if len(polluted_examples) < 5:
                    polluted_examples.append(f"{stem} / {k} = {v[:60]}")

    return {"name": name, "images": len(objs), "fields": total,
            "filled": filled, "polluted": polluted, "lens": lens,
            "examples": polluted_examples, "per_field": per_field_pollution,
            "field_filled": per_field_filled, "field_seen": per_field_seen}


def print_per_field(runs: list[dict]) -> None:
    """★★★ 逐字段填充率对比 —— 这才是现在的主指标。

    实测拼音图上业务字段几乎全空(订单号 0/100), 而非拼音对照组订单号 41/100。
    也就是说**拼音把正文字段整个打没了**, 不是"读不全"。
    这种情况下"污染率"那个判据用不了 —— 字段都是空的, 没有文本可以被污染。
    要看的是**每个字段有多少张读出来了**, 以及有没有往对照组的水平靠。
    """
    fields = []
    for r in runs:
        for k in r["field_seen"]:
            if k not in fields:
                fields.append(k)
    if not fields:
        return
    print("=" * 66)
    print("逐字段非空率(这是主指标)")
    print("=" * 66)
    head = f"{'字段':<22}" + "".join(f"{r['name']:>11}" for r in runs)
    print(head)
    print("-" * len(head))
    # 按第一组的填充率排, 空的排最后
    def rate(r, k):
        n = r["field_seen"].get(k, 0)
        return (r["field_filled"].get(k, 0) / n) if n else 0.0
    for k in sorted(fields, key=lambda k: -rate(runs[0], k)):
        row = f"{k:<22}"
        for r in runs:
            n = r["field_seen"].get(k, 0)
            row += f"{rate(r, k):>10.0%} " if n else f"{'-':>11}"
        print(row)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="名字=目录, 可以给多次。例: --run 原图=D:\\a --run 切片=D:\\b")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        if "=" not in spec:
            print(f"格式应当是 名字=目录, 收到的是: {spec}")
            return
        name, path = spec.split("=", 1)
        d = Path(path)
        if not d.exists():
            print(f"目录不在: {d}")
            return
        objs = load_dir(d)
        if not objs:
            print(f"{d} 里没读到结果 JSON")
            return
        runs.append(analyse(name, objs))

    print("=" * 66)
    print(f"{'跑法':<12}{'图数':>6}{'字段':>7}{'非空':>7}{'非空率':>8}"
          f"{'混进拼音':>9}{'污染率':>8}")
    print("=" * 66)
    for r in runs:
        fr = r["filled"] / r["fields"] if r["fields"] else 0
        pr = r["polluted"] / r["filled"] if r["filled"] else 0
        print(f"{r['name']:<12}{r['images']:>6}{r['fields']:>7}{r['filled']:>7}"
              f"{fr:>7.1%}{r['polluted']:>9}{pr:>8.1%}")
    print()

    print_per_field(runs)

    base = runs[0]
    b_pr = base["polluted"] / base["filled"] if base["filled"] else 0
    for r in runs[1:]:
        pr = r["polluted"] / r["filled"] if r["filled"] else 0
        fr0 = base["filled"] / base["fields"] if base["fields"] else 0
        fr1 = r["filled"] / r["fields"] if r["fields"] else 0
        print(f"★ {r['name']} 对比 {base['name']}:")
        print(f"    污染率 {b_pr:.1%} -> {pr:.1%}"
              + ("   **改善**" if pr < b_pr else
                 "   没改善" if pr == b_pr else "   **变差了**"))
        print(f"    非空率 {fr0:.1%} -> {fr1:.1%}"
              + ("   **更多字段读出来了**" if fr1 > fr0 else
                 "   持平" if fr1 == fr0 else "   **读出来的反而少了**"))
    print()

    for r in runs:
        if r["examples"]:
            print(f"--- {r['name']} 里混进拼音的例子 ---")
            for e in r["examples"]:
                print(f"    {e}")
            if r["per_field"]:
                top = ", ".join(f"{k}({n})" for k, n in r["per_field"].most_common(5))
                print(f"    最常中招的字段: {top}")
            print()

    print("★★★ 现在的主指标是**逐字段非空率**, 不是污染率。")
    print("  实测: 非拼音对照组 订单号 41%, 拼音组 0% —— 拼音把正文字段整个打没了,")
    print("  不是'读不全'。字段都空的时候污染率永远是 0, 判不出东西来。")
    print("★★ 下面这组参考数**只对 pp-final-v4-r2 那个交付包成立**。")
    print("   2026-09-15 查清楚: 线上跑的是 PaddleOCR, 那个包是没被接受的候选。")
    print("   对着 receipt_inference.cli --ocr paddle 跑出来的结果, 不要跟这组数比。")
    print("   (非拼音对照 100 张, 那个包: transfer_status 94  payment_method 91")
    print("    amount 46  transfer_time 42  voucher_number 41  recipient_name 12)")
    print("★ 真正该看的是**同一批图、同一条管线、只差擦没擦**这两列的差。")
    print("  横着跟别的管线比没有意义。")


if __name__ == "__main__":
    main()
