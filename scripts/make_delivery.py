"""把交付内容归到一个文件夹:结果 jsonl + 标注样例 + 说明.txt(含数量统计)。

给经理/运营看的那个文件夹,一条命令生成。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

CN = {"ios": "苹果", "android": "安卓", "uncertain": "不确定", "unknown": "未知"}


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True, help="final_device*.jsonl")
    ap.add_argument("--annotated", type=Path, default=None, help="标注图目录(可选)")
    ap.add_argument("--output", type=Path, required=True, help="交付文件夹")
    args = ap.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.results.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = Counter(r.get("device") for r in rows)
    src = Counter(r.get("source") for r in rows)
    total = len(rows)

    shutil.copy(args.results, args.output / args.results.name)

    n_ann = 0
    if args.annotated and args.annotated.exists():
        dst = args.output / "标注样例"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(args.annotated, dst)
        n_ann = sum(1 for _ in dst.glob("*") if _.is_file())

    lines = [
        "支付宝转账成功截图 —— 设备识别结果",
        "",
        f"总张数：{total}",
        f"    苹果：{dev.get('ios', 0)}",
        f"    安卓：{dev.get('android', 0)}",
        f"    不确定：{dev.get('uncertain', 0)}",
        "",
        f"判定来源：分辨率直接判 {src.get('resolution', 0)} 张，状态栏模型判 {src.get('cnn', 0)} 张",
        "",
        "文件说明：",
        f"    {args.results.name} —— 全量结果，每张图一行：file(文件) / device(设备) / source(判定来源) / confidence(置信度)",
        f"    标注样例/ —— 随机抽的 {n_ann} 张，红字标了设备类型，给运营看",
        "",
        "device 取值：ios=苹果，android=安卓，uncertain=不确定",
    ]
    (args.output / "说明.txt").write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"交付文件夹已生成 -> {args.output}（全量结果 + {n_ann} 张标注样例 + 说明.txt）")
    print(f"  分布：苹果{dev.get('ios',0)} / 安卓{dev.get('android',0)} / 不确定{dev.get('uncertain',0)}")


if __name__ == "__main__":
    main()
