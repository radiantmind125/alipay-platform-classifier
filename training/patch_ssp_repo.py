r"""把官方 SSP 仓库改成能在"支付宝单类 + 现代环境"下重训(审计发现的三个坑,一键修,带备份)。

改动(幂等,先存 .bak):
1. options.py: choices 默认 [0,0,0,0,1,0,0,0] -> [2,2,2,2,1,2,2,2]
   —— 否则 get_val_loader 会去验证全部 8 个 GenImage 目录(不存在)-> FileNotFoundError 开训即崩。
   改后只在 sdv4 槽 训(==1)+验(0/1),其余 7 个 ==2 跳过。数据放 <root>/imagenet_ai_0419_sdv4/。
2. tdataloader.py: `from scipy.ndimage.filters import gaussian_filter`
   -> `from scipy.ndimage import gaussian_filter`(旧路径 scipy>=1.14 已删,改后不必钉 scipy==1.10.1)。

不动 train_val.py 的 .cuda()(服务器有 GPU,原样即可)。

用法(服务器,指向 SSP 源码目录):
  python training/patch_ssp_repo.py --repo D:\SSP-AI-Generated-Image-Detection-main
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 每条:(文件, [可能的原串候选…], 目标串)。命中任一候选即替换;都没有且未达标才告警。
_PATCHES = [
    ("options.py",
     ["default=[0, 0, 0, 0, 1, 0, 0, 0]", "default=[0,0,0,0,1,0,0,0]"],
     "default=[2, 2, 2, 2, 1, 2, 2, 2]"),
    ("utils/tdataloader.py",
     ["from scipy.ndimage.filters import gaussian_filter"],
     "from scipy.ndimage import gaussian_filter"),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="给官方 SSP 仓库打支付宝重训补丁")
    ap.add_argument("--repo", type=Path, required=True, help="SSP 源码目录")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    applied = 0
    skipped = 0
    for rel, olds, new in _PATCHES:
        f = args.repo / rel
        if not f.exists():
            print(f"跳过(文件不存在): {rel}")
            continue
        txt = f.read_text(encoding="utf-8")
        if new in txt:
            print(f"已是目标状态: {rel} :: {new}")
            skipped += 1
            continue
        hit = next((o for o in olds if o in txt), None)
        if hit is None:
            print(f"!! 没找到待替换串,请人工核对: {rel} :: {olds[0]}")
            continue
        if args.dry_run:
            print(f"[dry-run] 将改 {rel}: {hit} -> {new}")
            continue
        if not (f.with_suffix(f.suffix + ".bak")).exists():
            shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))
        f.write_text(txt.replace(hit, new), encoding="utf-8")
        print(f"已改 {rel}: {hit} -> {new}(原文件备份 .bak)")
        applied += 1

    print(f"\n完成:改 {applied} 处,已达标 {skipped} 处。")
    print("提醒:数据放 <root>/imagenet_ai_0419_sdv4/{train,val}/{nature,ai};val/ai 必须非空;"
          "resnet50(pretrained=True) 首次训练会下 ImageNet 权重(离线要预放 ~/.cache/torch/hub/checkpoints/)。")


if __name__ == "__main__":
    main()
