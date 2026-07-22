r"""红队·分辨率攻击(直接打生产设备模型 StatusBarDeviceClassifier)。

最省事的伪造:把一张**已知安卓**的截图缩放到 iPhone **精确分辨率**——不碰像素、只改尺寸,
就能命中 Tier-0 的 iPhone 分辨率表。本脚本量化生产模型对这种攻击有多稳:
  (1) Tier-0(分辨率规则)失守率——缩放到 iPhone 分辨率后几乎必然被判成 iOS;
  (2) 生产模型的 CNN 交叉核查(device_prior_conflict)能挡回多少——CNN 看状态栏仍认出是安卓、判冲突;
  (3) 攻击成功率(分辨率+CNN 都被骗成 iOS)——这就是设备模型对"缩放伪造"的失守下界。

标签来自**构造**(缩放前分辨率规则判的安卓,可靠),不需人工采集。自足:只要图片目录 + 权重。
只读原图;--out-dir 时另存少量攻击样张供查看。必须用带 torch 的 venv 跑。

用法(服务器):
  D:\alipay-ai-data\alipay-ai-inference\.venv\Scripts\python.exe `
    <platform-classifier>\scripts\redteam_resolution_prod.py `
    --hx-src D:\Hx.AI.py\src `
    --image-root D:\download\TempFakeImages `
    --checkpoint D:\Hx.AI.py\checkpoints\statusbar_device_v1\best.pt `
    --device cuda --limit 500 --out-dir D:\download\redteam_prod
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="红队·分辨率攻击 打生产设备模型")
    ap.add_argument("--hx-src", type=Path, required=True, help=r"Hx.AI.py 的 src 目录,如 D:\Hx.AI.py\src")
    ap.add_argument("--image-root", type=Path, required=True, help="图片目录(会先分类挑出已知安卓)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="设备模型权重 statusbar_device_v1/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=500, help="攻击多少张已知安卓(默认500)")
    ap.add_argument("--scan-limit", type=int, default=4000, help="最多扫描多少张图去凑够安卓(默认4000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save", type=int, default=8, help="另存几张攻击样张")
    args = ap.parse_args()

    sys.path.insert(0, str(args.hx_src))
    try:
        from transfer_receipt_ai.device_statusbar import (
            IPHONE_RESOLUTIONS,
            StatusBarDeviceClassifier,
            resolution_platform,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"导入失败(--hx-src 指向 Hx.AI.py\\src 了吗?用带 torch 的 venv 了吗?):{type(exc).__name__}: {exc}")
        raise

    images = [p for p in args.image_root.rglob("*") if p.suffix.lower() in _EXTS]
    if not images:
        print(f"目录里没找到图片:{args.image_root}")
        return
    random.seed(args.seed)
    random.shuffle(images)

    targets = sorted({(w, h) for (w, h) in IPHONE_RESOLUTIONS if h > w}) or sorted(IPHONE_RESOLUTIONS)
    clf = StatusBarDeviceClassifier(args.checkpoint, device=args.device)
    if args.out_dir:
        (args.out_dir / "attacked").mkdir(parents=True, exist_ok=True)

    scanned = 0
    n = tier0_fooled = caught = succeeded = saved = 0
    for path in images:
        if n >= args.limit or scanned >= args.scan_limit:
            break
        scanned += 1
        try:
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            continue
        base = clf.classify(np.asarray(im))
        # 只攻击"缩放前分辨率规则判的安卓"(构造性真值)
        if not (base.get("source") == "resolution" and base.get("platform") == "android"):
            continue
        tw, th = targets[n % len(targets)]
        attacked = im.resize((tw, th))
        n += 1
        if resolution_platform(tw, th) == "ios":
            tier0_fooled += 1
        out = clf.classify(np.asarray(attacked))
        if out.get("device_prior_conflict"):
            caught += 1                       # CNN 交叉核查判冲突 → 抓到伪造
        elif out.get("platform") == "ios":
            succeeded += 1                    # 分辨率+CNN 都被骗成 iOS → 攻击成功
        if args.out_dir and saved < args.save:
            attacked.save(args.out_dir / "attacked" / f"attack_{tw}x{th}_{path.stem}.jpg", quality=90)
            saved += 1

    if not n:
        print(f"扫描了 {scanned} 张,没凑到 source=resolution 的安卓样本;加大 --scan-limit 或换图库")
        return
    print(f"\n红队·分辨率攻击(生产模型):扫描 {scanned} 张,取到 {n} 张已知安卓 → 缩放到 iPhone 精确分辨率")
    print(f"  Tier-0(分辨率规则)失守:{tier0_fooled}/{n} = {tier0_fooled / n:.1%}  (缩到 iPhone 分辨率,几乎必然被判 iOS)")
    print(f"  生产模型抓到(device_prior_conflict 冲突):{caught}/{n} = {caught / n:.1%}  ← CNN 交叉核查挡回")
    print(f"  攻击成功(分辨率+CNN 都判 iOS、漏过):{succeeded}/{n} = {succeeded / n:.1%}  ← 设备模型对'缩放伪造'的失守下界")
    if args.out_dir:
        print(f"  攻击样张另存 {saved} 张 -> {args.out_dir / 'attacked'}")


if __name__ == "__main__":
    main()
