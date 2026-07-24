r"""红队·状态栏拼接攻击(打生产设备模型 StatusBarDeviceClassifier)。两种攻击:

  A 拼接:安卓机身 + 贴上真 iOS 状态栏 → 尺寸不变、分辨率仍判安卓,CNN 被假栏骗成 iOS → 冲突 → 应被抓。
      测"分辨率↔状态栏交叉核查"对'贴状态栏'这种伪造是否也有效。
  B 组合:A 再缩放到 iPhone 精确分辨率 → 分辨率判 iOS、CNN 也判 iOS → 两信号一致 → 抓不到。
      这是"好假图让所有信号一致"的根本上限:任何图内信号都拦不住,只能靠外部硬证据(账户设备史/服务端遥测)。

标签来自构造(缩放前是安卓/iOS,分辨率规则判的),不需人工。自足:只要图库 + 权重。只读原图。

用法(带 torch 的 venv):
  D:\alipay-ai-data\alipay-ai-inference\.venv\Scripts\python.exe `
    <platform-classifier>\scripts\redteam_statusbar_composite_prod.py `
    --hx-src D:\Hx.AI.py\src `
    --image-root D:\download\TempFakeImages `
    --checkpoint D:\Hx.AI.py\checkpoints\statusbar_device_v1\best.pt `
    --device cuda --limit 300 --out-dir D:\download\redteam_composite
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_FRAC = 0.08  # 贴满 CNN 会看的顶部那条(与 device_statusbar._STRIP_FRACTION 一致)


def _upright(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _composite(android: Image.Image, ios: Image.Image) -> Image.Image:
    """安卓机身顶部贴上 iOS 的状态栏条。"""
    aw, ah = android.size
    bar_h = max(1, int(ah * _FRAC))
    ios_bar = ios.crop((0, 0, ios.width, max(1, int(ios.height * _FRAC)))).resize((aw, bar_h))
    out = android.copy()
    out.paste(ios_bar, (0, 0))
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="红队·状态栏拼接攻击 打生产设备模型")
    ap.add_argument("--hx-src", type=Path, required=True, help=r"Hx.AI.py 的 src,如 D:\Hx.AI.py\src")
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="设备模型权重 statusbar_device_v1/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=300, help="攻击多少张安卓机身")
    ap.add_argument("--scan-limit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, str(args.hx_src))
    try:
        from transfer_receipt_ai.device_statusbar import (
            IPHONE_RESOLUTIONS,
            StatusBarDeviceClassifier,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"导入失败(--hx-src 指向 Hx.AI.py\\src?用带 torch 的 venv?):{type(exc).__name__}: {exc}")
        raise

    clf = StatusBarDeviceClassifier(args.checkpoint, device=args.device)

    images = [p for p in args.image_root.rglob("*") if p.suffix.lower() in _EXTS]
    if not images:
        print(f"目录里没图片:{args.image_root}")
        return
    random.seed(args.seed)
    random.shuffle(images)

    andro: list[Path] = []
    ios: list[Path] = []
    need_ios = max(200, args.limit)
    scanned = 0
    for p in images:
        if scanned >= args.scan_limit or (len(andro) >= args.limit and len(ios) >= need_ios):
            break
        try:
            im = _upright(p)
        except Exception:
            continue
        scanned += 1
        out = clf.classify(np.asarray(im))
        if out.get("source") == "resolution":
            plat = out.get("platform")
            if plat == "android" and len(andro) < args.limit:
                andro.append(p)
            elif plat == "ios" and len(ios) < need_ios:
                ios.append(p)
    if not andro or not ios:
        print(f"扫描 {scanned} 张,安卓 {len(andro)} / iOS {len(ios)};两边都要有 source=resolution 的样本才能拼")
        return

    targets = sorted({(w, h) for (w, h) in IPHONE_RESOLUTIONS if h > w}) or sorted(IPHONE_RESOLUTIONS)
    if args.out_dir:
        (args.out_dir / "samples").mkdir(parents=True, exist_ok=True)

    n = caught_a = caught_b = succeeded_b = saved = 0
    for i, af in enumerate(andro):
        try:
            a = _upright(af)
            iimg = _upright(ios[i % len(ios)])
        except Exception:
            continue
        if a.width >= a.height:  # 只用竖屏,保持干净
            continue
        comp = _composite(a, iimg)                       # 攻击 A
        n += 1
        out_a = clf.classify(np.asarray(comp))
        if out_a.get("device_prior_conflict"):
            caught_a += 1
        tw, th = targets[i % len(targets)]
        comb = comp.resize((tw, th))                     # 攻击 B
        out_b = clf.classify(np.asarray(comb))
        if out_b.get("device_prior_conflict"):
            caught_b += 1
        elif out_b.get("platform") == "ios":
            succeeded_b += 1
        if args.out_dir and saved < args.save:
            comp.save(args.out_dir / "samples" / f"A_composite_{af.stem}.jpg", quality=90)
            comb.save(args.out_dir / "samples" / f"B_combined_{tw}x{th}_{af.stem}.jpg", quality=90)
            saved += 1

    if not n:
        print("没凑到竖屏安卓样本")
        return
    print(f"\n红队·状态栏拼接(生产模型):{n} 个「安卓机身 + 贴 iOS 状态栏」的伪造")
    print(f"  A 拼接(尺寸不变、分辨率仍判安卓):被交叉核查抓到 = {caught_a}/{n} = {caught_a / n:.1%}")
    print("     → 越高说明交叉核查对'贴状态栏'这种伪造也有效")
    print(f"  B 组合(拼接 + 再缩放到 iPhone 分辨率):抓到 = {caught_b}/{n} = {caught_b / n:.1%};攻击成功(判 iOS 漏过) = {succeeded_b}/{n} = {succeeded_b / n:.1%}")
    print("     → B 抓不到是预期的:分辨率和状态栏两个信号都被伪造成 iOS、彼此一致,图内没有信号能拦;")
    print("        这是根本上限,得靠外部硬证据(账户设备史/服务端遥测)。")
    if args.out_dir:
        print(f"  攻击样张另存 {saved} 组 -> {args.out_dir / 'samples'}")


if __name__ == "__main__":
    main()
