r"""红队·状态栏拼接攻击(标签来自构造,不需人工)。

两种攻击,量化管线拦截率:
  (A) 拼接:安卓机身 + 贴上真 iOS 状态栏 → 分辨率仍判安卓、CNN 被假栏骗成 iOS → 矛盾 → **应被抓**。
      测"堵洞的交叉核查"对状态栏伪造也有效。
  (B) 组合(拼接 + 再缩放到 iPhone 精确分辨率)→ 分辨率判 iOS、CNN 也判 iOS → **两信号一致 → 抓不到**。
      这就是"好假图让所有信号一致"的根本上限:任何图像内部信号都拦不住,只能靠外部硬证据
      (账户设备历史/服务端遥测)或勾等其它取证。

需 torch + 模型(--model)量化 CNN 与拦截率;--out-dir 另存少量攻击样张供查看/给经理看。

用法:
  python scripts\redteam_statusbar_composite.py --results runs\pool_full\final_device_v2.jsonl --image-root D:\download\TempFakeImages --model training\runs\statusbar_v2\best.pt --limit 300 --out-dir runs\pool_full\redteam_composite
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alipay_platform.bootstrap import resolution_platform  # noqa: E402
from alipay_platform.fusion import device_prior_conflict  # noqa: E402
from alipay_platform.metadata_seed import IPHONE_RESOLUTIONS  # noqa: E402

FRAC = 0.08  # 与 CNN 裁条比例一致:贴满 CNN 会看的那条


def _upright(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _composite(android: Image.Image, ios: Image.Image) -> Image.Image:
    """安卓机身顶部贴上 iOS 的状态栏条。"""
    aw, ah = android.size
    bar_h = max(1, int(ah * FRAC))
    ios_bar = ios.crop((0, 0, ios.width, max(1, int(ios.height * FRAC)))).resize((aw, bar_h))
    out = android.copy()
    out.paste(ios_bar, (0, 0))
    return out


def _cnn_scorer(model_path: Path, dev: str):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from preprocess import crop_status_strip, strip_to_canvas, normalize  # noqa
    from model import build_model  # noqa

    device = dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu"
    payload = torch.load(model_path, map_location=device, weights_only=False)
    m = build_model(2, width=float(payload.get("width", 1.0))).to(device)
    m.load_state_dict(payload["model_state"])
    m.eval()

    @torch.no_grad()
    def score(pil: Image.Image) -> tuple[str, float]:
        strip = crop_status_strip(np.asarray(pil))
        x = torch.from_numpy(normalize(strip_to_canvas(strip))).unsqueeze(0).to(device)
        p = torch.softmax(m(x), 1)[0, 1].item()
        return ("ios" if p > 0.5 else "android", max(p, 1 - p))

    return score


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.results.read_text(encoding="utf-8").splitlines() if l.strip()]

    def pool(dev: str) -> list[str]:
        fs = [r["file"] for r in rows if r.get("device") == dev and r.get("source") == "resolution"]
        random.seed(args.seed if dev == "android" else args.seed + 1)
        random.shuffle(fs)
        return fs

    andro, ios = pool("android"), pool("ios")
    if not andro or not ios:
        print("需要同时有 source=resolution 的安卓与 iOS 样本")
        return
    andro = andro[: args.limit]

    scorer = _cnn_scorer(args.model, args.device) if args.model else None
    targets = sorted({(w, h) for (w, h) in IPHONE_RESOLUTIONS if h > w}) or sorted(IPHONE_RESOLUTIONS)
    if args.out_dir:
        (args.out_dir / "samples").mkdir(parents=True, exist_ok=True)

    n = caughtA = caughtB = saved = 0
    for i, af in enumerate(andro):
        try:
            a = _upright(args.image_root / af)
            iimg = _upright(args.image_root / ios[i % len(ios)])
            if a.width >= a.height:  # 只用竖屏,保持干净
                continue
        except Exception:
            continue
        comp = _composite(a, iimg)                                   # 攻击 A
        n += 1
        # A:拼接(尺寸不变=安卓)。分辨率判安卓;CNN 看假栏
        resA = resolution_platform(*comp.size)
        if scorer is not None:
            cdev, cconf = scorer(comp)
            if device_prior_conflict(resA, cdev, cconf):             # 矛盾=抓到
                caughtA += 1
        # B:组合(再缩放到 iPhone 分辨率)
        tw, th = targets[i % len(targets)]
        comb = comp.resize((tw, th))
        resB = resolution_platform(tw, th)
        if scorer is not None:
            cdev2, cconf2 = scorer(comb)
            if device_prior_conflict(resB, cdev2, cconf2):
                caughtB += 1
        if args.out_dir and saved < args.save:
            comp.save(args.out_dir / "samples" / f"A_composite_{Path(af).stem}.jpg", quality=90)
            comb.save(args.out_dir / "samples" / f"B_combined_{tw}x{th}_{Path(af).stem}.jpg", quality=90)
            saved += 1

    print(f"\n红队·状态栏拼接:{n} 个 安卓机身+iOS状态栏 的伪造")
    if scorer is not None:
        print(f"  A 拼接(尺寸不变):被交叉核查抓到(分辨率↔状态栏矛盾) = {caughtA}/{n} = {caughtA/max(1,n):.1%}"
              f"  → 交叉核查对状态栏伪造也有效")
        print(f"  B 组合(拼接+缩放到iPhone分辨率):抓到 = {caughtB}/{n} = {caughtB/max(1,n):.1%}"
              f"  → 越低越说明'两信号一致'的假图拦不住(根本上限,需外部硬证据)")
    else:
        print("  (未给 --model,只构造样张;量化拦截率请加 --model)")
    if args.out_dir:
        print(f"  攻击样张另存 {saved} 组 -> {args.out_dir/'samples'}")


if __name__ == "__main__":
    main()
