r"""红队·分辨率攻击探针(标签来自构造,不需人工)。

攻击:把**已知安卓**图缩放到 iPhone **精确分辨率**——这是最省事的伪造:不碰像素、只改尺寸,
就能命中 Tier-0 的 iPhone 分辨率表。量化两件事:
  (1) 光靠分辨率规则,这类攻击能骗过多少(Tier-0 失守率);
  (2) 若在 Tier-0 命中时**也跑一遍状态栏 CNN**(建议的修复),CNN 还能不能认出它其实是安卓
      (= 修复能挡回多少)。这正对着 fusion.merge_device 命中分辨率就给 0.99、根本不咨询 CNN 的洞。

只读原图;--out-dir 时另存少量攻击样张供查看。CNN 部分需 torch+模型(--model);不给就只报 (1)。

用法:
  python scripts\redteam_resolution_attack.py --results runs\pool_full\final_device_v2.jsonl --image-root D:\download\TempFakeImages --model training\runs\statusbar_v2\best.pt --limit 500 --out-dir runs\pool_full\redteam
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
from alipay_platform.metadata_seed import IPHONE_RESOLUTIONS  # noqa: E402


def _portrait_targets() -> list[tuple[int, int]]:
    t = sorted({(w, h) for (w, h) in IPHONE_RESOLUTIONS if h > w})
    return t or sorted(IPHONE_RESOLUTIONS)


def _cnn_scorer(model_path: Path, dev: str):
    """惰性加载 CNN;返回 f(pil_rgb)->'ios'/'android'。只有 --model 时才会走到这里。"""
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
    def score(pil: Image.Image) -> str:
        strip = crop_status_strip(np.asarray(pil.convert("RGB")))
        x = torch.from_numpy(normalize(strip_to_canvas(strip))).unsqueeze(0).to(device)
        p = torch.softmax(m(x), 1)[0, 1].item()
        return "ios" if p > 0.5 else "android"

    return score


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None, help="给了才做 CNN 交叉核查")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save", type=int, default=8, help="另存几张攻击样张供查看")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.results.read_text(encoding="utf-8").splitlines() if l.strip()]
    # 已知安卓 = 分辨率规则(短边面板宽)判的安卓,构造性真值,可靠
    andro = [r for r in rows if r.get("device") == "android" and r.get("source") == "resolution"]
    random.seed(args.seed)
    random.shuffle(andro)
    andro = andro[: args.limit]
    if not andro:
        print("没有 source=resolution 的安卓样本可攻击")
        return

    targets = _portrait_targets()
    scorer = _cnn_scorer(args.model, args.device) if args.model else None
    saved = 0
    if args.out_dir:
        (args.out_dir / "attacked").mkdir(parents=True, exist_ok=True)

    tier0_fooled = 0          # Tier-0 被骗成 iOS
    cnn_caught = 0            # CNN 仍判安卓(修复能挡)
    cnn_total = 0
    for i, r in enumerate(andro):
        tw, th = targets[i % len(targets)]
        try:
            im = ImageOps.exif_transpose(Image.open(args.image_root / r["file"])).convert("RGB")
        except Exception:
            continue
        attacked = im.resize((tw, th))                      # 攻击:缩放到 iPhone 精确分辨率
        if resolution_platform(tw, th) == "ios":            # Tier-0 现在判什么
            tier0_fooled += 1
        if scorer is not None:
            cnn_total += 1
            if scorer(attacked) == "android":               # CNN 交叉核查:仍认出是安卓
                cnn_caught += 1
        if args.out_dir and saved < args.save:
            attacked.save(args.out_dir / "attacked" / f"attack_{tw}x{th}_{Path(r['file']).stem}.jpg", quality=90)
            saved += 1

    n = len(andro)
    print(f"\n红队·分辨率攻击:{n} 张已知安卓图 → 缩放到 iPhone 精确分辨率")
    print(f"  Tier-0(分辨率规则)失守:判成 iOS 的 = {tier0_fooled}/{n} = {tier0_fooled/n:.1%}"
          f"  → 光靠分辨率规则,这类攻击几乎全过(且 fusion 给 0.99、不咨询 CNN)")
    if scorer is not None:
        print(f"  修复(命中 Tier-0 也跑 CNN 交叉核查):CNN 仍判安卓 = {cnn_caught}/{cnn_total} = {cnn_caught/max(1,cnn_total):.1%}"
              f"  → 这批本可被 device_prior_conflict 挡回")
    else:
        print("  (未给 --model,跳过 CNN 交叉核查;加上模型可量化'修复能挡回多少')")
    if args.out_dir:
        print(f"  攻击样张另存 {saved} 张 -> {args.out_dir/'attacked'}")


if __name__ == "__main__":
    main()
