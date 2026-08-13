r"""分块打分(tiled) —— 补"只改一小块"的结构盲区, **不重训**, 且比原版快很多。

问题(已实测坐实): SSP 打分是在**整张图**随机取 64 个 32x32 小块, 只留**纹理最丰富的一个**。
收据上最"丰富"的通常是底部彩色缩略图, 金额那小块几乎选不中 ->
只改金额的假图 300 张 **100% 漏检**(median 0.002 = 和真图一样), 整图重绘对照组 98.7% 拦下。

思路: 把图切成若干块, **每块单独选块打分**, 取最高分当整图分数。
块内的"最富纹理块"只能在这块里选 -> 改动所在的块就有机会被看见。复用现有 v6, 无需重训。

**速度**(原版慢的原因: 每块要做 64 次随机裁剪 + 逐个算纹理, 一张图上万次计算):
- 用积分图**一次性**算出块内所有 32x32 窗口的纹理和, 直接取最大 -> 免掉 64 次随机试。
- 所有分块**合成一个 batch 一次前向** -> GPU 跑一次而不是 18 次。
- 附带好处: **结果确定**(同图同分), 生产环境本来就该如此(原版随机取块 同图两次分数会不同)。

代价: 打分机会变多 -> **误杀可能上升**。按经理"绝对不能误杀"的红线, 必须在真图上重测。

产出 summary.csv 兼容 eval_summary.py; 同时存 max/top3/mean, 调阈值不用重跑。

用法:
  python training/predict_tiled.py --ssp-repo D:\SSP --model <v6.pth> --input D:\probe\localedit --output_dir D:\probe\localedit_tiled --device cuda
  # 真图误杀(量大可先抽样): 加 --limit 500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _tiles(w: int, h: int, cols: int, rows: int, overlap: float):
    """切成 cols x rows 块, 相邻块按 overlap 比例重叠(防改动区正好被切成两半)。"""
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tw - ox)); y0 = max(0, int(r * th - oy))
            x1 = min(w, int((c + 1) * tw + ox)); y1 = min(h, int((r + 1) * th + oy))
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                out.append((x0, y0, x1, y1))
    return out


def _texture_map(a: np.ndarray) -> np.ndarray:
    """每像素的纹理强度(与官方 compute() 同口径: 水平+垂直+两条对角的绝对差之和)。"""
    x = a.astype(np.int32)
    h, w = x.shape[:2]
    t = np.zeros((h, w), np.float32)
    t[:, :-1] += np.abs(x[:, :-1] - x[:, 1:]).sum(2)
    t[:-1, :] += np.abs(x[:-1, :] - x[1:, :]).sum(2)
    t[:-1, :-1] += np.abs(x[:-1, :-1] - x[1:, 1:]).sum(2)
    t[:-1, :-1] += np.abs(x[1:, :-1] - x[:-1, 1:]).sum(2)
    return t


def _is_blue_page(a: np.ndarray) -> bool:
    """蓝底转账页判定。

    为什么要判: 蓝图的金额是**白字蓝底**, 而 locate_amount 找的是"浅底深字" —— 它看不见真金额,
    却会误锁到页面里唯一的深字浅底元素: **红包促销卡**(如"到店支付红包 ¥0.6" 那种彩色卡片)。
    那块是花哨的合成图形, 打分自然偏高 -> 系统性误杀。实测最高分真图里就有这种。
    所以蓝图直接不出局部信号(它需要自己的金额定位器, 见蓝白分开的计划)。
    """
    top = a[: max(1, a.shape[0] // 3)]
    r, g, b = (float(top[:, :, i].mean()) for i in range(3))
    return b > r + 25 and b > g + 15


def _richest_patch(a: np.ndarray, ps: int) -> np.ndarray:
    """块内纹理最丰富的 ps x ps 窗口(积分图一次算完所有窗口, 免 64 次随机裁剪; 确定性)。"""
    h, w = a.shape[:2]
    if h < ps or w < ps:                      # 太小就放大到至少 ps(与官方 min_len<patch_size 时 resize 同思路)
        s = max(ps / max(1, h), ps / max(1, w))
        a = cv2.resize(a, (max(ps, int(w * s + 0.5)), max(ps, int(h * s + 0.5))), interpolation=cv2.INTER_LINEAR)
        h, w = a.shape[:2]
    ii = cv2.integral(_texture_map(a))        # (h+1, w+1)
    win = ii[ps:, ps:] - ii[:-ps, ps:] - ii[ps:, :-ps] + ii[:-ps, :-ps]
    y, x = np.unravel_index(int(np.argmax(win)), win.shape)
    return a[y:y + ps, x:x + ps]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="分块打分(补局部改动盲区, 不重训, 快且确定)")
    ap.add_argument("--ssp-repo", required=True, help="SSP 预测仓库路径(取 networks/), 如 D:\\SSP")
    ap.add_argument("--model", required=True, help="Net_epoch_best.pth")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=6, help="截图是竖长条, 行多列少更合理")
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--roi-amount", action="store_true",
                    help="**只看金额那一块**(用 locate_amount 定位, 外扩后切少量小块)。"
                         "篡改就发生在这里, 只看这儿 = 信号最强 + 误杀面最小。定位失败则退回 --roi-top。"
                         "注: 金额定位是按白底账单详情调的, 蓝图页可能定位不到。")
    ap.add_argument("--blue-locator", action="store_true",
                    help="蓝图改用**蓝图专用金额定位器**(白字蓝底)来出信号, 而不是跳过。"
                         "要求模型见过蓝图金额区(白字蓝底和深字白底长得很不一样, 多半要把蓝图裁块加进训练)")
    ap.add_argument("--skip-blue", action="store_true",
                    help="蓝底转账页不出局部信号(默认关, 建议开)。蓝图金额是白字蓝底, locate_amount 看不见, "
                         "会误锁红包促销卡 -> 那是彩色合成图形, 打分偏高 = 系统性误杀源")
    ap.add_argument("--amount-pad", type=float, default=1.0,
                    help="金额框上下左右各外扩几倍框高(1.0=各扩一个框高, 给点上下文)")
    ap.add_argument("--roi-top", type=float, default=1.0,
                    help="只看图像上部的这个比例(如 0.6 = 上面五分之三)。"
                         "下半部分是广告条/优惠券缩略图/图标, 纹理最杂 —— 真图在那儿最容易被误判成AI(误杀主要来源), "
                         "而金额等关键字段都在上部。经理给 OCR 定的'裁掉下面减少误差'同一个道理。")
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--agg", default="max", choices=["max", "top3", "mean"])
    ap.add_argument("--limit", type=int, default=0, help="只跑**排序后的前 N 张**(先看趋势用; 0=全跑)")
    ap.add_argument("--sample", type=int, default=0,
                    help="**随机**抽 N 张(0=不抽)。文件名是 s3_voucher_<雪花ID>_<时间戳>, "
                         "排序≈按时间排序, 所以 `--limit` 抽到的永远是**最早那一批** —— "
                         "拿它当标定池会带时间偏置。**标定池必须用 --sample。**")
    ap.add_argument("--sample-seed", type=int, default=0, help="--sample 的随机种子, 便于复现")
    ap.add_argument("--reject_threshold", type=float, default=0.90)
    ap.add_argument("--review_threshold", type=float, default=0.60)
    ap.add_argument("--ai_label", type=int, default=0, choices=[0, 1])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _locate = _locate_blue = None
    if args.roi_amount:                 # 复用 engine_b_tamper 的金额定位(同目录, 直接 import)
        try:
            from engine_b_tamper import locate_amount as _locate
        except Exception as exc:  # noqa: BLE001
            print(f"载入金额定位失败({exc}), --roi-amount 将退回 --roi-top", flush=True)
        if args.blue_locator:
            try:
                from locate_blue import locate_amount_blue as _locate_blue
            except Exception as exc:  # noqa: BLE001
                print(f"载入蓝图定位失败({exc}), 蓝图将按 --skip-blue 处理", flush=True)

    sys.path.insert(0, str(Path(args.ssp_repo)))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageFile
    from networks.resnet import resnet50
    from networks.srm_conv import SRMConv2d_simple
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class SSPNet(nn.Module):          # 与官方 networks/ssp.py 同构(pretrained=False 免下载)
        def __init__(self):
            super().__init__()
            self.srm = SRMConv2d_simple()
            self.disc = resnet50(pretrained=False)
            self.disc.fc = nn.Linear(2048, 1)

        def forward(self, x):
            x = F.interpolate(x, (256, 256), mode="bilinear")
            return self.disc(self.srm(x))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("没有 CUDA, 改用 CPU"); args.device = "cpu"
    device = torch.device(args.device)

    sd = torch.load(args.model, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model = SSPNet(); model.load_state_dict(sd, strict=False); model.to(device).eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    root = Path(args.input)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in _EXTS) if root.is_dir() else [root]
    if args.sample > 0 and len(imgs) > args.sample:
        import random
        imgs = sorted(random.Random(args.sample_seed).sample(imgs, args.sample))
        print(f"随机抽样 {len(imgs)} 张(种子 {args.sample_seed})", flush=True)
    elif args.limit > 0:
        imgs = imgs[:args.limit]
    if not imgs:
        print("没找到图片"); return
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"图片 {len(imgs)} 张 | 切块 {args.cols}x{args.rows} 重叠 {args.overlap} | 只看上部 {args.roi_top} | "
          f"聚合 {args.agg} | 设备 {device}", flush=True)

    rows_out = []
    n_located = 0
    for i, ip in enumerate(imgs, 1):
        try:
            arr = np.asarray(Image.open(ip).convert("RGB"))
            cols, rows = args.cols, args.rows
            located = False
            if args.roi_amount:                   # 只看金额那块: 篡改就在这儿, 信号最强误杀面最小
                if _is_blue_page(arr):
                    # 蓝图: 有专用定位器就用它; 否则不出信号
                    # (白图定位器在蓝图上会误锁红包促销卡 -> 系统性误杀, 绝不能用)
                    loc = _locate_blue(arr) if _locate_blue else (
                        None if args.skip_blue else (_locate(arr) if _locate else None))
                else:
                    loc = _locate(arr) if _locate else None
                if loc:
                    bx0, by0, bx1, by1 = loc[0], loc[1], loc[2], loc[3]
                    pad = int(max(8, (by1 - by0) * args.amount_pad))
                    H, W = arr.shape[:2]
                    arr = arr[max(0, by0 - pad):min(H, by1 + pad), max(0, bx0 - pad):min(W, bx1 + pad)]
                    cols, rows = 2, 2             # 区域已经很小, 切少量块就够
                    located = True
            if not located and 0 < args.roi_top < 1.0:   # 退回: 只保留上部(甩掉下面广告/券区=误杀主源)
                arr = arr[:max(32, int(arr.shape[0] * args.roi_top))]
            h, w = arr.shape[:2]
            patches = [_richest_patch(arr[y0:y1, x0:x1], args.patch_size)
                       for (x0, y0, x1, y1) in _tiles(w, h, cols, rows, args.overlap)]
            batch = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).float().div_(255.0).to(device)
            batch = (batch - MEAN) / STD
            with torch.no_grad():                      # 所有分块一次前向
                s = torch.sigmoid(model(batch).view(-1))
                s = (1.0 - s) if args.ai_label == 0 else s
            ts = sorted((float(v) for v in s), reverse=True)
            s_max, s_top3, s_mean = ts[0], sum(ts[:3]) / min(3, len(ts)), sum(ts) / len(ts)
            final = {"max": s_max, "top3": s_top3, "mean": s_mean}[args.agg]
            dec = ("Reject" if final >= args.reject_threshold else
                   "ManualReview" if final >= args.review_threshold else "PassToBusinessVerify")
            rows_out.append({"image": str(ip), "image_name": ip.name,
                             "final_ai_score": round(final, 6),
                             "tile_max": round(s_max, 6), "tile_top3": round(s_top3, 6),
                             "tile_mean": round(s_mean, 6), "n_tiles": len(ts),
                             "decision": dec, "roi_amount_located": int(located),
                             "scores_json": json.dumps({"tiled": round(final, 6)})})
            n_located += int(located)
            if i % 200 == 0:
                print(f"  已算 {i}/{len(imgs)}...", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  失败 {ip.name}: {str(exc)[:150]}", flush=True)

    sp = out_dir / "summary.csv"
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=["image", "image_name", "final_ai_score", "tile_max",
                                             "tile_top3", "tile_mean", "n_tiles", "decision",
                                             "roi_amount_located", "scores_json"])
        wcsv.writeheader(); wcsv.writerows(rows_out)
    print(f"完成 {len(rows_out)} 张 -> {sp}")
    if args.roi_amount:
        print(f"其中 {n_located}/{len(rows_out)} 张成功定位到金额区(其余按 --roi-top 处理)。"
              f"定位率低说明这批不是白底账单详情页。")
    print("下一步: python training/eval_summary.py <上面的 summary.csv> --kind fake|genuine")
    print("提示: CSV 存了 tile_max/tile_top3/tile_mean, 换聚合方式不用重跑。")


if __name__ == "__main__":
    main()
