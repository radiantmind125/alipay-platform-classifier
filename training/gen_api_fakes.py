r"""用 DMXAPI(聚合接口)批量造国内大模型的假收据 —— 豆包 Seedream / 千问 / 可灵 等。

为什么要它: 之前豆包是手动一张张在 App 造的(100 张就很耗时)。有了接口可以批量造,
既能测覆盖, 也能造训练量级的数据把置信度训上去(同当初 ostris 76%->95%、真 Qwen 70%漏->0漏)。

要点:
- **API key 从环境变量读, 绝不写进代码/仓库**(仓库是公开的)。跑之前先 set DMX_KEY。
- `--watermark` 默认关: 接口能直接出不带"AI生成"水印的图 -> 省掉去水印那一步。
- 图片以 base64 内联上传, 不用先传到图床。
- 断点续跑: 已存在的输出文件跳过, 中断了重跑不会重复花钱。
- 每张都写 manifest.csv(记模型/提示词), 便于后面按生成器切分训练/测试。

用法(先设 key; **PowerShell 和 cmd 语法不一样, 别用错**):
  PowerShell:  $env:DMX_KEY = "sk-xxxx"      (提示符 PS D:\...> 用这个)
  cmd:         set DMX_KEY=sk-xxxx
  设完在同一个窗口跑下面的命令。
  python training/gen_api_fakes.py --src-root D:\download\TempFakeImages --out D:\api_fakes\seedream45 --n 50 --model doubao-seedream-4-5-251128
  python training/gen_api_fakes.py --src-root D:\download\TempFakeImages --out D:\api_fakes\qwenedit --n 50 --model qwen-image-edit
  # 局部改金额(测结构盲区, 用真模型而不是我本地模拟):
  python training/gen_api_fakes.py --src-root D:\download2\TempFakeImages --out D:\api_fakes\localamt --n 50 --model qwen-image-edit --prompt-mode local-amount
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

from api_image import generate_image

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_OPT = (33434, 33437, 34855, 37386)   # 光学拍摄参数; 有=相机照片, 不当真图源

# 提示词: redraw=整张重画; local-amount=让模型改金额
# 实测重要发现: Seedream 收到 local-amount 也是**整张重新生成**(只是金额变了) —— 两种模式出图的
# 幻觉细节完全一致, 说明它做不了"只动金额那几个像素"的外科式修改。
# 含义: 骗子用消费级 AI App"把金额改掉"拿到的其实是整张重绘图 -> 正是 v6 已经 0 漏检的那一类。
# 真正的"只改一小块"(99%真像素)要用带掩码的局部重绘 -> 见 gen_local_ai_edit.py(本地模拟那条路)。
_PROMPTS = {
    "redraw": "照着这张图 重新画一张一模一样的图片 所有文字 数字和排版都保持不变",
    "local-amount": "把这张图里的金额数字改成 8888.88 其他所有内容 排版 颜色 字体都完全不要动",
}


def _is_clean_screenshot(im: Image.Image) -> bool:
    w, h = im.size
    short = min(w, h)
    if not short:
        return False
    aspect = max(w, h) / short
    try:
        ifd = im.getexif().get_ifd(0x8769)
    except Exception:
        ifd = {}
    return (not any(t in ifd for t in _OPT)) and short <= 1500 and 1.6 <= aspect <= 2.6


def _collect(root: Path, n: int, seed: int) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in _EXTS]
    random.Random(seed).shuffle(files)
    out: list[Path] = []
    for p in files:
        if len(out) >= n:
            break
        try:
            with Image.open(p) as im:
                if _is_clean_screenshot(im):
                    out.append(p)
        except Exception:
            continue
    return out


def _post(url: str, key: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:      # 把接口真正的错误内容带出来, 否则只看到 400 没法排查
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="用 DMXAPI 批量造国内大模型假收据")
    ap.add_argument("--src-root", type=Path, required=True, help="真收据源目录")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="doubao-seedream-4-5-251128",
                    help="实测能图生图的: doubao-seedream-5-0-pro-260628 / 4-5-251128 / 4-0-250828 / 5.0-lite(豆包即梦); "
                         "wan2.7-image / wan2.7-image-pro / wan2.6-image(阿里万相)。"
                         "qwen-image-edit(千问, 走 DashScope 形状的 body, api_image 已处理); "
                         "可灵只有视频接口; 即梦/SeedEdit 这平台没渠道")
    ap.add_argument("--prompt-mode", default="redraw", choices=list(_PROMPTS),
                    help="redraw=整张重画; local-amount=只改金额(测局部编辑盲区)")
    ap.add_argument("--prompt", default="", help="自定义提示词(盖过 --prompt-mode)")
    ap.add_argument("--base", default="https://www.dmxapi.cn", help="DMXAPI 域名(.cn 可用, .com 拒 key)")
    ap.add_argument("--size", default="2K")
    ap.add_argument("--watermark", action="store_true", help="加上则出图带AI水印(默认不带, 省去去水印)")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--sleep", type=float, default=1.0, help="每张之间歇一下, 别把接口打崩")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("DMX_KEY", "").strip()
    if not key:
        raise SystemExit(
            "没读到环境变量 DMX_KEY(key 不写进代码/仓库, 只放环境变量)。按你的终端选一种:\n"
            '  PowerShell(提示符是 PS D:\\...> ):  $env:DMX_KEY = "sk-xxxx"\n'
            "  cmd(提示符是 D:\\> ):              set DMX_KEY=sk-xxxx\n"
            "注意: PowerShell 里 `set DMX_KEY=...` 是无效的(那是 cmd 语法, 在 PS 里只建了个普通变量)。\n"
            "设完在**同一个窗口**里跑本脚本; 换窗口要重设。可用 `echo $env:DMX_KEY` 确认。")
    prompt = args.prompt or _PROMPTS[args.prompt_mode]

    args.out.mkdir(parents=True, exist_ok=True)
    srcs = _collect(args.src_root, args.n, args.seed)
    if not srcs:
        print("没采到真图源"); return
    print(f"真图源 {len(srcs)} 张 | 模型 {args.model} | 模式 {args.prompt_mode} | 水印 {args.watermark}", flush=True)
    print(f"提示词: {prompt}", flush=True)

    mp = args.out / "manifest.csv"; new = not mp.exists()
    mf = open(mp, "a", newline="", encoding="utf-8-sig"); mw = csv.writer(mf)
    if new:
        mw.writerow(["file", "model", "prompt_mode", "src"])

    made = failed = skipped = 0
    for i, sp in enumerate(srcs):
        fn = f"api_{args.prompt_mode}_{args.model.replace('.', '-')}_{i:05d}.jpg"
        fp = args.out / fn
        if fp.exists():           # 断点续跑: 已有的跳过, 不重复花钱
            skipped += 1
            continue
        try:
            # 各厂商请求格式不同(豆包 images/generations vs 万相 responses), 统一走 api_image
            fp.write_bytes(generate_image(args.model, prompt, sp.read_bytes(), key,
                                          base=args.base, size=args.size,
                                          timeout=args.timeout, watermark=bool(args.watermark)))
            mw.writerow([fn, args.model, args.prompt_mode, sp.name]); mf.flush()
            made += 1
            if made % 10 == 0:
                print(f"  已造 {made} 张...", flush=True)
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"  失败 {sp.name}: {str(exc)[:200]}", flush=True)
            if failed >= 10 and made == 0:
                print("连续失败太多, 先停下来检查(key/额度/模型名/网络)。"); break

    mf.close()
    print(f"完成: 造了 {made} 张 -> {args.out}(跳过已存在 {skipped}, 失败 {failed})")
    print("下一步: 直接用 predict_all_models.py + eval_summary.py 测(不带水印, 不用再去水印)。")


if __name__ == "__main__":
    main()
