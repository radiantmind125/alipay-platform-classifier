r"""找**逐字节完全相同**的重复提交 —— 不用模型, 零误报, 只读文件。

为什么这条值得单独做
--------------------
2026-08-31 在 `white/TempFakeImages` **138,614 张**上实扫:

  逐字节相同            **998 组 / 2,618 张 = 189/万**
  其中像真收据的        **907 组 / 2,139 张 = 154/万**
  多余提交(每组留一张)  **1,232 张 = 89/万**   <- **对外引用这个更稳**

逐张开图确认过一组: 是一张**真真正正的账单详情**(交易成功, 银行卡付款,
订单号与商家订单号俱全, **设备原生分辨率**, 画面没有任何篡改痕迹),
**在 4 分半钟内用 9 个不同凭证号提交了 9 次**。
(**故意不写金额/银行/卡号/时间** —— 本仓库是公开的, 真实进件的细节一律不入库。)

★★ **这一类 SSP 和 OCR 都抓不到, 因为图是百分之百真的。**
  SSP 问"这块像素是不是模型画的" -> 不是。OCR 读文字 -> 文字都是真的。
  **欺诈发生在"复用"这个维度上, 根本不在图里面。**

对照(同一批数据上的口径):
  SSP 自动拒 **2.7/万** | AI 假图到达 **6.5/万** | 负号几何离群 **8~11/万**
  -> **重复提交比 SSP 自动拒常见约 57 倍。**

怎么做到又快又准
----------------
**先按文件大小分组, 只对大小相同的才去算 sha256。** 实测 13.8 万张里
只有 5.7 万张需要哈希, 省掉六成磁盘读。逐字节相等**没有阈值也没有模棱两可**, 所以**零误报**。

必须分开报的两类
----------------
池子里**混着非收据**: 实扫最大的两组(132 张 / 41 张)分别是
**赌博 app 的筹码明细页**和**支付宝收银台的付款前扫码页** —— 那是垃圾提交, 不是收据复用。
所以脚本用线上同一个 `locate_amount_auto` 把两类分开报。

★ **这个数要不要紧, 取决于他们后台是不是已经查重了。** 报出去之前先问一句。

用法
----
  python training/dup_scan.py --input D:\download2\OtherImages --out D:\probe\dups.json
  python training/dup_scan.py --input D:\download2\BlueImages  --out D:\probe\dups_blue.json --no-classify

**只读**: 不动源目录, 只往 --out 写。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TS = re.compile(r"_(\d{14})")


def _sha(p: str, buf: int = 1 << 20) -> str | None:
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while True:
                b = f.read(buf)
                if not b:
                    break
                h.update(b)
    except Exception:
        return None
    return h.hexdigest()


def _looks_like_receipt(p: str) -> bool:
    """能不能定位到金额行 —— 用来把"收据复用"和"垃圾重复提交"分开。"""
    try:
        import numpy as np
        from PIL import Image
        from locate_blue import locate_amount_auto
        rgb = np.asarray(Image.open(p).convert("RGB"))
        loc, _pg = locate_amount_auto(rgb)
        return bool(loc and len(loc[4]) >= 4)
    except Exception:
        return False


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="找逐字节完全相同的重复提交(零误报)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="重复组明细写这里(json)")
    ap.add_argument("--no-classify", action="store_true",
                    help="跳过'像不像收据'的判定(快, 但收据类和垃圾类混在一起报)")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(args.input):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _EXTS:
                files.append(os.path.join(dp, fn))
    n_all = len(files)
    print(f"文件 {n_all:,} 个", flush=True)
    if not n_all:
        raise SystemExit("!! 一个文件都没有")

    # ---- 先按大小分组: 大小不同的**一定**不是逐字节相同, 直接跳过, 省掉大部分磁盘读 ----
    bysize: dict[int, list[str]] = collections.defaultdict(list)
    for p in files:
        try:
            bysize[os.path.getsize(p)].append(p)
        except OSError:
            pass
    cand = [v for v in bysize.values() if len(v) > 1]
    n_cand = sum(len(v) for v in cand)
    print(f"大小相同的组 {len(cand):,} 个, 需要哈希 {n_cand:,} 个 "
          f"(省掉 {100.0*(n_all-n_cand)/n_all:.0f}% 的磁盘读)", flush=True)

    dups: dict[str, list[str]] = collections.defaultdict(list)
    done = 0
    for grp in cand:
        for p in grp:
            h = _sha(p)
            if h:
                dups[h].append(p)
            done += 1
            if done % 10000 == 0:
                print(f"   已哈希 {done:,}/{n_cand:,}", flush=True)

    real = {h: v for h, v in dups.items() if len(v) > 1}
    tot = sum(len(v) for v in real.values())
    excess = tot - len(real)
    print(f"\n{'='*62}")
    print(f"逐字节完全相同: **{len(real):,} 组 / {tot:,} 张**")
    print(f"  占全池 {100.0*tot/n_all:.2f}%  = **{10000.0*tot/n_all:.0f}/万**")
    print(f"  多余提交(每组留一张): **{excess:,} 张 = {10000.0*excess/n_all:.0f}/万**  <- 对外用这个更稳")

    if not args.no_classify and real:
        print("\n分类中(判断每组的代表图像不像收据)...", flush=True)
        rec_g = rec_f = junk_g = 0
        for h, v in real.items():
            if _looks_like_receipt(sorted(v)[0]):
                rec_g += 1
                rec_f += len(v)
            else:
                junk_g += 1
        print(f"  **像真收据**: {rec_g:,} 组 / {rec_f:,} 张 = **{10000.0*rec_f/n_all:.0f}/万**")
        print(f"  非收据(收银台/游戏记录/其它垃圾): {junk_g:,} 组")
        print("  ★ 实扫见过: 最大的两组分别是**赌博 app 筹码明细**和**收银台付款前页**, 不是收据复用。")

    # ---- 时间跨度: 用来排除"同一次提交被系统重复记录"这个可能 ----
    cross = same = 0
    for v in real.values():
        ts = sorted(m.group(1) for m in (_TS.search(os.path.basename(x)) for x in v) if m)
        if len(ts) < 2:
            continue
        if ts[0][:8] != ts[-1][:8]:
            cross += 1
        else:
            same += 1
    if cross or same:
        print(f"\n组内**跨天**的: {cross:,} / {cross+same:,} ({100.0*cross/(cross+same):.0f}%)")
        print("  跨天 = **隔天又拿同一张图来用**, 不可能是一次提交被重复抓取。")

    print(f"\n提交次数最多的 {args.show} 组:")
    for h, v in sorted(real.items(), key=lambda kv: -len(kv[1]))[: args.show]:
        ts = sorted(m.group(1) for m in (_TS.search(os.path.basename(x)) for x in v) if m)
        span = f"{ts[0]} -> {ts[-1]}" if ts else ""
        print(f"  {len(v):>4} 次  {h[:16]}  {span}")
        for x in sorted(v)[:2]:
            print(f"         {os.path.basename(x)[:56]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({h: [os.path.basename(x) for x in v] for h, v in real.items()},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n明细 -> {args.out}")

    print("\n★ 报给上面之前先问一句: **后台是不是已经在查重了。**")
    print("  已经查了 -> 这些图本来就被拦下来了, 这个数不能当成'漏掉的欺诈'。")
    print("  没查     -> 这是一条零误报、零算力的新拦截口。")
    print("\n(下一步可做: **近似重复** —— 同一张收据重新截图/换压缩率, 逐字节就不同了。")
    print("  那需要感知哈希, 会引入误报, 得单独定阈值 —— 这一版**故意只做逐字节**, 保住零误报。)")


if __name__ == "__main__":
    main()
