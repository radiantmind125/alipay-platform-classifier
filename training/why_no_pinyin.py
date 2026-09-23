"""查一张图上的拼音**卡在哪一道**没被检测出来。

起因
----
蓝图拼音检测只有 82.49%(白图 96.31%)。看图发现漏的不只是大标题:

```
转账成功 上面的 zhuǎnzhàngchéng gōng    没检测到
回首页   上面的 huí shǒu yè             没检测到
收款方   上面的 shōu kuǎn fāng          没检测到   <- 这个字号很正常
付款方式 上面的 fù kuǎn fāng shì        没检测到   <- 这个也是
```

★★★★★ `收款方` `付款方式` **字号根本不大**, 所以**不是大小判据的锅** ——
   local_ratio 那一版修不好它们, 实测也确实没修好。

   它们的共同点是**蓝色渐变背景上的白字**, 而底下白卡片里的拼音**检测到了**。
   所以嫌疑在**掩膜**或者**连通块的筛选**, 不在大小判据。

★★ 但这仍然是猜。这一程已经猜错四次(蓝图摊薄、标签噪声、边界太窄、声调),
   所以不猜, **把每一道闸的存活数打出来**, 看东西是在哪一步没的。

这条链上的闸
------------
```
1. text_mask          有没有进掩膜        <- 进不了掩膜, 后面全白搭
2. 面积 >= 8          太小的当噪声扔掉
3. 高 < 0.05*H        太高的当装饰扔掉
4. 宽 < 0.4*W         太宽的当分隔线扔掉
5. 小块 <= 0.55*big_h 够不够"小"
6. 底下有大块压住      配对
7. 同高度凑成一排      MIN_RUN, 防止把汉字的散笔画当拼音
```

按**横带**分段统计, 这样能看出是"整页都漏"还是"只有蓝色那一带漏"。

用法
----
    python why_no_pinyin.py --img D:\\download2\\pinyin_hits_blue\\xxx.jpg ^
        --save D:\\alipay-ai-data\\mask_look
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erase_pinyin import (ANNOT_GAP_RATIO, MIN_RUN,  # noqa: E402
                          annotation_labels, local_background, text_mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    ap.add_argument("--bands", type=int, default=10, help="把页面横着切几带")
    ap.add_argument("--save", type=Path, default=None,
                    help="存掩膜图 —— 先看拼音有没有进掩膜, 这是第一道")
    ap.add_argument("--local-ratio", action="store_true")
    a = ap.parse_args()

    img = cv2.imdecode(np.fromfile(str(a.img), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  图读不到: {a.img}")
        return
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = text_mask(gray, local_background(gray))
    n, labels, st, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)

    print("=" * 74)
    print("  WHY NO PINYIN  (ASCII only - safe to paste)")
    print("=" * 74)
    print(f"  {a.img.name}   {W} x {H}")
    print(f"  掩膜里连通块总数 {n - 1:,}")
    print()

    # 每一道闸的存活
    keep1 = [i for i in range(1, n) if st[i][4] >= 8]
    keep2 = [i for i in keep1 if st[i][3] < 0.05 * H]
    keep3 = [i for i in keep2 if st[i][2] < 0.4 * W]
    print("  逐道闸还剩多少:")
    print(f"    掩膜里的块          {n-1:>8,}")
    print(f"    面积 >= 8           {len(keep1):>8,}"
          f"   (扔掉 {n-1-len(keep1):,})")
    print(f"    高 < 5% 页高        {len(keep2):>8,}"
          f"   (扔掉 {len(keep1)-len(keep2):,})")
    print(f"    宽 < 40% 页宽       {len(keep3):>8,}"
          f"   (扔掉 {len(keep2)-len(keep3):,})")
    if len(keep3) < 60:
        print(f"    ★★★★★ 只剩 {len(keep3)} 个, **不到 60 就直接放弃整页**,")
        print( "       annotation_labels 会返回空, 这一页一个拼音都不会检测到")
    print()

    hs = np.array([st[i][3] for i in keep3], float)
    if len(hs):
        big_h = float(np.percentile(hs, 75))
        small = [i for i in keep3 if st[i][3] <= 0.55 * big_h]
        big = [i for i in keep3 if st[i][3] >= 0.8 * big_h]
        print(f"    big_h(高度75分位)   {big_h:>8.1f} px")
        print(f"    算'小'的            {len(small):>8,}   (<= {0.55*big_h:.1f})")
        print(f"    算'大'的            {len(big):>8,}   (>= {0.8*big_h:.1f})")
        if len(small) < 20 or len(big) < 20:
            print( "    ★★★★★ 小的或大的**不到 20 个, 直接放弃整页**")
        print()

    _lab, kept, _cnt, _bb = annotation_labels(mask, local_ratio=a.local_ratio)
    print(f"  最终判成注音的 {len(kept):,}"
          f"{'   (开了 local_ratio)' if a.local_ratio else ''}")
    print()

    # ★ 按横带看, 分得出"整页都漏"还是"只有某一带漏"
    kept_set = set(kept)
    step = max(1, H // a.bands)
    print(f"  按横带看(每带 {step} 像素):")
    print(f"    {'y 范围':<16}{'块':>7}{'过筛':>7}{'判成注音':>10}")
    print("    " + "-" * 40)
    for b in range(a.bands):
        y0, y1 = b * step, min(H, (b + 1) * step)
        inb = [i for i in range(1, n) if y0 <= st[i][1] < y1]
        inb3 = [i for i in keep3 if y0 <= st[i][1] < y1]
        ann = [i for i in inb3 if i in kept_set]
        flag = ""
        if len(inb3) >= 30 and not ann:
            flag = "   <- 有字没拼音, 可疑"
        print(f"    {y0:>6}-{y1:<9}{len(inb):>7}{len(inb3):>7}{len(ann):>10}"
              f"{flag}")
    print()
    print(f"  (MIN_RUN={MIN_RUN}, ANNOT_GAP_RATIO={ANNOT_GAP_RATIO})")

    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
        # ★★★★★ 第一件要看的就是这张: 拼音**有没有进掩膜**。
        #   进不了掩膜, 后面所有判据都无关紧要。
        out = a.save / f"mask_{a.img.stem[:40]}.png"
        cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)[1].tofile(
            str(out))
        print(f"  掩膜存到 {out.name}   <- 先看白字拼音在不在掩膜里")
        vis = img.copy()
        if kept:
            vis[np.isin(_lab, kept)] = (0, 0, 255)
        out2 = a.save / f"ann_{a.img.stem[:40]}.png"
        cv2.imencode(".png", vis)[1].tofile(str(out2))
        print(f"  判成注音的涂红存到 {out2.name}")


if __name__ == "__main__":
    main()
