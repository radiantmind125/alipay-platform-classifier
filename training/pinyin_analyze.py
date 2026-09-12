"""对拼音全量扫描的结果做几个必须做的检查。

★ 只报一个"占 0.67%"是不够的, 有三件事会让这个数字的含义完全不同:

1. **是不是集中在某几天。** 如果只有某一天高, 那可能是那天的数据源不一样,
   不是真实分布。(我之前就栽过一次: 拿排序后的前 3000 张估频率,
   结果 2995 张是同一天的。)

2. ★★ **是不是集中在少数几个用户。** 拼音是安卓的系统开关, 一个用户开了,
   他提交的**每一张**图都带拼音。
   如果上千张命中其实来自几十个用户, 那结论就不是"每 150 张有 1 张",
   而是"有一小批用户开了这个开关, 他们的单子会全部被拒" ——
   后者是更严重的问题, 而且处理方式完全不同。
   ★ 判法: 在**名次空间**里和随机分布比, 不能拿"凭证号差小于某个数"来判 ——
     凭证号是 19 位, 相邻两张图的号差中位数就有 92 亿, 随便拍一个阈值
     必然把所有命中都判成孤立的, 那样得出的"没有聚集"是无效结论。

3. **阈值稳不稳。** 把线从 0.20 挪到 0.35, 命中数如果是平缓变化说明分得开,
   如果某一段陡然跳变说明线卡在分布中间, 不稳。
"""
import argparse
import csv
import io
import math
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - hw), min(1.0, c + hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--corpus', type=int, default=768000,
                    help='线上总量, 用来推算张数')
    a = ap.parse_args()

    rows = []
    with io.open(a.csv, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                r['pinyin_ratio'] = float(r['pinyin_ratio'])
                r['stacked'] = int(r['stacked'])
                r['flat'] = float(r['flat'])
            except (KeyError, ValueError):
                continue
            rows.append(r)

    def hit(r, mr=0.25):
        return r['pinyin_ratio'] >= mr and r['stacked'] >= 15 and r['flat'] >= 0.15

    hits = [r for r in rows if hit(r)]
    n, k = len(rows), len(hits)
    lo, hi = wilson(k, n)
    print('=' * 66)
    print(f'总计 量到 {n:,} 张, 命中 {k:,} 张')
    print(f'  频率 {k/n*100:.3f}%   95% 区间 [{lo*100:.3f}%, {hi*100:.3f}%]')
    print(f'  按 {a.corpus:,} 张推算约 {int(k/n*a.corpus):,} 张, 每 {int(n/max(1,k))} 张有 1 张')

    # ---- 1. 按天 ----
    print()
    print('=' * 66)
    print('1. 按天看 —— 频率稳不稳')
    tot, ht = Counter(), Counter()
    for r in rows:
        try:
            d = r['name'].rsplit('_', 1)[1][:8]
        except Exception:
            continue
        tot[d] += 1
        if hit(r):
            ht[d] += 1
    print(f"   {'日期':<10}{'总数':>9}{'命中':>7}{'频率':>9}{'95% 区间':>20}")
    for d in sorted(tot):
        t, h = tot[d], ht[d]
        l, u = wilson(h, t)
        print(f'   {d:<10}{t:>9,}{h:>7}{h/t*100:>8.2f}%   [{l*100:.2f}%, {u*100:.2f}%]')

    # ---- 2. 是不是集中在少数用户 ----
    print()
    print('=' * 66)
    print('2. 是不是集中在少数用户')
    ranked = []
    for r in rows:
        try:
            ranked.append((int(r['name'].split('GWCZ')[1].split('_')[0]), hit(r)))
        except (ValueError, IndexError):
            pass
    ranked.sort()
    N = len(ranked)
    hr = [i for i, (_, h) in enumerate(ranked) if h]
    K = len(hr)
    if N > 100 and K > 10:
        # ★ 不能拿"凭证号差小于某个数"来判 —— 凭证号是 19 位, 相邻两张的号差
        #   本身就有几十亿, 随便拍一个阈值必然把所有命中都判成孤立的。
        #   要在**名次空间**里比, 并且拿随机分布当零假设。
        gaps = [hr[i + 1] - hr[i] for i in range(K - 1)]
        p = K / N
        print(f'   总 {N:,} 张, 命中 {K:,} 张')
        print(f'   相邻两个命中的名次差: 实测均值 {sum(gaps)/len(gaps):.0f}, '
              f'随机情况下应当 {N/K:.0f}')
        print(f"   {'名次差':>8}{'实测':>8}{'随机应当':>11}{'倍数':>8}")
        worst = 1.0
        for kk in (1, 2, 3, 5, 10):
            obs = sum(1 for g in gaps if g <= kk)
            exp = (1 - (1 - p) ** kk) * len(gaps)
            ratio = obs / exp if exp > 0 else 0
            worst = max(worst, ratio)
            print(f'   {"<= " + str(kk):>8}{obs:>8}{exp:>11.1f}{ratio:>7.1f}倍')
        if worst >= 2.0:
            print('   ★ 挨在一起的明显多于随机 —— 说明有用户开了拼音开关,')
            print('     他提交的每一张都会带。结论要说成"有一批用户受影响",')
            print('     不能说成"随机每 N 张有 1 张"。')
        else:
            print('   和随机撒的没区别 —— 可以按"每 N 张有 1 张"来说。')

    # ---- 3. 阈值敏感度 ----
    print()
    print('=' * 66)
    print('3. 阈值挪动看命中数变化(希望是平缓的)')
    print(f"   {'比例线':>8}{'命中':>9}{'占比':>9}{'较上一档变化':>14}")
    prev = None
    for mr in (0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35):
        kk = sum(1 for r in rows if hit(r, mr))
        ch = '' if prev is None else f'{(kk-prev)/max(1,prev)*100:+.1f}%'
        print(f'   {mr:>8.3f}{kk:>9,}{kk/n*100:>8.3f}%{ch:>14}')
        prev = kk

    # ---- 4. 贴线的有多少 ----
    print()
    print('=' * 66)
    print('4. 命中里贴着线的有多少(人工要重点看这些)')
    band = [r for r in hits if r['pinyin_ratio'] < 0.32]
    print(f'   比例 0.25~0.32 的 {len(band):,} 张 (占命中 {len(band)/max(1,k)*100:.1f}%)')
    fl = sorted(r['flat'] for r in hits)
    rr = sorted(r['pinyin_ratio'] for r in hits)
    if rr:
        print(f'   命中的比例: 最低 {rr[0]:.3f} 中位 {rr[len(rr)//2]:.3f} 最高 {rr[-1]:.3f}')
        print(f'   命中的平坦占比: 最低 {fl[0]:.3f} (线是 0.15)')
    near = [r for r in rows if not hit(r) and 0.15 <= r['pinyin_ratio'] < 0.25]
    print(f'   差一点的(0.15~0.25 没命中) {len(near):,} 张 —— 人工要抽看找漏判')


if __name__ == '__main__':
    main()
