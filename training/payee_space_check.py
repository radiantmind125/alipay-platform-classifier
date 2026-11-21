r"""收款方空白检查 —— PayeeSpaceCheck.cs 的 Python 参考实现 + 服务器批量量(2026-11-17, 经理: 针对这个写个算法 检测这个空白位置;
2026-11-21 加第 8 步 行尾: 假图在名字后面加个点, 第 6 步判不了, 漏过去 2 张)。

页面
----
蓝色页头的 转账成功 页: 转账成功 / ¥金额 / 收款方 ... 名字(**某) / 付款方式 ... / 下面白卡片 立即通知收款人。
经理给的 3 张假图(1320x2868)页头 收款方 那一行是 名字（**某）: 名字和左括号之间空一大块(34~35 像素, 字高 42),
真图同尺寸只空 12~14 像素(字高 40)。OCR 读出来有时带空格有时不带, 所以按像素量。

做法(只看图, 和 C# 一步一步对应)
--------------------------------
1. 页头: 从 0.12H 往下找第一行"蓝底像素不到一半"的行, 那以上是蓝色页头(最多到 0.55H)。
   蓝底像素: B >= 200, R <= 130, B - R >= 100。
2. 页头里的字: R >= 150 的像素(蓝底 R 只有 50~80, 白字 200 以上)。按行数字像素, 连续有字的行是一条行带。
3. 最高的行带是金额; 金额下面第一条行带是 收款方 那一行(拼音页: 比下一条矮一半以上的是拼音, 跳过)。
4. 这一行按列切成一段段(连续有字的列), 最大的空把左边标签和右边取值分开;
   标签要从 0.25W 以内起、宽 2.0~4.0 个字高(收款方 三个字, 拼音字体窄一些), 取值要到 0.75W 以后才完。
5. 取值最后一段要像右括号; 找左括号紧跟的第一个星号: 竖着只占一小截(<= 0.45 行高)、不在最底下、宽 <= 0.75 行高、宽 >= 0.4 倍自己的高。
   星号前一段是左括号(窄: 宽 <= 0.4 行高、宽 <= 0.6 倍自己的高、高 >= 0.4 行高), 再前一段是名字的最后一个字。
6. 名字最后一个字判不了的几种(宁可不判):
     太窄(宽 < 0.3 行高: 1 l 川 的一竖、括号)、太矮(高 < 0.5 行高: 。，、);
     ？ ！ 收尾(竖着两截以上、最下面一截是小点: 高 <= 0.25 行高、宽 <= 0.4 行高), 全角 ？ ！ 右边本来就空;
     那个字到左括号之间 B < 120 的像素 >= 行高平方 / 40(深色 / 彩色表情);
     名字和括号之间(两边各让 max(2, 行高/10) 像素)和底色差 > 40 的像素 >= 行高平方 / 40(蓝色、灰色表情 💤 👤),
     底色 = 标签和取值之间那一大段空白每个通道的下中位数。
7. 空白 = 左括号最左的墨 - 名字最后一个字最右的墨(像素); 比例 = 空白 / 行高。空白 * 100 >= GAP_MIN_PCT * 行高 判可疑。
8. 行尾(2026-11-21 加, 走到第 5 步就量, 第 6、7 步出结果以后再用):
   只在苹果原图截图尺寸(IPHONE_SIZES)+ 常规字号(行高 / 倍数 12.5~17.5: 3 倍屏 38~52 像素, 2 倍屏 25~35)时用。
   行尾 = 图宽 - 这一行最右的墨(右括号右沿)。行尾 * 100 >= END_MAX_PCT * 行高 判可疑(第 7 步已经可疑的不改)。
   右括号右边到图右边(最右 2% 不看, 有的图带一两像素黑边)和底色差 > 40 的像素 >= 行高平方 / 40 的, 这条不用
   (右边盖着浮窗、灰条)。底色和第 6 步一样。
   为什么: 苹果原图的排版是死的, 真图页头收款方是半角括号 名字(**某), 右括号紧贴右边距;
   假图模板用全角括号 名字（**某）, 全角右括号自己右边带一大块空, 模板右边距也宽一点, 所以整行离右边远一截。
   这一条不管名字后面放什么(点、句号、一竖、空格)都一样。

实测(2026-11-17)
    经理的 3 张假图 0.81~0.83; 经理截的图(780x1280 压过)0.82; JPEG 90~40、缩 0.75 / 0.5、TG 1280、BGRA 0.81~0.90
    服务器 11 月蓝图全部 95,810 张, 量到约 2 万张(其余不是这个页面、或者是上面判不了的几种):
        苹果原生截图 0.29~0.34 居多(名字和括号之间本来就空半个字), 安卓 0.02~0.25,
        最后一个字是 月 日 一类(右边留白多)、拉丁字母、＆ 0.45~0.50, 方块字(□)0.50;
        >= 0.65 只有 1 张: 昵称自己带了几个空格(晨SIR   (**英), 1.19), 这种分不出来。
        第 6 步 ？！ 和表情两条是看了服务器结果加的: 第一轮(没有这两条, 门槛 0.55)判可疑 29 张 = ？！ 收尾 23 + 表情 5 + 空格 1,
        全是真图; 没有这两条按 0.65 是 21 张(？！ 15 + 表情 5 + 空格 1)。两条一共挡掉 37 张(？！ 26、表情 11)。
        复核方法: 服务器存下的收款方那一行小图(比例 >= 0.30 的全部)左边补回原来的 x, 直接跑 measure_line。
    服务器 11 月白图随机 3 万张: 0 张量到(没有蓝色页头)。
    本机真图 3,302 张整页(7~11 月): 0 张可疑。

实测第 8 步(2026-11-21)
    漏过的 2 张(11-20, 1320x2868, 后台截图里 1/4 大小): 消失的.（**涛）、L.（*旺）。和 11-17 的 3 张是同一个假图模板
    (收款方那一行在 y 653, 真图 571; 页头底 830; 全角括号; 字细)。名字后面加了个点, 点 -> 左括号 0.76 / 0.69 个字高,
    第 6 步 "名字最后一个字太窄" 判不了 -> 0。
    不能改成从点量起: 真图有人昵称就是一个句号, 。(**燕) 句号到半角左括号 0.80 个字高, 和假图一样。
    行尾(个字高):
        服务器 11 月蓝图全部 95,810 张(11-01..04, 2026-11-21 用这一版跑, 26 分钟): 第 8 步改判 0 张;
        第 1~5 步的位置和 11-17 那一轮逐张一样; 可疑还是只有 1 张(晨SIR 昵称自带空格, 第 7 步判的, 和 11-17 一样)。
        苹果尺寸走到括号的 14,279 张: 用上第 8 步的 14,235 张 行尾 1.490~2.050(中位 1.878, 99.9% 1.925, >= 2.1 的 0 张;
        按像素 3 倍屏 71~83(77 有 11,580 张), 2 倍屏 49~51, 排版是死的); 右边不干净不用的 35 张(悬浮球、浮窗)行尾都 <= 1.925;
        字号不常规不用的 9 张(8 张行带和别的行并在一起 行高 71~83; 1 张 1284 右边盖灰条 行高 37 行尾 4.03, 只看右边干不干净也会不用)。
        苹果尺寸第 6 步判不了的 1,399 张(昵称以 。，、！？ 一竖 表情收尾)要是从最后一个点量起按 0.65 判, 会误判 296 张; 第 8 步 0 张。
        白图随机 2 万张: 0 张量到。服务器存的 1,498 张小图放回原 x 用本版复核, 和服务器结果逐张一样。
        本机苹果真图(含 。 窄字 表情收尾判不了的)1.82~1.93;
        11-17 的 3 张假图 2.71(114 像素), 11-21 两张约 2.6~2.7(在 1/4 截图上量, 真图同样缩过量出来和原尺寸一样)。
    合成: 3 张假图名字后面加 点 / 句号 / 一竖(放在原来的空里或名字左移) 18 张: 旧版全是 0, 新版全判可疑, JPEG 95~40 也是;
        缩 0.75 / 0.5 以后不是苹果原图尺寸, 第 8 步不用, 这 18 张又是 0(要传上传的原图)。
        真图 307 张加同样的东西 1,540 张(每张再做 JPEG 95~40 四份、缩 0.75 / 0.5 两份): 0 张可疑。
    边界样本: 行尾刚好 97 像素(2.31)判可疑、96(2.29)不判; 右边涂灰 6% 第 8 步不用; 最右 3 像素黑边照用。
    旧版 11-17 对拍用的 108,074 张图: 新旧结果、理由逐字一样(第 8 步在真图上一张没改)。

返回: v 0 判不了 / 1 正常 / 2 可疑(和别的检查一样 Ok=1 Suspicious=2 CannotDetermine=0)。

用法
----
    python -I training/payee_space_check.py 图 [图 ...]
    python -u training/payee_space_check.py --scan E:/BlueImages --since 20261101 --workers 16 --max-minutes 60 ^
        --out E:/_move_logs/space_1121/blue.csv --crops E:/_move_logs/space_1121/crops
    小图存哪些: 可疑全部 / 第 8 步右边不干净全部 / 行尾 >= 2.10 全部 / 苹果尺寸第 6 步判不了的全部 / 苹果尺寸其余抽 1/200。
和 training/demo/PayeeSpaceCheck.cs 逐步对应, 改一边要同步改另一边。
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy 之前

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

GAP_MIN_PCT = 65        # 空白 / 行高 >= 65% 判可疑; 整数比较, C# 和这里不会差在小数上
#                         服务器 11 月蓝图真图(排除下面几种判不了的)最高 0.50, 假图 0.81 起, 取中间
INK_R = 150             # 页头白字: R >= 这个
FOREIGN_B = 120         # 页头里 B < 这个的像素既不是白字也不是蓝底(深色表情、头像、浮窗);
#                         JPEG 在白字边上把蓝底的 B 压到 140~160(色度抽样), 所以不能用 200
NONBG_DIST = 40         # 空白里和底色差(三通道差的绝对值之和) > 40 的像素算"有东西"(蓝色 / 灰色表情 💤 👤 等)
END_MAX_PCT = 230       # 第 8 步(苹果原图尺寸): 行最右的墨(右括号)离图右边 >= 2.30 个字高判可疑; 整数比较
#                         真图是半角右括号, 离右边 1.8~2.05 个字高; 假图模板用全角右括号, 2.6~2.7
# 苹果手机原图截图尺寸(和 YenCheck / AmountYCheck 一样): 这些尺寸的页面排版是死的, 收款方那一行的右端每张都在同一个位置
IPHONE_SIZES = {(640, 1136), (750, 1334), (1242, 2208), (828, 1792), (1125, 2436), (1242, 2688), (1170, 2532),
                (1284, 2778), (1179, 2556), (1290, 2796), (1206, 2622), (1320, 2868), (1260, 2736)}

UPLOAD_DAY = re.compile(r"_([0-9]{8})[0-9]{6}\.")


def _runs(on):
    """一维 bool -> [(起, 止)), 连续为真的段"""
    out = []
    x, n = 0, len(on)
    while x < n:
        if on[x]:
            x0 = x
            while x < n and on[x]:
                x += 1
            out.append((x0, x))
        else:
            x += 1
    return out


def check_bgr(img):
    """img: 原图解码成的 BGR(uint8)。返回 dict, v / reason 一定有, 量到的话还有 gap / h / ratio 和位置"""
    H, W = img.shape[:2]
    r = {"v": 0, "W": W, "H": H}
    if H < 400 or W < 300:
        r["reason"] = "图太小"
        return r
    B = img[..., 0].astype(np.int32)
    R = img[..., 2].astype(np.int32)
    lim = int(0.55 * H)
    # 1. 蓝色页头的底
    blue = (B[:lim] >= 200) & (R[:lim] <= 130) & (B[:lim] - R[:lim] >= 100)
    bcnt = blue.sum(axis=1)
    yb = lim
    for y in range(int(0.12 * H), lim):
        if bcnt[y] * 2 < W:
            yb = y
            break
    r["yb"] = yb
    if yb < int(0.15 * H):
        r["reason"] = "没有蓝色页头"
        return r
    # 2. 行带
    xa, xz = int(0.02 * W), int(0.98 * W)
    ink = R[:yb] >= INK_R
    rowcnt = ink[:, xa:xz].sum(axis=1)
    bands = _runs(rowcnt >= 2)
    if len(bands) < 3:
        r["reason"] = "页头字行不到 3 条"
        return r
    hs = [b - a for a, b in bands]
    ia = int(np.argmax(hs))
    # 3. 金额下面第一条(拼音行跳过)
    pay = -1
    for i in range(ia + 1, len(bands)):
        if i + 1 < len(bands) and hs[i] * 10 < 6 * hs[i + 1]:
            continue
        pay = i
        break
    if pay < 0:
        r["reason"] = "金额下面没有字行"
        return r
    y0, y1 = bands[pay]
    return measure_line(img, y0, y1, r)


def _lowmed(v):
    """uint8 数组的下中位数(第 (n+1)//2 小的值), 用直方图算, C# 一样"""
    c = np.cumsum(np.bincount(v.ravel(), minlength=256))
    return int(np.searchsorted(c, (v.size + 1) // 2))


def measure_line(img, y0, y1, r):
    """第 4~7 步: 已经知道 收款方 那一行是 [y0, y1) 行。服务器小图复核也直接调这个(左边补底色到原来的 x)。"""
    H, W = img.shape[:2]
    B = img[y0:y1, :, 0].astype(np.int32)
    R = img[y0:y1, :, 2].astype(np.int32)
    h = y1 - y0
    r["y0"], r["y1"], r["h"] = y0, y1, h
    if h < 12:
        r["reason"] = "字太小"
        return r
    band = R >= INK_R
    seg = _runs(band.sum(axis=0) > 0)
    if len(seg) < 5:
        r["reason"] = "收款方那一行字太少"
        return r
    # 4. 标签 / 取值
    gaps = [seg[i + 1][0] - seg[i][1] for i in range(len(seg) - 1)]
    k = int(np.argmax(gaps))
    lab, val = seg[:k + 1], seg[k + 1:]
    lw = lab[-1][1] - lab[0][0]
    r["label_x0"], r["label_w"] = lab[0][0], round(lw / h, 3)
    if lab[0][0] > 0.25 * W or not (2.0 * h <= lw <= 4.0 * h) or val[-1][1] < 0.75 * W:
        r["reason"] = "这一行不像 收款方 ... 名字"
        return r
    # 5. 每一段的宽、上沿、下沿
    feats = []
    for a, b in val:
        rows = np.nonzero(band[:, a:b].any(axis=1))[0]
        feats.append((a, b, b - a, int(rows[0]), int(rows[-1]) + 1))
    # 星号 = 竖着只占一小截、不在最底下、不太宽也不太扁; 左括号 = 窄、比自己宽高得多、至少 0.4 行高。
    # 要"左括号紧跟星号"才算: 小写字母(o a u ...)也像星号, 但前面不是左括号
    def is_star(f):
        a, b, w, t, bt = f
        e = bt - t
        return e * 100 <= 45 * h and t * 100 < 60 * h and w * 100 <= 75 * h and w * 10 >= 4 * e

    def is_bracket(f):
        a, b, w, t, bt = f
        e = bt - t
        return w * 10 <= 4 * h and w * 10 <= 6 * e and e * 10 >= 4 * h

    if not is_bracket(feats[-1]):
        r["reason"] = "取值最后不是右括号(不是 名字(**某) 的样子)"
        return r
    any_star = False
    star = -1
    for i in range(len(feats)):
        if is_star(feats[i]):
            any_star = True
            if i >= 1 and is_bracket(feats[i - 1]):
                star = i
                break
    if star < 0:
        r["reason"] = ("星号前面不是左括号" if any_star else "取值里没有星号") + "(不是 名字(**某) 的样子)"
        return r
    if star < 2:
        r["reason"] = "左括号前面没有名字"
        return r
    ba, bb, bw, bt0, bt1 = feats[star - 1]
    pa, pb, pw, pt0, pt1 = feats[star - 2]
    r["br_x0"], r["prev_x1"] = ba, pb
    # 8(先量, 第 6、7 步算完再用). 行尾: 苹果原图尺寸、常规字号时, 看右括号离图右边多远
    _line_end(img, y0, y1, h, val[-1][1], seg[k][1], seg[k + 1][0], r)
    # 6. 名字最后一个字
    if pw * 10 < 3 * h:
        r["reason"] = "名字最后一个字太窄(1 l 川 或括号一类), 量不准"
        return _end_rule(r)
    if (pt1 - pt0) * 10 < 5 * h:
        r["reason"] = "名字最后一个是标点(。，、一类), 量不准"
        return _end_rule(r)
    # ？ ！ 收尾: 竖着分成两截以上, 最下面一截是个小点(高 <= 0.25 行高、宽 <= 0.4 行高); 全角的 ？ ！ 右边本来就空一大块
    rr = _runs(band[:, pa:pb].any(axis=1))
    if len(rr) >= 2:
        la, lb = rr[-1]
        lc = np.nonzero(band[la:lb, pa:pb].any(axis=0))[0]
        if (lb - la) * 4 <= h and (int(lc[-1]) - int(lc[0]) + 1) * 10 <= 4 * h:
            r["reason"] = "名字最后一个是 ？ ！ 一类(下面一个点), 量不准"
            return _end_rule(r)
    if int((B[:, pa:ba] < FOREIGN_B).sum()) * 40 >= h * h:
        r["reason"] = "名字后面有表情图标一类的彩色东西, 量不准"
        return _end_rule(r)
    # 空白里有没有别的东西(蓝色 / 灰色的表情, R 不够白字、B 又不低): 和底色比; 底色 = 标签和取值之间那一大段空白的下中位数
    m = max(2, h // 10)
    if ba - m > pb + m:
        ga, gb = seg[k][1], seg[k + 1][0]
        bg = [_lowmed(img[y0:y1, ga:gb, c]) for c in range(3)]
        reg = img[y0:y1, pb + m:ba - m].astype(np.int32)
        dist = np.abs(reg[..., 0] - bg[0]) + np.abs(reg[..., 1] - bg[1]) + np.abs(reg[..., 2] - bg[2])
        if int((dist > NONBG_DIST).sum()) * 40 >= h * h:
            r["reason"] = "名字和括号之间有表情一类的东西, 量不准"
            return _end_rule(r)
    # 7. 空白
    gap = ba - pb
    r["gap"] = gap
    r["ratio"] = round(gap / h, 4)
    r["v"] = 2 if gap * 100 >= GAP_MIN_PCT * h else 1
    q = (gap * 200 + h) // (2 * h)            # 比例的百分位, 四舍五入用整数算(C# 一样), 理由里两边逐字相同
    txt = f"{q // 100}.{q % 100:02d}"
    r["reason"] = (f"可疑: 名字和左括号之间空 {gap} 像素 = {txt} 个字高 >= 0.{GAP_MIN_PCT:02d}(真图一般 0.1~0.5)"
                   if r["v"] == 2 else f"正常: 名字和左括号之间空 {gap} 像素 = {txt} 个字高")
    return _end_rule(r)


def _line_end(img, y0, y1, h, xe, ga, gb, r):
    """第 8 步的量法: 行最右的墨(右括号的右沿)xe 离图右边 end = W - xe 像素。
    只在苹果原图截图尺寸 + 常规字号(行高 / 倍数 12.5~17.5: 3 倍屏 38~52 像素, 2 倍屏 25~35)时用;
    右括号右边到图右边(最右 2% 不看, 有的图带一两像素的黑边)要是干净的底色, 不干净(右边盖着灰条、浮窗)不用这条。
    底色 = 标签和取值之间那一大段空白每个通道的下中位数(和第 6 步一样)。
    写进 r: end / end_h / iphone / end_rule(-1 不适用, 0 右边不干净不用, 1 用了) / end_hit(0/1)"""
    W = img.shape[1]
    H = r.get("H", img.shape[0])               # 服务器小图复核时整页高从 r 来; 正常调用 r["H"] 就是图高
    end = W - xe
    s = 2 if W < 1000 else 3                   # 2 倍屏: 640 / 750 / 828 宽; 别的都是 3 倍屏
    r["end"] = end
    r["end_h"] = round(end / h, 4)
    r["iphone"] = int((W, H) in IPHONE_SIZES)
    r["end_rule"] = -1
    r["end_hit"] = 0
    if not r["iphone"] or not (25 * s <= 2 * h <= 35 * s):
        return
    m = max(2, h // 10)
    xa, xb = xe + m, W - W // 50
    if xb > xa:
        bg = [_lowmed(img[y0:y1, ga:gb, c]) for c in range(3)]
        reg = img[y0:y1, xa:xb].astype(np.int32)
        dist = np.abs(reg[..., 0] - bg[0]) + np.abs(reg[..., 1] - bg[1]) + np.abs(reg[..., 2] - bg[2])
        if int((dist > NONBG_DIST).sum()) * 40 >= h * h:
            r["end_rule"] = 0
            return
    r["end_rule"] = 1
    r["end_hit"] = int(end * 100 >= END_MAX_PCT * h)


def _end_rule(r):
    """第 6、7 步出了结果(判不了 / 正常 / 可疑)以后: 第 8 步中了、而第 7 步还没判可疑 -> 改判可疑, 原来的理由接在后面。
    gap_v 记第 6、7 步自己的结果(批量统计用)。"""
    r["gap_v"] = r["v"]
    if r["end_hit"] and r["v"] != 2:
        e, h = r["end"], r["h"]
        q = (e * 200 + h) // (2 * h)
        txt = f"{q // 100}.{q % 100:02d}"
        r["v"] = 2
        r["reason"] = (f"可疑: 右括号离右边 {e} 像素 = {txt} 个字高 >= {END_MAX_PCT // 100}.{END_MAX_PCT % 100:02d}"
                       f"(苹果原图尺寸真图 1.8~2.05, 假图全角括号 2.6 以上); 空白那条: {r['reason']}")
    return r


def read_bgr(path):
    import cv2
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 批量

COLS = ["path", "err", "W", "H", "v", "reason", "yb", "y0", "y1", "h", "label_x0", "label_w", "prev_x1", "br_x0", "gap",
        "ratio", "iphone", "end", "end_h", "end_rule", "end_hit", "gap_v", "sec"]


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


_CROPS = ""


def _init_crops(crops):
    global _CROPS
    _init_worker()
    _CROPS = crops


def score_one(path):
    t0 = time.time()
    try:
        img = read_bgr(path)
        if img is None:
            return {"path": path, "err": "unreadable"}
        r = check_bgr(img)
        out = {"path": path, "err": ""}
        out.update({k: r.get(k, "") for k in COLS if k not in ("path", "err", "sec")})
        out["sec"] = round(time.time() - t0, 3)
        if _CROPS and "y0" in r and "label_x0" in r:
            # 收款方那一行的小图(给人看, 也能放回原 x 用 measure_line 复核), 上下各多 1 个行高, 右边到图边。
            # 存哪些(第 8 步那一轮): 可疑的全存; 第 8 步右边不干净的全存; 行尾 >= 2.10 个字高的全存;
            # 苹果尺寸里第 6 步判不了的(点、窄字、？！、表情收尾)全存; 苹果尺寸用了第 8 步的再抽 1/200
            import cv2
            import zlib
            h = r["h"]
            c = img[max(0, r["y0"] - h):min(img.shape[0], r["y1"] + h), max(0, r["label_x0"] - 10):]
            hsh = zlib.crc32(os.path.basename(path).encode("utf-8"))
            b = None
            if r["v"] == 2:
                b = "SUSP"
            elif r.get("end_rule") == 0:
                b = "ENDDIRTY"
            elif r.get("end_rule") == 1 and r["end"] * 100 >= 210 * h:
                b = "ENDHI"
            elif r.get("iphone") and r.get("gap_v") == 0:
                b = "IPCANNOT"
            elif r.get("end_rule") == 1 and hsh % 200 == 0:
                b = "IPSAMPLE"
            if b:
                d = os.path.join(_CROPS, b)
                os.makedirs(d, exist_ok=True)
                cv2.imencode(".png", c)[1].tofile(os.path.join(d, f"{r.get('end_h', 0):.3f}_" +
                                                              os.path.splitext(os.path.basename(path))[0] + ".png"))
        return out
    except Exception as e:  # noqa: BLE001
        return {"path": path, "err": repr(e)[:200], "sec": round(time.time() - t0, 3)}


def _inputs(args):
    paths = []
    for a in args:
        if a.startswith("@"):
            with open(a[1:], encoding="utf-8") as f:
                paths += [x.strip() for x in f if x.strip()]
        elif os.path.isdir(a):
            for root, _dirs, files in os.walk(a):
                paths += [os.path.join(root, x) for x in files if x.lower().endswith((".jpg", ".jpeg", ".png"))]
        else:
            paths.append(a)
    return paths


def scan(a):
    paths = _inputs(a.inputs)
    n_all = len(paths)
    if a.since:
        paths = [p for p in paths if (UPLOAD_DAY.search(os.path.basename(p)) or [None, ""])[1] >= a.since]
    rng = random.Random(a.seed)
    rng.shuffle(paths)
    if a.sample:
        paths = paths[:a.sample]
    print(f"images found {n_all}, since {a.since or '-'}: {len(paths)} to check", flush=True)
    from multiprocessing import Pool
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.crops:
        os.makedirs(a.crops, exist_ok=True)
    t0 = time.time()
    n = err = 0
    vs = {0: 0, 1: 0, 2: 0}
    reasons = {}
    ratios = []
    trans = {}              # 第 6、7 步的结果 -> 最后结果(第 8 步改了哪些)
    endrule = {}            # 苹果尺寸、量到行尾的页: 第 8 步 -1 不适用 / 0 右边不干净 / 1 用了
    ends = {}               # 用了第 8 步的: (W, H) -> [end_h ...]
    stopped = False
    block = max(256, a.workers * 64)
    with Pool(a.workers, initializer=_init_crops, initargs=(a.crops,)) as pool, \
            open(a.out, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for i in range(0, len(paths), block):
            if a.max_minutes and (time.time() - t0) / 60 >= a.max_minutes:
                stopped = True
                break
            for r in pool.imap_unordered(score_one, paths[i:i + block], chunksize=8):
                w.writerow(r)
                n += 1
                if r.get("err"):
                    err += 1
                    continue
                vs[r["v"]] += 1
                key = r["reason"].split(":")[0] if r["v"] else r["reason"]
                reasons[key] = reasons.get(key, 0) + 1
                if r["ratio"] != "":
                    ratios.append(float(r["ratio"]))
                if r["gap_v"] != "":
                    t = (r["gap_v"], r["v"])
                    trans[t] = trans.get(t, 0) + 1
                if r["iphone"] == 1:
                    endrule[r["end_rule"]] = endrule.get(r["end_rule"], 0) + 1
                    if r["end_rule"] == 1:
                        ends.setdefault((r["W"], r["H"]), []).append(float(r["end_h"]))
            fo.flush()
            el = time.time() - t0
            print(f"{n} / {len(paths)}  {n / max(el, 1e-9):.2f} pages/s  measured {vs[1] + vs[2]}  suspicious {vs[2]}  err {err}",
                  flush=True)
    print()
    print(f"pages {n}  errors {err}  ok {vs[1]}  suspicious {vs[2]}  cannot {vs[0]}"
          + ("  STOPPED BY --max-minutes" if stopped else ""))
    for k, c in sorted(reasons.items(), key=lambda t: -t[1]):
        print(f"  {c:7d}  {k}")
    if ratios:
        q = np.percentile(np.array(ratios), [1, 5, 25, 50, 75, 95, 99, 99.5, 99.9])
        print("ratio percentiles 1/5/25/50/75/95/99/99.5/99.9:", " ".join(f"{x:.3f}" for x in q))
        for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80):
            k = sum(x >= t for x in ratios)
            print(f"  >= {t:.2f}: {k}  ({100.0 * k / len(ratios):.3f} % of measured)")
    print("step 6-7 verdict -> final verdict (pages that reached the bracket):",
          "  ".join(f"{a}->{b}: {c}" for (a, b), c in sorted(trans.items())))
    print("changed by step 8 (line end):", sum(c for (a, b), c in trans.items() if a != b))
    print("iPhone-size pages that reached the bracket, step 8 -1 n/a / 0 right side not clean / 1 used:",
          "  ".join(f"{k}: {c}" for k, c in sorted(endrule.items())))
    alle = []
    for wh, e in sorted(ends.items(), key=lambda t: -len(t[1])):
        alle += e
        q = np.percentile(np.array(e), [0, 1, 50, 99, 100])
        print(f"  {wh[0]}x{wh[1]}: {len(e)}  end_h min/p1/p50/p99/max " + " ".join(f"{x:.3f}" for x in q))
    for t in (2.0, 2.1, 2.2, 2.3, 2.4):
        print(f"  end_h >= {t:.1f}: {sum(x >= t for x in alle)}")
    print(f"SPACE_SCAN_DONE pages={n} minutes={(time.time() - t0) / 60:.1f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--since", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1117)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--crops", default="")
    a = ap.parse_args()
    if a.scan:
        if not a.out:
            ap.error("--scan 要 --out")
        scan(a)
        return
    for p in a.inputs:
        img = read_bgr(p)
        if img is None:
            print("UNREADABLE", p)
            continue
        r = check_bgr(img)
        print(r["W"], r["H"], r["v"], r.get("ratio", ""), r.get("end_h", ""), r.get("end_rule", ""), r["reason"], p)


if __name__ == "__main__":
    main()
