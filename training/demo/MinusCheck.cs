// 负号几何检查 —— .NET 版, 不用模型, 不用显卡, 只做算术。
//
// ★ 这个文件**不依赖任何第三方包**, 只用 System.*, 直接拖进工程就能编。
//   入口收的是**解码好的像素字节数组**, 你现在用什么解码器都行, 不用为它装新包。
//   **一个文件就够, 不需要别的。**
//
// ★★ 拿到先看这两条, 不然容易走弯路
// ---------------------------------
// 1. **拿 03.jpg / 04.jpg 自验会看到 CannotDetermine, 那不是没生效。**
//    它们数字高只有 47 和 53, 低于默认的 60 px 弃权线。详见下面"怎么自验"那一节。
// 2. **用 System.Drawing 解码的话, `stride` 和 `PixelOrder.Bgr` 两个都必须传。**
//    少传任何一个都会**静默出错**(整图错位 / 页型认反), 不会抛异常。
//    文件最后有现成的 GDI 调用示例, 照抄即可。
//
// 干什么
// ------
// 白底账单详情页里, 量**金额前面那个负号的宽度 / 数字的中位宽度**。
// 真图这个比值很集中, 被拉长过的会明显偏出去。4 万张白单实测的中位数:
//   4 位金额 **0.6739**(n=6,092) / 5 位 **0.7021**(n=28,020) / 6 位 **0.7021**(n=928)
// ★ 位数不同基线略有差别(中位数字宽取决于出现了哪些数字, "1" 比 "0" 窄),
//   但差距很小, 所以下面用一个统一阈值 0.78, 不按位数分。
//
// 怎么调
// ------
//   var r = MinusCheck.Check(rgb, width, height);      // rgb = 解码好的字节数组
//   if (r.Verdict == MinusVerdict.Suspicious)
//       amount = "";                                   // 比例不对, 金额输出空字符串
//   // Ok 和 CannotDetermine 都照常走 OCR
//
// ★ **多线程直接调没问题**: 这个类只有 const, 没有任何可变的静态状态, 全是局部变量。
//   (实测 700 张图, 16 路并行跑和单线程串行跑, **700/700 结果完全一样**。)
// ★ 内存: 每次调用大约占 图宽*图高*2 字节的临时空间(灰度图 + 标记用的位图),
//   300 万像素的截图约 6 MB, 用完就回收。并发度高的话按这个估。
//
// ★★ 你的解码器如果每行末尾有对齐填充, **一定要把行跨距 stride 传进来**:
//       MinusCheck.Check(rgb, w, h, PixelOrder.Bgr, stride);
//   **System.Drawing / GDI 就有这个填充**, 宽度不是 4 的倍数时每行补 1~3 个字节。
//   实测本地进件 1,200 张里 **33.8% 的宽度不是 4 的倍数**
//   (1179 / 1290 / 1170 / 1206 这些都是, 1080 / 1320 / 1284 才是刚好的)。
//   不传 stride 的话整张图会**逐行斜着错开**, 量出来全是垃圾, 而且**一声不吭不报错**。
//
// 为什么阈值是 0.78 而不是更高
// ---------------------------
// 按"负号加 6 个像素"这个真实攻击尺度反推的。
// 做法: 拿真图, **把负号往左描宽 6 px**, 再看报不报得出来。1,473 对实测:
//
//   阈值 0.76   加6px 能抓到 99.8%
//   阈值 0.78   加6px 能抓到 99.5%   <- 用这个
//   阈值 0.80   加6px 能抓到 89.3%
//   阈值 0.85   加6px 能抓到 32.7%
//   阈值 0.90   加6px 能抓到  0.0%   <- 太高, 一张都抓不到
//
// ★ 抓不到的那 7 张(0.5%)有共同点: **本来的负号就偏短**(中位 0.655, 正常 0.700),
//   而且**字特别大**(数字高中位 99 px, 正常 72 px)。
//   `bar_width` 是个比值, 所以**字越大, 固定的 6 px 占的比例越小** ——
//   有一张数字高 182 px, 加 6 px 几乎看不出来。**字特别大的图这条判据会偏弱。**
//
// 整体报出率(4 万张白单随机抽样, 可量 35,045 张, 阈值 0.78)
// --------------------------------------------------------
//   数字高 40~60 px    3 / 2,190    = **13.7/万**
//   数字高 60 px 以上  34 / 32,615  = **10.4/万**
//
// ★ 这是**报出率**, **不是误报率**。进件池没有逐张的真伪标注,
//   报出来的里面本来就有真的被改过的, 所以不能当误报率讲。
//
// ★★★ 误杀最可能的来源(按实测排序)
// --------------------------------
// **(一) 蓝底转账页没跳过 —— 这个最大。**
//   蓝图金额**不带负号**, 这条线对蓝图覆盖率本来就是 0。
//   但页面上那条**横向分隔线**会被当成负号量。随机抽 **11,928 张蓝图**实测:
//     · **40 张**被量出了 bar_width(**1/298**, 合 33.5/万)
//     · 其中 **36 张(90%)会判成 Suspicious** —— bar_width 最大到 **12.23**(正常才 0.70)
//     · 这 40 张的数字高是 **69~182**, **全部在 60 以上, 那道字号闸一张都拦不住**
//   蓝图池 12 万张量级, 折合**三四百张金额被无故清空**。
//   **这一版开头就把蓝图整个跳掉。**
//
//   ★ 先前只抽了 3,970 张, 命中 8 张而且**碰巧个个都很极端**(全部 >= 8.45, 8/8 都会报),
//     于是写成了"100% 会报"。样本放大到 1.2 万张、命中 40 张之后,
//     bar_width 实际是 **0.0439 ~ 12.2314**, 报出比例是 **90% 而不是 100%**。
//     **8 个事件的样本不足以支撑"全部"这种话。**
//
// **(二) 三道有效性闸没做。** 真正压住噪声的是这三道(见下面第 4 步):
//   数字高度变异 > 8% / 宽度变异 > 30% / 宽高比不在 0.45~0.75 —— 任一条不满足就放弃。
//   没有这三道的话**一整张二维码**会混进来, 而且排在离群榜第一。
//
// **(三) 字号闸 —— 实测作用比原先写的小得多, 这里如实记一笔。**
//   原来的注释写"小字号报出 200~400/万、p99 到 1.18~1.78、两档差 15~30 倍"。
//   **4 万张实测都不成立**: 小字号档 13.7/万 对 10.4/万, **只差 1.3 倍**;
//   各档 p99 全在 **0.73~0.76**, 没有一档接近 1.18。
//   ★ 原因: 那几个旧数字是**加上面三道有效性闸之前**的口径。
//     闸加上以后小字号本身已经不散了。(`minus_outlier.py` 里那段注释同样是旧口径。)
//   **开这道闸的实际代价和收益**: 丢掉 **6.9%** 可量的图,
//   整体报出率从 10.6/万 只降到 **10.4/万**。
//   而且报出率和字号**并不是单调关系** —— 实测最高的一档反而是 90~120 px(177/万)。
//   -> **默认仍然保留 60**(偏保守, 和之前的行为一致), 但**要关掉就把 MinDigitHeight 设成 0**。
//      这是覆盖率和精度的取舍, 谁用谁定。
//
// 怎么接
// ------
//   Suspicious       -> 负号比例不对, **金额输出空字符串**, 别硬给一个数
//   CannotDetermine  -> 这条判据没意见(蓝图/小字/定位不到), **OCR 照常走**
//   Ok               -> 负号正常
//
// 覆盖边界(别让人以为管得更宽)
// ----------------------------
// - **只管白底账单详情页**, 蓝底转账页不管。
// - 只管"负号被拉长"这一种手法。**改数字、复制粘贴数字, 这条一律看不见。**
// - 判的是"这个负号和同一张图里的数字比例不对", **不是**"这张图是生成的"。
//
// 怎么自验(不用传任何图, 你手上那两张就行)
// ----------------------------------------
//   03.jpg   BarWidth = 1.0000   DigitHeight = 47   DigitAspect = 0.5745   DigitCount = 5
//   04.jpg   BarWidth = 0.7097   DigitHeight = 53   DigitAspect = 0.5849   DigitCount = 5
//
// ★★★ **先看这条, 免得以为没生效**:
//   **按默认设置, 这两张都会返回 CannotDetermine, 不是 Suspicious。**
//   因为它们的数字高是 **47 和 53, 都低于默认的 60 px 弃权线**。
//   **要对的是 BarWidth 这个数, 不是 Verdict。**
//
// ★★ 把 `MinDigitHeight` 改成 0(即关掉那道弃权闸)再跑, 这两张就是:
//       03.jpg -> **Suspicious**  <- 就是你说的那张假图
//       04.jpg -> **Ok**          <- 正常那张
//   **两张都判对了。** 也就是说判据本身没问题, 是那道 60 px 的闸把它们一起挡掉了。
//   ★ 而且 03.jpg 落在 **40~50 px** 这一档, 这一档 4 万张实测**一张都没报出**(0/1,519) ——
//     那里并没有噪声需要挡。要不要关这道闸, 看上面"(三)"那一节的取舍, 谁用谁定。
//
// 这一版和 Python 那版对过账
// --------------------------
// 逐张比 bar_width, 三批一共 **3,551 次测量, 数值全部相同**(差 < 5e-5):
//   · **随机**抽的白单 699 张 —— 605 张两边都量得出且数值一致, 94 张两边都量不出,
//     **"量得出还是量不出"699 张全对得上**
//   · 另外挑出来的 1,473 张(能量、数字高 >= 60、本来判 Ok 的), 数值一致
//   · 同样这 1,473 张**把负号描宽 6 px** 之后再比一遍, 数值仍然一致
//
// 再放大到 9,000 张跑了一次(独立抽样):
//   · 两边都量得出的 **7,930 张, 7,930 张数值完全相同** —— **一个数值分歧都没有**
//   · **两边报出的名单完全一样**(各 6 张)
//   · 有 **12 张(0.13%)** 是 **C# 量得出而 Python 量不出** —— 这类图 **PIL 直接报错不读**
//     (截断的 JPEG、扩展名骗人的 HEIC 之类), 而 ImageSharp 会照读。反过来一张都没有。
//   · 另有 **2 张(0.02%)** ImageSharp 自己也读不了, 会抛异常 —— 调用方要 try/catch。
//
// ★ **别把这说成"永远一张不差"**。两边用的是不同的 JPEG 解码器(PIL/libjpeg 对 ImageSharp),
//   同一张图解出来的像素本来就可能差一点点, 原理上就可能让 bar_width 差一档。
//   我这 7,930 张里没碰到, 但**不能保证零**。
//   **能保证的是: 同样的像素喂进去, 两边算出来完全一样** —— 分歧只可能来自解码器, 不来自这里的算法。
//   ★ 要和 Python 完全同口径, 就把**截断的 JPEG 也当成读不了**。
//
// 速度: 单线程 **约 38 ms 一张**(含解码; 本机空载, 300 万像素左右的截图)。不需要显卡。
//   ★ 同一批图 Python 那版是 35 ms —— **两边差不多**, 别指望换成 C# 就快很多
//     (Python 那边真正干活的是 OpenCV 的 C 代码)。机器一忙这个数会涨好几倍,
//     实测同机满载时会掉到 60~170 ms, 按容量规划的话要用你自己机器的实测值。
//
// ★ 对过账的图都是**七月**的白单。八月那批在服务器上, 本地没有, 没在这版上重跑过。
//
// ★ 唯一一处**故意和 Python 不一样**: 这版开头就把蓝底页整个跳掉,
//   Python 那版没跳, 靠"蓝图上找不到横杠"自然落空。

#nullable enable      // 显式打开, 这样不管你工程开没开 nullable, 编译都是零警告

using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssp
{
    public enum MinusVerdict
    {
        // ★ CannotDetermine 放在 0 位, 这样 new MinusResult() 默认是"不下结论"而不是"正常"。
        //   万一哪条路径忘了赋值, 默认弃权是安全的, 默认 Ok 就是漏判了。
        CannotDetermine = 0,   // 判不了 -> OCR 照常
        Ok = 1,                // 负号正常
        Suspicious = 2,        // 负号偏长 -> 金额输出空
    }

    /// <summary>像素在字节数组里的通道顺序。</summary>
    public enum PixelOrder
    {
        Rgb = 0,   // ImageSharp Rgb24 / SkiaSharp Rgba8888
        Bgr = 1,   // ★ System.Drawing 的 Format24bppRgb 内存里其实是 BGR, 别选错
    }

    public sealed class MinusResult
    {
        public MinusVerdict Verdict;
        public double BarWidth;      // 负号宽 / 数字中位宽  <- 主判据
        public double DigitHeight;   // 数字中位高
        public double DigitAspect;   // 数字中位宽 / 中位高
        public int DigitCount;
        public bool Measured;        // 有没有量出 BarWidth(false 时上面几个都没意义)
        public string Reason = "";
    }

    public static class MinusCheck
    {
        // ---- 可调的三个数, 都有实测依据, 别随手改 ----
        public const double Threshold = 0.78;      // 见文件头的取舍表
        public const double MinDigitHeight = 60;   // 低于这个不判, 见文件头
        const int GrayDark = 140;                  // 定位金额行时"算深色"的灰度阈

        // 安全阀: 过完尺寸筛之后的连通块超过这个数就直接放弃(见 LocateAmount 里的说明)。
        // ★ 实测 150 张真实账单: 过筛后**中位 6 个, p99 10 个, 最多 17 个**。
        //   两万是这个的一千倍以上, 真图永远碰不到; 只用来挡恶意构造的图。
        public const int MaxComponents = 20_000;

        struct Comp   // 一个连通块
        {
            public int X, Y, W, H, Area;
        }

        /// <summary>
        /// 入口。<paramref name="pixels"/> 是解码后的像素, 每像素 3 字节。
        /// 第 y 行第 x 列的第一个分量在 <c>y*stride + x*3</c>。
        /// </summary>
        /// <param name="order">通道顺序。System.Drawing 出来的要传 <see cref="PixelOrder.Bgr"/>。</param>
        /// <param name="stride">
        /// 每行占的字节数。传 0 表示行与行之间没有填充(即 width*3)。
        /// ★ GDI / System.Drawing 的 BitmapData.Stride 通常**大于** width*3, 必须如实传进来,
        ///   否则整张图会逐行错开, 而且不会报错。
        /// </param>
        public static MinusResult Check(byte[] pixels, int width, int height,
                                        PixelOrder order = PixelOrder.Rgb, int stride = 0)
        {
            if (pixels == null) throw new ArgumentNullException(nameof(pixels));
            if (width <= 0 || height <= 0)
                throw new ArgumentException("宽高必须为正");
            if (stride == 0) stride = width * 3;
            if (stride < width * 3)
                throw new ArgumentException(
                    $"stride {stride} 小于一行需要的 {width * 3} 字节");
            long need = (long)(height - 1) * stride + (long)width * 3;
            if (pixels.Length < need)
                throw new ArgumentException(
                    $"像素数组长度 {pixels.Length} 不够, 按 stride={stride} 需要 {need} 字节");

            var res = new MinusResult { Verdict = MinusVerdict.CannotDetermine };
            if (width < 16 || height < 16) { res.Reason = "图太小"; return res; }

            int ri = order == PixelOrder.Rgb ? 0 : 2;   // 红在一个像素里的第几个字节
            int bi = order == PixelOrder.Rgb ? 2 : 0;   // 蓝在第几个字节

            // ★ 蓝底转账页的金额**不带符号**, 这条判据对它覆盖率为 0, 整个跳过。
            //   不跳的话页面上任何一根横条都可能被当成负号 —— 纯虚报。
            if (IsBluePage(pixels, width, height, stride, ri, bi))
            { res.Reason = "蓝底转账页, 金额不带负号"; return res; }

            var gray = ToGray(pixels, width, height, stride, ri, bi);
            return Check(gray, width, height, res);
        }

        static MinusResult Check(byte[,] gray, int W, int H, MinusResult res)
        {
            // ---- 1. 定位金额行 ----
            var box = LocateAmount(gray, W, H);
            if (box == null) { res.Reason = "定位不到金额行"; return res; }
            int bx0 = box[0], by0 = box[1], bx1 = box[2], by1 = box[3];

            // ---- 2. 在这一行内重新取块 ----
            // ★ 必须重取, 而且**不设高度下限**: 定位那一步有 hh < 0.02*H 的闸,
            //   对 1280 高的图就是 25.6 px 下限, 而负号只有 7 px —— 会被整个滤掉。
            // ★ 左边要大幅外扩: 定位给的 x0 是**只用数字**算的(负号已被那道闸丢掉),
            //   负号落在它左边, pad 小了根本够不着。实测只留 15% 行高的 pad, 03/04 都取不到负号。
            int pad = Math.Max(2, (int)((by1 - by0) * 0.15));
            int padx = Math.Max(pad, (int)((bx1 - bx0) * 0.30));
            int cx0 = Math.Max(0, bx0 - padx), cy0 = Math.Max(0, by0 - pad);
            int cx1 = Math.Min(W, bx1 + pad), cy1 = Math.Min(H, by1 + pad);
            int cw = cx1 - cx0, ch = cy1 - cy0;
            if (cw < 8 || ch < 8) { res.Reason = "金额行裁出来太小"; return res; }

            var sub = new byte[ch, cw];
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                    sub[y, x] = gray[cy0 + y, cx0 + x];

            int otsu = Otsu(sub, ch, cw);
            var fg = new bool[ch, cw];
            long on = 0;
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                { fg[y, x] = sub[y, x] <= otsu; if (fg[y, x]) on++; }
            // 白图浅底深字 —— 前景过半说明极性反了, 翻回来。
            // ★ Python 那边是 `th.mean() > 127`, th 的取值是 0/255, 所以门槛是 127/255 = 0.498039,
            //   **不是 0.5**。这里照抄成整数比较, 免得在那条窄缝里和 Python 判得不一样。
            if (255L * on > 127L * ch * cw)
                for (int y = 0; y < ch; y++)
                    for (int x = 0; x < cw; x++) fg[y, x] = !fg[y, x];

            var glyphs = Label(fg, ch, cw, minArea: 6);   // ★ 不设高度下限
            if (glyphs.Count < 4) { res.Reason = "块数不够"; return res; }

            // ---- 3. 分出数字和横杠 ----
            var hs = glyphs.Select(g => g.H).OrderBy(v => v).ToList();
            double medH = hs[hs.Count / 2];
            if (medH < 20) { res.Reason = "字太小"; return res; }

            var digits = glyphs.Where(g => g.H > 0.75 * medH && g.W < 1.5 * g.H).ToList();
            var bars = glyphs.Where(g => g.W >= 1.5 * g.H && g.H <= 0.45 * medH).ToList();
            if (digits.Count < 4) { res.Reason = "数字不够 4 个"; return res; }
            if (bars.Count == 0) { res.Reason = "没有负号"; return res; }

            // ---- 4. 有效性闸 ----
            // ★ 没有这道闸, **一整张二维码**会混进来并且排在离群榜第一
            //   (实测有一张 bar_width=1.3964 的就是二维码, 它同样满足"4 个近方块 + 1 根横条")。
            //   真金额行的数字**高度几乎完全一致**, 二维码和图标行不会。
            double dhMean = digits.Average(g => (double)g.H);
            double dhStd = Std(digits.Select(g => (double)g.H));
            if (dhStd / dhMean > 0.08) { res.Reason = "数字高度不齐, 不像一行数字"; return res; }
            double dwMean = digits.Average(g => (double)g.W);
            double dwStd = Std(digits.Select(g => (double)g.W));
            if (dwStd / dwMean > 0.30) { res.Reason = "数字宽度差太多"; return res; }

            double mw = Median(digits.Select(g => (double)g.W));
            double mh = Median(digits.Select(g => (double)g.H));
            double ar = mw / mh;
            if (ar < 0.45 || ar > 0.75) { res.Reason = "数字宽高比不对"; return res; }

            var bar = bars.OrderBy(g => g.X).First();   // 最靠左那条 = 负号
            res.Measured = true;
            res.BarWidth = bar.W / mw;
            res.DigitHeight = mh;
            res.DigitAspect = ar;
            res.DigitCount = digits.Count;

            // ---- 5. 判 ----
            // ★★ 小字号一律弃权 —— 误杀主要就是从这里来的, 见文件头
            if (mh < MinDigitHeight)
            {
                res.Verdict = MinusVerdict.CannotDetermine;
                res.Reason = $"数字高 {mh:F0} < {MinDigitHeight}, 判不准所以不判";
                return res;
            }
            res.Verdict = res.BarWidth >= Threshold ? MinusVerdict.Suspicious : MinusVerdict.Ok;
            res.Reason = $"负号宽比 {res.BarWidth:F4} (阈值 {Threshold})";
            return res;
        }

        // ---------- 下面都是工具 ----------

        // 蓝底判定: 上三分之一的三通道均值, 蓝明显压过红和绿
        static bool IsBluePage(byte[] px, int w, int h, int stride, int ri, int bi)
        {
            int h3 = Math.Max(1, h / 3);
            double r = 0, g = 0, b = 0; long n = (long)h3 * w;
            for (int y = 0; y < h3; y++)
            {
                int row = y * stride;
                for (int x = 0; x < w; x++)
                {
                    int o = row + x * 3;
                    r += px[o + ri]; g += px[o + 1]; b += px[o + bi];
                }
            }
            r /= n; g /= n; b /= n;
            return b > r + 25 && b > g + 15;
        }

        // 灰度化。★ 这不是"约等于"OpenCV, 是**逐位相同**。
        //
        //   OpenCV 的 cvtColor(RGB2GRAY) 走的是定点实现, 不是浮点:
        //       (9798*R + 19235*G + 3735*B + 16384) >> 15
        //   实测把 400 万个随机像素跑完, 和 cv2 4.11 **零个不一样**。
        //
        // ★ 别"顺手"改成 0.299/0.587/0.114 的浮点写法, 那样约 0.13% 的像素会差 1 个灰阶;
        //   也别把蓝色系数写成 3736 —— 那是 0.114*32768 四舍五入的值, 而 **OpenCV 用的是截断的 3735**,
        //   差这 1 会让 0.38% 的像素对不上。这一条是查了很久才定下来的。
        //
        // 灰度只在两处用: `< 140` 那道闸和 Otsu。差 1 个灰阶就有可能翻掉一个像素,
        // 所以这里做到逐位一致, 是把整条链上唯一的浮点不确定性去掉。
        static byte[,] ToGray(byte[] px, int w, int h, int stride, int ri, int bi)
        {
            var gray = new byte[h, w];
            for (int y = 0; y < h; y++)
            {
                int row = y * stride;
                for (int x = 0; x < w; x++)
                {
                    int o = row + x * 3;
                    gray[y, x] = (byte)((px[o + ri] * 9798 + px[o + 1] * 19235
                                         + px[o + bi] * 3735 + 16384) >> 15);
                }
            }
            return gray;
        }

        // 找金额行: 只在图的 8%~55% 那一段找, 取"同一行里字最高的那一行"。
        // ★ 早先是按"最高的连通块"选的, 被**商家头像**顶掉了(头像 125~138 px, 金额才 71~78 px),
        //   白图定位率只有 73%。改成按行聚类取中位高最大的一行之后是 98%。
        static int[]? LocateAmount(byte[,] gray, int W, int H)
        {
            int y0b = (int)(H * 0.08), y1b = (int)(H * 0.55);
            int bh = y1b - y0b;
            if (bh < 8) return null;
            var dark = new bool[bh, W];
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < W; x++) dark[y, x] = gray[y0b + y, x] < GrayDark;

            var all = Label(dark, bh, W, minArea: 20);
            var comps = new List<Comp>();
            foreach (var c in all)
            {
                // ★ 这道高度闸是给"找金额数字"用的, 它会把负号滤掉 —— 所以第 2 步要重取
                if (c.H < 0.02 * H || c.H > 0.22 * H || c.W > 0.5 * W) continue;
                comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
            }
            if (comps.Count == 0) return null;

            // ★ 安全阀。下面的按行聚类最坏是 O(N^2): 每个块都自成一行时, 第 N 个块要比 N-1 行。
            //   真实账单这一段里的块数是几百量级, 但**恶意构造的图**(比如一整片 1px 竖条)
            //   能把 N 顶到十万级, 那就是几分钟的 CPU —— 进件服务不能有这种无上限的路径。
            //   超过上限直接当"判不了"返回, **失败方向是安全的**(OCR 照常走)。
            //   (这个上限远高于任何真实截图; Python 那版没有这道闸, 这里是有意加的。)
            if (comps.Count > MaxComponents) return null;

            // 按"同一行"聚类: 纵向重叠 >= 较矮那个高度的一半。
            // 金额的每一位都在同一条基线上, 商家头像不在 —— 全部依据就是这一条。
            // ★ 每行的上下界随加入增量维护, 不要每次都对全行成员求一遍 Min/Max ——
            //   结果完全一样(本来就是成员的 min/max), 但省掉一层 N。
            var rows = new List<List<Comp>>();
            var bounds = new List<(int Y0, int Y1)>();
            foreach (var c in comps.OrderBy(k => k.Y))
            {
                bool placed = false;
                for (int i = 0; i < rows.Count; i++)
                {
                    int ry0 = bounds[i].Y0, ry1 = bounds[i].Y1;
                    if (Math.Min(c.Y + c.H, ry1) - Math.Max(c.Y, ry0)
                        >= 0.5 * Math.Min(c.H, ry1 - ry0))
                    {
                        rows[i].Add(c);
                        bounds[i] = (Math.Min(ry0, c.Y), Math.Max(ry1, c.Y + c.H));
                        placed = true; break;
                    }
                }
                if (!placed)
                {
                    rows.Add(new List<Comp> { c });
                    bounds.Add((c.Y, c.Y + c.H));
                }
            }

            List<Comp>? best = null; double bestMed = -1;
            foreach (var r in rows)
            {
                if (r.Count < 2) continue;                   // 至少要有几位数字
                int x0 = r.Min(k => k.X), x1 = r.Max(k => k.X + k.W);
                if (x1 - x0 < 0.1 * W) continue;             // 太窄的一行不是金额
                var hh = r.Select(k => k.H).OrderBy(v => v).ToList();
                double med = hh[r.Count / 2];                // 用中位高, 一个异常块带不偏
                if (med > bestMed) { bestMed = med; best = r; }
            }
            if (best == null) return null;
            return new[] { best.Min(k => k.X), best.Min(k => k.Y),
                           best.Max(k => k.X + k.W), best.Max(k => k.Y + k.H) };
        }

        // 8 连通标记。用显式栈, 不用递归 —— 大连通块会把调用栈撑爆。
        static List<Comp> Label(bool[,] fg, int h, int w, int minArea)
        {
            var seen = new bool[h, w];
            var outp = new List<Comp>();
            var stack = new Stack<int>();          // 存 y*w+x, 比存元组省内存
            for (int sy = 0; sy < h; sy++)
                for (int sx = 0; sx < w; sx++)
                {
                    if (!fg[sy, sx] || seen[sy, sx]) continue;
                    // ★ 光栅扫描下, 第一次碰到某个连通块的像素一定是它**最上面那一行**的,
                    //   所以 minY 就是 sy, 不用再更新。minX/maxX 不成立, 必须更新。
                    int minX = sx, maxX = sx, maxY = sy, area = 0;
                    stack.Push(sy * w + sx); seen[sy, sx] = true;
                    while (stack.Count > 0)
                    {
                        int v = stack.Pop();
                        int y = v / w, x = v - y * w;
                        area++;
                        if (x < minX) minX = x; else if (x > maxX) maxX = x;
                        if (y > maxY) maxY = y;
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            int ny = y + dy;
                            if (ny < 0 || ny >= h) continue;
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (dy == 0 && dx == 0) continue;
                                int nx = x + dx;
                                if (nx < 0 || nx >= w) continue;
                                if (fg[ny, nx] && !seen[ny, nx])
                                { seen[ny, nx] = true; stack.Push(ny * w + nx); }
                            }
                        }
                    }
                    if (area >= minArea)
                        outp.Add(new Comp { X = minX, Y = sy, W = maxX - minX + 1,
                                            H = maxY - sy + 1, Area = area });
                }
            // ★ 不排序, 保持"从上到下、从左到右第一次碰到"的发现顺序, 结果是确定的。
            //   顺序只在**并列**的时候有影响(行聚类按 Y 排序、挑负号按 X 排序时的并列),
            //   OpenCV 用的是块并行算法, 并列时的编号顺序不保证和这里一样 ——
            //   699 张实测没有因此产生任何差异, 但真要较真的话这是唯一可能的分歧点。
            return outp;
        }

        // Otsu: 直方图最大类间方差
        static int Otsu(byte[,] a, int h, int w)
        {
            var hist = new long[256];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) hist[a[y, x]]++;
            long total = (long)h * w, sum = 0;
            for (int i = 0; i < 256; i++) sum += (long)i * hist[i];
            long sumB = 0, wB = 0; double maxVar = -1; int thr = 0;
            for (int t = 0; t < 256; t++)
            {
                wB += hist[t]; if (wB == 0) continue;
                long wF = total - wB; if (wF == 0) break;
                sumB += (long)t * hist[t];
                double mB = (double)sumB / wB, mF = (double)(sum - sumB) / wF;
                double v = (double)wB * wF * (mB - mF) * (mB - mF);
                if (v > maxVar) { maxVar = v; thr = t; }
            }
            return thr;
        }

        // ★ 必须和 numpy 的 median 一致: **偶数个时取中间两个的平均**。
        //   数字个数经常是偶数(四位、六位金额), 图省事取上中位的话,
        //   算出来的 BarWidth 就和 Python 那版对不上, 自验的 0.7097 也复现不了。
        static double Median(IEnumerable<double> xs)
        {
            var v = xs.OrderBy(k => k).ToList();
            if (v.Count == 0) return 0;
            return v.Count % 2 == 1 ? v[v.Count / 2]
                                    : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2.0;
        }

        // 总体标准差(除以 n), 和 numpy 默认的 ddof=0 一致
        static double Std(IEnumerable<double> xs)
        {
            var v = xs.ToList(); if (v.Count == 0) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Sum(k => (k - m) * (k - m)) / v.Count);
        }
    }
}

// ---------------------------------------------------------------------------
// 用 System.Drawing 解码的现成写法(Windows 上不用装任何包)
//
//   using System.Drawing;
//   using System.Drawing.Imaging;
//
//   static MinusResult CheckFileGdi(string path)
//   {
//       using var bmp = new Bitmap(path);
//       var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
//       var d = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
//       try
//       {
//           var px = new byte[(long)d.Stride * bmp.Height];
//           System.Runtime.InteropServices.Marshal.Copy(d.Scan0, px, 0, px.Length);
//           // ★★ 下面两个参数都不能少, 少了**不会报错**, 只会安静地给错结果:
//           //   1. d.Stride —— GDI 每行会补齐到 4 的倍数。实测本地进件里
//           //      **33.8% 的宽度不是 4 的倍数**(1179 / 1290 / 1170 / 1206 这些),
//           //      不传 stride 的话整张图**逐行斜着错开**, 量出来全是垃圾。
//           //   2. PixelOrder.Bgr —— Format24bppRgb 名字叫 Rgb,
//           //      **内存里其实是 BGR**。传错的话蓝底页会被当成白底页,
//           //      蓝图那道闸就废了(蓝图是误杀最大的来源, 见文件头)。
//           return MinusCheck.Check(px, bmp.Width, bmp.Height, PixelOrder.Bgr, d.Stride);
//       }
//       finally { bmp.UnlockBits(d); }
//   }
//
// ★ 用别的解码器也一样: 只要拿到"每像素 3 字节"的缓冲区, 把 宽、高、通道顺序、行跨距
//   如实传进来就行。行与行之间没有填充就传 stride = 0。
// ---------------------------------------------------------------------------
