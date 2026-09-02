// 负号几何检查 —— .NET 版, 不用模型, 不用显卡, 只做算术。
//
// ★ 这个文件**不依赖任何第三方包**, 只用 System.*, 直接拖进工程就能编。
//   入口收的是**解码好的 RGB 字节数组**, 你现在用什么解码器都行, 不用为它装新包。
//   (如果需要一个现成的读文件封装, 见同目录 MinusCheckLoader.cs, 那个是可选的。)
//
// 干什么
// ------
// 白底账单详情页里, 量**金额前面那个负号的宽度 / 数字的中位宽度**。
// 真图这个比值很集中(五位数金额四千多张实测中位 0.70), 被拉长过的会明显偏出去。
//
// 为什么阈值是 0.78 而不是更高
// ---------------------------
// 按"负号加 6 个像素"这个真实攻击尺度反推的。在主力档(数字高 60~90 px, 占进件 92%)实测:
//
//   阈值 0.76  真图报出 20/万   加6px 能报出 100%
//   阈值 0.78  真图报出 14/万   加6px 能报出 100%   <- 用这个
//   阈值 0.85  真图报出 10/万   加6px 能报出  32%
//   阈值 0.90  真图报出 10/万   加6px 能报出   0%   <- 太高, 一张都抓不到
//
// ★★★ 误杀大多是这两条漏掉造成的
// ------------------------------
// **(一) 数字太小的图必须弃权。** 同样的阈值, 报出率随字号差 15~30 倍:
//
//   数字高 60~90 px(占 92%)   真图报出 **14/万**
//   数字高 40~60 px           真图报出 **200~400/万**
//
//   小字号上真图自己就散(p99 到 1.18~1.78), 判了既误报高又抓不到。
//   **所以数字高 < 60 px 一律返回 CannotDetermine, 不给结论。**
//
// **(二) 蓝底转账页要整个跳过。** 蓝图金额**不带负号**, 这条线对蓝图覆盖率是 0。
//   不跳过的话, 页面上任何一根横条都可能被当成负号量一遍。
//   实测: 60 张蓝图里有 1 张能被"量出"一个 bar_width —— 那 1 张纯属虚报。
//
// 如果你那版误杀偏大, 先对一下这两条。
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
// ★ 对 **BarWidth**, 别对 Verdict —— 那两张数字高都 < 60,
//   按上面那道闸会返回 CannotDetermine。要验的是量得准不准, 不是判得对不对。
//
// 这一版和 Python 那版对过账
// --------------------------
// 随机抽 699 张白单, 两边逐张比 `bar_width`:
//   **两边都量得出的 605 张, 605 张数值完全相同(差 < 5e-5), 一张不差。**
//   量得出/量不出的判断也完全一致 —— 没有"这边能量那边不能"的情况。
//   报出的张数一样(各 1 张)。
// 速度: 单线程 CPU **约 38 ms 一张**(含 JPEG/PNG 解码), 不需要显卡。
//
// ★ 唯一一处**故意和 Python 不一样**: 这版**开头就把蓝底页整个跳掉**,
//   Python 那版没跳, 靠"蓝图上找不到横杠"自然落空。实测 60 张蓝图里
//   有 1 张真被它量出了一个 bar_width —— 那是纯虚报。这版不会。

#nullable enable      // 显式打开, 这样不管你工程开没开 nullable, 编译都是零警告

using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssp
{
    public enum MinusVerdict
    {
        Ok,                 // 负号正常
        Suspicious,         // 负号偏长 -> 金额输出空
        CannotDetermine     // 判不了 -> OCR 照常
    }

    /// <summary>像素在字节数组里的通道顺序。</summary>
    public enum PixelOrder
    {
        Rgb,   // ImageSharp Rgb24 / SkiaSharp Rgba8888
        Bgr    // ★ System.Drawing 的 Format24bppRgb 内存里其实是 BGR, 别选错
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
        public const double MinDigitHeight = 60;   // 低于这个不判(误杀主要来源)
        const int GrayDark = 140;                  // 定位金额行时"算深色"的灰度阈

        struct Comp   // 一个连通块
        {
            public int X, Y, W, H, Area;
        }

        /// <summary>
        /// 入口。<paramref name="pixels"/> 是**解码后紧密排列**的像素, 每像素 3 字节,
        /// 按行优先(第 y 行第 x 列的红色分量在 (y*width + x)*3)。
        /// 长度必须正好 width*height*3。
        /// </summary>
        /// <param name="order">通道顺序。System.Drawing 出来的要传 <see cref="PixelOrder.Bgr"/>。</param>
        public static MinusResult Check(byte[] pixels, int width, int height,
                                        PixelOrder order = PixelOrder.Rgb)
        {
            if (pixels == null) throw new ArgumentNullException(nameof(pixels));
            if (width <= 0 || height <= 0)
                throw new ArgumentException("宽高必须为正");
            if ((long)pixels.Length < (long)width * height * 3)
                throw new ArgumentException(
                    $"像素数组长度 {pixels.Length} 不够, 需要 {(long)width * height * 3} " +
                    "(每像素 3 字节, 行与行之间不能有 padding)");

            var res = new MinusResult { Verdict = MinusVerdict.CannotDetermine };
            if (width < 16 || height < 16) { res.Reason = "图太小"; return res; }

            int ri = order == PixelOrder.Rgb ? 0 : 2;   // 红在第几个字节
            int bi = order == PixelOrder.Rgb ? 2 : 0;   // 蓝在第几个字节

            // ★ 蓝底转账页的金额**不带符号**, 这条判据对它覆盖率为 0, 整个跳过。
            //   不跳的话页面上任何一根横条都可能被当成负号 —— 纯虚报。
            if (IsBluePage(pixels, width, height, ri, bi))
            { res.Reason = "蓝底转账页, 金额不带负号"; return res; }

            var gray = ToGray(pixels, width, height, ri, bi);
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
            // 白图浅底深字 —— 前景过半说明极性反了, 翻回来
            if (on * 2 > (long)ch * cw)
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
        static bool IsBluePage(byte[] px, int w, int h, int ri, int bi)
        {
            int h3 = Math.Max(1, h / 3);
            double r = 0, g = 0, b = 0; long n = (long)h3 * w;
            for (int y = 0; y < h3; y++)
            {
                int row = y * w * 3;
                for (int x = 0; x < w; x++)
                {
                    int o = row + x * 3;
                    r += px[o + ri]; g += px[o + 1]; b += px[o + bi];
                }
            }
            r /= n; g /= n; b /= n;
            return b > r + 25 && b > g + 15;
        }

        // ★ 系数必须和 OpenCV 的 RGB2GRAY 一致, 换了系数量出来的值就对不上上面那两个自验数
        static byte[,] ToGray(byte[] px, int w, int h, int ri, int bi)
        {
            var gray = new byte[h, w];
            for (int y = 0; y < h; y++)
            {
                int row = y * w * 3;
                for (int x = 0; x < w; x++)
                {
                    int o = row + x * 3;
                    gray[y, x] = (byte)Math.Round(
                        0.299 * px[o + ri] + 0.587 * px[o + 1] + 0.114 * px[o + bi]);
                }
            }
            return gray;
        }

        // 找金额行: 只在图的 8%~55% 那一段找, 取"同一行里字最高的那一行"
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

            // 按"同一行"聚类: 纵向重叠 >= 较矮那个高度的一半。
            // 金额的每一位都在同一条基线上, 商家头像不在 —— 全部依据就是这一条。
            var rows = new List<List<Comp>>();
            foreach (var c in comps.OrderBy(k => k.Y))
            {
                bool placed = false;
                foreach (var r in rows)
                {
                    int ry0 = r.Min(k => k.Y), ry1 = r.Max(k => k.Y + k.H);
                    if (Math.Min(c.Y + c.H, ry1) - Math.Max(c.Y, ry0)
                        >= 0.5 * Math.Min(c.H, ry1 - ry0))
                    { r.Add(c); placed = true; break; }
                }
                if (!placed) rows.Add(new List<Comp> { c });
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
            var stack = new Stack<int>();          // 存 y*w+x, 比存元组省一半内存
            for (int sy = 0; sy < h; sy++)
                for (int sx = 0; sx < w; sx++)
                {
                    if (!fg[sy, sx] || seen[sy, sx]) continue;
                    int minX = sx, maxX = sx, minY = sy, maxY = sy, area = 0;
                    stack.Push(sy * w + sx); seen[sy, sx] = true;
                    while (stack.Count > 0)
                    {
                        int v = stack.Pop();
                        int y = v / w, x = v - y * w;
                        area++;
                        if (x < minX) minX = x; else if (x > maxX) maxX = x;
                        if (y > maxY) maxY = y;              // 扫描顺序保证 y >= minY
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
                        outp.Add(new Comp { X = minX, Y = minY, W = maxX - minX + 1,
                                            H = maxY - minY + 1, Area = area });
                }
            // ★ 不排序, 保持"从上到下、从左到右第一次碰到"的发现顺序 ——
            //   和 OpenCV 的标号顺序一致。行聚类里按 Y 排序遇到并列时, 顺序会影响归到哪一行。
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
