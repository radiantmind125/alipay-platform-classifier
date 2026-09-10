#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum DotVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // 小数点是圆的
    }

    public sealed class DotResult
    {
        public DotVerdict Verdict;
        public double Fill;          // 小数点面积 / 它的外接框面积。实心方点 = 1.0, 圆点 = 0.785
        public int DotWidth;         // 小数点外接框宽
        public int DotHeight;        // 小数点外接框高
        public double DotArea;       // 小数点的前景像素数
        public double DigitHeight;   // 金额数字中位高, 用来判断字号够不够大
        public bool MerchantPage;    // 左上角是关闭图标的商家账单页
        /// <summary>
        /// 蓝底转账页那道闸是否跑过。入参是单通道(已转好的灰度图)时它需要颜色, 跑不了, 这里为 false。
        /// 喂灰度是合法用法(省一次整帧转换), 但要知道自己少了一道闸: 实测约 0.5% 的图判定会变。
        /// </summary>
        public bool BlueGateRan;
        public bool Rephotograph;    // 疑似翻拍
        public bool Measured;        // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 白底账单详情页的**小数点形状**检查。
    ///
    /// 支付宝金额字体的小数点是一个**实心方块**, 填充率(面积/外接框面积) 1.000。
    /// 换字体重打时大多数字体的句点是**圆的**, 圆在方框里只占 pi/4 = 0.785。
    /// 真图 png 最低 0.9121, 圆点最高 0.8947, 两侧不重叠, 阈值取 0.90。
    ///
    /// ★ 判的是**形状不是大小**。上一版量"面积 / 数字高的平方", 因为一个圆点和一个更小的
    ///   方点面积可以相同, 拿真实圆点图去测只抓到 10%, 已废弃。
    ///
    /// 三种情况不判, 都是实测定下来的:
    ///   - **有损图(JPEG)**: 压缩会把方点的角压圆, 真图 jpg 最低到 0.7059, 和真圆点分不开。
    ///   - **翻拍图**: 拍屏幕会把角拍糊, 翻拍里 29% 填充率低于 0.90, 干净截图只有 0.14%。
    ///   - **商家账单页**: 那个页型金额字体和详情页不一样。
    ///
    /// 这条和负号那条抓的不是同一种手法: 只改负号的那种它看不见, 两条都响的基本是整个金额重打过。
    ///
    /// 入参只读, 内部全部用 ROI 视图, 不复制整图, 不碰 System.Drawing。
    /// 唯一的整帧分配是 BGR 转灰度; 单通道入参时连这次也没有。无静态可变状态, 可多线程调用。
    ///
    /// ★ 和 <see cref="MinusCheck"/> 同时调用会把定位金额行那套活做两遍(耗时约翻倍)。
    ///   调用方自己转一次灰度再喂两个类可以省掉一部分, 但那样蓝底页那道闸就跑不了, 见 BlueGateRan。
    /// </summary>
    public static class DotCheck
    {
        /// <summary>填充率低于这个值判为圆点。真图 png 最小 0.9121, 圆点最大 0.8947, 两侧不重叠。</summary>
        public const double FillLow = 0.90;

        /// <summary>数字低于这个高度不判; 字太小的时候小数点只有几个像素, 量不准。</summary>
        public const double MinDigitHeight = 60;

        const int GrayDark = 140;                  // 定位金额行时的深色阈值
        public const int MaxComponents = 20_000;   // 上限保护, 见 LocateAmount

        struct Comp
        {
            public int X, Y, W, H, Area;
        }

        /// <param name="image">
        /// 8 位或 16 位图, 1 / 3 / 4 通道均可; 16 位会先降成 8 位。
        /// 位深或通道数不支持时返回 CannotDetermine, **不抛异常**。
        /// ★ 多通道必须是 **BGR(A)** 序(Cv2.ImRead / ImDecode 就是)。Mat 不携带通道序信息,
        ///   喂 RGB 不报错也不抛异常, 只会让结果悄悄漂 —— 实测约 2% 的图判定不同。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        /// <param name="lossless">
        /// 源图是不是无损格式。png 传 true, jpg/jpeg/webp 传 false。
        /// 依据是**源文件的容器格式**, 不是 Mat 的内容 —— Mat 解码后不带格式信息, 只能由调用方传。
        ///
        /// ★ **故意不给默认值。** 传 false 时这条判据整个不工作: 会在赋 Verdict 之前就返回,
        ///   任何输入都是 CannotDetermine, 不抛异常也不打日志。给了默认值的话, 调用点照着
        ///   MinusCheck.Check(mat) 写成 DotCheck.Check(mat) 会编译通过而判据静默全关。
        /// </param>
        public static DotResult Check(Mat image, bool lossless)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new DotResult { Verdict = DotVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }

            // 这两种是数据问题不是编程错误, 批量跑图时返回"不判"而不是中断流程
            int cn = image.Channels();
            if (cn != 1 && cn != 3 && cn != 4)
            { res.Reason = $"不支持 {cn} 通道, 不判"; return res; }
            bool is8 = image.Depth() == MatType.CV_8U;
            if (!is8 && image.Depth() != MatType.CV_16U)
            { res.Reason = "位深不是 8 位或 16 位, 不判"; return res; }

            int W = image.Width, H = image.Height;
            if (W < 16 || H < 16) { res.Reason = "图太小"; return res; }

            res.Rephotograph = IsRephotograph(W, H);

            Mat? owned8 = null;
            Mat? owned = null;
            try
            {
                // 16 位按 1/256 降到 8 位。真实图池里约 3% 是 16 位 PNG, 标定时读图也是降位读的,
                // 所以这里必须降位, 不能跳过更不能抛异常。与 OpenCV 自己的降位差最多 1 个灰阶。
                Mat src = image;
                if (!is8)
                {
                    owned8 = new Mat();
                    image.ConvertTo(owned8, MatType.CV_8U, 1.0 / 256.0);
                    src = owned8;
                }

                // 蓝底转账页金额排版不同
                // ★ 必须排在降位之后: 这里用的是 8 位绝对阈值, 16 位上会失效
                res.BlueGateRan = cn >= 3;
                if (res.BlueGateRan && IsBluePage(src))
                { res.Reason = "蓝底转账页, 不判"; return res; }

                Mat gray;
                if (cn == 1) gray = src;
                else
                {
                    owned = new Mat();
                    Cv2.CvtColor(src, owned,
                        cn == 4 ? ColorConversionCodes.BGRA2GRAY : ColorConversionCodes.BGR2GRAY);
                    gray = owned;
                }

                if (IsMerchantPage(gray))
                {
                    res.MerchantPage = true;
                    res.Reason = "商家账单页, 金额字体和详情页不一样, 不判";
                    return res;
                }
                return Check(gray, res, lossless);
            }
            finally { owned?.Dispose(); owned8?.Dispose(); }
        }

        static DotResult Check(Mat gray, DotResult res, bool lossless)
        {
            int W = gray.Width, H = gray.Height;

            var box = LocateAmount(gray);
            if (box == null) { res.Reason = "定位不到金额行"; return res; }
            int bx0 = box.Value.Left, by0 = box.Value.Top;
            int bx1 = box.Value.Right, by1 = box.Value.Bottom;

            int pad = Math.Max(2, (int)((by1 - by0) * 0.15));
            int padx = Math.Max(pad, (int)((bx1 - bx0) * 0.30));
            int cx0 = Math.Max(0, bx0 - padx), cy0 = Math.Max(0, by0 - pad);
            int cx1 = Math.Min(W, bx1 + pad), cy1 = Math.Min(H, by1 + pad);
            int cw = cx1 - cx0, ch = cy1 - cy0;
            if (cw < 8 || ch < 8) { res.Reason = "金额行裁出来太小"; return res; }

            using var sub = new Mat(gray, new Rect(cx0, cy0, cw, ch));
            using var fg = new Mat();
            Cv2.Threshold(sub, fg, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
            long on = Cv2.CountNonZero(fg);
            if (255L * on > 127L * ch * cw) Cv2.BitwiseNot(fg, fg);

            var glyphs = Label(fg, minArea: 6);
            if (glyphs.Count < 4) { res.Reason = "块数不够"; return res; }

            var hs = glyphs.Select(g => g.H).OrderBy(v => v).ToList();
            double medH = hs[hs.Count / 2];
            if (medH < 20) { res.Reason = "字太小"; return res; }

            var digits = glyphs.Where(g => g.H > 0.75 * medH && g.W < 1.5 * g.H).ToList();
            var bars = glyphs.Where(g => g.W >= 1.5 * g.H && g.H <= 0.45 * medH).ToList();
            if (digits.Count < 4) { res.Reason = "数字不够 4 个"; return res; }

            // ★ 贴着切块左右边缘的块是**被切断的**, 尺寸不可信, 先剔掉。
            //   实测三张 -499.99 的首块 X=0、高只有其余的 0.857, 会被下面的 ¥ 判据误伤;
            //   真正的 ¥ 和真图首字都离边缘很远。不剔掉还会让 n_digit 多算一位。
            digits = digits.Where(g => g.X > 0 && g.X + g.W < cw).ToList();
            if (digits.Count < 4) { res.Reason = "去掉贴边的块之后数字不够"; return res; }

            // ★ 把 ¥ 从数字里剔出去。留着它会把 mw 拉宽, 进而压低负号宽比 —— 造成漏判。
            //   ¥ 和数字**一样高**, 所以按高度判会失效; 而且它不止一种字形, 要三选一:
            //     又宽又稀(宽比 1.41~1.62) / 明显偏矮(0.901) / 后面留大空(1.021)
            //   对照真图首字: 宽比 0.55~1.00, 高比 0.98~1.00, 间隔 0.23~0.57, 三条都不沾。
            digits = digits.OrderBy(g => g.X).ToList();
            if (digits.Count >= 5)
            {
                var rest = digits.Skip(1).ToList();
                double restW = Median(rest.Select(g => (double)g.W));
                double restH = Median(rest.Select(g => (double)g.H));
                double restFill = Median(rest.Select(g => g.Area / (double)(g.W * g.H)));
                var d0 = digits[0];
                double f0 = d0.Area / (double)(d0.W * d0.H);
                if (restW > 0 && restH > 0 && restFill > 0)
                {
                    double gap0 = (rest[0].X - (d0.X + d0.W)) / restW;
                    bool wideSparse = d0.W > 1.25 * restW && f0 < 0.85 * restFill;
                    bool shortGlyph = d0.H < 0.95 * restH;
                    // 宽度条件防误伤: 金额如 -1.50 首字后面就是小数点, 间隔天生大,
                    // 但那种情况首字是窄的 "1"
                    bool wideGap = gap0 > 0.85 && d0.W > 0.95 * restW;
                    if (wideSparse || shortGlyph || wideGap) digits = rest;
                }
            }
            if (digits.Count < 4) { res.Reason = "去掉 ¥ 之后数字不够"; return res; }

            // ★ 必须有负号才判。这不是为了量负号, 是为了**把判定限制在标过阈值的那批图上** ——
            //   0.90 这个阈值和它背后的样本全部来自带负号的图, 没有负号的是收款那一类,
            //   金额字体从来没量过。去掉这道闸会多放进约 4% 没标定过的图。
            //   (不同页型的金额字体确实不一样, 是量出来的, 不是假设。)
            if (bars.Count == 0) { res.Reason = "没有负号, 这类图的阈值没标过, 不判"; return res; }

            // 与 MinusCheck 相同的有效性闸: 挡掉定位跑偏、把二维码当成金额行的情况
            double dhMean = digits.Average(g => (double)g.H);
            double dhStd = Std(digits.Select(g => (double)g.H));
            if (dhStd / dhMean > 0.08) { res.Reason = "数字高度不齐, 不像一行数字"; return res; }
            double dwMean = digits.Average(g => (double)g.W);
            double dwStd = Std(digits.Select(g => (double)g.W));
            if (dwStd / dwMean > 0.30) { res.Reason = "数字宽度差太多"; return res; }

            double mw = Median(digits.Select(g => (double)g.W));
            double mh = Median(digits.Select(g => (double)g.H));
            if (mw / mh < 0.45 || mw / mh > 0.75) { res.Reason = "数字宽高比不对"; return res; }
            res.DigitHeight = mh;

            // 小数点: 既不是数字也不是横条, 又小又坐在数字基线上。
            // 必须恰好一个 —— 多于一个说明切块里有噪点, 那时挑出来的可能根本不是小数点。
            double baseline = Median(digits.Select(g => (double)(g.Y + g.H)));
            var dots = glyphs.Where(g => !digits.Contains(g) && !bars.Contains(g)
                                      && g.H <= 0.30 * mh && g.W <= 0.60 * mw
                                      && Math.Abs((g.Y + g.H) - baseline) <= 0.06 * mh).ToList();
            if (dots.Count != 1)
            { res.Reason = dots.Count == 0 ? "找不到小数点" : $"小数点候选有 {dots.Count} 个, 判不了"; return res; }

            var d = dots[0];
            if (d.W <= 0 || d.H <= 0) { res.Reason = "小数点尺寸异常"; return res; }
            res.Measured = true;
            res.DotWidth = d.W;
            res.DotHeight = d.H;
            res.DotArea = d.Area;
            res.Fill = d.Area / (double)(d.W * d.H);

            if (mh < MinDigitHeight)
            {
                res.Reason = $"数字高 {mh:F0} < {MinDigitHeight}, 小数点太小量不准, 不判";
                return res;
            }
            if (!lossless)
            {
                res.Reason = $"填充率 {res.Fill:F4}, 但不是无损图, 不判";
                return res;
            }
            if (res.Rephotograph)
            {
                res.Reason = $"填充率 {res.Fill:F4}, 但疑似翻拍, 不判";
                return res;
            }

            if (res.Fill < FillLow)
            {
                res.Verdict = DotVerdict.Suspicious;
                res.Reason = $"小数点是圆的, 填充率 {res.Fill:F4} (实心方点应为 1.0, 阈值 {FillLow})";
            }
            else
            {
                res.Verdict = DotVerdict.Ok;
                res.Reason = $"小数点填充率 {res.Fill:F4}";
            }
            return res;
        }

        // 拿相机拍屏幕会把方点的角拍糊, 翻拍里 29% 填充率低于 0.90, 干净截图只有 0.14%
        static bool IsRephotograph(int w, int h)
        {
            long px = (long)w * h;
            double ar = Math.Max(w, h) / (double)Math.Min(w, h);
            return px >= 6_000_000 || ar < 1.7;
        }

        // 左上角近正方形 = X 关闭图标 = 商家账单页; 返回箭头是 1.7 比 1
        static bool IsMerchantPage(Mat gray)
        {
            int W = gray.Width, H = gray.Height;
            int cw = (int)(W * 0.13), ch = (int)(H * 0.12);
            if (cw < 16 || ch < 16) return false;
            using var corner = new Mat(gray, new Rect(0, 0, cw, ch));
            using var dark = new Mat();
            Cv2.Threshold(corner, dark, 169, 255, ThresholdTypes.BinaryInv);
            using var labels = new Mat();
            using var stats = new Mat();
            using var cent = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(dark, labels, stats, cent,
                                                     PixelConnectivity.Connectivity8, MatType.CV_32S);
            int bestY = -1, bw = 0, bh = 0;
            for (int i = 1; i < n; i++)
            {
                int a = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                int w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
                int h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
                if (a < 100 || h < 25 || h > 80 || w < 15 || w > 80) continue;
                int y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
                if (y > bestY) { bestY = y; bw = w; bh = h; }
            }
            if (bestY < 0 || bw == 0) return false;
            double ar = bh / (double)bw;
            return ar >= 0.85 && ar <= 1.15;
        }

        static bool IsBluePage(Mat bgr)
        {
            int h3 = Math.Max(1, bgr.Height / 3);
            using var top = new Mat(bgr, new Rect(0, 0, bgr.Width, h3));
            var m = Cv2.Mean(top);            // BGR: Val0=B, Val1=G, Val2=R
            return m.Val0 > m.Val2 + 25 && m.Val0 > m.Val1 + 15;
        }

        // 在图像 8%~55% 高度范围内, 取同一行中字最高的那一行
        static Rect? LocateAmount(Mat gray)
        {
            int W = gray.Width, H = gray.Height;
            int y0b = (int)(H * 0.08), y1b = (int)(H * 0.55);
            int bh = y1b - y0b;
            if (bh < 8) return null;

            using var band = new Mat(gray, new Rect(0, y0b, W, bh));
            using var dark = new Mat();
            // 判据是 gray < 140, 即 <= 139; BinaryInv 的判据是 src <= thresh
            Cv2.Threshold(band, dark, GrayDark - 1, 255, ThresholdTypes.BinaryInv);

            var comps = new List<Comp>();
            foreach (var c in Label(dark, minArea: 20))
            {
                if (c.H < 0.02 * H || c.H > 0.22 * H || c.W > 0.5 * W) continue;
                comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
            }
            if (comps.Count == 0 || comps.Count > MaxComponents) return null;

            var rows = new List<List<Comp>>();
            var bounds = new List<(int Y0, int Y1)>();
            // 排序取全序: 连通域的返回顺序依实现而定, 并列时不能靠它决定
            foreach (var c in comps.OrderBy(k => k.Y).ThenBy(k => k.X))
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
                if (r.Count < 2) continue;
                int x0 = r.Min(k => k.X), x1 = r.Max(k => k.X + k.W);
                if (x1 - x0 < 0.1 * W) continue;
                var hh = r.Select(k => k.H).OrderBy(v => v).ToList();
                double med = hh[r.Count / 2];
                if (med > bestMed) { bestMed = med; best = r; }
            }
            if (best == null) return null;
            return Rect.FromLTRB(best.Min(k => k.X), best.Min(k => k.Y),
                                 best.Max(k => k.X + k.W), best.Max(k => k.Y + k.H));
        }

        static List<Comp> Label(Mat bin, int minArea)
        {
            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(bin, labels, stats, centroids,
                                                     PixelConnectivity.Connectivity8, MatType.CV_32S);
            var outp = new List<Comp>(Math.Max(0, n - 1));
            for (int i = 1; i < n; i++)      // 0 是背景
            {
                int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                // 这句不能省: OpenCV 自己不做面积过滤。实测 600 个金额裁块里有 4 个(0.7%)
                // 会多出最多 53 个碎块, 它们会把 medH 带偏, 进而搞乱数字和横条的分类。
                if (area < minArea) continue;
                outp.Add(new Comp
                {
                    X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                    Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                    W = stats.At<int>(i, (int)ConnectedComponentsTypes.Width),
                    H = stats.At<int>(i, (int)ConnectedComponentsTypes.Height),
                    Area = area,
                });
            }
            return outp;
        }

        static double Median(IEnumerable<double> xs)
        {
            var v = xs.OrderBy(k => k).ToList();
            if (v.Count == 0) return 0;
            return v.Count % 2 == 1 ? v[v.Count / 2]
                                    : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2.0;
        }

        static double Std(IEnumerable<double> xs)
        {
            var v = xs.ToList(); if (v.Count == 0) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Sum(k => (k - m) * (k - m)) / v.Count);
        }
    }
}
