#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum MinusVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // 负号偏长
    }

    public sealed class MinusResult
    {
        public MinusVerdict Verdict;
        public double BarWidth;      // 负号宽 / 数字中位宽
        public double DigitHeight;   // 数字中位高
        public double DigitAspect;   // 数字中位宽 / 中位高
        public int DigitCount;
        public bool MerchantPage;    // 左上角是 X 关闭图标 = 商家账单页, 字体不一样, 不判
        /// <summary>
        /// 蓝底转账页那道闸是否跑过。入参是单通道(已转好的灰度图)时它需要颜色, 跑不了, 这里为 false。
        /// 喂灰度是合法用法(省一次整帧转换), 但要知道自己少了一道闸: 实测约 0.5% 的图判定会变。
        /// </summary>
        public bool BlueGateRan;
        public bool Measured;        // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 白底账单详情页的**负号**字形检查: 负号宽 / 数字中位宽, 同一张图内部的比值, 不查任何表。
    /// 小数点那条在 <see cref="DotCheck"/>, 这里不管。
    ///
    /// 参照物(负号)是**恒定形状**: 不管什么机型、什么金额、哪个用户它都是同一个字形。
    /// 这是这条判据不用查表、不挑机型、不随改版失效的原因。
    ///
    /// 入参只读, 内部全部用 ROI 视图, 不复制整图, 不碰 System.Drawing。
    /// 唯一的整帧分配是 BGR 转灰度; 单通道入参时连这次也没有。
    /// 无静态可变状态, 可多线程调用。
    ///
    /// ★ 和 <see cref="DotCheck"/> 同时调用会把定位金额行那套活做两遍(耗时约翻倍)。
    ///   调用方自己转一次灰度再喂两个类可以省掉一部分, 但那样蓝底页那道闸就跑不了,
    ///   见 <see cref="MinusResult.BlueGateRan"/>。
    /// </summary>
    public static class MinusCheck
    {
        public const double Threshold = 0.78;

        /// <summary>
        /// 负号偏窄的下界。换字体重打时负号往往偏窄, 高侧只抓偏宽的, 低侧补上这一半。
        /// 真图负号宽比中位 0.70, 0.625 在 p1 以下。
        /// </summary>
        public const double ThresholdLow = 0.625;

        public const double MinDigitHeight = 60;   // 数字低于此高度不判定; 设为 0 可关闭
        const int GrayDark = 140;                  // 定位金额行时的深色阈值

        public const int MaxComponents = 20_000;   // 上限保护, 见 LocateAmount

        struct Comp
        {
            public int X, Y, W, H, Area;
        }

        /// <summary>
        /// 左上角的图标是不是近正方形。是的话按**商家账单页**处理(左上角 X 关闭图标),
        /// 不是账单详情页(左上角 &lt; 返回箭头, 高宽比约 1.7)。
        ///
        /// ★ 这两种页的金额字体不一样, 商家账单页的负号天生更宽, 不跳过就会大量误报。
        ///   跳过的代价是约 0.5% 的图不判。
        ///   已知不足: 顶部有通知横幅或状态栏图标时, 这里可能认错, 把普通详情页也跳过。
        /// </summary>
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
                if (y > bestY) { bestY = y; bw = w; bh = h; }   // 取最靠下的
            }
            if (bestY < 0 || bw == 0) return false;
            double ar = bh / (double)bw;
            return ar >= 0.85 && ar <= 1.15;      // 近正方形 = X 关闭图标
        }

        /// <param name="image">
        /// 8 位或 16 位图, 1 / 3 / 4 通道均可; 16 位会先降成 8 位。
        /// 位深或通道数不支持时返回 CannotDetermine, **不抛异常**。
        /// ★ 多通道必须是 **BGR(A)** 序(Cv2.ImRead / ImDecode 就是)。Mat 不携带通道序信息,
        ///   喂 RGB 不报错也不抛异常, 只会让结果悄悄漂 —— 实测约 2% 的图判定不同。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        public static MinusResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new MinusResult { Verdict = MinusVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }

            // 这两种是数据问题不是编程错误, 批量跑图时返回"不判"而不是中断流程
            int cn = image.Channels();
            if (cn != 1 && cn != 3 && cn != 4)
            { res.Reason = $"不支持 {cn} 通道, 不判"; return res; }
            bool is8 = image.Depth() == MatType.CV_8U;
            if (!is8 && image.Depth() != MatType.CV_16U)
            { res.Reason = "位深不是 8 位或 16 位, 不判"; return res; }
            if (image.Width < 16 || image.Height < 16) { res.Reason = "图太小"; return res; }

            // 单通道 8 位时直接用入参, 不复制也不释放; 其余情况才转换出自己的图
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

                // 蓝底转账页金额不带符号, 不处理
                // ★ 必须排在降位之后: 这里用的是 8 位绝对阈值, 16 位上会失效
                res.BlueGateRan = cn >= 3;
                if (res.BlueGateRan && IsBluePage(src))
                { res.Reason = "蓝底转账页, 金额不带负号"; return res; }

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
                    res.Reason = "商家账单页(左上角是关闭图标), 金额字体和详情页不一样, 不判";
                    return res;
                }
                return Check(gray, res);
            }
            finally { owned?.Dispose(); owned8?.Dispose(); }
        }

        static MinusResult Check(Mat gray, MinusResult res)
        {
            int W = gray.Width, H = gray.Height;

            var box = LocateAmount(gray);
            if (box == null) { res.Reason = "定位不到金额行"; return res; }
            int bx0 = box.Value.Left, by0 = box.Value.Top;
            int bx1 = box.Value.Right, by1 = box.Value.Bottom;

            // 行内重新取块, 不设高度下限: 定位阶段的高度过滤会把负号滤掉。
            // 左侧外扩较多, 因为定位框只覆盖数字, 负号落在框外。
            int pad = Math.Max(2, (int)((by1 - by0) * 0.15));
            int padx = Math.Max(pad, (int)((bx1 - bx0) * 0.30));
            int cx0 = Math.Max(0, bx0 - padx), cy0 = Math.Max(0, by0 - pad);
            int cx1 = Math.Min(W, bx1 + pad), cy1 = Math.Min(H, by1 + pad);
            int cw = cx1 - cx0, ch = cy1 - cy0;
            if (cw < 8 || ch < 8) { res.Reason = "金额行裁出来太小"; return res; }

            // ROI 是视图, 不复制
            using var sub = new Mat(gray, new Rect(cx0, cy0, cw, ch));
            using var fg = new Mat();
            // BinaryInv 得到 src <= otsu 的前景, 与手写版的判据一致
            Cv2.Threshold(sub, fg, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

            // 前景过半说明极性相反, 取反; 阈值为 127/255
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
            if (bars.Count == 0) { res.Reason = "没有负号"; return res; }

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

            // 有效性检查: 金额行的数字高度基本一致, 二维码等区域不满足
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

            // 排序取全序: 连通域的返回顺序依实现而定(cv2 按 2x2 块光栅序), 并列时不能靠它决定
            var bar = bars.OrderBy(g => g.X).ThenBy(g => g.Y).ThenBy(g => g.W).First();
            res.Measured = true;
            res.BarWidth = bar.W / mw;
            res.DigitHeight = mh;
            res.DigitAspect = ar;
            res.DigitCount = digits.Count;

            if (mh < MinDigitHeight)
            {
                res.Verdict = MinusVerdict.CannotDetermine;
                res.Reason = $"数字高 {mh:F0} < {MinDigitHeight}, 判不准所以不判";
                return res;
            }

            // 保持已上线的高侧阈值 0.78 不动, 低侧 0.625 是纯增量, 不会丢掉现在能抓的。
            if (res.BarWidth >= Threshold || res.BarWidth < ThresholdLow)
            {
                res.Verdict = MinusVerdict.Suspicious;
                res.Reason = $"负号宽比 {res.BarWidth:F4} (正常 {ThresholdLow}~{Threshold})";
            }
            else
            {
                res.Verdict = MinusVerdict.Ok;
                res.Reason = $"负号宽比 {res.BarWidth:F4}";
            }
            return res;
        }

        // 上三分之一的通道均值判断蓝底; Mean 收 ROI 视图, 不复制
        static bool IsBluePage(Mat bgr)
        {
            int h3 = Math.Max(1, bgr.Height / 3);
            using var top = new Mat(bgr, new Rect(0, 0, bgr.Width, h3));
            var m = Cv2.Mean(top);            // BGR 顺序: Val0=B, Val1=G, Val2=R
            return m.Val0 > m.Val2 + 25 && m.Val0 > m.Val1 + 15;
        }

        // 在图像 8%~55% 高度范围内, 取同一行中字最高的那一行
        static Rect? LocateAmount(Mat gray)
        {
            int W = gray.Width, H = gray.Height;
            int y0b = (int)(H * 0.08), y1b = (int)(H * 0.55);
            int bh = y1b - y0b;
            if (bh < 8) return null;

            using var band = new Mat(gray, new Rect(0, y0b, W, bh));   // 视图
            using var dark = new Mat();
            // 手写版判据是 gray < 140, 即 <= 139; BinaryInv 的判据是 src <= thresh
            Cv2.Threshold(band, dark, GrayDark - 1, 255, ThresholdTypes.BinaryInv);

            var all = Label(dark, minArea: 20);
            var comps = new List<Comp>();
            foreach (var c in all)
            {
                // 此高度过滤会滤掉负号, 所以后面要在行内重取
                if (c.H < 0.02 * H || c.H > 0.22 * H || c.W > 0.5 * W) continue;
                comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
            }
            if (comps.Count == 0) return null;

            // 上限保护: 下面的行聚类最坏为 O(N^2)
            if (comps.Count > MaxComponents) return null;

            // 按纵向重叠聚类成行; 每行上下界增量维护
            var rows = new List<List<Comp>>();
            var bounds = new List<(int Y0, int Y1)>();
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
                if (x1 - x0 < 0.1 * W) continue;             // 太窄的一行不是金额
                var hh = r.Select(k => k.H).OrderBy(v => v).ToList();
                double med = hh[r.Count / 2];
                if (med > bestMed) { bestMed = med; best = r; }
            }
            if (best == null) return null;
            return Rect.FromLTRB(best.Min(k => k.X), best.Min(k => k.Y),
                                 best.Max(k => k.X + k.W), best.Max(k => k.Y + k.H));
        }

        // 8 连通标记; 非零算前景, 与手写版的 fg 口径一致
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

        // 偶数个时取中间两个的平均
        static double Median(IEnumerable<double> xs)
        {
            var v = xs.OrderBy(k => k).ToList();
            if (v.Count == 0) return 0;
            return v.Count % 2 == 1 ? v[v.Count / 2]
                                    : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2.0;
        }

        // 总体标准差
        static double Std(IEnumerable<double> xs)
        {
            var v = xs.ToList(); if (v.Count == 0) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Sum(k => (k - m) * (k - m)) / v.Count);
        }
    }
}
