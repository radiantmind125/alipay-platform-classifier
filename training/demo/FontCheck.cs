#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum FontVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // 金额字号偏大
    }

    public sealed class FontResult
    {
        public FontVerdict Verdict;
        public int DigitHeight;        // 金额数字中位高(像素)
        public int NormalHeight;       // 该分辨率的常态高; 0 表示表里没有
        public double SizeRatio;       // DigitHeight / NormalHeight
        public int BodyHeight;         // 正文字高(像素)
        public double AmountToBody;    // DigitHeight / BodyHeight
        public int DigitCount;
        // 下面两项只做输出, 不参与判定, 原因见类注释
        public int BaselineSpread;     // 行内数字底边的最大偏差(像素)
        public double HeightSpread;    // 行内数字高度的变异系数
        public bool Measured;          // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 白底账单详情页的金额字号检查。
    ///
    /// 金额数字高在同一分辨率下几乎是个定值: 实测 12,000 张真图, 每个分辨率的高度
    /// 集中在两个相邻整数上(如 1179x2556 是 72 或 71, 合计占 98%), 组内变异系数 0.8%~4.2%。
    /// 所以"金额被重打成更大的字号"是可以量出来的。
    ///
    /// 只判偏大, 不判偏小: 常态往下 1~2 像素就是另一个正常取值, 低侧没有余量。
    ///
    /// ★ 只看"金额相对屏幕多大"是不够的: 开了系统大字体的用户, 整页字号都被放大,
    /// 金额高会到常态的 1.35 倍, 但这是正常用户。实测四张这样的真图, 金额/正文分别是
    /// 2.425 / 2.459 / 2.432 / 2.282, 正落在真图中位 2.414 上 —— 整页一起放大, 金额本身没被动。
    /// 所以必须同时要求"金额相对同一张图里的正文也偏大", 两个条件都满足才报。
    /// 少了这一条, 这个 check 会系统性地误伤需要大字体的用户。
    ///
    /// 行内一致性(底边对齐、高度齐不齐)只输出不判定。它原理上只能抓"逐位重打/贴图",
    /// 抓不到"整行重画"(整行重渲染时每位数字仍然共用同一基线); 而且实测 PNG 与 JPEG
    /// 之间的报出率差 43 倍, 阈值没法统一。留着这两个数是给人工复核排序用的。
    ///
    /// 只读入参, 内部全部用 ROI 视图, 不复制整图。无静态可变状态, 可多线程调用。
    /// </summary>
    public static class FontCheck
    {
        /// <summary>金额字高超过本分辨率常态的这个倍数, 才算"偏大"。</summary>
        public const double SizeThreshold = 1.03;

        /// <summary>金额字高 / 正文字高 的真图常态(实测中位 2.414)。</summary>
        public const double NormalAmountToBody = 2.41;

        /// <summary>
        /// 上面那个比值超过常态的这个倍数, 才算"只有金额被放大"。
        /// 两个条件必须同时满足才报 —— 见类注释里系统大字体那一段。
        /// </summary>
        public const double RatioThreshold = 1.08;

        const int GrayDark = 140;                  // 定位金额行时的深色阈值
        public const int MaxComponents = 20_000;   // 上限保护, 见 LocateAmount

        /// <summary>
        /// 各分辨率的金额数字常态高(像素), 2026-07 的 12,000 张真图实测众数。
        /// 只收了每组样本 150 张以上的分辨率; 不在表里的一律返回 CannotDetermine, 不猜。
        ///
        /// 支付宝改版会让这张表整体失效, 需要按月重新标定 —— 这是绝对像素判据的固有代价,
        /// MinusCheck 那种比值判据没有这个问题。
        /// </summary>
        static readonly Dictionary<(int, int), int> NormalDigitHeight = new()
        {
            { (1179, 2556), 72 }, { (1320, 2868), 79 }, { (1290, 2796), 78 },
            { (1080, 2400), 66 }, { (1170, 2532), 71 }, { (1206, 2622), 73 },
            { (1260, 2800), 77 }, { (1080, 2412), 66 }, { (1284, 2778), 78 },
            { (1080, 2340), 66 }, {  (720, 1600), 43 }, {  (828, 1792), 51 },
            { (1080, 2376), 66 }, { (1080, 2408), 66 },
        };

        struct Comp
        {
            public int X, Y, W, H, Area;
        }

        /// <param name="image">
        /// 8 位图, 1 / 3 / 4 通道均可; 多通道按 OpenCV 惯例视为 BGR(A)。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        public static FontResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new FontResult { Verdict = FontVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U)
                throw new ArgumentException("只支持 8 位图");

            int cn = image.Channels();
            if (cn != 1 && cn != 3 && cn != 4)
                throw new ArgumentException($"不支持 {cn} 通道");

            int W = image.Width, H = image.Height;
            if (W < 16 || H < 16) { res.Reason = "图太小"; return res; }

            if (!NormalDigitHeight.TryGetValue((W, H), out int normal))
            {
                res.Reason = $"{W}x{H} 不在常态表里, 不判";
                return res;
            }
            res.NormalHeight = normal;

            // 蓝底转账页金额的排版和白底页不一样, 常态表不适用
            if (cn >= 3 && IsBluePage(image))
            { res.Reason = "蓝底转账页, 常态表不适用"; return res; }

            Mat? owned = null;
            try
            {
                Mat gray;
                if (cn == 1) gray = image;
                else
                {
                    owned = new Mat();
                    Cv2.CvtColor(image, owned,
                        cn == 4 ? ColorConversionCodes.BGRA2GRAY : ColorConversionCodes.BGR2GRAY);
                    gray = owned;
                }
                return Check(gray, res);
            }
            finally { owned?.Dispose(); }
        }

        static FontResult Check(Mat gray, FontResult res)
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
            if (digits.Count < 4) { res.Reason = "数字不够 4 个"; return res; }

            // ★ 必须有负号才判。常态表是在"带负号的付款页"上标定的,
            // 而收款页是另一套排版, 金额字号大得多 —— 实测 1170x2532 上付款页 71 像素,
            // 收款页 97~109 像素, 差 1.37 倍。拿付款页的表去套收款页, 会把
            // 一整类正常页面全判成偏大(实测误报从 27/万 涨到 200/万)。
            // 收款页要判就得另立一张表, 不能共用这一张。
            var bars = glyphs.Where(g => g.W >= 1.5 * g.H && g.H <= 0.45 * medH).ToList();
            if (bars.Count == 0) { res.Reason = "没有负号, 不是付款页, 常态表不适用"; return res; }

            // 与 MinusCheck 相同的三道有效性闸: 挡掉定位跑偏、把二维码或多行文字当成金额行的情况。
            // 实测不加这三道闸, 本底里会混进"金额行"高度 43~148 像素这种明显错框的样本。
            double dhMean = digits.Average(g => (double)g.H);
            double dhStd = Std(digits.Select(g => (double)g.H));
            if (dhStd / dhMean > 0.08) { res.Reason = "数字高度不齐, 不像一行数字"; return res; }
            double dwMean = digits.Average(g => (double)g.W);
            double dwStd = Std(digits.Select(g => (double)g.W));
            if (dwStd / dwMean > 0.30) { res.Reason = "数字宽度差太多"; return res; }

            double mw = Median(digits.Select(g => (double)g.W));
            double mh = Median(digits.Select(g => (double)g.H));
            if (mw / mh < 0.45 || mw / mh > 0.75) { res.Reason = "数字宽高比不对"; return res; }

            // 正文字高: 金额行下方那一片明显更小的文字。用它把"整页放大"和"只放大金额"分开。
            int bodyH = BodyHeight(gray, by1);
            if (bodyH <= 0) { res.Reason = "量不到正文字高, 判不准所以不判"; return res; }

            var bottoms = digits.Select(g => g.Y + g.H).ToList();

            res.Measured = true;
            res.BodyHeight = bodyH;
            res.AmountToBody = mh / bodyH;
            res.DigitHeight = (int)Math.Round(mh);
            res.DigitCount = digits.Count;
            res.SizeRatio = mh / res.NormalHeight;
            res.BaselineSpread = bottoms.Max() - bottoms.Min();
            res.HeightSpread = dhStd / dhMean;

            bool tooBigOnScreen = res.SizeRatio > SizeThreshold;
            bool tooBigInPage = res.AmountToBody > NormalAmountToBody * RatioThreshold;
            if (tooBigOnScreen && tooBigInPage)
            {
                res.Verdict = FontVerdict.Suspicious;
                res.Reason = $"金额字高 {res.DigitHeight} 是常态 {res.NormalHeight} 的 {res.SizeRatio:F3} 倍, "
                           + $"且金额/正文 {res.AmountToBody:F3} 高于常态 {NormalAmountToBody}";
            }
            else
            {
                res.Verdict = FontVerdict.Ok;
                res.Reason = tooBigOnScreen
                    ? $"金额字高 {res.DigitHeight} 是常态的 {res.SizeRatio:F3} 倍, "
                      + $"但金额/正文 {res.AmountToBody:F3} 正常, 整页字号偏大而已"
                    : $"金额字高 {res.DigitHeight}, 常态 {res.NormalHeight}, {res.SizeRatio:F3} 倍";
            }
            return res;
        }

        /// <summary>
        /// 金额行下方正文文字的中位块高。取块数不足时返回 0, 由调用方弃权。
        /// </summary>
        static int BodyHeight(Mat gray, int amountBottom)
        {
            int W = gray.Width, H = gray.Height;
            int y0 = amountBottom + (int)(H * 0.01), y1 = Math.Min(H, (int)(H * 0.75));
            if (y1 - y0 < 50) return 0;

            using var band = new Mat(gray, new Rect(0, y0, W, y1 - y0));   // 视图
            using var dark = new Mat();
            Cv2.Threshold(band, dark, GrayDark - 1, 255, ThresholdTypes.BinaryInv);

            var hs = new List<double>();
            foreach (var c in Label(dark, minArea: 20))
                if (c.H > 0.008 * H && c.H < 0.030 * H && c.W < 0.4 * W) hs.Add(c.H);
            if (hs.Count < 8) return 0;
            return (int)Math.Round(Median(hs));
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
            // 判据是 gray < 140, 即 <= 139; BinaryInv 的判据是 src <= thresh
            Cv2.Threshold(band, dark, GrayDark - 1, 255, ThresholdTypes.BinaryInv);

            var comps = new List<Comp>();
            foreach (var c in Label(dark, minArea: 20))
            {
                if (c.H < 0.02 * H || c.H > 0.22 * H || c.W > 0.5 * W) continue;
                comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
            }
            if (comps.Count == 0 || comps.Count > MaxComponents) return null;

            // 按纵向重叠聚类成行; 每行上下界增量维护。
            // 排序取全序: 连通域的返回顺序依实现而定, 并列时不能靠它决定
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

        // 8 连通标记; 非零算前景
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
