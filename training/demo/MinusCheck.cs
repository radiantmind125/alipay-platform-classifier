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
        public bool Measured;        // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 白底账单详情页的负号几何检查: 量负号宽度与数字中位宽度的比值。
    /// 直接收 Mat, 内部全部用 ROI 视图, 不复制像素。无静态可变状态, 可多线程调用。
    /// </summary>
    public static class MinusCheck
    {
        public const double Threshold = 0.78;
        public const double MinDigitHeight = 60;   // 数字低于此高度不判定; 设为 0 可关闭
        const int GrayDark = 140;                  // 定位金额行时的深色阈值

        public const int MaxComponents = 20_000;   // 上限保护, 见 LocateAmount

        struct Comp
        {
            public int X, Y, W, H, Area;
        }

        /// <param name="image">
        /// 8 位图, 1 / 3 / 4 通道均可; 多通道按 OpenCV 惯例视为 BGR(A)。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        public static MinusResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new MinusResult { Verdict = MinusVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U)
                throw new ArgumentException("只支持 8 位图");

            int cn = image.Channels();
            if (cn != 1 && cn != 3 && cn != 4)
                throw new ArgumentException($"不支持 {cn} 通道");
            if (image.Width < 16 || image.Height < 16) { res.Reason = "图太小"; return res; }

            // 蓝底转账页金额不带符号, 不处理
            if (cn >= 3 && IsBluePage(image))
            { res.Reason = "蓝底转账页, 金额不带负号"; return res; }

            // 单通道时直接用入参, 不复制也不释放; 多通道才需要转换出一张自己的灰度图
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

            var bar = bars.OrderBy(g => g.X).First();   // 最左侧的横条即负号
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
            res.Verdict = res.BarWidth >= Threshold ? MinusVerdict.Suspicious : MinusVerdict.Ok;
            res.Reason = $"负号宽比 {res.BarWidth:F4} (阈值 {Threshold})";
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
