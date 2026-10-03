#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum TimelineVerdict
    {
        CannotDetermine = 0,   // 判不了(没有时间轴、找不到取值列等), 走正常流程
        Ok = 1,
        Suspicious = 2,        // 进度圆点没有往右缩进: 和取值列文字对齐, 或者比文字还靠左
    }

    public sealed class TimelineResult
    {
        public TimelineVerdict Verdict;
        public bool TimelineFound;   // 找到了"处理进度"时间轴(同一列竖着排开 >=2 个蓝色对勾圆点)
        public int CircleCount;      // 时间轴上的圆点个数
        public double Diameter;      // 圆点直径(各圆点宽的中位数), 像素
        public int CircleLeft;       // 圆点左缘 x(各圆点的中位数)
        public int CircleTop;        // 第一个圆点顶边 y
        public int ValueLeft;        // 圆点上方取值列文字的左缘 x
        public int OffsetPixels;     // CircleLeft - ValueLeft
        public double Offset;        // OffsetPixels / Diameter, 判定就看它
        public double LabelInk;      // 第一个圆点那一行、取值列左边的灰字像素 / D^2(只记录, 不参与判定)
        public bool LabelFound;      // 左边有没有"处理进度"标签(只记录, 不参与判定)
        public bool Measured;        // 为 false 时 ValueLeft / OffsetPixels / Offset 无意义
        public string Reason = "";
    }

    /// <summary>
    /// 账单详情页"处理进度"时间轴的位置检查。
    ///
    /// 真图的蓝色进度圆点比上面取值列的文字(如"招商银行")往右缩进几个像素;
    /// 假图的圆点和文字左缘对齐, 甚至往左凸出来。从文字左缘往下拉一条竖线,
    /// 真图从圆点左边擦过, 假图切进圆点里 —— 这里把这条线量准:
    ///     Offset = (圆点左缘 - 取值列左缘) / 圆点直径,  &lt;= Threshold 判可疑。
    ///
    /// 用圆点直径做单位, 不用页面宽度的百分比: 不同机型、不同显示缩放下,
    /// 圆点和文字跟着一起缩放, 两者的相对位置不变; 按页面百分比就会随机型漂。
    ///
    /// 本地 13,815 张实测(其中 489 张有时间轴):
    ///     正常字体真图         +0.19 ~ +0.24(各分辨率都在这里)
    ///     拼音手写字体真图     最低 +0.018(这种字体的字几乎不留左边白, 是最贴近的一类)
    ///     假图                 -0.036 ~ -0.068
    ///
    /// 取值列左缘取离时间轴最近两行文字各自最左的墨点, 再取更靠左的那个:
    /// 每个字左边留白不一样("中"比"账"多两三个像素), 只看一行会把字形差当成位置差。
    ///
    /// 和 training/timeline_scan.py 逐步对应(那边是批量扫描用的参考实现), 改一边要同步改另一边。
    /// 只读入参, 内部全部用 ROI 视图, 不复制整图(4 通道图除外, 要先去掉 alpha)。
    /// 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class TimelineCheck
    {
        /// <summary>Offset 小于等于这个值判可疑。按直径 46~62 像素算, 就是圆点比文字至少靠左 1 个像素。</summary>
        public const double Threshold = -0.01;

        /// <summary>圆点直径小于这个像素数不判: 图被缩得太小, 一个像素就是 0.05 个直径, 量不准。</summary>
        public const double MinDiameter = 20;

        /// <summary>标签灰字像素 / D^2 达到这个值算有"处理进度"标签。</summary>
        public const double LabelInkMin = 0.1;

        // 支付宝蓝(OpenCV 的 HSV, H 是 0~180)
        static readonly Scalar BlueLo = new(98, 120, 150), BlueHi = new(118, 255, 255);

        const int DarkGray = 110;        // 正文黑字: 灰度 < 110 且饱和度 < 60(排除蓝字)
        const int DarkSat = 60;
        const int LabelGrayLo = 120;     // 标签灰字: 灰度 120~205 且饱和度 < 40
        const int LabelGrayHi = 205;
        const int LabelSat = 40;

        /// <param name="image">
        /// 8 位彩色图, 3 / 4 通道, 按 OpenCV 惯例视为 BGR(A)。单通道图找不了蓝色圆点, 返回 CannotDetermine。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        public static TimelineResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new TimelineResult { Verdict = TimelineVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U)
                throw new ArgumentException("只支持 8 位图");

            int cn = image.Channels();
            if (cn == 1) { res.Reason = "单通道图, 圆点要靠蓝色找, 不判"; return res; }
            if (cn != 3 && cn != 4)
                throw new ArgumentException($"不支持 {cn} 通道");
            if (image.Width < 300 || image.Height < 300) { res.Reason = "图太小"; return res; }

            if (cn == 3) return Run(image, res);
            using var bgr = new Mat();
            Cv2.CvtColor(image, bgr, ColorConversionCodes.BGRA2BGR);
            return Run(bgr, res);
        }

        static TimelineResult Run(Mat bgr, TimelineResult res)
        {
            int W = bgr.Width, H = bgr.Height;

            var cs = Timeline(FindCircles(bgr));
            if (cs.Count == 0) { res.Reason = "没有处理进度时间轴"; return res; }

            double D = Median(cs.Select(c => (double)c.Width));
            int cl = (int)Median(cs.Select(c => (double)c.X));
            var c0 = cs[0];
            res.TimelineFound = true;
            res.CircleCount = cs.Count;
            res.Diameter = D;
            res.CircleLeft = cl;
            res.CircleTop = c0.Y;
            if (D < MinDiameter) { res.Reason = $"圆点直径 {D:0.#} 像素, 太小量不准"; return res; }

            // 取值列: 第一个圆点上方 6D 以内, 圆点左右 [-1.5D, +4D] 的窗口。
            // 窗口不从页面左边开始, 免得量到返回箭头、左边的标签栏。
            int wx0 = Math.Max(0, (int)(cl - 1.5 * D)), wx1 = Math.Min(W, (int)(cl + 4 * D));
            int wy0 = Math.Max(0, (int)(c0.Y - 6 * D)), wy1 = Math.Max(0, (int)(c0.Y - 0.3 * D));
            var lefts = new List<int>();
            if (wy1 > wy0 && wx1 > wx0)
            {
                byte[] dark = Mask(bgr, new Rect(wx0, wy0, wx1 - wx0, wy1 - wy0), dark: true);
                int ww = wx1 - wx0, wh = wy1 - wy0;
                var on = new bool[wh];
                for (int y = 0; y < wh; y++)
                {
                    int n = 0;
                    for (int x = 0; x < ww; x++) if (dark[y * ww + x] != 0) n++;
                    on[y] = n >= 2;
                }
                var rows = Runs(on).Where(r => r.b - r.a + 1 >= 0.4 * D).ToList();
                for (int i = rows.Count - 1; i >= 0; i--)          // 从下往上, 离时间轴最近的先
                {
                    int first = ww;
                    for (int y = rows[i].a; y <= rows[i].b; y++)
                        for (int x = 0; x < first; x++)
                            if (dark[y * ww + x] != 0) { first = x; break; }
                    if (first <= 1) continue;                      // 贴着窗口左边: 是从标签栏伸过来的
                    lefts.Add(first + wx0);
                }
            }

            // "处理进度"标签: 第一个圆点那一行、取值列左边的灰字。只记录, 不参与判定。
            int ly0 = Math.Max(0, (int)(c0.Y - 0.3 * D)), ly1 = Math.Min(H, (int)(c0.Y + c0.Height + 0.3 * D));
            int lx0 = (int)(0.03 * W), lx1 = Math.Max(lx0 + 1, (int)(cl - 1.5 * D));
            if (ly1 > ly0)
            {
                byte[] grey = Mask(bgr, new Rect(lx0, ly0, lx1 - lx0, ly1 - ly0), dark: false);
                int g = grey.Count(v => v != 0);
                res.LabelInk = g / (D * D);
                res.LabelFound = g >= LabelInkMin * D * D;
            }

            if (lefts.Count == 0) { res.Reason = "圆点上方找不到取值列文字"; return res; }

            int vl = lefts.Count >= 2 ? Math.Min(lefts[0], lefts[1]) : lefts[0];
            res.ValueLeft = vl;
            res.OffsetPixels = cl - vl;
            res.Offset = (cl - vl) / D;
            res.Measured = true;

            if (res.Offset <= Threshold)
            {
                res.Verdict = TimelineVerdict.Suspicious;
                res.Reason = $"圆点左缘 {cl} 不在取值列文字左缘 {vl} 的右边, 偏移 {res.Offset:+0.000;-0.000} 个直径"
                             + (res.LabelFound ? "" : ", 左边也没有处理进度标签");
            }
            else
            {
                res.Verdict = TimelineVerdict.Ok;
                res.Reason = $"圆点比取值列文字往右缩进 {res.OffsetPixels} 像素, 偏移 {res.Offset:+0.000;-0.000} 个直径";
            }
            return res;
        }

        /// <summary>
        /// 带白色对勾的蓝色实心圆。只在页面宽度 15%~55% 这一竖条里找。
        /// 先用横条开运算去掉圆点之间那根细竖线(它比核窄), 再闭运算把对勾的白缝补上。
        /// </summary>
        static List<Rect> FindCircles(Mat bgr)
        {
            int W = bgr.Width, H = bgr.Height;
            int x0 = (int)(0.15 * W), x1 = (int)(0.55 * W);
            using var roi = new Mat(bgr, new Rect(x0, 0, x1 - x0, H));
            using var hsv = new Mat();
            Cv2.CvtColor(roi, hsv, ColorConversionCodes.BGR2HSV);

            using var blue = new Mat();
            Cv2.InRange(hsv, BlueLo, BlueHi, blue);
            int k = Math.Max(5, (int)(W * 0.012)) | 1;
            using var opened = new Mat();
            using var closed = new Mat();
            using (var ker = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(k, 1)))
                Cv2.MorphologyEx(blue, opened, MorphTypes.Open, ker);
            using (var ker = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(k, k)))
                Cv2.MorphologyEx(opened, closed, MorphTypes.Close, ker);

            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(closed, labels, stats, centroids,
                                                     PixelConnectivity.Connectivity8, MatType.CV_32S);
            var outList = new List<Rect>();
            using var white = new Mat();
            for (int i = 1; i < n; i++)      // 0 是背景
            {
                int x = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
                int y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
                int w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
                int h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
                int a = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                if (!(0.030 * W <= w && w <= 0.056 * W && 0.80 * h <= w && w <= 1.25 * h && a * 2 >= w * h))
                    continue;
                if (!(0.22 * W <= x + x0 && x + x0 <= 0.42 * W))
                    continue;
                // 圆心一块里要有白色对勾: 亮(V > 200)而不饱和(S < 60)的像素 >= 4%
                int r = Math.Max(1, w / 2);
                int ix = x + w / 4, iy = y + h / 4;
                int iw = Math.Min(r, hsv.Cols - ix), ih = Math.Min(r, hsv.Rows - iy);
                if (iw <= 0 || ih <= 0) continue;
                using (var inner = new Mat(hsv, new Rect(ix, iy, iw, ih)))
                    Cv2.InRange(inner, new Scalar(0, 0, 201), new Scalar(255, 59, 255), white);
                if (Cv2.CountNonZero(white) * 25 < iw * ih) continue;
                outList.Add(new Rect(x + x0, y, w, h));
            }
            return outList;
        }

        /// <summary>
        /// 同一列(左缘差 &lt;= 0.15D, 直径差 &lt;= 0.2D)、竖着间隔 2.2D~7D 排开的圆点, 取最长的一组;
        /// 不到 2 个返回空。结果按 y 从上到下排。
        /// </summary>
        static List<Rect> Timeline(List<Rect> cs)
        {
            var best = new List<Rect>();
            foreach (var c in cs)
            {
                var grp = cs.Where(d => Math.Abs(d.X - c.X) <= 0.15 * c.Width
                                        && Math.Abs(d.Width - c.Width) <= 0.2 * c.Width)
                            .OrderBy(d => d.Y).ThenBy(d => d.X).ToList();
                if (grp.Count < 2) continue;
                bool ok = true;
                for (int i = 1; i < grp.Count && ok; i++)
                {
                    double g = (double)(grp[i].Y - grp[i - 1].Y) / grp[i - 1].Width;
                    ok = g >= 2.2 && g <= 7.0;
                }
                if (ok && grp.Count > best.Count) best = grp;
            }
            return best.Count >= 2 ? best : new List<Rect>();
        }

        /// <summary>
        /// 一块区域的掩码, 行优先展开成字节数组(非 0 = 命中)。
        /// dark = true: 正文黑字; false: 标签灰字。
        /// </summary>
        static byte[] Mask(Mat bgr, Rect rect, bool dark)
        {
            using var roi = new Mat(bgr, rect);
            using var gray = new Mat();
            using var hsv = new Mat();
            Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(roi, hsv, ColorConversionCodes.BGR2HSV);
            using var sat = new Mat();
            Cv2.ExtractChannel(hsv, sat, 1);

            using var m1 = new Mat();
            using var m2 = new Mat();
            using var m = new Mat();
            if (dark)
            {
                Cv2.InRange(gray, new Scalar(0), new Scalar(DarkGray - 1), m1);
                Cv2.InRange(sat, new Scalar(0), new Scalar(DarkSat - 1), m2);
            }
            else
            {
                Cv2.InRange(gray, new Scalar(LabelGrayLo), new Scalar(LabelGrayHi), m1);
                Cv2.InRange(sat, new Scalar(0), new Scalar(LabelSat - 1), m2);
            }
            Cv2.BitwiseAnd(m1, m2, m);
            var buf = new byte[rect.Width * rect.Height];
            // InRange 的输出是新分配的连续矩阵, 可以整块拷
            System.Runtime.InteropServices.Marshal.Copy(m.Data, buf, 0, buf.Length);
            return buf;
        }

        static List<(int a, int b)> Runs(bool[] on)
        {
            var segs = new List<(int, int)>();
            int i = 0, n = on.Length;
            while (i < n)
            {
                if (on[i])
                {
                    int a = i;
                    while (i < n && on[i]) i++;
                    segs.Add((a, i - 1));
                }
                i++;
            }
            return segs;
        }

        static double Median(IEnumerable<double> values)
        {
            var v = values.OrderBy(t => t).ToArray();
            int n = v.Length;
            return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
        }
    }
}
