#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum StepAlignVerdict
    {
        CannotDetermine = 0,   // 判不了(没有时间轴、图太小、量不到等), 走正常流程
        Ok = 1,
        Suspicious = 2,        // 进度圆点和右边的文字上下没对齐(假图常见: 越往下越偏, 第三个圆点偏下)
    }

    /// <summary>每个圆点量到的值(像素, 整图坐标)。</summary>
    public sealed class StepAlignCircle
    {
        public double Top, Bottom;      // 圆点上下边(半高处, 小数像素)
        public double TextBottom;       // 右边步骤文字的下边
        public int TextLeft;            // 右边步骤文字的左缘
        public int TextB;               // 步骤文字最后一行墨点的 y
        public double DateMid;          // 文字下面灰色日期的中线
        public int DateLeft;            // 日期左缘
        public int DateA;               // 日期第一行墨点的 y
        public int DateH;               // 日期行高
        public double Center => (Top + Bottom) / 2.0;
    }

    public sealed class StepAlignResult
    {
        public StepAlignVerdict Verdict;
        public bool TimelineFound;      // 找到了"处理进度"时间轴(同一列竖着排开 >=2 个蓝色对勾圆点)
        public int CircleCount;
        public double Diameter;         // 圆点宽的中位数, 像素(D)
        public bool Measured;           // 为 false 时下面几个数无意义
        public double Score;            // Min(SpreadText, SpreadDate), 判定就看它
        public double SpreadText;       // (圆点中心 - 文字下边) 在几个圆点之间的最大差 / D
        public double SpreadDate;       // (圆点中心 - 日期中线) 在几个圆点之间的最大差 / D
        public double[] OffsetText = Array.Empty<double>();   // 每个圆点: 圆点中心 - 文字下边(像素)
        public double[] OffsetDate = Array.Empty<double>();   // 每个圆点: 圆点中心 - 日期中线(像素)
        public List<StepAlignCircle?> Circles = new();        // 每个圆点量到的值, 量不到的是 null
        public string Reason = "";
    }

    /// <summary>
    /// 账单详情页"处理进度"三个蓝色对勾和右边文字的上下对齐检查("蓝色对勾要整齐")。
    ///
    /// 真图: 三个圆点(提交银行处理 / 银行处理中 / 银行入账成功)各自正对着右边那行字,
    ///       三个圆点相对各自文字的上下位置一模一样(同一张图里差不到 1 像素)。
    /// 假图: 圆点一列的间距和文字间距对不上(2026-11-13 的三张假图都是圆点间距比文字间距大 3.6%),
    ///       越往下越偏, 第三个"银行入账成功"的圆点明显偏下。
    /// 和 TimelineCheck(圆点左右缩进)是两回事, 两个可以一起用: 11-13 那张假图 TimelineCheck 判 Ok, 这里判 Suspicious。
    ///
    /// 量法(D = 圆点宽):
    ///   1. 找圆点: 和 TimelineCheck 完全一样。
    ///   2. 圆点上下中心: 圆点宽度 16%~31% 和 69%~84% 处 8 列(避开中间的连线), 每列取半高处的上下边(小数像素), 平均。
    ///   3. 右边步骤文字: 圆点上下各放宽 0.8D, 黑字按行切段, 取离圆点中心最近、高 >= 0.3D 的一段, 量它的下边
    ///      (用下边不用中线: 拼音字体的拼音在汉字正上方, 时连时断, 上边会跳, 下边不受影响)。
    ///   4. 文字下面的灰色日期(如 11-12 04:06): 1.8D 以内第一段高 >= 0.25D 的灰字, 量中线。三行日期通常是同一串数字, 最稳。
    ///   5. 圆点中心 - 文字下边、圆点中心 - 日期中线, 各自在几个圆点之间的最大差 / D;
    ///      Score = 两者里小的那个(两种参照都说没对齐才算), &gt; Threshold 判可疑。
    ///
    /// 本地实测: 真图时间轴页 822 张, Score 最高 0.033; 每张再过 6 种缩放 x 4 种 JPEG 压缩(19,363 份),
    /// 圆点直径 &gt;= 20 的最高 0.045。假图: 11-13 原图 0.233, 以前 112 抓到的假图里 53 张 0.14~0.90
    /// (所有缩放压缩下最低 0.14), 另外 26 张圆点没挪(0.03 以下), 那批靠 TimelineCheck 抓。
    /// 服务器实测(2026-11-13): 白图 217,166 张, 有时间轴 70,196 张, 判可疑 44 张, 逐张看过圆点都明显对不齐;
    /// 其中 39 张 TimelineCheck 也判可疑, 5 张 TimelineCheck 放过。没判可疑的最高 0.071。
    /// 阈值从 0.08 提到 0.10: 有一批安卓真图中间那个圆点固定差 1dp, 分数 0.067, 0.08 太贴;
    /// 已知假图(本地原图 + 缩放压缩 + 服务器 44 张)没有一张落在 0.06 ~ 0.12, 提到 0.10 一张不丢。
    ///
    /// 判不了(CannotDetermine)的几种, 都是宁可不判也不误判:
    ///   圆点直径 &lt; MinDiameter(图被缩得太小); 有圆点量不到圆心、步骤文字或日期;
    ///   几个圆点高矮差 &gt; 10%(有东西压在圆点上); 文字或日期左缘前后差 &gt; 0.25D(认到的不是同一种行);
    ///   文字到日期的间距前后差 &gt; 0.15D 或日期行高矮差 &gt; 25%(日期行认错了)。
    ///
    /// 和 training/step_align.py 逐步对应(整数列号、整数墨量和、同样的求和顺序, 两边结果逐位一样), 改一边要同步改另一边。
    /// 只读入参, 内部用 ROI 视图; 无静态可变状态, 可多线程调用。从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class StepAlignCheck
    {
        /// <summary>Score 大于这个值判可疑。真图本地(直径 &gt;= 20)最高 0.045, 服务器没判可疑的 7 万张最高 0.071; 假图最低 0.14。</summary>
        public const double Threshold = 0.10;

        /// <summary>圆点直径(宽)小于这个像素数不判。手写字体真图缩到 40% 再压缩, 有一份到了 0.15。</summary>
        public const double MinDiameter = 20;

        public const double ChordSpread = 0.10;   // 圆点高矮差 / 中位数
        public const double LeftSpread = 0.25;    // 文字、日期左缘前后差, 单位 D
        public const double GapSpread = 0.15;     // 文字下边到日期上边的间距前后差, 单位 D
        public const double DateHSpread = 0.25;   // 日期行高矮差 / 中位数

        static readonly int[] Cols = { 16, 21, 26, 31, 69, 74, 79, 84 };   // 量圆点上下边的列: 圆点宽的百分之几处
        const int Margin = 6;            // 圆点框上下各多看几个像素
        const double TextFrac = 0.15;    // 文字下边: 墨量曲线从背景往上 15% 处
        const double DateFrac = 0.25;    // 日期上下边: 25% 处

        // 支付宝蓝(OpenCV 的 HSV, H 是 0~180), 和 TimelineCheck 一样
        static readonly Scalar BlueLo = new(98, 120, 150), BlueHi = new(118, 255, 255);
        const int DarkGray = 150;        // 正文黑字: 灰度 < 150 且饱和度 < 60
        const int DarkSat = 60;
        const int GreyLo = 120;          // 灰字(日期): 灰度 120~205 且饱和度 < 40
        const int GreyHi = 205;
        const int GreySat = 40;

        /// <param name="image">8 位彩色图, 3 / 4 通道(BGR / BGRA)。传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static StepAlignResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new StepAlignResult { Verdict = StepAlignVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn == 1) { res.Reason = "单通道图, 圆点要靠蓝色找, 不判"; return res; }
            if (cn != 3 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");
            if (image.Width < 300 || image.Height < 300) { res.Reason = "图太小"; return res; }
            if (cn == 3) return Run(image, res);
            using var bgr = new Mat();
            Cv2.CvtColor(image, bgr, ColorConversionCodes.BGRA2BGR);
            return Run(bgr, res);
        }

        static StepAlignResult Run(Mat bgr, StepAlignResult res)
        {
            var cs = Timeline(FindCircles(bgr));
            if (cs.Count == 0) { res.Reason = "没有处理进度时间轴"; return res; }
            double D = Median(cs.Select(c => (double)c.Width));
            res.TimelineFound = true;
            res.CircleCount = cs.Count;
            res.Diameter = D;
            if (D < MinDiameter) { res.Reason = $"圆点直径小于 {MinDiameter} 像素, 图被缩小过, 不判"; return res; }

            foreach (var c in cs)
            {
                var e = CircleEdges(bgr, c);
                StepAlignCircle? m = null;
                if (e != null)
                {
                    var t = StepText(bgr, c, D);
                    if (t != null)
                    {
                        var d = DateRow(bgr, t.Value.left, t.Value.b, D);
                        if (d != null)
                            m = new StepAlignCircle
                            {
                                Top = e.Value.top, Bottom = e.Value.bot,
                                TextBottom = t.Value.bot, TextLeft = t.Value.left, TextB = t.Value.b,
                                DateMid = d.Value.mid, DateLeft = d.Value.left, DateA = d.Value.a, DateH = d.Value.h,
                            };
                    }
                }
                res.Circles.Add(m);
            }
            Decide(D, res);
            return res;
        }

        /// <summary>量好之后的判定, 和 step_align.py 的 decide() 一一对应。</summary>
        static void Decide(double D, StepAlignResult res)
        {
            var m = res.Circles;
            if (m.Count < 2 || m.Any(v => v == null)) { res.Reason = "有圆点量不到圆心、右边的步骤文字或下面的日期, 不判"; return; }
            var ms = m.Select(v => v!).ToList();
            var chords = ms.Select(v => v.Bottom - v.Top).ToList();
            if ((chords.Max() - chords.Min()) / Median(chords) > ChordSpread)
            { res.Reason = "几个圆点高矮不一样(有东西压在圆点上), 不判"; return; }
            var tl = ms.Select(v => v.TextLeft).ToList();
            var dl = ms.Select(v => v.DateLeft).ToList();
            if (tl.Max() - tl.Min() > LeftSpread * D || dl.Max() - dl.Min() > LeftSpread * D)
            { res.Reason = "步骤文字或日期的左缘前后对不齐, 认到的不是同一种行, 不判"; return; }
            var gaps = ms.Select(v => v.DateA - v.TextB).ToList();
            var dhs = ms.Select(v => (double)v.DateH).ToList();
            if (gaps.Max() - gaps.Min() > GapSpread * D || (dhs.Max() - dhs.Min()) / Median(dhs) > DateHSpread)
            { res.Reason = "日期行认得不一致, 不判"; return; }

            res.OffsetText = ms.Select(v => v.Center - v.TextBottom).ToArray();
            res.OffsetDate = ms.Select(v => v.Center - v.DateMid).ToArray();
            res.SpreadText = (res.OffsetText.Max() - res.OffsetText.Min()) / D;
            res.SpreadDate = (res.OffsetDate.Max() - res.OffsetDate.Min()) / D;
            res.Score = Math.Min(res.SpreadText, res.SpreadDate);
            res.Measured = true;
            if (res.Score > Threshold)
            {
                res.Verdict = StepAlignVerdict.Suspicious;
                res.Reason = $"圆点和右边文字上下没对齐: 几个圆点之间差 {res.Score * D:0.0} 像素({res.Score:0.000} 个直径)";
            }
            else
            {
                res.Verdict = StepAlignVerdict.Ok;
                res.Reason = $"圆点和右边文字对齐, 差 {res.Score * D:0.0} 像素({res.Score:0.000} 个直径)";
            }
        }

        /// <summary>圆点的上下边(小数像素): Cols 那 8 列各自量半高处的上下边, 再各自平均。量得到的列 &lt; 4 返回 null。</summary>
        static (double top, double bot)? CircleEdges(Mat bgr, Rect c)
        {
            int W = bgr.Width, H = bgr.Height;
            int y0 = Math.Max(0, c.Y - Margin), y1 = Math.Min(H, c.Y + c.Height + Margin);
            if (y1 - y0 < 8) return null;
            int xs = Math.Max(0, c.X), xe = Math.Min(W, c.X + c.Width);
            if (xe <= xs) return null;
            byte[] g = Gray(bgr, new Rect(xs, y0, xe - xs, y1 - y0));
            int gw = xe - xs, n = y1 - y0;
            var tops = new List<double>();
            var bots = new List<double>();
            var p = new long[n];
            foreach (int k in Cols)
            {
                int col = c.X + c.Width * k / 100;
                if (col < 0 || col >= W || col < xs || col >= xe) continue;
                for (int i = 0; i < n; i++) p[i] = 255 - g[i * gw + (col - xs)];
                long bg = Math.Max(Math.Max(p[0], p[1]), Math.Max(p[n - 2], p[n - 1]));
                long pk = p.Max();
                if (pk - bg < 50) continue;
                double thr = bg + 0.5 * (pk - bg);
                tops.Add(FirstCross(p, thr)!.Value + y0);
                bots.Add(LastCross(p, thr)!.Value + y0);
            }
            if (tops.Count < 4) return null;
            double st = 0, sb = 0;
            foreach (var v in tops) st += v;
            foreach (var v in bots) sb += v;
            return (st / tops.Count, sb / bots.Count);
        }

        /// <summary>圆点右边的步骤文字: 离圆点中心最近、高 &gt;= 0.3D 的黑字行。</summary>
        static (double bot, int left, int b)? StepText(Mat bgr, Rect c, double D)
        {
            int W = bgr.Width, H = bgr.Height;
            int x0 = Math.Min(W - 1, (int)(c.X + c.Width + 0.15 * D)), x1 = Math.Min(W, (int)(c.X + c.Width + 6 * D));
            int y0 = Math.Max(0, (int)(c.Y - 0.8 * D)), y1 = Math.Min(H, (int)(c.Y + c.Height + 0.8 * D));
            if (x1 - x0 < 5 || y1 - y0 < 5) return null;
            var rect = new Rect(x0, y0, x1 - x0, y1 - y0);
            byte[] dark = Mask(bgr, rect, dark: true);
            int ww = rect.Width, wh = rect.Height;
            double cyr = c.Y + c.Height / 2.0 - y0;
            int ba = -1, bb = -1;
            double bd = 0;
            foreach (var (a, b) in Runs(RowsOn(dark, ww, wh)))
            {
                if (b - a + 1 < 0.3 * D) continue;
                double d = Math.Abs((a + b) / 2.0 - cyr);
                if (ba < 0 || d < bd) { bd = d; ba = a; bb = b; }
            }
            if (ba < 0) return null;
            var (c0, c1) = ColSpan(dark, ww, ba, bb);
            byte[] g = Gray(bgr, rect);
            var (p, lo) = Profile(g, ww, wh, ba, bb, c0, c1);
            long bg = Math.Min(p[0], p[p.Length - 1]), pk = p.Max();
            if (pk <= bg) return null;
            double bot = LastCross(p, bg + TextFrac * (pk - bg))!.Value;
            return (bot + lo + y0, c0 + x0, bb + y0);
        }

        /// <summary>步骤文字下面那行灰色日期: 文字下方 1.8D 以内第一段高 &gt;= 0.25D 的灰字。</summary>
        static (double mid, int left, int a, int h)? DateRow(Mat bgr, int textLeft, int textB, double D)
        {
            int W = bgr.Width, H = bgr.Height;
            int x0 = Math.Max(0, (int)(textLeft - 0.3 * D)), x1 = Math.Min(W, (int)(textLeft + 6 * D));
            int y0 = textB + 1, y1 = Math.Min(H, (int)(textB + 1.8 * D));
            if (x1 - x0 < 5 || y1 - y0 < 5) return null;
            var rect = new Rect(x0, y0, x1 - x0, y1 - y0);
            byte[] grey = Mask(bgr, rect, dark: false);
            int ww = rect.Width, wh = rect.Height;
            int pa = -1, pb = -1;
            foreach (var (a, b) in Runs(RowsOn(grey, ww, wh)))
                if (b - a + 1 >= 0.25 * D) { pa = a; pb = b; break; }
            if (pa < 2) return null;               // 没找到, 或紧贴着文字(分不清是日期还是文字的毛边)
            var (c0, c1) = ColSpan(grey, ww, pa, pb);
            byte[] g = Gray(bgr, rect);
            var (p, lo) = Profile(g, ww, wh, pa, pb, c0, c1);
            long bg = Math.Min(p[0], p[p.Length - 1]), pk = p.Max();
            if (pk <= bg) return null;
            double thr = bg + DateFrac * (pk - bg);
            double top = FirstCross(p, thr)!.Value, bot = LastCross(p, thr)!.Value;
            return ((top + bot) / 2.0 + lo + y0, c0 + x0, pa + y0, pb - pa + 1);
        }

        /// <summary>行 [a-3, b+3] 在列 [c0, c1] 上的墨量(255 - 灰度 的整数和)。</summary>
        static (long[] p, int lo) Profile(byte[] g, int ww, int wh, int a, int b, int c0, int c1)
        {
            int lo = Math.Max(0, a - 3), hi = Math.Min(wh, b + 4);
            var p = new long[hi - lo];
            for (int i = lo; i < hi; i++)
            {
                long s = 0;
                for (int x = c0; x <= c1; x++) s += 255 - g[i * ww + x];
                p[i - lo] = s;
            }
            return (p, lo);
        }

        static double? FirstCross(long[] p, double thr)
        {
            for (int i = 0; i < p.Length; i++)
                if (p[i] >= thr)
                    return i == 0 ? 0.0 : (i - 1) + (thr - p[i - 1]) / (p[i] - p[i - 1]);
            return null;
        }

        static double? LastCross(long[] p, double thr)
        {
            int n = p.Length;
            for (int i = n - 1; i >= 0; i--)
                if (p[i] >= thr)
                    return i == n - 1 ? i : i + (p[i] - thr) / (p[i] - p[i + 1]);
            return null;
        }

        static bool[] RowsOn(byte[] m, int ww, int wh)
        {
            var on = new bool[wh];
            for (int y = 0; y < wh; y++)
            {
                int n = 0;
                for (int x = 0; x < ww; x++) if (m[y * ww + x] != 0) n++;
                on[y] = n >= 2;
            }
            return on;
        }

        static (int c0, int c1) ColSpan(byte[] m, int ww, int a, int b)
        {
            int c0 = ww, c1 = -1;
            for (int y = a; y <= b; y++)
                for (int x = 0; x < ww; x++)
                    if (m[y * ww + x] != 0) { if (x < c0) c0 = x; if (x > c1) c1 = x; }
            return (c0, c1);
        }

        static byte[] Gray(Mat bgr, Rect rect)
        {
            using var roi = new Mat(bgr, rect);
            using var gray = new Mat();
            Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
            var buf = new byte[rect.Width * rect.Height];
            System.Runtime.InteropServices.Marshal.Copy(gray.Data, buf, 0, buf.Length);   // CvtColor 输出是连续的
            return buf;
        }

        // ---- 下面三个和 TimelineCheck 里的同名函数一字不差(两个文件各自独立, 只拷一个也能用) ----

        static List<Rect> FindCircles(Mat bgr)
        {
            int W = bgr.Width, H = bgr.Height;
            int x0 = (int)(0.15 * W), x1 = (int)(0.55 * W);
            using var roi = new Mat(bgr, new Rect(x0, 0, x1 - x0, H));
            using var hsv = new Mat();
            Cv2.CvtColor(roi, hsv, ColorConversionCodes.BGR2HSV);
            using var gray = new Mat();
            Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);

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
            for (int i = 1; i < n; i++)
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
                int r = Math.Max(1, w / 2);
                int ix = x + w / 4, iy = y + h / 4;
                int iw = Math.Min(r, gray.Cols - ix), ih = Math.Min(r, gray.Rows - iy);
                if (iw <= 0 || ih <= 0) continue;
                using (var inner = new Mat(gray, new Rect(ix, iy, iw, ih)))
                    Cv2.InRange(inner, new Scalar(171), new Scalar(255), white);
                if (Cv2.CountNonZero(white) * 25 < iw * ih) continue;

                int xa = Math.Max(0, x - 4), xb = Math.Min(gray.Cols, x + Math.Max(1, w / 4));
                int ya = y + h / 4, yb = Math.Min(gray.Rows, y + h - h / 4);
                int firstCol = -1;
                for (int cx = xa; cx < xb && firstCol < 0; cx++)
                    for (int cy = ya; cy < yb; cy++)
                        if (gray.At<byte>(cy, cx) < 180) { firstCol = cx; break; }
                if (firstCol > xa) x = firstCol;

                outList.Add(new Rect(x + x0, y, w, h));
            }
            return outList;
        }

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
                Cv2.InRange(gray, new Scalar(GreyLo), new Scalar(GreyHi), m1);
                Cv2.InRange(sat, new Scalar(0), new Scalar(GreySat - 1), m2);
            }
            Cv2.BitwiseAnd(m1, m2, m);
            var buf = new byte[rect.Width * rect.Height];
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
