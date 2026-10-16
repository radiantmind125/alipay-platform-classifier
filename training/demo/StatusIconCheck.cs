#nullable enable

using System;
using OpenCvSharp;

namespace Ssp
{
    public enum StatusIconVerdict
    {
        CannotDetermine = 0,   // 判不了(不是蓝图、通知条盖住了页头、圆里的图认不出等), 走正常流程
        Ok = 1,                // 成功页的页头(标题左边空心细圈打勾 + 大号金额)
        Suspicious = 2,        // 到账失败(实心白圆里是感叹号); flagPending 为 true 时 提交成功(实心白圆里是时钟)也算
    }

    public enum StatusIconKind
    {
        None = 0,
        Failed = 1,            // 到账失败 资金将原路返回你的付款账户
        Pending = 2,           // 提交成功 预计 2 小时内到账(钱还没到)
        Success = 3,           // 转账成功 / 到账成功 / 支付成功 / 代付成功 / 授权成功 这一类的页头
        UnknownDisk = 4,       // 有实心白圆, 但圆里的图认不出
    }

    public sealed class StatusIconResult
    {
        public StatusIconVerdict Verdict;
        public StatusIconKind Kind;
        public bool DiskFound;        // 找到了标题正上方的实心白圆
        public bool RingFound;        // 找到了成功页标题左边的空心细圈
        public double CenterX;        // 圆心横坐标 / 页宽
        public double Top;            // 圆顶 / 页高
        public double Diameter;       // 圆的宽 / 页宽
        public double WhiteFill;      // 白色像素 / 外接框面积
        public double FilledFill;     // 填掉圆里的字以后 / 外接框面积(正圆 0.785)
        public double GlyphWidth;     // 圆里蓝字的宽 / 直径: 感叹号 0.05~0.13, 时钟 0.22~0.28
        public double GlyphHeight;    // 圆里蓝字的高 / 直径: 两种都在 0.44~0.49
        public int GlyphParts;        // 圆里蓝字有几块(小于全部 5% 的碎块不算): 感叹号 2(一竖 + 一点), 时钟 1
        public double GlyphOffset;    // 圆里蓝字的中心横向偏移 / 直径(时钟偏右 +0.03~0.04)
        public string Reason = "";
    }

    /// <summary>
    /// 蓝图(支付宝转账结果页)页头的状态检查: 到账失败 / 提交成功 的页面被当成付款凭证传上来。
    ///
    /// 经理 10/14 要把这类异常页挑出来。到账失败 / 提交成功 的页头和成功页长得不一样:
    ///     成功页(转账成功 到账成功 支付成功 代付成功 授权成功): 标题左边一个空心细圈里打勾, 下面是大号白字金额;
    ///     到账失败: 标题正上方一个实心白圆, 圆里是蓝色感叹号, 下面一行灰字"资金将原路返回你的付款账户", 页头里没有大金额;
    ///     提交成功: 同样的实心白圆, 圆里是蓝色时钟指针, 下面一行"预计 2 小时内到账 查看到账进度"。
    /// 只看图, 不做 OCR。
    ///
    /// 服务器 2026-09 蓝图库 678,191 张全扫: 判 到账失败 763 张, 再用 OCR 读页头核对 761 张确实是到账失败
    /// (1 张拼音字体读不出字, 1 张页头是感叹号、标题却写着"到账成功 资金已转入你的银行卡", 要人看);
    /// 判 提交成功 1,124 张, OCR 核对全部是提交成功(旧版圆里字形的判法多判了 1 张被通知条盖住的成功页, 现在这版不判)。
    /// 1.8 万张随机图上和 OCR 逐张比: 标题看得见的 到账失败 7/7、提交成功 16/16 都判出来, 判可疑而 OCR 说成功的 0 张。
    /// 判不出来的(设计如此): 通知条盖住页头的到账失败(1.8 万张里 1 张)、等待对方领取(白卡片页面)、
    /// 授权成功 / 代付成功(页头和成功页一模一样, 只能靠字分)。
    ///
    /// 怎么判:
    ///     页头 5%~20% 高度的平均色不是支付宝蓝(B &gt; 150 且 B &gt; R + 60)  -&gt; CannotDetermine
    ///     在 2%~17% 高度里找白色连通块(三个通道都 &gt; 215), 挑横向正中、直径 0.035~0.09 页宽、近正方、实心的:
    ///         圆里的蓝字(红通道 &lt; 160, 圆往里收一圈, 去掉小于 5% 的碎块) 字宽 &lt;= 0.16 直径 -&gt; 感叹号 -&gt; Suspicious, Failed
    ///         字宽 0.19~0.34 直径而且只有一块                                 -&gt; 时钟   -&gt; Pending(flagPending 为 true 时 Suspicious, 否则 Ok)
    ///         字高 &lt; 0.30 直径(压坏了)或者别的                                -&gt; CannotDetermine, UnknownDisk
    ///
    /// 压缩和缩放: 样本 7 张到账失败 + 12 张提交成功 + 347 张成功页各过 9 种 JPEG 压缩 / 缩放(JPEG 95~40、缩到 0.75 / 0.5、
    /// 缩放加 JPEG、宽 360 加 JPEG 60): 成功页一次也没判可疑; 提交成功每种都 12/12; 到账失败除了最狠的两种
    /// (缩到 0.5 加 JPEG 40 掉 1 张、宽 360 加 JPEG 60 掉 2 张, 都是判不了)都 7/7, 没有一张被认成提交成功或成功。
    ///     没有实心圆, 有成功页标题左边的空心细圈 -&gt; Ok, Success
    ///     都没有(通知条盖住了页头等)            -&gt; CannotDetermine
    ///
    /// 和 training/status_icon.py 逐步对应(那边是批量扫描用的参考实现), 改一边要同步改另一边。
    /// 只读入参, 内部用 ROI 视图(4 通道图先去掉 alpha)。无静态可变状态, 可多线程调用。
    /// 从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class StatusIconCheck
    {
        const double RoiTop = 0.02, RoiBottom = 0.17;   // 找图标的高度范围(占页高)
        const int White = 215;                          // 三个通道都大于这个算白

        // 实心白圆(到账失败 / 提交成功)
        const double DiskCenterTolerance = 0.04;        // 圆心离页面正中不超过 0.04 页宽
        const double DiskMinD = 0.035, DiskMaxD = 0.09; // 直径占页宽
        const double DiskMinTop = 0.035, DiskMaxTop = 0.13;
        const double MinAspect = 0.85, MaxAspect = 1.18;
        const double DiskMinWhite = 0.55, DiskMaxWhite = 0.80;
        const double DiskMinFilled = 0.68, DiskMaxFilled = 0.84;
        // 圆里的蓝字: 只认红通道 < GlyphRed 的像素(蓝字 R 只有 20~60, JPEG 在白圆里压出来的灰点 R 在 200 以上)
        const int GlyphRed = 160;
        const double BangMaxWidth = 0.16;               // 感叹号: 字宽 <= 0.16 直径
        const double ClockMinWidth = 0.19, ClockMaxWidth = 0.34;   // 时钟指针: 字宽 0.19~0.34 直径, 只有一块
        const double GlyphMinHeight = 0.30;             // 字高 < 0.30 直径: 压坏了, 认不出

        // 成功页标题左边的空心细圈
        const double RingMinCx = 0.25, RingMaxCx = 0.48;
        const double RingMinD = 0.025, RingMaxD = 0.065;
        const double RingMinWhite = 0.05, RingMaxWhite = 0.30;
        const double RingMinFilled = 0.68, RingMaxFilled = 0.86;

        /// <param name="image">8 位彩色图, 3 / 4 通道, 按 OpenCV 惯例视为 BGR(A)。不会被修改或释放。</param>
        /// <param name="flagPending">提交成功(钱还没到)也判 Suspicious; false 时判 Ok, Kind 仍是 Pending。</param>
        public static StatusIconResult Check(Mat image, bool flagPending = true)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new StatusIconResult { Verdict = StatusIconVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) { res.Reason = "不是 8 位图"; return res; }
            if (image.Channels() != 3 && image.Channels() != 4) { res.Reason = "不是彩色图"; return res; }

            Mat? bgrOwned = null;
            try
            {
                Mat bgr = image;
                if (image.Channels() == 4)
                {
                    bgrOwned = new Mat();
                    Cv2.CvtColor(image, bgrOwned, ColorConversionCodes.BGRA2BGR);
                    bgr = bgrOwned;
                }
                int H = bgr.Rows, W = bgr.Cols;
                if (W < 300 || H < 400) { res.Reason = "图太小"; return res; }

                // 页头颜色
                int hy0 = (int)(H * 0.05), hy1 = (int)(H * 0.20);
                using (var head = new Mat(bgr, new Rect(0, hy0, W, hy1 - hy0)))
                {
                    Scalar m = Cv2.Mean(head);
                    if (!(m.Val0 > 150 && m.Val0 > m.Val2 + 60)) { res.Reason = "不是蓝图"; return res; }
                }

                int y0 = (int)(H * RoiTop), y1 = (int)(H * RoiBottom);
                using var roi = new Mat(bgr, new Rect(0, y0, W, y1 - y0));
                using var white = new Mat();
                {
                    Mat[] ch = Cv2.Split(roi);
                    try
                    {
                        using var m0 = new Mat(); using var m1 = new Mat(); using var m2 = new Mat();
                        Cv2.Threshold(ch[0], m0, White, 1, ThresholdTypes.Binary);
                        Cv2.Threshold(ch[1], m1, White, 1, ThresholdTypes.Binary);
                        Cv2.Threshold(ch[2], m2, White, 1, ThresholdTypes.Binary);
                        Cv2.BitwiseAnd(m0, m1, white);
                        Cv2.BitwiseAnd(white, m2, white);
                    }
                    finally { foreach (var c in ch) c.Dispose(); }
                }
                using var labels = new Mat();
                using var stats = new Mat();
                using var cents = new Mat();
                int n = Cv2.ConnectedComponentsWithStats(white, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);

                bool ring = false;
                double rcx = 0, rtop = 0, rd = 0, rfw = 0, rfa = 0;
                for (int k = 1; k < n; k++)
                {
                    int x = stats.At<int>(k, (int)ConnectedComponentsTypes.Left);
                    int y = stats.At<int>(k, (int)ConnectedComponentsTypes.Top);
                    int w = stats.At<int>(k, (int)ConnectedComponentsTypes.Width);
                    int h = stats.At<int>(k, (int)ConnectedComponentsTypes.Height);
                    int area = stats.At<int>(k, (int)ConnectedComponentsTypes.Area);
                    double d = (double)w / W, cx = (x + w / 2.0) / W, top = (double)(y + y0) / H;
                    double aspect = (double)w / Math.Max(h, 1);
                    if (aspect < MinAspect || aspect > MaxAspect || d < Math.Min(DiskMinD, RingMinD) || d > DiskMaxD) continue;

                    Shape(labels, k, x, y, w, h, area, out double fw, out double fa, out Mat filled);
                    using (filled)
                    {
                        if (Math.Abs(cx - 0.5) <= DiskCenterTolerance && d >= DiskMinD && d <= DiskMaxD
                            && top >= DiskMinTop && top <= DiskMaxTop
                            && fw >= DiskMinWhite && fw <= DiskMaxWhite && fa >= DiskMinFilled && fa <= DiskMaxFilled)
                        {
                            double gw, gh, off;
                            int parts;
                            using (var crop = new Mat(roi, new Rect(x, y, w, h)))
                                Glyph(crop, filled, w, h, out gw, out gh, out parts, out off);
                            res.DiskFound = true;
                            res.CenterX = cx; res.Top = top; res.Diameter = d; res.WhiteFill = fw; res.FilledFill = fa;
                            res.GlyphWidth = gw; res.GlyphHeight = gh; res.GlyphParts = parts; res.GlyphOffset = off;
                            if (gh >= GlyphMinHeight && gw <= BangMaxWidth)
                            {
                                res.Kind = StatusIconKind.Failed;
                                res.Verdict = StatusIconVerdict.Suspicious;
                                res.Reason = $"标题上方实心白圆里是感叹号(字宽 {gw:F3} 直径, {parts} 块): 到账失败一类";
                            }
                            else if (gh >= GlyphMinHeight && gw >= ClockMinWidth && gw <= ClockMaxWidth && parts == 1)
                            {
                                res.Kind = StatusIconKind.Pending;
                                res.Verdict = flagPending ? StatusIconVerdict.Suspicious : StatusIconVerdict.Ok;
                                res.Reason = $"标题上方实心白圆里是时钟(字宽 {gw:F3} 直径): 提交成功, 钱还没到";
                            }
                            else
                            {
                                res.Kind = StatusIconKind.UnknownDisk;
                                res.Reason = $"标题上方有实心白圆, 圆里的图认不出(字宽 {gw:F3} 字高 {gh:F3} 直径, {parts} 块)";
                            }
                            return res;
                        }
                        if (cx >= RingMinCx && cx <= RingMaxCx && d >= RingMinD && d <= RingMaxD
                            && fw >= RingMinWhite && fw <= RingMaxWhite && fa >= RingMinFilled && fa <= RingMaxFilled)
                        {
                            ring = true;
                            rcx = cx; rtop = top; rd = d; rfw = fw; rfa = fa;   // 和 Python 一样留最后一个
                        }
                    }
                }
                if (ring)
                {
                    res.RingFound = true;
                    res.CenterX = rcx; res.Top = rtop; res.Diameter = rd; res.WhiteFill = rfw; res.FilledFill = rfa;
                    res.Kind = StatusIconKind.Success;
                    res.Verdict = StatusIconVerdict.Ok;
                    res.Reason = "标题左边空心细圈打勾: 成功页的页头";
                }
                else
                {
                    res.Reason = "没找到图标(通知条盖住了页头等)";
                }
                return res;
            }
            finally
            {
                bgrOwned?.Dispose();
            }
        }

        /// <summary>一个白色连通块: 白色占外接框、填掉里面的洞以后占外接框、填好的掩膜(0/255, 调用方释放)。</summary>
        static void Shape(Mat labels, int k, int x, int y, int w, int h, int area, out double fw, out double fa, out Mat filled)
        {
            using var lab = new Mat(labels, new Rect(x, y, w, h));
            using var comp = new Mat();
            Cv2.Compare(lab, new Scalar(k), comp, CmpType.EQ);          // 255 / 0
            filled = comp.Clone();
            Cv2.FindContours(comp, out Point[][] cnts, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            Cv2.DrawContours(filled, cnts, -1, Scalar.All(255), -1);
            fw = (double)area / (w * h);
            fa = (double)Cv2.CountNonZero(filled) / (w * h);
        }

        /// <summary>圆里的蓝字: 字宽 / 直径、字高 / 直径、块数、中心横向偏移 / 直径。见 GlyphRed 的注。</summary>
        static void Glyph(Mat crop, Mat filled, int w, int h, out double gw, out double gh, out int parts, out double off)
        {
            gw = 0; gh = 0; parts = 0; off = 0;
            using var inner = new Mat();
            using (var kernel = new Mat(3, 3, MatType.CV_8UC1, Scalar.All(1)))
                Cv2.Erode(filled, inner, kernel, null, Math.Max(1, w / 20));
            using var red = new Mat();
            Cv2.ExtractChannel(crop, red, 2);
            using var ink = new Mat();
            Cv2.Threshold(red, ink, GlyphRed - 1, 255, ThresholdTypes.BinaryInv);   // R < 160 -> 255
            Cv2.BitwiseAnd(ink, inner, ink);
            int total = Cv2.CountNonZero(ink);
            if (total < 4) return;
            using var labels = new Mat();
            using var stats = new Mat();
            using var cents = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(ink, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
            using var keep = new Mat(ink.Size(), MatType.CV_8UC1, Scalar.All(0));
            using var one = new Mat();
            for (int j = 1; j < n; j++)
            {
                if (stats.At<int>(j, (int)ConnectedComponentsTypes.Area) < 0.05 * total) continue;
                parts++;
                Cv2.Compare(labels, new Scalar(j), one, CmpType.EQ);
                Cv2.BitwiseOr(keep, one, keep);
            }
            if (parts == 0) return;
            Rect bb = Cv2.BoundingRect(keep);
            gw = (double)bb.Width / w;
            gh = (double)bb.Height / h;
            Moments mo = Cv2.Moments(keep, true);
            off = (mo.M10 / mo.M00 - w / 2.0) / w;
        }
    }
}
