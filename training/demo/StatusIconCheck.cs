#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace Ssp
{
    public enum StatusIconVerdict
    {
        CannotDetermine = 0,   // 判不了(不是蓝图也不是白图账单详情、通知条盖住了页头、圆里的图认不出、彩色字对不上等), 走正常流程
        Ok = 1,                // 蓝图成功页的页头(标题左边空心细圈打勾 + 大号金额)
        Suspicious = 2,        // 异常页: 到账失败 / 提交成功(flagPending) / 等待对方领取 / 白图未完成 / 白图已全额退款 / 白图退款记录
    }

    public enum StatusIconKind
    {
        None = 0,
        Failed = 1,             // 蓝图 到账失败 资金将原路返回你的付款账户
        Pending = 2,            // 蓝图 提交成功 预计 2 小时内到账(钱还没到)
        Success = 3,            // 蓝图 转账成功 / 到账成功 / 支付成功 / 代付成功 / 授权成功 这一类的页头
        UnknownDisk = 4,        // 蓝图 有实心白圆, 但圆里的图认不出
        Waiting = 5,            // 蓝图 转账详情 等待对方领取(白卡片里实心绿圆, 圆里白色时钟)
        WhitePending = 6,       // 白图账单详情 金额下面橙字 3 个字或 6 个字(处理中 领取中 待到账 缴费中 / 等待对方领取 等待对方付款 等待确认收货)
        WhiteRefunded = 7,      // 白图账单详情 金额下面红字 5 个字(已全额退款)
        WhiteRefundRecord = 8,  // 白图账单详情 退款记录(金额上面蓝圆白色"退"字头像, 下面灰字 退款成功)
        WhiteUnknownText = 9,   // 白图 金额下面有彩色字, 但字数或位置对不上(支付成功 交易成功 等待付款 这类 4 个字、浮窗、印章), 判不了
        WhiteNotAlipayCard = 10,// 白图 像是异常, 但不是支付宝账单详情的卡片(收银台、银行回单、别的 App), 判不了
    }

    public sealed class StatusIconResult
    {
        public StatusIconVerdict Verdict;
        public StatusIconKind Kind;
        // ---- 蓝图页头 ----
        public bool DiskFound;        // 找到了标题正上方的实心白圆(或 等待对方领取 的实心绿圆)
        public bool RingFound;        // 找到了成功页标题左边的空心细圈
        public double CenterX;        // 圆心横坐标 / 页宽
        public double Top;            // 圆顶 / 页高
        public double Diameter;       // 圆的宽 / 页宽
        public double WhiteFill;      // 白色像素 / 外接框面积(绿圆不量, 是 0)
        public double FilledFill;     // 填掉圆里的字以后 / 外接框面积(正圆 0.785)
        public double GlyphWidth;     // 圆里字的宽 / 直径: 感叹号 0.05~0.13, 时钟 0.22~0.28; 绿圆里白色时钟 0.20~0.24
        public double GlyphHeight;    // 圆里字的高 / 直径
        public int GlyphParts;        // 圆里字有几块(小于全部 5% 的碎块不算): 感叹号 2(一竖 + 一点), 时钟 1
        public double GlyphOffset;    // 圆里字的中心横向偏移 / 直径
        // ---- 白图账单详情 ----
        public bool AmountFound;      // 找到了卡片里的金额那一行
        public double AmountTop;      // 金额行上沿 / 页高
        public double AmountBottom;   // 金额行下沿 / 页高
        public double Orange;         // 金额下面 橙色像素 / 金额字高^2
        public double Red;            // 金额下面 红色像素 / 金额字高^2
        public double LineWidth;      // 彩色字那一行的宽 / 金额字高: 3 个字 1.5~1.9, 4 个字 2.0 以上, 5 个字 2.6~3.0, 6 个字 3.1~3.9
        public double LineHeight;     // 彩色字那一行的高 / 金额字高
        public double LineOffset;     // 彩色字中心横向偏移 / 页宽
        public double TuiIou = -1;    // 金额上面蓝圆头像里的白字和"退"字模板的 IoU; 没有蓝圆头像是 -1
        public string Card = "";      // 支付宝账单详情卡片: A 支付宝 / A0 支付宝老版整页白 / O 别的 App / 空 没有这种卡片; 只在要判可疑时才量
        public string Reason = "";
    }

    /// <summary>
    /// 付款凭证里的异常页: 钱没到、退回来了、还在处理, 却被当成付款凭证传上来。只看图, 不做 OCR。
    ///
    /// 经理 10/14 要把这类异常页挑出来。一共认 6 种, 都判 Suspicious:
    ///   蓝图(支付宝转账结果页, 蓝色页头):
    ///     到账失败      标题正上方一个实心白圆, 圆里是蓝色感叹号, 下面一行"资金将原路返回你的付款账户"
    ///     提交成功      同样的实心白圆, 圆里是蓝色时钟指针, 下面一行"预计 2 小时内到账"(flagPending 为 false 时判 Ok)
    ///     等待对方领取  蓝色顶栏 + 白卡片, 卡片里一个大的实心绿圆, 圆里白色时钟
    ///   白图(支付宝账单详情, 白色页头, 卡片里一行大号金额):
    ///     白图未完成      金额正下方一行橙字, 3 个字(处理中 领取中 待到账 缴费中)或 6 个字(等待对方领取 等待对方付款 等待确认收货)
    ///     白图已全额退款  金额正下方一行红字, 5 个字(已全额退款)
    ///     白图退款记录    金额上面居中一个蓝色实心圆头像, 圆里白色"退"字(下面一行灰字 退款成功)
    ///   蓝图成功页(标题左边空心细圈打勾 + 大号金额)判 Ok; 别的都判不了(CannotDetermine), 走正常流程。
    ///
    /// 白图为什么要数字数: 淘宝这类商户付款的 支付成功 / 交易成功 也是橙字或红字, 和没付款的 等待付款 一样是 4 个字, 分不开,
    /// 所以 4 个字的一律判不了; 盖在金额下面的浮窗、贴纸、印章, 宽高位置也对不上, 判不了。
    /// 字数按 彩色字那一行的宽 / 金额字高 算, 行宽用宽松的暖色量(压缩以后浅的笔画还在, 宽不会缩), 界线都偏向判不了。
    /// 白图的三种还要过一道门槛: 页面得是支付宝账单详情的卡片(和 TextColorCheck 一样的量法: 左灰标签右深色取值、对齐的行 >= 3,
    /// 页边是支付宝灰底 238~250 且标签色 148~165; 或者整页白但取值是 #333 的老版)。收银台、银行回单、别的 App 金额下面也常有红字,
    /// 过不了这道门槛, 判不了。宽小于 500 或高小于 900 的图量不了卡片, 白图的三种都判不了。
    ///
    /// 测试结果见 使用说明。和 training/status_icon.py 逐步对应(那边是批量扫描用的参考实现), 改一边要同步改另一边。
    /// 只读入参, 内部用 ROI 视图和拷出来的像素(4 通道图先去掉 alpha)。无静态可变状态, 可多线程调用。
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

        // 等待对方领取: 顶栏(2%~8% 高度)是支付宝蓝; 5%~40% 高度里横向居中的实心绿圆(G > 140, G - R > 60, G - B > 30),
        // 圆里一块白色时钟(三个通道都 > 200)
        const double GreenRoiTop = 0.05, GreenRoiBottom = 0.40;
        const double GreenMinD = 0.12, GreenMaxD = 0.22;
        const double GreenMinTop = 0.10, GreenMaxTop = 0.25;
        const double GreenMinFilled = 0.70, GreenMaxFilled = 0.86;
        const double GreenGlyphMinW = 0.17, GreenGlyphMaxW = 0.30;
        const double GreenGlyphMinH = 0.33, GreenGlyphMaxH = 0.50;

        // 白图金额: 12%~40% 高度里字最高、横向居中、宽 0.12~0.6 页宽、至少 3 个字的黑字行(三个通道最大值 < 100)
        const double AmountRoiTop = 0.12, AmountRoiBottom = 0.40;
        const double AmountMaxW = 0.60;
        const int AmountDark = 100;
        const double AmountMinH = 0.012, AmountMaxH = 0.05;   // 一个字的高 占页高
        // 金额下面那行彩色字(金额下面 0.15~1.4 个金额字高、金额中心左右 0.3 页宽)
        //   有没有彩色字、橙还是红: 严格的暖色 R > 180, R - B > 90, R - G > 40; 橙 = 再 G - B >= 40, 其余算红
        //   行的宽高: 宽松的暖色 R > 150, R - B > 40, R - G > 15
        const double WarmMin = 0.10;                    // 橙 + 红 像素 / 金额字高^2 >= 这个: 有彩色字
        static readonly (double lo, double hi)[] OrangeWidths = { (1.40, 1.95), (3.05, 3.95) };   // 3 个字 / 6 个字
        const double RedMinW = 2.60, RedMaxW = 3.20;    // 5 个字
        const double LineMinH = 0.30, LineMaxH = 1.00;
        const double LineMaxOffset = 0.05;
        const double TuiMinIou = 0.70;
        const int TuiN = 32;

        // 支付宝账单详情的卡片(和 TextColorCheck 同一个量法)
        const int CardEdgeLo = 238, CardEdgeHi = 250, CardLabelLo = 148, CardLabelHi = 165, CardGreyLo = 38, CardGreyHi = 64;

        /// <param name="image">8 位彩色图, 3 / 4 通道, 按 OpenCV 惯例视为 BGR(A)。不会被修改或释放。</param>
        /// <param name="flagPending">蓝图 提交成功(钱还没到)也判 Suspicious; false 时判 Ok, Kind 仍是 Pending。别的种类不受影响。</param>
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

                // 页头颜色: 支付宝蓝 -> 蓝图; 三个通道平均都 > 200 -> 白图账单详情
                int hy0 = (int)(H * 0.05), hy1 = (int)(H * 0.20);
                Scalar m;
                using (var head = new Mat(bgr, new Rect(0, hy0, W, hy1 - hy0)))
                    m = Cv2.Mean(head);
                if (!(m.Val0 > 150 && m.Val0 > m.Val2 + 60))
                {
                    if (Math.Min(m.Val0, Math.Min(m.Val1, m.Val2)) > 200) { WhitePage(bgr, res); return res; }
                    res.Reason = "不是蓝图, 也不是白图账单详情";
                    return res;
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
                else if (GreenWaiting(bgr, res))
                {
                    res.Kind = StatusIconKind.Waiting;
                    res.Verdict = StatusIconVerdict.Suspicious;
                    res.Reason = $"白卡片里实心绿圆, 圆里白色时钟(字宽 {res.GlyphWidth:F3} 字高 {res.GlyphHeight:F3} 直径): 等待对方领取, 钱还没到";
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

        // ================================ 蓝图 ================================

        /// <summary>一个白色连通块: 白色占外接框、填掉里面的洞以后占外接框、填好的掩膜(0/255, 调用方释放)。</summary>
        static void Shape(Mat labels, int k, int x, int y, int w, int h, int area, out double fw, out double fa, out Mat filled)
        {
            filled = Filled(labels, k, x, y, w, h);
            fw = (double)area / (w * h);
            fa = (double)Cv2.CountNonZero(filled) / (w * h);
        }

        /// <summary>连通块 k 在外接框里的掩膜, 里面的洞填满(0/255, 调用方释放)。</summary>
        static Mat Filled(Mat labels, int k, int x, int y, int w, int h)
        {
            using var lab = new Mat(labels, new Rect(x, y, w, h));
            using var comp = new Mat();
            Cv2.Compare(lab, new Scalar(k), comp, CmpType.EQ);          // 255 / 0
            var filled = comp.Clone();
            Cv2.FindContours(comp, out Point[][] cnts, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            Cv2.DrawContours(filled, cnts, -1, Scalar.All(255), -1);
            return filled;
        }

        /// <summary>圆里的蓝字: 字宽 / 直径、字高 / 直径、块数、中心横向偏移 / 直径。见 GlyphRed 的注。</summary>
        static void Glyph(Mat crop, Mat filled, int w, int h, out double gw, out double gh, out int parts, out double off)
        {
            using var inner = new Mat();
            using (var kernel = new Mat(3, 3, MatType.CV_8UC1, Scalar.All(1)))
                Cv2.Erode(filled, inner, kernel, null, Math.Max(1, w / 20));
            using var red = new Mat();
            Cv2.ExtractChannel(crop, red, 2);
            using var ink = new Mat();
            Cv2.Threshold(red, ink, GlyphRed - 1, 255, ThresholdTypes.BinaryInv);   // R < 160 -> 255
            Cv2.BitwiseAnd(ink, inner, ink);
            InkShape(ink, w, h, out gw, out gh, out parts, out off);
        }

        /// <summary>掩膜(0/255)里去掉小于全部 5% 的碎块, 剩下的: 宽 / w、高 / h、块数、中心横向偏移 / w。</summary>
        static void InkShape(Mat ink, int w, int h, out double gw, out double gh, out int parts, out double off)
        {
            gw = 0; gh = 0; parts = 0; off = 0;
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

        /// <summary>蓝图 转账详情页的 等待对方领取。找到时把圆和字填进 res。</summary>
        static bool GreenWaiting(Mat bgr, StatusIconResult res)
        {
            int H = bgr.Rows, W = bgr.Cols;
            int ty0 = (int)(H * 0.02), ty1 = (int)(H * 0.08);
            using (var topBand = new Mat(bgr, new Rect(0, ty0, W, ty1 - ty0)))
            {
                Scalar tm = Cv2.Mean(topBand);
                if (!(tm.Val0 > 150 && tm.Val0 > tm.Val2 + 60)) return false;
            }
            int y0 = (int)(H * GreenRoiTop), y1 = (int)(H * GreenRoiBottom);
            int rh = y1 - y0;
            byte[] px = Pixels(bgr, new Rect(0, y0, W, rh));
            var gm = new byte[rh * W];
            for (int i = 0, j = 0; i < gm.Length; i++, j += 3)
            {
                int b = px[j], g = px[j + 1], r = px[j + 2];
                if (g > 140 && g - r > 60 && g - b > 30) gm[i] = 1;
            }
            using var green = MaskMat(gm, rh, W);
            using var labels = new Mat();
            using var stats = new Mat();
            using var cents = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(green, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
            for (int k = 1; k < n; k++)
            {
                int x = stats.At<int>(k, (int)ConnectedComponentsTypes.Left);
                int y = stats.At<int>(k, (int)ConnectedComponentsTypes.Top);
                int w = stats.At<int>(k, (int)ConnectedComponentsTypes.Width);
                int h = stats.At<int>(k, (int)ConnectedComponentsTypes.Height);
                double d = (double)w / W, cx = (x + w / 2.0) / W, top = (double)(y + y0) / H;
                double aspect = (double)w / Math.Max(h, 1);
                if (Math.Abs(cx - 0.5) > DiskCenterTolerance || d < GreenMinD || d > GreenMaxD || top < GreenMinTop || top > GreenMaxTop
                    || aspect < MinAspect || aspect > MaxAspect) continue;
                using var filled = Filled(labels, k, x, y, w, h);
                double fa = (double)Cv2.CountNonZero(filled) / (w * h);
                if (fa < GreenMinFilled || fa > GreenMaxFilled) continue;
                using var inner = new Mat();
                using (var kernel = new Mat(3, 3, MatType.CV_8UC1, Scalar.All(1)))
                    Cv2.Erode(filled, inner, kernel, null, Math.Max(1, w / 20));
                // 圆里的白字: 三个通道都 > 200
                var wm = new byte[w * h];
                for (int yy = 0; yy < h; yy++)
                    for (int xx = 0; xx < w; xx++)
                    {
                        int j = ((y + yy) * W + x + xx) * 3;
                        if (px[j] > 200 && px[j + 1] > 200 && px[j + 2] > 200) wm[yy * w + xx] = 255;
                    }
                using var ink = MaskMat(wm, h, w);
                Cv2.BitwiseAnd(ink, inner, ink);
                InkShape(ink, w, h, out double gw, out double gh, out int parts, out double off);
                if (parts == 0) continue;
                if (gw >= GreenGlyphMinW && gw <= GreenGlyphMaxW && gh >= GreenGlyphMinH && gh <= GreenGlyphMaxH && parts == 1)
                {
                    res.DiskFound = true;
                    res.CenterX = cx; res.Top = top; res.Diameter = d; res.WhiteFill = 0; res.FilledFill = fa;
                    res.GlyphWidth = gw; res.GlyphHeight = gh; res.GlyphParts = parts; res.GlyphOffset = off;
                    return true;
                }
            }
            return false;
        }

        // ================================ 白图账单详情 ================================

        static void WhitePage(Mat bgr, StatusIconResult res)
        {
            int H = bgr.Rows;
            if (!AmountRow(bgr, out int a0, out int a1, out int ax0, out int ax1))
            {
                res.Reason = "白图: 找不到金额那一行";
                return;
            }
            res.AmountFound = true;
            res.AmountTop = (double)a0 / H;
            res.AmountBottom = (double)a1 / H;
            StatusLine(bgr, a0, a1, ax0, ax1, res);
            var kind = StatusIconKind.None;
            string what = "";
            if (res.Orange + res.Red >= WarmMin)
            {
                bool line = res.LineHeight >= LineMinH && res.LineHeight <= LineMaxH && Math.Abs(res.LineOffset) <= LineMaxOffset;
                string geo = $"宽 {res.LineWidth:F2} 高 {res.LineHeight:F2} 个金额字高, 偏 {res.LineOffset:F3} 页宽";
                if (res.Red > res.Orange)
                {
                    if (line && res.LineWidth >= RedMinW && res.LineWidth <= RedMaxW)
                    { kind = StatusIconKind.WhiteRefunded; what = $"金额下面一行红字 5 个字({geo}): 已全额退款"; }
                    else
                    { kind = StatusIconKind.WhiteUnknownText; what = $"金额下面有红字, 字数或位置对不上({geo})"; }
                }
                else
                {
                    bool fit = false;
                    foreach (var (lo, hi) in OrangeWidths) if (res.LineWidth >= lo && res.LineWidth <= hi) fit = true;
                    if (line && fit)
                    {
                        kind = StatusIconKind.WhitePending;
                        what = $"金额下面一行橙字 {(res.LineWidth < 2.5 ? 3 : 6)} 个字({geo}): 处理中 / 领取中 / 等待对方领取 一类, 钱还没到对方";
                    }
                    else
                    { kind = StatusIconKind.WhiteUnknownText; what = $"金额下面有橙字, 字数或位置对不上({geo}), 支付成功 交易成功 这类 4 个字的不判"; }
                }
            }
            else
            {
                res.TuiIou = TuiIou(bgr, a0);
                if (res.TuiIou >= TuiMinIou)
                { kind = StatusIconKind.WhiteRefundRecord; what = $"金额上面蓝圆头像里是\"退\"字(IoU {res.TuiIou:F2}): 退款记录"; }
            }
            if (kind == StatusIconKind.None)
            {
                res.Reason = "白图: 没看出异常";
                return;
            }
            if (kind == StatusIconKind.WhiteUnknownText)
            {
                res.Kind = kind;
                res.Reason = "白图: " + what + ", 判不了";
                return;
            }
            res.Card = CardGate(bgr);
            if (res.Card == "A" || res.Card == "A0")
            {
                res.Kind = kind;
                res.Verdict = StatusIconVerdict.Suspicious;
                res.Reason = "白图账单详情: " + what;
            }
            else
            {
                res.Kind = StatusIconKind.WhiteNotAlipayCard;
                res.Reason = "白图: " + what + ", 但页面不是支付宝账单详情的卡片, 判不了";
            }
        }

        /// <summary>金额那一行(12%~40% 高度里字最高、横向居中、宽 0.12~0.6 页宽、至少 3 个字的黑字行): 上、下(不含)、左、右(不含)。</summary>
        static bool AmountRow(Mat bgr, out int top, out int bottom, out int left, out int right)
        {
            top = bottom = left = right = 0;
            int H = bgr.Rows, W = bgr.Cols;
            int y0 = (int)(H * AmountRoiTop), y1 = (int)(H * AmountRoiBottom);
            int rh = y1 - y0;
            byte[] px = Pixels(bgr, new Rect(0, y0, W, rh));
            var dm = new byte[rh * W];
            for (int i = 0, j = 0; i < dm.Length; i++, j += 3)
                if (Math.Max(px[j], Math.Max(px[j + 1], px[j + 2])) < AmountDark) dm[i] = 1;
            using var dark = MaskMat(dm, rh, W);
            using var labels = new Mat();
            using var stats = new Mat();
            using var cents = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(dark, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
            var comps = new List<(int x, int y, int w, int h)>();
            for (int k = 1; k < n; k++)
            {
                int w = stats.At<int>(k, (int)ConnectedComponentsTypes.Width);
                int h = stats.At<int>(k, (int)ConnectedComponentsTypes.Height);
                if (AmountMinH * H <= h && h <= AmountMaxH * H && w <= 0.12 * W)
                    comps.Add((stats.At<int>(k, (int)ConnectedComponentsTypes.Left), stats.At<int>(k, (int)ConnectedComponentsTypes.Top), w, h));
            }
            // 按上沿从上往下(稳定排序, 和 Python sorted 一样), 和已有的行上下重叠超过矮的那个一半就并进去
            var rows = new List<int[]>();     // 左 上 右 下 字数 最高的字
            foreach (var (x, y, w, h) in comps.OrderBy(c => c.y))
            {
                bool merged = false;
                foreach (var r in rows)
                {
                    if (Math.Min(r[3], y + h) - Math.Max(r[1], y) > 0.5 * Math.Min(h, r[3] - r[1]))
                    {
                        r[0] = Math.Min(r[0], x); r[1] = Math.Min(r[1], y); r[2] = Math.Max(r[2], x + w); r[3] = Math.Max(r[3], y + h);
                        r[4]++; r[5] = Math.Max(r[5], h);
                        merged = true;
                        break;
                    }
                }
                if (!merged) rows.Add(new[] { x, y, x + w, y + h, 1, h });
            }
            int[]? best = null;
            foreach (var r in rows)
            {
                double mid = (r[0] + r[2]) / 2.0 / W;
                if (r[4] < 3 || mid < 0.35 || mid > 0.65 || r[2] - r[0] < 0.12 * W || r[2] - r[0] > AmountMaxW * W) continue;
                if (best == null || r[5] > best[5]) best = r;    // 字最高的那行; 一样高取先找到的
            }
            if (best == null) return false;
            top = best[1] + y0; bottom = best[3] + y0; left = best[0]; right = best[2];
            return true;
        }

        /// <summary>金额下面那行彩色字: 橙 / 红像素 / 金额字高^2, 行的宽高 / 金额字高, 中心横向偏移 / 页宽(没有彩色字时后三个是 0)。</summary>
        static void StatusLine(Mat bgr, int a0, int a1, int ax0, int ax1, StatusIconResult res)
        {
            int H = bgr.Rows, W = bgr.Cols;
            int h = a1 - a0, cx = (ax0 + ax1) / 2;
            int by0 = a1 + (int)(0.15 * h), by1 = Math.Min(H, a1 + (int)(1.4 * h));
            int bx0 = Math.Max(0, cx - (int)(0.3 * W)), bx1 = Math.Min(W, cx + (int)(0.3 * W));
            if (by1 <= by0 || bx1 <= bx0) return;
            int bw = bx1 - bx0, bh = by1 - by0;
            byte[] px = Pixels(bgr, new Rect(bx0, by0, bw, bh));
            int nOrange = 0, nRed = 0, nLoose = 0;
            var lm = new byte[bw * bh];
            for (int i = 0, j = 0; i < lm.Length; i++, j += 3)
            {
                int b = px[j], g = px[j + 1], r = px[j + 2];
                if (r > 180 && r - b > 90 && r - g > 40)
                {
                    // 橙 / 红按 G - B 分: JPEG 把红字和白底混成粉色时 G 会升高, 但 G 和 B 差不多; 橙字混白以后 G 仍比 B 高很多
                    if (g - b >= 40) nOrange++; else nRed++;
                }
                if (r > 150 && r - b > 40 && r - g > 15) { lm[i] = 1; nLoose++; }
            }
            res.Orange = (double)nOrange / ((double)h * h);
            res.Red = (double)nRed / ((double)h * h);
            if (nOrange + nRed < 4 || nLoose < 4) return;
            // 一行字的笔画先横向连起来(闭运算, 核 3 行 x 金额字高 / 6 列), 再取面积 >= 宽松暖色像素 5% 的块合起来的外接框
            using var loose = MaskMat(lm, bh, bw);
            using var joined = new Mat();
            using (var kernel = new Mat(3, Math.Max(3, h / 6), MatType.CV_8UC1, Scalar.All(1)))
                Cv2.MorphologyEx(loose, joined, MorphTypes.Close, kernel);
            using var labels = new Mat();
            using var stats = new Mat();
            using var cents = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(joined, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
            int xs0 = int.MaxValue, xs1 = int.MinValue, ys0 = int.MaxValue, ys1 = int.MinValue;
            for (int k = 1; k < n; k++)
            {
                if (stats.At<int>(k, (int)ConnectedComponentsTypes.Area) < 0.05 * nLoose) continue;
                int x = stats.At<int>(k, (int)ConnectedComponentsTypes.Left), y = stats.At<int>(k, (int)ConnectedComponentsTypes.Top);
                xs0 = Math.Min(xs0, x); xs1 = Math.Max(xs1, x + stats.At<int>(k, (int)ConnectedComponentsTypes.Width));
                ys0 = Math.Min(ys0, y); ys1 = Math.Max(ys1, y + stats.At<int>(k, (int)ConnectedComponentsTypes.Height));
            }
            if (xs0 == int.MaxValue) return;
            res.LineWidth = (double)(xs1 - xs0) / h;
            res.LineHeight = (double)(ys1 - ys0) / h;
            res.LineOffset = (bx0 + (xs0 + xs1) / 2.0 - cx) / W;
        }

        /// <summary>"退"字模板(32x32, '#' 是字), 从真图的退款记录头像里取的。</summary>
        static readonly string[] TuiRef =
        {
            "................................", "................................",
            "................................", "................................",
            "................................", "................................",
            "................................", "................................",
            "........##...##########.........", "........###..##########.........",
            ".........##..###.....##.........", ".........##..##########.........",
            ".............##########.........", ".......#####.###.....##.........",
            ".......#####.##########.........", "........####.###########........",
            ".........###.###..#...##........", ".........###.###.#######........",
            ".........###.###.######.........", ".........###.######.####........",
            "........####.#####...####.......", ".......########........##.......",
            ".......###.##############.......", ".......##...#############.......",
            "................................", "................................",
            "................................", "................................",
            "................................", "................................",
            "................................", "................................",
        };

        /// <summary>金额上面居中的蓝色实心圆头像, 圆里的白字缩到 32x32 和"退"字模板比的 IoU; 没有这种头像返回 -1。</summary>
        static double TuiIou(Mat bgr, int amountTop)
        {
            int H = bgr.Rows, W = bgr.Cols;
            int y0 = Math.Max(0, amountTop - (int)(0.13 * H));
            int rh = amountTop - y0;
            if (rh <= 0) return -1;
            using var roi = new Mat(bgr, new Rect(0, y0, W, rh));
            byte[] px = Pixels(bgr, new Rect(0, y0, W, rh));
            var bm = new byte[rh * W];
            for (int i = 0, j = 0; i < bm.Length; i++, j += 3)
            {
                int b = px[j], g = px[j + 1], r = px[j + 2];
                if (b > 190 && b - r > 110 && g > 90 && g < 190) bm[i] = 1;
            }
            using var blue = MaskMat(bm, rh, W);
            using var labels = new Mat();
            using var stats = new Mat();
            using var cents = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(blue, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
            for (int k = 1; k < n; k++)
            {
                int x = stats.At<int>(k, (int)ConnectedComponentsTypes.Left);
                int y = stats.At<int>(k, (int)ConnectedComponentsTypes.Top);
                int w = stats.At<int>(k, (int)ConnectedComponentsTypes.Width);
                int h = stats.At<int>(k, (int)ConnectedComponentsTypes.Height);
                double aspect = (double)w / Math.Max(h, 1);
                if (w < 0.04 * W || Math.Abs((x + w / 2.0) / W - 0.5) > 0.05 || aspect < MinAspect || aspect > MaxAspect) continue;
                using var filled = Filled(labels, k, x, y, w, h);
                if ((double)Cv2.CountNonZero(filled) / (w * h) < 0.65) continue;
                using var gray = new Mat();
                using (var crop = new Mat(roi, new Rect(x, y, w, h)))
                using (var g0 = new Mat())
                {
                    Cv2.CvtColor(crop, g0, ColorConversionCodes.BGR2GRAY);
                    Cv2.Resize(g0, gray, new Size(TuiN, TuiN), 0, 0, InterpolationFlags.Area);
                }
                using var disk = new Mat();
                Cv2.Resize(filled, disk, new Size(TuiN, TuiN), 0, 0, InterpolationFlags.Area);
                using var diskBin = new Mat();
                Cv2.Threshold(disk, diskBin, 127, 255, ThresholdTypes.Binary);        // > 127
                using var inner = new Mat();
                using (var kernel = new Mat(3, 3, MatType.CV_8UC1, Scalar.All(1)))
                    Cv2.Erode(diskBin, inner, kernel, null, 2);
                var gv = new byte[TuiN * TuiN];
                var iv = new byte[TuiN * TuiN];
                using (var gc = gray.Clone()) Marshal.Copy(gc.Data, gv, 0, gv.Length);
                using (var ic = inner.Clone()) Marshal.Copy(ic.Data, iv, 0, iv.Length);
                int inter = 0, uni = 0;
                for (int yy = 0; yy < TuiN; yy++)
                    for (int xx = 0; xx < TuiN; xx++)
                    {
                        bool mm = gv[yy * TuiN + xx] > 200 && iv[yy * TuiN + xx] == 255;
                        bool tt = TuiRef[yy][xx] == '#';
                        if (mm && tt) inter++;
                        if (mm || tt) uni++;
                    }
                return (double)inter / Math.Max(1, uni);     // 和 Python 一样: 第一个像样的蓝圆就定了
            }
            return -1;
        }

        /// <summary>
        /// 支付宝账单详情的卡片: 和 TextColorCheck(118)同一个量法, 只要它的量, 不判真假。
        /// 返回 A(支付宝: 对齐的行 >= 3, 页边灰底 238~250, 标签色 148~165) / A0(整页白但取值是 #333 的行占 60% 以上, 支付宝老版) /
        /// O(有这种卡片但不是支付宝的) / ""(没有这种卡片、深色模式、图太小)。
        /// </summary>
        static string CardGate(Mat bgr)
        {
            int W = bgr.Width, H = bgr.Height;
            if (W < 500 || H < 900) return "";
            int y0 = (int)(0.095 * H), y1 = (int)(0.96 * H);
            int x0 = (int)(0.03 * W), x1 = (int)(0.97 * W);
            using var grayM = new Mat();
            Cv2.CvtColor(bgr, grayM, ColorConversionCodes.BGR2GRAY);
            var ch = Cv2.Split(bgr);
            using var mx = new Mat();
            using var mn = new Mat();
            using var chromaM = new Mat();
            try
            {
                Cv2.Max(ch[0], ch[1], mx); Cv2.Max(mx, ch[2], mx);
                Cv2.Min(ch[0], ch[1], mn); Cv2.Min(mn, ch[2], mn);
                Cv2.Subtract(mx, mn, chromaM);
            }
            finally
            {
                foreach (var cm in ch) cm.Dispose();
            }
            var g = new byte[W * H];
            var c = new byte[W * H];
            Marshal.Copy(grayM.Data, g, 0, g.Length);
            Marshal.Copy(chromaM.Data, c, 0, c.Length);

            // 深色模式: 正文区域的中位灰度 < 128
            var gh = new int[256];
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++) gh[g[y * W + x]]++;
            if (MedianTwice(gh, (y1 - y0) * (x1 - x0)) < 256) return "";

            // 一行里深色字或灰色标签字(不彩色、灰度 <= 185) >= 3 个像素算有字
            var on = new bool[y1 - y0];
            for (int y = y0; y < y1; y++)
            {
                int cnt = 0;
                for (int x = x0; x < x1 && cnt < 3; x++)
                {
                    int i = y * W + x;
                    if (g[i] <= 185 && c[i] < 30) cnt++;
                }
                on[y - y0] = cnt >= 3;
            }
            int lx1 = (int)(0.27 * W), vx0 = (int)(0.22 * W);
            var cand = new List<(int y, int yb, int lcore, int vcore, int ll, int vl)>();
            var lh = new int[256];
            var vh = new int[256];
            foreach (var (ra, rb) in Runs(on))
            {
                int a = ra + y0, b = rb + y0;
                int hh = b - a + 1;
                if (hh < 0.008 * H || hh > 0.04 * H) continue;
                Array.Clear(lh); Array.Clear(vh);
                int ln = 0, vn = 0, ll = -1, vl = -1;
                for (int x = x0; x < lx1; x++)
                    for (int y = a; y <= b; y++)
                    {
                        int i = y * W + x;
                        if (!(g[i] >= 120 && g[i] <= 185 && c[i] < 30)) continue;
                        lh[g[i]]++; ln++;
                        if (ll < 0) ll = x;
                    }
                for (int x = vx0; x < x1; x++)
                    for (int y = a; y <= b; y++)
                    {
                        int i = y * W + x;
                        if (!(g[i] < 120 && c[i] < 30)) continue;
                        vh[g[i]]++; vn++;
                        if (vl < 0) vl = x;
                    }
                if (ln < 10 || vn < 10) continue;
                int lcore = ArgMax(lh), vcore = ArgMax(vh);
                if (lcore < 140 || lcore > 165) continue;
                cand.Add((a, b, lcore, vcore, ll, vl));
            }
            if (cand.Count < 3) return "";
            double llm = Median(cand.Select(t => (double)t.ll));
            double vlm = Median(cand.Select(t => (double)t.vl));
            double tol = 0.006 * W;
            if (!(0.03 * W <= llm && llm <= 0.10 * W && 0.20 * W <= vlm && vlm <= 0.34 * W)) return "";
            var pairs = cand.Where(t => Math.Abs(t.ll - llm) <= tol && Math.Abs(t.vl - vlm) <= tol).ToList();
            int rows = pairs.Count;
            if (rows < 3) return "";
            int rowsGrey = pairs.Count(t => t.vcore >= CardGreyLo && t.vcore <= CardGreyHi);
            int label = (int)Median(pairs.Select(t => (double)t.lcore));
            // 卡片左边外面(宽度 0.4%~1.6%)、对齐那几行高度上, 页面底色出现最多的灰度
            int xa = (int)(0.004 * W), xb = Math.Max(xa + 1, (int)(0.016 * W));
            var eh = new int[256];
            foreach (var t in pairs)
                for (int y = t.y; y <= t.yb; y++)
                    for (int x = xa; x < xb; x++) eh[g[y * W + x]]++;
            int edge = ArgMax(eh);
            if (edge >= CardEdgeLo && edge <= CardEdgeHi && label >= CardLabelLo && label <= CardLabelHi) return "A";
            if (edge > CardEdgeHi && rowsGrey * 5 >= rows * 3) return "A0";
            return "O";
        }

        // ================================ 小工具 ================================

        /// <summary>图里一块矩形的 BGR 像素, 按行连续(每行 r.Width * 3 个字节)。</summary>
        static byte[] Pixels(Mat bgr, Rect r)
        {
            using var view = new Mat(bgr, r);
            using var c = view.Clone();
            var a = new byte[r.Width * r.Height * 3];
            Marshal.Copy(c.Data, a, 0, a.Length);
            return a;
        }

        /// <summary>字节数组做成单通道 8 位图(调用方释放)。</summary>
        static Mat MaskMat(byte[] data, int rows, int cols)
        {
            var m = new Mat(rows, cols, MatType.CV_8UC1);
            Marshal.Copy(data, 0, m.Data, data.Length);
            return m;
        }

        /// <summary>出现最多的值; 并列取小的(和 np.bincount(...).argmax() 一样)。</summary>
        static int ArgMax(int[] hist)
        {
            int best = 0;
            for (int v = 1; v < hist.Length; v++) if (hist[v] > hist[best]) best = v;
            return best;
        }

        /// <summary>中位数的两倍(偶数个是中间两个之和, 奇数个是中间那个的两倍)。</summary>
        static int MedianTwice(int[] hist, int count)
        {
            int k1 = (count - 1) / 2, k2 = count / 2;
            int a = -1, b = -1, cum = 0;
            for (int v = 0; v < 256; v++)
            {
                cum += hist[v];
                if (a < 0 && cum > k1) a = v;
                if (b < 0 && cum > k2) { b = v; break; }
            }
            return a + b;
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

        /// <summary>和 numpy 的 median 一样: 偶数个取中间两个的平均。</summary>
        static double Median(IEnumerable<double> values)
        {
            var v = values.OrderBy(t => t).ToArray();
            int n = v.Length;
            return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
        }
    }
}
