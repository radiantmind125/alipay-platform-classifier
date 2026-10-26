#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum AmountFontVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // 金额数字不是支付宝那套数字字体
    }

    public sealed class AmountFontResult
    {
        public AmountFontVerdict Verdict;
        public double DigitHeight;          // 数字高(像素)
        public int DigitCount;
        public int ZeroCount;               // 认出来的 "0" 个数
        public double ZeroWidthRatio;       // ★ 判据一: 最胖的那个 0 的宽 / 数字高; 没有 0 时为 0
        public double ZeroHoleWidthRatio;   // 那个 0 的孔宽 / 0 的字宽, 只输出(真图 <= 0.50, 1026 假图 0.55)
        public bool HasDot;
        public double DotHeightRatio;       // ★ 判据二: 小数点高 / 数字高; 没有小数点时为 0
        public double DotWidthRatio;        // 小数点宽 / 数字高, 只输出
        public bool Measured;               // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// **蓝底页**金额数字的字体检查: 数字是不是支付宝 App 自带的那套数字字体。
    ///
    /// 原理
    /// ----
    /// 支付宝的金额数字是 App 自己带的字体, **苹果安卓一模一样**: 102 像素高的 0 都是 61 像素宽,
    /// 小数点都是 17 宽 19 高。P 图的人用别的字体把金额重打一遍, 字形就对不上 ——
    /// 2026-10-26 那张假图的 0 胖了一圈, 小数点扁了。
    /// 两条都是同一张图里自己跟自己比(除以数字高), 不看绝对像素, 也**不用知道是安卓还是苹果**,
    /// 图被缩放、加边框、裁边都不影响。
    ///
    /// 判据
    /// ----
    /// 1. "0" 的宽 / 数字高 > <see cref="ZeroWidthMax"/> 就报。
    ///    "0" = 正好一个孔、孔在正中、孔高 >= 字高 0.6 的数字块。支付宝的 4 也有一个封闭的孔,
    ///    但孔只有字高的一半, 不会被当成 0(4 本来就比 0 宽, 当成 0 会误报)。
    /// 2. 小数点的高 / 数字高 &lt; <see cref="DotHeightMin"/> 就报(数字高 >= <see cref="MinDigitHeightForDot"/> 才判, 这条是副的)。
    ///    小数点 = 最后两位数字前面紧挨着的那个小块, 底边和数字底齐。
    ///
    /// 标定(本机 2,273 张蓝图: 转账成功 / 到账成功 / 支付成功 / 代付等, 安卓为主, 苹果 47 张)
    /// --------------------------------------------------------------------------------
    ///                         真图                         2026-10-26 假图
    ///     0 宽 / 数字高        0.580 ~ 0.613(1,833 页)     0.690
    ///     小数点高 / 数字高    0.174 ~ 0.217(2,205 页)     0.147
    ///     0 孔宽 / 0 字宽      0.366 ~ 0.500               0.550     (只输出)
    ///   苹果 47 张和安卓 2,163 张的 0 宽 / 小数点高都在同一个范围里 —— 确实是同一套字体。
    ///   结论: 报出 1 张(就是这张假图), Ok 2,258, 不判 8(6 张是 TG 那种缩小图数字高不到 60, 2 张没有 0 也没有小数点)。真图 0 报。
    ///   9/16 和 10/25 那几张 ¥ 假图这里都正常 —— 它们的数字字体是对的, 只是 ¥ 不对, 归 YenCheck。
    ///
    /// 压缩缩放实测(三张图各 53 个变体: JPEG 95~40, 缩 0.4~1.33 倍 x 三种插值 x 有无 JPEG75, TG 1280, 加边框, 裁边;
    /// 另外按模糊 / 锐化 / 亮度 / 对比度 / gamma / 旋转 2 度以内 / JPEG 10~30 / 加边框 / 留黑边又做了 120 多个):
    ///     0 宽(按数字高分档 60~70 / 70~80 / 80~90 / 90~100 / 100~120 / 120 以上):
    ///         真图最大  0.607 / 0.605 / 0.613 / 0.606 / 0.607 / 0.606
    ///         假图最小  0.688 / 0.686 / 0.690 /   -   / 0.689 / 0.690    (不算旋转过的)
    ///     假图: 数字高 >= 60 的 157 个变体报出 156 个; 漏的那个是**整张转了 3 度**的 ——
    ///           整行斜了, 行高从 116 变 136, 0 宽比变成 0.596, 小数点也对不上底边。转 2 度以内的都报。
    ///     缩到数字高 60 以下的不判。
    ///     真图(含真图变体 144 个): 一个都不报。
    /// 本机所有图(连上面这些变体和切图, 7,510 个能量的)过了一遍: 报出的全是这张假图和它的变体, 真图 0 报。
    ///
    /// 两道护栏(都是测出来才加的, 不加会误报)
    /// --------------------------------------
    /// - **¥ 护栏**: 最左块必须像 ¥(高 / 数字高 0.60~0.80, 正好两道横杠), 和 YenCheck 一样。
    ///   不加的话, 图只截了上面一截时定位器会锁到状态栏的时间(19:40), 系统字体的 0 本来就胖(0.65~0.70);
    ///   彩票页的号码球(白圈里一个孔)也会被当成 0, 量出 1.20。
    /// - **上下被切的金额行不判**: 图正好裁在金额行上(或定位框没框全), 切掉一截的 0 更胖, 切剩一条的小数点更扁。
    ///   实测安卓真图只留上面 952~976 行、或者裁掉上面 254~256 行, 不加这道会报 7 张。
    ///
    /// 覆盖边界
    /// --------
    /// - 金额里没有 0、又量不到小数点就返回 CannotDetermine。本机 17% 的页没有 0, 但几乎都有小数点。
    /// - 数字高不到 <see cref="MinDigitHeight"/>(图被缩得很小)不判。原图上数字高约 100。
    /// - 整张图转了 3 度以上可能抓不到(行高被斜线撑大, 0 显得窄)。正常截图不会转。
    /// - 只认"0 变胖 / 小数点变扁"这两种字形差别。换一套 0 和小数点都跟支付宝很像的字体就绕过去了。
    /// - 只管蓝底页, 白底账单详情页的金额归 MinusCheck / DotCheck / FontCheck。
    ///
    /// 入参只读, 内部全部用 ROI 视图, 不复制整图。无静态可变状态, 可多线程调用。
    /// </summary>
    public static class AmountFontCheck
    {
        /// <summary>
        /// "0" 宽 / 数字高 的上限。数字高 >= 60 时真图最大 0.613(连变体), 假图 0.690(没转过的变体最小 0.686)。
        /// 取 0.65: 比真图最大高 6%, 比假图低 6%。支付宝最宽的数字是 4(0.671), 所以必须先认准是 0 再比。
        /// </summary>
        public const double ZeroWidthMax = 0.65;

        /// <summary>
        /// 小数点高 / 数字高 的下限。数字高 >= 80 时真图最小 0.175, 假图原图 0.147。
        /// 取 0.16。★ 这条比 0 宽那条薄: 假图的变体里糊过的能到 0.164(那几张靠 0 宽报), 所以它是副判据。
        /// </summary>
        public const double DotHeightMin = 0.16;

        /// <summary>
        /// 数字高低于此值整张不判(两条判据都不判)。原图上数字高约 100 像素, 这道闸只挡被缩得很小的图。
        /// ★ 实测必须: 真图缩到数字高 40 出头、再横向拉伸 3% 左右(CNN 训练那种 448x384 缩图),
        ///   二值化把 0 的上下圆弧削掉 1~2 像素, 0 宽/数字高 从 0.58 涨到 0.66~0.72, 小数点从 19 高削成 4 高。
        ///   数字高 >= 60 时没见过这种情况(变体里真图 0 宽最大 0.62, 小数点最小 0.177)。
        /// </summary>
        public const double MinDigitHeight = 60;

        /// <summary>
        /// 小数点那条要的最小数字高。小数点本身只有数字高的 0.19, 数字高 64 时才 12 像素,
        /// 糊掉一个像素就从 0.172 变成 0.156 —— 实测数字高 60~70 时真图最小 0.172、假图变体最大 0.156, 只隔一个像素。
        /// 数字高 >= 80 时真图最小 0.175(离线 0.16 有 1.2 像素以上)。0 宽那条每一档都稳(真图最大 0.613, 假图最小 0.686)。
        /// </summary>
        public const double MinDigitHeightForDot = 80;

        /// <summary>块高 >= 数字高的这个比例才算一个完整数字(认 0、找最后两位用)。</summary>
        const double DigitTall = 0.85;

        /// <summary>孔至少这么多像素, 挡 JPEG 噪点。</summary>
        const int HoleMinArea = 3;

        /// <summary>¥ 护栏: 最左块高 / 数字高 的范围(和 YenCheck 一样)。真图苹果 0.72~0.74, 安卓 0.66~0.74。</summary>
        public const double YenHeightLow = 0.60, YenHeightHigh = 0.80;

        /// <summary>¥ 护栏: 某一行的前景占 ¥ 宽度这个比例以上就算横杠所在行(和 YenCheck 一样)。</summary>
        const double BarFill = 0.70;

        public const int MaxComponents = 20_000;   // 上限保护

        struct Comp
        {
            public int X, Y, W, H, Area, Label;
        }

        /// <param name="image">
        /// 8 位或 16 位 **BGR(A)** 图(Cv2.ImRead / ImDecode 就是); 16 位会先降成 8 位。
        /// 单通道进来判不了(蓝底页这道闸要颜色)。位深/通道不支持时返回 CannotDetermine, 不抛异常。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        public static AmountFontResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new AmountFontResult { Verdict = AmountFontVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }

            int cn = image.Channels();
            if (cn != 3 && cn != 4)
            { res.Reason = $"{cn} 通道没有颜色, 判不了是不是蓝底页"; return res; }
            bool is8 = image.Depth() == MatType.CV_8U;
            if (!is8 && image.Depth() != MatType.CV_16U)
            { res.Reason = "位深不是 8 位或 16 位, 不判"; return res; }
            if (image.Width < 64 || image.Height < 128) { res.Reason = "图太小"; return res; }

            Mat? owned8 = null, ownedBgr = null, ownedGray = null;
            try
            {
                // 16 位按 1/256 降到 8 位。必须排在蓝底闸之前: 下面用的是 8 位绝对阈值
                Mat src = image;
                if (!is8)
                {
                    owned8 = new Mat();
                    image.ConvertTo(owned8, MatType.CV_8U, 1.0 / 256.0);
                    src = owned8;
                }
                if (cn == 4)
                {
                    ownedBgr = new Mat();
                    Cv2.CvtColor(src, ownedBgr, ColorConversionCodes.BGRA2BGR);
                    src = ownedBgr;
                }

                if (!IsBluePage(src))
                { res.Reason = "不是蓝底页, 本类只管蓝底页"; return res; }

                var box = LocateAmountBlue(src);
                if (box == null) { res.Reason = "定位不到金额行"; return res; }

                // 行内重新取块, 不设高度下限(小数点要留着); 左侧多外扩, 定位框的 x0 是按字算的
                int pad = Math.Max(2, (int)(box.Value.Height * 0.15));
                int padx = Math.Max(pad, (int)(box.Value.Width * 0.30));
                int cx0 = Math.Max(0, box.Value.Left - padx), cy0 = Math.Max(0, box.Value.Top - pad);
                int cx1 = Math.Min(src.Width, box.Value.Right + pad);
                int cy1 = Math.Min(src.Height, box.Value.Bottom + pad);
                if (cx1 - cx0 < 8 || cy1 - cy0 < 8)
                { res.Reason = "金额行裁出来太小"; return res; }

                using var bgrCrop = new Mat(src, Rect.FromLTRB(cx0, cy0, cx1, cy1));
                ownedGray = new Mat();
                Cv2.CvtColor(bgrCrop, ownedGray, ColorConversionCodes.BGR2GRAY);
                return Measure(ownedGray, res);
            }
            finally { ownedGray?.Dispose(); ownedBgr?.Dispose(); owned8?.Dispose(); }
        }

        // 上三分之一的通道均值判断蓝底(和 YenCheck 同一口径)
        static bool IsBluePage(Mat bgr)
        {
            int h3 = Math.Max(1, bgr.Height / 3);
            using var top = new Mat(bgr, new Rect(0, 0, bgr.Width, h3));
            var m = Cv2.Mean(top);            // BGR 顺序: Val0=B, Val1=G, Val2=R
            return m.Val0 > m.Val2 + 25 && m.Val0 > m.Val1 + 15;
        }

        /// <summary>蓝底页的金额行 —— 白字蓝底(和 YenCheck 同一口径)。</summary>
        static Rect? LocateAmountBlue(Mat bgr)
        {
            int W = bgr.Width, H = bgr.Height;
            int y0b = (int)(H * 0.05), y1b = (int)(H * 0.45);
            if (y1b - y0b < 8) return null;

            using var band = new Mat(bgr, new Rect(0, y0b, W, y1b - y0b));
            var ch = Cv2.Split(band);
            try
            {
                using var t = new Mat();
                using var mn = new Mat();
                using var mx = new Mat();
                Cv2.Min(ch[0], ch[1], t); Cv2.Min(t, ch[2], mn);
                Cv2.Max(ch[0], ch[1], t); Cv2.Max(t, ch[2], mx);
                using var diff = new Mat();
                Cv2.Subtract(mx, mn, diff);
                using var bright = new Mat();
                Cv2.Threshold(mn, bright, 175, 255, ThresholdTypes.Binary);      // mn > 175
                using var flat = new Mat();
                Cv2.Threshold(diff, flat, 44, 255, ThresholdTypes.BinaryInv);    // mx-mn < 45
                using var white = new Mat();
                Cv2.BitwiseAnd(bright, flat, white);

                var all = Label(white, minArea: 30);
                if (all.Count > MaxComponents) return null;
                var comps = new List<Comp>();
                foreach (var c in all)
                {
                    // 字高下限排掉"转账成功/收款方"这些小字; 上限和宽度上限排掉红包卡和大白块
                    if (c.H < 0.025 * H || c.H > 0.12 * H || c.W > 0.5 * W) continue;
                    comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
                }
                if (comps.Count == 0) return null;

                int maxh = comps.Max(c => c.H);
                var amt = comps.Where(c => c.H >= 0.6 * maxh).ToList();   // 只留最高的那一档
                if (amt.Count < 2) return null;
                double cy = Median(amt.Select(c => c.Y + c.H / 2.0));
                amt = amt.Where(c => Math.Abs((c.Y + c.H / 2.0) - cy) < 0.6 * maxh).ToList();
                if (amt.Count < 2) return null;

                int x0 = amt.Min(c => c.X), x1 = amt.Max(c => c.X + c.W);
                int y0 = amt.Min(c => c.Y), y1 = amt.Max(c => c.Y + c.H);
                if (x1 - x0 < 0.08 * W) return null;                      // 太窄不像金额
                return Rect.FromLTRB(x0, y0, x1, y1);
            }
            finally { foreach (var c in ch) c.Dispose(); }
        }

        /// <param name="sub">已经裁好、已经转成灰度的金额行小块。</param>
        /// <param name="res">量到的数和结论都写进这里。</param>
        static AmountFontResult Measure(Mat sub, AmountFontResult res)
        {
            int cw = sub.Width, ch = sub.Height;

            using var bin = new Mat();
            Cv2.Threshold(sub, bin, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
            // 前景过半说明极性相反, 取反(白图深字 / 蓝图白字极性正好反着)
            if (255L * Cv2.CountNonZero(bin) > 127L * cw * ch) Cv2.BitwiseNot(bin, bin);

            using var labels = new Mat();
            using var stats = new Mat();
            using var cent = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(bin, labels, stats, cent,
                                                     PixelConnectivity.Connectivity8, MatType.CV_32S);
            if (n - 1 > MaxComponents) { res.Reason = "块太多"; return res; }
            var glyphs = new List<Comp>();
            for (int i = 1; i < n; i++)
            {
                int a = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                if (a < 6) continue;              // 只挡真正的噪点, 不设高度下限
                glyphs.Add(new Comp
                {
                    X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                    Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                    W = stats.At<int>(i, (int)ConnectedComponentsTypes.Width),
                    H = stats.At<int>(i, (int)ConnectedComponentsTypes.Height),
                    Area = a,
                    Label = i,
                });
            }
            if (glyphs.Count < 3) { res.Reason = "块数不够"; return res; }
            // 排序取全序: 连通域返回顺序依实现而定, 并列时不能靠它决定
            glyphs = glyphs.OrderBy(g => g.X).ThenBy(g => g.Y).ThenBy(g => g.Label).ToList();

            // 最左边那块当 ¥, 不进统计; 贴着裁块左右边缘的块是被切断的, 也不要
            var rest = glyphs.Skip(1).Where(g => g.X > 0 && g.X + g.W < cw).ToList();
            if (rest.Count == 0) { res.Reason = "数字不够三个"; return res; }
            int tall = glyphs.Skip(1).Max(g => g.H);
            var digits = rest.Where(g => g.H >= 0.80 * tall).ToList();
            if (digits.Count < 3) { res.Reason = "数字不够三个"; return res; }    // 金额至少 x.xx 三位

            // ★ 金额行上下被切到了(图被裁得正好卡在金额行上, 或者定位框没框全)就不判:
            //   切掉一截的 0 看着更胖, 切剩一条的小数点看着更扁, 两条判据都会误报。
            //   实测: 安卓真图只留上面 952~976 行、或者裁掉上面 254~256 行时, 不加这道会报 7 张。
            if (digits.Any(g => g.Y == 0 || g.Y + g.H >= ch) || glyphs[0].Y == 0 || glyphs[0].Y + glyphs[0].H >= ch)
            { res.Reason = "金额行上下被切到了, 不判"; return res; }

            int dTop = digits.Min(g => g.Y), dBot = digits.Max(g => g.Y + g.H);
            double dH = dBot - dTop;
            if (dH <= 0) { res.Reason = "数字高算不出来"; return res; }
            double dhMean = digits.Average(g => (double)g.H);
            if (Std(digits.Select(g => (double)g.H)) / dhMean > 0.08)
            { res.Reason = "数字高度不齐, 不像一行数字"; return res; }

            labels.GetRectangularArray(out int[,] lab);

            // ---- ¥ 护栏(和 YenCheck 同一口径): 最左块高比 0.60~0.80, 而且正好两道横杠 ----
            // ★ 不能省: 图只截了上面一截时, 定位器会锁到状态栏的时间(19:40 这种), 系统字体的 0 本来就胖,
            //   实测苹果真图截上 1/6 量出 0 宽 0.652 直接误报。核一下最左块是 ¥, 才说明锁的是金额行。
            var yen = glyphs[0];
            double yenHRatio = yen.H / dH;
            if (yenHRatio < YenHeightLow || yenHRatio > YenHeightHigh)
            { res.Reason = $"最左块高比 {yenHRatio:F3} 不像 ¥, 可能没锁到金额行"; return res; }
            int bars = CountBars(lab, yen);
            if (bars != 2)
            { res.Reason = $"最左块有 {bars} 道横杠, 不像 ¥, 可能没锁到金额行"; return res; }

            // ---- 判据一的量: 认出 0, 取最胖的那个 ----
            int zeros = 0;
            double zeroW = 0, zeroHoleW = 0;
            foreach (var d in digits)
            {
                if (d.H < DigitTall * dH) continue;
                var hs = Holes(lab, d);
                if (!IsZero(d, hs)) continue;
                zeros++;
                double wr = d.W / dH;
                if (wr > zeroW) zeroW = wr;
                double hw = (double)hs[0].W / d.W;
                if (hw > zeroHoleW) zeroHoleW = hw;
            }

            // ---- 判据二的量: 最后两位数字前面紧挨着的那个小块, 底边和数字底齐 ----
            Comp? dot = null;
            var big = rest.Where(g => g.H >= DigitTall * dH).ToList();
            if (big.Count >= 2)
            {
                int xLast2 = big[big.Count - 2].X;
                var cands = rest.Where(g => g.X + g.W <= xLast2 && g.H < 0.35 * dH && g.W < 0.35 * dH
                                            && Math.Abs(dBot - (g.Y + g.H)) <= 0.08 * dH).ToList();
                if (cands.Count > 0)
                {
                    var k = cands[cands.Count - 1];
                    // 它和最后两位之间不能再夹着别的数字
                    if (!big.Any(b => k.X < b.X && b.X < xLast2)) dot = k;
                }
            }

            res.Measured = true;
            res.DigitHeight = dH;
            res.DigitCount = digits.Count;
            res.ZeroCount = zeros;
            res.ZeroWidthRatio = zeroW;
            res.ZeroHoleWidthRatio = zeroHoleW;
            res.HasDot = dot != null;
            res.DotHeightRatio = dot != null ? dot.Value.H / dH : 0;
            res.DotWidthRatio = dot != null ? dot.Value.W / dH : 0;

            if (dH < MinDigitHeight)
            {
                res.Reason = $"数字高 {dH:F0} < {MinDigitHeight}, 判不准所以不判";
                return res;
            }

            var bad = new List<string>();
            if (zeros > 0 && zeroW > ZeroWidthMax)
                bad.Add($"0 宽/数字高 {zeroW:F4} (真图 <= {ZeroWidthMax})");
            bool dotUsable = res.HasDot && dH >= MinDigitHeightForDot;
            if (dotUsable && res.DotHeightRatio < DotHeightMin)
                bad.Add($"小数点高/数字高 {res.DotHeightRatio:F4} (真图 >= {DotHeightMin})");

            if (bad.Count > 0)
            {
                res.Verdict = AmountFontVerdict.Suspicious;
                res.Reason = "金额数字不像支付宝的字体: " + string.Join("; ", bad);
            }
            else if (zeros == 0 && !dotUsable)
            {
                res.Reason = "金额里没有 0, 也没有能量的小数点, 不判";
            }
            else
            {
                res.Verdict = AmountFontVerdict.Ok;
                res.Reason = (zeros > 0 ? $"0 宽/数字高 {zeroW:F4}" : "没有 0")
                           + (dotUsable ? $", 小数点高/数字高 {res.DotHeightRatio:F4}" : "");
            }
            return res;
        }

        readonly struct Hole
        {
            public readonly double Cx, Cy;     // 中心, 相对字块外接框
            public readonly int W, H, Area;
            public Hole(double cx, double cy, int w, int h, int area) { Cx = cx; Cy = cy; W = w; H = h; Area = area; }
        }

        /// <summary>
        /// 一个字块里的孔: 外接框外补一圈背景, 背景按 4 连通找连通块, 不碰外圈的就是孔。
        /// ★ 口径和 Python 侧 amount_font_scan.holes 一样(那边用 OpenCV 4 连通标记, 这里自己走一遍, 点集相同)。
        /// </summary>
        static List<Hole> Holes(int[,] lab, Comp c)
        {
            int h = c.H + 2, w = c.W + 2;
            var bgPix = new bool[h, w];            // true = 背景(不是这个字块的像素)
            for (int r = 0; r < h; r++)
                for (int x = 0; x < w; x++)
                {
                    int gy = c.Y + r - 1, gx = c.X + x - 1;
                    bool inBox = r >= 1 && r <= c.H && x >= 1 && x <= c.W;
                    bgPix[r, x] = !(inBox && lab[gy, gx] == c.Label);
                }
            var seen = new bool[h, w];
            var outp = new List<Hole>();
            var stack = new Stack<(int r, int x)>();
            for (int r0 = 0; r0 < h; r0++)
                for (int x0 = 0; x0 < w; x0++)
                {
                    if (!bgPix[r0, x0] || seen[r0, x0]) continue;
                    int minX = x0, maxX = x0, minY = r0, maxY = r0, area = 0;
                    long sx = 0, sy = 0;
                    seen[r0, x0] = true;
                    stack.Push((r0, x0));
                    while (stack.Count > 0)
                    {
                        var (r, x) = stack.Pop();
                        area++; sx += x; sy += r;
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (r < minY) minY = r;
                        if (r > maxY) maxY = r;
                        if (r > 0 && bgPix[r - 1, x] && !seen[r - 1, x]) { seen[r - 1, x] = true; stack.Push((r - 1, x)); }
                        if (r + 1 < h && bgPix[r + 1, x] && !seen[r + 1, x]) { seen[r + 1, x] = true; stack.Push((r + 1, x)); }
                        if (x > 0 && bgPix[r, x - 1] && !seen[r, x - 1]) { seen[r, x - 1] = true; stack.Push((r, x - 1)); }
                        if (x + 1 < w && bgPix[r, x + 1] && !seen[r, x + 1]) { seen[r, x + 1] = true; stack.Push((r, x + 1)); }
                    }
                    // 碰到外圈的是字外面那一大块背景, 不是孔
                    if (minX == 0 || minY == 0 || maxX == w - 1 || maxY == h - 1) continue;
                    if (area < HoleMinArea) continue;
                    outp.Add(new Hole((double)sx / area - 1.0, (double)sy / area - 1.0,
                                      maxX - minX + 1, maxY - minY + 1, area));
                }
            return outp;
        }

        /// <summary>¥ 的横杠数: 逐行数这个字块的像素, 够 BarFill 的行算横杠行, 连续的算一道。</summary>
        static int CountBars(int[,] lab, Comp c)
        {
            int spans = 0;
            bool prev = false;
            for (int r = 0; r < c.H; r++)
            {
                int on = 0;
                for (int x = 0; x < c.W; x++)
                    if (lab[c.Y + r, c.X + x] == c.Label) on++;
                bool bar = on >= BarFill * c.W;
                if (bar && !prev) spans++;
                prev = bar;
            }
            return spans;
        }

        // 总体标准差
        static double Std(IEnumerable<double> xs)
        {
            var v = xs.ToList(); if (v.Count == 0) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Sum(k => (k - m) * (k - m)) / v.Count);
        }

        /// <summary>"0": 正好一个孔, 孔心在字块正中(横竖都在 40%~60%), 孔高 >= 字高 0.6。</summary>
        static bool IsZero(Comp c, List<Hole> hs)
        {
            if (hs.Count != 1) return false;
            var o = hs[0];
            return o.Cx / c.W >= 0.40 && o.Cx / c.W <= 0.60
                && o.Cy / c.H >= 0.40 && o.Cy / c.H <= 0.60
                && o.H >= 0.60 * c.H;
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
            for (int i = 1; i < n; i++)          // 0 是背景
            {
                int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                if (area < minArea) continue;    // 这句不能省: OpenCV 自己不做面积过滤
                outp.Add(new Comp
                {
                    X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                    Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                    W = stats.At<int>(i, (int)ConnectedComponentsTypes.Width),
                    H = stats.At<int>(i, (int)ConnectedComponentsTypes.Height),
                    Area = area,
                    Label = i,
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
    }
}
