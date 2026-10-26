#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum CircleVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,                // 标题前的圈是正圆, 圈里的对号位置大小正常
        Suspicious = 2,        // 圈不圆 / 对号不对 / 成功页标题前没有圈对号
    }

    public sealed class CircleResult
    {
        public CircleVerdict Verdict;
        // ---- 找到圈的时候 ----
        public bool RingFound;                       // 找到了标题左边的空心圈
        public double Roundness = double.NaN;        // ★ 判据一: 36 个方向的直径 最大 / 最小(正圆 1.000)
        public double RatioHV = double.NaN;          // 横向直径 / 竖向直径, 只输出(假图 1.033, 横向拉宽了)
        public double RingDiameter = double.NaN;     // 圈外径(像素, 72 条射线的平均)
        public double RingToAmount = double.NaN;     // 圈外径 / 金额行高, 只输出(真图 0.45 ~ 0.56)
        public int RayCount;                         // 72 条射线里量到圈边的条数
        public bool CheckFound;                      // 圈里有对号
        public double CheckWidth = double.NaN;       // ★ 判据三: 对号宽 / 圈外径
        public double CheckHeight = double.NaN;      // 对号高 / 圈外径
        public double CheckOffsetX = double.NaN;     // (对号中心 - 圈心) 横向 / 圈外径
        public double CheckOffsetY = double.NaN;     // (对号中心 - 圈心) 竖向 / 圈外径
        // ---- 没找到圈的时候 ----
        public bool TitleRowFound;                   // 找到了金额上面那一行白色标题字
        public double TitleBrightness = double.NaN;  // 标题字像素三通道最小值的 90 分位(护栏: 页面被压暗时发灰)
        public double LeadGap = double.NaN;          // 标题最左一簇到第二簇的空隙 / 金额行高(护栏)
        public double LeadTall = double.NaN;         // 标题最左一簇的高 / 其余各簇高的中位数(护栏)
        public double SlotFraction = double.NaN;     // ★ 判据二: 标题左边圈该在的那块里"不是底色"的像素占比
        public double ForeignRatio = double.NaN;     // 标题行上下半行里别的颜色(不在 底色 -> 白 线段上)的占比(护栏)
        public string Reason = "";
    }

    /// <summary>
    /// **蓝底成功页**(转账成功 / 到账成功 / 支付成功)标题前面那个"圈 + 对号"图标的检查。
    ///
    /// 背景
    /// ----
    /// 2026-10-26 经理在一张真图上圈了三样: 标题前的圈对号、¥、两个 0, 要"圈符号的检测"和"对号的检测"。
    /// ¥ 归 YenCheck, 0 归 AmountFontCheck, 这里管圈对号。量下来假图有两种做法:
    ///   - 9-16 假图(到账成功 1179x2556)、10-25 假图(支付成功 1320x2868)、TG 小图 013.jpg 是**同一个贴上去的图标**:
    ///     外框 58x56, 横向拉宽了, 是椭圆。支付宝自己的圈是正圆(同尺寸的真图 54x54, 线粗 3)。
    ///   - 10-26 假图(安卓到账成功 1220x2712)标题前**根本没有圈对号**, 标题是粗体, 位置也偏高。
    ///
    /// 判据
    /// ----
    /// 1. **圈不圆**(<see cref="RoundMax"/>): 从圈心沿 72 个方向取剖面, 圈的内沿、外沿各用自己那一侧的底色算半高位置
    ///    (这样圈外有光晕、底色不匀也不受影响), 中线 = 内外沿的平均; 对边相加得 36 个方向的直径,
    ///    每个方向取前后 20 度扇区平均, 最大 / 最小 > 1.023 就报。不用知道机型, 图等比缩放了也不影响。
    /// 2. **标题前没有圈对号**: 找不到圈的时候, 找金额上面那一行白色标题字, 标题第一个字左边圈本来该在的那块
    ///    (左沿往左 0.10 ~ 0.95 个金额行高, 上下各 0.42 个金额行高)全是底色, 就报。
    /// 3. **对号不对**: 圈里要有对号, 对号相对圈外径的宽、高、偏移在真图范围以外就报(范围放得很宽, 是兜底的)。
    ///
    /// 标定(本机 10,986 张: 2,339 张蓝底真图 + 白图等别的页 + 已知假图 + 可疑集)
    /// ------------------------------------------------------------------------
    ///                                  真图                                      假图
    ///     圆度(直径 最大/最小)         原图 1.0002 ~ 1.0064(2,161 张)            1.0322 ~ 1.0326(三张贴图标的)
    ///                                  压缩缩小的变体最大 1.0187(见下)            连变体最小 1.0274, 一般 1.03 ~ 1.04
    ///     对号宽 / 圈外径              0.393 ~ 0.521                             0.444 ~ 0.503(假图对号本身没问题)
    ///     对号高 / 圈外径              0.242 ~ 0.335                             0.274 ~ 0.295
    ///   结论: 真图 0 报。蓝底真图 2,339 张里 Ok 2,161, 不判 178(105 张定位不到金额行, 其余多是通知横幅 / 拼音软件
    ///   把标题和圈遮了或染了色, 圈认不出来, 按护栏不判); 白图等别的页一律不判。四张已知假图全报(三张圈不圆, 1026 那张没有圈)。
    ///   以前查出来的可疑集(AI 生成图、红队造的缩放攻击图)77 张里报 11 张, 都是圈不圆 —— 椭圆是画出来 / 拉伸出来的。
    ///   连同下面的变体、AmountFontCheck 评审时造的各种刁难图(加框线 / 下划线 / 横幅、截断、16 位、镜像等)一共 31,710 个输入:
    ///   真图和真图变体 0 报。
    ///
    /// 压缩缩放实测
    /// ------------
    ///   蓝底真图(连已有的各种变体, 5,000 多个来源)每张再做 4 种: TG 590 宽 q80 / 原尺寸 q40 / 0.5 倍 q75 / 0.75 倍 q60;
    ///   另抽 900 张做更狠的 4 种: TG 590 宽 q50 / q40, 0.6 倍 q50, 0.55 倍 q40。量得到圈的一共 30,875 个, 圆度按圈外径分档:
    ///       外径 24~28 最大 1.0187 / 28~32 1.0163 / 32~40 1.0132 / 40~60 1.0127 / 60 以上 1.0080, 一个都不报。
    ///   假图(三张贴图标的)和它们的变体: 圆度最小 1.0274, 圈外径 >= 24 的全报。阈值 1.023 取在两边正中。
    ///   1026 假图: 原尺寸、JPEG 40~95、0.55 倍以上缩放、加边框都报; 缩到金额行高 60 以下、裁掉了标题、标题旁边加了框线的不判。
    ///
    /// 护栏(都是测出误报才加的)
    /// ----------------------
    /// - 高 / 宽 &lt; 1.6 不判: 拍屏的 4:3 照片有透视, 正圆会拍成椭圆(本机两张拍屏图量出 1.026 / 1.030)。
    /// - 圈外径 &lt; 24 像素不判; &lt; 30 只看圆不圆, 对号只看有没有(小图对号被糊掉一圈)。
    /// - "没有圈"那条, 这些情况一律不判:
    ///   金额行高 &lt; 60(小图里对号和圈粘成一块, 圈认不出来, 实测 0.4 倍的拼音页会误报);
    ///   标题字发灰(<see cref="TitleBright"/>, 页面被半透明遮罩压暗, 圈也发灰, 白色门槛下圈碎成几个点);
    ///   标题最左一簇单独隔开、比别的字高、或者近似正方形又带孔(多半是没认出来的圈或别的图标);
    ///   标题正上方还有白块(比如上面一个"支"字标志、下面小字"支付成功"的那种页, 本来就没有圈对号);
    ///   标题这一行别处有个圈(整张图是镜像的, 圈跑到了标题右边);
    ///   标题行附近有别的颜色(通知横幅、贴纸、描边)。
    /// - 圈里没有东西: 不是成功页的那种图标, 不判。
    ///
    /// 覆盖边界
    /// --------
    /// - 只管蓝底页; 别的页、横屏、拍屏、裁掉标题的图都返回 CannotDetermine。
    /// - 圈做得正圆、对号放对位置的假图这里抓不到(要靠 YenCheck / AmountFontCheck 等别的检查)。
    /// - 没有圈的那条只认"标题左边是空的"; 标题左边放了个别的东西就判不了。
    /// - 整张图被**不等比**缩放过(拉成别的长宽比, 比如训练用的 288x640 缩图再放大), 正圆也会被拉扁, 会报。
    ///   手机截图和聊天软件的压缩都是等比的, 不会这样; 本机 3,583 张 288x640 训练缩图原样都不判(圈太小)。
    ///
    /// 和 training/circle_scan.py 逐项对数: 同样的像素进来, 结论、原因和量到的数逐位一样
    /// (31,710 个输入里不一样的 28 个都是解码层面的: 截断的 JPEG 两边解出来的像素不同, 这时 C# 这边都是不判)。
    /// 入参只读, 内部只拷标题那一条; 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class CircleCheck
    {
        /// <summary>高 / 宽 低于这个不判(拍屏 4:3 照片的透视会把正圆拍扁; 裁过的图、横屏也不判)。</summary>
        public const double MinAspect = 1.6;

        /// <summary>标题带里的"白": 三个通道都大于这个值。蓝底的 R 才二十多, 白字和圈是 255。</summary>
        public const int WhiteMin = 140;

        /// <summary>
        /// 圆度(直径 最大 / 最小)超过就报。真图原图最大 1.0064, 压缩缩小的变体最大 1.0187; 假图连变体最小 1.0274。
        /// </summary>
        public const double RoundMax = 1.023;

        /// <summary>圈外径(像素)小于这个整张不判。</summary>
        public const double MinRingDiameter = 24.0;

        /// <summary>圈外径小于这个只看圆不圆、对号只看有没有, 不比对号大小位置。</summary>
        public const double MinRingDiameterForCheck = 30.0;

        /// <summary>对号宽 / 圈外径 的范围(真图 0.393 ~ 0.521)。</summary>
        public const double CheckWidthLow = 0.34, CheckWidthHigh = 0.60;

        /// <summary>对号高 / 圈外径 的范围(真图 0.242 ~ 0.335)。</summary>
        public const double CheckHeightLow = 0.20, CheckHeightHigh = 0.40;

        /// <summary>(对号中心 - 圈心) 横向 / 圈外径 的范围(真图 0.019 ~ 0.043)。</summary>
        public const double CheckOffsetXLow = -0.05, CheckOffsetXHigh = 0.11;

        /// <summary>(对号中心 - 圈心) 竖向 / 圈外径 的范围(真图 -0.007 ~ 0.021)。</summary>
        public const double CheckOffsetYLow = -0.08, CheckOffsetYHigh = 0.10;

        /// <summary>"没有圈"那条要的最小金额行高(像素)。约等于图宽 650 以上。</summary>
        public const int MinAmountHeightForMissing = 60;

        /// <summary>标题字像素三通道最小值的 90 分位低于这个(发灰), "没有圈"那条不判。正常页 240 以上。</summary>
        public const int TitleBright = 200;

        /// <summary>标题左边空位里"不是底色"的像素占比不超过这个就算空(给 JPEG 噪点留的余量)。</summary>
        public const double SlotTolerance = 0.002;

        /// <summary>标题行上下半行里别的颜色占比超过这个, "没有圈"那条不判。</summary>
        public const double ForeignMax = 0.01;

        public const int MaxComponents = 20_000;   // 上限保护

        const int MinArea = 6;                       // 连通块至少这么多像素
        const int HoleMinArea = 3;                   // 孔至少这么多像素
        const double RingCxLow = 0.10, RingCxHigh = 0.62;   // 圈心横坐标 / 图宽
        const double RingHLow = 0.010, RingHHigh = 0.08;    // 圈高 / 图宽
        const double RingArLow = 0.8, RingArHigh = 1.25;    // 圈的宽 / 高
        const double HoleBig = 0.5;                  // 大孔: 宽、高都 >= 圈的这个比例
        const int TitleMin = 3;                      // 圈右边同一行至少这么多个字块
        const int NAng = 72;                         // 射线条数
        const double Step = 0.25;                    // 射线采样步长(像素)
        const double RLow = 0.62, RHigh = 1.30;      // 射线从 0.62 R0 量到 1.30 R0(R0 = 外框半径), 避开里面的对号
        const double MinContrast = 40;               // 剖面峰值比两侧底色至少亮这么多
        const int MinRays = 60;                      // 有效射线至少这么多条
        const int MinSectors = 18;                   // 有效扇区至少这么多个(共 36 个)
        const double SectorHalf = 20;                // 扇区半宽(度)

        const double RowHLow = 0.12, RowHHigh = 0.70;       // 标题字块高 / 金额行高
        const double RowWMax = 1.2;                          // 字块宽上限 / 金额行高
        const double RowCyLow = 0.35, RowCyHigh = 2.20;     // 字块中心在金额上沿以上多少个金额行高
        const double RowXLow = 0.15, RowXHigh = 0.85;       // 字块中心横坐标 / 图宽(排掉右上角的 完成 / 回首页)
        const double RowTrLow = 0.18, RowTrHigh = 0.75;     // 标题行高 / 金额行高
        const int RowMinTr = 16;                             // 标题行高至少这么多像素
        const double RowCxLow = 0.30, RowCxHigh = 0.75;     // 标题行中心横坐标 / 图宽
        const double MainH = 0.5;                            // 定标题左右沿只用高 >= 行高这个比例的字簇
        const double LeadGapMax = 0.15;                      // 最左一簇和第二簇隔开这么多个金额行高以上: 当它是图标
        const double LeadTallMax = 1.10;                     // 最左一簇比其余簇高 10% 以上: 当它是图标
        const double LeadSquareLow = 0.75, LeadSquareHigh = 1.33;   // 最左一簇近似正方形 ...
        const double LeadHole = 0.10;                        // ... 且孔的总面积 >= 外框 10%: 当它是没认出来的圈
        const double AboveH = 0.12;                          // 标题正上方有这么高(金额行高)的白块: 别的页型
        const double AbovePad = 0.5;                         // "正上方" 横向比标题行左右各宽这么多个金额行高
        const double RingFill = 0.35;                        // 标题行别处找圈: 白像素 <= 外框 35%(真圈 0.19 ~ 0.21) ...
        const double RingHole = 0.75;                        // ... 且正好一个宽高都 >= 外框 75% 的孔(真圈 0.89)
        const double SlotXFar = 0.95, SlotXNear = 0.10;      // 空位横向: 标题左沿往左 0.95 到 0.10 个金额行高
        const double SlotHalf = 0.42;                        // 空位竖向: 标题行中心上下各 0.42 个金额行高
        const double SlotMinW = 0.5;                         // 空位裁剪后至少这么宽(金额行高)
        const double SlotDist2 = 1600;                       // 离底色 > 40 算有东西(距离平方)
        const double ForeignDist2 = 6400;                    // 离 底色 -> 白 线段 > 80 算别的颜色(距离平方)

        struct Comp
        {
            public int X, Y, W, H, Area, Label;
        }

        readonly struct HoleBox
        {
            public readonly int X, Y, W, H, Area;    // 坐标相对字块外接框
            public HoleBox(int x, int y, int w, int h, int area) { X = x; Y = y; W = w; H = h; Area = area; }
        }

        sealed class Cluster
        {
            public int L, R, T, B;
            public readonly List<Comp> Members = new List<Comp>();
        }

        /// <param name="image">
        /// 8 位或 16 位 **BGR(A)** 图(Cv2.ImRead / ImDecode 就是); 16 位会先降成 8 位。
        /// 单通道进来判不了(蓝底页这道闸要颜色)。位深/通道不支持时返回 CannotDetermine, 不抛异常。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        public static CircleResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new CircleResult { Verdict = CircleVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }

            int cn = image.Channels();
            if (cn != 3 && cn != 4)
            { res.Reason = $"{cn} 通道没有颜色, 判不了是不是蓝底页"; return res; }
            bool is8 = image.Depth() == MatType.CV_8U;
            if (!is8 && image.Depth() != MatType.CV_16U)
            { res.Reason = "位深不是 8 位或 16 位, 不判"; return res; }
            if (image.Width < 64 || image.Height < 128) { res.Reason = "图太小"; return res; }
            if (image.Height < MinAspect * image.Width)
            { res.Reason = "长宽比不像手机截图(拍屏 / 裁过 / 横屏), 不判"; return res; }

            Mat? owned8 = null, ownedBgr = null;
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

                int W = src.Width, H = src.Height;
                int ay0 = box.Value.Top, adh = box.Value.Height;
                int top = (int)(0.04 * H), bot = (int)(ay0 - 0.15 * adh);
                if (bot - top < 20) { res.Reason = "金额上面没地方放标题"; return res; }
                return Measure(src, top, bot, ay0, adh, res);
            }
            finally { ownedBgr?.Dispose(); owned8?.Dispose(); }
        }

        static CircleResult Measure(Mat src, int top, int bot, int ay0, int adh, CircleResult res)
        {
            int W = src.Width, bh = bot - top;
            using var roi = new Mat(src, new Rect(0, top, W, bh));
            using var band = roi.Clone();              // ROI 视图不连续, 拷一份再整块取
            band.GetRectangularArray(out Vec3b[,] px);

            var ch = Cv2.Split(band);
            byte[,] mn;
            int[,] lab;
            var comps = new List<Comp>();
            try
            {
                using var t = new Mat();
                using var mnMat = new Mat();
                Cv2.Min(ch[0], ch[1], t); Cv2.Min(t, ch[2], mnMat);
                mnMat.GetRectangularArray(out mn);
                using var white = new Mat();
                Cv2.Threshold(mnMat, white, WhiteMin, 255, ThresholdTypes.Binary);   // mn > 140

                using var labels = new Mat();
                using var stats = new Mat();
                using var cent = new Mat();
                int n = Cv2.ConnectedComponentsWithStats(white, labels, stats, cent,
                                                         PixelConnectivity.Connectivity8, MatType.CV_32S);
                if (n - 1 > MaxComponents) { res.Reason = "白块太多(花图), 不判"; return res; }
                labels.GetRectangularArray(out lab);
                for (int i = 1; i < n; i++)
                {
                    int a = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                    if (a < MinArea) continue;
                    comps.Add(new Comp
                    {
                        X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                        Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                        W = stats.At<int>(i, (int)ConnectedComponentsTypes.Width),
                        H = stats.At<int>(i, (int)ConnectedComponentsTypes.Height),
                        Area = a,
                        Label = i,
                    });
                }
            }
            finally { foreach (var c in ch) c.Dispose(); }

            var ring = FindRing(W, comps, lab);
            if (ring != null) return JudgeRing(res, ring.Value.ring, ring.Value.hole, comps, mn, lab, adh);
            return JudgeMissing(res, px, mn, lab, comps, adh, ay0 - top, W, bh);
        }

        // ================= 判据一、三: 找到了圈 =================

        /// <summary>圈: 外框近似正方形、正好一个大孔、右边同一行 >= 3 个标题字块; 多个取最靠下的。</summary>
        static (Comp ring, HoleBox hole)? FindRing(int W, List<Comp> comps, int[,] lab)
        {
            (Comp ring, HoleBox hole)? best = null;
            foreach (var c in comps)
            {
                double cx = c.X + c.W / 2.0;
                if (!(RingCxLow * W <= cx && cx <= RingCxHigh * W)) continue;
                if (!(RingHLow * W <= c.H && c.H <= RingHHigh * W)) continue;
                double ar = (double)c.W / c.H;
                if (!(RingArLow <= ar && ar <= RingArHigh)) continue;
                var big = HolesBox(lab, c).Where(o => o.W >= HoleBig * c.W && o.H >= HoleBig * c.H).ToList();
                if (big.Count != 1) continue;
                double cy = c.Y + c.H / 2.0;
                int right = 0;
                foreach (var o in comps)
                    if (c.X + c.W < o.X && o.X < c.X + c.W + 8 * c.H
                        && Math.Abs((o.Y + o.H / 2.0) - cy) < 0.6 * c.H && o.H >= 0.4 * c.H)
                        right++;
                if (right < TitleMin) continue;
                // 取最靠下的; 一样低取最靠左的(不靠连通域编号顺序, 各版本 OpenCV 编号可能不同)
                if (best == null || c.Y > best.Value.ring.Y
                    || (c.Y == best.Value.ring.Y && c.X < best.Value.ring.X)) best = (c, big[0]);
            }
            return best;
        }

        static CircleResult JudgeRing(CircleResult res, Comp ring, HoleBox hole, List<Comp> comps,
                                      byte[,] mn, int[,] lab, int adh)
        {
            res.RingFound = true;
            RingRays(mn, lab, ring, out double cx, out double cy, out double[] rin, out double[] rout);
            int half = NAng / 2;
            var dm = new double[half];
            var dOut = new double[half];
            int nray = 0;
            var rmid = new double[NAng];
            for (int k = 0; k < NAng; k++)
            {
                rmid[k] = (rin[k] + rout[k]) / 2.0;
                if (!double.IsNaN(rmid[k])) nray++;
            }
            for (int j = 0; j < half; j++)
            {
                dm[j] = rmid[j] + rmid[j + half];
                dOut[j] = rout[j] + rout[j + half];
            }
            res.RayCount = nray;
            double acc = 0; int cnt = 0;
            for (int j = 0; j < half; j++)
                if (!double.IsNaN(dOut[j])) { acc += dOut[j]; cnt++; }
            double ringD = cnt > 0 ? acc / cnt : double.NaN;
            res.RingDiameter = ringD;
            res.RingToAmount = ringD / adh;

            var secs = new double?[36];
            var good = new List<double>();
            for (int s = 0; s < 36; s++)
            {
                secs[s] = SectorMean(dm, s * 5);
                if (secs[s] != null) good.Add(secs[s]!.Value);
            }
            if (secs[0] != null && secs[18] != null) res.RatioHV = secs[0]!.Value / secs[18]!.Value;
            if (good.Count >= MinSectors) res.Roundness = good.Max() / good.Min();

            // 对号: 中心落在大孔外接框里的白块
            var inside = comps.Where(o => o.Label != ring.Label
                && ring.X + hole.X <= o.X + o.W / 2.0 && o.X + o.W / 2.0 <= ring.X + hole.X + hole.W
                && ring.Y + hole.Y <= o.Y + o.H / 2.0 && o.Y + o.H / 2.0 <= ring.Y + hole.Y + hole.H).ToList();
            if (inside.Count > 0 && !double.IsNaN(ringD))
            {
                res.CheckFound = true;
                int l = inside.Min(o => o.X), r = inside.Max(o => o.X + o.W);
                int t = inside.Min(o => o.Y), b = inside.Max(o => o.Y + o.H);
                res.CheckWidth = (r - l) / ringD;
                res.CheckHeight = (b - t) / ringD;
                res.CheckOffsetX = ((l + r) / 2.0 - cx) / ringD;
                res.CheckOffsetY = ((t + b) / 2.0 - cy) / ringD;
            }

            if (double.IsNaN(ringD) || ringD < MinRingDiameter)
            { res.Reason = "圈太小, 不判"; return res; }
            if (nray < MinRays || double.IsNaN(res.Roundness))
            { res.Reason = "圈的边量不全(被别的东西挡了), 不判"; return res; }
            if (res.Roundness > RoundMax)
            {
                res.Verdict = CircleVerdict.Suspicious;
                res.Reason = $"圈不圆: 直径 最大/最小 = {F4(res.Roundness)}(真图 <= 1.019)";
                return res;
            }
            if (!res.CheckFound)
            { res.Reason = "圈里没有对号, 不是成功页的图标, 不判"; return res; }
            if (ringD < MinRingDiameterForCheck)
            {
                res.Verdict = CircleVerdict.Ok;
                res.Reason = $"圈是正圆({F4(res.Roundness)}); 图小, 对号只看有没有";
                return res;
            }
            var bad = new List<string>();
            if (!(CheckWidthLow <= res.CheckWidth && res.CheckWidth <= CheckWidthHigh))
                bad.Add($"对号宽/圈 {F3(res.CheckWidth)}");
            if (!(CheckHeightLow <= res.CheckHeight && res.CheckHeight <= CheckHeightHigh))
                bad.Add($"对号高/圈 {F3(res.CheckHeight)}");
            if (!(CheckOffsetXLow <= res.CheckOffsetX && res.CheckOffsetX <= CheckOffsetXHigh))
                bad.Add($"对号横向偏移 {S3(res.CheckOffsetX)}");
            if (!(CheckOffsetYLow <= res.CheckOffsetY && res.CheckOffsetY <= CheckOffsetYHigh))
                bad.Add($"对号竖向偏移 {S3(res.CheckOffsetY)}");
            if (bad.Count > 0)
            {
                res.Verdict = CircleVerdict.Suspicious;
                res.Reason = "对号不对: " + string.Join(", ", bad);
                return res;
            }
            res.Verdict = CircleVerdict.Ok;
            res.Reason = $"圈是正圆({F4(res.Roundness)}), 对号位置大小正常";
            return res;
        }

        /// <summary>
        /// 沿 72 个方向量圈的内沿 / 外沿半径(量不到的是 NaN)。
        /// 每条射线从 0.62 R0 到 1.30 R0 每 0.25 像素双线性取一个"三通道最小值", 找峰值;
        /// 内沿 = 峰值往里第一次降到 (峰值 + 里侧底色) / 2 的位置, 外沿同理用外侧底色; 都线性插值到亚像素。
        /// 峰值比两侧底色亮不到 40、半高范围里碰到别的白块(对号、标题字), 这条就不要。
        /// </summary>
        static void RingRays(byte[,] mn, int[,] lab, Comp c, out double cx, out double cy,
                             out double[] rin, out double[] rout)
        {
            long sx = 0, sy = 0; int cnt = 0;
            for (int r = 0; r < c.H; r++)
                for (int x = 0; x < c.W; x++)
                    if (lab[c.Y + r, c.X + x] == c.Label) { sx += x; sy += r; cnt++; }
            cx = c.X + (double)sx / cnt;
            cy = c.Y + (double)sy / cnt;
            double r0 = (c.W + c.H) / 4.0;
            double lo = RLow * r0, hi = RHigh * r0;
            int n = (int)Math.Ceiling((hi - lo) / Step);
            var rs = new double[n];
            for (int k = 0; k < n; k++) rs[k] = lo + k * Step;
            rin = new double[NAng];
            rout = new double[NAng];
            int h = mn.GetLength(0), w = mn.GetLength(1);
            var px = new double[n];
            var py = new double[n];
            var p = new double[n];
            for (int k = 0; k < NAng; k++)
            {
                rin[k] = double.NaN; rout[k] = double.NaN;
                double t = 2.0 * Math.PI * k / NAng;
                double ct = Math.Cos(t), st = Math.Sin(t);
                bool ok = true;
                for (int i = 0; i < n; i++)
                {
                    px[i] = cx + rs[i] * ct;
                    py[i] = cy + rs[i] * st;
                    int x0 = (int)Math.Floor(px[i]), y0 = (int)Math.Floor(py[i]);
                    if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) { ok = false; break; }
                }
                if (!ok) continue;
                for (int i = 0; i < n; i++)
                {
                    int x0 = (int)Math.Floor(px[i]), y0 = (int)Math.Floor(py[i]);
                    double fx = px[i] - x0, fy = py[i] - y0;
                    p[i] = (double)mn[y0, x0] * (1 - fx) * (1 - fy) + (double)mn[y0, x0 + 1] * fx * (1 - fy)
                         + (double)mn[y0 + 1, x0] * (1 - fx) * fy + (double)mn[y0 + 1, x0 + 1] * fx * fy;
                }
                int pk = 0;
                for (int i = 1; i < n; i++) if (p[i] > p[pk]) pk = i;
                if (pk == 0 || pk == n - 1) continue;
                double topv = p[pk], bIn = double.MaxValue, bOut = double.MaxValue;
                for (int i = 0; i < pk; i++) if (p[i] < bIn) bIn = p[i];
                for (int i = pk; i < n; i++) if (p[i] < bOut) bOut = p[i];
                if (topv - Math.Max(bIn, bOut) < MinContrast) continue;
                double? a = Cross(p, rs, pk, 0, (topv + bIn) / 2.0);
                double? b = Cross(p, rs, pk, n - 1, (topv + bOut) / 2.0);
                if (a == null || b == null) continue;
                // 半高范围(各外扩 1 像素)里碰到别的白块就不要这条
                int kLo = Math.Max(0, SearchSorted(rs, a.Value) - 4);
                int kHi = Math.Min(n - 1, SearchSorted(rs, b.Value) + 4);
                bool hit = false;
                for (int kk = kLo; kk <= kHi && !hit; kk++)
                {
                    int gx = (int)Math.Floor(px[kk] + 0.5), gy = (int)Math.Floor(py[kk] + 0.5);
                    if (gx >= 0 && gx < w && gy >= 0 && gy < h)
                    {
                        int v = lab[gy, gx];
                        if (v != 0 && v != c.Label) hit = true;
                    }
                }
                if (hit) continue;
                rin[k] = a.Value; rout[k] = b.Value;
            }
        }

        /// <summary>剖面 p 从下标 k0 往 k1 走, 第一次越过 level 的亚像素半径; 没越过返回 null。</summary>
        static double? Cross(double[] p, double[] rs, int k0, int k1, double level)
        {
            int step = k1 >= k0 ? 1 : -1;
            for (int k = k0; k != k1; k += step)
            {
                double a = p[k], b = p[k + step];
                if ((a - level) * (b - level) <= 0 && a != b)
                {
                    double t = (level - a) / (b - a);
                    return rs[k] + t * (rs[k + step] - rs[k]);
                }
            }
            return null;
        }

        // 第一个 >= v 的下标(都比 v 小时返回长度), 和 numpy.searchsorted(side="left") 一样
        static int SearchSorted(double[] rs, double v)
        {
            for (int i = 0; i < rs.Length; i++) if (rs[i] >= v) return i;
            return rs.Length;
        }

        /// <summary>以 centerDeg 为中心、前后 20 度扇区里的直径平均(不到 3 个返回 null)。逐个相加, 和 Python 对数。</summary>
        static double? SectorMean(double[] d, double centerDeg)
        {
            double acc = 0; int cnt = 0;
            for (int j = 0; j < NAng / 2; j++)
            {
                double ang = j * (360.0 / NAng);
                double m = ((ang - centerDeg) + 90.0) % 180.0;
                if (m < 0) m += 180.0;                 // C# 的 % 对负数给负数, Python 给正数
                if (Math.Abs(m - 90.0) <= SectorHalf && !double.IsNaN(d[j])) { acc += d[j]; cnt++; }
            }
            return cnt >= 3 ? acc / cnt : (double?)null;
        }

        // ================= 判据二: 没找到圈 =================

        static CircleResult JudgeMissing(CircleResult res, Vec3b[,] px, byte[,] mn, int[,] lab, List<Comp> comps,
                                         int adh, int ay0Band, int W, int bh)
        {
            if (adh < MinAmountHeightForMissing)
            { res.Reason = "没找到圈; 图太小, 不判有没有圈"; return res; }
            var row = TitleRow(comps, adh, ay0Band, W);
            if (row == null)
            { res.Reason = "没找到圈, 也没找到白色标题行, 不判"; return res; }
            int rt = row.Min(c => c.Y), rb = row.Max(c => c.Y + c.H);
            int tr = rb - rt;
            // 字块按横向重叠并成簇, 只拿够高的簇定标题左右沿: 发灰的圈压暗后只剩几个碎点, 不能让碎点把左沿拉过去
            var cl = RowClusters(row).Where(k => k.B - k.T >= MainH * tr).ToList();
            if (cl.Count < 2)
            { res.Reason = "没找到圈, 标题行够高的字簇不到两个, 不判"; return res; }
            int rl = cl[0].L, rr = cl[cl.Count - 1].R;
            double rowCx = (rl + rr) / 2.0;
            if (tr < RowMinTr || !(RowTrLow * adh <= tr && tr <= RowTrHigh * adh) || rr - rl < 2.0 * tr
                || !(RowCxLow * W <= rowCx && rowCx <= RowCxHigh * W))
            { res.Reason = "没找到圈, 标题行不像标题, 不判"; return res; }
            res.TitleRowFound = true;

            // 标题字够不够白: 页面被半透明遮罩压暗时圈也发灰, 白色门槛下圈会散掉, 这种不判
            var hist = new long[256];
            long tot = 0;
            foreach (var k in cl)
                foreach (var c in k.Members)
                    for (int y = c.Y; y < c.Y + c.H; y++)
                        for (int x = c.X; x < c.X + c.W; x++)
                            if (lab[y, x] == c.Label) { hist[mn[y, x]]++; tot++; }
            long want = (long)Math.Ceiling(0.9 * tot), accu = 0;
            int bright = 255;
            for (int v = 0; v < 256; v++) { accu += hist[v]; if (accu >= want) { bright = v; break; } }
            res.TitleBrightness = bright;
            if (bright < TitleBright)
            { res.Reason = $"标题字发灰({bright}), 页面可能被压暗, 不判"; return res; }

            res.LeadGap = (cl[1].L - cl[0].R) / (double)adh;
            var others = cl.Skip(1).Select(k => k.B - k.T).OrderBy(v => v).ToList();
            int m = others.Count;
            double med = m % 2 == 1 ? others[m / 2] : (others[m / 2 - 1] + others[m / 2]) / 2.0;
            res.LeadTall = (cl[0].B - cl[0].T) / med;
            if (res.LeadGap >= LeadGapMax || res.LeadTall > LeadTallMax)
            { res.Reason = "标题行最左边有个单独的东西(可能是没认出来的图标), 不判"; return res; }
            // 最左一簇近似正方形又带孔: 多半是对号和圈粘在一起、没认出来的圈
            int lw = cl[0].R - cl[0].L, lh = cl[0].B - cl[0].T;
            long holeArea = 0;
            foreach (var c in cl[0].Members)
                foreach (var o in HolesBox(lab, c)) holeArea += o.Area;
            double lar = (double)lw / lh;
            if (LeadSquareLow <= lar && lar <= LeadSquareHigh && holeArea >= LeadHole * lw * lh)
            { res.Reason = "标题行最左边的字块像个圈(带孔), 不判"; return res; }
            // 标题正上方还有白块(不碰标题带上沿, 那是状态栏): 图标在标题上面的别的页型
            foreach (var c in comps)
            {
                double ccx = c.X + c.W / 2.0;
                if (c.Y > 0 && c.Y + c.H <= rt && c.H >= AboveH * adh
                    && rl - AbovePad * adh <= ccx && ccx <= rr + AbovePad * adh)
                { res.Reason = "标题上面还有图标或字(别的页型), 不判"; return res; }
            }
            // 标题这一行别处有个圈(比如整张图是镜像的, 圈跑到标题右边): 有图标只是不在左边, 不判。
            // 圈 = 近似正方形、细(像素 <= 外框 35%)、正好一个占外框 75% 以上的大孔; 字里的 口 日 贝 孔小、笔画粗, 不会算进来
            foreach (var c in comps)
            {
                double ar = (double)c.W / c.H, ccy = c.Y + c.H / 2.0;
                if (!(RingHLow * W <= c.H && c.H <= RingHHigh * W && RingArLow <= ar && ar <= RingArHigh
                      && c.Area <= RingFill * c.W * c.H)) continue;
                if (!(rt - 0.5 * tr <= ccy && ccy <= rb + 0.5 * tr)) continue;
                if (HolesBox(lab, c).Count(o => o.W >= RingHole * c.W && o.H >= RingHole * c.H) == 1)
                { res.Reason = "标题这一行别处有个圈(镜像图或别的版式), 不判"; return res; }
            }

            double rcy = (rt + rb) / 2.0;
            int sx0 = Math.Max(0, (int)(rl - SlotXFar * adh));
            int sx1 = Math.Max(0, (int)(rl - SlotXNear * adh));
            int sy0 = Math.Max(0, (int)(rcy - SlotHalf * adh));
            int sy1 = Math.Min(bh, (int)(rcy + SlotHalf * adh));
            if (sx1 - sx0 < SlotMinW * adh || sy1 - sy0 < 4)
            { res.Reason = "标题左边没有空间, 不判"; return res; }
            int gx0 = (int)(0.05 * W), gx1 = (int)(0.95 * W);
            int fy0 = Math.Max(0, (int)(rt - 0.5 * tr));
            int fy1 = Math.Min(bh, (int)(rb + 0.5 * tr));
            var bg = BgAndForeign(px, mn, gx0, gx1, rt, rb, fy0, fy1, out double foreign);
            res.ForeignRatio = foreign;
            if (bg == null) { res.Reason = "标题行没有底色, 不判"; return res; }
            if (foreign > ForeignMax)
            { res.Reason = $"标题行附近有别的颜色({F3(foreign)}), 不判"; return res; }

            var (bR, bG, bB) = bg.Value;
            long nb = 0, all = 0;
            for (int y = sy0; y < sy1; y++)
                for (int x = sx0; x < sx1; x++)
                {
                    var v = px[y, x];
                    double dR = v.Item2 - bR, dG = v.Item1 - bG, dB = v.Item0 - bB;
                    if (dR * dR + dG * dG + dB * dB > SlotDist2) nb++;
                    all++;
                }
            res.SlotFraction = (double)nb / all;
            if (res.SlotFraction <= SlotTolerance)
            {
                res.Verdict = CircleVerdict.Suspicious;
                res.Reason = "标题前没有圈对号(标题左边是空的蓝底)";
                return res;
            }
            res.Reason = "标题前有东西, 但认不出是圈对号, 不判";
            return res;
        }

        /// <summary>金额上面那一行白色标题字: 竖向重叠 >= 一半的字块连成行, 取最靠下、>= 3 块的那一行。</summary>
        static List<Comp>? TitleRow(List<Comp> comps, int adh, int ay0Band, int W)
        {
            var cand = new List<Comp>();
            foreach (var c in comps)
            {
                double above = ay0Band - (c.Y + c.H / 2.0), ccx = c.X + c.W / 2.0;
                if (RowHLow * adh <= c.H && c.H <= RowHHigh * adh && c.W <= RowWMax * adh
                    && RowCyLow * adh <= above && above <= RowCyHigh * adh
                    && RowXLow * W <= ccx && ccx <= RowXHigh * W)
                    cand.Add(c);
            }
            int n = cand.Count;
            if (n < TitleMin) return null;
            var parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;
            int Find(int a)
            {
                while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
                return a;
            }
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++)
                {
                    int t = Math.Max(cand[p].Y, cand[q].Y);
                    int b = Math.Min(cand[p].Y + cand[p].H, cand[q].Y + cand[q].H);
                    if (b - t >= 0.5 * Math.Min(cand[p].H, cand[q].H))
                    {
                        int ra = Find(p), rb = Find(q);
                        if (ra != rb) parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
                    }
                }
            var order = new List<int>();
            var groups = new Dictionary<int, List<Comp>>();
            for (int p = 0; p < n; p++)
            {
                int r = Find(p);
                if (!groups.TryGetValue(r, out var g)) { g = new List<Comp>(); groups[r] = g; order.Add(r); }
                g.Add(cand[p]);
            }
            List<Comp>? best = null;
            int bestBot = int.MinValue, bestLeft = int.MaxValue;
            foreach (int r in order)
            {
                var g = groups[r];
                if (g.Count < TitleMin) continue;
                int bot = g.Max(c => c.Y + c.H), left = g.Min(c => c.X);
                // 取最靠下的行; 一样低取最靠左的
                if (best == null || bot > bestBot || (bot == bestBot && left < bestLeft))
                { best = g; bestBot = bot; bestLeft = left; }
            }
            return best;
        }

        /// <summary>一行字块按横向重叠并成簇, 从左到右。</summary>
        static List<Cluster> RowClusters(List<Comp> row)
        {
            var outp = new List<Cluster>();
            foreach (var c in row.OrderBy(c => c.X).ThenBy(c => c.Y))
            {
                if (outp.Count > 0 && c.X < outp[outp.Count - 1].R)
                {
                    var k = outp[outp.Count - 1];
                    k.R = Math.Max(k.R, c.X + c.W);
                    k.T = Math.Min(k.T, c.Y);
                    k.B = Math.Max(k.B, c.Y + c.H);
                    k.Members.Add(c);
                }
                else
                {
                    var k = new Cluster { L = c.X, R = c.X + c.W, T = c.Y, B = c.Y + c.H };
                    k.Members.Add(c);
                    outp.Add(k);
                }
            }
            return outp;
        }

        /// <summary>
        /// 底色 = [y0,y1) x [x0,x1) 里非白像素(三通道最小值 &lt;= 140)每个通道的下中位数;
        /// 再算 [fy0,fy1) x [x0,x1) 里离"底色 -> 白"线段超过 80 的像素占比。
        /// ★ 口径和 Python 一样: 按 R、G、B 的顺序累加, 拿距离平方和 6400 比, 不开方。
        /// </summary>
        static (double R, double G, double B)? BgAndForeign(Vec3b[,] px, byte[,] mn, int x0, int x1, int y0, int y1,
                                                            int fy0, int fy1, out double foreign)
        {
            var hb = new long[256]; var hg = new long[256]; var hr = new long[256];
            long n = 0;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                    if (mn[y, x] <= WhiteMin) { var v = px[y, x]; hb[v.Item0]++; hg[v.Item1]++; hr[v.Item2]++; n++; }
            if (n == 0) { foreign = 1.0; return null; }
            double aR = LowerMedian(hr, n), aG = LowerMedian(hg, n), aB = LowerMedian(hb, n);
            double dR = 255 - aR, dG = 255 - aG, dB = 255 - aB;
            double L = dR * dR + dG * dG + dB * dB;
            long tot = 0, far = 0;
            for (int y = fy0; y < fy1; y++)
                for (int x = x0; x < x1; x++)
                {
                    var v = px[y, x];
                    double pR = v.Item2 - aR, pG = v.Item1 - aG, pB = v.Item0 - aB;
                    double t = L == 0 ? 0 : (pR * dR + pG * dG + pB * dB) / L;
                    if (t < 0) t = 0; else if (t > 1) t = 1;
                    double eR = pR - t * dR, eG = pG - t * dG, eB = pB - t * dB;
                    tot++;
                    if (eR * eR + eG * eG + eB * eB > ForeignDist2) far++;
                }
            foreign = tot == 0 ? 0 : (double)far / tot;
            return (aR, aG, aB);
        }

        static double LowerMedian(long[] hist, long n)
        {
            long want = (n + 1) / 2, acc = 0;
            for (int v = 0; v < 256; v++)
            {
                acc += hist[v];
                if (acc >= want) return v;
            }
            return 255;
        }

        // ================= 公用 =================

        /// <summary>字块里的孔: 外接框外补一圈背景, 背景 4 连通, 不碰外圈的就是孔。坐标相对字块外接框。</summary>
        static List<HoleBox> HolesBox(int[,] lab, Comp c)
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
            var outp = new List<HoleBox>();
            var stack = new Stack<(int r, int x)>();
            for (int r0 = 0; r0 < h; r0++)
                for (int x0 = 0; x0 < w; x0++)
                {
                    if (!bgPix[r0, x0] || seen[r0, x0]) continue;
                    int minX = x0, maxX = x0, minY = r0, maxY = r0, area = 0;
                    seen[r0, x0] = true;
                    stack.Push((r0, x0));
                    while (stack.Count > 0)
                    {
                        var (r, x) = stack.Pop();
                        area++;
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
                    outp.Add(new HoleBox(minX - 1, minY - 1, maxX - minX + 1, maxY - minY + 1, area));
                }
            return outp;
        }

        // 上三分之一的通道均值判断蓝底(和 YenCheck / AmountFontCheck 同一口径)
        static bool IsBluePage(Mat bgr)
        {
            int h3 = Math.Max(1, bgr.Height / 3);
            using var top = new Mat(bgr, new Rect(0, 0, bgr.Width, h3));
            var m = Cv2.Mean(top);            // BGR 顺序: Val0=B, Val1=G, Val2=R
            return m.Val0 > m.Val2 + 25 && m.Val0 > m.Val1 + 15;
        }

        /// <summary>蓝底页的金额行 —— 白字蓝底(和 YenCheck / AmountFontCheck 同一口径)。</summary>
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

        static string F4(double v) => v.ToString("F4", System.Globalization.CultureInfo.InvariantCulture);
        static string F3(double v) => v.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);
        static string S3(double v) => (v >= 0 ? "+" : "") + F3(v);
    }
}
