#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum YenVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // ¥ 的笔画粗细和真图对不上
    }

    /// <summary>
    /// 这张图是苹果还是安卓 —— **由调用方给**, 本类不自己判。
    ///
    /// ★★★★★ 为什么必须外面给: ¥ 是**系统回退字形**, 苹果细安卓粗。
    ///   实测**安卓真图的数值和苹果假图一模一样**(0.726~0.733 对 0.736~0.739),
    ///   拿 ¥ 自己去定机型再回头判 ¥ 是循环论证。机型只能来自**状态栏**那条独立信号
    ///   (2026-07-19 交付的设备模型, 12 万张跑过)。
    /// </summary>
    public enum YenPlatform
    {
        Unknown = 0,
        Apple = 1,
        Android = 2,
    }

    public sealed class YenResult
    {
        public YenVerdict Verdict;
        public double StemRatio;        // ★ 判据: ¥ 竖笔等效宽 / 数字笔画等效宽
        /// <summary>
        /// ¥ 横杠等效厚 / 数字笔画等效宽。**只输出, 现在不参与判定。**
        ///
        /// ★★ 它**接近**一条独立判据, 但**不是完整的**, 留着当后手:
        ///   5,199 张 iPhone 真图 min 0.2730  p50 0.3161  p99 0.3203  p99.9 0.3243  max 0.3881;
        ///   两张已知假图 0.3630 / 0.3740。
        ///   卡在 0.363 的话真图会漏出 **1 张 / 5,199 = 0.019%** —— 不是零重叠。
        ///
        /// ★★★★★ 这一条是**被更大的样本推翻过一次的**: 先在 4,141 张上量, 真图最大才 0.3256,
        ///   当时写的是"都在真图最大值之上, 零重叠"; 换成 9,529 张重扫, 冒出一张 0.3881。
        ///   **max 是最不稳的统计量, 样本一大就变。** 以后凡是靠"最大值"下的结论都要这么复核一遍。
        ///
        /// ★ 仍然有用的地方: 量的是**横杠**, StemRatio 量的是**竖笔**, 两处不同的笔画。
        ///   对方把竖笔调对了, 横杠还得再对一次。
        /// ★ 现在不接进判定: 每多一条判据多一份误报面, 而且这条**还没有安卓标定**
        ///   (安卓真图中位 0.4072, 离苹果的 0.3161 很远, 必须单独标)。
        /// </summary>
        public double BarRatio;
        public double YenHeightRatio;   // ¥ 高 / 数字高; 真图很稳, 用来确认抓到的确实是 ¥
        /// <summary>
        /// (数字底 - ¥ 底) / 数字高。客户(玄武)说苹果真图这里应当为 0(底边齐平)。
        ///
        /// ★★★★★ **只输出, 绝不判定 —— 这条在语料上已经被证伪了。**
        ///   5,199 张 iPhone 真图: min 0.0000  p50 0.0098  p99 0.0145  max 0.0194。
        ///   假图 013 = 0.0185 落在第 **99.79** 百分位, 假图 2.png = 0.0280 在第 100 百分位。
        ///   **但真图 014.jpg 是 0.0192, 落在第 99.81 百分位 —— 比假图 013 还靠上。**
        ///   方向是对的(假的确实更悬空), 可是真假压在一起, **卡哪儿都会误伤真图**。
        ///
        /// ★ 客户描述的现象是真的, 但"人眼看得出"不等于"量得开"。留着这一栏是为了
        ///   以后有更多样本时能重新评估, 不是为了现在拿来判。
        /// </summary>
        public double BottomGap;
        public double DigitHeight;      // 数字中位高(像素)
        public double DigitStroke;      // 数字笔画等效宽(像素, 亚像素)
        /// <summary>
        /// 金额数字自己的笔画粗细 = DigitStroke / DigitHeight。
        /// ★★★ 这一项是 <see cref="StemRatio"/> 的**分母的健康度**。真图两个平台都在 0.155~0.168,
        ///   偏出去说明**数字本身**就不是真图那个字体 —— 那时候 StemRatio 低不代表 ¥ 细,
        ///   而是分母被撑大了。两种异常要分开说, 不然报告会把人往错的方向带。
        /// </summary>
        public double DigitStrokeNorm;
        public double StemWidth;        // ¥ 竖笔等效宽(像素, 亚像素)
        public int DigitCount;
        public YenPlatform Platform;    // 原样回显调用方给的机型
        public bool Measured;           // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// **蓝底转账页**金额前面那个 ¥ 的字形检查: ¥ 竖笔粗细 / 同一行数字笔画粗细。
    /// 同一张图内部的比值, 不查表、不挑机型、不随分辨率变。
    ///
    /// 和谁的关系
    /// ----------
    /// MinusCheck / DotCheck / FontCheck 管的是
    /// **白底账单详情页**, 而且 MinusCheck 明确写着"蓝底转账页金额不带负号"直接返回。
    /// 本类反过来: **只管蓝底页**, 白底页直接返回 CannotDetermine。两边不重叠, 互不引用。
    ///
    /// 判据从哪来
    /// ----------
    /// 经理 2026-09-16 派活: "你试试用上次识别负号的方式" "只能硬写算法了" "差距太小了"。
    /// 客户(国王支付-玄武)给的现象: **苹果蓝图的 ¥ 是细线, 安卓的是粗线**。
    /// 经理的解释: "他p图用错字体了"。
    ///
    /// 在经理给的两张原图上量(2.png 假 / true.png 真), 以及客户发的四张压缩图上复核:
    ///
    ///     苹果 真   0.5880 / 0.5884
    ///     苹果 假   0.7362 / 0.7392
    ///     安卓 真   0.7260 / 0.7327
    ///
    /// ★★★ 真图那两个数一个来自 1170x2532 原图, 一个来自 591x1280 的压缩图,
    ///   **量出来只差 0.0004** —— 这条判据对缩放和压缩是稳的。
    ///   (经理 2026-08-30 担心的"缩放后就没法区分", 靠的就是下面那个亚像素量法解决的。)
    ///
    /// ★★★★★ 安卓真图 0.726~0.733 和苹果假图 0.736~0.739 **挨在一起**。
    ///   所以**不给机型就不判**, 见 <see cref="YenPlatform"/>。
    ///
    /// 怎么量粗细 —— 不是数像素
    /// ------------------------
    /// 抗锯齿让笔画边缘是一道斜坡。二值化之后数像素, 阈值卡高卡低会让**细笔画**的
    /// 相对误差远大于粗笔画 —— 同样六张图, 阈值取 200 量出 0.500/0.750,
    /// 换 Otsu 就变成 0.625/0.706, **判据强弱全看阈值定在哪**, 站不住。
    ///
    /// 所以改成**等效宽度**: 把一段笔画的灰度积分再除以字底色差
    ///     宽度 = Σ (I - 底色) / (字色 - 底色)
    /// 模糊只是把能量摊开, **积分不变**。于是对阈值、抗锯齿、轻度压缩都不敏感,
    /// 而且是连续值, 不会被"4 像素和 6 像素之间取不到值"卡死。
    /// 二值图只用来**找笔画在哪**, 不用来量它有多宽。
    ///
    /// 语料标定(本机 TempFakeImages 抽 8,000 张, 量到 7,609 张蓝图)
    /// -----------------------------------------------------------
    /// 比值分布是**干净的双峰**, 中间几乎是空的:
    ///
    ///     0.55~0.60   4,200 张        &lt;- 低簇
    ///     0.60~0.65      12 张        &lt;- 空档
    ///     0.65~0.70     801 张
    ///     0.70~0.75   2,574 张        &lt;- 高簇
    ///
    /// ★★★★★ 按分辨率分组, **42 种分辨率没有一种落在空档里**, 而且分得和机型完全一致:
    ///   低簇全是 iPhone 分辨率(1170x2532 / 1179x2556 / 1284x2778 / 1290x2796 /
    ///   1206x2622 / 1320x2868 / 1242x2688 / 1125x2436 / 828x1792),
    ///   高簇全是安卓分辨率(1080x2400 / 1080x2340 / 720x1600 / 1440x3200 ...)。
    ///   **客户说的"苹果细线安卓粗线", 在 7,609 张上独立复核成立。**
    ///
    /// 覆盖边界
    /// --------
    /// - 只认"¥ 画错了"这一种痕迹。照着正确字体把 ¥ 打出来就绕过去了。
    /// - 报出来的意思是"这张图的 ¥ 和真图对不上", **不是**"这张图一定是假的"。
    /// - 安卓那条带是拿**真图**标出来的(没有安卓假样本), 只能说"不像真安卓"。
    ///
    /// ★★★★★ 最大的误报来源不是这条判据, 是**机型判错** —— 这条是量出来的, 不是推的
    /// ------------------------------------------------------------------------
    /// 两条带几乎不重叠(苹果 0.50~0.65, 安卓 0.64~0.80)。所以**机型给错一张, 基本必报一张**:
    /// 真安卓图被当成苹果 -> 0.727 落在苹果带外 -> 报;
    /// 真苹果图被当成安卓 -> 0.585 落在安卓带外 -> 报。
    ///
    /// 语料上两个数摆在一起就很清楚:
    ///
    ///     机型认对(按 Apple 官方分辨率表)  5,199 张 iPhone  ->  **误报 0 张 = 0.000%**
    ///     机型用分辨率粗略代理            4,209 张 非iPhone ->  报出 91 张 = 2.162%
    ///         91 张**全是偏细的, 一张偏粗都没有**; 其中 90.1% 落在 iPhone 的正常区间里,
    ///         数字笔画也正常 —— 就是**被裁过/缩过的 iPhone 截图**, 不是字形有问题
    ///
    /// ★★★★★ **判据自己的误报是 0, 2.37% 全是机型认错贡献的。**
    ///
    /// ★★★ 所以调用方要么只在设备模型**高置信**时才传机型, 要么就传 Unknown 让它只量不判。
    ///   拿不准的时候传 Unknown 是安全的 —— 数照样出得来。
    ///
    /// 经理提的"图片定位"(模板库)方案 —— 测过了, **它是行的**
    /// ------------------------------------------------------
    /// 已经做出来了, 就是 `YenTemplateCheck.cs`(平均模板 + 相关系数)。
    ///     苹果真图 2,296 张, 阈值取 p0.1 = 0.941, 误报 2 张 = 0.087%
    ///     两张已知假图 0.8278 / 0.8652, 都报得出来
    ///
    /// ★★★ 更正一条我先前写错的: 我曾按 2 张安卓样本(0.9134/0.9343)说"卡 0.95
    ///   能把苹果真图、安卓真图、假图三者都分开"。换 1,764 张安卓真图重算,
    ///   **安卓最高能到 0.9962** —— 0.95 分不开。**模板法同样分不了机型。**
    ///
    /// ★★★ 选了标量做主判据, 理由是**抗缩放**(两张原图各做 16 个缩放/重压缩变体实测):
    ///                1.0 倍    0.35 倍    空档缩水
    ///     标量空档    0.149     0.125      -16%
    ///     模板空档    0.212     0.072      **-66%**
    ///   降采样会把区别糊掉, **假图的模板分反而从 0.788 爬到 0.883**; 标量几乎不动。
    ///   另外模板法要随代码带一份库、还得分平台维护, 支付宝换字体就得重建。
    ///
    /// ★ 两条方法互相独立(笔画粗细比 vs 形状相关), 在 6 张有标注的样本上结论完全一致。
    ///   要再加一道保险, 模板法是现成的第二意见。
    ///
    /// 还有两类图这条判据管不了
    /// ------------------------
    /// - **翻拍照片**(实测 3072x4096 这种 4:3 的相机图): 镜头畸变 + 屏幕摩尔纹会把笔画量歪,
    ///   实测 11 张里 7 张落在苹果区间。调用方该先用截图/照片判据挡一道。
    /// - **裁过或缩过的截图**: 尺寸不再是原生分辨率, 机型代理失效(见上)。
    ///
    /// 入参只读, 内部全部用 ROI 视图, 不复制整图, 不碰 System.Drawing。
    /// 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class YenCheck
    {
        /// <summary>
        /// 苹果真图的正常区间。两侧都判: 换字体重打既可能偏粗也可能偏细。
        ///
        /// ★★★ 标定口径: 机型用 **Apple 官方分辨率表**认(不是用比值自己认, 那是循环论证)。
        ///   9,529 张语料里 5,199 张 iPhone 分辨率的蓝图:
        ///     min 0.5456   p1 0.5547   p50 0.5857   p99 0.5889   p99.9 0.6004   max 0.6086
        ///   **整个真图分布只占 0.546~0.609**, 两张已知假图在 0.7365 / 0.7392, 中间空了一大片。
        ///
        /// ★★★★★ 上界为什么取 0.65 而不是贴着真图上界:
        ///   0.62~0.66 在这 5,199 张上**误报都是 0 张**, 所以这一段随便取。
        ///   取 0.65 = 比真图最大值(0.6089)高 6.7%, 比已知假图(0.7365)低 11.7%, 两头都留足。
        ///   **贴着 0.62 只比真图最大值高 1.8%** —— 而我们手上**一张"到账成功"页的真图都没有**,
        ///   那个页型万一 ¥ 差个 3% 就会整片误报。宁可留余量。
        ///   等拿到到账成功页的真样本, 确认它也落在 0.548~0.609, 再收紧不迟。
        ///
        /// ★ 下界 0.50 比真图最小值(0.5480)低 8.8%。低侧纯属补位(没有"¥ 偏细"的已知假图)。
        /// </summary>
        public const double AppleLow = 0.500, AppleHigh = 0.650;

        /// <summary>
        /// 安卓真图的正常区间。
        ///
        /// ★★ **没有一张安卓假样本。** 这条带是拿真图分布标出来的,
        ///   所以它只能说"不像真安卓", 不能说"是假的"。
        ///
        /// 剔掉"整组都像苹果"的分辨率(见下)之后, 估计的真安卓 3,317 张:
        ///     p1 0.6799   p5 0.6915   p50 0.7270   p95 0.7368   p99 0.7434   max 0.7855
        ///   [0.64, 0.80] 报出 25 张 = 0.754%。★ 里面**没有一张是偏粗的**, 全是偏细。
        ///
        /// ★★★★★ 那 0.754% 基本不是这条判据的误报, 是**机型认错**:
        ///   拿分辨率当机型代理时, 被裁过/缩过的 iPhone 截图会落到非 iPhone 分辨率上
        ///   (实测 1260x2736 / 1280x2781 / 1280x2774 / 1280x2769 / 960x2079 这几种
        ///   **整组 100% 都像苹果**, 数字笔画却完全正常 —— 就是 iPhone 图换了个尺寸)。
        ///   线上用状态栏那个设备模型(对缩放伪造实测拦 99.8%)就不会这么错。
        /// </summary>
        public const double AndroidLow = 0.640, AndroidHigh = 0.800;

        /// <summary>
        /// 数字中位高低于此值不判定; 设为 0 可关闭。
        /// ★ 缩放实测: 缩到数字高 36 像素 + JPEG 65, 真假仍差 0.1255 且不重叠,
        ///   所以 40 是留了余量的。语料里数字高 p1 = 68, 这道闸基本不损覆盖。
        /// </summary>
        public const double MinDigitHeight = 40;

        /// <summary>
        /// 金额数字笔画粗细(笔画/字高)的正常区间 —— **和机型无关**, 苹果 0.1576 安卓 0.1582, 一样。
        ///
        /// ★★★★★ 这条是**测出来才加的**。拿阈值在语料上报出 11 张, 逐张看图 + 回查数值发现:
        ///   其中 9 张的 StemRatio 之所以低, 不是 ¥ 细, 是**数字粗**(0.188~0.204, 真图 p95 才 0.167)。
        ///   只按 StemRatio 报, 理由会写成"¥ 竖笔比 0.5451", 而那张图的 ¥ 其实没问题 ——
        ///   **理由是错的, 会把复核的人带偏。**
        ///
        /// 标定(9,529 张的那一轮, 机型按 Apple 官方分辨率表认):
        ///   5,199 张 iPhone 真图  min 0.1531  p50 0.1576  p99 0.1681  p99.9 0.1698  max 0.1753
        ///   上界扫描: 0.175 -> 误报 2 张(0.038%);  **0.176 及以上 -> 0 张**
        ///   查出来的"数字偏粗"异常聚在 0.188~0.204, 另有一张 0.2520。
        ///
        /// ★ 所以取 0.178: 比真图最大值(0.1753)高 1.4%, 比异常最低值(0.188)低 5.4%。
        ///   两边都不宽裕 —— 这条判据本身就比竖笔比那条紧得多, 报出来的更要人工看。
        /// ★★ 先写 0.175 是在 6,801 张上定的, 换 9,529 张重扫就冒出 2 张误报。
        ///   和 BarRatio 那次一样, **都是样本变大之后尾巴伸出来**。
        /// </summary>
        public const double DigitStrokeNormLow = 0.145, DigitStrokeNormHigh = 0.178;

        /// <summary>¥ 高 / 数字高 的护栏。真图实测 0.691~0.726, 卡宽一点。</summary>
        public const double YenHeightLow = 0.60, YenHeightHigh = 0.80;

        /// <summary>某一行的前景占 ¥ 宽度这个比例以上, 就算横杠所在行。</summary>
        const double BarFill = 0.70;

        /// <summary>量等效宽度时向两侧各让这么多像素, 把抗锯齿的斜坡收进来。</summary>
        const int Margin = 2;

        public const int MaxComponents = 20_000;   // 上限保护

        struct Comp
        {
            public int X, Y, W, H, Area, Label;
        }

        /// <param name="image">
        /// 8 位或 16 位 **BGR(A)** 图(Cv2.ImRead / ImDecode 就是); 16 位会先降成 8 位。
        /// ★ 单通道进来判不了 —— 蓝底页这道闸要颜色。位深/通道不支持时返回
        ///   CannotDetermine, **不抛异常**。传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        /// <param name="platform">
        /// 这张图是苹果还是安卓, 由调用方的设备模型给。
        /// ★ 不给(Unknown)时**照样把所有数量出来**, 只是不下判定 ——
        ///   这样没接设备模型也能先看数, 接上之后不用改调用处。
        /// </param>
        public static YenResult Check(Mat image, YenPlatform platform = YenPlatform.Unknown)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new YenResult { Verdict = YenVerdict.CannotDetermine, Platform = platform };
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
                // 16 位按 1/256 降到 8 位。★ 必须排在蓝底闸之前: 下面用的是 8 位绝对阈值
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
                { res.Reason = "不是蓝底转账页, 本类只管蓝底页"; return res; }

                var box = LocateAmountBlue(src);
                if (box == null) { res.Reason = "定位不到金额行"; return res; }

                // 行内重新取块, **不设高度下限**: 定位阶段的高度闸会把小数点之类滤掉。
                // 左侧外扩较多, 因为定位框的 x0 是按字算出来的。
                int pad = Math.Max(2, (int)(box.Value.Height * 0.15));
                int padx = Math.Max(pad, (int)(box.Value.Width * 0.30));
                int cx0 = Math.Max(0, box.Value.Left - padx), cy0 = Math.Max(0, box.Value.Top - pad);
                int cx1 = Math.Min(src.Width, box.Value.Right + pad);
                int cy1 = Math.Min(src.Height, box.Value.Bottom + pad);
                if (cx1 - cx0 < 8 || cy1 - cy0 < 8)
                { res.Reason = "金额行裁出来太小"; return res; }

                // 只把**这一小块**转灰度, 不转整帧 —— 后面一个整帧像素都用不到。
                // ★ 诚实说明: 本来以为这是耗时大头, 实测**几乎没差**(36.9 -> 39.5ms, 在噪声里)。
                //   真正的大头是逐像素取数组, 见 ToBytes。这里保留只裁不转整帧, 是因为
                //   少一次整帧分配(大图上省内存), 不是因为它快。
                using var bgrCrop = new Mat(src, Rect.FromLTRB(cx0, cy0, cx1, cy1));
                ownedGray = new Mat();
                Cv2.CvtColor(bgrCrop, ownedGray, ColorConversionCodes.BGR2GRAY);
                return Measure(ownedGray, res);
            }
            finally { ownedGray?.Dispose(); ownedBgr?.Dispose(); owned8?.Dispose(); }
        }

        // 上三分之一的通道均值判断蓝底; Mean 收 ROI 视图, 不复制
        static bool IsBluePage(Mat bgr)
        {
            int h3 = Math.Max(1, bgr.Height / 3);
            using var top = new Mat(bgr, new Rect(0, 0, bgr.Width, h3));
            var m = Cv2.Mean(top);            // BGR 顺序: Val0=B, Val1=G, Val2=R
            return m.Val0 > m.Val2 + 25 && m.Val0 > m.Val1 + 15;
        }

        /// <summary>
        /// 蓝底页的金额行 —— 白字蓝底。口径和 Python 侧 `locate_blue.locate_amount_blue` 一致。
        /// ★ 白底页那套(浅底深字)在蓝图上完全失效, 会误锁到红包促销卡, 所以要单独一套。
        /// </summary>
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
        static YenResult Measure(Mat sub, YenResult res)
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
            if (glyphs.Count < 4) { res.Reason = "块数不够"; return res; }   // ¥ 加至少三个数字
            // 排序取全序: 连通域返回顺序依实现而定, 并列时不能靠它决定
            glyphs = glyphs.OrderBy(g => g.X).ThenBy(g => g.Y).ThenBy(g => g.Label).ToList();

            var yen = glyphs[0];                                    // 最左边那块
            var rest = glyphs.Skip(1).ToList();
            int tall = rest.Max(g => g.H);
            // ★ 贴着裁块左右边缘的块是被切断的, 尺寸不可信
            var digits = rest.Where(g => g.H >= 0.80 * tall && g.X > 0 && g.X + g.W < cw).ToList();
            if (digits.Count < 3) { res.Reason = "数字不够三个"; return res; }

            int dTop = digits.Min(g => g.Y), dBot = digits.Max(g => g.Y + g.H);
            double dH = dBot - dTop;
            if (dH <= 0) { res.Reason = "数字高算不出来"; return res; }
            double dhMean = digits.Average(g => (double)g.H);
            if (Std(digits.Select(g => (double)g.H)) / dhMean > 0.08)
            { res.Reason = "数字高度不齐, 不像一行数字"; return res; }

            // ---- 把整块裁图取成数组, 下面要逐像素积分 ----
            var g8 = ToBytes(sub);
            var lab = ToInts(labels);

            // ---- ¥ 的两道护栏: 高度比例 + 必须正好两道横杠 ----
            // ★ 没有这两条, 版式一变就会把别的字当 ¥ 量, 而且量出来的数看着很正常。
            double yenHRatio = yen.H / dH;
            if (yenHRatio < YenHeightLow || yenHRatio > YenHeightHigh)
            { res.Reason = $"最左块高比 {yenHRatio:F3} 不像 ¥"; return res; }

            var barRows = new bool[yen.H];
            for (int r = 0; r < yen.H; r++)
            {
                int on = 0;
                for (int c = 0; c < yen.W; c++)
                    if (lab[yen.Y + r, yen.X + c] == yen.Label) on++;
                barRows[r] = on >= BarFill * yen.W;
            }
            var barSpans = Spans(barRows);
            if (barSpans.Count != 2)
            { res.Reason = $"最左块有 {barSpans.Count} 道横杠, 不像 ¥"; return res; }

            if (!Levels(g8, ToBytes(bin), cw, ch, out double bg, out double fg))
            { res.Reason = "金额行对比度不够"; return res; }

            // ---- 数字笔画: 每块自己的外接框内逐行量等效宽度 ----
            var dWidths = new List<double>();
            foreach (var d in digits) AddWidths(g8, lab, d, bg, fg, 0, d.H, dWidths);
            if (dWidths.Count == 0) { res.Reason = "数字量不出笔画"; return res; }
            double dStroke = Median(dWidths);
            if (dStroke <= 0) { res.Reason = "数字笔画为零"; return res; }

            // ---- ¥ 竖笔: 取**最后一道横杠以下**那几行, 那里只剩竖笔 ----
            // ★ 不要固定取"底部百分之多少": 有的字形横杠压得很低, 会把横杠当竖笔量, 比值直接翻倍。
            int stemFrom = barSpans[1].End;
            if (yen.H - stemFrom < 2) { res.Reason = "横杠以下没有竖笔"; return res; }
            var sWidths = new List<double>();
            AddWidths(g8, lab, yen, bg, fg, stemFrom, yen.H, sWidths);
            if (sWidths.Count == 0) { res.Reason = "竖笔量不出来"; return res; }
            double stem = Median(sWidths);

            res.Measured = true;
            res.DigitHeight = dH;
            res.DigitCount = digits.Count;
            res.DigitStroke = dStroke;
            res.DigitStrokeNorm = dStroke / dH;
            res.StemWidth = stem;
            res.StemRatio = stem / dStroke;
            res.BarRatio = BarThickness(g8, lab, yen, bg, fg, barSpans[0].Start, barSpans[1].End)
                           / dStroke;
            res.YenHeightRatio = yenHRatio;
            res.BottomGap = (dBot - (yen.Y + yen.H)) / dH;

            if (dH < MinDigitHeight)
            {
                res.Reason = $"数字高 {dH:F0} < {MinDigitHeight}, 判不准所以不判";
                return res;
            }

            // ★★★ 先判数字自己的笔画 —— 这一条**和机型无关**, 所以不给机型也能判。
            //   而且必须排在 StemRatio 前面: 数字粗了会把 StemRatio 的分母撑大,
            //   这时候报"¥ 竖笔比偏低"是**错的理由**。
            if (res.DigitStrokeNorm < DigitStrokeNormLow || res.DigitStrokeNorm > DigitStrokeNormHigh)
            {
                res.Verdict = YenVerdict.Suspicious;
                res.Reason = $"金额数字笔画 {res.DigitStrokeNorm:F4} 异常 "
                           + $"(真图 {DigitStrokeNormLow}~{DigitStrokeNormHigh}), 问题在数字字体不在 ¥";
                return res;
            }

            double lo, hi;
            switch (res.Platform)
            {
                case YenPlatform.Apple: lo = AppleLow; hi = AppleHigh; break;
                case YenPlatform.Android: lo = AndroidLow; hi = AndroidHigh; break;
                default:
                    // ★ 安卓真图的数和苹果假图挨在一起, 不知道机型就只能量不能判
                    res.Reason = $"竖笔比 {res.StemRatio:F4}; 数字笔画正常; 不知道是安卓还是苹果, 只量不判";
                    return res;
            }
            if (res.StemRatio < lo || res.StemRatio > hi)
            {
                res.Verdict = YenVerdict.Suspicious;
                res.Reason = $"¥ 竖笔比 {res.StemRatio:F4} (该机型正常 {lo}~{hi})";
            }
            else
            {
                res.Verdict = YenVerdict.Ok;
                res.Reason = $"¥ 竖笔比 {res.StemRatio:F4}";
            }
            return res;
        }

        /// <summary>
        /// 在一块的外接框内逐行量每一段笔画的**等效宽度**, 累加到 outp。
        ///
        /// ★★ 口径必须和 Python 侧 `yen_scan._stroke_widths` 一模一样:
        ///   窗口向两侧各让 Margin 个像素, 但**夹在同一行相邻两段之间, 并且不越出这一块的外接框**。
        ///   贴着框边的那一笔会少收半个像素的斜坡 —— 固定偏差, 比值里基本抵消。
        ///   要改口径必须两边一起改, 然后重新标定阈值。
        /// </summary>
        static void AddWidths(byte[,] gray, int[,] lab, Comp c, double bg, double fg,
                              int rowFrom, int rowTo, List<double> outp)
        {
            double amp = fg - bg;
            if (amp == 0) return;
            var row = new bool[c.W];
            for (int r = rowFrom; r < rowTo; r++)
            {
                int gy = c.Y + r;
                for (int x = 0; x < c.W; x++) row[x] = lab[gy, c.X + x] == c.Label;
                var spans = Spans(row);
                for (int i = 0; i < spans.Count; i++)
                {
                    int lo = spans[i].Start - Margin, hi = spans[i].End + Margin;
                    if (i > 0) lo = Math.Max(lo, spans[i - 1].End);
                    if (i + 1 < spans.Count) hi = Math.Min(hi, spans[i + 1].Start);
                    if (lo < 0) lo = 0;
                    if (hi > c.W) hi = c.W;
                    if (hi <= lo) continue;
                    double sum = 0;
                    for (int x = lo; x < hi; x++)
                    {
                        double v = (gray[gy, c.X + x] - bg) / amp;
                        sum += v > 1 ? 1 : (v < 0 ? 0 : v);
                    }
                    outp.Add(sum);
                }
            }
        }

        /// <summary>横杠厚度: 把 ¥ 的横杠那几行转置过来量, 竖着的一段就是横杠的厚。</summary>
        static double BarThickness(byte[,] gray, int[,] lab, Comp c, double bg, double fg,
                                   int rowFrom, int rowTo)
        {
            double amp = fg - bg;
            if (amp == 0 || rowTo <= rowFrom) return 0;
            int h = rowTo - rowFrom;
            var col = new bool[h];
            var outp = new List<double>();
            for (int x = 0; x < c.W; x++)
            {
                for (int r = 0; r < h; r++) col[r] = lab[c.Y + rowFrom + r, c.X + x] == c.Label;
                var spans = Spans(col);
                for (int i = 0; i < spans.Count; i++)
                {
                    int lo = spans[i].Start - Margin, hi = spans[i].End + Margin;
                    if (i > 0) lo = Math.Max(lo, spans[i - 1].End);
                    if (i + 1 < spans.Count) hi = Math.Min(hi, spans[i + 1].Start);
                    if (lo < 0) lo = 0;
                    if (hi > h) hi = h;
                    if (hi <= lo) continue;
                    double sum = 0;
                    for (int r = lo; r < hi; r++)
                    {
                        double v = (gray[c.Y + rowFrom + r, c.X + x] - bg) / amp;
                        sum += v > 1 ? 1 : (v < 0 ? 0 : v);
                    }
                    outp.Add(sum);
                }
            }
            return outp.Count == 0 ? 0 : Median(outp);
        }

        /// <summary>从裁块估底色和字色。取前景的亮端/暗端当字色, 避开被抗锯齿拉平的边缘像素。</summary>
        static bool Levels(byte[,] gray, byte[,] bin, int w, int h, out double bg, out double fg)
        {
            bg = fg = 0;
            var hFg = new int[256];
            var hBg = new int[256];
            long nFg = 0, nBg = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    if (bin[y, x] != 0) { hFg[gray[y, x]]++; nFg++; }
                    else { hBg[gray[y, x]]++; nBg++; }
                }
            if (nFg < 20 || nBg < 20) return false;
            double mFg = Percentile(hFg, nFg, 50), mBg = Percentile(hBg, nBg, 50);
            bg = mBg;
            fg = Percentile(hFg, nFg, mFg > mBg ? 90 : 10);
            return Math.Abs(fg - bg) >= 30;
        }

        static double Percentile(int[] hist, long total, double q)
        {
            long want = (long)Math.Ceiling(total * q / 100.0);
            if (want < 1) want = 1;
            long acc = 0;
            for (int v = 0; v < 256; v++)
            {
                acc += hist[v];
                if (acc >= want) return v;
            }
            return 255;
        }

        readonly struct Span
        {
            public readonly int Start, End;      // End 是开区间
            public Span(int s, int e) { Start = s; End = e; }
        }

        static List<Span> Spans(bool[] row)
        {
            var outp = new List<Span>();
            int s = -1;
            for (int i = 0; i <= row.Length; i++)
            {
                bool on = i < row.Length && row[i];
                if (on && s < 0) s = i;
                else if (!on && s >= 0) { outp.Add(new Span(s, i)); s = -1; }
            }
            return outp;
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

        // ★★★ 用整块拷贝, **不要**逐像素 GetGenericIndexer。
        //   分阶段量过(200 张, 中位 1200x2622): 蓝底闸 1.2ms, 定位 14.7ms,
        //   剩下的 21.0ms 几乎全耗在这两个数组上。换成 GetRectangularArray 之后
        //   整个 Check 中位 36.9 -> 21.7ms, p95 91.4 -> 38.6ms, 最大 482 -> 173ms。
        //   200 张逐位重对过, 判定结果一个字节都没变。
        static byte[,] ToBytes(Mat m)
        {
            m.GetRectangularArray(out byte[,] a);
            return a;
        }

        static int[,] ToInts(Mat m)
        {
            m.GetRectangularArray(out int[,] a);
            return a;
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
