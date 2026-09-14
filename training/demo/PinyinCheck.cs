#nullable enable

using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace Ssp
{
    public enum PinyinVerdict
    {
        CannotDetermine = 0,   // 量不了, 走正常流程
        NoPinyin = 1,
        HasPinyin = 2,         // 带拼音标注, OCR 的行切分会乱
    }

    public sealed class PinyinResult
    {
        public PinyinVerdict Verdict;
        public double Ratio;         // 压住的小块数 / 小块总数
        public int Stacked;          // 有多少小块正下方紧贴着一个大块
        public int SmallCount;       // 小块总数
        public int BigCount;
        public double BigHeight;     // 正文字高(块高的 75 分位)
        public double Flat;          // 出现最多的那个灰度值占了多少像素 —— 用来分截图和照片
        public bool Measured;        // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 判断截图里是不是带**拼音标注**(安卓的无障碍功能: 每个汉字正上方多一行小字拼音)。
    ///
    /// 为什么要判
    /// ----------
    /// 带拼音的图会让 OCR 的**行切分**出错: 每两行真文字之间多插了一行小字,
    /// 订单号要么和上一行拼接、要么被劈成两段, 于是读不出来,
    /// 系统按"订单号不符合规则"直接拒单 —— 真交易被误拒。
    /// ★ 注意不是字体变形, 字本身是正常的, 坏的是**版面**。
    ///
    /// 怎么判
    /// ------
    /// 不做 OCR, 只看版面几何。拼音的特征很干净:
    /// 块高呈**双峰**(小块约 8 像素是拼音字母, 大块约 21 像素是汉字),
    /// 并且每个拼音块的**正下方紧贴一个汉字块**, 水平方向还重叠。
    /// 普通截图里也有小字(时间戳、说明文字), 但不会系统性地压在大字正上方。
    ///
    /// ★★ 取字看的是"**和局部底色差多少**", 不是"够不够暗"。
    ///   用全局阈值会两头出错, 因为一个页面上既有蓝底白字又有白卡片黑字:
    ///       蓝底(灰度约 120)比阈值暗 -> **背景**被当成文字, 被白字切成一堆碎片,
    ///                                    碎片凑出假的"小压大" -> 误判
    ///       白字(灰度 255)比阈值亮   -> 真的文字取不到           -> 漏判
    ///   两个方向的错来自**同一个**原因, 所以只补一头是补不好的(试过, 更糟)。
    ///
    /// 实测(本地 262,748 张, 人工核对过)
    /// ---------------------------------
    ///   命中 1,740 张 = 0.663%, 大约每 150 张有 1 张
    ///   随机抽 102 张人工看, 102 张都确实带拼音
    ///   145 张人工看过的真图, 0 误判
    ///   漏判约一成
    ///
    /// 入参只读, 不碰 System.Drawing, 无静态可变状态, 可多线程调用。
    /// 单张 1080x2412 约 30 毫秒; 4000x3000 的翻拍照片要贵得多(估底色跟面积走)。
    /// </summary>
    public static class PinyinCheck
    {
        /// <summary>
        /// 判为带拼音的比例下界。
        /// ★ 这条线是**跟着取字方式走的** —— 换了取字方式必须重新定。
        ///   我第一次就是把旧方法的 0.25 原样搬过来, 结果贴线那段 76% 是误判。
        ///   379 格人工标注里, 所有误判的比例都 &lt;= 0.340, 真命中的中位数是 0.578,
        ///   所以取 0.35。0.32~0.40 之间命中数变化都在 4% 以内, 线落在平台上。
        /// </summary>
        public const double Threshold = 0.35;

        /// <summary>
        /// 压住数的下界。文字少的页面比例会抖(分母小), 光看比例分不开,
        /// 实测两张确实带拼音的只有 0.294 和 0.299, 而不带拼音的红包弹窗页是 0.256;
        /// 但绝对数差得很清楚(20 和 79 对 10)。所以比例和绝对数两个都要过。
        /// </summary>
        public const int MinStacked = 15;

        /// <summary>
        /// 平坦占比的下界 —— 用来把**不是截图**的图挡掉。
        /// 截图有大片灰度值完全相同的底色; 照片因为传感器噪声做不到。
        /// 实测照片 0.009~0.050, 截图 0.312~0.880, 中间是空的。
        /// ★ 别改用长宽比来挡: 有张真带拼音的截图是 1200x1920, 按长宽比会误杀。
        /// </summary>
        public const double MinFlat = 0.15;

        const int DiffThreshold = 28;      // 和局部底色差多少才算文字
        const int MinArea = 8;             // 太小的连通块是噪点
        const int MinComponents = 60;      // 块太少说明这张图没什么文字, 判不了
        const int BucketWidth = 50;        // 按 x 建桶, 免得配对成 O(N^2)

        /// <summary>
        /// 上限保护, 只挡真正病态的输入(整幅噪声之类)。
        /// ★ 这个数一开始按 MinusCheck 的量级设成 6 万, **设小了**: 一张 3024x4032 的
        ///   翻拍图连通块就有 88,459 个, 会被直接判成"判不了", 而 Python 版没有这条,
        ///   照样算得出来 —— 两边结果就对不上了。逐张对比时抓到的。
        ///   30 万对正常图片留了 3 倍以上余量。
        /// </summary>
        public const int MaxComponents = 300_000;

        struct Comp
        {
            public int X, Y, W, H;
        }

        /// <summary>
        /// 判断这张图是不是带拼音标注。入参可以是 BGR 或单通道灰度, 只读。
        /// </summary>
        public static PinyinResult Check(Mat image)
        {
            var res = new PinyinResult();
            if (image == null || image.Empty())
            {
                res.Reason = "空图";
                return res;
            }
            if (image.Rows < 200 || image.Cols < 200)
            {
                res.Reason = "图太小, 不判";
                return res;
            }

            int H = image.Rows, W = image.Cols;

            // ---- 转灰度 ----
            // 16 位图先降到 8 位; ConvertTo 是四舍五入, 和 IMREAD_COLOR 的截断有一两个灰阶的差,
            // 对"和底色差 28"这个判据没有影响。
            using var gray = new Mat();
            using var tmp = new Mat();
            Mat src = image;
            if (image.Depth() != MatType.CV_8U)
            {
                double scale = image.Depth() == MatType.CV_16U ? 1.0 / 256.0 : 1.0;
                image.ConvertTo(tmp, MatType.CV_8U, scale);
                src = tmp;
            }
            if (src.Channels() == 1)
            {
                src.CopyTo(gray);
            }
            else if (src.Channels() == 3)
            {
                Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
            }
            else if (src.Channels() == 4)
            {
                Cv2.CvtColor(src, gray, ColorConversionCodes.BGRA2GRAY);
            }
            else
            {
                res.Reason = $"通道数 {src.Channels()} 不支持";
                return res;
            }

            // ---- 平坦占比: 出现最多的那个灰度值占了多少像素 ----
            // 缩图必须用 Nearest, 插值会把"数值完全一样"抹掉。
            res.Flat = FlatFraction(gray);

            // ---- 取文字掩膜: 和局部底色的绝对差 ----
            using var mask = TextMask(gray);

            // ---- 连通块 ----
            var comps = new List<Comp>();
            using (var labels = new Mat())
            using (var stats = new Mat())
            using (var centroids = new Mat())
            {
                int n = Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids,
                                                         PixelConnectivity.Connectivity8);
                if (n > MaxComponents)
                {
                    res.Reason = $"连通块过多 ({n}), 不判";
                    return res;
                }
                double hLimit = 0.05 * H, wLimit = 0.4 * W;
                for (int i = 1; i < n; i++)
                {
                    int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                    if (area < MinArea) continue;
                    int ch = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
                    if (ch >= hLimit) continue;
                    int cw = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
                    if (cw >= wLimit) continue;
                    comps.Add(new Comp
                    {
                        X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                        Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                        W = cw,
                        H = ch,
                    });
                }
            }

            if (comps.Count < MinComponents)
            {
                res.Reason = $"文字块太少 ({comps.Count}), 判不了";
                return res;
            }

            // ---- 正文字高 = 块高的 75 分位 ----
            var heights = new int[comps.Count];
            for (int i = 0; i < comps.Count; i++) heights[i] = comps[i].H;
            Array.Sort(heights);
            double bigH = heights[Math.Min(heights.Length - 1, (int)(heights.Length * 0.75))];
            if (bigH < 8)
            {
                res.Reason = "正文字太小, 判不了";
                return res;
            }
            res.BigHeight = bigH;

            var small = new List<Comp>();
            var big = new List<Comp>();
            foreach (var c in comps)
            {
                if (c.H <= 0.55 * bigH) small.Add(c);
                if (c.H >= 0.80 * bigH) big.Add(c);
            }
            res.SmallCount = small.Count;
            res.BigCount = big.Count;

            if (small.Count < 20 || big.Count < 20)
            {
                res.Measured = true;
                res.Verdict = PinyinVerdict.NoPinyin;
                res.Reason = "大小两档的块都不够, 不像有拼音";
                return res;
            }

            // ---- 数"小块正下方紧贴一个大块且水平压住" ----
            var buckets = new Dictionary<int, List<Comp>>();
            foreach (var b in big)
            {
                int k0 = b.X / BucketWidth, k1 = (b.X + b.W) / BucketWidth;
                for (int k = k0; k <= k1; k++)
                {
                    if (!buckets.TryGetValue(k, out var list))
                    {
                        list = new List<Comp>();
                        buckets[k] = list;
                    }
                    list.Add(b);
                }
            }

            int stacked = 0;
            foreach (var s in small)
            {
                if (IsStacked(s, buckets)) stacked++;
            }

            res.Stacked = stacked;
            res.Ratio = stacked / (double)small.Count;
            res.Measured = true;

            if (res.Flat < MinFlat)
            {
                res.Verdict = PinyinVerdict.CannotDetermine;
                res.Reason = $"不像截图(平坦占比 {res.Flat:F3}), 多半是翻拍或别的照片, 不判";
                return res;
            }

            res.Verdict = (res.Ratio >= Threshold && stacked >= MinStacked)
                ? PinyinVerdict.HasPinyin
                : PinyinVerdict.NoPinyin;
            res.Reason = res.Verdict == PinyinVerdict.HasPinyin
                ? $"带拼音 (比例 {res.Ratio:F3}, 压住 {stacked})"
                : $"没有拼音 (比例 {res.Ratio:F3}, 压住 {stacked})";
            return res;
        }

        /// <summary>
        /// 取文字掩膜: 看每个像素**和周围底色差多少**, 而不是它本身够不够暗。
        /// 文字不管是深底浅字还是浅底深字, 和底色的差都大;
        /// 平整的背景和自己的底色差约等于 0, 根本不进掩膜 —— 背景碎片这一类从源头没了。
        /// 缩到 1/4 再取中值当底色: 中值不会被细笔画带跑, 缩小是为了快。
        /// </summary>
        static Mat TextMask(Mat gray, int diffThreshold = DiffThreshold)
        {
            int sw = Math.Max(1, gray.Cols / 4), sh = Math.Max(1, gray.Rows / 4);
            using var small = new Mat();
            Cv2.Resize(gray, small, new Size(sw, sh), 0, 0, InterpolationFlags.Area);
            // MedianBlur 的核必须是奇数; 小图上核不能超过边长
            int k = Math.Min(21, Math.Min(sw, sh));
            if (k % 2 == 0) k--;
            if (k >= 3) Cv2.MedianBlur(small, small, k);

            using var bg = new Mat();
            Cv2.Resize(small, bg, new Size(gray.Cols, gray.Rows), 0, 0, InterpolationFlags.Linear);
            using var diff = new Mat();
            Cv2.Absdiff(gray, bg, diff);
            var mask = new Mat();
            Cv2.Threshold(diff, mask, diffThreshold, 255, ThresholdTypes.Binary);
            return mask;
        }

        /// <summary>
        /// 出现次数最多的那个灰度值占了多少像素。截图的底色是大片同值的, 照片做不到。
        /// </summary>
        static double FlatFraction(Mat gray)
        {
            Mat g = gray;
            Mat? shrunk = null;
            try
            {
                int longest = Math.Max(gray.Rows, gray.Cols);
                if (longest > 1400)
                {
                    double sc = 1400.0 / longest;
                    shrunk = new Mat();
                    // ★ 必须用 Nearest: 任何插值都会把"数值完全相同"抹掉, 这个量就废了
                    Cv2.Resize(gray, shrunk, new Size(Math.Max(1, (int)(gray.Cols * sc)),
                                                      Math.Max(1, (int)(gray.Rows * sc))),
                               0, 0, InterpolationFlags.Nearest);
                    g = shrunk;
                }
                using var hist = new Mat();
                Cv2.CalcHist(new[] { g }, new[] { 0 }, null, hist, 1,
                             new[] { 256 }, new[] { new Rangef(0, 256) });
                Cv2.MinMaxLoc(hist, out _, out double maxVal);
                long total = (long)g.Rows * g.Cols;
                return total > 0 ? maxVal / total : 0.0;
            }
            finally
            {
                shrunk?.Dispose();
            }
        }

        /// <summary>
        /// 这个小块的正下方是不是紧贴着一个大块, 并且水平方向压住了它。
        /// </summary>
        static bool IsStacked(Comp s, Dictionary<int, List<Comp>> buckets)
        {
            int k0 = s.X / BucketWidth, k1 = (s.X + s.W) / BucketWidth;
            for (int k = k0; k <= k1; k++)
            {
                if (!buckets.TryGetValue(k, out var list)) continue;
                foreach (var b in list)
                {
                    int gap = b.Y - (s.Y + s.H);
                    if (gap < -2 || gap > 0.7 * b.H) continue;      // 必须紧贴在上方
                    int ov = Math.Min(s.X + s.W, b.X + b.W) - Math.Max(s.X, b.X);
                    if (ov > 0.5 * Math.Min(s.W, b.W)) return true; // 水平要压住
                }
            }
            return false;
        }
    }
}
