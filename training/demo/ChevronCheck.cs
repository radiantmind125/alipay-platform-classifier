#nullable enable

using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace Ssp
{
    public enum ChevronVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // 返回箭头偏小且发虚
    }

    public sealed class ChevronResult
    {
        public ChevronVerdict Verdict;
        public int ArrowHeight;      // 返回箭头的高
        public int ArrowWidth;       // 返回箭头的宽
        public int ArrowTop;         // 箭头顶边在整图中的 y
        public double Blur;          // 半灰边缘像素 / 实心笔画像素
        public bool Measured;        // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 账单详情页左上角返回箭头的几何检查。
    ///
    /// iOS 的导航栏图标由系统按点数渲染, 所以苹果各机型不管屏幕分辨率多少,
    /// 这个箭头都是 56x33 像素, 边缘很利。实测五种苹果分辨率, 92%~94% 都是这个尺寸。
    /// 用固定尺寸素材拼出来的图, 箭头会偏小而且发虚 —— 发虚是因为它被缩放过,
    /// 原生渲染无论多大都是利的。
    ///
    /// 只对苹果机型有效。安卓各家 ROM 自己画导航栏, 同一分辨率下实测有九百多种不同的
    /// 箭头位图, 最常见的才占 7%, 没有可比的基准; 而且安卓的正常箭头尺寸落在
    /// 和异常箭头相同的区间里, 分不开。所以遇到非苹果分辨率一律不判。
    ///
    /// 只读入参, 内部全部用 ROI 视图, 不复制整图。无静态可变状态, 可多线程调用。
    /// </summary>
    public static class ChevronCheck
    {
        // 真图箭头的标准尺寸(苹果 @3x)
        public const int NormalHeight = 56;
        public const int NormalWidth = 33;

        // 判定阈值: 高和宽都低于标准的 92%, 且虚实比超过这个值
        public const double SizeFraction = 0.92;
        public const double BlurThreshold = 0.65;

        const int GrayDark = 170;    // 算"深色"的灰度阈
        const int CoreThr = 100;     // 低于此算实心笔画
        const int MidThr = 200;      // 介于 CoreThr 和此值之间算半灰边缘

        // 认作返回箭头的尺寸范围。真图 56x33, 异常的 47x28, 都在这个范围内;
        // 范围外的一律不当箭头, 免得在没有箭头的页面上挑中别的图标。
        const int MinPlausibleH = 40, MaxPlausibleH = 70;
        const int MinPlausibleW = 22, MaxPlausibleW = 42;

        /// <summary>
        /// 适用的分辨率。iOS 导航图标按点数渲染, 所以这些机型的箭头尺寸完全一致。
        /// 出现新机型时往这里加; 不在表里的一律返回 CannotDetermine, 不猜。
        /// </summary>
        static readonly HashSet<(int, int)> Supported = new()
        {
            (1179, 2556), (1290, 2796), (1170, 2532),
            (1320, 2868), (1206, 2622), (1284, 2778),
            (1125, 2436), (1242, 2688),
        };

        /// <param name="image">
        /// 8 位图, 1 / 3 / 4 通道均可; 多通道按 OpenCV 惯例视为 BGR(A)。
        /// 传进来的 Mat 不会被修改, 也不会被释放。
        /// </param>
        public static ChevronResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            var res = new ChevronResult { Verdict = ChevronVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U)
                throw new ArgumentException("只支持 8 位图");

            int cn = image.Channels();
            if (cn != 1 && cn != 3 && cn != 4)
                throw new ArgumentException($"不支持 {cn} 通道");

            int width = image.Width, height = image.Height;
            if (!Supported.Contains((width, height)))
            {
                res.Reason = $"{width}x{height} 不是已知的苹果机型分辨率, 不判";
                return res;
            }

            // 只看左上角一小块, 导航栏一定在这个范围内。ROI 是视图, 不复制。
            int cw = (int)(width * 0.13), ch = (int)(height * 0.12);
            if (cw < 16 || ch < 16) { res.Reason = "图太小"; return res; }

            using var corner = new Mat(image, new Rect(0, 0, cw, ch));
            using var gray = new Mat();
            if (cn == 1) corner.CopyTo(gray);
            else Cv2.CvtColor(corner, gray,
                cn == 4 ? ColorConversionCodes.BGRA2GRAY : ColorConversionCodes.BGR2GRAY);

            var arrow = LocateArrow(gray);
            if (arrow == null) { res.Reason = "找不到返回箭头"; return res; }
            var a = arrow.Value;

            // 虚实比: 半灰边缘像素数 / 实心笔画像素数。
            // 原生渲染的图标边缘干净, 这个比值小; 缩放过的会糊出一圈半灰, 比值明显变大。
            using var box = new Mat(gray, a);
            using var mask = new Mat();
            Cv2.InRange(box, 0, CoreThr - 1, mask);
            int core = Cv2.CountNonZero(mask);
            Cv2.InRange(box, CoreThr, MidThr - 1, mask);
            int mid = Cv2.CountNonZero(mask);
            if (core == 0) { res.Reason = "箭头没有实心像素"; return res; }

            res.Measured = true;
            res.ArrowHeight = a.Height;
            res.ArrowWidth = a.Width;
            res.ArrowTop = a.Top;
            res.Blur = (double)mid / core;

            bool small = a.Height < NormalHeight * SizeFraction && a.Width < NormalWidth * SizeFraction;
            bool fuzzy = res.Blur >= BlurThreshold;
            if (small && fuzzy)
            {
                res.Verdict = ChevronVerdict.Suspicious;
                res.Reason = $"箭头 {a.Height}x{a.Width} 小于常态 {NormalHeight}x{NormalWidth}, 虚实比 {res.Blur:F3}";
            }
            else
            {
                res.Verdict = ChevronVerdict.Ok;
                res.Reason = $"箭头 {a.Height}x{a.Width}, 虚实比 {res.Blur:F3}";
            }
            return res;
        }

        /// <summary>
        /// 找导航栏的返回箭头。用连通域找, 不用固定窗口 ——
        /// 状态栏高度各机型不一样, 固定窗口会切到箭头上或者整个错过。
        /// </summary>
        static Rect? LocateArrow(Mat gray)
        {
            using var dark = new Mat();
            // 判据是 gray < 170, 即 <= 169; BinaryInv 的判据是 src <= thresh
            Cv2.Threshold(gray, dark, GrayDark - 1, 255, ThresholdTypes.BinaryInv);

            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(dark, labels, stats, centroids,
                                                     PixelConnectivity.Connectivity8, MatType.CV_32S);

            Rect? best = null;
            for (int i = 1; i < n; i++)      // 0 是背景
            {
                int bw = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
                int bh = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
                int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                // 箭头: 竖着比横着长, 有一定墨量, 且尺寸落在合理范围内。
                // 尺寸下限很重要: 没有返回箭头的页面上, 不加下限会挑中别的小图标
                // (实测挑到 18x11、19x15 这种), 误报会从万分之 0.8 涨到万分之 6。
                if (bh < MinPlausibleH || bh > MaxPlausibleH) continue;
                if (bw < MinPlausibleW || bw > MaxPlausibleW) continue;
                if (bh < bw || area < 100) continue;
                int x = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
                int y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
                // 状态栏的图标也可能过筛, 取最靠下的那个 —— 导航栏在状态栏下面
                if (best == null || y > best.Value.Top)
                    best = new Rect(x, y, bw, bh);
            }
            return best;
        }
    }
}
