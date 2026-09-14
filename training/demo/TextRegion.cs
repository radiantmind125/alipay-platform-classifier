#nullable enable

using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace Ssp
{
    /// <summary>
    /// 按**位置**取图上的文字区域 —— 按矩形取, 或者按一条水平线取那一行。
    ///
    /// 为什么要这个
    /// ------------
    /// 带**拼音标注**的截图(安卓无障碍功能: 每个汉字正上方多一行小字)会把 OCR 的
    /// **行切分**搞乱: 每两行真文字之间多插了一行小字, 订单号要么和上一行拼接、
    /// 要么被劈成两段, 于是读不出来, 系统按"订单号不符合规则"直接拒单。
    /// ★ 注意坏的不是字形(字本身是正常的), 是**版面**。
    ///
    /// 按位置取值**不依赖行切分**, 所以天然免疫这个问题。
    /// 而且 <see cref="FindTextRows"/> 是自己做投影切行的, 可以按行高把拼音那种细行滤掉。
    ///
    /// 实测(经理给的 011.jpg, 1080x2412):
    ///     拼音行   11~14 像素
    ///     真文字行 27~28 像素
    ///   两者在水平投影上是**分开的两行**(453-467 拼音, 468-496 真文字),
    ///   所以按行高一滤就只剩真文字行, 位置也就稳了。
    ///
    /// ★ 关于取出来的 Mat
    /// -------------------
    /// <see cref="ByRect(Mat, Rect, int)"/> 返回的是**ROI 视图, 不是拷贝** ——
    /// 它和原图共用同一块内存(经理 2026-09-09: "别用内存拷贝")。
    /// 所以: **原图还活着的时候才能用它**, 原图 Dispose 了视图就悬空。
    /// 需要独立副本的话调用方自己 <c>.Clone()</c>。
    ///
    /// 不碰 System.Drawing, 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class TextRegion
    {
        /// <summary>
        /// 行高低于"正文行高 × 这个比例"的行会被当成注音/装饰行滤掉。
        /// 实测拼音行大约是正文行高的 0.4 倍, 取 0.6 留了余量。
        /// 传 0 表示不滤, 什么行都要。
        /// </summary>
        public const double DefaultMinRowHeightRatio = 0.6;

        const int DiffThreshold = 28;    // 和局部底色差多少才算文字, 和 PinyinCheck 一致

        /// <summary>
        /// 按**绝对像素**矩形取一块。<paramref name="yOffset"/> 是竖直偏移(像素, 可正可负)。
        /// 越界会自动夹到图内; 夹完如果是空的, 返回 null。
        /// ★ 返回的是 ROI 视图, 不是拷贝, 见类注释。
        /// </summary>
        public static Mat? ByRect(Mat image, Rect rect, int yOffset = 0)
        {
            if (image == null || image.Empty()) return null;
            var r = Clamp(new Rect(rect.X, rect.Y + yOffset, rect.Width, rect.Height),
                          image.Width, image.Height);
            if (r.Width <= 0 || r.Height <= 0) return null;
            return new Mat(image, r);
        }

        /// <summary>
        /// 按**相对坐标**(0~1)矩形取一块, 换分辨率不用改参数。
        /// <paramref name="yOffsetRel"/> 也是相对图高的偏移(可正可负)。
        ///
        /// ★ 什么时候**不能**用相对坐标: 手机的状态栏/导航栏在不同机型上是
        ///   **固定像素高度, 不是固定比例**。整页按比例取的话, 状态栏一高一矮
        ///   下面的内容就整体上下飘。要取靠近页面顶部的东西时,
        ///   宁可先用 <see cref="FindTextRows"/> 定位, 再按行取。
        /// </summary>
        public static Mat? ByRect(Mat image, double xRel, double yRel,
                                  double wRel, double hRel, double yOffsetRel = 0)
        {
            if (image == null || image.Empty()) return null;
            int W = image.Width, H = image.Height;
            var r = new Rect((int)Math.Round(xRel * W),
                             (int)Math.Round((yRel + yOffsetRel) * H),
                             (int)Math.Round(wRel * W),
                             (int)Math.Round(hRel * H));
            r = Clamp(r, W, H);
            if (r.Width <= 0 || r.Height <= 0) return null;
            return new Mat(image, r);
        }

        /// <summary>
        /// 按**一条水平线**取: 给一个竖直位置(相对图高 0~1), 返回这条线穿过的那一行文字的框。
        /// 线落在两行之间时取**最近的一行**。找不到任何文字行返回 null。
        /// <paramref name="yOffsetRel"/> 在找行之前先把线上下挪一挪。
        /// </summary>
        public static Rect? RowAt(Mat image, double yRel, double yOffsetRel = 0,
                                  double minRowHeightRatio = DefaultMinRowHeightRatio)
        {
            var rows = FindTextRows(image, minRowHeightRatio);
            if (rows.Count == 0) return null;
            int y = (int)Math.Round((yRel + yOffsetRel) * image.Height);

            Rect best = rows[0];
            int bestDist = int.MaxValue;
            foreach (var r in rows)
            {
                // 线在行内 -> 距离 0; 否则算到行边的距离
                int d = y < r.Y ? r.Y - y : (y > r.Y + r.Height ? y - (r.Y + r.Height) : 0);
                if (d < bestDist) { bestDist = d; best = r; }
                if (d == 0) break;
            }
            return best;
        }

        /// <summary>
        /// 按**一条线段**取: 给线段的两个端点, 取一条包住它的水平带。
        /// 线段是斜的也按它的竖直范围取(再各加 <paramref name="padY"/> 像素)。
        /// ★ 返回 ROI 视图, 不是拷贝。
        /// </summary>
        public static Mat? ByLine(Mat image, Point a, Point b, int padY = 0, int yOffset = 0)
        {
            if (image == null || image.Empty()) return null;
            int x0 = Math.Min(a.X, b.X), x1 = Math.Max(a.X, b.X);
            int y0 = Math.Min(a.Y, b.Y) - padY + yOffset;
            int y1 = Math.Max(a.Y, b.Y) + padY + yOffset;
            var r = Clamp(new Rect(x0, y0, x1 - x0, y1 - y0), image.Width, image.Height);
            if (r.Width <= 0 || r.Height <= 0) return null;
            return new Mat(image, r);
        }

        /// <summary>
        /// 自己做水平投影切出所有文字行, 从上到下。
        ///
        /// ★ 这是这个类的关键: **不用 OCR 的行切分**, 所以拼音多插的那些细行
        ///   不会把真文字行搅乱。切完之后按行高把细行滤掉(见
        ///   <see cref="DefaultMinRowHeightRatio"/>), 剩下的就是正文行。
        ///
        /// 取字用的是"和局部底色差多少", 不是"够不够暗" —— 和 <see cref="PinyinCheck"/>
        /// 一致, 这样蓝底白字页(转账成功页)也切得出来。
        /// </summary>
        public static IReadOnlyList<Rect> FindTextRows(
            Mat image, double minRowHeightRatio = DefaultMinRowHeightRatio)
        {
            var empty = Array.Empty<Rect>();
            if (image == null || image.Empty()) return empty;
            if (image.Rows < 8 || image.Cols < 8) return empty;

            using var gray = new Mat();
            using var tmp = new Mat();
            Mat src = image;
            if (image.Depth() == MatType.CV_16U || image.Depth() == MatType.CV_16S)
            {
                image.ConvertTo(tmp, MatType.CV_8U, 1.0 / 256.0);
                src = tmp;
            }
            else if (image.Depth() != MatType.CV_8U)
            {
                return empty;    // 浮点图不猜, 见 PinyinCheck 里同样的理由
            }
            if (src.Channels() == 1) src.CopyTo(gray);
            else if (src.Channels() == 3) Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
            else if (src.Channels() == 4) Cv2.CvtColor(src, gray, ColorConversionCodes.BGRA2GRAY);
            else return empty;

            int H = gray.Rows, W = gray.Cols;

            // 和局部底色的绝对差 -> 文字掩膜
            using var small = new Mat();
            Cv2.Resize(gray, small, new Size(Math.Max(1, W / 4), Math.Max(1, H / 4)),
                       0, 0, InterpolationFlags.Area);
            int k = Math.Min(21, Math.Min(small.Width, small.Height));
            if (k % 2 == 0) k--;
            if (k >= 3) Cv2.MedianBlur(small, small, k);
            using var bg = new Mat();
            Cv2.Resize(small, bg, new Size(W, H), 0, 0, InterpolationFlags.Linear);
            using var diff = new Mat();
            Cv2.Absdiff(gray, bg, diff);
            using var mask = new Mat();
            Cv2.Threshold(diff, mask, DiffThreshold, 1, ThresholdTypes.Binary);

            // 每行的前景像素数。
            // ★ 这里必须用 ReduceDimension.Column: 它把矩阵压成**一列**(H 行 x 1 列),
            //   也就是"每个图像行一个数"。
            //   我一开始写的是 ReduceDimension.Row —— 那是压成**一行**(1 x W),
            //   每个数对应一**列**; 再按 At<int>(y, 0) 去读 y 直到图高, 就读到缓冲区外面,
            //   直接 AccessViolationException **硬崩**(比给个错答案严重得多)。
            using var proj = new Mat();
            Cv2.Reduce(mask, proj, ReduceDimension.Column, ReduceTypes.Sum, MatType.CV_32S);
            if (proj.Rows != H || proj.Cols != 1) return empty;   // 维度不对就别往下读

            // 一次性拷进托管数组再遍历: 比逐个 At<T>() 快, 也不会踩到指针
            var ink = new int[H];
            proj.GetArray(out int[] projArr);
            if (projArr == null || projArr.Length < H) return empty;
            Array.Copy(projArr, ink, H);

            int minInk = Math.Max(3, (int)(W * 0.003));   // 挡掉零星噪点
            var raw = new List<Rect>();
            int y = 0;
            while (y < H)
            {
                if (ink[y] <= minInk) { y++; continue; }
                int y0 = y;
                while (y < H && ink[y] > minInk) y++;
                raw.Add(new Rect(0, y0, W, y - y0));
            }
            if (raw.Count == 0 || minRowHeightRatio <= 0) return raw;

            // ★ 按行高滤掉注音/装饰那种细行。
            //   基准用**75 分位**而不是中位数: 带拼音的图上细行和正文行数量相当,
            //   中位数会被细行拖下来, 滤不干净。
            var heights = new List<int>(raw.Count);
            foreach (var r in raw) heights.Add(r.Height);
            heights.Sort();
            double p75 = heights[Math.Min(heights.Count - 1, (int)(heights.Count * 0.75))];
            double minH = p75 * minRowHeightRatio;

            var kept = new List<Rect>(raw.Count);
            foreach (var r in raw)
                if (r.Height >= minH) kept.Add(r);
            return kept.Count > 0 ? kept : raw;    // 全被滤光就退回不滤, 免得一行都取不到
        }

        static Rect Clamp(Rect r, int w, int h)
        {
            int x0 = Math.Max(0, r.X), y0 = Math.Max(0, r.Y);
            int x1 = Math.Min(w, r.X + r.Width), y1 = Math.Min(h, r.Y + r.Height);
            return new Rect(x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0));
        }
    }
}
