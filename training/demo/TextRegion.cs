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
        /// 判注音行时, 它相对下面那一行的高度上限。实测拼音行大约是 0.44 倍, 取 0.6 留余量。
        /// </summary>
        public const double AnnotationHeightRatio = 0.6;

        /// <summary>
        /// 判注音行时, 它和下面那一行的间距上限(相对下面那行的高)。
        /// </summary>
        public const double AnnotationGapRatio = 0.5;

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
                                  bool dropAnnotationRows = false)
        {
            var rows = FindTextRows(image, dropAnnotationRows);
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
        /// 按**一条线段**取: 给线段的两个端点, 取一条包住它的带。
        /// 线段是斜的也按它的竖直范围取, 再各加 <paramref name="padY"/> / <paramref name="padX"/> 像素。
        /// 两个端点顺序无所谓。
        /// ★ 竖直线段(两端点 x 相同)要传 <paramref name="padX"/>, 否则宽度是 0, 返回 null。
        /// ★ 返回 ROI 视图, 不是拷贝。
        /// </summary>
        public static Mat? ByLine(Mat image, Point a, Point b,
                                  int padY = 0, int yOffset = 0, int padX = 0)
        {
            if (image == null || image.Empty()) return null;
            int x0 = Math.Min(a.X, b.X) - padX, x1 = Math.Max(a.X, b.X) + padX;
            int y0 = Math.Min(a.Y, b.Y) - padY + yOffset;
            int y1 = Math.Max(a.Y, b.Y) + padY + yOffset;
            var r = Clamp(new Rect(x0, y0, x1 - x0, y1 - y0), image.Width, image.Height);
            if (r.Width <= 0 || r.Height <= 0) return null;
            return new Mat(image, r);
        }

        /// <summary>
        /// 自己做水平投影切出所有文字行, 从上到下。
        /// **不用 OCR 的行切分**, 所以拼音多插的那些细行不会把真文字行的位置搅乱。
        ///
        /// <paramref name="dropAnnotationRows"/> ——
        /// ★★ **默认 false, 不要随便打开。**
        ///   只有在**已经确认这一页带拼音**(用 <see cref="PinyinCheck.Check"/> 判过)
        ///   的时候才传 true。
        ///
        ///   为什么: "一行矮的压在一行高的上方, 而且贴着它"这个形状,
        ///   **拼音和正常版面是一样的** —— 标题压在金额上、字段名压在值上, 都是这个形状。
        ///   实测拿 61 个真注音配对和 74 个非注音配对比:
        ///       间距/自身高   真注音 中位 1.12   非注音 中位 1.59
        ///       高度比        真注音 0.44        非注音 0.43   ← 几乎一模一样
        ///   两组**完全重叠**, 单看一行根本分不开。卡任何一条线都是
        ///   "留住 43% 的注音, 同时误伤 3% 的正文"这种赔本买卖。
        ///
        ///   ★ <see cref="PinyinCheck"/> 之所以判得准, 是因为它在**整页**上数
        ///     "小块压在大块上方"的比例 —— 是**通篇每一行都这样**才叫拼音,
        ///     单独一处这样只是普通版面。所以该不该滤, 得先由整页判定说了算。
        ///
        ///   这一页确实带拼音时, 这个开关很有用: 实测 011.jpg 切出 30 行,
        ///   滤掉的 11 行全是拼音行, 正文行位置一点没动。
        ///
        /// 取字用的是"和局部底色差多少", 不是"够不够暗" —— 和 <see cref="PinyinCheck"/>
        /// 一致, 这样蓝底白字页(转账成功页)也切得出来。
        /// </summary>
        public static IReadOnlyList<Rect> FindTextRows(
            Mat image, bool dropAnnotationRows = false)
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
            if (raw.Count == 0 || !dropAnnotationRows) return raw;

            // ★★ 判一行是不是注音行, 只看它**和紧挨着的下一行**的关系, 不看整页。
            //
            //   注音行的定义就是局部的: 它比**它下面那一行**矮很多, 而且**贴着**它。
            //   一整页里有没有别的大块(卡片、大额数字、印章)和它无关。
            //
            //   ★ 我第一版用的是整页行高的 75 分位当基准, **错得很厉害**:
            //     一张回单上 收款人 / 收款人账号 / 转账时间 / 备注 都是 33~45 像素的正常行,
            //     但页上有个 280 像素的大块, 75 分位被顶到 102, 门槛算成 61,
            //     **四行正文全被当成注音行滤掉了** —— 恰恰是最该取的那几行。
            //     200 张随机真图里有 11 张被滤掉一半以上, 最狠的滤掉 75%。
            //     换成局部判断之后就没这个问题了。
            var kept = new List<Rect>(raw.Count);
            for (int i = 0; i < raw.Count; i++)
            {
                var cur = raw[i];
                bool annotation = false;
                if (i + 1 < raw.Count)
                {
                    var next = raw[i + 1];
                    int gap = next.Y - (cur.Y + cur.Height);
                    // 比下面那行矮得多, 而且紧贴着它
                    if (cur.Height <= AnnotationHeightRatio * next.Height &&
                        gap >= 0 && gap <= AnnotationGapRatio * next.Height)
                    {
                        annotation = true;
                    }
                }
                if (!annotation) kept.Add(cur);
            }
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
