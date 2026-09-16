#nullable enable

using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace Ssp
{
    /// <summary>
    /// 按**位置**处理图上的文字区域, 两件事:
    ///   **判断** —— <see cref="InRect"/> / <see cref="OnSegment"/>, 给一个文字框, 返回在不在(bool)
    ///   **取值** —— <see cref="ByRect(Mat, Rect, int)"/> / <see cref="ByLine"/> / <see cref="RowAt"/>,
    ///               按矩形或线段把那一块图取出来
    ///
    /// ★ 判和取用的是**同一套偏移口径**(yOffset 都是像素, 向上为负, 动的都是矩形/线段而不是文字),
    ///   所以"判为真"就等于"同样参数去取, 能把这块文字取进来"。两边不会打架。
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

        /// <summary>
        /// ★★★★★ 带拼音时, 文字框的**顶**被顶高了多少 —— 相对**正文字高**的倍数。
        ///
        /// 拼音把 OCR 的框撑大了: 框会把上面那行拼音一起圈进去, 顶就比正常位置高。
        /// 按位置判字段的规则要补这个差, 补多少就是这个数。
        ///
        /// <c>yOffset = -(int)Math.Round(AnnotationShiftRatio * 正文字高)</c>
        /// 或者直接用 <see cref="AnnotationOffset"/>。
        ///
        /// ★ 这个 0.46 是**量出来的**, 不是拍的。
        ///   2026-09-16 在 800 张带拼音的白底回单上量:
        /// <code>
        ///                   中位   25分位  75分位  95分位
        ///     拼音行高       9.0     8.0    10.0    11.0
        ///     拼音底到汉字顶 3.0     2.0     4.0     5.0
        ///     框顶上移量    12.0    10.0    13.0    16.0     <- 就是这个
        ///     正文字高      26.0    23.0    30.0    35.0
        /// </code>
        ///   12 / 26 = 0.46。对得上类注释里 011.jpg 那组(框 29 -> 44, 顶高 15 像素)。
        ///
        /// ★★ 给倍数而不是给像素, 是因为**不同机型分辨率不一样**, 像素数会跟着变;
        ///   除以正文字高之后就稳了。页高 2412 的图上, 0.46 x 26 就是 12 像素。
        ///
        /// ★★★ **这个偏移是个常数, 不随行号变。** 本来担心每行上面都加一行拼音,
        ///   下面的行会被累积推下去(那样第 N 行就得偏 N 倍, 一个数就不够用)。
        ///   实测行距(归一化到页高): 带拼音 0.04751, 不带 0.04829, 比值 0.98;
        ///   行距除以行高 3.04 对 3.03。**拼音是挤在原有行距里的, app 没另开空间**,
        ///   所以没有累积推移, 一个常数就够。
        /// </summary>
        public const double AnnotationShiftRatio = 0.46;

        /// <summary>
        /// 拼音把框顶高了多少(像素, **正数**)。<paramref name="bodyTextHeight"/> 传正文字高 ——
        /// <c>PinyinCheck.Check</c> 返回的 <c>PinyinResult.BigHeight</c> 就是这个数。
        ///
        /// ★ 这只是个量, 怎么用见 <see cref="ExpandForAnnotation"/>。
        /// ★ 高度不合理(小于等于 0 / 大得离谱 / NaN)时返回 0 —— 宁可不动, 也不要按瞎算的数动。
        /// </summary>
        public static int AnnotationShift(double bodyTextHeight)
        {
            if (double.IsNaN(bodyTextHeight) || bodyTextHeight <= 0) return 0;
            if (bodyTextHeight > 1000) return 0;          // 明显不是字高, 别硬算
            return (int)Math.Round(AnnotationShiftRatio * bodyTextHeight,
                                   MidpointRounding.AwayFromZero);
        }

        /// <summary>
        /// 带拼音时, 把字段格子的**顶往上撑开**, 好把被拼音顶高的文字框也罩进去。
        /// **底不动, 高度变大** —— 不是把整个格子挪上去。
        ///
        /// <code>
        ///     var pr = PinyinCheck.Check(image);
        ///     var box = pr.Verdict == PinyinVerdict.HasPinyin
        ///             ? TextRegion.ExpandForAnnotation(fieldRect, pr.BigHeight)
        ///             : fieldRect;                        // 普通图原样
        ///     bool hit = TextRegion.InRect(textBox, box);  // yOffset 保持 0
        /// </code>
        ///
        /// ★★★★★ **为什么是"撑开"不是"挪"** —— 2026-09-16 在 200 张真图
        ///   (2153 个带拼音的行)上量过, 三种做法:
        /// <code>
        ///     做法              通过率    覆盖比例中位
        ///     不动              89.1%      0.658
        ///     整个挪上去        85.8%      0.645     <- **比不动还差**
        ///     ★ 顶往上撑开      99.5%      0.971
        /// </code>
        ///   挪为什么更差: 字段格子本来就和汉字行对齐、整个落在合并框里面,
        ///   往上挪反而有一截跑到框外, 交集变小。
        ///   撑开才对: 合并框往上长出来的那一截, 正好被撑开的部分罩住。
        ///
        /// ★★ 我一开始实现成了"挪"(<c>AnnotationOffset</c>), 而且写了个测试说它管用 ——
        ///   那个测试用的是 011.jpg 的几个**合成矩形**, 格子比正文行低 10 像素,
        ///   那个错位是当初为了演示"中心法会翻"特意造的, 不是真实版面。
        ///   拿真图一量就露馅了。**合成用例过了不等于真数据上成立。**
        ///
        /// ★ 撑多少: 见 <see cref="AnnotationShiftRatio"/>, 0.46 倍正文字高(中位 12 像素)。
        /// </summary>
        public static Rect ExpandForAnnotation(Rect field, double bodyTextHeight)
        {
            int up = AnnotationShift(bodyTextHeight);
            if (up <= 0) return field;
            return new Rect(field.X, field.Y - up, field.Width, field.Height + up);
        }

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
                int d = SpanDistance(y, r.Y, r.Height);
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
        /// 判"文字在矩形内"时, 文字框至少要有多大比例落在矩形里。
        /// 兄弟类里同形状的判据都取 0.5(PinyinCheck 判水平压住、MinusCheck 按纵向重叠聚行),
        /// 这里是按字段格子定位, 该比那两处紧一点, 先取 0.6。
        /// ★ 这个数**还没拿真图标过**, 只是照兄弟类的量级开的口, 标完必须重定。
        ///   标法: 拿已经挑出来的 7205 张拼音图, 用真实字段矩形扫 0.3~0.9, 看订单号命中率的拐点。
        /// </summary>
        public const double CoverRatio = 0.6;

        /// <summary>
        /// 判断一块**文字**是不是落在给定的矩形里。
        ///
        /// <paramref name="text"/> 传文字的**外接框**: OCR 的 det 框(四点框先
        /// <c>Cv2.BoundingRect</c> 一下)、连通块框, 或 <see cref="FindTextRows"/> 切出来的行框。
        /// 不收单个点 —— 要对付的正是"框被切坏", 只给一个点就没有范围可判了。
        ///
        /// <paramref name="yOffset"/> 偏移的是**矩形**, 单位像素, 可正可负, **向上是负数**,
        /// 和 <see cref="ByRect(Mat, Rect, int)"/> 的同名参数逐字一致。
        /// 所以这条判为真, 就等于同样参数的 ByRect 会把这块文字裁进去 —— 判和取不会打架。
        ///
        /// 判据是"**框有多少落在里面**": 交集面积 / 文字框面积 >= <paramref name="coverRatio"/>。
        /// ★ 不用"整个框都在内": 带拼音的图上, 框被劈成两段、或者和上面那行拼在一起本来就是常态;
        ///   卡"完全在内"的话, 最该捞回来的那几个恰好全判否。
        /// ★ 不用"中心在内": 中心对**竖直方向**最敏感, 而竖直正是拼音搞坏的那个方向。
        ///   011.jpg 实测拼音行 453-467、正文行 468-496(见类注释); 两行一旦被并成一个框,
        ///   中心就往上跑 7 像素 —— 分界线只要落在这 7 像素里, 中心法就**判到上一个字段去了**,
        ///   而且是**言之凿凿地判错**, 不是判不出。
        ///
        /// ★★ **但要说清楚: 面积法并不能把并行的情况救回来。**
        ///   实测(见 trtest D 组): 拼音行并进正文行之后, 框从 29 像素长到 44 像素,
        ///   分母跟着变大, 落在目标格子里的比例掉到 **0.432** —— 低于默认的 0.6, 照样判否。
        ///   面积法真正买到的只有一样: 它**不会判到错的格子上去**(上一格也是否),
        ///   宁可答"不知道"也不答错。对误杀要赔钱的场景, 这个取舍是对的。
        ///
        ///   想把并行的那条认出来, **不要去放低阈值** —— 实测要放到 0.43 以下才认得出,
        ///   那时候"半个框在别的格子里"也算中, 相邻字段就分不开了。
        ///   **正确的办法是先把拼音行滤掉**:
        ///     1. <c>PinyinCheck.Check</c> 判这一页带不带拼音
        ///     2. 带的话 <see cref="FindTextRows"/> 传 <c>dropAnnotationRows: true</c>
        ///     3. 滤完再用默认 0.6 判 —— 实测就正常了(trtest D5/D6)
        ///
        /// ★★ <see cref="FindTextRows"/> 返回的行框是**整幅宽**的(x=0, w=图宽),
        ///   拿它判"在不在某个字段的小矩形里"永远过不了。行框要么先按 x 收窄,
        ///   要么改用 <see cref="OnSegment"/> —— 按行判是那个方法的活。
        ///
        /// 纯几何, 不看图, 不分配内存, 可多线程调用。
        /// </summary>
        public static bool InRect(Rect text, Rect region, int yOffset = 0,
                                  double coverRatio = CoverRatio)
        {
            long area = (long)text.Width * text.Height;
            if (area <= 0) return false;                    // 空框一律判否, 不引入第三态
            var moved = region + new Point(0, yOffset);     // 和 ByRect 一样: 动的是矩形
            var hit = Rect.Intersect(text, moved);
            return (long)hit.Width * hit.Height >= coverRatio * area;
        }

        /// <summary>
        /// 判断一块**文字**是不是压在给定的**线段**上。
        ///
        /// <paramref name="a"/> / <paramref name="b"/> 是线段两个端点, 顺序无所谓,
        /// 和 <see cref="ByLine"/> 收的是同一对点。
        /// <paramref name="yOffset"/> 偏移的是**线段**(像素, **向上是负数**), 也和 <see cref="ByLine"/>
        /// 一致 —— 判为真就等于同样参数的 ByLine 会把这块文字带出来。
        /// <paramref name="tol"/> 是容差(像素): 差这么一点没挨上, 也算压上。
        ///
        /// 判据是"**线段到文字框的距离** &lt;= tol"(穿过去时距离为 0), 不是"框的中心离线段多远"。
        /// ★ 为什么不按中心: 011.jpg 实测正文行高 27~28 像素, 紧贴在上面的拼音行 11~14 像素。
        ///   想靠中心距离把相邻两行分开, 容差就得小于半个行高; 可拼音一并进来,
        ///   框的中心自己先往上跑 7.5 像素(算式见 <see cref="InRect"/>) —— 该中的那行反倒判否。
        ///   按"线段有没有从框里穿过去"就没这个毛病: 框长高了, 原来那条基线仍然在框内。
        /// ★ 用的是点到**线段**的真实距离, 不假设线是水平的, 斜线竖线一样判
        ///   (这点和 <see cref="RowAt"/> 不同, 那个收的是一条水平线的 y)。
        ///
        /// <paramref name="tol"/> 默认 0 = 必须真碰上, 和 <see cref="ByLine"/> 的 padY / padX
        /// 默认 0 是同一个口径。要留容差的话, 大约取四分之一行高(实测行高 27~28, 即 7 上下)。
        /// ★ 这个 7 同样**没标定过**。
        ///
        /// 纯几何, 不看图, 不分配内存, 可多线程调用。
        /// </summary>
        public static bool OnSegment(Rect text, Point a, Point b, int yOffset = 0, int tol = 0)
        {
            if (text.Width <= 0 || text.Height <= 0) return false;
            var box = Rect.Inflate(text, tol, tol);         // 容差做成把框放大, 全整数, 不算浮点距离
            var p = new Point(a.X, a.Y + yOffset);          // 和 ByLine 一样: 动的是线段
            var q = new Point(b.X, b.Y + yOffset);
            return SegmentHitsRect(p, q, box);
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

        // 一维: 点 v 到区间 [lo, lo+len] 的距离, 在区间内为 0
        static int SpanDistance(int v, int lo, int len)
            => v < lo ? lo - v : (v > lo + len ? v - (lo + len) : 0);

        // 线段和矩形有没有交: 端点落在框内, 或线段和四条边中任意一条相交
        static bool SegmentHitsRect(Point p, Point q, Rect r)
        {
            if (r.Width <= 0 || r.Height <= 0) return false;
            if (r.Contains(p) || r.Contains(q)) return true;    // 退化成一个点时也走这里
            int x0 = r.X, y0 = r.Y, x1 = r.X + r.Width, y1 = r.Y + r.Height;
            var tl = new Point(x0, y0); var tr = new Point(x1, y0);
            var br = new Point(x1, y1); var bl = new Point(x0, y1);
            return SegHitsSeg(p, q, tl, tr) || SegHitsSeg(p, q, tr, br)
                || SegHitsSeg(p, q, br, bl) || SegHitsSeg(p, q, bl, tl);
        }

        // 叉积符号法, 含共线重叠。全整数, 用 long 防溢出
        static bool SegHitsSeg(Point a, Point b, Point c, Point d)
        {
            long d1 = Cross(c, d, a), d2 = Cross(c, d, b);
            long d3 = Cross(a, b, c), d4 = Cross(a, b, d);
            if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
                ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
            if (d1 == 0 && InSpan(c, d, a)) return true;
            if (d2 == 0 && InSpan(c, d, b)) return true;
            if (d3 == 0 && InSpan(a, b, c)) return true;
            if (d4 == 0 && InSpan(a, b, d)) return true;
            return false;
        }

        static long Cross(Point o, Point p, Point q)
            => (long)(p.X - o.X) * (q.Y - o.Y) - (long)(p.Y - o.Y) * (q.X - o.X);

        static bool InSpan(Point a, Point b, Point p)
            => Math.Min(a.X, b.X) <= p.X && p.X <= Math.Max(a.X, b.X)
            && Math.Min(a.Y, b.Y) <= p.Y && p.Y <= Math.Max(a.Y, b.Y);

        static Rect Clamp(Rect r, int w, int h)
        {
            int x0 = Math.Max(0, r.X), y0 = Math.Max(0, r.Y);
            int x1 = Math.Min(w, r.X + r.Width), y1 = Math.Min(h, r.Y + r.Height);
            return new Rect(x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0));
        }
    }
}
