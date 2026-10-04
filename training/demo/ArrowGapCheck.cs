#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum ArrowGapVerdict
    {
        CannotDetermine = 0,   // 判不了(没有账单管理那几行、图太小、贴线、拼音页), 走正常流程
        Ok = 1,                // 真图
        Suspicious = 2,        // 假图: 箭头离文字太近(而且箭头大一号)
    }

    public sealed class ArrowGapResult
    {
        public ArrowGapVerdict Verdict;
        public bool Measured;        // 找到了账单管理那一列(>= 2 行), 下面几项才有意义
        public double Ratio;         // 间距 / 箭头高, 判定就看它
        public double Gap;           // 文字右缘到箭头左缘的间距(各行中位数), 像素
        public double ArrowHeight;   // 箭头高(各行中位数), 像素
        public double ArrowWidth;
        public int ArrowGray;        // 箭头最深的灰度(各行中位数)
        public int RowCount;         // 量到的行数
        public int Margin;           // 箭头离页面右边的距离
        public int[] RowTops = Array.Empty<int>();
        public bool PinyinChecked;   // 跑过拼音检查(只在要判可疑时才跑)
        public bool Pinyin;
        public string Reason = "";
    }

    /// <summary>
    /// 账单详情页"账单管理"那几行(账单分类 转账红包 &gt;、标签 请选择 &gt;、备注 添加 &gt;)
    /// 右对齐灰字和后面 &gt; 箭头的间距检查(经理说的 111: 假图都很靠近后面的箭头, 真图是有距离的)。
    /// 按颜色像素找, 不靠图片定位。
    ///
    /// 量 R = 间距 / 箭头高。假图两处不对而且同时出现: 箭头离文字近、箭头大一号, R 把两处都算进去;
    /// 用箭头高做单位不用文字高, 拼音页的文字行带着拼音, 文字高不准, 箭头不受影响。
    ///
    /// ★ 支付宝 8 月到 9 月之间改过这一块(箭头变小、离远了), 真图有新老两种:
    ///     9 月以后的真图(新版)         R 1.4~1.8(服务器 9 月样本 98.6% 是新版, 中位 1.50)
    ///     7~8 月的真图(老版)           R 0.86~0.94
    ///     拼音页(箭头本来就大一号)     R 0.95 起
    ///     假图                         R 0.60~0.76
    ///   没升级的用户截出来是老版的样子, 所以阈值放在假图和老版真图之间(0.78), 不放在假图和新版之间。
    ///
    /// 服务器 2026-09-01~16 抽的 101,380 张(54,291 张量到): 判可疑约 150 张(0.3%), 抽看都带假图特征
    /// (订单号日期和支付时间对不上、苹果状态栏出现在苹果截不出来的尺寸上等); 其中有时间轴的 14 张里
    /// 13 张 112 也判了假。有 142 张没有处理进度时间轴(收钱码、商家付款), 112 管不到, 这一项正好补上。
    ///
    /// 判不了(CannotDetermine)的几种, 都是宁可不判也不误判:
    ///     找不到账单管理那一列(至少 2 行箭头离右边一样远、箭头大小相近);
    ///     箭头高 &lt; MinArrowHeight: 图被缩小过;
    ///     (间距 + 1) / 箭头高 还够得着阈值: 差在一个像素以内, 小图自动更保守;
    ///     拼音页(用 PinyinCheck 判, 只在要判可疑时才跑)。
    ///
    /// 和 training/arrow_scan.py 逐步对应, 改一边要同步改另一边。
    /// 只读入参, 无静态可变状态, 可多线程调用。从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class ArrowGapCheck
    {
        /// <summary>R = 间距 / 箭头高 小于它判可疑。</summary>
        public const double Threshold = 0.78;

        /// <summary>箭头高小于这个像素数不判: 图被缩小过。</summary>
        public const int MinArrowHeight = 14;

        const int InkGray = 215, InkSat = 60;       // 字和箭头(含灰字): 灰度 < 215 且饱和度 < 60
        const int DarkGray = 110;                   // 左边黑字标签: 灰度 < 110 且饱和度 < 60

        /// <param name="image">8 位彩色图, 3 / 4 通道(BGR / BGRA)。传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static ArrowGapResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new ArrowGapResult { Verdict = ArrowGapVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn == 1) { res.Reason = "单通道图, 要靠颜色分灰字, 不判"; return res; }
            if (cn != 3 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");
            if (image.Width < 500 || image.Height < 800) { res.Reason = "图太小"; return res; }

            if (cn == 3) return Run(image, res);
            using var bgr = new Mat();
            Cv2.CvtColor(image, bgr, ColorConversionCodes.BGRA2BGR);
            return Run(bgr, res);
        }

        struct Row
        {
            public int Y, H, Margin, Gap, ArrowW, ArrowH, ArrowGray;
        }

        static ArrowGapResult Run(Mat bgr, ArrowGapResult res)
        {
            int W = bgr.Width, H = bgr.Height;
            using var grayM = new Mat();
            using var hsv = new Mat();
            Cv2.CvtColor(bgr, grayM, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(bgr, hsv, ColorConversionCodes.BGR2HSV);
            using var satM = new Mat();
            Cv2.ExtractChannel(hsv, satM, 1);
            var g = new byte[W * H];
            var s = new byte[W * H];
            System.Runtime.InteropServices.Marshal.Copy(grayM.Data, g, 0, g.Length);   // CvtColor 的输出是连续的
            System.Runtime.InteropServices.Marshal.Copy(satM.Data, s, 0, s.Length);

            var rows = FindRows(g, s, W, H);
            if (rows.Count == 0) { res.Reason = "没有账单管理那几行(灰字 + 箭头)"; return res; }

            // 账单管理那一列: 箭头离右边距离最常见的那个(+-2), 再去掉箭头大小差 20% 以上的; 至少 2 行。
            // 只有 1 行的多半是别的东西(蓝底转账页的"预约转账 >"、促销倒计时条)。
            // 最常见的有并列时取先出现的(和 Python 的 Counter.most_common 一样)。
            var counts = new Dictionary<int, int>();
            var order = new List<int>();
            foreach (var r in rows)
            {
                if (!counts.ContainsKey(r.Margin)) { counts[r.Margin] = 0; order.Add(r.Margin); }
                counts[r.Margin]++;
            }
            int mc = order[0];
            foreach (var m in order) if (counts[m] > counts[mc]) mc = m;
            var col = rows.Where(r => Math.Abs(r.Margin - mc) <= 2).ToList();
            double ah = Median(col.Select(r => (double)r.ArrowH));
            col = col.Where(r => Math.Abs(r.ArrowH - ah) <= 0.2 * ah).ToList();
            if (col.Count < 2) { res.Reason = "账单管理那一列不到 2 行, 不判"; return res; }

            double gap = Median(col.Select(r => (double)r.Gap));
            res.Measured = true;
            res.Gap = gap;
            res.ArrowHeight = ah;
            res.ArrowWidth = Median(col.Select(r => (double)r.ArrowW));
            res.ArrowGray = (int)Median(col.Select(r => (double)r.ArrowGray));
            res.RowCount = col.Count;
            res.Margin = mc;
            res.RowTops = col.Select(r => r.Y).ToArray();
            res.Ratio = ah > 0 ? gap / ah : 0;

            if (ah < MinArrowHeight) { res.Reason = $"箭头高 {ah:0.#} 像素, 小于 {MinArrowHeight}, 图被缩小过, 不判"; return res; }
            if (gap / ah >= Threshold)
            {
                res.Verdict = ArrowGapVerdict.Ok;
                res.Reason = $"箭头离文字 {gap:0.#} 像素, 箭头高 {ah:0.#}, R {res.Ratio:0.00}";
                return res;
            }
            if ((gap + 1) / ah >= Threshold)
            {
                res.Reason = $"R {res.Ratio:0.00}, 间距再量长一个像素就过线了, 差在测量误差以内, 不判";
                return res;
            }
            // 拼音模式下支付宝把箭头画大一号、间距也小一点, 真图 R 在 1.0 上下, 离假图太近, 不判
            res.PinyinChecked = true;
            res.Pinyin = PinyinCheck.Check(bgr).Verdict == PinyinVerdict.HasPinyin;
            if (res.Pinyin) { res.Reason = $"R {res.Ratio:0.00}, 但是拼音页(箭头本来就大), 不判"; return res; }

            res.Verdict = ArrowGapVerdict.Suspicious;
            res.Reason = $"箭头离文字只有 {gap:0.#} 像素, 箭头高 {ah:0.#}, R {res.Ratio:0.00} 小于 {Threshold}";
            return res;
        }

        /// <summary>所有"右对齐灰字 + 灰色 &gt; 箭头 + 左边有黑字标签"的行。</summary>
        static List<Row> FindRows(byte[] g, byte[] s, int W, int H)
        {
            int x0 = (int)(0.45 * W);
            int rw = W - x0;
            bool Ink(int y, int x) => g[y * W + x] < InkGray && s[y * W + x] < InkSat;
            bool Dark(int y, int x) => g[y * W + x] < DarkGray && s[y * W + x] < InkSat;

            var on = new bool[H];
            for (int y = 0; y < H; y++)
            {
                int n = 0;
                for (int x = x0; x < W && n < 2; x++) if (Ink(y, x)) n++;
                on[y] = n >= 2;
            }
            var outRows = new List<Row>();
            int lx0 = (int)(0.03 * W), lx1 = (int)(0.25 * W);
            foreach (var (a, b) in Runs(on))
            {
                int h = b - a + 1;
                if (h < 0.015 * W || h > 0.05 * W) continue;
                var colAny = new bool[rw];
                for (int x = 0; x < rw; x++)
                    for (int y = a; y <= b; y++)
                        if (Ink(y, x0 + x)) { colAny[x] = true; break; }
                var cr = Runs(colAny);
                if (cr.Count < 2) continue;
                var (ca, cb) = cr[^1];
                int cw = cb - ca + 1;
                int top = -1, bot = -1;
                for (int y = a; y <= b; y++)
                    for (int x = ca; x <= cb; x++)
                        if (Ink(y, x0 + x)) { if (top < 0) top = y; bot = y; break; }
                int ch = bot - top + 1;
                int margin = W - (x0 + cb) - 1;
                if (!(2 <= cw && cw <= 0.75 * ch && ch >= 0.4 * h && 0.03 * W <= margin && margin <= 0.14 * W)) continue;
                if (!IsChevron(Ink, a, b, x0 + ca, x0 + cb)) continue;

                int arrowGray = 255;
                for (int y = a; y <= b; y++)
                    for (int x = ca; x <= cb; x++)
                        if (Ink(y, x0 + x) && g[y * W + x0 + x] < arrowGray) arrowGray = g[y * W + x0 + x];
                if (arrowGray < 100) continue;                 // 黑箭头: 蓝底胶囊里的那种, 不是账单管理的行

                var (ta, tb) = cr[^2];
                var hist = new int[256];
                int cnt = 0;
                for (int y = a; y <= b; y++)
                    for (int x = 0; x <= tb; x++)
                        if (Ink(y, x0 + x)) { hist[g[y * W + x0 + x]]++; cnt++; }
                int textGray = MedianFloor(hist, cnt);
                if (textGray < 115 || textGray > 205) continue;  // 取值是灰字

                bool label = false;
                for (int y = a; y <= b && !label; y++)
                    for (int x = lx0; x < lx1; x++)
                        if (Dark(y, x)) { label = true; break; }
                if (!label) continue;                            // 左边要有黑字标签

                // 取值文字的起点: 从结尾往左, 字和字之间的空小于一个行高就还算同一段
                int j = cr.Count - 2;
                while (j > 0 && cr[j].a - cr[j - 1].b - 1 < h) j--;
                // 账单管理的取值是右对齐的短字, 左边到标签之间是一大片空白;
                // 起点左边 1.5 个行高以内有字, 说明是左对齐的长取值, 不要
                int tl = x0 + cr[j].a;
                bool inkLeft = false;
                for (int y = a; y <= b && !inkLeft; y++)
                    for (int x = Math.Max(0, tl - (int)(1.5 * h)); x < tl; x++)
                        if (Ink(y, x)) { inkLeft = true; break; }
                if (inkLeft) continue;

                outRows.Add(new Row { Y = a, H = h, Margin = margin, Gap = ca - tb - 1, ArrowW = cw, ArrowH = ch, ArrowGray = arrowGray });
            }
            return outRows;
        }

        /// <summary>
        /// '&gt;' 形: 每一行只有一笔(一段), 笔画细(尖上除外); 每行那一笔的中心从上往中间往右走, 再往下往左回来。
        /// 只比"最右点"不够: 数字 9、5, 往下的 ∨ 都能混过去(服务器上真把时间戳最后一位当成了箭头)。
        /// </summary>
        static bool IsChevron(Func<int, int, bool> ink, int a, int b, int xa, int xb)
        {
            int w = xb - xa + 1;
            int top = -1, bot = -1;
            for (int y = a; y <= b; y++)
                for (int x = xa; x <= xb; x++)
                    if (ink(y, x)) { if (top < 0) top = y; bot = y; break; }
            if (top < 0) return false;
            int h = bot - top + 1;
            int rowsWithInk = 0;
            for (int y = top; y <= bot; y++)
                for (int x = xa; x <= xb; x++)
                    if (ink(y, x)) { rowsWithInk++; break; }
            if (rowsWithInk < 6) return false;

            var cx = new double?[h];
            var line = new bool[w];
            for (int y = top; y <= bot; y++)
            {
                for (int x = 0; x < w; x++) line[x] = ink(y, xa + x);
                var segs = Runs(line);
                if (segs.Count != 1)
                {
                    if (segs.Count > 1) return false;
                    cx[y - top] = null;
                    continue;
                }
                var (sa, sb) = segs[0];
                if (sb - sa + 1 > 0.7 * w && !(h * 0.35 <= y - top && y - top <= h * 0.65)) return false;
                cx[y - top] = (sa + sb) / 2.0;
            }
            int q = Math.Max(1, h / 5);
            double? Mean(int lo, int hi)
            {
                lo = Math.Max(0, lo); hi = Math.Min(h, hi);
                double sum = 0; int n = 0;
                for (int i = lo; i < hi; i++) if (cx[i].HasValue) { sum += cx[i]!.Value; n++; }
                return n > 0 ? sum / n : null;
            }
            var t = Mean(0, q);
            var m = Mean(h / 2 - q / 2, h / 2 + q / 2 + 1);
            var btm = Mean(h - q, h);
            return t.HasValue && m.HasValue && btm.HasValue && m.Value - t.Value >= 0.25 * w && m.Value - btm.Value >= 0.25 * w;
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

        /// <summary>和 int(np.median(x)) 一样: 偶数个取中间两个的平均再往下取整。</summary>
        static int MedianFloor(int[] hist, int count)
        {
            if (count == 0) return 0;
            int k1 = (count - 1) / 2, k2 = count / 2;
            int a = -1, b = -1, cum = 0;
            for (int v = 0; v < 256; v++)
            {
                cum += hist[v];
                if (a < 0 && cum > k1) a = v;
                if (b < 0 && cum > k2) { b = v; break; }
            }
            return (int)((a + b) / 2.0);
        }
    }
}
