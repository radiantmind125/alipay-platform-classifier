#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum TextColorVerdict
    {
        CannotDetermine = 0,   // 判不了(不是账单详情那张卡片、深色模式、图太小、拼音页、颜色说不清), 走正常流程
        Ok = 1,                // 真图: 取值字是 #333
        Suspicious = 2,        // 假图: 取值字是纯黑 #000
    }

    public sealed class TextColorResult
    {
        public TextColorVerdict Verdict;
        public bool Measured;          // 找到了 >= 3 行对齐的"左标签 + 右取值", 下面几项才有意义
        public int Rows;               // 对齐的行数
        public int RowsBlack;          // 取值核心色 <= BlackMax 的行数
        public int RowsGrey;           // 取值核心色在 GreyLo~GreyHi 的行数
        public double BlackShare;      // RowsBlack / Rows
        public double GreyShare;       // RowsGrey / Rows
        public int LabelCore;          // 标签核心色(各行中位数): 真图 153, 造假工具 150; 不在 LabelLo~LabelHi 不判
        public int Background;         // 卡片里左边的底色, 只记录不判
        public int Edge;               // 卡片外面页面边上的颜色: 支付宝 245, 造假工具 244/246, 微信等整页白 255
        public int[] ValueCores = Array.Empty<int>();   // 每行取值的核心色(前 30 行)
        public bool PinyinChecked;     // 跑过拼音检查(只在要判可疑时才跑)
        public bool Pinyin;
        public string Reason = "";
    }

    /// <summary>
    /// 账单详情页正文字的颜色深度检查(经理说的 118: 这个颜色的……直接分析深度就能搞定)。
    ///
    /// 支付宝账单详情上面那张卡片: 左边灰标签 #999999(灰度 153), 右边取值 #333333(灰度 51)。
    /// 造假工具(和 111/112/116 的假图同一个, 1080x2400 苹果样式)取值是纯黑 #000000, 标签 150, 底色 244。
    /// JPEG 压过之后真图取值的核心颜色还是稳稳的 50~51, 造假工具的纯黑还是 0。
    /// 苹果的状态栏、导航栏标题本来就是纯黑, 所以只看导航栏下面(高度 9.5% 往下)。
    ///
    /// 量法:
    ///   1. 不彩色的像素(三通道最大减最小 &lt; 30)里, 灰度 &lt; 120 算深色字, 120~185 算灰色标签字, 按行切开。
    ///      深色模式(中位灰度 &lt; 128)不判。
    ///   2. 只要"左边灰标签 + 右边深色取值"的行: 标签在宽度 27% 以左、取值从 22% 往右, 两边都要 &gt;= 10 个像素,
    ///      标签核心色在 140~165。核心色 = 这一段像素里出现最多的灰度(笔画中间那种, 不受抗锯齿影响)。
    ///   3. 这些行要上下对齐: 至少 3 行, 标签左缘中位数在宽度 3%~10%, 取值左缘中位数在 20%~34%,
    ///      每行和中位数差 &lt;= 0.6% 宽度。对不齐的(收银台、支付结果页、银行短信、别的 App)不判,
    ///      它们本来就用纯黑字。
    ///   4. 只判支付宝的页面: 卡片外面页面边上的颜色要是支付宝的灰底(EdgeLo~EdgeHi), 标签色要在 LabelLo~LabelHi。
    ///      微信账单也是左标签右取值, 字是黑色 90%(25)或纯黑; 不加这一条服务器上会把 200 多张微信账单当成假图。
    ///      支付宝 8 月以前的老版是整页白, 也不判。
    ///   5. 每行一票: 纯黑(&lt;= BlackMax)的行占 &gt;= 60% 判可疑; #333 的行占 &gt;= 60% 判 Ok; 其他判不了。
    ///      纯黑只认 0 附近: 两种造假工具都是正好 0, 微信、支付宝话费充值页、银行短信是 21~25。
    ///   6. 要判可疑之前先看是不是拼音页(PinyinCheck), 拼音页不判: 拼音字体有细体的变种, 字会压得很深。
    ///
    /// 服务器 09-01~16 抽的 101,285 张: 判可疑 66 张(0.07%), 逐张看过都是支付宝账单详情的假图
    /// (34 张 111/112 也判了假; 32 张只有这一项判出来, 订单号是乱的、两笔转账同一个订单号、或者底色是造假工具的)。
    /// 页面左右被裁掉(灰底没了)判不了。
    ///
    /// 和 training/color_scan.py 逐步对应, 改一边要同步改另一边。
    /// 只读入参, 无静态可变状态, 可多线程调用。从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class TextColorCheck
    {
        /// <summary>行的取值核心色 &lt;= 这个算纯黑(造假工具都是 0; 微信、话费充值页、银行短信 21~25 不能算)。</summary>
        public const int BlackMax = 12;
        /// <summary>卡片外面页面边上的颜色范围: 支付宝 #F5F5F5(245), 造假工具 244/246; 整页白(255)的不是支付宝页面。</summary>
        public const int EdgeLo = 238, EdgeHi = 250;
        /// <summary>标签颜色范围: 支付宝 #999(153), 造假工具 150。</summary>
        public const int LabelLo = 148, LabelHi = 158;
        /// <summary>行的取值核心色在这个范围算 #333。</summary>
        public const int GreyLo = 38, GreyHi = 64;
        /// <summary>纯黑(或 #333)的行占比 &gt;= 60% 就下结论(按 x * 5 &gt;= rows * 3 整数比, 不受浮点影响)。</summary>
        public const double Share = 0.60;

        const int DarkGray = 120, LabelGrayHi = 185, MaxChroma = 30;

        /// <param name="image">8 位彩色图, 3 / 4 通道(BGR / BGRA)。传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static TextColorResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new TextColorResult { Verdict = TextColorVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn == 1) { res.Reason = "单通道图, 要靠颜色分灰字, 不判"; return res; }
            if (cn != 3 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");
            if (image.Width < 500 || image.Height < 900) { res.Reason = "图太小"; return res; }

            if (cn == 3) return Run(image, res);
            using var bgr = new Mat();
            Cv2.CvtColor(image, bgr, ColorConversionCodes.BGRA2BGR);
            return Run(bgr, res);
        }

        struct Cand
        {
            public int Y, YB, LabelCore, ValueCore, LabelLeft, ValueLeft;
        }

        static TextColorResult Run(Mat bgr, TextColorResult res)
        {
            int W = bgr.Width, H = bgr.Height;
            int y0 = (int)(0.095 * H), y1 = (int)(0.96 * H);
            int x0 = (int)(0.03 * W), x1 = (int)(0.97 * W);

            // 灰度和"彩色程度"(三通道最大减最小); 输出都是新建的 Mat, 连续, 可以整块拷出来
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
                foreach (var m in ch) m.Dispose();
            }
            var g = new byte[W * H];
            var c = new byte[W * H];
            System.Runtime.InteropServices.Marshal.Copy(grayM.Data, g, 0, g.Length);
            System.Runtime.InteropServices.Marshal.Copy(chromaM.Data, c, 0, c.Length);

            // 左边页边(宽度 1%)的底色, 只记录
            int bw = Math.Max(1, (int)(0.01 * W));
            var bh = new int[256];
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x0 + bw; x++) bh[g[y * W + x]]++;
            res.Background = ArgMax(bh);

            // 深色模式: 正文区域的中位灰度 < 128(和 np.median 一样, 偶数个取中间两个的平均, 这里比两倍免得出现 .5)
            var gh = new int[256];
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++) gh[g[y * W + x]]++;
            if (MedianTwice(gh, (y1 - y0) * (x1 - x0)) < 256) { res.Reason = "深色模式"; return res; }

            bool Dark(int i) => g[i] < DarkGray && c[i] < MaxChroma;
            bool Label(int i) => g[i] >= DarkGray && g[i] <= LabelGrayHi && c[i] < MaxChroma;

            // 一行里深色字或灰色标签字 >= 3 个像素算有字
            var on = new bool[y1 - y0];
            for (int y = y0; y < y1; y++)
            {
                int n = 0;
                for (int x = x0; x < x1 && n < 3; x++)
                {
                    int i = y * W + x;
                    if (g[i] <= LabelGrayHi && c[i] < MaxChroma) n++;
                }
                on[y - y0] = n >= 3;
            }

            int lx1 = (int)(0.27 * W);     // 标签栏: x0 ~ 27% 宽度
            int vx0 = (int)(0.22 * W);     // 取值栏: 22% 宽度 ~ x1
            var cand = new List<Cand>();
            var lh = new int[256];
            var vh = new int[256];
            foreach (var (ra, rb) in Runs(on))
            {
                int a = ra + y0, b = rb + y0;
                int h = b - a + 1;
                if (h < 0.008 * H || h > 0.04 * H) continue;
                Array.Clear(lh); Array.Clear(vh);
                int ln = 0, vn = 0, ll = -1, vl = -1;
                for (int x = x0; x < lx1; x++)
                    for (int y = a; y <= b; y++)
                    {
                        int i = y * W + x;
                        if (!Label(i)) continue;
                        lh[g[i]]++; ln++;
                        if (ll < 0) ll = x;
                    }
                for (int x = vx0; x < x1; x++)
                    for (int y = a; y <= b; y++)
                    {
                        int i = y * W + x;
                        if (!Dark(i)) continue;
                        vh[g[i]]++; vn++;
                        if (vl < 0) vl = x;
                    }
                if (ln < 10 || vn < 10) continue;
                int lcore = ArgMax(lh), vcore = ArgMax(vh);
                if (lcore < 140 || lcore > 165) continue;
                cand.Add(new Cand { Y = a, YB = b, LabelCore = lcore, ValueCore = vcore, LabelLeft = ll, ValueLeft = vl });
            }

            // 账单详情上面那张卡片: 标签都从左边 3%~10% 起、取值都从同一个 x 起(20%~34%), 上下对齐。
            // 收银台、支付结果页、订单说明这些别的页面对不齐, 就不判(它们本来就用纯黑字)。
            var pairs = new List<Cand>();
            if (cand.Count >= 3)
            {
                double llm = Median(cand.Select(t => (double)t.LabelLeft));
                double vlm = Median(cand.Select(t => (double)t.ValueLeft));
                double tol = 0.006 * W;
                if (0.03 * W <= llm && llm <= 0.10 * W && 0.20 * W <= vlm && vlm <= 0.34 * W)
                    pairs = cand.Where(t => Math.Abs(t.LabelLeft - llm) <= tol && Math.Abs(t.ValueLeft - vlm) <= tol).ToList();
            }
            int rows = pairs.Count;
            res.Rows = rows;
            res.RowsBlack = pairs.Count(t => t.ValueCore <= BlackMax);
            res.RowsGrey = pairs.Count(t => t.ValueCore >= GreyLo && t.ValueCore <= GreyHi);
            if (rows > 0)
            {
                res.BlackShare = (double)res.RowsBlack / rows;
                res.GreyShare = (double)res.RowsGrey / rows;
                res.LabelCore = (int)Median(pairs.Select(t => (double)t.LabelCore));
                res.ValueCores = pairs.Take(30).Select(t => t.ValueCore).ToArray();
                // 卡片左边外面(宽度 0.4%~1.6%)、对齐那几行高度上, 页面底色出现最多的灰度
                int xa = (int)(0.004 * W), xb = Math.Max(xa + 1, (int)(0.016 * W));
                var eh = new int[256];
                foreach (var t in pairs)
                    for (int y = t.Y; y <= t.YB; y++)
                        for (int x = xa; x < xb; x++) eh[g[y * W + x]]++;
                res.Edge = ArgMax(eh);
            }
            if (rows < 3) { res.Reason = "不是左标签右取值的账单详情页"; return res; }
            res.Measured = true;

            // 只判支付宝的页面(微信账单、别的 App、支付宝老版整页白; 标签颜色不对的别的页面)
            if (res.Edge < EdgeLo || res.Edge > EdgeHi) { res.Reason = $"页面边上是 {res.Edge}, 不是支付宝的灰底(微信账单、别的 App), 不判"; return res; }
            if (res.LabelCore < LabelLo || res.LabelCore > LabelHi) { res.Reason = $"标签颜色 {res.LabelCore}, 不是支付宝的, 不判"; return res; }

            // 按行数算(每行一票), 不按像素: 一大块黑色(按钮、图片)会把像素占比带偏
            if (res.RowsBlack * 5 >= rows * 3)
            {
                // 拼音检查比较贵, 只在要判可疑时才跑
                res.PinyinChecked = true;
                res.Pinyin = PinyinCheck.Check(bgr).Verdict == PinyinVerdict.HasPinyin;
                if (res.Pinyin) { res.Reason = "拼音页, 拼音字体的字本来就深, 不判"; return res; }
                res.Verdict = TextColorVerdict.Suspicious;
                res.Reason = $"取值字是纯黑({res.RowsBlack}/{rows} 行), 支付宝是 #333";
                return res;
            }
            if (res.RowsGrey * 5 >= rows * 3)
            {
                res.Verdict = TextColorVerdict.Ok;
                res.Reason = $"取值字是 #333({res.RowsGrey}/{rows} 行)";
                return res;
            }
            res.Reason = "取值字颜色说不清";
            return res;
        }

        /// <summary>出现最多的灰度; 并列取小的(和 np.bincount(...).argmax() 一样)。</summary>
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
