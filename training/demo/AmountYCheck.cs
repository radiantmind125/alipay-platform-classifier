#nullable enable

using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace Ssp
{
    public enum AmountYVerdict
    {
        CannotDetermine = 0,   // 判不了(不是账单详情这种页面、名字是标点 / 字母 / 符号、图太小), 走正常流程
        Ok = 1,                // 金额上下位置正常
        Suspicious = 2,        // 金额偏下(假图)
    }

    public sealed class AmountYResult
    {
        public AmountYVerdict Verdict;
        public bool Measured;            // 量到了, 下面的 Up / D / Ratio 才有意义
        public bool Found;               // 找到了 名字 / 金额 / 交易成功 / 第一行, 下面的位置才有意义
        public int NameTop, NameBottom;          // 名字那一行(原图 y, 下沿不含)
        public int AmountTop, AmountBottom;      // 金额那一行
        public int AmountLeft, AmountRight;      // 金额左右(原图 x, 右沿不含)
        public int StatusTop, StatusBottom;      // 交易成功 那一行
        public int StatusLeft, StatusRight;
        public int RowTop, RowBottom;            // 下面第一行(创建时间 / 服务费 ...)
        public int Up;                   // 金额顶 - 名字底, 像素
        public int D;                    // 交易成功底 - 名字底, 像素
        public double Ratio;             // Up / D
        public int NameDark, StatusDark; // 名字 / 交易成功 里深灰中性像素数(第 5 步)
        public int GapOdd = -1;          // 名字和金额之间和底色差 > 40 的像素数(第 5 步; -1 = 没量)
        public bool IsIPhoneSize;        // 图的尺寸正好是 iPhone 原生截图尺寸
        public int ThresholdPercent;     // 这张图用的门槛(30 或 32)
        public string Reason = "";
    }

    /// <summary>
    /// 金额上下位置检查(2026-11-20, 经理: 假图金额Y坐标不对)。
    ///
    /// 页面: 白底 账单详情 页, 卡片里居中一摞 logo / 名字 / 金额(-800.00) / 交易成功, 下面左对齐一行行 创建时间 ...。
    /// 经理给的 3 张假图(1179x2556 苹果截图)把 金额 + 交易成功 整块贴低了:
    /// 名字底到金额顶 73~80 像素, 真图同一种手机(3 倍图)只有 56~61 像素; 交易成功 也跟着往下, 但没金额挪得多。
    ///
    /// 比例 = (金额顶 - 名字底) / (交易成功底 - 名字底)。只看图, 不用 OCR, 不用传机型:
    ///   名字底、交易成功底是中文字的下沿(拼音字体的拼音在字上面, 不影响), 金额数字的上沿很稳(逗号、小数点只往下);
    ///   手机大小、显示缩放都按比例变, 比例不变。Up * 100 &gt;= 门槛 * D 判可疑(整数比较), 门槛分两种:
    ///     图的尺寸正好是 iPhone 原生截图尺寸: RatioMaxPercentIPhone = 30。苹果不能换系统字体, 排版是死的,
    ///       同一尺寸的真图位置差不过 2 像素; 服务器 11 月 1~4 日 iPhone 尺寸的 账单详情 真图最高约 0.28。
    ///     别的尺寸: RatioMaxPercent = 32。安卓可以换手写体、拼音字体, 中文字下沿会往上抬, 真图最高到 0.307。
    ///
    /// 实测(2026-11-20):
    ///     经理的 3 张假图(1179x2556) 0.326 / 0.341 / 0.348;
    ///     JPEG 95~40、缩 0.75 / 0.5、TG 1280、长边 2000(变成别的尺寸就用 0.32) 0.323~0.350, 75 个变体全判可疑。
    ///     服务器 11 月白图全部 195,078 张(11-01~11-04), 量到 164,647 张, p50 0.261, p99 0.276。
    ///       苹果尺寸的真图一档一档很整齐: 例如 1170x2532 一万二千多张, 名字底到金额顶 56~58 像素、
    ///       到交易成功底 214~218 像素, 金额数字高都是 71(带逗号 82)。
    ///       第一版(没有第 5 步)按 0.31 判可疑 65 张, 看了小图:
    ///         别的页面 41 张(银行 app 交易详情 -¥500.00 / 余额 / 交易卡号 14 张、支付成功 + 银行立减金 25 张、转钱包 2 张)
    ///           和 账单详情 上盖了弹窗 1 张 —— 第 5 步挡掉;
    ///         剩下 23 张都是 账单详情, 金额比同尺寸真图低 14~25 像素, 苹果尺寸的数字还都比真图小一号
    ///           (1170 宽 62~69, 真图 71), 里面有以前 111 箭头检查抓到的假图 —— 都是改过的图。
    ///       定版(本文件)同样 195,078 张再跑一遍: 量到 163,810 张, 可疑 16 张(iPhone 尺寸 12、别的 4),
    ///         一张张看过都是 账单详情: 金额比同尺寸真图的常见位置低 6~22 像素, 14 张数字比真图小一号
    ///         (1080 宽 50~57, 真图 66; 1320x2868 59~68, 真图 79), 另 2 张金额没有小数(-20000)、数字纯黑。
    ///         第 5 步挡掉 837 张: 比例 &gt;= 0.30 的 42 张全是别的页面和 1 张弹窗; 其余是 收款金额 / 付款成功 / 转账成功
    ///         一类别的页面, 少数是状态不是深灰字的 账单详情(处理中、等待对方领取)。
    ///         iPhone 尺寸量到 76,405 张, 99.9% 在 0.2785 以下; 0.30~0.32 的安卓图是以前的假图和手写体 / 拼音字体真图(最高 0.307)。
    ///     服务器 11 月蓝图随机 2 万张: 0 张量到(不是这种页面)。
    ///
    /// 做法(和 training/amount_y_check.py 一步一步对应, 改一边要同步改另一边):
    ///  1. 整屏竖截图才判: W &gt;= 300, H &gt;= 1.6W。
    ///  2. 墨 = min(B, G, R) &lt; 200。只看 0.04W~0.96W、上面 0.6H。每行 &gt;= 2 个墨点算有字, 连续有字的行是一条行带;
    ///     行带之间的缝 &lt;= max(2, W/250) 的合并(logo 中间断开)。每条行带记左右墨的范围。
    ///  3. 从上往下找金额: 居中(左沿 &gt; 0.12W 且 |左 + 右 - W| &lt;= 0.06W)、高 0.040W~0.095W、宽 &gt;= 1.5 倍高、
    ///     深色中性(min &lt; 160 的像素里 max - min &lt; 40 的 &gt;= 80%, logo 是彩色的);
    ///     上一条(名字)居中; 下一条(交易成功)居中、宽 &lt;= 0.5W; 再下一条从 0.12W 以内起(第一行);
    ///     金额高 &gt;= 1.3 倍 交易成功 高; 名字高在 交易成功 高的 0.9~1.3 倍
    ///     (名字是标点、字母、~ 一类时下沿不是中文字的下沿, 不判)。第一个全对上的就是。
    ///  4. D = 交易成功底 - 名字底, 不到 60 像素不判; Up = 金额顶 - 名字底。
    ///  5. 是不是 账单详情 那一摞(宁可不判):
    ///     名字、交易成功 两行里深灰中性像素(min &lt; 110 且 max - min &lt; 40)都要 &gt;= 行高平方 / 16
    ///       (银行页面的 余额 是浅灰字、支付成功页的 支付成功 是蓝字、划掉的原价是浅灰字, 都是 0);
    ///     名字底和金额顶之间(上下各让 max(2, Up/8), 左右 0.12W~0.88W)和底色差 &gt; 40 的像素 &lt; 千分之五
    ///       (底色 = 这块每个通道的下中位数; 弹窗、水印、阴影会超)。
    ///  6. 比例 = Up / D, 和门槛比。
    ///
    /// ★ 传上传的原图(不要先缩放、不要先转灰度): 原图尺寸才认得出 iPhone(门槛 0.30), 缩放过就按 0.32 判;
    ///   太小(D &lt; 60)判不了。
    /// 只读入参, 无静态可变状态, 可多线程调用。从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class AmountYCheck
    {
        /// <summary>图的尺寸正好是 iPhone 原生截图尺寸时: Up / D &gt;= 这个百分比判可疑。</summary>
        public const int RatioMaxPercentIPhone = 30;

        /// <summary>别的尺寸: Up / D &gt;= 这个百分比判可疑(安卓能换字体, 真图最高 0.307)。</summary>
        public const int RatioMaxPercent = 32;

        const int Ink = 200;       // min(B, G, R) < 200 算墨
        const int Dark = 160;      // 金额要深色: min < 160
        const int Neutral = 40;    // 中性(灰黑): max - min < 40
        const int MinD = 60;       // 名字底到交易成功底不到 60 像素: 图太小
        const int TextDark = 110;  // 名字 / 交易成功 的深灰字(#333): min < 110 且中性
        const int GapDist = 40;    // 名字和金额之间, 和底色差(三通道差的绝对值之和) > 40 的像素算"有东西"
        const int GapMaxPermille = 5;  // 有东西的像素 >= 千分之五: 不判

        /// <summary>iPhone 原生截图尺寸(竖屏), 和 YenCheck 的表一样。</summary>
        static readonly HashSet<(int W, int H)> IPhoneSizes = new()
        {
            (640, 1136), (750, 1334), (1242, 2208), (828, 1792), (1125, 2436), (1242, 2688), (1170, 2532),
            (1284, 2778), (1179, 2556), (1290, 2796), (1206, 2622), (1320, 2868), (1260, 2736),
        };

        /// <param name="image">8 位图, 1 / 3 / 4 通道(灰度 / BGR / BGRA)。传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static AmountYResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new AmountYResult { Verdict = AmountYVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn == 3)
            {
                if (image.IsContinuous()) return Run(image, res);
                using var c = image.Clone();
                return Run(c, res);
            }
            if (cn != 1 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");
            using var bgr = new Mat();
            Cv2.CvtColor(image, bgr, cn == 4 ? ColorConversionCodes.BGRA2BGR : ColorConversionCodes.GRAY2BGR);
            return Run(bgr, res);
        }

        /// <summary>一维 bool -> [起, 止) 连续为真的段</summary>
        static List<(int A, int B)> Runs(bool[] on)
        {
            var o = new List<(int, int)>();
            int x = 0, n = on.Length;
            while (x < n)
            {
                if (on[x])
                {
                    int x0 = x;
                    while (x < n && on[x]) x++;
                    o.Add((x0, x));
                }
                else x++;
            }
            return o;
        }

        static AmountYResult Run(Mat bgr, AmountYResult res)
        {
            int W = bgr.Width, H = bgr.Height;
            if (W < 300 || H * 10 < 16 * W) { res.Reason = "不是整屏竖截图"; return res; }
            int yz = H * 60 / 100;
            int xa = W * 4 / 100, xz = W * 96 / 100;
            var px = new byte[(long)W * yz * 3];
            System.Runtime.InteropServices.Marshal.Copy(bgr.Data, px, 0, px.Length);   // 连续的 BGR, 只要上面 0.6H
            var mn = new byte[(long)W * yz];
            var mx = new byte[(long)W * yz];
            for (long i = 0, o = 0; i < mn.Length; i++, o += 3)
            {
                byte b = px[o], g = px[o + 1], r = px[o + 2];
                mn[i] = Math.Min(b, Math.Min(g, r));
                mx[i] = Math.Max(b, Math.Max(g, r));
            }
            bool InkAt(int y, int x) => mn[(long)y * W + x] < Ink;

            // 2. 行带
            var rowOn = new bool[yz];
            for (int y = 0; y < yz; y++)
            {
                int cnt = 0;
                for (int x = xa; x < xz; x++) if (InkAt(y, x)) cnt++;
                rowOn[y] = cnt >= 2;
            }
            var raw = Runs(rowOn);
            int gapMax = Math.Max(2, W / 250);
            var bands = new List<(int A, int B)>();
            foreach (var (a, b) in raw)
            {
                if (bands.Count > 0 && a - bands[bands.Count - 1].B <= gapMax)
                    bands[bands.Count - 1] = (bands[bands.Count - 1].A, b);
                else bands.Add((a, b));
            }
            var B = new List<(int A, int B, int X0, int X1)>();
            foreach (var (a, b) in bands)
            {
                int c0 = -1, c1 = -1;
                for (int x = xa; x < xz; x++)
                {
                    bool any = false;
                    for (int y = a; y < b; y++) if (InkAt(y, x)) { any = true; break; }
                    if (any) { if (c0 < 0) c0 = x; c1 = x; }
                }
                B.Add((a, b, c0, c1 + 1));
            }

            bool Centered((int A, int B, int X0, int X1) t) =>
                t.X0 * 100 > 12 * W && Math.Abs(t.X0 + t.X1 - W) * 100 <= 6 * W;

            // 3. 找金额
            int amt = -1;
            bool nameBad = false;
            for (int i = 0; i < B.Count; i++)
            {
                var t = B[i];
                int h = t.B - t.A;
                if (!Centered(t) || !(40 * W <= h * 1000 && h * 1000 <= 95 * W) || (t.X1 - t.X0) * 2 < 3 * h) continue;
                int dark = 0, neu = 0;
                for (int y = t.A; y < t.B; y++)
                    for (int x = t.X0; x < t.X1; x++)
                    {
                        long k = (long)y * W + x;
                        if (mn[k] < Dark)
                        {
                            dark++;
                            if (mx[k] - mn[k] < Neutral) neu++;
                        }
                    }
                if (dark == 0 || neu * 5 < dark * 4) continue;
                if (i < 1 || i + 2 >= B.Count) continue;
                var nm = B[i - 1]; var st = B[i + 1]; var row = B[i + 2];
                if (!(Centered(nm) && Centered(st)) || (st.X1 - st.X0) * 2 > W || row.X0 * 100 > 12 * W) continue;
                int sh = st.B - st.A, nh = nm.B - nm.A;
                if (h * 10 < 13 * sh) continue;
                if (!(9 * sh <= nh * 10 && nh * 10 <= 13 * sh)) { nameBad = true; continue; }
                amt = i;
                break;
            }
            if (amt < 0)
            {
                res.Reason = nameBad ? "名字太矮或太高(标点、字母、符号一类), 量不准"
                                     : "没找到 名字 / 金额 / 交易成功 / 第一行 这样的一摞(不是账单详情这种页面)";
                return res;
            }
            var N = B[amt - 1]; var A = B[amt]; var S = B[amt + 1]; var R = B[amt + 2];
            res.Found = true;
            res.NameTop = N.A; res.NameBottom = N.B;
            res.AmountTop = A.A; res.AmountBottom = A.B; res.AmountLeft = A.X0; res.AmountRight = A.X1;
            res.StatusTop = S.A; res.StatusBottom = S.B; res.StatusLeft = S.X0; res.StatusRight = S.X1;
            res.RowTop = R.A; res.RowBottom = R.B;

            // 4. 间距
            int d = S.B - N.B;
            int up = A.A - N.B;
            res.D = d; res.Up = up;
            if (d < MinD) { res.Reason = "字太小"; return res; }

            // 5. 是不是 账单详情 那一摞
            int DarkText((int A, int B, int X0, int X1) t)
            {
                int c = 0;
                for (int y = t.A; y < t.B; y++)
                    for (int x = t.X0; x < t.X1; x++)
                    {
                        long k = (long)y * W + x;
                        if (mn[k] < TextDark && mx[k] - mn[k] < Neutral) c++;
                    }
                return c;
            }
            int nameH = N.B - N.A, statusH = S.B - S.A;
            res.NameDark = DarkText(N); res.StatusDark = DarkText(S);
            if (res.NameDark * 16 < nameH * nameH || res.StatusDark * 16 < statusH * statusH)
            {
                res.Reason = "名字或交易成功不是深灰色的字(不像账单详情, 比如银行页面的余额、支付成功页), 不判"; return res;
            }
            int mg = Math.Max(2, up / 8);
            int ga = N.B + mg, gb = A.A - mg;
            int gx0 = W * 12 / 100, gx1 = W * 88 / 100;
            if (gb > ga)
            {
                var bg = new int[3];
                long area = (long)(gb - ga) * (gx1 - gx0);
                for (int ch = 0; ch < 3; ch++)
                {
                    var hist = new long[256];
                    for (int y = ga; y < gb; y++)
                        for (int x = gx0; x < gx1; x++) hist[px[((long)y * W + x) * 3 + ch]]++;
                    long need = (area + 1) / 2, acc = 0;
                    int v = 0;
                    for (; v < 256; v++) { acc += hist[v]; if (acc >= need) break; }
                    bg[ch] = v;
                }
                int odd = 0;
                for (int y = ga; y < gb; y++)
                    for (int x = gx0; x < gx1; x++)
                    {
                        long o = ((long)y * W + x) * 3;
                        int dd = Math.Abs(px[o] - bg[0]) + Math.Abs(px[o + 1] - bg[1]) + Math.Abs(px[o + 2] - bg[2]);
                        if (dd > GapDist) odd++;
                    }
                res.GapOdd = odd;
                if ((long)odd * 1000 >= GapMaxPermille * area)
                {
                    res.Reason = "名字和金额之间有别的东西(弹窗、水印、阴影一类), 量不准"; return res;
                }
            }

            // 6. 比例
            bool iphone = IPhoneSizes.Contains((W, H));
            int thr = iphone ? RatioMaxPercentIPhone : RatioMaxPercent;
            res.IsIPhoneSize = iphone; res.ThresholdPercent = thr;
            res.Measured = true;
            res.Ratio = (double)up / d;
            res.Verdict = up * 100 >= thr * d ? AmountYVerdict.Suspicious : AmountYVerdict.Ok;
            int q = (up * 2000 + d) / (2 * d);      // 比例的千分位, 四舍五入用整数算(Python 一样), 理由里两边逐字相同
            string txt = $"{q / 1000}.{q % 1000:000}";
            string kind = iphone ? "iPhone 原图尺寸, 真图最高约 0.28" : "真图最高约 0.31";
            res.Reason = res.Verdict == AmountYVerdict.Suspicious
                ? $"可疑: 金额偏下, 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, 比例 {txt} >= 0.{thr:00}({kind})"
                : $"正常: 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, 比例 {txt}(门槛 0.{thr:00})";
            return res;
        }
    }
}
