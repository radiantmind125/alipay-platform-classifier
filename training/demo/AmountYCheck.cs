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
        public string Reason = "";
    }

    /// <summary>
    /// 金额上下位置检查(2026-11-20, 经理: 假图金额Y坐标不对)。
    ///
    /// 页面: 白底 账单详情 页, 卡片里居中一摞 logo / 名字 / 金额(-800.00) / 交易成功, 下面左对齐一行行 创建时间 ...。
    /// 经理给的 3 张假图(1179x2556 苹果截图)把 金额 + 交易成功 整块贴低了:
    /// 名字底到金额顶 73~80 像素, 真图同一种手机(3 倍图)只有 56~61 像素; 交易成功 也跟着往下, 但没金额挪得多。
    ///
    /// 比例 = (金额顶 - 名字底) / (交易成功底 - 名字底)。只看图, 不用 OCR, 不用知道是什么手机:
    ///   名字底、交易成功底是中文字的下沿(拼音字体的拼音在字上面, 不影响), 金额数字的上沿很稳(逗号、小数点只往下);
    ///   手机大小、显示缩放都按比例变, 比例不变。
    ///     真图: 苹果 0.254~0.259, 安卓 0.22~0.28, 拼音字体最高 0.291
    ///     经理的 3 张假图 0.326 / 0.341 / 0.348; JPEG 95~40、缩 0.75 / 0.5、TG 1280、长边 2000 都还是 0.323~0.350
    /// Up * 100 &gt;= RatioMaxPercent * D 判可疑(整数比较)。
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
    ///
    /// ★ 传上传的原图(不要先转灰度)。缩放过比例基本不变, 但太小(D &lt; 60)判不了。
    /// 只读入参, 无静态可变状态, 可多线程调用。从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class AmountYCheck
    {
        /// <summary>Up / D &gt;= 这个百分比判可疑。</summary>
        public const int RatioMaxPercent = 31;

        const int Ink = 200;       // min(B, G, R) < 200 算墨
        const int Dark = 160;      // 金额要深色: min < 160
        const int Neutral = 40;    // 中性(灰黑): max - min < 40
        const int MinD = 60;       // 名字底到交易成功底不到 60 像素: 图太小

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

            // 4. 比例
            int d = S.B - N.B;
            int up = A.A - N.B;
            res.D = d; res.Up = up;
            if (d < MinD) { res.Reason = "字太小"; return res; }
            res.Measured = true;
            res.Ratio = (double)up / d;
            res.Verdict = up * 100 >= RatioMaxPercent * d ? AmountYVerdict.Suspicious : AmountYVerdict.Ok;
            int q = (up * 2000 + d) / (2 * d);      // 比例的千分位, 四舍五入用整数算(Python 一样), 理由里两边逐字相同
            string txt = $"{q / 1000}.{q % 1000:000}";
            res.Reason = res.Verdict == AmountYVerdict.Suspicious
                ? $"可疑: 金额偏下, 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, 比例 {txt} >= 0.{RatioMaxPercent:00}(真图一般 0.22~0.29)"
                : $"正常: 名字底到金额顶 {up} 像素, 名字底到交易成功底 {d} 像素, 比例 {txt}";
            return res;
        }
    }
}
