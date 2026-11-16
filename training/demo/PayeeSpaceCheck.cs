#nullable enable

using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace Ssp
{
    public enum PayeeSpaceVerdict
    {
        CannotDetermine = 0,   // 判不了(不是这个页面、名字是标点 / 表情 / 太窄的字、图太小), 走正常流程
        Ok = 1,                // 名字和左括号之间的空白正常
        Suspicious = 2,        // 名字和左括号之间空一大块(假图)
    }

    public sealed class PayeeSpaceResult
    {
        public PayeeSpaceVerdict Verdict;
        public bool Measured;        // 量到了, 下面几项才有意义
        public int Gap;              // 名字最后一个字最右的墨 到 左括号最左的墨, 像素
        public int LineHeight;       // 收款方那一行的行高(有字的行), 像素
        public double Ratio;         // Gap / LineHeight
        public int LineTop, LineBottom;     // 收款方那一行(原图坐标)
        public int NameRight, BracketLeft;  // 量的两个位置(原图 x)
        public string Reason = "";
    }

    /// <summary>
    /// 收款方空白检查(2026-11-17, 经理: 针对这个写个算法 检测这个空白位置)。
    ///
    /// 页面: 蓝色页头的 转账成功 页(转账成功 / ¥金额 / 收款方 ... 名字(**某) / 付款方式 ... / 白卡片 立即通知收款人)。
    /// 量页头 收款方 那一行: 名字最后一个字 和 左括号 之间空多少, 按这一行的行高算比例。
    /// OCR 读出来有时带空格有时不带, 所以按像素量。只看图, 不用 OCR, 不用知道是什么手机。
    ///
    ///     经理给的 3 张假图(1320x2868)            0.81~0.83(空 34~35 像素, 行高 42)
    ///     经理截的图(780x1280, 压过)              0.82
    ///     苹果原生截图真图                        0.32~0.34(名字和括号之间本来就空半个字)
    ///     安卓真图                                0.02~0.25
    ///     本机所有真图(3,300 张整页, 7~11 月)     量到 205 张, 最高 0.46(装饰字), 0 张可疑
    /// 比例 &gt;= GapMinPercent / 100 判可疑。
    ///
    /// 做法(和 training/payee_space_check.py 一步一步对应, 改一边要同步改另一边):
    ///  1. 页头: 从 0.12H 往下找第一行"蓝底像素不到一半宽"的行, 那以上是蓝色页头(最多到 0.55H, 不到 0.15H 判不了)。
    ///     蓝底像素: B &gt;= 200, R &lt;= 130, B - R &gt;= 100。
    ///  2. 页头里的字: R &gt;= 150(蓝底 R 只有 50~80)。每行在 0.02W~0.98W 之间数字像素, &gt;= 2 个算有字, 连续有字的行是一条行带。
    ///  3. 最高的行带是金额; 金额下面第一条是 收款方 那一行(比下一条矮四成以上的是拼音行, 跳过)。
    ///  4. 这一行按列切段(整行宽, 连续有字的列), 最大的空把标签和取值分开;
    ///     标签从 0.25W 以内起、宽 2~4 个行高(收款方), 取值到 0.75W 以后才完。
    ///  5. 取值最后一段要像右括号; 找第一个"前面紧跟左括号"的星号; 再前一段是名字最后一个字。
    ///     星号: 高 &lt;= 0.45 行高、上沿 &lt; 0.6 行高、宽 &lt;= 0.75 行高、宽 &gt;= 0.4 倍自己的高。
    ///     括号: 宽 &lt;= 0.4 行高、宽 &lt;= 0.6 倍自己的高、高 &gt;= 0.4 行高。
    ///  6. 名字最后一个字宽 &lt; 0.3 行高(1 l 川 的一竖、括号)、高 &lt; 0.5 行高(。，、), 或者从它到左括号之间
    ///     B &lt; 120 的像素(表情图标一类的彩色东西) &gt;= 行高平方 / 40, 判不了。
    ///  7. Gap = 左括号左沿 - 名字右沿; Gap * 100 &gt;= GapMinPercent * 行高 判可疑(整数比较)。
    ///
    /// ★ 传上传的原图(不要先缩放、不要先转灰度): 要靠颜色分蓝底和白字。缩放过比例基本不变, 但字太小(行高 &lt; 12)判不了。
    /// 只读入参, 无静态可变状态, 可多线程调用。从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class PayeeSpaceCheck
    {
        /// <summary>Gap / 行高 &gt;= 这个百分比判可疑。</summary>
        public const int GapMinPercent = 55;

        const int InkR = 150;          // 页头白字: R >= 150
        const int ForeignB = 120;      // 页头里 B < 120: 不是白字也不是蓝底(JPEG 在白字边上把蓝底的 B 压到 140~160)

        /// <param name="image">8 位彩色图, 3 / 4 通道(BGR / BGRA)。传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static PayeeSpaceResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new PayeeSpaceResult { Verdict = PayeeSpaceVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn == 1) { res.Reason = "单通道图, 要靠颜色分蓝底白字, 不判"; return res; }
            if (cn != 3 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");
            if (cn == 3)
            {
                if (image.IsContinuous()) return Run(image, res);
                using var c = image.Clone();
                return Run(c, res);
            }
            using var bgr = new Mat();
            Cv2.CvtColor(image, bgr, ColorConversionCodes.BGRA2BGR);
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

        static PayeeSpaceResult Run(Mat bgr, PayeeSpaceResult res)
        {
            int W = bgr.Width, H = bgr.Height;
            if (H < 400 || W < 300) { res.Reason = "图太小"; return res; }
            var px = new byte[(long)W * H * 3];
            System.Runtime.InteropServices.Marshal.Copy(bgr.Data, px, 0, px.Length);   // 连续的 BGR
            int Bv(int y, int x) => px[((long)y * W + x) * 3];
            int Rv(int y, int x) => px[((long)y * W + x) * 3 + 2];

            // 1. 蓝色页头的底
            int lim = (int)(0.55 * H);
            int yb = lim;
            for (int y = (int)(0.12 * H); y < lim; y++)
            {
                int cnt = 0;
                for (int x = 0; x < W; x++)
                {
                    int b = Bv(y, x), r = Rv(y, x);
                    if (b >= 200 && r <= 130 && b - r >= 100) cnt++;
                }
                if (cnt * 2 < W) { yb = y; break; }
            }
            if (yb < (int)(0.15 * H)) { res.Reason = "没有蓝色页头"; return res; }

            // 2. 行带
            int xa = (int)(0.02 * W), xz = (int)(0.98 * W);
            var rowOn = new bool[yb];
            for (int y = 0; y < yb; y++)
            {
                int cnt = 0;
                for (int x = xa; x < xz; x++) if (Rv(y, x) >= InkR) cnt++;
                rowOn[y] = cnt >= 2;
            }
            var bands = Runs(rowOn);
            if (bands.Count < 3) { res.Reason = "页头字行不到 3 条"; return res; }
            int ia = 0;
            for (int i = 1; i < bands.Count; i++)
                if (bands[i].B - bands[i].A > bands[ia].B - bands[ia].A) ia = i;     // 并列取先出现的(和 np.argmax 一样)

            // 3. 金额下面第一条(拼音行跳过)
            int pay = -1;
            for (int i = ia + 1; i < bands.Count; i++)
            {
                int hi = bands[i].B - bands[i].A;
                if (i + 1 < bands.Count && hi * 10 < 6 * (bands[i + 1].B - bands[i + 1].A)) continue;
                pay = i;
                break;
            }
            if (pay < 0) { res.Reason = "金额下面没有字行"; return res; }
            int y0 = bands[pay].A, y1 = bands[pay].B, h = y1 - y0;
            res.LineTop = y0; res.LineBottom = y1; res.LineHeight = h;
            if (h < 12) { res.Reason = "字太小"; return res; }

            var colOn = new bool[W];
            for (int x = 0; x < W; x++)
                for (int y = y0; y < y1; y++)
                    if (Rv(y, x) >= InkR) { colOn[x] = true; break; }
            var seg = Runs(colOn);
            if (seg.Count < 5) { res.Reason = "收款方那一行字太少"; return res; }

            // 4. 标签 / 取值
            int k = 0;
            for (int i = 1; i < seg.Count - 1; i++)
                if (seg[i + 1].A - seg[i].B > seg[k + 1].A - seg[k].B) k = i;
            int labX0 = seg[0].A, lw = seg[k].B - seg[0].A;
            int valEnd = seg[seg.Count - 1].B;
            if (labX0 > 0.25 * W || !(2 * h <= lw && lw <= 4 * h) || valEnd < 0.75 * W)
            {
                res.Reason = "这一行不像 收款方 ... 名字"; return res;
            }

            // 5. 每一段的宽、上沿、下沿
            var feats = new List<(int A, int B, int Wd, int T, int Bt)>();
            for (int i = k + 1; i < seg.Count; i++)
            {
                int a = seg[i].A, b = seg[i].B, t = -1, bt = -1;
                for (int y = y0; y < y1; y++)
                {
                    bool any = false;
                    for (int x = a; x < b; x++) if (Rv(y, x) >= InkR) { any = true; break; }
                    if (any) { if (t < 0) t = y - y0; bt = y - y0 + 1; }
                }
                feats.Add((a, b, b - a, t, bt));
            }
            // 星号 = 竖着只占一小截、不在最底下、不太宽也不太扁; 括号 = 窄、比自己宽高得多、至少 0.4 行高。
            // 要"左括号紧跟星号"才算: 小写字母(o a u ...)也像星号, 但前面不是左括号
            bool IsStar((int A, int B, int Wd, int T, int Bt) f)
            {
                int e = f.Bt - f.T;
                return e * 100 <= 45 * h && f.T * 100 < 60 * h && f.Wd * 100 <= 75 * h && f.Wd * 10 >= 4 * e;
            }
            bool IsBracket((int A, int B, int Wd, int T, int Bt) f)
            {
                int e = f.Bt - f.T;
                return f.Wd * 10 <= 4 * h && f.Wd * 10 <= 6 * e && e * 10 >= 4 * h;
            }
            if (!IsBracket(feats[feats.Count - 1])) { res.Reason = "取值最后不是右括号(不是 名字(**某) 的样子)"; return res; }
            bool anyStar = false;
            int star = -1;
            for (int i = 0; i < feats.Count; i++)
            {
                if (!IsStar(feats[i])) continue;
                anyStar = true;
                if (i >= 1 && IsBracket(feats[i - 1])) { star = i; break; }
            }
            if (star < 0) { res.Reason = (anyStar ? "星号前面不是左括号" : "取值里没有星号") + "(不是 名字(**某) 的样子)"; return res; }
            if (star < 2) { res.Reason = "左括号前面没有名字"; return res; }
            var br = feats[star - 1];
            var pv = feats[star - 2];
            res.BracketLeft = br.A; res.NameRight = pv.B;

            // 6. 名字最后一个字
            if (pv.Wd * 10 < 3 * h) { res.Reason = "名字最后一个字太窄(1 l 川 或括号一类), 量不准"; return res; }
            if ((pv.Bt - pv.T) * 10 < 5 * h) { res.Reason = "名字最后一个是标点(。，、一类), 量不准"; return res; }
            int foreign = 0;
            for (int y = y0; y < y1; y++)
                for (int x = pv.A; x < br.A; x++)
                    if (Bv(y, x) < ForeignB) foreign++;
            if (foreign * 40 >= h * h) { res.Reason = "名字后面有表情图标一类的彩色东西, 量不准"; return res; }

            // 7. 空白
            int gap = br.A - pv.B;
            res.Measured = true;
            res.Gap = gap;
            res.Ratio = (double)gap / h;
            res.Verdict = gap * 100 >= GapMinPercent * h ? PayeeSpaceVerdict.Suspicious : PayeeSpaceVerdict.Ok;
            int q = (gap * 200 + h) / (2 * h);      // 比例的百分位, 四舍五入用整数算(Python 一样), 理由里两边逐字相同
            string txt = $"{q / 100}.{q % 100:00}";
            res.Reason = res.Verdict == PayeeSpaceVerdict.Suspicious
                ? $"可疑: 名字和左括号之间空 {gap} 像素 = {txt} 个字高 >= 0.{GapMinPercent:00}(真图一般 0.1~0.35)"
                : $"正常: 名字和左括号之间空 {gap} 像素 = {txt} 个字高";
            return res;
        }
    }
}
