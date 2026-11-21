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
        Suspicious = 2,        // 名字和左括号之间空一大块, 或者(苹果原图尺寸)右括号离右边远一截(假图)
    }

    public sealed class PayeeSpaceResult
    {
        public PayeeSpaceVerdict Verdict;
        public bool Measured;        // 第 7 步量到了名字和左括号之间的空白, Gap / Ratio 才有意义(第 8 步看 EndGap / EndRule)
        public int Gap;              // 名字最后一个字最右的墨 到 左括号最左的墨, 像素
        public int LineHeight;       // 收款方那一行的行高(有字的行), 像素
        public double Ratio;         // Gap / LineHeight
        public int LineTop, LineBottom;     // 收款方那一行(原图坐标)
        public int NameRight, BracketLeft;  // 量的两个位置(原图 x)
        // 第 8 步(行尾, 2026-11-21 加): 只有走到第 5 步(找到 名字(**某) 的左右括号和星号)才量
        public int EndGap = -1;      // 行最右的墨(右括号右沿)到图右边, 像素; -1 = 没走到第 5 步
        public bool IsIPhoneSize;    // 苹果手机原图截图尺寸(第 8 步只在这些尺寸上用)
        public int EndRule = -1;     // 第 8 步: -1 不适用(没走到第 5 步 / 不是苹果原图尺寸 / 字号不常规), 0 右括号右边不干净不用, 1 用了
        public bool EndHit;          // 第 8 步判中: EndGap * 100 >= EndMaxPercent * 行高
        public string Reason = "";
    }

    /// <summary>
    /// 收款方空白检查(2026-11-17, 经理: 针对这个写个算法 检测这个空白位置;
    /// 2026-11-21 加第 8 步 行尾: 假图在名字后面加个点, 第 6 步判不了, 漏过去 2 张)。
    ///
    /// 页面: 蓝色页头的 转账成功 页(转账成功 / ¥金额 / 收款方 ... 名字(**某) / 付款方式 ... / 白卡片 立即通知收款人)。
    /// 量页头 收款方 那一行: 名字最后一个字 和 左括号 之间空多少, 按这一行的行高算比例。
    /// OCR 读出来有时带空格有时不带, 所以按像素量。只看图, 不用 OCR, 不用知道是什么手机。
    ///
    ///     经理给的 3 张假图(1320x2868)            0.81~0.83(空 34~35 像素, 行高 42)
    ///     经理截的图(780x1280, 压过)              0.82
    ///     假图 JPEG 90~40 / 缩 0.75 0.5 / TG 1280  0.81~0.90, 都判可疑
    ///     服务器 11 月蓝图全部 95,810 张(2026-11-17), 量到约 2 万张, 其余不是这个页面或者是下面判不了的几种:
    ///         苹果原生截图 0.29~0.34 居多(名字和括号之间本来就空半个字), 安卓 0.02~0.25,
    ///         最后一个字是 月 日 一类(右边留白多)、拉丁字母、＆ 0.45~0.50;
    ///         &gt;= 0.65 只有 1 张: 昵称自己带了几个空格(晨SIR   (**英), 1.19), 这种分不出来。
    ///     服务器 11 月白图随机 3 万张: 0 张量到(没有蓝色页头)。本机真图 3,302 张整页(7~11 月): 0 张可疑。
    /// Gap / 行高 &gt;= GapMinPercent / 100(0.65, 真图最高 0.50 和假图 0.81 中间)判可疑。
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
    ///  6. 名字最后一个字判不了的几种(宁可不判):
    ///     宽 &lt; 0.3 行高(1 l 川 的一竖、括号)、高 &lt; 0.5 行高(。，、);
    ///     ？ ！ 收尾: 竖着两截以上、最下面一截是小点(高 &lt;= 0.25 行高、宽 &lt;= 0.4 行高), 全角 ？ ！ 右边本来就空一大块
    ///       (服务器 11 月蓝图上这条挡掉 26 张; 没有它按 0.65 会误判 15 张, 0.65~0.71);
    ///     从它到左括号之间 B &lt; 120 的像素 &gt;= 行高平方 / 40(深色、彩色表情);
    ///     名字和括号之间(两边各让 max(2, 行高/10) 像素)和底色差 &gt; 40 的像素 &gt;= 行高平方 / 40(蓝色、灰色表情 💤 👤);
    ///       底色 = 标签和取值之间那一大段空白每个通道的下中位数
    ///       (服务器上这条挡掉 11 张; 没有它按 0.65 会误判 5 张, 1.37~1.70)。
    ///  7. Gap = 左括号左沿 - 名字右沿; Gap * 100 &gt;= GapMinPercent * 行高 判可疑(整数比较)。
    ///  8. 行尾(2026-11-21 加; 走到第 5 步就量, 第 6、7 步出结果以后再用, 第 7 步已经可疑的不改):
    ///     只在苹果原图截图尺寸 + 常规字号(行高 / 倍数 12.5~17.5: 3 倍屏 38~52 像素, 2 倍屏 25~35)时用。
    ///     EndGap = 图宽 - 这一行最右的墨(右括号右沿); EndGap * 100 &gt;= EndMaxPercent * 行高 判可疑(第 6 步判不了的也一样)。
    ///     右括号右边到图右边(最右 2% 不看)和底色差 &gt; 40 的像素 &gt;= 行高平方 / 40 的, 这条不用(右边盖着浮窗、灰条)。
    ///     为什么: 苹果原图排版是死的, 真图页头是半角括号 名字(**某), 右括号紧贴右边距;
    ///     假图模板用全角括号 名字（**某）, 全角右括号自己右边带一大块空, 整行离右边远一截。名字后面放什么(点、句号、一竖)都一样。
    ///
    /// 实测第 8 步(2026-11-21):
    ///     漏过的 2 张(11-20, 1320x2868): 消失的.（**涛）、L.（*旺）, 和 11-17 的 3 张同一个假图模板; 名字后面加了个点,
    ///       点 -&gt; 左括号 0.76 / 0.69 个字高, 第 6 步判不了 -&gt; 0。不能改成从点量起: 真图昵称是一个句号的 。(**燕) 也是 0.80。
    ///     服务器 11 月蓝图全部 95,810 张(11-01..04, 用这一版跑): 第 8 步改判 0 张, 可疑还是只有 1 张(晨SIR 自带空格, 第 7 步判的)。
    ///       苹果尺寸用上第 8 步的 14,235 张 行尾 1.490~2.050 个字高(中位 1.878, 99.9% 1.925, &gt;= 2.1 的 0 张; 3 倍屏 71~83 像素);
    ///       右边不干净不用的 35 张(悬浮球、浮窗)行尾都 &lt;= 1.925。白图随机 2 万张 0 张量到。
    ///       苹果尺寸第 6 步判不了的 1,399 张(昵称以 。，、！？ 一竖 表情收尾)要是从最后一个点量起按 0.65 判会误判 296 张, 第 8 步 0 张。
    ///     本机苹果真图(含 。 窄字 表情收尾)1.82~1.93; 11-17 的 3 张假图 2.71(114 像素); 11-21 两张约 2.6~2.7。
    ///     合成: 3 张假图名字后面加 点 / 句号 / 一竖 18 张: 旧版全是 0, 新版全判可疑(JPEG 95~40 也是);
    ///       真图 307 张加同样的东西 1,540 张(含 JPEG / 缩放): 0 张可疑。11-17 对拍的 108,074 张图: 新旧结果和理由逐字一样。
    ///     缩放过的图不是苹果原图尺寸, 第 8 步不用(加点的假图缩放后又是 0), 所以一定传上传的原图。
    ///
    /// ★ 传上传的原图(不要先缩放、不要先转灰度): 要靠颜色分蓝底和白字。缩放过比例基本不变, 但字太小(行高 &lt; 12)判不了。
    /// 只读入参, 无静态可变状态, 可多线程调用。从文件读图请用 ImreadModes.Color。
    /// </summary>
    public static class PayeeSpaceCheck
    {
        /// <summary>Gap / 行高 &gt;= 这个百分比判可疑。</summary>
        public const int GapMinPercent = 65;

        /// <summary>第 8 步(苹果原图尺寸): 右括号离图右边 / 行高 &gt;= 这个百分比判可疑。</summary>
        public const int EndMaxPercent = 230;

        const int InkR = 150;          // 页头白字: R >= 150
        const int ForeignB = 120;      // 页头里 B < 120: 不是白字也不是蓝底(JPEG 在白字边上把蓝底的 B 压到 140~160)
        const int NonBgDist = 40;      // 空白里和底色差(三通道差的绝对值之和) > 40 的像素算"有东西"

        // 苹果手机原图截图尺寸(和 YenCheck / AmountYCheck 一样): 这些尺寸的排版是死的, 收款方那一行的右端每张都在同一个位置
        static readonly HashSet<(int W, int H)> IPhoneSizes = new HashSet<(int W, int H)>
        {
            (640, 1136), (750, 1334), (1242, 2208), (828, 1792), (1125, 2436), (1242, 2688), (1170, 2532),
            (1284, 2778), (1179, 2556), (1290, 2796), (1206, 2622), (1320, 2868), (1260, 2736),
        };

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

            // 底色 = 标签和取值之间那一大段空白每个通道的下中位数(第 8 步和第 6 步都用)
            int[] LowMedBg()
            {
                int ga = seg[k].B, gb = seg[k + 1].A;
                var bgc = new int[3];
                for (int ch = 0; ch < 3; ch++)
                {
                    var hist = new int[256];
                    for (int y = y0; y < y1; y++)
                        for (int x = ga; x < gb; x++) hist[px[((long)y * W + x) * 3 + ch]]++;
                    int need = ((y1 - y0) * (gb - ga) + 1) / 2, acc = 0, v = 0;
                    for (; v < 256; v++) { acc += hist[v]; if (acc >= need) break; }
                    bgc[ch] = v;
                }
                return bgc;
            }
            // 和底色差 > NonBgDist 的像素个数, 区域 行 [ra, rz) x 列 [ca, cz)
            int CountNonBg(int[] bgc, int ra, int rz, int ca, int cz)
            {
                int c = 0;
                for (int y = ra; y < rz; y++)
                    for (int x = ca; x < cz; x++)
                    {
                        long o = ((long)y * W + x) * 3;
                        int d = Math.Abs(px[o] - bgc[0]) + Math.Abs(px[o + 1] - bgc[1]) + Math.Abs(px[o + 2] - bgc[2]);
                        if (d > NonBgDist) c++;
                    }
                return c;
            }

            // 8(先量, 第 6、7 步算完再用). 行尾: 苹果原图尺寸、常规字号(3 倍屏行高 38~52, 2 倍屏 25~35)时,
            //    右括号右沿离图右边 >= 2.30 个行高判可疑。右括号右边到图右边(最右 2% 不看, 有的图带一两像素黑边)要是干净的底色,
            //    不干净(右边盖着灰条、浮窗)这条不用。
            res.EndGap = W - valEnd;
            res.IsIPhoneSize = IPhoneSizes.Contains((W, H));
            int sc = W < 1000 ? 2 : 3;                 // 2 倍屏: 640 / 750 / 828 宽; 别的都是 3 倍屏
            if (res.IsIPhoneSize && 25 * sc <= 2 * h && 2 * h <= 35 * sc)
            {
                int me = Math.Max(2, h / 10);
                int xa8 = valEnd + me, xb8 = W - W / 50;
                bool clean = true;
                if (xb8 > xa8) clean = CountNonBg(LowMedBg(), y0, y1, xa8, xb8) * 40 < h * h;
                res.EndRule = clean ? 1 : 0;
                res.EndHit = clean && res.EndGap * 100 >= EndMaxPercent * h;
            }

            // 6. 名字最后一个字
            if (pv.Wd * 10 < 3 * h) { res.Reason = "名字最后一个字太窄(1 l 川 或括号一类), 量不准"; return EndRule8(res, h); }
            if ((pv.Bt - pv.T) * 10 < 5 * h) { res.Reason = "名字最后一个是标点(。，、一类), 量不准"; return EndRule8(res, h); }
            // ？ ！ 收尾: 竖着分成两截以上, 最下面一截是个小点(高 <= 0.25 行高、宽 <= 0.4 行高); 全角的 ？ ！ 右边本来就空一大块
            var pvRow = new bool[h];
            for (int y = y0; y < y1; y++)
                for (int x = pv.A; x < pv.B; x++)
                    if (Rv(y, x) >= InkR) { pvRow[y - y0] = true; break; }
            var rr = Runs(pvRow);
            if (rr.Count >= 2)
            {
                var (la, lb) = rr[rr.Count - 1];
                int c0 = -1, c1 = -1;
                for (int x = pv.A; x < pv.B; x++)
                {
                    bool any = false;
                    for (int y = y0 + la; y < y0 + lb; y++) if (Rv(y, x) >= InkR) { any = true; break; }
                    if (any) { if (c0 < 0) c0 = x; c1 = x; }
                }
                if ((lb - la) * 4 <= h && (c1 - c0 + 1) * 10 <= 4 * h)
                {
                    res.Reason = "名字最后一个是 ？ ！ 一类(下面一个点), 量不准"; return EndRule8(res, h);
                }
            }
            int foreign = 0;
            for (int y = y0; y < y1; y++)
                for (int x = pv.A; x < br.A; x++)
                    if (Bv(y, x) < ForeignB) foreign++;
            if (foreign * 40 >= h * h) { res.Reason = "名字后面有表情图标一类的彩色东西, 量不准"; return EndRule8(res, h); }
            // 空白里有没有别的东西(蓝色 / 灰色的表情, R 不够白字、B 又不低): 和底色比; 底色 = 标签和取值之间那一大段空白的下中位数
            int m = Math.Max(2, h / 10);
            if (br.A - m > pv.B + m)
            {
                if (CountNonBg(LowMedBg(), y0, y1, pv.B + m, br.A - m) * 40 >= h * h)
                {
                    res.Reason = "名字和括号之间有表情一类的东西, 量不准"; return EndRule8(res, h);
                }
            }

            // 7. 空白
            int gap = br.A - pv.B;
            res.Measured = true;
            res.Gap = gap;
            res.Ratio = (double)gap / h;
            res.Verdict = gap * 100 >= GapMinPercent * h ? PayeeSpaceVerdict.Suspicious : PayeeSpaceVerdict.Ok;
            int q = (gap * 200 + h) / (2 * h);      // 比例的百分位, 四舍五入用整数算(Python 一样), 理由里两边逐字相同
            string txt = $"{q / 100}.{q % 100:00}";
            res.Reason = res.Verdict == PayeeSpaceVerdict.Suspicious
                ? $"可疑: 名字和左括号之间空 {gap} 像素 = {txt} 个字高 >= 0.{GapMinPercent:00}(真图一般 0.1~0.5)"
                : $"正常: 名字和左括号之间空 {gap} 像素 = {txt} 个字高";
            return EndRule8(res, h);
        }

        /// <summary>第 6、7 步出了结果(判不了 / 正常 / 可疑)以后: 第 8 步判中、而第 7 步还没判可疑 -> 改判可疑, 原来的理由接在后面。</summary>
        static PayeeSpaceResult EndRule8(PayeeSpaceResult res, int h)
        {
            if (res.EndHit && res.Verdict != PayeeSpaceVerdict.Suspicious)
            {
                int e = res.EndGap;
                int q = (e * 200 + h) / (2 * h);
                res.Verdict = PayeeSpaceVerdict.Suspicious;
                res.Reason = $"可疑: 右括号离右边 {e} 像素 = {q / 100}.{q % 100:00} 个字高 >= {EndMaxPercent / 100}.{EndMaxPercent % 100:00}"
                    + $"(苹果原图尺寸真图 1.8~2.05, 假图全角括号 2.6 以上); 空白那条: {res.Reason}";
            }
            return res;
        }
    }
}
