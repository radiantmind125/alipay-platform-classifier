#nullable enable

using System;
using System.Globalization;
using OpenCvSharp;

namespace Ssp
{
    public enum BlurVerdict
    {
        CannotDetermine = 0,   // 判不了(图上干净的强边太少), 走正常流程
        Ok = 1,                // 清楚
        Suspicious = 2,        // 模糊: 建议送人工
    }

    /// <summary>只按原图宽高分的页面类别, 决定用哪个门槛(见 <see cref="BlurCheck.PageKind"/>)。</summary>
    public enum BlurPageKind
    {
        Screen = 0,        // 截图(其它所有尺寸), 门槛 BlurMin
        VoucherImage = 1,  // 支付宝生成的转账凭证图, 本身就虚, 门槛 VoucherBlurMin
        Photo = 2,         // 照片尺寸(拿相机拍的), 门槛和截图一样, 只做标记
    }

    public sealed class BlurResult
    {
        public BlurVerdict Verdict;
        public BlurPageKind Kind;       // 按尺寸分的类别
        public double Threshold;        // 这一类用的门槛
        public int Edges;               // 量到的干净强边条数
        public double EdgeWidth;        // ★ 判据: 强边宽的中位数(像素); 截图 1.3~1.7, 放大 / 拍屏 / 糊过 2.3 以上
        public double CrispFraction;    // 宽 <= 1.5 像素的边占比, 只输出
        public bool Measured;           // false 时上面两项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 凭证模糊检查: 量页面上字、图标、按钮的边有多"软"(2026-11-16, 模糊图片单独处理)。
    ///
    /// 原理
    /// ----
    /// 截图是手机直接渲染的, 边很硬: 从底色到字色 1~2 个像素就过去了。
    /// 图被放大过、拍屏、整张糊过一遍, 边就变软, 要好几个像素才过去。
    /// 只看这一张图自己, 不用知道是苹果还是安卓, 白图蓝图一样算。
    ///
    /// 怎么量
    /// ------
    /// 1. 转灰度(BGR -> GRAY, 和 OpenCV 一样)。
    /// 2. 每一行从左到右、每一列从上到下: 相邻两个像素的差的绝对值 g, 在 g 的局部最大处(>= 左边、> 右边、>= MinStep)
    ///    取边两侧 6 个像素 p0..p5(局部最大那一步在 p2 -> p3 之间)。
    /// 3. 两头落差 C = |p5 - p0| >= MinContrast, 且中间单调(反方向的一步不超过 C 的 MonoTol), 才算一条干净的强边;
    ///    这条边的宽 E = C / g(像素)。一步到位的硬边 E = 1, 越软越大。
    /// 4. 所有边 E 的中位数(偶数条取中间两个的平均)= EdgeWidth。边少于 MinEdges 条判不了。
    /// 5. 按原图宽高分类(PageKind), EdgeWidth >= 这一类的门槛判模糊:
    ///        截图 BlurMin 2.4; 转账凭证图 VoucherBlurMin 3.1; 照片尺寸 BlurMin 2.4(只做标记)。
    ///
    /// 服务器实测(2026-11-16, 11 月真图白图 / 蓝图各随机 3 万张, 量到 29,939 / 29,932 张)
    /// ---------------------------------------------------------------------------------
    ///     截图        白 中位 1.41 / 99% 1.94, >= 2.4 的 30 张(0.10%); 蓝 中位 1.40 / 99% 1.65, 10 张(0.03%)
    ///                 这些是被别的程序放大 / 缩放过的截图(平台收银台 1080x2400 2.96、954x2094 等), 看得清但不是原图
    ///     照片尺寸    白 212 张 / 蓝 160 张, 中位 2.57 / 2.60, >= 2.4 的 143 / 125 张(拿另一部手机拍屏幕)
    ///     转账凭证图  白 565 张(1.9%, 1304x2038 / 1304x2122 / 1252x1954 ...)真图最高 2.88, 0 张 >= 3.1; 蓝图只有 14 张
    ///     合计判模糊  白 0.58%(照片 143 + 截图 30), 蓝 0.45%(照片 125 + 截图 10)
    ///     (只用一个门槛 2.4 的话白图 2.39%, 多出来的全是转账凭证图)
    /// 本机变体(最终代码, 截图 1,869 张 / 转账凭证图 30 张 / 照片 7 张, 判模糊的比例)
    ///     截图        JPEG 60 / 40 重存 1 / 4 张(最高 2.40 / 2.48); 缩小一半、TG 1280 0 张;
    ///                 高斯 0.8 92%, 高斯 1.2 100%, 先缩 0.7 再放回 97%, 缩 0.5 再放回 99.9%
    ///     转账凭证图  原图最高 2.85, JPEG 60 / 40 最高 2.97 / 2.98 -> 0 张;
    ///                 高斯 0.8 80%, 高斯 1.2 97%, 先缩 0.7 再放回 63%, 缩 0.5 再放回 13%(本来就是放大出来的, 再放大变化小)
    /// 已知漏洞
    ///     宽 1100~1310、高宽比 1.45~1.66 的清楚截图(裁过的, 服务器 6 万张里 19 张)也用 3.1, 轻微的糊抓不到。
    ///     转账凭证图再缩小到宽 1100 以下(比如 TG 1280 -> 819x1280)就按截图算, 本机 30 张里 20 张判模糊(2.40~2.60);
    ///     服务器上传的图里这种尺寸 >= 2.4 的 6 万张里只有 1 张, 所以没放宽。
    ///
    /// ★ 传上传的原图, 不要先缩放: 类别按原图宽高认, 缩小会让边变硬、分数变低。
    /// 和 training/blur_check.py 逐步对应(解码成彩色 -> 转灰 -> 量边 -> 中位数 -> 按类别比门槛), 改一边要同步改另一边。
    /// 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class BlurCheck
    {
        public const int MinContrast = 100;     // 边两头至少差这么多灰度(深字白底约 200, 蓝底白字约 150)
        public const double MonoTol = 0.08;     // 中间反方向的一步最多是落差的 8%
        public const int MinStep = 12;          // 局部最大那一步至少这么大
        public const int MinEdges = 200;        // 干净的强边少于这么多条判不了
        public const double BlurMin = 2.4;      // 截图 / 照片: 中位边宽 >= 这个判模糊
        public const double VoucherBlurMin = 3.1;   // 转账凭证图: 本身就虚(真图最高 2.88, JPEG 40 重存最高 2.98)

        /// <summary>
        /// 只按原图宽高分类(和 blur_check.py 的 page_kind 一样):
        ///   转账凭证图 = 宽 1304、高 1950~2450(支付宝生成的转账凭证图),
        ///                或存下来又缩小 / 裁过的: 宽 1100~1310、高宽比 1.45~1.66(手机截图都在 1.77 以上;
        ///                宽 1320 的是苹果 Pro Max 截图裁过的, 是清楚的, 不算);
        ///   照片尺寸   = 长边 >= 2600 且长短边比 &lt; 1.45;
        ///   截图       = 其它。
        /// </summary>
        public static BlurPageKind PageKind(int W, int H)
        {
            if (W <= 0 || H <= 0) return BlurPageKind.Screen;
            double r = (double)H / W;
            if ((W == 1304 && H >= 1950 && H <= 2450) || (W >= 1100 && W <= 1310 && r >= 1.45 && r <= 1.66))
                return BlurPageKind.VoucherImage;
            int lo = Math.Min(W, H), hi = Math.Max(W, H);
            if (hi >= 2600 && (double)hi / lo < 1.45)
                return BlurPageKind.Photo;
            return BlurPageKind.Screen;
        }

        static string KindName(BlurPageKind k) =>
            k == BlurPageKind.VoucherImage ? "转账凭证图" : k == BlurPageKind.Photo ? "照片尺寸" : "截图";

        /// <param name="image">上传的原图(不要先缩放), 8 位, 1 / 3 / 4 通道(GRAY / BGR / BGRA)。
        /// 传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static BlurResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new BlurResult { Verdict = BlurVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn != 1 && cn != 3 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");

            int W = image.Width, H = image.Height;
            res.Kind = PageKind(W, H);
            res.Threshold = res.Kind == BlurPageKind.VoucherImage ? VoucherBlurMin : BlurMin;
            double t = res.Threshold;
            byte[] g = new byte[W * H];
            using (var gray = new Mat())
            {
                if (cn == 3) Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
                else if (cn == 4) Cv2.CvtColor(image, gray, ColorConversionCodes.BGRA2GRAY);
                else image.CopyTo(gray);
                using var cont = gray.IsContinuous() ? gray : gray.Clone();
                System.Runtime.InteropServices.Marshal.Copy(cont.Data, g, 0, g.Length);
            }

            var e = new System.Collections.Generic.List<double>(1 << 16);
            // 沿行(竖边): 第 y 行的第 x 个像素 = g[y * W + x]
            Edges1D(g, H, W, W, 1, e);
            // 沿列(横边): 第 x 列的第 y 个像素 = g[y * W + x]
            Edges1D(g, W, H, 1, W, e);

            res.Edges = e.Count;
            if (e.Count < MinEdges)
            {
                res.Reason = $"干净的强边只有 {e.Count} 条(少于 {MinEdges}), 判不了";
                return res;
            }
            var a = e.ToArray();
            Array.Sort(a);
            int n = a.Length;
            double med = n % 2 == 1 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2.0;
            int crisp = 0;
            foreach (double v in a) if (v <= 1.5) crisp++;
            res.EdgeWidth = med;
            res.CrispFraction = (double)crisp / n;
            res.Measured = true;
            var ci = CultureInfo.InvariantCulture;
            string kn = KindName(res.Kind);
            if (med >= t)
            {
                res.Verdict = BlurVerdict.Suspicious;
                res.Reason = $"模糊: {kn}, 强边宽中位数 {med.ToString("0.00", ci)} 像素 >= {t.ToString("0.0", ci)}(截图一般 1.3~1.7), 边 {n} 条";
            }
            else
            {
                res.Verdict = BlurVerdict.Ok;
                res.Reason = $"清楚: {kn}, 强边宽中位数 {med.ToString("0.00", ci)} 像素 < {t.ToString("0.0", ci)}, 边 {n} 条";
            }
            return res;
        }

        /// <summary>
        /// 沿一个方向找边。lines 条线, 每条 len 个像素; 第 i 条线第 k 个像素 = g[i * lineStep + k * pixStep]。
        /// 和 blur_check.py 的 _edges_1d 一样: d[k] = p[k+1] - p[k]; k 在 [2, len-4] 且 |d[k]| >= |d[k-1]|、> |d[k+1]|、>= MinStep;
        /// 取 p[k-2] .. p[k+3]。
        /// </summary>
        static void Edges1D(byte[] g, int lines, int len, int lineStep, int pixStep, System.Collections.Generic.List<double> outE)
        {
            if (len < 6) return;
            var p = new int[6];
            for (int i = 0; i < lines; i++)
            {
                int b = i * lineStep;
                for (int k = 2; k <= len - 4; k++)
                {
                    int dk = g[b + (k + 1) * pixStep] - g[b + k * pixStep];
                    int ak = Math.Abs(dk);
                    if (ak < MinStep) continue;
                    int am = Math.Abs(g[b + k * pixStep] - g[b + (k - 1) * pixStep]);
                    int ap = Math.Abs(g[b + (k + 2) * pixStep] - g[b + (k + 1) * pixStep]);
                    if (ak < am || ak <= ap) continue;
                    for (int j = 0; j < 6; j++) p[j] = g[b + (k - 2 + j) * pixStep];
                    int C = p[5] - p[0];
                    int Ca = Math.Abs(C);
                    if (Ca < MinContrast) continue;
                    int s = C > 0 ? 1 : -1;
                    int minStep = int.MaxValue;
                    for (int j = 0; j < 5; j++)
                    {
                        int st = (p[j + 1] - p[j]) * s;
                        if (st < minStep) minStep = st;
                    }
                    if ((double)minStep < -MonoTol * Ca) continue;
                    outE.Add((double)Ca / ak);
                }
            }
        }
    }
}
