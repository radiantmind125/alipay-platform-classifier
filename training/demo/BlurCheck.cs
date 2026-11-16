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

    public sealed class BlurResult
    {
        public BlurVerdict Verdict;
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
    /// 5. EdgeWidth >= BlurMin 判模糊。
    ///
    /// 实测(本机真图 1,886 张: 白图 920 + 蓝图 966; 另 900 张各做 10 种变体)
    /// ----------------------------------------------------------------------
    ///     原图                 白 中位 1.52 / 99% 2.14     蓝 中位 1.45 / 99% 2.04
    ///     JPEG 60 / 40 重存     白中位 1.84 / 2.01
    ///     缩小一半 / TG 1280    1.57 / 1.66(缩小不会变软)
    ///     先缩 0.7 / 0.5 再放回 最低 2.27 / 2.36
    ///     高斯模糊 0.8 / 1.2 / 1.6   最低 2.04 / 2.84 / 3.31
    ///   真图里 >= 2.4 的约 0.7%: 拿另一部手机拍屏幕的照片(3024x4032 等, 2.4~3.0),
    ///   和一种"转账成功 / 转出成功"凭证图(带拼音字体和支付宝水印, 宽 1112~1304, 2.55~2.78, 看得清但发虚)。
    ///   门槛 BlurMin 暂定 2.4, 按服务器上白图 / 蓝图的分布再定。
    ///
    /// 和 training/blur_check.py 逐步对应(解码成彩色 -> 转灰 -> 量边 -> 中位数), 改一边要同步改另一边。
    /// 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class BlurCheck
    {
        public const int MinContrast = 100;     // 边两头至少差这么多灰度(深字白底约 200, 蓝底白字约 150)
        public const double MonoTol = 0.08;     // 中间反方向的一步最多是落差的 8%
        public const int MinStep = 12;          // 局部最大那一步至少这么大
        public const int MinEdges = 200;        // 干净的强边少于这么多条判不了
        public const double BlurMin = 2.4;      // 中位边宽 >= 这个判模糊(暂定)

        /// <param name="image">8 位图, 1 / 3 / 4 通道(GRAY / BGR / BGRA)。传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static BlurResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new BlurResult { Verdict = BlurVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn != 1 && cn != 3 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");

            int W = image.Width, H = image.Height;
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
            if (med >= BlurMin)
            {
                res.Verdict = BlurVerdict.Suspicious;
                res.Reason = $"模糊: 强边宽中位数 {med.ToString("0.00", ci)} 像素 >= {BlurMin.ToString("0.0", ci)}(截图一般 1.3~1.7), 边 {n} 条";
            }
            else
            {
                res.Verdict = BlurVerdict.Ok;
                res.Reason = $"清楚: 强边宽中位数 {med.ToString("0.00", ci)} 像素 < {BlurMin.ToString("0.0", ci)}, 边 {n} 条";
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
