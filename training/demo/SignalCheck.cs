#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum SignalVerdict
    {
        CannotDetermine = 0,   // 判不了(深色模式、不是苹果原尺寸、没找到信号格、少见的样子), 走正常流程
        Ok = 1,                // 苹果原尺寸, 信号格是真图的样子
        Suspicious = 2,        // 状态栏和已知造假工具画的一样
    }

    public sealed class SignalResult
    {
        public SignalVerdict Verdict;
        public bool BarsFound;        // 找到了 4 格信号
        /// <summary>签名: 宽|4 格的高|3 个间距|下面一排的点数(双卡)。4 格宽度不一样时, 宽前面带 ~。</summary>
        public string Signature = "";
        public int[] Widths = Array.Empty<int>();
        public int[] Heights = Array.Empty<int>();
        public int[] Gaps = Array.Empty<int>();
        public int Dots;              // 信号格下面那排小方点的个数(苹果双卡是 4)
        public int BarLeft;           // 第一格左缘 x(整图坐标)
        public int BarBottom;         // 信号格底边 y
        public bool DarkMode;
        public bool IphoneSize;       // 尺寸是苹果原尺寸截图之一
        public string Reason = "";
    }

    /// <summary>
    /// 状态栏信号格检查(按颜色像素找, 不靠图片定位)。
    ///
    /// 苹果的信号格是系统按点数画的, 同一尺寸、同一系统版本上每一格的像素宽高和间距是**固定值**,
    /// 比如 1179x2556 单卡是 宽 10、高 14 21 29 37、间距 6; 双卡上排 高 12 15 20 25, 下面多一排 4 个灰点。
    ///
    /// 经理给的 116 假图(和 111、112 的假图是同一个状态栏, 1080x2400)的信号格是自己画的:
    /// 宽 8~9、高 13 20 28 35、间距 5~6 —— 高度比例像苹果, 但格子细(宽/最高 0.23, 苹果 0.27)、
    /// 间距大, 第一格比别的宽一个像素。同一个工具每张都画在同一个位置, 这里按模板认出来。
    ///
    /// 服务器 2026-09-01~16 抽的 101,380 张实测:
    ///     判可疑 2 张, 都是这个工具画的, 111 和 112 也都判了可疑;
    ///     苹果原尺寸、信号格是真图样子的 42,980 张判 Ok。
    ///
    /// ★ 苹果真图的信号格不止一种样子(系统版本、双卡、没信号), 白名单是按服务器数据定的
    ///   (每个尺寸出现 >= 10 张、跨 >= 5 天), 覆盖苹果尺寸量到的 99.2%。只拿一个模板比的话,
    ///   苹果真图会误判 7% 左右(比如 1170 宽有 1,709 张是 宽 9 间距 5 的老系统样子)。
    /// ★ 白名单外的少见样子不判可疑, 只给 CannotDetermine: 服务器上这 351 张和 111/112 判出的假图
    ///   一张都不重合。苹果尺寸上的假图都是直接拿真图的状态栏(或者开飞行模式没有信号格), 这一项抓不到它们。
    ///
    /// 判"不是彩色"用 三通道最大减最小(绝对色度) &lt; 40, 不用 HSV 饱和度:
    /// 接近黑色的像素饱和度很不稳, JPEG 里 (5,3,8) 这种黑点饱和度能到 95, 会把黑色信号格切碎。
    ///
    /// 和 training/signal_scan.py 逐步对应, 改一边要同步改另一边。
    /// 只读入参, 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class SignalCheck
    {
        // 真图白名单: 苹果原尺寸 -> 签名集合(服务器数据定的, 见 signal_scan.py 的注释)
        static readonly string[] A = { "10|14,21,29,37|6,6,6|0", "10|12,15,20,25|6,6,6|4", "10|10,10,10,10|6,6,6|0", "10|10,10,10,10|6,6,6|4" };
        static readonly string[] A2 = { "9|12,14,19,24|6,6,6|4", "9|13,20,27,35|6,6,6|0", "10|14,17,21,25|6,6,6|4" };
        static readonly string[] B = { "11|16,23,32,41|7,7,7|0", "11|13,17,22,27|7,7,7|4", "11|11,11,11,11|7,7,7|0", "11|11,11,11,11|7,7,7|4" };

        public static readonly IReadOnlyDictionary<(int W, int H), HashSet<string>> Genuine =
            new Dictionary<(int, int), HashSet<string>>
            {
                [(1170, 2532)] = Set(A, new[] { "9|12,18,26,34|5,5,5|0", "9|11,14,18,22|5,5,5|4", "9|9,9,9,9|5,5,5|0" }),
                [(1179, 2556)] = Set(A, A2),
                [(1206, 2622)] = Set(A, A2),
                [(1284, 2778)] = Set(A, B),
                [(1290, 2796)] = Set(B, new[] { "10|15,22,30,39|7,7,7|0", "10|12,16,21,26|7,7,7|4", "11|21,27,35,43|7,7,7|0" }),
                [(1320, 2868)] = Set(B, new[] { "11|21,27,35,43|7,7,7|0", "11|14,18,23,28|7,7,7|4" }),
                [(1260, 2736)] = Set(new[] { "11|16,23,32,41|7,7,7|0", "11|13,17,22,27|7,7,7|4" }),   // iPhone Air
                [(1125, 2436)] = Set(new[] { "9|12,18,25,32|5,5,5|0", "10|14,21,29,37|6,6,6|0", "9|11,16,24,32|5,5,5|0",
                                             "9|9,9,9,9|5,5,5|0", "8|10,13,17,21|5,5,5|4" }),
                [(1242, 2688)] = Set(new[] { "10|13,19,27,35|5,5,5|0", "9|11,14,18,22|5,5,5|4", "10|10,10,10,10|5,5,5|0" }),
                [(828, 1792)] = Set(new[] { "7|9,13,18,23|3,3,3|0", "6|8,10,13,16|3,3,3|4", "7|7,7,7,7|3,3,3|0" }),
            };

        /// <summary>已知造假工具画的状态栏。位置、大小差 1 个像素以内算同一个(JPEG 压过会差一点)。</summary>
        public sealed class Template
        {
            public int W, H, Left, Bottom;
            public int[] Heights = Array.Empty<int>();
            public int[] Widths = Array.Empty<int>();
            public int[] Gaps = Array.Empty<int>();
        }

        public static readonly IReadOnlyList<Template> FakeTemplates = new[]
        {
            // 经理给的 116/111/112 三张 false.jpg, 服务器上 111/112 判出的 S_002、S_013 也是它
            new Template { W = 1080, H = 2400, Left = 793, Bottom = 80,
                           Heights = new[] { 13, 20, 28, 35 }, Widths = new[] { 8, 9 }, Gaps = new[] { 5, 6 } },
        };

        static HashSet<string> Set(params string[][] parts) => new(parts.SelectMany(p => p));

        /// <param name="image">8 位彩色图, 3 / 4 通道(BGR / BGRA)。传进来的 Mat 不会被修改, 也不会被释放。</param>
        public static SignalResult Check(Mat image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new SignalResult { Verdict = SignalVerdict.CannotDetermine };
            if (image.Empty()) { res.Reason = "图为空"; return res; }
            if (image.Depth() != MatType.CV_8U) throw new ArgumentException("只支持 8 位图");
            int cn = image.Channels();
            if (cn == 1) { res.Reason = "单通道图, 要靠颜色判, 不判"; return res; }
            if (cn != 3 && cn != 4) throw new ArgumentException($"不支持 {cn} 通道");
            int W = image.Width, H = image.Height;
            if (W < 500 || H < 900) { res.Reason = "图太小"; return res; }
            res.IphoneSize = Genuine.ContainsKey((W, H));

            if (cn == 3) return Run(image, res);
            using var bgr = new Mat();
            Cv2.CvtColor(image, bgr, ColorConversionCodes.BGRA2BGR);
            return Run(bgr, res);
        }

        static SignalResult Run(Mat bgr, SignalResult res)
        {
            int W = bgr.Width, H = bgr.Height;
            bool found = FindBars(bgr, res);
            if (res.DarkMode) { res.Reason = "深色模式"; return res; }
            res.BarsFound = found;

            if (found && MatchTemplate(W, H, res))
            {
                res.Verdict = SignalVerdict.Suspicious;
                res.Reason = $"状态栏和已知造假工具画的一样(信号格 {res.Signature})";
                return res;
            }
            if (Genuine.TryGetValue((W, H), out var ok))
            {
                if (!found) { res.Reason = "苹果尺寸, 没找到信号格(飞行模式、灵动岛展开等)"; return res; }
                if (ok.Contains(res.Signature))
                {
                    res.Verdict = SignalVerdict.Ok;
                    res.Reason = $"信号格是这个尺寸真图的样子({res.Signature})";
                    return res;
                }
                res.Reason = $"苹果尺寸, 信号格样子少见({res.Signature}), 不判";
                return res;
            }
            res.Reason = "不是苹果原尺寸";
            return res;
        }

        static bool MatchTemplate(int W, int H, SignalResult r)
        {
            foreach (var t in FakeTemplates)
            {
                if (W != t.W || H != t.H) continue;
                if (Math.Abs(r.BarLeft - t.Left) > 3 || Math.Abs(r.BarBottom - t.Bottom) > 2) continue;
                bool hOk = r.Heights.Zip(t.Heights, (h, th) => Math.Abs(h - th) <= 1).All(x => x);
                bool wOk = r.Widths.All(w => t.Widths.Contains(w));
                bool gOk = r.Gaps.All(g => t.Gaps.Contains(g));
                if (hOk && wOk && gOk) return true;
            }
            return false;
        }

        struct Comp
        {
            public int X, Y, W, H, Area;
            public int Bottom => Y + H;
        }

        /// <summary>状态栏(高度 6.5% 以内、宽度 55% 往右)里的 4 格信号。找到返回 true 并填好 r。</summary>
        static bool FindBars(Mat bgr, SignalResult r)
        {
            int W = bgr.Width, H = bgr.Height;
            int y1 = (int)(0.065 * H), x0 = (int)(0.55 * W);
            using var band = new Mat(bgr, new Rect(x0, 0, W - x0, y1));
            int bw = band.Width, bh = band.Height;
            using var gray = new Mat();
            Cv2.CvtColor(band, gray, ColorConversionCodes.BGR2GRAY);

            var g = new byte[bw * bh];
            var mask = new byte[bw * bh];
            var hist = new int[256];
            var gi = gray.GetGenericIndexer<byte>();
            var ci = band.GetGenericIndexer<Vec3b>();
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < bw; x++)
                {
                    byte v = gi[y, x];
                    g[y * bw + x] = v;
                    hist[v]++;
                }
            int bg = MedianFloor(hist, bw * bh);
            if (bg < 128) { r.DarkMode = true; return false; }
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < bw; x++)
                {
                    var p = ci[y, x];
                    int mx = Math.Max(p.Item0, Math.Max(p.Item1, p.Item2));
                    int mn = Math.Min(p.Item0, Math.Min(p.Item1, p.Item2));
                    if (g[y * bw + x] < bg - 30 && mx - mn < 40) mask[y * bw + x] = 255;
                }

            using var m = new Mat(bh, bw, MatType.CV_8UC1);
            System.Runtime.InteropServices.Marshal.Copy(mask, 0, m.Data, mask.Length);
            using var labels = new Mat();
            using var stats = new Mat();
            using var cent = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(m, labels, stats, cent, PixelConnectivity.Connectivity8, MatType.CV_32S);
            var cs = new List<Comp>();
            for (int i = 1; i < n; i++)
            {
                var c = new Comp
                {
                    X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                    Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                    W = stats.At<int>(i, (int)ConnectedComponentsTypes.Width),
                    H = stats.At<int>(i, (int)ConnectedComponentsTypes.Height),
                    Area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area),
                };
                if (c.Area >= 6) cs.Add(c);
            }

            // 信号格: 底边对齐(+-2)、宽度相近(+-2)、竖着的(高 >= 0.8 宽)、间距均匀的一排, 从左到右不变矮, 取最左边的那组
            List<int>? best = null;
            for (int ic = 0; ic < cs.Count; ic++)
            {
                var c = cs[ic];
                var grp = Enumerable.Range(0, cs.Count)
                    .Where(j => Math.Abs(cs[j].Bottom - c.Bottom) <= 2 && Math.Abs(cs[j].W - c.W) <= 2
                                && cs[j].H >= 0.8 * cs[j].W && cs[j].W >= 3)
                    .OrderBy(j => cs[j].X).ToList();
                int at = grp.IndexOf(ic);
                if (at < 0) continue;
                var run = new List<int> { ic };
                for (int k = at + 1; k < grp.Count; k++)
                {
                    var d = cs[grp[k]];
                    var last = cs[run[^1]];
                    int gap = d.X - (last.X + last.W);
                    if (gap < 0 || gap > 1.5 * c.W) break;
                    if (run.Count >= 2)
                    {
                        int g0 = cs[run[1]].X - (cs[run[0]].X + cs[run[0]].W);
                        if (Math.Abs(gap - g0) > 2) break;
                    }
                    run.Add(grp[k]);
                    if (run.Count == 4) break;
                }
                if (run.Count != 4) continue;
                bool up = true;
                for (int k = 0; k < 3; k++) if (cs[run[k]].H > cs[run[k + 1]].H) up = false;
                if (!up) continue;
                if (best == null || cs[run[0]].X < cs[best[0]].X) best = run;
            }
            if (best == null) return false;

            var bars = best.Select(j => cs[j]).ToArray();
            int[] widths = bars.Select(b => b.W).ToArray();
            int[] heights = bars.Select(b => b.H).ToArray();
            int[] gaps = Enumerable.Range(0, 3).Select(k => bars[k + 1].X - (bars[k].X + bars[k].W)).ToArray();
            var ws = widths.OrderBy(v => v).ToArray();
            int w = (int)((ws[1] + ws[2]) / 2.0);                 // numpy 的中位数: 偶数个取中间两个的平均, 再取整
            int bottom = bars[0].Bottom;
            int left = bars[0].X, right = bars[3].X + bars[3].W;
            int dots = cs.Count(d => d.Y > bottom && d.Y - bottom <= 2 * w && Math.Abs(d.W - d.H) <= 2
                                     && left - 2 <= d.X && d.X <= right + 2);

            r.Widths = widths;
            r.Heights = heights;
            r.Gaps = gaps;
            r.Dots = dots;
            r.BarLeft = x0 + left;
            r.BarBottom = bottom;
            string wPart = widths.Max() - widths.Min() <= 0 ? w.ToString() : "~" + w;
            r.Signature = $"{wPart}|{string.Join(",", heights)}|{string.Join(",", gaps)}|{dots}";
            return true;
        }

        /// <summary>和 int(np.median(x)) 一样: 偶数个取中间两个的平均再往下取整。</summary>
        static int MedianFloor(int[] hist, int count)
        {
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
