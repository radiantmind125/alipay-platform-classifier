#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssp
{
    public enum MinusVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // 负号偏长
    }

    /// <summary>像素在字节数组里的通道顺序。</summary>
    public enum PixelOrder
    {
        Rgb = 0,
        Bgr = 1,   // System.Drawing 的 Format24bppRgb 实际是 BGR
    }

    public sealed class MinusResult
    {
        public MinusVerdict Verdict;
        public double BarWidth;      // 负号宽 / 数字中位宽
        public double DigitHeight;   // 数字中位高
        public double DigitAspect;   // 数字中位宽 / 中位高
        public int DigitCount;
        public bool Measured;        // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 白底账单详情页的负号几何检查: 量负号宽度与数字中位宽度的比值。
    /// 无外部依赖, 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class MinusCheck
    {
        public const double Threshold = 0.78;
        public const double MinDigitHeight = 60;   // 数字低于此高度不判定; 设为 0 可关闭
        const int GrayDark = 140;                  // 定位金额行时的深色阈值

        public const int MaxComponents = 20_000;   // 上限保护, 见 LocateAmount

        struct Comp
        {
            public int X, Y, W, H, Area;
        }

        /// <summary>
        /// 每像素 3 字节, 第 y 行第 x 列的首字节位于 y*stride + x*3。
        /// </summary>
        /// <param name="order">通道顺序; System.Drawing 解出的数据传 Bgr。</param>
        /// <param name="stride">
        /// 每行字节数, 传 0 表示 width*3。GDI 的 BitmapData.Stride 会补齐到 4 的倍数,
        /// 需如实传入, 否则逐行错位且不会抛异常。
        /// </param>
        public static MinusResult Check(byte[] pixels, int width, int height,
                                        PixelOrder order = PixelOrder.Rgb, int stride = 0)
        {
            if (pixels == null) throw new ArgumentNullException(nameof(pixels));
            if (width <= 0 || height <= 0)
                throw new ArgumentException("宽高必须为正");
            if (stride == 0) stride = width * 3;
            if (stride < width * 3)
                throw new ArgumentException(
                    $"stride {stride} 小于一行需要的 {width * 3} 字节");
            long need = (long)(height - 1) * stride + (long)width * 3;
            if (pixels.Length < need)
                throw new ArgumentException(
                    $"像素数组长度 {pixels.Length} 不够, 按 stride={stride} 需要 {need} 字节");

            var res = new MinusResult { Verdict = MinusVerdict.CannotDetermine };
            if (width < 16 || height < 16) { res.Reason = "图太小"; return res; }

            int ri = order == PixelOrder.Rgb ? 0 : 2;
            int bi = order == PixelOrder.Rgb ? 2 : 0;

            // 蓝底转账页金额不带符号, 不处理
            if (IsBluePage(pixels, width, height, stride, ri, bi))
            { res.Reason = "蓝底转账页, 金额不带负号"; return res; }

            var gray = ToGray(pixels, width, height, stride, ri, bi);
            return Check(gray, width, height, res);
        }

        static MinusResult Check(byte[,] gray, int W, int H, MinusResult res)
        {
            var box = LocateAmount(gray, W, H);
            if (box == null) { res.Reason = "定位不到金额行"; return res; }
            int bx0 = box[0], by0 = box[1], bx1 = box[2], by1 = box[3];

            // 行内重新取块, 不设高度下限: 定位阶段的高度过滤会把负号滤掉。
            // 左侧外扩较多, 因为定位框只覆盖数字, 负号落在框外。
            int pad = Math.Max(2, (int)((by1 - by0) * 0.15));
            int padx = Math.Max(pad, (int)((bx1 - bx0) * 0.30));
            int cx0 = Math.Max(0, bx0 - padx), cy0 = Math.Max(0, by0 - pad);
            int cx1 = Math.Min(W, bx1 + pad), cy1 = Math.Min(H, by1 + pad);
            int cw = cx1 - cx0, ch = cy1 - cy0;
            if (cw < 8 || ch < 8) { res.Reason = "金额行裁出来太小"; return res; }

            var sub = new byte[ch, cw];
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                    sub[y, x] = gray[cy0 + y, cx0 + x];

            int otsu = Otsu(sub, ch, cw);
            var fg = new bool[ch, cw];
            long on = 0;
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                { fg[y, x] = sub[y, x] <= otsu; if (fg[y, x]) on++; }
            // 前景过半说明极性相反, 取反; 阈值为 127/255
            if (255L * on > 127L * ch * cw)
                for (int y = 0; y < ch; y++)
                    for (int x = 0; x < cw; x++) fg[y, x] = !fg[y, x];

            var glyphs = Label(fg, ch, cw, minArea: 6);
            if (glyphs.Count < 4) { res.Reason = "块数不够"; return res; }

            var hs = glyphs.Select(g => g.H).OrderBy(v => v).ToList();
            double medH = hs[hs.Count / 2];
            if (medH < 20) { res.Reason = "字太小"; return res; }

            var digits = glyphs.Where(g => g.H > 0.75 * medH && g.W < 1.5 * g.H).ToList();
            var bars = glyphs.Where(g => g.W >= 1.5 * g.H && g.H <= 0.45 * medH).ToList();
            if (digits.Count < 4) { res.Reason = "数字不够 4 个"; return res; }
            if (bars.Count == 0) { res.Reason = "没有负号"; return res; }

            // 有效性检查: 金额行的数字高度基本一致, 二维码等区域不满足
            double dhMean = digits.Average(g => (double)g.H);
            double dhStd = Std(digits.Select(g => (double)g.H));
            if (dhStd / dhMean > 0.08) { res.Reason = "数字高度不齐, 不像一行数字"; return res; }
            double dwMean = digits.Average(g => (double)g.W);
            double dwStd = Std(digits.Select(g => (double)g.W));
            if (dwStd / dwMean > 0.30) { res.Reason = "数字宽度差太多"; return res; }

            double mw = Median(digits.Select(g => (double)g.W));
            double mh = Median(digits.Select(g => (double)g.H));
            double ar = mw / mh;
            if (ar < 0.45 || ar > 0.75) { res.Reason = "数字宽高比不对"; return res; }

            var bar = bars.OrderBy(g => g.X).First();   // 最左侧的横条即负号
            res.Measured = true;
            res.BarWidth = bar.W / mw;
            res.DigitHeight = mh;
            res.DigitAspect = ar;
            res.DigitCount = digits.Count;

            if (mh < MinDigitHeight)
            {
                res.Verdict = MinusVerdict.CannotDetermine;
                res.Reason = $"数字高 {mh:F0} < {MinDigitHeight}, 判不准所以不判";
                return res;
            }
            res.Verdict = res.BarWidth >= Threshold ? MinusVerdict.Suspicious : MinusVerdict.Ok;
            res.Reason = $"负号宽比 {res.BarWidth:F4} (阈值 {Threshold})";
            return res;
        }

        // 上三分之一的通道均值判断蓝底
        static bool IsBluePage(byte[] px, int w, int h, int stride, int ri, int bi)
        {
            int h3 = Math.Max(1, h / 3);
            double r = 0, g = 0, b = 0; long n = (long)h3 * w;
            for (int y = 0; y < h3; y++)
            {
                int row = y * stride;
                for (int x = 0; x < w; x++)
                {
                    int o = row + x * 3;
                    r += px[o + ri]; g += px[o + 1]; b += px[o + bi];
                }
            }
            r /= n; g /= n; b /= n;
            return b > r + 25 && b > g + 15;
        }

        // 定点系数, 与 OpenCV cvtColor RGB2GRAY 结果一致, 不要改成浮点写法
        static byte[,] ToGray(byte[] px, int w, int h, int stride, int ri, int bi)
        {
            var gray = new byte[h, w];
            for (int y = 0; y < h; y++)
            {
                int row = y * stride;
                for (int x = 0; x < w; x++)
                {
                    int o = row + x * 3;
                    gray[y, x] = (byte)((px[o + ri] * 9798 + px[o + 1] * 19235
                                         + px[o + bi] * 3735 + 16384) >> 15);
                }
            }
            return gray;
        }

        // 在图像 8%~55% 高度范围内, 取同一行中字最高的那一行
        static int[]? LocateAmount(byte[,] gray, int W, int H)
        {
            int y0b = (int)(H * 0.08), y1b = (int)(H * 0.55);
            int bh = y1b - y0b;
            if (bh < 8) return null;
            var dark = new bool[bh, W];
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < W; x++) dark[y, x] = gray[y0b + y, x] < GrayDark;

            var all = Label(dark, bh, W, minArea: 20);
            var comps = new List<Comp>();
            foreach (var c in all)
            {
                // 此高度过滤会滤掉负号, 所以后面要在行内重取
                if (c.H < 0.02 * H || c.H > 0.22 * H || c.W > 0.5 * W) continue;
                comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
            }
            if (comps.Count == 0) return null;

            // 上限保护: 下面的行聚类最坏为 O(N^2)
            if (comps.Count > MaxComponents) return null;

            // 按纵向重叠聚类成行; 每行上下界增量维护
            var rows = new List<List<Comp>>();
            var bounds = new List<(int Y0, int Y1)>();
            foreach (var c in comps.OrderBy(k => k.Y))
            {
                bool placed = false;
                for (int i = 0; i < rows.Count; i++)
                {
                    int ry0 = bounds[i].Y0, ry1 = bounds[i].Y1;
                    if (Math.Min(c.Y + c.H, ry1) - Math.Max(c.Y, ry0)
                        >= 0.5 * Math.Min(c.H, ry1 - ry0))
                    {
                        rows[i].Add(c);
                        bounds[i] = (Math.Min(ry0, c.Y), Math.Max(ry1, c.Y + c.H));
                        placed = true; break;
                    }
                }
                if (!placed)
                {
                    rows.Add(new List<Comp> { c });
                    bounds.Add((c.Y, c.Y + c.H));
                }
            }

            List<Comp>? best = null; double bestMed = -1;
            foreach (var r in rows)
            {
                if (r.Count < 2) continue;
                int x0 = r.Min(k => k.X), x1 = r.Max(k => k.X + k.W);
                if (x1 - x0 < 0.1 * W) continue;             // 太窄的一行不是金额
                var hh = r.Select(k => k.H).OrderBy(v => v).ToList();
                double med = hh[r.Count / 2];
                if (med > bestMed) { bestMed = med; best = r; }
            }
            if (best == null) return null;
            return new[] { best.Min(k => k.X), best.Min(k => k.Y),
                           best.Max(k => k.X + k.W), best.Max(k => k.Y + k.H) };
        }

        // 8 连通标记, 显式栈避免递归过深
        static List<Comp> Label(bool[,] fg, int h, int w, int minArea)
        {
            var seen = new bool[h, w];
            var outp = new List<Comp>();
            var stack = new Stack<int>();
            for (int sy = 0; sy < h; sy++)
                for (int sx = 0; sx < w; sx++)
                {
                    if (!fg[sy, sx] || seen[sy, sx]) continue;
                    // 扫描顺序保证种子点位于该块最上一行, 故 minY 即 sy
                    int minX = sx, maxX = sx, maxY = sy, area = 0;
                    stack.Push(sy * w + sx); seen[sy, sx] = true;
                    while (stack.Count > 0)
                    {
                        int v = stack.Pop();
                        int y = v / w, x = v - y * w;
                        area++;
                        if (x < minX) minX = x; else if (x > maxX) maxX = x;
                        if (y > maxY) maxY = y;
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            int ny = y + dy;
                            if (ny < 0 || ny >= h) continue;
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (dy == 0 && dx == 0) continue;
                                int nx = x + dx;
                                if (nx < 0 || nx >= w) continue;
                                if (fg[ny, nx] && !seen[ny, nx])
                                { seen[ny, nx] = true; stack.Push(ny * w + nx); }
                            }
                        }
                    }
                    if (area >= minArea)
                        outp.Add(new Comp { X = minX, Y = sy, W = maxX - minX + 1,
                                            H = maxY - sy + 1, Area = area });
                }
            return outp;   // 保持发现顺序
        }

        // Otsu 阈值
        static int Otsu(byte[,] a, int h, int w)
        {
            var hist = new long[256];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) hist[a[y, x]]++;
            long total = (long)h * w, sum = 0;
            for (int i = 0; i < 256; i++) sum += (long)i * hist[i];
            long sumB = 0, wB = 0; double maxVar = -1; int thr = 0;
            for (int t = 0; t < 256; t++)
            {
                wB += hist[t]; if (wB == 0) continue;
                long wF = total - wB; if (wF == 0) break;
                sumB += (long)t * hist[t];
                double mB = (double)sumB / wB, mF = (double)(sum - sumB) / wF;
                double v = (double)wB * wF * (mB - mF) * (mB - mF);
                if (v > maxVar) { maxVar = v; thr = t; }
            }
            return thr;
        }

        // 偶数个时取中间两个的平均
        static double Median(IEnumerable<double> xs)
        {
            var v = xs.OrderBy(k => k).ToList();
            if (v.Count == 0) return 0;
            return v.Count % 2 == 1 ? v[v.Count / 2]
                                    : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2.0;
        }

        // 总体标准差
        static double Std(IEnumerable<double> xs)
        {
            var v = xs.ToList(); if (v.Count == 0) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Sum(k => (k - m) * (k - m)) / v.Count);
        }
    }
}

// System.Drawing 调用示例:
//
//   using var bmp = new Bitmap(path);
//   var d = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height),
//                        ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
//   try
//   {
//       var px = new byte[(long)d.Stride * bmp.Height];
//       Marshal.Copy(d.Scan0, px, 0, px.Length);
//       return MinusCheck.Check(px, bmp.Width, bmp.Height, PixelOrder.Bgr, d.Stride);
//   }
//   finally { bmp.UnlockBits(d); }
//
// Stride 与 Bgr 两个参数都要传, 少传不会抛异常, 但结果是错的。
