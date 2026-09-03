// 负号几何检查 —— 不用模型, 不用显卡, 只做算术。
//
// 白底账单详情页里, 量**负号宽度 / 数字中位宽度**。真图这个比值很集中, 被拉长过的会明显偏出去。
// 4 万张白单实测中位: 4 位金额 0.6739 / 5 位 0.7021 / 6 位 0.7021。
//
// **零依赖**, 只用 System.*, 这一个文件放进工程就能编。
//
// ★★ 先看这两条
// -------------
// 1. **拿 03.jpg / 04.jpg 自验会返回 CannotDetermine, 那不是没生效。**
//    它们数字高只有 47 和 53, 低于默认的 60 px 弃权线。**要对的是 BarWidth**:
//        03.jpg   BarWidth 1.0000   DigitHeight 47
//        04.jpg   BarWidth 0.7097   DigitHeight 53
//    把 `MinDigitHeight` 改成 0 再跑, 这两张就是 **Suspicious** 和 **Ok**, 都判对。
// 2. **用 System.Drawing 解码时, `stride` 和 `PixelOrder.Bgr` 两个都必须传。**
//    少传任何一个都会**静默出错**(整图逐行错位 / 页型认反), 不抛异常。
//    文件末尾有现成的 GDI 写法, 照抄即可。
//
// 怎么调
// ------
//   var r = MinusCheck.Check(rgb, width, height);            // rgb = 解码好的字节数组
//   if (r.Verdict == MinusVerdict.Suspicious) amount = "";   // 比例不对, 金额输出空
//   // Ok 和 CannotDetermine 都照常走 OCR
//
// 多线程直接调没问题(没有可变的静态状态)。单线程约 38 ms 一张, 含解码。
//
// 阈值
// ----
// 0.78 是按"负号加 6 px"这个攻击尺度定的。1,473 对真图 + 加宽图实测能抓到的比例:
//   0.76 -> 99.8%    **0.78 -> 99.5%**    0.80 -> 89.3%    0.85 -> 32.7%    0.90 -> 0%
// 报出率(4 万张随机白单): 数字高 60 以上 **10.4/万**, 40~60 档 **13.7/万**。
// ★ 这是**报出率, 不是误报率** —— 进件池没有逐张的真伪标注, 报出来的里面有真被改过的。
//
// ★ `MinDigitHeight` 这道闸实测作用不大: 报出率只从 10.6/万 降到 10.4/万,
//   代价是丢掉 **6.9%** 可量的图, 而且报出率和字号并不是单调关系。
//   **要关掉就设成 0。** 这是覆盖率和精度的取舍, 谁用谁定。
//
// 覆盖边界(别让人以为管得更宽)
// ----------------------------
// - 只管**白底账单详情页**。**蓝底转账页直接跳过**, 那种金额不带负号。
//   不跳的话页面上的横线会被当成负号: 实测 1.2 万张蓝图有 40 张被量出, 其中九成会误报。
// - 只管"负号被拉长"这一种手法。**改数字、复制粘贴数字, 这条一律看不见。**
// - 判的是"负号和同一张图里的数字比例不对", **不是**"这张图是生成的"。

#nullable enable      // 显式打开, 不管工程开没开 nullable, 编译都是零警告

using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssp
{
    public enum MinusVerdict
    {
        // ★ CannotDetermine 放在 0 位: 默认构造是"不下结论"而不是"正常", 漏赋值时失败方向安全。
        CannotDetermine = 0,   // 判不了 -> OCR 照常
        Ok = 1,                // 负号正常
        Suspicious = 2,        // 负号偏长 -> 金额输出空
    }

    /// <summary>像素在字节数组里的通道顺序。</summary>
    public enum PixelOrder
    {
        Rgb = 0,   // ImageSharp Rgb24 / SkiaSharp Rgba8888
        Bgr = 1,   // ★ System.Drawing 的 Format24bppRgb 内存里其实是 BGR, 别选错
    }

    public sealed class MinusResult
    {
        public MinusVerdict Verdict;
        public double BarWidth;      // 负号宽 / 数字中位宽  <- 主判据
        public double DigitHeight;   // 数字中位高
        public double DigitAspect;   // 数字中位宽 / 中位高
        public int DigitCount;
        public bool Measured;        // 没量出 BarWidth 时上面几个都没意义
        public string Reason = "";
    }

    public static class MinusCheck
    {
        public const double Threshold = 0.78;      // 见文件头
        public const double MinDigitHeight = 60;   // 低于这个不判; 设成 0 即关掉, 见文件头
        const int GrayDark = 140;                  // 定位金额行时"算深色"的灰度阈

        // 安全阀, 只用来挡恶意构造的图: 真实账单过筛后中位 6 个块, 最多见过 17 个。
        public const int MaxComponents = 20_000;

        struct Comp   // 一个连通块
        {
            public int X, Y, W, H, Area;
        }

        /// <summary>
        /// 入口。<paramref name="pixels"/> 是解码后的像素, 每像素 3 字节,
        /// 第 y 行第 x 列的第一个分量在 <c>y*stride + x*3</c>。
        /// </summary>
        /// <param name="order">通道顺序。System.Drawing 出来的要传 <see cref="PixelOrder.Bgr"/>。</param>
        /// <param name="stride">
        /// 每行占的字节数, 传 0 表示行间没有填充(即 width*3)。
        /// ★ GDI 的 BitmapData.Stride 会补齐到 4 的倍数, 通常**大于** width*3, 必须如实传,
        ///   否则整张图逐行错开, 而且不会报错。
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

            int ri = order == PixelOrder.Rgb ? 0 : 2;   // 红在一个像素里的第几个字节
            int bi = order == PixelOrder.Rgb ? 2 : 0;   // 蓝在第几个字节

            // ★ 蓝底转账页金额不带符号, 覆盖率为 0; 不跳的话页面上的横线会被当成负号。
            if (IsBluePage(pixels, width, height, stride, ri, bi))
            { res.Reason = "蓝底转账页, 金额不带负号"; return res; }

            var gray = ToGray(pixels, width, height, stride, ri, bi);
            return Check(gray, width, height, res);
        }

        static MinusResult Check(byte[,] gray, int W, int H, MinusResult res)
        {
            // ---- 1. 定位金额行 ----
            var box = LocateAmount(gray, W, H);
            if (box == null) { res.Reason = "定位不到金额行"; return res; }
            int bx0 = box[0], by0 = box[1], bx1 = box[2], by1 = box[3];

            // ---- 2. 在这一行内重新取块 ----
            // ★ 必须重取, 而且**不设高度下限**: 定位那步的 hh < 0.02*H 会把负号整个滤掉
            //   (1280 高的图下限就是 25.6 px, 而负号只有 7 px)。
            // ★ 左边要**大幅**外扩: 定位给的 x0 只用数字算, 负号落在它左边, pad 小了够不着。
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
            // 前景过半说明极性反了, 翻回来。★ 门槛是 127/255 = 0.498039, **不是 0.5**。
            if (255L * on > 127L * ch * cw)
                for (int y = 0; y < ch; y++)
                    for (int x = 0; x < cw; x++) fg[y, x] = !fg[y, x];

            var glyphs = Label(fg, ch, cw, minArea: 6);   // ★ 不设高度下限
            if (glyphs.Count < 4) { res.Reason = "块数不够"; return res; }

            // ---- 3. 分出数字和横杠 ----
            var hs = glyphs.Select(g => g.H).OrderBy(v => v).ToList();
            double medH = hs[hs.Count / 2];
            if (medH < 20) { res.Reason = "字太小"; return res; }

            var digits = glyphs.Where(g => g.H > 0.75 * medH && g.W < 1.5 * g.H).ToList();
            var bars = glyphs.Where(g => g.W >= 1.5 * g.H && g.H <= 0.45 * medH).ToList();
            if (digits.Count < 4) { res.Reason = "数字不够 4 个"; return res; }
            if (bars.Count == 0) { res.Reason = "没有负号"; return res; }

            // ---- 4. 有效性闸 ----
            // ★ 没有这三道, **一整张二维码**会混进来并排在离群榜第一 —— 它同样满足
            //   "4 个近方块 + 1 根横条"。真金额行的数字高度几乎完全一致, 二维码不会。
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

            var bar = bars.OrderBy(g => g.X).First();   // 最靠左那条 = 负号
            res.Measured = true;
            res.BarWidth = bar.W / mw;
            res.DigitHeight = mh;
            res.DigitAspect = ar;
            res.DigitCount = digits.Count;

            // ---- 5. 判 ----
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

        // ---------- 下面都是工具 ----------

        // 蓝底判定: 上三分之一的三通道均值, 蓝明显压过红和绿
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

        // 灰度化。★ 这三个常数**别动** —— 它和 OpenCV 的 cvtColor(RGB2GRAY) 是**逐位相同**的
        //   (400 万随机像素实测零误差)。改成 0.299/0.587/0.114 的浮点写法会有 0.13% 的像素差 1 阶;
        //   蓝色系数写成 3736(四舍五入值)而不是截断的 3735, 会差 0.38%。
        //   灰度用在 `< 140` 那道闸和 Otsu 上, 差 1 阶就可能翻掉一个像素。
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

        // 找金额行: 只在图的 8%~55% 那一段找, 取"同一行里字最高的那一行"。
        // ★ 不能按"最高的连通块"选 —— 会被商家头像顶掉(头像 125~138 px, 金额才 71~78 px)。
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
                // ★ 这道高度闸会把负号滤掉 —— 所以第 2 步要在行内重取
                if (c.H < 0.02 * H || c.H > 0.22 * H || c.W > 0.5 * W) continue;
                comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
            }
            if (comps.Count == 0) return null;

            // ★ 安全阀: 下面的按行聚类最坏是 O(N^2), 恶意构造的图能把 N 顶到十万级。
            //   超上限直接当"判不了", 失败方向安全。
            if (comps.Count > MaxComponents) return null;

            // 按"同一行"聚类: 纵向重叠 >= 较矮那个高度的一半。
            // 金额的每一位都在同一条基线上, 商家头像不在 —— 全部依据就是这一条。
            // ★ 每行上下界增量维护(结果与逐次求 Min/Max 完全一样, 省一层 N)。
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
                if (r.Count < 2) continue;                   // 至少要有几位数字
                int x0 = r.Min(k => k.X), x1 = r.Max(k => k.X + k.W);
                if (x1 - x0 < 0.1 * W) continue;             // 太窄的一行不是金额
                var hh = r.Select(k => k.H).OrderBy(v => v).ToList();
                double med = hh[r.Count / 2];                // 用中位高, 一个异常块带不偏
                if (med > bestMed) { bestMed = med; best = r; }
            }
            if (best == null) return null;
            return new[] { best.Min(k => k.X), best.Min(k => k.Y),
                           best.Max(k => k.X + k.W), best.Max(k => k.Y + k.H) };
        }

        // 8 连通标记。用显式栈, 不用递归 —— 大连通块会把调用栈撑爆。
        static List<Comp> Label(bool[,] fg, int h, int w, int minArea)
        {
            var seen = new bool[h, w];
            var outp = new List<Comp>();
            var stack = new Stack<int>();          // 存 y*w+x, 比存元组省内存
            for (int sy = 0; sy < h; sy++)
                for (int sx = 0; sx < w; sx++)
                {
                    if (!fg[sy, sx] || seen[sy, sx]) continue;
                    // ★ 光栅扫描下第一次碰到某块的像素必然在它最上面一行, 所以 minY 就是 sy,
                    //   不用再更新; minX/maxX 不成立, 必须更新。
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
            return outp;   // ★ 不排序: 保持发现顺序, 结果确定
        }

        // Otsu: 直方图最大类间方差
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

        // ★ 偶数个时必须取中间两个的平均(和 numpy 的 median 一致)。
        //   金额位数经常是偶数, 图省事取上中位的话 04.jpg 的 0.7097 就复现不出来。
        static double Median(IEnumerable<double> xs)
        {
            var v = xs.OrderBy(k => k).ToList();
            if (v.Count == 0) return 0;
            return v.Count % 2 == 1 ? v[v.Count / 2]
                                    : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2.0;
        }

        // 总体标准差(除以 n)
        static double Std(IEnumerable<double> xs)
        {
            var v = xs.ToList(); if (v.Count == 0) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Sum(k => (k - m) * (k - m)) / v.Count);
        }
    }
}

// ---------------------------------------------------------------------------
// 用 System.Drawing 解码的现成写法(Windows 上不用装任何包)
//
//   using System.Drawing;
//   using System.Drawing.Imaging;
//
//   static MinusResult CheckFileGdi(string path)
//   {
//       using var bmp = new Bitmap(path);
//       var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
//       var d = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
//       try
//       {
//           var px = new byte[(long)d.Stride * bmp.Height];
//           System.Runtime.InteropServices.Marshal.Copy(d.Scan0, px, 0, px.Length);
//           // ★★ 下面两个参数都不能少, 少了**不报错**, 只会安静地给错结果:
//           //   1. d.Stride —— GDI 每行补齐到 4 的倍数。实测进件里 33.8% 的宽度不是 4 的倍数
//           //      (1179 / 1290 / 1170 / 1206 这些), 不传就整张图逐行错开。
//           //   2. PixelOrder.Bgr —— Format24bppRgb 名字叫 Rgb, 内存里其实是 BGR。
//           //      传错的话蓝底页会被当成白底页, 蓝图那道闸就废了。
//           return MinusCheck.Check(px, bmp.Width, bmp.Height, PixelOrder.Bgr, d.Stride);
//       }
//       finally { bmp.UnlockBits(d); }
//   }
//
// ★ 用别的解码器也一样: 拿到"每像素 3 字节"的缓冲区, 把宽、高、通道顺序、行跨距如实传进来。
//   行间没有填充就传 stride = 0。
// ---------------------------------------------------------------------------
