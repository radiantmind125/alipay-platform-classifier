#nullable enable

using System;
using System.Collections.Generic;

namespace Ssp
{
    public enum ChevronVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // 返回箭头偏小且发虚
    }

    public sealed class ChevronResult
    {
        public ChevronVerdict Verdict;
        public int ArrowHeight;      // 返回箭头的高
        public int ArrowWidth;       // 返回箭头的宽
        public int ArrowTop;         // 箭头顶边在整图中的 y
        public double Blur;          // 半灰边缘像素 / 实心笔画像素
        public bool Measured;        // 为 false 时上面几项无意义
        public string Reason = "";
    }

    /// <summary>
    /// 账单详情页左上角返回箭头的几何检查。
    ///
    /// iOS 的导航栏图标由系统按点数渲染, 所以苹果各机型不管屏幕分辨率多少,
    /// 这个箭头都是 56x33 像素, 边缘很利。实测五种苹果分辨率, 92%~94% 都是这个尺寸。
    /// 用固定尺寸素材拼出来的图, 箭头会偏小而且发虚 —— 发虚是因为它被缩放过,
    /// 原生渲染无论多大都是利的。
    ///
    /// 只对苹果机型有效。安卓各家 ROM 自己画导航栏, 同一分辨率下实测有五百多种不同的
    /// 箭头位图, 最常见的才占 7%, 没有可比的基准; 而且安卓的正常箭头尺寸落在
    /// 和异常箭头相同的区间里, 分不开。所以遇到非苹果分辨率一律不判。
    ///
    /// 无外部依赖, 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class ChevronCheck
    {
        // 真图箭头的标准尺寸(苹果 @3x)
        public const int NormalHeight = 56;
        public const int NormalWidth = 33;

        // 判定阈值: 高和宽都低于标准的 92%, 且虚实比超过这个值
        public const double SizeFraction = 0.92;
        public const double BlurThreshold = 0.65;

        const int GrayDark = 170;    // 算"深色"的灰度阈
        const int CoreThr = 100;     // 低于此算实心笔画
        const int MidThr = 200;      // 介于 CoreThr 和此值之间算半灰边缘

        // 认作返回箭头的尺寸范围。真图 56x33, 异常的 47x28, 都在这个范围内;
        // 范围外的一律不当箭头, 免得在没有箭头的页面上挑中别的图标。
        const int MinPlausibleH = 40, MaxPlausibleH = 70;
        const int MinPlausibleW = 22, MaxPlausibleW = 42;

        /// <summary>
        /// 适用的分辨率。iOS 导航图标按点数渲染, 所以这些机型的箭头尺寸完全一致。
        /// 出现新机型时往这里加; 不在表里的一律返回 CannotDetermine, 不猜。
        /// </summary>
        static readonly HashSet<(int, int)> Supported = new()
        {
            (1179, 2556), (1290, 2796), (1170, 2532),
            (1320, 2868), (1206, 2622), (1284, 2778),
            (1125, 2436), (1242, 2688),
        };

        struct Comp { public int X, Y, W, H, Area; }

        /// <summary>
        /// 每像素 3 字节, 第 y 行第 x 列的首字节位于 y*stride + x*3。
        /// </summary>
        /// <param name="order">通道顺序; System.Drawing 解出的数据传 Bgr。</param>
        /// <param name="stride">
        /// 每行字节数, 传 0 表示 width*3。GDI 的 BitmapData.Stride 会补齐到 4 的倍数,
        /// 需如实传入, 否则逐行错位且不会抛异常。
        /// </param>
        public static ChevronResult Check(byte[] pixels, int width, int height,
                                          PixelOrder order = PixelOrder.Rgb, int stride = 0)
        {
            if (pixels == null) throw new ArgumentNullException(nameof(pixels));
            if (width <= 0 || height <= 0)
                throw new ArgumentException("宽高必须为正");
            if (stride == 0) stride = width * 3;
            if (stride < width * 3)
                throw new ArgumentException($"stride {stride} 小于一行需要的 {width * 3} 字节");
            long need = (long)(height - 1) * stride + (long)width * 3;
            if (pixels.Length < need)
                throw new ArgumentException(
                    $"像素数组长度 {pixels.Length} 不够, 按 stride={stride} 需要 {need} 字节");

            var res = new ChevronResult { Verdict = ChevronVerdict.CannotDetermine };

            if (!Supported.Contains((width, height)))
            {
                res.Reason = $"{width}x{height} 不是已知的苹果机型分辨率, 不判";
                return res;
            }

            int ri = order == PixelOrder.Rgb ? 0 : 2;
            int bi = order == PixelOrder.Rgb ? 2 : 0;

            // 只看左上角一小块, 导航栏一定在这个范围内
            int cw = (int)(width * 0.13), ch = (int)(height * 0.12);
            if (cw < 16 || ch < 16) { res.Reason = "图太小"; return res; }

            var gray = new byte[ch, cw];
            for (int y = 0; y < ch; y++)
            {
                int row = y * stride;
                for (int x = 0; x < cw; x++)
                {
                    int o = row + x * 3;
                    gray[y, x] = (byte)((pixels[o + ri] * 9798 + pixels[o + 1] * 19235
                                         + pixels[o + bi] * 3735 + 16384) >> 15);
                }
            }

            var arrow = LocateArrow(gray, ch, cw);
            if (arrow == null) { res.Reason = "找不到返回箭头"; return res; }
            var a = arrow.Value;

            // 虚实比: 半灰边缘像素数 / 实心笔画像素数。
            // 原生渲染的图标边缘干净, 这个比值小; 缩放过的会糊出一圈半灰, 比值明显变大。
            int core = 0, mid = 0;
            for (int y = a.Y; y < a.Y + a.H; y++)
                for (int x = a.X; x < a.X + a.W; x++)
                {
                    int g = gray[y, x];
                    if (g < CoreThr) core++;
                    else if (g < MidThr) mid++;
                }
            if (core == 0) { res.Reason = "箭头没有实心像素"; return res; }

            res.Measured = true;
            res.ArrowHeight = a.H;
            res.ArrowWidth = a.W;
            res.ArrowTop = a.Y;
            res.Blur = (double)mid / core;

            bool small = a.H < NormalHeight * SizeFraction && a.W < NormalWidth * SizeFraction;
            bool fuzzy = res.Blur >= BlurThreshold;
            if (small && fuzzy)
            {
                res.Verdict = ChevronVerdict.Suspicious;
                res.Reason = $"箭头 {a.H}x{a.W} 小于常态 {NormalHeight}x{NormalWidth}, 虚实比 {res.Blur:F3}";
            }
            else
            {
                res.Verdict = ChevronVerdict.Ok;
                res.Reason = $"箭头 {a.H}x{a.W}, 虚实比 {res.Blur:F3}";
            }
            return res;
        }

        /// <summary>
        /// 找导航栏的返回箭头。用连通域找, 不用固定窗口 ——
        /// 状态栏高度各机型不一样, 固定窗口会切到箭头上或者整个错过。
        /// </summary>
        static Comp? LocateArrow(byte[,] gray, int h, int w)
        {
            var fg = new bool[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    fg[y, x] = gray[y, x] < GrayDark;

            Comp? best = null;
            var seen = new bool[h, w];
            var stack = new Stack<int>();
            for (int sy = 0; sy < h; sy++)
                for (int sx = 0; sx < w; sx++)
                {
                    if (!fg[sy, sx] || seen[sy, sx]) continue;
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
                    int bh = maxY - sy + 1, bw = maxX - minX + 1;
                    // 箭头: 竖着比横着长, 有一定墨量, 且尺寸落在合理范围内。
                    // 尺寸下限很重要: 没有返回箭头的页面上, 不加下限会挑中别的小图标
                    // (实测挑到 18x11、19x15 这种), 误报会从万分之 0.8 涨到万分之 6。
                    if (bh < MinPlausibleH || bh > MaxPlausibleH) continue;
                    if (bw < MinPlausibleW || bw > MaxPlausibleW) continue;
                    if (bh < bw || area < 100) continue;
                    // 状态栏的图标也可能过筛, 取最靠下的那个 —— 导航栏在状态栏下面
                    if (best == null || sy > best.Value.Y)
                        best = new Comp { X = minX, Y = sy, W = bw, H = bh, Area = area };
                }
            return best;
        }
    }
}
