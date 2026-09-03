// 可选: 把图片文件读成 MinusCheck 要的 RGB 字节数组。
//
// 这个文件是可选的。MinusCheck.cs 本身不依赖任何第三方包,
// 如果你的服务里已经把图解码好了(不管用什么解码器), 直接调
// MinusCheck.Check(pixels, width, height, order, stride) 就行, 这个文件可以不要。
// 推荐走这条路 —— 见下面第 1 条, ImageSharp 的授权不是能随手引的。
//
// ⚠ ImageSharp 的三件事, 先看清楚再决定用不用
// -------------------------------------------
// 1. 授权: 3.x 和 4.x 都是 Six Labors Split License(商业使用要买 license)。
// 只有 2.x 是 Apache-2.0。
// 3.x 和 4.x 的区别只是有没有在编译期强制检查, 不是授权本身的区别:
// · 4.x 没有 license 时 `dotnet build -c Release` 直接报错(Debug 只是警告), 实测过:
// error : No Six Labors license found.
// · 3.x 不卡编译, 但授权义务完全一样。另外 3.1.5 有 NU1903(高危)/ NU1902(中危)告警。
// 别以为"钉在 3.x 就绕开商业授权了" —— 绕不开。这条写清楚免得被当成结论传出去。
//
// 2. EXIF 摆正: ImageSharp 解码时不会自动摆正。 实测过(3.1.5):
// 造一张 120x60、EXIF Orientation=6 的 JPEG, ImageSharp 读出来仍然是 120x60,
// 没有转; PIL 的 `Image.open().convert("RGB")` 也没有转。两边一致, 不用做任何处理。
// (仓库里 ONNX_PORT_SPEC 那句"ImageSharp 默认会自动摆正"是不对的, 已实测。)
// 但如果你换成别的解码器, 要先确认它摆不摆 —— 摆了就会和线上 Python 对不上。
//
// 3. 这个方法会抛异常。 实测 9,000 张真实进件里 2 张(0.02%) ImageSharp 读不了
// (截断、坏文件、扩展名骗人的 HEIC 之类)。
// 调用方自己 try/catch, 读不了就当"这条判据没意见", 让 OCR 照常走。
//
// 顺带一个已知的口径差: PIL 对截断的 JPEG 会直接报错, ImageSharp 会照读。
// 同一批 9,000 张里, 有 12 张(0.13%)是 C# 量得出而 Python 量不出的, 反过来一张都没有。
// 要和 Python 完全同口径的话, 这类图应当也当成读不了。

#nullable enable

using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;

namespace Ssp
{
    public static class MinusCheckLoader
    {
        /// <summary>
        /// 像素数上限, 防"解压炸弹"(几十 KB 的 PNG 能解出上亿像素)。
        /// 手机截图一般 300 万像素上下, 5000 万已经非常宽松了。
        /// </summary>
        public const long MaxPixels = 50_000_000;

        /// <summary>用 ImageSharp 读文件并跑负号检查。文件读不了会抛异常, 调用方要接。</summary>
        public static MinusResult CheckFile(string path)
        {
            var (px, w, h) = LoadRgb(path);
            return MinusCheck.Check(px, w, h);
        }

        /// <summary>读成紧密排列的 RGB24 字节数组(长度 = w*h*3, 行间无填充)。</summary>
        public static (byte[] pixels, int width, int height) LoadRgb(
            string path, long maxPixels = MaxPixels)
        {
            // 先只读文件头拿尺寸, 不解码 —— 否则"图太大"这道闸等于没有,
            // 等发现太大的时候内存已经吃进去了。
            var info = Image.Identify(path);
            long n = (long)info.Width * info.Height;
            if (n > maxPixels)
                throw new InvalidOperationException(
                    $"图太大: {info.Width}x{info.Height} = {n:N0} 像素, 超过上限 {maxPixels:N0}");

            // MaxFrames = 1: 多帧 GIF / WebP 只解第一帧。
            // 不限制的话一个多帧文件会把每一帧都解出来留在内存里, 而我们只用第一帧。
            var opts = new DecoderOptions { MaxFrames = 1 };
            using var img = Image.Load<Rgb24>(opts, path);

            int w = img.Width, h = img.Height;
            var px = new byte[(long)w * h * 3];   // 上面已经卡过上限, 这里不会溢出 int
            img.CopyPixelDataTo(px);              // Rgb24 本来就是紧密排列的 RGB, stride = w*3
            return (px, w, h);
        }
    }
}

// ---------------------------------------------------------------------------
// 不想引 ImageSharp 的话, 用 System.Drawing 就行, Windows 上不用装任何包 ——
// 现成的写法在 MinusCheck.cs 文件末尾, 那边一并写清楚了 stride 和 BGR 这两个坑。
// 那条路只需要 MinusCheck.cs 一个文件, 这个文件可以整个不要。
// ---------------------------------------------------------------------------
