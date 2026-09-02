// 可选: 把图片文件读成 MinusCheck 要的 RGB 字节数组。
//
// ★ 这个文件是**可选**的。MinusCheck.cs 本身不依赖任何第三方包,
//   如果你的服务里已经把图解码好了(不管用什么解码器), 直接调
//   MinusCheck.Check(pixels, width, height) 就行, 这个文件可以不要。
//
// ⚠ ImageSharp 的两个坑, 先看清楚再决定用不用
// -------------------------------------------
// 1. **4.x 起是商业授权, 而且是在编译期卡的。** 没有 license 时
//    `dotnet build -c Release` 直接**报错**(Debug 只是警告), 实测过:
//        error : No Six Labors license found.
//    2.x 是 Apache-2.0 不卡; 3.x 不卡但有已知高危漏洞告警(NU1903)。
// 2. **ImageSharp 默认按 EXIF Orientation 自动摆正**, 而 GDI+/SkiaSharp/WIC 默认不摆。
//    截图大多没有 Orientation, **但翻拍图有** —— 摆不摆正会改变量出来的几何量。
//    下面显式关掉了自动摆正, 和线上 Python 那边(PIL 不自动摆)保持一致。
//
// **所以更推荐用你工程里已经有的解码器**, 别为这一个检查新引一个包 ——
// 下面同时给了 System.Drawing 的写法(Windows 服务上不用装任何东西)。

#nullable enable

using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;

namespace Ssp
{
    public static class MinusCheckLoader
    {
        /// <summary>用 ImageSharp 读文件并跑负号检查。</summary>
        public static MinusResult CheckFile(string path)
        {
            var (px, w, h) = LoadRgb(path);
            return MinusCheck.Check(px, w, h);
        }

        /// <summary>读成紧密排列的 RGB24 字节数组(长度 = w*h*3)。</summary>
        public static (byte[] pixels, int width, int height) LoadRgb(string path)
        {
            // ★ 关掉 EXIF 自动摆正, 和线上 Python 侧一致
            var opts = new DecoderOptions { SkipMetadata = false };
            using var img = Image.Load<Rgb24>(opts, path);
            img.Metadata.ExifProfile = null;      // 防止后续操作再摆一次

            int w = img.Width, h = img.Height;
            var px = new byte[(long)w * h * 3 is var n && n <= int.MaxValue ? (int)n
                              : throw new InvalidOperationException("图太大")];
            img.CopyPixelDataTo(px);              // Rgb24 本来就是紧密排列的 RGB
            return (px, w, h);
        }
    }
}

// ---------------------------------------------------------------------------
// 不想引 ImageSharp 的话, Windows 上用 System.Drawing 是这样(不用装包):
//
//   using System.Drawing;
//   using System.Drawing.Imaging;
//
//   static (byte[], int, int) LoadRgbGdi(string path)
//   {
//       using var bmp = new Bitmap(path);
//       var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
//       var d = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
//       try
//       {
//           var px = new byte[bmp.Width * bmp.Height * 3];
//           for (int y = 0; y < bmp.Height; y++)                 // ★ 必须逐行拷,
//               System.Runtime.InteropServices.Marshal.Copy(     //   Stride 通常有 padding,
//                   d.Scan0 + y * d.Stride, px,                  //   整块拷会错位
//                   y * bmp.Width * 3, bmp.Width * 3);
//           return (px, bmp.Width, bmp.Height);
//       }
//       finally { bmp.UnlockBits(d); }
//   }
//
//   ★ 拿到之后调用要传 BGR —— Format24bppRgb 名字叫 Rgb, **内存里其实是 BGR**:
//       MinusCheck.Check(px, w, h, PixelOrder.Bgr);
//     传错的话蓝底页会被当成白底页, 蓝图那道闸就废了。
// ---------------------------------------------------------------------------
