// ONNX 调用最小示例 —— 只示范"怎么把小块喂进去、怎么把分数读出来"。
//
// ★ 这份**不是**完整实现, 故意不是。
//   取哪些块、金额怎么定位、分数怎么判 —— 那些在 ONNX_PORT_SPEC.md 里, 要你自己写。
//   这里只解决一件事: **张量的形状和内存排布对不对**。
//   那正是跨语言最容易悄悄写错、而且写错不会报错的地方。
//
// 怎么自验
// --------
//   `vectors.json` 里 case1 的 `stage3_chosen_positions` 已经给了 16 个块的坐标,
//   `stage4_patch_scores` 给了这 16 块各自应得的分数。
//   所以这份 demo 按**现成坐标**裁块 -> 喂模型 -> 和现成分数对。
//   对得上 = 你的张量构造是对的; 对不上 = 就是这一层的问题, 与取块算法无关。
//
// 依赖
// ----
//   dotnet add package Microsoft.ML.OnnxRuntime
//   dotnet add package SixLabors.ImageSharp
//
//   ⚠ ImageSharp 默认会按 EXIF Orientation 自动摆正, 真实进件必须**关掉**
//     (见 ONNX_PORT_SPEC §0.5(1))。这里用的 7 张对照图是程序生成的 PNG, 不带 EXIF,
//     所以 demo 里看不出差别 —— 但你的正式代码里要处理。
//
// 用法
// ----
//   dotnet run -- aigen_v7.onnx conformance/vectors.json conformance/images

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

class OnnxCallDemo
{
    const int PatchSize = 32;

    static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.WriteLine("用法: dotnet run -- <aigen_v7.onnx> <vectors.json> <images 目录>");
            return 1;
        }
        string onnxPath = args[0], vectorsPath = args[1], imagesDir = args[2];

        // ---- 1. 建会话 ----
        // ★ 线程数一定要显式设, 不设的话 onnxruntime 按核数开, 多进程部署会互相抢核。
        //   具体开几个见 ONNX_PORT_SPEC §0.7(2)。
        var so = new SessionOptions();
        so.IntraOpNumThreads = 2;
        so.InterOpNumThreads = 1;
        using var session = new InferenceSession(onnxPath, so);

        // 顺带把模型的输入输出名字打出来, 免得记错
        Console.WriteLine("输入: " + string.Join(", ", session.InputMetadata.Keys));
        Console.WriteLine("输出: " + string.Join(", ", session.OutputMetadata.Keys));
        Console.WriteLine();

        using var doc = JsonDocument.Parse(File.ReadAllText(vectorsPath));
        var cases = doc.RootElement.GetProperty("cases");

        int okCases = 0, badCases = 0;
        foreach (var c in cases.EnumerateArray())
        {
            string name = c.GetProperty("image").GetString()!;
            string imgPath = Path.Combine(imagesDir, name);
            if (!File.Exists(imgPath)) { Console.WriteLine($"  跳过(找不到图) {name}"); continue; }

            // ---- 2. 解码整图 ----
            using var img = Image.Load<Rgb24>(imgPath);

            // ---- 3. 按向量里现成的坐标裁 16 个块 ----
            //   stage3_chosen_positions 是 [[y, x], ...], **先 y 后 x**(顺序反了取到的块完全不同)
            var pos = c.GetProperty("stage3_chosen_positions");
            int n = pos.GetArrayLength();

            // ★★ 关键: NHWC 排布, 也就是 (批, 高, 宽, 通道), 且是 **uint8 原始像素**。
            //   不要归一化、不要除 255、不要转 NCHW、不要自己放大到 256 ——
            //   这四样**都已经包在 onnx 图里**了。自己再做一遍, 分数就废了, 而且不会报错。
            var tensor = new DenseTensor<byte>(new[] { n, PatchSize, PatchSize, 3 });

            int i = 0;
            foreach (var yx in pos.EnumerateArray())
            {
                int y = yx[0].GetInt32();
                int x = yx[1].GetInt32();
                for (int dy = 0; dy < PatchSize; dy++)
                {
                    for (int dx = 0; dx < PatchSize; dx++)
                    {
                        Rgb24 p = img[x + dx, y + dy];     // ImageSharp 索引是 [x, y]
                        tensor[i, dy, dx, 0] = p.R;
                        tensor[i, dy, dx, 1] = p.G;
                        tensor[i, dy, dx, 2] = p.B;
                    }
                }
                i++;
            }

            // ---- 4. 跑 ----
            // ★ 正式代码里建议**一次喂 4 个块**而不是 16 个: 内存少占 265 MB, 速度一样,
            //   分数逐位不变(见 ONNX_PORT_SPEC §0.7(1))。这里为了和向量逐块对照, 一次喂完。
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("patches", tensor)
            };
            using var results = session.Run(inputs);
            float[] got = results.First().AsEnumerable<float>().ToArray();

            // ---- 5. 和向量里的期望分数比 ----
            var want = c.GetProperty("stage4_patch_scores")
                        .EnumerateArray().Select(v => v.GetSingle()).ToArray();

            double maxDiff = 0;
            for (int k = 0; k < Math.Min(got.Length, want.Length); k++)
                maxDiff = Math.Max(maxDiff, Math.Abs(got[k] - want[k]));

            // 台阶 4 的容差是 1e-4(前提是两边都跑 CPU, 见 §5)
            bool ok = got.Length == want.Length && maxDiff <= 1e-4;
            if (ok) okCases++; else badCases++;

            Console.WriteLine($"  {name,-26} 块数 {got.Length,2}  最大差 {maxDiff:E2}  {(ok ? "对上了" : "★ 对不上")}");

            if (!ok)
            {
                Console.WriteLine($"      期望前三个: {string.Join(", ", want.Take(3).Select(v => v.ToString("F6")))}");
                Console.WriteLine($"      实际前三个: {string.Join(", ", got.Take(3).Select(v => v.ToString("F6")))}");
            }
        }

        Console.WriteLine();
        Console.WriteLine($"对上 {okCases} 张, 对不上 {badCases} 张");
        if (badCases > 0)
        {
            Console.WriteLine();
            Console.WriteLine("对不上先查这三样, 按可能性排:");
            Console.WriteLine("  1. 张量排布写成 NCHW 了 —— 这里要的是 **NHWC**(批, 高, 宽, 通道)");
            Console.WriteLine("  2. 自己又做了一遍归一化或者除了 255 —— 图里已经做过了, 喂原始 uint8 就行");
            Console.WriteLine("  3. 裁块时把 y 和 x 弄反了 —— 向量里是 [y, x], 而 ImageSharp 索引是 [x, y]");
            return 1;
        }
        Console.WriteLine("-> 张量这一层是对的。接下来照 ONNX_PORT_SPEC 把取块和定位写完, 再对台阶 1~3 和 6~12。");
        return 0;
    }
}
