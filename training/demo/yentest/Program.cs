using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using Ssp;

namespace Ssp.YenTest
{
    /// <summary>
    /// 对一批图跑 <see cref="YenCheck.Check"/>, 逐张打印结论, 量到的写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/yentest
    ///   dotnet run -c Release -- 图或目录 [图或目录 ...] [--list 清单.txt] [--plat apple|android|unknown] [--out 结果.csv]
    ///
    ///   --plat 不给就是 unknown(和经理那边只传一个 Mat 的调用一样)。
    ///   给的是文件就逐张打印; 给的是目录就递归扫, 只打印可疑的和汇总。
    ///   退出码: 0 = 跑完; 2 = 参数不对。
    /// </summary>
    static class Program
    {
        static readonly string[] Exts = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

        static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            string? outCsv = null;
            var plat = YenPlatform.Unknown;
            var inputs = new List<string>();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--out" && i + 1 < args.Length) outCsv = args[++i];
                else if (args[i] == "--list" && i + 1 < args.Length)
                {
                    // 一行一个路径的清单(命令行塞不下几千个路径时用)
                    foreach (var line in File.ReadAllLines(args[++i], Encoding.UTF8))
                        if (line.Trim().Length > 0) inputs.Add(line.Trim());
                }
                else if (args[i] == "--plat" && i + 1 < args.Length)
                {
                    string p = args[++i].ToLowerInvariant();
                    if (p == "apple") plat = YenPlatform.Apple;
                    else if (p == "android") plat = YenPlatform.Android;
                    else if (p == "unknown") plat = YenPlatform.Unknown;
                    else { Console.Error.WriteLine($"--plat 只认 apple / android / unknown: {p}"); return 2; }
                }
                else inputs.Add(args[i]);
            }
            if (inputs.Count == 0)
            {
                Console.Error.WriteLine("用法: YenTest 图或目录 [图或目录 ...] [--plat apple|android|unknown] [--out 结果.csv]");
                return 2;
            }

            var files = new List<string>();
            bool anyDir = false;
            foreach (var p in inputs)
            {
                if (Directory.Exists(p))
                {
                    anyDir = true;
                    files.AddRange(Directory.EnumerateFiles(p, "*", SearchOption.AllDirectories)
                        .Where(f => Exts.Contains(Path.GetExtension(f).ToLowerInvariant())));
                }
                else if (File.Exists(p)) files.Add(p);
                else { Console.Error.WriteLine($"不存在: {p}"); return 2; }
            }
            files.Sort(StringComparer.Ordinal);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var results = new ConcurrentBag<(string path, int w, int h, YenResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                YenResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) return;
                    w = img.Width; h = img.Height;
                    r = YenCheck.Check(img, plat);
                }
                catch (Exception e)
                {
                    r = new YenResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
            });
            sw.Stop();

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            var inv = CultureInfo.InvariantCulture;
            foreach (var (path, w, h, r) in all)
            {
                if (anyDir && r.Verdict != YenVerdict.Suspicious) continue;
                Console.WriteLine($"{(int)r.Verdict} {r.Verdict,-15} {Path.GetFileName(path)}  {w}x{h}  "
                    + (r.Measured
                        ? $"stem {r.StemRatio.ToString("F4", inv)} yenH {r.YenHeightRatio.ToString("F4", inv)} "
                          + $"bar {r.BarRatio.ToString("F4", inv)} norm {r.DigitStrokeNorm.ToString("F4", inv)} "
                          + $"dh {r.DigitHeight.ToString("F0", inv)}  "
                        : "")
                    + r.Reason);
            }
            int ok = all.Count(t => t.r.Verdict == YenVerdict.Ok);
            int sus = all.Count(t => t.r.Verdict == YenVerdict.Suspicious);
            int cd = all.Count(t => t.r.Verdict == YenVerdict.CannotDetermine);
            Console.WriteLine($"共 {all.Count} 张  Ok {ok}  Suspicious {sus}  CannotDetermine {cd}  机型 {plat}  "
                              + $"用时 {sw.Elapsed.TotalSeconds:F1}s");

            if (outCsv != null)
            {
                var sb = new StringBuilder();
                sb.AppendLine("path,W,H,verdict,measured,stem_ratio,yen_h_ratio,bar_ratio,bot_gap,stroke_norm,digit_h,d_stroke,y_stem,reason");
                foreach (var (path, w, h, r) in all)
                {
                    sb.Append('"').Append(path.Replace("\"", "\"\"")).Append('"').Append(',')
                      .Append(w).Append(',').Append(h).Append(',')
                      .Append((int)r.Verdict).Append(',').Append(r.Measured ? 1 : 0).Append(',')
                      .Append(r.StemRatio.ToString("R", inv)).Append(',')
                      .Append(r.YenHeightRatio.ToString("R", inv)).Append(',')
                      .Append(r.BarRatio.ToString("R", inv)).Append(',')
                      .Append(r.BottomGap.ToString("R", inv)).Append(',')
                      .Append(r.DigitStrokeNorm.ToString("R", inv)).Append(',')
                      .Append(r.DigitHeight.ToString("R", inv)).Append(',')
                      .Append(r.DigitStroke.ToString("R", inv)).Append(',')
                      .Append(r.StemWidth.ToString("R", inv)).Append(',')
                      .Append('"').Append(r.Reason.Replace("\"", "\"\"")).Append('"')
                      .AppendLine();
                }
                File.WriteAllText(outCsv, sb.ToString(), new UTF8Encoding(true));
                Console.WriteLine($"写了 {outCsv}");
            }
            return 0;
        }
    }
}
