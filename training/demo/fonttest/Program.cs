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

namespace Ssp.FontTest
{
    /// <summary>
    /// 对一批图跑 <see cref="AmountFontCheck.Check"/>, 逐张打印结论, 量到的写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/fonttest
    ///   dotnet run -c Release -- 图或目录 [图或目录 ...] [--list 清单.txt] [--out 结果.csv]
    ///
    ///   给的是文件就逐张打印; 给的是目录就递归扫, 只打印可疑的和汇总。
    ///   csv 的列和 training/amount_font_scan.py 的 measure() 对得上。
    ///   退出码: 0 = 跑完; 2 = 参数不对。
    /// </summary>
    static class Program
    {
        static readonly string[] Exts = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

        static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            string? outCsv = null;
            var inputs = new List<string>();
            bool quiet = false;
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--out" && i + 1 < args.Length) outCsv = args[++i];
                else if (args[i] == "--list" && i + 1 < args.Length)
                {
                    // 一行一个路径的清单(命令行塞不下几千个路径时用); 用清单时只打印可疑的
                    foreach (var line in File.ReadAllLines(args[++i], Encoding.UTF8))
                        if (line.Trim().Length > 0) inputs.Add(line.Trim());
                    quiet = true;
                }
                else inputs.Add(args[i]);
            }
            if (inputs.Count == 0)
            {
                Console.Error.WriteLine("用法: FontTest 图或目录 [图或目录 ...] [--list 清单.txt] [--out 结果.csv]");
                return 2;
            }

            var files = new List<string>();
            foreach (var p in inputs)
            {
                if (Directory.Exists(p))
                {
                    quiet = true;
                    files.AddRange(Directory.EnumerateFiles(p, "*", SearchOption.AllDirectories)
                        .Where(f => Exts.Contains(Path.GetExtension(f).ToLowerInvariant())));
                }
                else if (File.Exists(p)) files.Add(p);
                else { Console.Error.WriteLine($"不存在: {p}"); return 2; }
            }
            files.Sort(StringComparer.Ordinal);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var results = new ConcurrentBag<(string path, int w, int h, AmountFontResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                AmountFontResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) return;
                    w = img.Width; h = img.Height;
                    r = AmountFontCheck.Check(img);
                }
                catch (Exception e)
                {
                    r = new AmountFontResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
            });
            sw.Stop();

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            var inv = CultureInfo.InvariantCulture;
            foreach (var (path, w, h, r) in all)
            {
                if (quiet && r.Verdict != AmountFontVerdict.Suspicious) continue;
                Console.WriteLine($"{(int)r.Verdict} {r.Verdict,-15} {Path.GetFileName(path)}  {w}x{h}  "
                    + (r.Measured
                        ? $"0宽 {r.ZeroWidthRatio.ToString("F4", inv)} 小数点高 {r.DotHeightRatio.ToString("F4", inv)} "
                          + $"dh {r.DigitHeight.ToString("F0", inv)}  "
                        : "")
                    + r.Reason);
            }
            int ok = all.Count(t => t.r.Verdict == AmountFontVerdict.Ok);
            int sus = all.Count(t => t.r.Verdict == AmountFontVerdict.Suspicious);
            int cd = all.Count(t => t.r.Verdict == AmountFontVerdict.CannotDetermine);
            Console.WriteLine($"共 {all.Count} 张  Ok {ok}  Suspicious {sus}  CannotDetermine {cd}  用时 {sw.Elapsed.TotalSeconds:F1}s");

            if (outCsv != null)
            {
                var sb = new StringBuilder();
                sb.AppendLine("path,W,H,verdict,measured,digit_h,n_digit,n_zero,zero_w,zero_hole_w,zero_edge,has_dot,dot_h,dot_w,band_foreign,reason");
                foreach (var (path, w, h, r) in all)
                {
                    sb.Append('"').Append(path.Replace("\"", "\"\"")).Append('"').Append(',')
                      .Append(w).Append(',').Append(h).Append(',')
                      .Append((int)r.Verdict).Append(',').Append(r.Measured ? 1 : 0).Append(',')
                      .Append(r.DigitHeight.ToString("R", inv)).Append(',')
                      .Append(r.DigitCount).Append(',').Append(r.ZeroCount).Append(',')
                      .Append(r.ZeroWidthRatio.ToString("R", inv)).Append(',')
                      .Append(r.ZeroHoleWidthRatio.ToString("R", inv)).Append(',')
                      .Append(r.ZeroEdgeFill.ToString("R", inv)).Append(',')
                      .Append(r.HasDot ? 1 : 0).Append(',')
                      .Append(r.DotHeightRatio.ToString("R", inv)).Append(',')
                      .Append(r.DotWidthRatio.ToString("R", inv)).Append(',')
                      .Append(r.BandForeignRatio.ToString("R", inv)).Append(',')
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
