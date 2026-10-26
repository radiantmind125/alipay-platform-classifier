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

namespace Ssp.CircleTest
{
    /// <summary>
    /// 对一批图跑 <see cref="CircleCheck.Check"/>, 逐张打印结论, 量到的写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/circletest
    ///   dotnet run -c Release -- 图或目录 [图或目录 ...] [--list 清单.txt] [--out 结果.csv]
    ///
    ///   给的是文件就逐张打印; 给的是目录或清单就只打印可疑的和汇总。
    ///   csv 的列和 training/circle_scan.py 的输出对得上(没量到的写 nan)。
    ///   退出码: 0 = 跑完; 2 = 参数不对。
    /// </summary>
    static class Program
    {
        static readonly string[] Exts = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

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
                Console.Error.WriteLine("用法: CircleTest 图或目录 [图或目录 ...] [--list 清单.txt] [--out 结果.csv]");
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
            var results = new ConcurrentBag<(string path, int w, int h, CircleResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                CircleResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) return;
                    w = img.Width; h = img.Height;
                    r = CircleCheck.Check(img);
                }
                catch (Exception e)
                {
                    r = new CircleResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
            });
            sw.Stop();

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            foreach (var (path, w, h, r) in all)
            {
                if (quiet && r.Verdict != CircleVerdict.Suspicious) continue;
                Console.WriteLine($"{(int)r.Verdict} {r.Verdict,-15} {Path.GetFileName(path)}  {w}x{h}  "
                    + (r.RingFound ? $"圆度 {N(r.Roundness)} 外径 {N(r.RingDiameter)}  " : "")
                    + r.Reason);
            }
            int ok = all.Count(t => t.r.Verdict == CircleVerdict.Ok);
            int sus = all.Count(t => t.r.Verdict == CircleVerdict.Suspicious);
            int cd = all.Count(t => t.r.Verdict == CircleVerdict.CannotDetermine);
            Console.WriteLine($"共 {all.Count} 张  Ok {ok}  Suspicious {sus}  CannotDetermine {cd}  用时 {sw.Elapsed.TotalSeconds:F1}s");

            if (outCsv != null)
            {
                var sb = new StringBuilder();
                sb.AppendLine("path,W,H,verdict,reason,ring_found,roundness,ratio_hv,ring_d,ring_d_adh,n_rays,check_found,"
                              + "ck_w,ck_h,ck_dx,ck_dy,row_found,lead_gap,lead_tall,slot_frac,foreign,title_bright");
                foreach (var (path, w, h, r) in all)
                {
                    sb.Append('"').Append(path.Replace("\"", "\"\"")).Append('"').Append(',')
                      .Append(w).Append(',').Append(h).Append(',')
                      .Append((int)r.Verdict).Append(',')
                      .Append('"').Append(r.Reason.Replace("\"", "\"\"")).Append('"').Append(',')
                      .Append(r.RingFound ? 1 : 0).Append(',')
                      .Append(N(r.Roundness)).Append(',').Append(N(r.RatioHV)).Append(',')
                      .Append(N(r.RingDiameter)).Append(',').Append(N(r.RingToAmount)).Append(',')
                      .Append(r.RayCount).Append(',').Append(r.CheckFound ? 1 : 0).Append(',')
                      .Append(N(r.CheckWidth)).Append(',').Append(N(r.CheckHeight)).Append(',')
                      .Append(N(r.CheckOffsetX)).Append(',').Append(N(r.CheckOffsetY)).Append(',')
                      .Append(r.TitleRowFound ? 1 : 0).Append(',')
                      .Append(N(r.LeadGap)).Append(',').Append(N(r.LeadTall)).Append(',')
                      .Append(N(r.SlotFraction)).Append(',').Append(N(r.ForeignRatio)).Append(',')
                      .Append(N(r.TitleBrightness))
                      .AppendLine();
                }
                File.WriteAllText(outCsv, sb.ToString(), new UTF8Encoding(true));
                Console.WriteLine($"写了 {outCsv}");
            }
            return 0;
        }

        static string N(double v) => double.IsNaN(v) ? "nan" : v.ToString("R", Inv);
    }
}
