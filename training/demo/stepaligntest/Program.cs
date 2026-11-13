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

namespace Ssp.StepAlignTest
{
    /// <summary>
    /// 对一批图跑 <see cref="StepAlignCheck.Check"/>, 逐张打印结论, 有时间轴的写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/stepaligntest
    ///   dotnet run -c Release -- 图或目录或@清单 [...] [--out 结果.csv]
    ///
    ///   给的是文件就逐张打印; 给的是目录(递归)或 @清单(一行一个路径)就只打印可疑的和汇总。
    ///   csv 里每个圆点的量值用 R 格式(往返精确), 可以和 training/step_align.py 的 m 列逐位比。
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
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--out" && i + 1 < args.Length) outCsv = args[++i];
                else inputs.Add(args[i]);
            }
            if (inputs.Count == 0)
            {
                Console.Error.WriteLine("用法: StepAlignTest 图或目录或@清单 [...] [--out 结果.csv]");
                return 2;
            }

            var files = new List<string>();
            bool many = false;
            foreach (var p in inputs)
            {
                if (p.StartsWith("@"))
                {
                    many = true;
                    files.AddRange(File.ReadAllLines(p.Substring(1)).Select(l => l.Trim()).Where(l => l.Length > 0));
                }
                else if (Directory.Exists(p))
                {
                    many = true;
                    files.AddRange(Directory.EnumerateFiles(p, "*", SearchOption.AllDirectories)
                        .Where(f => Exts.Contains(Path.GetExtension(f).ToLowerInvariant())));
                }
                else if (File.Exists(p)) files.Add(p);
                else { Console.Error.WriteLine($"不存在: {p}"); return 2; }
            }
            files.Sort(StringComparer.Ordinal);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var results = new ConcurrentBag<(string path, int w, int h, StepAlignResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                StepAlignResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) return;
                    w = img.Width; h = img.Height;
                    r = StepAlignCheck.Check(img);
                }
                catch (Exception e)
                {
                    r = new StepAlignResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
            });

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            var ci = CultureInfo.InvariantCulture;
            foreach (var (path, w, h, r) in all)
                if (!many || r.Verdict == StepAlignVerdict.Suspicious)
                    Console.WriteLine($"{r.Verdict,-15} {(r.Measured ? r.Score.ToString("0.000", ci) : "     ")}  {w}x{h}  {path}\n                {r.Reason}");

            var tl = all.Where(t => t.r.TimelineFound).ToList();
            Console.WriteLine();
            Console.WriteLine($"图 {all.Count:N0} 张, 有时间轴 {tl.Count:N0}, 用时 {sw.Elapsed.TotalSeconds:F0} 秒");
            Console.WriteLine($"  Suspicious       {tl.Count(t => t.r.Verdict == StepAlignVerdict.Suspicious):N0}");
            Console.WriteLine($"  Ok               {tl.Count(t => t.r.Verdict == StepAlignVerdict.Ok):N0}");
            Console.WriteLine($"  CannotDetermine  {tl.Count(t => t.r.Verdict == StepAlignVerdict.CannotDetermine):N0}  (有时间轴但量不了)");

            if (outCsv != null)
            {
                using var o = new StreamWriter(outCsv, false, new UTF8Encoding(true));
                o.WriteLine("path,W,H,verdict,D,n,score,sp_text,sp_date,m,reason");
                foreach (var (path, w, h, r) in tl)
                {
                    string m = string.Join(";", r.Circles.Select(c => c == null ? "null" : string.Join("|",
                        c.Top.ToString("R", ci), c.Bottom.ToString("R", ci), c.TextBottom.ToString("R", ci),
                        c.TextLeft.ToString(ci), c.TextB.ToString(ci), c.DateMid.ToString("R", ci),
                        c.DateLeft.ToString(ci), c.DateA.ToString(ci), c.DateH.ToString(ci))));
                    o.WriteLine(string.Join(",", Csv(path), w, h, r.Verdict, r.Diameter.ToString("R", ci), r.CircleCount,
                        r.Measured ? r.Score.ToString("R", ci) : "", r.Measured ? r.SpreadText.ToString("R", ci) : "",
                        r.Measured ? r.SpreadDate.ToString("R", ci) : "", m, Csv(r.Reason)));
                }
                Console.WriteLine($"csv -> {outCsv}");
            }
            return 0;
        }

        static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
    }
}
