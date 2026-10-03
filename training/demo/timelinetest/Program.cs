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

namespace Ssp.TimelineTest
{
    /// <summary>
    /// 对一批图跑 <see cref="TimelineCheck.Check"/>, 逐张打印结论, 有时间轴的写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/timelinetest
    ///   dotnet run -c Release -- 图或目录 [图或目录 ...] [--out 结果.csv]
    ///
    ///   给的是文件就逐张打印; 给的是目录就递归扫, 只打印可疑的和汇总。
    ///   csv 的列和 training/timeline_scan.py 的输出对得上, 可以逐行比两边算得一不一样。
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
                Console.Error.WriteLine("用法: TimelineTest 图或目录 [图或目录 ...] [--out 结果.csv]");
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
            var results = new ConcurrentBag<(string path, int w, int h, TimelineResult r)>();
            int done = 0;
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                TimelineResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) return;
                    w = img.Width; h = img.Height;
                    r = TimelineCheck.Check(img);
                }
                catch (Exception e)
                {
                    r = new TimelineResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
                int d = System.Threading.Interlocked.Increment(ref done);
                if (anyDir && d % 2000 == 0) Console.WriteLine($"  {d:N0}/{files.Count:N0}");
            });

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            foreach (var (path, w, h, r) in all)
            {
                if (!anyDir || r.Verdict == TimelineVerdict.Suspicious)
                    Console.WriteLine($"{r.Verdict,-15} {Fmt(r)}  {w}x{h}  {path}\n                {r.Reason}");
            }

            var tl = all.Where(t => t.r.TimelineFound).ToList();
            Console.WriteLine();
            Console.WriteLine($"图 {all.Count:N0} 张, 有时间轴 {tl.Count:N0}, 用时 {sw.Elapsed.TotalSeconds:F0} 秒");
            Console.WriteLine($"  Suspicious       {tl.Count(t => t.r.Verdict == TimelineVerdict.Suspicious):N0}");
            Console.WriteLine($"  Ok               {tl.Count(t => t.r.Verdict == TimelineVerdict.Ok):N0}");
            Console.WriteLine($"  CannotDetermine  {tl.Count(t => t.r.Verdict == TimelineVerdict.CannotDetermine):N0}  (有时间轴但量不了)");

            if (outCsv != null)
            {
                var ci = CultureInfo.InvariantCulture;
                using var sw2 = new StreamWriter(outCsv, false, new UTF8Encoding(true));
                sw2.WriteLine("path,W,H,verdict,offset,offset_px,circle_left,value_left,D,n,circle_top,label,label_ink,reason");
                foreach (var (path, w, h, r) in tl)
                {
                    sw2.WriteLine(string.Join(",",
                        Csv(path), w, h, r.Verdict,
                        r.Measured ? r.Offset.ToString("0.0000", ci) : "",
                        r.Measured ? r.OffsetPixels.ToString(ci) : "",
                        r.CircleLeft, r.Measured ? r.ValueLeft.ToString(ci) : "",
                        r.Diameter.ToString("0.###", ci), r.CircleCount, r.CircleTop,
                        r.LabelFound ? 1 : 0, r.LabelInk.ToString("0.000", ci), Csv(r.Reason)));
                }
                Console.WriteLine($"csv -> {outCsv}");
            }
            return 0;
        }

        static string Fmt(TimelineResult r) =>
            r.Measured ? $"偏移 {r.Offset,7:+0.000;-0.000} ({r.OffsetPixels,3} px, 直径 {r.Diameter:0.#})" : "".PadRight(32);

        static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
    }
}
