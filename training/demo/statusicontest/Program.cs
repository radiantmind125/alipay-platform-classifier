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

namespace Ssp.StatusIconTest
{
    /// <summary>
    /// 对一批图跑 <see cref="StatusIconCheck.Check"/>, 打印汇总和判可疑的, 全部写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/statusicontest
    ///   dotnet run -c Release -- 图或目录 [图或目录 ...] [--out 结果.csv]
    ///
    ///   csv 的列和 training/status_icon.py 对得上(verdict kind 圆的位置大小 圆里蓝字的形状), 可以逐行比两边一不一样。
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
                Console.Error.WriteLine("用法: StatusIconTest 图或目录 [图或目录 ...] [--out 结果.csv]");
                return 2;
            }
            var files = new List<string>();
            foreach (var p in inputs)
            {
                if (Directory.Exists(p))
                    files.AddRange(Directory.EnumerateFiles(p, "*", SearchOption.AllDirectories)
                        .Where(f => Exts.Contains(Path.GetExtension(f).ToLowerInvariant())));
                else if (File.Exists(p)) files.Add(p);
                else { Console.Error.WriteLine($"不存在: {p}"); return 2; }
            }
            files.Sort(StringComparer.Ordinal);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var results = new ConcurrentBag<(string path, int w, int h, StatusIconResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                StatusIconResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) { r = new StatusIconResult { Reason = "读不出" }; }
                    else { w = img.Width; h = img.Height; r = StatusIconCheck.Check(img); }
                }
                catch (Exception e) { r = new StatusIconResult { Reason = "出错: " + e.Message }; }
                results.Add((f, w, h, r));
            });
            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            Console.WriteLine($"  图 {all.Count:N0} 张   {sw.Elapsed.TotalSeconds:F1} 秒   {all.Count / Math.Max(sw.Elapsed.TotalSeconds, 1e-9):F1} 张/秒");
            foreach (var g in all.GroupBy(t => $"{t.r.Verdict}/{t.r.Kind}").OrderByDescending(g => g.Count()))
                Console.WriteLine($"    {g.Key,-34}{g.Count(),8:N0}");
            foreach (var t in all.Where(t => t.r.Verdict == StatusIconVerdict.Suspicious))
                Console.WriteLine($"  可疑 {Path.GetFileName(t.path)}  {t.r.Reason}");
            if (outCsv != null)
            {
                var ci = CultureInfo.InvariantCulture;
                var sb = new StringBuilder("path,W,H,verdict,kind,disk,ring,cx,top,d,white,filled,gw,gh,parts,off,reason\n");
                foreach (var t in all)
                {
                    var r = t.r;
                    sb.Append('"').Append(t.path).Append("\",").Append(t.w).Append(',').Append(t.h).Append(',')
                      .Append(r.Verdict).Append(',').Append(r.Kind).Append(',').Append(r.DiskFound ? 1 : 0).Append(',').Append(r.RingFound ? 1 : 0).Append(',')
                      .Append(r.CenterX.ToString("F4", ci)).Append(',').Append(r.Top.ToString("F4", ci)).Append(',')
                      .Append(r.Diameter.ToString("F4", ci)).Append(',').Append(r.WhiteFill.ToString("F4", ci)).Append(',')
                      .Append(r.FilledFill.ToString("F4", ci)).Append(',').Append(r.GlyphWidth.ToString("F4", ci)).Append(',').Append(r.GlyphHeight.ToString("F4", ci)).Append(',')
                      .Append(r.GlyphParts).Append(',').Append(r.GlyphOffset.ToString("F4", ci)).Append(",\"").Append(r.Reason.Replace("\"", "'")).Append("\"\n");
                }
                File.WriteAllText(outCsv, sb.ToString(), new UTF8Encoding(true));
                Console.WriteLine($"  -> {outCsv}");
            }
            return 0;
        }
    }
}
