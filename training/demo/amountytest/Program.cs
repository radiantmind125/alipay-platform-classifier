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

namespace Ssp.AmountYTest
{
    /// <summary>
    /// 对一批图跑 <see cref="AmountYCheck.Check"/>。
    ///
    /// 用法:
    ///   cd training/demo/amountytest
    ///   dotnet run -c Release -- 图或目录或@清单 [...] [--out 结果.csv]
    ///
    ///   给的是文件就逐张打印; 给的是目录(递归)或 @清单(一行一个路径)就只打印可疑的和汇总。
    ///   csv 的列和 training/amount_y_check.py 的 nm_a / nm_b / am_a / am_b / st_a / st_b / row_a / row_b / up / d / v 一一对应, 可以逐张比。
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
                Console.Error.WriteLine("用法: AmountYTest 图或目录或@清单 [...] [--out 结果.csv]");
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
            var results = new ConcurrentBag<(string path, int w, int h, AmountYResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                AmountYResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码; 和 amount_y_check.py 一样解码成彩色
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) { results.Add((f, 0, 0, new AmountYResult { Reason = "读不了" })); return; }
                    w = img.Width; h = img.Height;
                    r = AmountYCheck.Check(img);
                }
                catch (Exception e)
                {
                    r = new AmountYResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
            });

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            var ci = CultureInfo.InvariantCulture;
            foreach (var (path, w, h, r) in all)
                if (!many || r.Verdict == AmountYVerdict.Suspicious)
                    Console.WriteLine($"{r.Verdict,-15} {(r.Measured ? r.Ratio.ToString("0.000", ci) : "     ")}  {w}x{h}  {path}\n                {r.Reason}");

            Console.WriteLine();
            Console.WriteLine($"图 {all.Count:N0} 张, 用时 {sw.Elapsed.TotalSeconds:F1} 秒");
            Console.WriteLine($"  Suspicious       {all.Count(t => t.r.Verdict == AmountYVerdict.Suspicious):N0}");
            Console.WriteLine($"  Ok               {all.Count(t => t.r.Verdict == AmountYVerdict.Ok):N0}");
            Console.WriteLine($"  CannotDetermine  {all.Count(t => t.r.Verdict == AmountYVerdict.CannotDetermine):N0}");

            if (outCsv != null)
            {
                using var o = new StreamWriter(outCsv, false, new UTF8Encoding(true));
                o.WriteLine("path,W,H,v,nm_a,nm_b,am_a,am_b,st_a,st_b,row_a,row_b,up,d,reason");
                foreach (var (path, w, h, r) in all)
                {
                    string F(int v) => r.Found ? v.ToString(ci) : "";
                    o.WriteLine(string.Join(",", Csv(path), w, h, (int)r.Verdict,
                        F(r.NameTop), F(r.NameBottom), F(r.AmountTop), F(r.AmountBottom), F(r.StatusTop), F(r.StatusBottom),
                        F(r.RowTop), F(r.RowBottom), F(r.Up), F(r.D), Csv(r.Reason)));
                }
                Console.WriteLine($"csv -> {outCsv}");
            }
            return 0;
        }

        static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
    }
}
