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

namespace Ssp.ArrowTest
{
    /// <summary>
    /// 对一批图跑 <see cref="ArrowGapCheck.Check"/>, 逐张打印结论, 量到的写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/arrowtest
    ///   dotnet run -c Release -- 图或目录 [图或目录 ...] [--out 结果.csv]
    ///
    ///   给的是文件就逐张打印; 给的是目录就递归扫, 只打印可疑的和汇总。
    ///   csv 的列和 training/arrow_scan.py 的输出对得上(verdict / ratio / gap / arrow_h / margin / n / ys)。
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
                Console.Error.WriteLine("用法: ArrowTest 图或目录 [图或目录 ...] [--out 结果.csv]");
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
            var results = new ConcurrentBag<(string path, int w, int h, ArrowGapResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                ArrowGapResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) return;
                    w = img.Width; h = img.Height;
                    r = ArrowGapCheck.Check(img);
                }
                catch (Exception e)
                {
                    r = new ArrowGapResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
            });

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            foreach (var (path, w, h, r) in all)
                if (!anyDir || r.Verdict == ArrowGapVerdict.Suspicious)
                    Console.WriteLine($"{r.Verdict,-15} {w}x{h}  {path}\n                {r.Reason}");

            var meas = all.Where(t => t.r.Measured).ToList();
            Console.WriteLine();
            Console.WriteLine($"图 {all.Count:N0} 张, 量到账单管理那几行的 {meas.Count:N0} 张, 用时 {sw.Elapsed.TotalSeconds:F0} 秒");
            Console.WriteLine($"  Suspicious       {meas.Count(t => t.r.Verdict == ArrowGapVerdict.Suspicious):N0}");
            Console.WriteLine($"  Ok               {meas.Count(t => t.r.Verdict == ArrowGapVerdict.Ok):N0}");
            Console.WriteLine($"  CannotDetermine  {meas.Count(t => t.r.Verdict == ArrowGapVerdict.CannotDetermine):N0}  (量到了但不判)");

            if (outCsv != null)
            {
                var ci = CultureInfo.InvariantCulture;
                using var sw2 = new StreamWriter(outCsv, false, new UTF8Encoding(true));
                sw2.WriteLine("path,W,H,verdict,ratio,gap,arrow_h,arrow_w,arrow_gray,margin,n,ys,pinyin,reason");
                foreach (var (path, w, h, r) in meas)
                {
                    sw2.WriteLine(string.Join(",", Csv(path), w, h, r.Verdict,
                        r.Ratio.ToString("0.0000", ci), r.Gap.ToString("0.###", ci), r.ArrowHeight.ToString("0.###", ci),
                        r.ArrowWidth.ToString("0.###", ci), r.ArrowGray, r.Margin, r.RowCount, string.Join(" ", r.RowTops),
                        r.PinyinChecked ? (r.Pinyin ? "1" : "0") : "", Csv(r.Reason)));
                }
                Console.WriteLine($"csv -> {outCsv}");
            }
            return 0;
        }

        static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
    }
}
