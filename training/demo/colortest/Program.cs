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

namespace Ssp.ColorTest
{
    /// <summary>
    /// 对一批图跑 <see cref="TextColorCheck.Check"/>, 逐张打印结论, 量到的写进 csv。
    ///
    /// 用法:
    ///   cd training/demo/colortest
    ///   dotnet run -c Release -- 图或目录 [图或目录 ...] [--out 结果.csv]
    ///
    ///   给的是文件就逐张打印; 给的是目录就递归扫, 只打印可疑的和汇总。
    ///   csv 的列和 training/color_scan.py 的输出对得上(verdict / black_share / grey_share / rows / label_core / cores)。
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
                Console.Error.WriteLine("用法: ColorTest 图或目录 [图或目录 ...] [--out 结果.csv]");
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
            var results = new ConcurrentBag<(string path, int w, int h, TextColorResult r)>();
            Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, f =>
            {
                TextColorResult r;
                int w = 0, h = 0;
                try
                {
                    // Cv2.ImRead 在 Windows 上读不了中文路径, 先读字节再解码
                    using var img = Cv2.ImDecode(File.ReadAllBytes(f), ImreadModes.Color);
                    if (img.Empty()) return;
                    w = img.Width; h = img.Height;
                    r = TextColorCheck.Check(img);
                }
                catch (Exception e)
                {
                    r = new TextColorResult { Reason = "出错: " + e.Message };
                }
                results.Add((f, w, h, r));
            });

            var all = results.OrderBy(t => t.path, StringComparer.Ordinal).ToList();
            foreach (var (path, w, h, r) in all)
                if (!anyDir || r.Verdict == TextColorVerdict.Suspicious)
                    Console.WriteLine($"{r.Verdict,-15} {w}x{h}  {path}\n                {r.Reason}");

            var meas = all.Where(t => t.r.Measured).ToList();
            Console.WriteLine();
            Console.WriteLine($"图 {all.Count:N0} 张, 量到账单详情卡片的 {meas.Count:N0} 张, 用时 {sw.Elapsed.TotalSeconds:F0} 秒");
            Console.WriteLine($"  Suspicious       {meas.Count(t => t.r.Verdict == TextColorVerdict.Suspicious):N0}");
            Console.WriteLine($"  Ok               {meas.Count(t => t.r.Verdict == TextColorVerdict.Ok):N0}");
            Console.WriteLine($"  CannotDetermine  {meas.Count(t => t.r.Verdict == TextColorVerdict.CannotDetermine):N0}  (量到了但不判)");

            if (outCsv != null)
            {
                // 每张都写(包括没量到的), 方便和 Python 逐张对
                var ci = CultureInfo.InvariantCulture;
                using var sw2 = new StreamWriter(outCsv, false, new UTF8Encoding(true));
                sw2.WriteLine("path,W,H,verdict,black_share,grey_share,rows,rows_black,rows_grey,label_core,edge,pinyin,bg,cores,reason");
                foreach (var (path, w, h, r) in all)
                {
                    sw2.WriteLine(string.Join(",", Csv(path), w, h, r.Verdict,
                        r.Rows > 0 ? r.BlackShare.ToString("0.000", ci) : "", r.Rows > 0 ? r.GreyShare.ToString("0.000", ci) : "",
                        r.Rows, r.RowsBlack, r.RowsGrey, r.Rows > 0 ? r.LabelCore.ToString(ci) : "", r.Rows > 0 ? r.Edge.ToString(ci) : "",
                        r.PinyinChecked ? (r.Pinyin ? "1" : "0") : "", r.Background, string.Join(" ", r.ValueCores), Csv(r.Reason)));
                }
                Console.WriteLine($"csv -> {outCsv}");
            }
            return 0;
        }

        static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
    }
}
