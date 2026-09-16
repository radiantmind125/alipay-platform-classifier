using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using OpenCvSharp;
using Ssp;

namespace Ssp.RowDump
{
    /// <summary>
    /// 把带拼音的截图按**正文行**裁出来。
    ///
    /// 每张图做三件:
    ///   1. <see cref="PinyinCheck.Check"/>  判是不是拼音页
    ///   2. <see cref="TextRegion.FindTextRows"/> 带 dropAnnotationRows 拿正文行
    ///   3. 按行裁出来存成 png
    ///
    /// 同时出一份清单, 记每一行来自哪张图、在原图的哪个位置 ——
    /// 后面 OCR 出文字之后要按这个清单配对。
    /// </summary>
    static class Program
    {
        static readonly string[] Exts = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

        /// 裁的时候上下留的余量(像素)。
        /// ★ 不能留大: 留多了会把上面那行拼音带进来, 白裁了。
        const int Pad = 2;

        /// 太矮的行不要 —— 多半是分隔线或者噪声
        const int MinRowHeight = 8;

        static int Main(string[] args)
        {
            string? input = null, output = null;
            int limit = 0;
            bool onlyPinyin = true;

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--input" when i + 1 < args.Length: input = args[++i]; break;
                    case "--output" when i + 1 < args.Length: output = args[++i]; break;
                    case "--limit" when i + 1 < args.Length:
                        int.TryParse(args[++i], out limit); break;
                    case "--all":
                        onlyPinyin = false; break;   // 连不带拼音的也裁, 用来做对照
                    default: break;
                }
            }

            if (input == null || output == null)
            {
                Console.WriteLine("用法: dotnet run -- --input 图目录 --output 行图目录 [--limit N] [--all]");
                Console.WriteLine("  --all  连判为不带拼音的也裁(默认只裁带拼音的)");
                return 2;
            }
            if (!Directory.Exists(input))
            {
                Console.WriteLine($"目录不在: {input}");
                return 2;
            }

            var files = Directory.EnumerateFiles(input)
                .Where(f => Exts.Contains(Path.GetExtension(f).ToLowerInvariant()))
                .OrderBy(f => f, StringComparer.Ordinal)
                .ToList();
            if (limit > 0) files = files.Take(limit).ToList();
            if (files.Count == 0)
            {
                Console.WriteLine("没找到图");
                return 2;
            }

            Directory.CreateDirectory(output);
            var manifestPath = Path.Combine(output, "_rows.csv");

            Console.WriteLine($"要处理 {files.Count:N0} 张");

            int nPinyin = 0, nSkipped = 0, nUnreadable = 0, nRows = 0;
            // 判定分布, 用来核对和 Python 那边挑图的口径一致不一致
            var verdicts = new Dictionary<PinyinVerdict, int>();

            using var sw = new StreamWriter(manifestPath, false, new UTF8Encoding(true));
            sw.WriteLine("row_file,source,row_index,x,y,w,h,pinyin_ratio,verdict");

            for (int i = 0; i < files.Count; i++)
            {
                var f = files[i];
                Mat? img = null;
                try
                {
                    // ★ 不用 Cv2.ImRead: 路径带中文时它会读不出来。先读字节再解码。
                    var bytes = File.ReadAllBytes(f);
                    img = Cv2.ImDecode(bytes, ImreadModes.Color);
                    if (img == null || img.Empty()) { nUnreadable++; continue; }

                    var pr = PinyinCheck.Check(img);
                    verdicts.TryGetValue(pr.Verdict, out int c);
                    verdicts[pr.Verdict] = c + 1;

                    if (onlyPinyin && pr.Verdict != PinyinVerdict.HasPinyin)
                    {
                        nSkipped++;
                        continue;
                    }
                    nPinyin++;

                    // ★ dropAnnotationRows: true —— 拼音那几行在这一步就被滤掉了
                    var rows = TextRegion.FindTextRows(img, dropAnnotationRows: true);
                    string stem = Path.GetFileNameWithoutExtension(f);

                    int kept = 0;
                    for (int r = 0; r < rows.Count; r++)
                    {
                        var rc = rows[r];
                        if (rc.Height < MinRowHeight) continue;

                        int y0 = Math.Max(0, rc.Y - Pad);
                        int y1 = Math.Min(img.Rows, rc.Y + rc.Height + Pad);
                        if (y1 - y0 < MinRowHeight) continue;

                        // 横向裁整宽: 左标签和右取值在同一行, 都要
                        using var crop = new Mat(img, new Rect(0, y0, img.Cols, y1 - y0));
                        string name = $"{stem}_r{kept:D2}.png";
                        var dst = Path.Combine(output, name);
                        Cv2.ImEncode(".png", crop, out var buf);
                        File.WriteAllBytes(dst, buf);

                        sw.WriteLine(string.Join(",",
                            name,
                            Quote(f),
                            kept.ToString(CultureInfo.InvariantCulture),
                            "0",
                            y0.ToString(CultureInfo.InvariantCulture),
                            img.Cols.ToString(CultureInfo.InvariantCulture),
                            (y1 - y0).ToString(CultureInfo.InvariantCulture),
                            pr.Ratio.ToString("F4", CultureInfo.InvariantCulture),
                            pr.Verdict.ToString()));
                        kept++;
                        nRows++;
                    }
                }
                catch (Exception ex)
                {
                    // 一张图坏了不能让整批停下来
                    Console.WriteLine($"  [跳过] {Path.GetFileName(f)}: {ex.GetType().Name}");
                    nUnreadable++;
                }
                finally
                {
                    img?.Dispose();
                }

                if ((i + 1) % 200 == 0)
                {
                    sw.Flush();
                    Console.WriteLine($"  {i + 1:N0}/{files.Count:N0}   裁出 {nRows:N0} 行");
                }
            }

            Console.WriteLine();
            Console.WriteLine(new string('=', 60));
            Console.WriteLine($"判为带拼音   {nPinyin:N0}");
            Console.WriteLine($"跳过(不带)   {nSkipped:N0}");
            Console.WriteLine($"读不了       {nUnreadable:N0}");
            Console.WriteLine($"裁出正文行   {nRows:N0}");
            if (nPinyin > 0)
                Console.WriteLine($"平均每张     {(double)nRows / nPinyin:F1} 行");
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("判定分布:");
            foreach (var kv in verdicts.OrderByDescending(k => k.Value))
                Console.WriteLine($"  {kv.Key,-16} {kv.Value:N0}");
            Console.WriteLine();
            Console.WriteLine($"清单 -> {manifestPath}");
            Console.WriteLine("★ 下一步: 对这些行图跑 OCR, 出 (行图, 文字) 的配对。");
            Console.WriteLine("  验收看纯拉丁行的比例 —— 整页送 OCR 是 8 张 129 行,");
            Console.WriteLine("  按行裁要能压到个位数, 这条路才算通。");
            return 0;
        }

        static string Quote(string s) =>
            s.Contains(',') || s.Contains('"')
                ? "\"" + s.Replace("\"", "\"\"") + "\""
                : s;
    }
}
