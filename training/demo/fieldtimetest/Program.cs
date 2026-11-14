using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Ssp;

namespace Ssp.FieldTimeTest
{
    /// <summary>
    /// <see cref="FieldTimeCheck.Check"/> 的测试。
    ///
    /// 用法:
    ///   cd training/demo/fieldtimetest
    ///   dotnet run -c Release -- --selftest              固定用例(和 field_time_scan.py --selftest 同一批), 全过退出码 0
    ///   dotnet run -c Release -- 输入.jsonl 输出.jsonl    逐行跑规则, 和 field_time_scan.py --rules-jsonl 的输出逐条比
    ///
    ///   输入每行 {"id", "pay", "create", "order", "upload"(14 位数字或空), "name"(upload 空时从文件名取)}。
    ///   退出码: 0 = 跑完 / 全过; 1 = 有用例没过; 2 = 参数不对。
    /// </summary>
    static class Program
    {
        // (支付时间, 创建时间, 订单号, 上传 14 位或文件名, 期望, 说明), 和 field_time_scan.py 的 SELFTEST 一样
        static readonly (string? Pay, string? Create, string? Order, string? Upload, int Want, string Why)[] Cases =
        {
            (null, "2026-11-14 04:35:45", "20261114200040011100059255558788", "20261114043520", 2, "11-14 假图: 创建时间比上传晚 25 秒"),
            ("2026-11-01 04:45:24", null, "20261101044526584798398267229281", "x_20261101044657.jpg", 2, "11 月箭头判不出的假图: 订单号 = 支付时间 + 2 秒"),
            ("2026-11-04 07:28:01", null, "20261104072802950852632894448512", "20261104072932", 2, "同上 + 1 秒"),
            ("2026-11-03 22:45:06", null, "2026110323001442801444266253", "20261103224543", 1, "真图收钱码: 2300 格式不看, 上传晚 37 秒"),
            ("2026-11-03 23:00:10", null, "2026110323001442801444266253", "20261103230100", 1, "整点撞上 23:00:14 的真图, 2300 格式不看"),
            ("2026-11-01 20:00:38", null, "20261101200040011100390084455147", "20261101200110", 1, "转账 2000 格式撞 20:00:40, 不看"),
            ("2026-11-03 22:20:16", null, "20261103221950013887", "20261103222100", 1, "商家订单号误传: 支付前 26 秒, 不判"),
            ("2026-11-03 22:20:16", null, "", "20261103222016", 1, "上传和页面同一秒: 不判"),
            ("2026-11-03 22:20:17", null, "", "20261103222016", 2, "页面比上传晚 1 秒"),
            ("2026-11-03 22:21:16", null, "", "20261103222016", 2, "晚 60 秒"),
            ("2026-11-03 22:21:17", null, "", "20261103222016", 1, "晚 61 秒: 不判(多半认错字)"),
            ("2026-11-03 22:20:16", null, "", "20261103142016", 1, "晚 8 小时(上传传成 UTC): 不判"),
            ("2026-11-03 22:20:16", null, "20261103222031999", null, 2, "订单号晚 15 秒"),
            ("2026-11-03 22:20:16", null, "20261103222032999", null, 1, "晚 16 秒: 不判"),
            ("2026-11-03 22:20:16", null, "20261103222015999", null, 1, "早 1 秒: 不判"),
            ("2026-11-03 22:20", null, "20261103222016999", "20261103222000", 0, "页面时间没有秒: 判不了"),
            (null, null, "20261103222016999", "20261103222000", 0, "没有页面时间: 判不了"),
            ("2026-11-03 22:20:16", null, "123", null, 0, "没上传时间, 订单号太短: 判不了"),
            ("支付时间2026-11-0322:20:16", null, "订单号 2026 1103 2220 18 77", null, 2, "带标签、丢空格、订单号带空格"),
            ("2026/11/03 22：20：16", null, "", "20261103222020", 1, "斜杠日期 全角冒号, 上传晚 4 秒"),
            ("2026-02-30 22:20:16", "2026-11-03 22:20:16", "", "20261103222010", 2, "支付时间不合法 -> 用创建时间"),
            ("2026-11-03 22:20:16", null, "２０２６１１０３２２２０１８", null, 0, "全角数字不认"),
            ("2026-11-03 24:00:00", null, "", "20261103222010", 0, "24 点不合法"),
            ("2026-11-03 22:20:16", null, "", "a/b_20261103222016.jpg", 1, "从文件名取上传时间"),
        };

        static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            if (args.Length == 1 && args[0] == "--selftest") return SelfTest();
            if (args.Length == 2) return RunJsonl(args[0], args[1]);
            Console.Error.WriteLine("用法: FieldTimeTest --selftest | 输入.jsonl 输出.jsonl");
            return 2;
        }

        static DateTime? Upload(string? s)
        {
            if (string.IsNullOrEmpty(s)) return null;
            // 14 位数字走和文件名同一段代码(Python 那边也是同一个 _from14)
            return Regex.IsMatch(s, "^[0-9]{14}$")
                ? FieldTimeCheck.UploadTimeFromFileName("_" + s + ".jpg")
                : FieldTimeCheck.UploadTimeFromFileName(s);
        }

        static int SelfTest()
        {
            int bad = 0;
            foreach (var c in Cases)
            {
                var r = FieldTimeCheck.Check(c.Pay, c.Create, c.Order, Upload(c.Upload));
                bool ok = (int)r.Verdict == c.Want;
                if (!ok) bad++;
                Console.WriteLine($"{(ok ? "PASS" : "FAIL")} v={(int)r.Verdict} want={c.Want} d1={r.PageAfterUploadSeconds} d2={r.OrderNoAfterPageSeconds}  {c.Why}  | {r.Reason}");
            }
            // Kind = Utc 的上传时间先加 8 小时; 毫秒去掉
            var utc = FieldTimeCheck.Check(null, "2026-11-14 04:35:45", null, new DateTime(2026, 11, 13, 20, 35, 20, DateTimeKind.Utc));
            bool okUtc = utc.Verdict == FieldTimeVerdict.Suspicious && utc.PageAfterUploadSeconds == 25;
            var ms = FieldTimeCheck.Check(null, "2026-11-14 04:35:21", null, new DateTime(2026, 11, 14, 4, 35, 20, 900));
            bool okMs = ms.Verdict == FieldTimeVerdict.Suspicious && ms.PageAfterUploadSeconds == 1;
            Console.WriteLine($"{(okUtc ? "PASS" : "FAIL")} UTC 上传时间加 8 小时: {utc.Reason}");
            Console.WriteLine($"{(okMs ? "PASS" : "FAIL")} 上传时间毫秒去掉: {ms.Reason}");
            if (!okUtc) bad++;
            if (!okMs) bad++;
            Console.WriteLine(bad == 0 ? "FIELD_SELFTEST_OK" : $"FIELD_SELFTEST_FAILED {bad}");
            return bad == 0 ? 0 : 1;
        }

        static string? Str(JsonElement e, string name)
        {
            if (!e.TryGetProperty(name, out var v) || v.ValueKind == JsonValueKind.Null) return null;
            return v.GetString();
        }

        static string Ts(DateTime? t) => t == null ? "" : t.Value.ToString("yyyyMMddHHmmss", System.Globalization.CultureInfo.InvariantCulture);

        static int RunJsonl(string src, string dst)
        {
            int n = 0;
            var opt = new JsonWriterOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
            using var w = new StreamWriter(dst, false, new UTF8Encoding(false)) { NewLine = "\n" };
            foreach (var line in File.ReadLines(src, Encoding.UTF8))
            {
                if (line.Trim().Length == 0) continue;
                using var doc = JsonDocument.Parse(line);
                var x = doc.RootElement;
                string? up = Str(x, "upload");
                DateTime? upload = !string.IsNullOrEmpty(up) ? Upload(up) : FieldTimeCheck.UploadTimeFromFileName(Str(x, "name"));
                var r = FieldTimeCheck.Check(Str(x, "pay"), Str(x, "create"), Str(x, "order"), upload);

                using var ms = new MemoryStream();
                using (var jw = new Utf8JsonWriter(ms, opt))
                {
                    jw.WriteStartObject();
                    jw.WriteString("id", Str(x, "id"));
                    jw.WriteNumber("v", (int)r.Verdict);
                    jw.WriteBoolean("r1", r.LaterThanUpload);
                    jw.WriteBoolean("r2", r.OrderNoIsPayTime);
                    jw.WriteString("field", r.PageTimeField);
                    jw.WriteString("pt", Ts(r.PageTime));
                    jw.WriteString("up", Ts(r.UploadTime));
                    if (r.PageAfterUploadSeconds == null) jw.WriteNull("d1"); else jw.WriteNumber("d1", (long)r.PageAfterUploadSeconds.Value);
                    jw.WriteString("digits", r.OrderNoDigits);
                    jw.WriteBoolean("fmt", r.OrderNoIsAlipayFormat);
                    jw.WriteString("ot", Ts(r.OrderNoTime));
                    if (r.OrderNoAfterPageSeconds == null) jw.WriteNull("d2"); else jw.WriteNumber("d2", (long)r.OrderNoAfterPageSeconds.Value);
                    jw.WriteEndObject();
                }
                w.WriteLine(Encoding.UTF8.GetString(ms.ToArray()));
                n++;
            }
            Console.WriteLine($"RULES_JSONL_DONE {n}");
            return 0;
        }
    }
}
