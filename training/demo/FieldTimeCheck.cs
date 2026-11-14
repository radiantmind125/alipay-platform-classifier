#nullable enable

using System;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Ssp
{
    public enum FieldTimeVerdict
    {
        CannotDetermine = 0,   // 判不了(没读到页面时间, 或者上传时间和订单号都没有), 走正常流程
        Ok = 1,                // 两条都没犯
        Suspicious = 2,        // 假图: 页面时间比上传还晚, 或者订单号开头就是支付时间
    }

    public sealed class FieldTimeResult
    {
        public FieldTimeVerdict Verdict;
        public DateTime? PageTime;                 // 页面时间: 支付时间, 读不出来再用创建时间
        public string PageTimeField = "";          // "支付时间" / "创建时间" / ""
        public DateTime? UploadTime;               // 实际用的上传时间(去掉毫秒; Kind 是 Utc 的已加 8 小时)
        public double? PageAfterUploadSeconds;     // 页面时间 - 上传时间, 秒; 正数 = 页面比上传还晚
        public string OrderNoDigits = "";          // 订单号里的数字
        public bool OrderNoIsAlipayFormat;         // 订单号第 9~12 位是 2000 / 2100 / 2200 / 2300(支付宝自己的格式), 第二条不看
        public DateTime? OrderNoTime;              // 订单号前 14 位当成 年月日时分秒(不是合法时间就是 null)
        public double? OrderNoAfterPageSeconds;    // 订单号时间 - 页面时间, 秒
        public bool LaterThanUpload;               // 第一条犯了
        public bool OrderNoIsPayTime;              // 第二条犯了
        public string Reason = "";
    }

    /// <summary>
    /// 账单详情页两个字段时间的检查, 只看 OCR 读出来的字段, 不看图(2026-11-14 加, 和 ArrowGapCheck 一起发)。
    ///
    /// 第一条: 页面时间比凭证上传时间还晚 -> 可疑。
    ///   真图是先付款、再截图、再上传, 页面上的支付时间(或创建时间)一定在上传之前。
    ///   本地真图 313 张(带页面时间, 大多是 11 月): 上传都在页面时间之后, 最少 12 秒, 中位 40 秒, 没有一张反过来。
    ///   假图: 服务器上 ArrowGapCheck 抓到的 420 张里 112 张页面时间比上传晚 1~32 秒; 11-14 放过去的那张晚 25 秒
    ///   (创建时间 04:35:45, 凭证上传 04:35:20)。
    ///   只认晚 MinAheadSeconds ~ MaxAheadSeconds 秒(1~60 秒): 假图都在这个范围里;
    ///   晚几分钟以上的多半是认错了字(日期、小时认错一位)或者时区不对, 不判, 宁可放过不误判。
    ///
    /// 第二条: 订单号前 14 位就是页面上的支付时间(年月日时分秒), 晚 0~15 秒 -> 可疑。
    ///   支付宝自己的订单号是 日期 + 固定几位, 不是时间:
    ///     收钱码 / 商家 28 位   yyyyMMdd2300 14... / yyyyMMdd2200 14...
    ///     转账 32 位            yyyyMMdd2000 4001 1100...
    ///   本地真图 268 张有订单号, 没有一张是"日期 + 支付时分秒"。假图的订单号是造假工具按支付时间生成的:
    ///   ArrowGapCheck 抓到的 420 张里 43 张, 11 月服务器上箭头判不出的 3 张(GWCZ2116632447205687296 等)全是,
    ///   订单号时间比支付时间晚 0~10 秒。
    ///   ★ 第 9~12 位是 2000 / 2100 / 2200 / 2300 的不看: 支付宝的格式按时间读出来正好是 20:00 / 21:00 / 22:00 / 23:00,
    ///     晚上 8 点到 11 点整点那十几秒付的真图会撞上。这样漏掉的只是正好在这几个整点那一分钟做的假图。
    ///   ★ 时间窗口从 0 秒开始: 商家订单号(商户自己的单号)常常是时间开头, 但都在支付之前(真图早 13~40 秒),
    ///     万一字段取错了传成商家订单号, 也不会判。请传"订单号"那一行, 不是"商家订单号"。
    ///
    /// 两条任何一条犯了就判可疑。两条在 ArrowGapCheck 抓到的假图上不重叠, 合起来 155 / 420(37%)。
    ///
    /// 入参都是字段的值(带不带标签都行, 比如 "2026-11-14 04:35:45" 或 "创建时间2026-11-14 04:35:45"):
    ///   payTime     支付时间那一行的值, 没有就传 null
    ///   createTime  创建时间那一行的值(转账页只有创建时间), 没有就传 null
    ///   orderNo     订单号那一行的值(折行的话只要第一行也够, 只用前 14 位), 没有就传 null
    ///   uploadTime  凭证上传时间, 北京时间。我们是按服务器图片文件名最后的 14 位算的(...._20261114043520.jpg),
    ///               没有现成的字段可以用 <see cref="UploadTimeFromFileName"/> 从文件名取。
    ///               毫秒会去掉; DateTime.Kind 是 Utc 的会先加 8 小时。没有就传 null(只看第二条)。
    /// 时间只认带秒的 年-月-日 时:分:秒(- / . 分隔, 冒号半角全角都行), 读不全不判。
    ///
    /// 和 training/field_time_scan.py 的 check() 逐步对应, 改一边要同步改另一边。
    /// 无静态可变状态, 可多线程调用。
    /// </summary>
    public static class FieldTimeCheck
    {
        /// <summary>第一条: 页面时间 - 上传时间 在这个范围(秒, 含两端)判可疑。</summary>
        public const int MinAheadSeconds = 1;
        public const int MaxAheadSeconds = 60;

        /// <summary>第二条: 订单号时间 - 页面时间 在这个范围(秒, 含两端)判可疑。</summary>
        public const int OrderNoMinSeconds = 0;
        public const int OrderNoMaxSeconds = 15;

        /// <summary>订单号第 9~12 位是这几个的, 是支付宝自己的格式, 第二条不看。</summary>
        static readonly string[] AlipayFormatCodes = { "2000", "2100", "2200", "2300" };

        // 年-月-日 时:分:秒; 只认半角数字; 日期和时间之间可以没有空格(OCR 常把空格丢掉)
        static readonly Regex TimeRe = new Regex(
            "(20[0-9]{2})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})[ \\t\\u00a0\\u3000]*([0-9]{1,2})[:\\uff1a]([0-9]{2})[:\\uff1a]([0-9]{2})",
            RegexOptions.CultureInvariant);

        static readonly Regex UploadRe = new Regex("_([0-9]{14})\\.", RegexOptions.CultureInvariant);

        public static FieldTimeResult Check(string? payTime, string? createTime, string? orderNo, DateTime? uploadTime)
        {
            var res = new FieldTimeResult { Verdict = FieldTimeVerdict.CannotDetermine };

            res.PageTime = ParsePageTime(payTime);
            if (res.PageTime != null) res.PageTimeField = "支付时间";
            else
            {
                res.PageTime = ParsePageTime(createTime);
                if (res.PageTime != null) res.PageTimeField = "创建时间";
            }

            if (uploadTime != null)
            {
                var u = uploadTime.Value;
                if (u.Kind == DateTimeKind.Utc) u = u.AddHours(8);
                res.UploadTime = new DateTime(u.Ticks - u.Ticks % TimeSpan.TicksPerSecond, DateTimeKind.Unspecified);
            }

            res.OrderNoDigits = DigitsOf(orderNo);
            res.OrderNoTime = OrderNoTimeOf(res.OrderNoDigits, out res.OrderNoIsAlipayFormat);

            if (res.PageTime == null)
            {
                res.Reason = "没读到支付时间 / 创建时间";
                return res;
            }
            var pt = res.PageTime.Value;

            if (res.UploadTime != null)
            {
                double d1 = (pt - res.UploadTime.Value).TotalSeconds;
                res.PageAfterUploadSeconds = d1;
                res.LaterThanUpload = d1 >= MinAheadSeconds && d1 <= MaxAheadSeconds;
            }
            if (res.OrderNoTime != null)
            {
                double d2 = (res.OrderNoTime.Value - pt).TotalSeconds;
                res.OrderNoAfterPageSeconds = d2;
                res.OrderNoIsPayTime = d2 >= OrderNoMinSeconds && d2 <= OrderNoMaxSeconds;
            }

            if (res.LaterThanUpload || res.OrderNoIsPayTime)
            {
                res.Verdict = FieldTimeVerdict.Suspicious;
                var sb = new StringBuilder();
                var inv = CultureInfo.InvariantCulture;
                if (res.LaterThanUpload)
                    sb.Append("页面" + res.PageTimeField + " " + pt.ToString("yyyy-MM-dd HH:mm:ss", inv)
                              + " 比上传时间 " + res.UploadTime!.Value.ToString("yyyy-MM-dd HH:mm:ss", inv)
                              + " 还晚 " + res.PageAfterUploadSeconds!.Value.ToString("0", inv) + " 秒");
                if (res.OrderNoIsPayTime)
                {
                    if (sb.Length > 0) sb.Append("; ");
                    sb.Append("订单号开头 " + res.OrderNoDigits.Substring(0, 14) + " 就是" + res.PageTimeField
                              + "(晚 " + res.OrderNoAfterPageSeconds!.Value.ToString("0", inv) + " 秒), 支付宝的订单号不是这个格式");
                }
                res.Reason = sb.ToString();
                return res;
            }

            if (res.UploadTime == null && res.OrderNoDigits.Length < 14)
            {
                res.Reason = "没有上传时间, 也没读到订单号";
                return res;
            }
            res.Verdict = FieldTimeVerdict.Ok;
            var ok = new StringBuilder();
            if (res.PageAfterUploadSeconds == null) ok.Append("没有上传时间");
            else if (res.PageAfterUploadSeconds.Value <= 0) ok.Append("上传比页面时间晚 " + Math.Abs(res.PageAfterUploadSeconds.Value).ToString("0", CultureInfo.InvariantCulture) + " 秒");
            else ok.Append("页面比上传晚 " + res.PageAfterUploadSeconds.Value.ToString("0", CultureInfo.InvariantCulture) + " 秒, 超过 " + MaxAheadSeconds + " 秒不判(多半认错字或时区不对)");
            ok.Append("; ");
            if (res.OrderNoDigits.Length < 14) ok.Append("没读到订单号");
            else if (res.OrderNoIsAlipayFormat) ok.Append("订单号是支付宝的格式");
            else if (res.OrderNoAfterPageSeconds == null) ok.Append("订单号开头不是时间");
            else ok.Append("订单号时间和" + res.PageTimeField + "差 " + res.OrderNoAfterPageSeconds.Value.ToString("0", CultureInfo.InvariantCulture) + " 秒, 不是 " + OrderNoMinSeconds + "~" + OrderNoMaxSeconds + " 秒");
            res.Reason = ok.ToString();
            return res;
        }

        /// <summary>从字段值里找第一个 年-月-日 时:分:秒; 找不到、或者第一个不是合法时间, 返回 null。</summary>
        public static DateTime? ParsePageTime(string? text)
        {
            if (string.IsNullOrEmpty(text)) return null;
            var m = TimeRe.Match(text);
            if (!m.Success) return null;
            return Make(Num(m.Groups[1].Value), Num(m.Groups[2].Value), Num(m.Groups[3].Value),
                        Num(m.Groups[4].Value), Num(m.Groups[5].Value), Num(m.Groups[6].Value));
        }

        /// <summary>服务器图片文件名最后的 _yyyyMMddHHmmss. 就是上传时间(带不带目录都行); 没有返回 null。</summary>
        public static DateTime? UploadTimeFromFileName(string? fileName)
        {
            if (string.IsNullOrEmpty(fileName)) return null;
            string name = fileName.Replace('\\', '/');
            int k = name.LastIndexOf('/');
            if (k >= 0) name = name.Substring(k + 1);
            var m = UploadRe.Match(name);
            if (!m.Success) return null;
            return FromDigits14(m.Groups[1].Value);
        }

        static string DigitsOf(string? s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            var sb = new StringBuilder(s.Length);
            foreach (char c in s) if (c >= '0' && c <= '9') sb.Append(c);
            return sb.ToString();
        }

        static DateTime? OrderNoTimeOf(string digits, out bool alipayFormat)
        {
            alipayFormat = false;
            if (digits.Length < 14) return null;
            string code = digits.Substring(8, 4);
            foreach (var c in AlipayFormatCodes)
                if (code == c) { alipayFormat = true; return null; }
            return FromDigits14(digits.Substring(0, 14));
        }

        static DateTime? FromDigits14(string d)
        {
            return Make(Num(d.Substring(0, 4)), Num(d.Substring(4, 2)), Num(d.Substring(6, 2)),
                        Num(d.Substring(8, 2)), Num(d.Substring(10, 2)), Num(d.Substring(12, 2)));
        }

        // 只会收到 1~4 位半角数字
        static int Num(string s)
        {
            int v = 0;
            foreach (char c in s) v = v * 10 + (c - '0');
            return v;
        }

        static DateTime? Make(int y, int mo, int d, int h, int mi, int s)
        {
            if (y < 2000 || y > 2099 || mo < 1 || mo > 12 || h > 23 || mi > 59 || s > 59) return null;
            if (d < 1 || d > DateTime.DaysInMonth(y, mo)) return null;
            return new DateTime(y, mo, d, h, mi, s, DateTimeKind.Unspecified);
        }
    }
}
