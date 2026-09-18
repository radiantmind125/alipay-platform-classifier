#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace Ssp
{
    public enum YenTemplateVerdict
    {
        CannotDetermine = 0,   // 判不了, 走正常流程
        Ok = 1,
        Suspicious = 2,        // ¥ 的形状和真图模板对不上
    }

    /// <summary>
    /// 机型, 由调用方的设备模型给。
    ///
    /// ★ 这里**另起了一个枚举**而不是用 YenCheck 里的 YenPlatform ——
    ///   house rule 是每个判据文件**单独放也能编译, 互不引用**(9/7 那两个就是这么发的)。
    ///   共用枚举的话这个文件就离不开 YenCheck.cs 了; 而重复定义同名枚举,
    ///   两个文件放进同一个工程又会撞名(CS0101)。所以取个不同的名字, 两种放法都成立。
    ///   调用方映射一下即可, 数值也是对齐的。
    /// </summary>
    public enum YenTemplatePlatform
    {
        Unknown = 0,
        Apple = 1,
        Android = 2,
    }

    public sealed class YenTemplateResult
    {
        public YenTemplateVerdict Verdict;
        /// <summary>和**调用方给的那个机型**的模板的相关系数, 判定用的就是它。</summary>
        public double Score;
        /// <summary>和苹果模板的相关系数, 只做输出。</summary>
        public double AppleScore;
        /// <summary>
        /// 和安卓模板的相关系数, 只做输出。
        /// ★★★ **不要拿这两个分谁高谁低去猜机型。** 实测伪造的 ¥ 连高度都是错的,
        ///   两张已知假图的 ¥ 高比落在 iPhone 真图的第 0.3 百分位, 按形状猜会被猜成安卓,
        ///   然后拿安卓的标准去判 —— **两张假图全放过**。机型只能来自状态栏。
        /// </summary>
        public double AndroidScore;
        public double DigitHeight;
        public YenTemplatePlatform Platform;
        public bool Measured;
        public string Reason = "";
    }

    /// <summary>
    /// **蓝底转账页**金额前面那个 ¥ 的**模板比对** —— 经理 2026-09-16 提的"图片定位"那条路。
    ///
    /// 和 YenCheck 的关系
    /// ------------------
    /// YenCheck 量的是**笔画粗细比**(一个标量), 这个类比的是**整个字形**(形状相关)。
    /// 两条路互相独立。**YenCheck 是主判据, 这个是第二意见**, 两个类互不引用, 各自单独能编译。
    ///
    /// ★★ 覆盖面比 YenCheck 窄, 要说清楚: 六张有标注的样本里, 这个类**只判得了两张**
    ///   (全分辨率的 2.png 和 true.png, 都判对了); 另外四张是 TG 压过的,
    ///   数字高只有 52~56, 低于本类 80 的下限, 一律不判。YenCheck 六张全判对。
    ///
    /// ★★★★★ 那道 80 的闸不是保守, 是有反例的: 真图 014.jpg 被 TG 压过之后
    ///   模板分掉到 **0.9260**, 低于苹果阈值 0.941 —— **没有这道闸就会误报一张真图**。
    ///   同一张图 YenCheck 给 0.5879(原图 0.5878), 照判不误。
    ///
    /// ★★★ 为什么主用 YenCheck 而不是这个: 实测**抗缩放差很多**。
    /// 两张原图各做 16 个缩放/重压缩变体:
    ///     缩放倍数        1.0      0.35     空档缩水
    ///     笔画比空档      0.149    0.125     -16%
    ///     模板分空档      0.212    0.072    **-66%**
    /// 降采样把区别糊掉之后, **假图的模板分反而往上爬**(0.788 -> 0.883)。
    /// 所以这个类的分辨率闸要卡得比 YenCheck 高。
    ///
    /// 模板怎么来的
    /// ------------
    /// `training/yen_template.py` 从语料里挑真图的 ¥, 按金额字高归一化,
    /// **按平台各求一张平均模板**, 量化成 uint8 塞在下面的 base64 里。
    /// 要重新标定就跑那个脚本加 --emit-cs, 把打出来的常量贴回来。
    ///
    /// ★★★★★ 尺寸和阈值**只用真图定, 一张假图都没参与**
    /// -----------------------------------------------
    /// 第一版是按"真假空档最大"挑画布尺寸的。七种尺寸的空档是
    /// -0.002 / 0.028 / 0.034 / 0.061 / 0.061 / 0.066 / 0.107 —— **差 50 倍还穿过零**。
    /// 那等于**拿两张假图在挑超参**, 换两张新假图结论就变。
    /// 改成用"苹果真图和安卓真图分不分得开"挑尺寸, 阈值按**本平台真图的 p0.1** 定,
    /// 假图只在最后验一眼。这和 MinusCheck 按真图分布定阈值是同一个路子。
    ///
    /// 覆盖边界
    /// --------
    /// - 只管蓝底转账页; 白底账单详情页直接跳过。
    /// - **必须给机型**。不给就只算分不下结论 —— 理由同 YenCheck。
    /// - 安卓那条**没有假样本**, 阈值是拿真图定的, 只能说"不像真安卓"。
    /// - 支付宝换了金额字体, 模板就得重做。这是它和 YenCheck 最大的不同:
    ///   YenCheck 只有两个常数, 这个带着一份数据。
    ///
    /// 入参只读, 无静态可变状态(模板是 static readonly, 初始化一次), 可多线程调用。
    /// </summary>
    public static class YenTemplateCheck
    {
        // ====== 以下常量由 training/yen_template.py --emit-cs 生成 ======
        public const int TemplateH = 33, TemplateW = 27;
        public const double NormDigitHeight = 30;
        public const double AppleMinScore = 0.941;
        public const double AndroidMinScore = 0.749;
        
        // apple 平均模板  33x27  归一字高 30  891 字节  标定自 80 张真图
        const string AppleTemplateB64 =
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK71MYsAQEBAQEB" +
            "AQEBLcjUuwAAAAAAAAAAAAB5/P2eBgEBAQEBAQEGoP78eAAAAAAAAAAAAAAVzP7wPQEBAQEBAQE+8f7LEwAAAAAAAAAAAAACU/f+" +
            "uAkBAQEBAQm6/vdRAgAAAAAAAAAAAAABCa7++FACAQEBAlL4/q0IAQAAAAAAAAAAAAABATTt/sgSAQEBE8r+7TIBAQAAAAAAAAAA" +
            "AAABAQSI/ftqAgECa/v9hwQBAQAAAAAAAAAAAAABAQEe2v7bGwEb2/7ZHQEBAQAAAAAAAAAAAAABAQECYvv+gQWD/vthAgEBAQAA" +
            "AAAAAAAAAAABAQEBDcD+51Tn/r8MAQEBAQAAAAAAAAAAAAABAQICAkTy/u3+8kMCAgEBAQAAAAAAAAAAAAABWr+/v8P0/v7+8sO/" +
            "v79gAQAAAAAAAAAAAAABW8DAwMDB9v72wcDAwL9fAQAAAAAAAAAAAAABBw0NDQ0S3f7dEg0NDQ0GAQAAAAAAAAAAAAABAQEBAQEH" +
            "3P7bBgEBAQEBAQAAAAAAAAAAAAABHDs7Ozs/5P7jPjs7OzseAQAAAAAAAAAAAAABcfDw8PDx/f788PDw8PB4AQAAAAAAAAAAAAAB" +
            "LV1dXV1g6P7oYF1dXVwuAQAAAAAAAAAAAAABAgMDAwMI3P7bCAMDAwIBAQAAAAAAAAAAAAABAQEBAQEH3P7bBgEBAQEBAQAAAAAA" +
            "AAAAAAABAQEBAQEH3P7bBgEBAQEBAQAAAAAAAAAAAAABAQEBAQEH3P7aBgEBAQEBAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        // android 平均模板  33x27  归一字高 30  891 字节  标定自 80 张真图
        const string AndroidTemplateB64 =
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJDAwKAQAAAAAA" +
            "AAAECwwLBgAAAAAAAAAAAACm8PHUKwEBAQEBAQMkyPHsrgAAAAAAAAAAAABG6v75jAYBAQEBAQp3+f3pUwAAAAAAAAAAAAALmfz+" +
            "4DMBAQEBAyjW/venDgAAAAAAAAAAAAACMN/++5gIAQECC3/6/N8+AgAAAAAAAAAAAAABB3/7/uY8AQEDLtn+9ZIHAQAAAAAAAAAA" +
            "AAABAiHR/vyjCgILhvv81SoBAQAAAAAAAAAAAAABAQRk9/7sRQMy3/7ydwQBAQAAAAAAAAAAAAABAQEWvP78sBeO/PvFHAEBAQAA" +
            "AAAAAAAAAAABAQEDTfL+8ITk/u5dAwEBAQAAAAAAAAAAAAABBQsMF6z+/vf9/LocCwoDAQAAAAAAAAAAAAACM7vHyN/+/v7+/eXH" +
            "x71AAQAAAAAAAAAAAAACPOXz8/T4/v799vPz8+hRAQAAAAAAAAAAAAABEkZLS0t89f72gEtLS0gbAQAAAAAAAAAAAAACCx8gICBY" +
            "8v73ZSAgIB0IAQAAAAAAAAAAAAACNcTQ0NDb/P7939DQz8U9AQAAAAAAAAAAAAACNMzZ2dnm/f774tnZ2dBEAQAAAAAAAAAAAAAB" +
            "CycqKipi8/72aioqKigOAQAAAAAAAAAAAAABAQEBAQJE8P70TgEBAQEBAQAAAAAAAAAAAAABAQEBAQJE8P70TgEBAQEBAQAAAAAA" +
            "AAAAAAABAQEBAQJD7vzzTQEBAQEBAQAAAAAAAAAAAAABAQEBAQEppK+rOgEBAQEBAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        // ===============================================================

        /// <summary>
        /// 数字中位高低于此值不判定。★ 比 YenCheck 的 40 高 —— 模板法抗缩放差,
        /// 实测缩到数字高 36 像素时空档只剩 0.072(全尺寸时是 0.212)。
        /// </summary>
        public const double MinDigitHeight = 80;

        /// <summary>¥ 高 / 数字高 的护栏, 和 YenCheck 同一套。</summary>
        public const double YenHeightLow = 0.60, YenHeightHigh = 0.80;

        const double BarFill = 0.70;
        const int MaxComponents = 20_000;

        static readonly float[,] AppleTpl = Decode(AppleTemplateB64);
        static readonly float[,] AndroidTpl = Decode(AndroidTemplateB64);

        struct Comp { public int X, Y, W, H, Area, Label; }

        static float[,] Decode(string b64)
        {
            var raw = Convert.FromBase64String(b64);
            if (raw.Length != TemplateH * TemplateW)
                throw new InvalidOperationException(
                    $"模板字节数 {raw.Length} 和 {TemplateH}x{TemplateW} 对不上");
            var a = new float[TemplateH, TemplateW];
            int k = 0;
            for (int y = 0; y < TemplateH; y++)
                for (int x = 0; x < TemplateW; x++) a[y, x] = raw[k++] / 255f;
            return a;
        }

        /// <param name="image">8 位或 16 位 **BGR(A)** 图。不支持的返回 CannotDetermine, 不抛异常。</param>
        /// <param name="platform">机型, 由调用方的设备模型给。不给就只算分不下结论。</param>
        public static YenTemplateResult Check(Mat image, YenTemplatePlatform platform = YenTemplatePlatform.Unknown)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var res = new YenTemplateResult
            { Verdict = YenTemplateVerdict.CannotDetermine, Platform = platform };
            if (image.Empty()) { res.Reason = "图为空"; return res; }

            int cn = image.Channels();
            if (cn != 3 && cn != 4)
            { res.Reason = $"{cn} 通道没有颜色, 判不了是不是蓝底页"; return res; }
            bool is8 = image.Depth() == MatType.CV_8U;
            if (!is8 && image.Depth() != MatType.CV_16U)
            { res.Reason = "位深不是 8 位或 16 位, 不判"; return res; }
            if (image.Width < 64 || image.Height < 128) { res.Reason = "图太小"; return res; }

            Mat? owned8 = null, ownedBgr = null, ownedGray = null;
            try
            {
                Mat src = image;
                if (!is8)
                {
                    owned8 = new Mat();
                    image.ConvertTo(owned8, MatType.CV_8U, 1.0 / 256.0);
                    src = owned8;
                }
                if (cn == 4)
                {
                    ownedBgr = new Mat();
                    Cv2.CvtColor(src, ownedBgr, ColorConversionCodes.BGRA2BGR);
                    src = ownedBgr;
                }
                if (!IsBluePage(src))
                { res.Reason = "不是蓝底转账页, 本类只管蓝底页"; return res; }

                var box = LocateAmountBlue(src);
                if (box == null) { res.Reason = "定位不到金额行"; return res; }

                int pad = Math.Max(2, (int)(box.Value.Height * 0.15));
                int padx = Math.Max(pad, (int)(box.Value.Width * 0.30));
                int cx0 = Math.Max(0, box.Value.Left - padx), cy0 = Math.Max(0, box.Value.Top - pad);
                int cx1 = Math.Min(src.Width, box.Value.Right + pad);
                int cy1 = Math.Min(src.Height, box.Value.Bottom + pad);
                if (cx1 - cx0 < 8 || cy1 - cy0 < 8)
                { res.Reason = "金额行裁出来太小"; return res; }

                using var bgrCrop = new Mat(src, Rect.FromLTRB(cx0, cy0, cx1, cy1));
                ownedGray = new Mat();
                Cv2.CvtColor(bgrCrop, ownedGray, ColorConversionCodes.BGR2GRAY);
                return Measure(ownedGray, res);
            }
            finally { ownedGray?.Dispose(); ownedBgr?.Dispose(); owned8?.Dispose(); }
        }

        static YenTemplateResult Measure(Mat sub, YenTemplateResult res)
        {
            int cw = sub.Width, ch = sub.Height;
            using var bin = new Mat();
            Cv2.Threshold(sub, bin, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
            if (255L * Cv2.CountNonZero(bin) > 127L * cw * ch) Cv2.BitwiseNot(bin, bin);

            using var labels = new Mat();
            using var stats = new Mat();
            using var cent = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(bin, labels, stats, cent,
                        PixelConnectivity.Connectivity8, MatType.CV_32S);
            if (n - 1 > MaxComponents) { res.Reason = "块太多"; return res; }
            var glyphs = new List<Comp>();
            for (int i = 1; i < n; i++)
            {
                int a = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                if (a < 6) continue;
                glyphs.Add(new Comp
                {
                    X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                    Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                    W = stats.At<int>(i, (int)ConnectedComponentsTypes.Width),
                    H = stats.At<int>(i, (int)ConnectedComponentsTypes.Height),
                    Area = a,
                    Label = i,
                });
            }
            if (glyphs.Count < 4) { res.Reason = "块数不够"; return res; }
            glyphs = glyphs.OrderBy(g => g.X).ThenBy(g => g.Y).ThenBy(g => g.Label).ToList();

            var yen = glyphs[0];
            var rest = glyphs.Skip(1).ToList();
            int tall = rest.Max(g => g.H);
            var digits = rest.Where(g => g.H >= 0.80 * tall && g.X > 0 && g.X + g.W < cw).ToList();
            if (digits.Count < 3) { res.Reason = "数字不够三个"; return res; }
            double dH = digits.Max(g => g.Y + g.H) - digits.Min(g => g.Y);
            if (dH <= 0) { res.Reason = "数字高算不出来"; return res; }

            double yenHRatio = yen.H / dH;
            if (yenHRatio < YenHeightLow || yenHRatio > YenHeightHigh)
            { res.Reason = $"最左块高比 {yenHRatio:F3} 不像 ¥"; return res; }

            var lab = ToInts(labels);
            var barRows = new bool[yen.H];
            for (int r = 0; r < yen.H; r++)
            {
                int on = 0;
                for (int c = 0; c < yen.W; c++) if (lab[yen.Y + r, yen.X + c] == yen.Label) on++;
                barRows[r] = on >= BarFill * yen.W;
            }
            if (Spans(barRows).Count != 2)
            { res.Reason = "最左块不是两道横杠, 不像 ¥"; return res; }

            res.DigitHeight = dH;
            if (dH < MinDigitHeight)
            {
                res.Reason = $"数字高 {dH:F0} < {MinDigitHeight}, 模板法在小图上不准, 不判";
                return res;
            }

            // ---- 把 ¥ 渲染成和模板同样口径的贴片 ----
            using var patch = RenderPatch(sub, yen, dH);
            if (patch == null) { res.Reason = "¥ 放不进模板画布"; return res; }
            var p = ToFloats(patch);
            res.Measured = true;
            res.AppleScore = Ncc(p, AppleTpl);
            res.AndroidScore = Ncc(p, AndroidTpl);

            double thr;
            switch (res.Platform)
            {
                case YenTemplatePlatform.Apple: res.Score = res.AppleScore; thr = AppleMinScore; break;
                case YenTemplatePlatform.Android: res.Score = res.AndroidScore; thr = AndroidMinScore; break;
                default:
                    res.Reason = $"苹果模板 {res.AppleScore:F4} 安卓模板 {res.AndroidScore:F4}; "
                               + "不知道机型, 只算分不下结论";
                    return res;
            }
            if (res.Score < thr)
            {
                res.Verdict = YenTemplateVerdict.Suspicious;
                res.Reason = $"¥ 和该机型真图模板只有 {res.Score:F4} (真图至少 {thr})";
            }
            else
            {
                res.Verdict = YenTemplateVerdict.Ok;
                res.Reason = $"¥ 模板分 {res.Score:F4}";
            }
            return res;
        }

        /// <summary>
        /// 口径必须和 Python 侧 `yen_template.render` 一模一样:
        /// 先 min-max 归一到 0~1, 再按金额字高缩放, 最后居中贴到全 0 画布上。
        /// </summary>
        static Mat? RenderPatch(Mat gray, Comp yen, double dH)
        {
            using var roi = new Mat(gray, new Rect(yen.X, yen.Y, yen.W, yen.H));
            using var f = new Mat();
            roi.ConvertTo(f, MatType.CV_32F);
            Cv2.MinMaxLoc(f, out double mn, out double mx);
            if (mx - mn < 1e-6) return null;
            using var norm = new Mat();
            Cv2.Subtract(f, new Scalar(mn), norm);
            norm.ConvertTo(norm, MatType.CV_32F, 1.0 / (mx - mn));

            double sc = NormDigitHeight / dH;
            int nh = Math.Max(3, (int)Math.Round(yen.H * sc));
            int nw = Math.Max(3, (int)Math.Round(yen.W * sc));
            if (nh > TemplateH || nw > TemplateW) return null;
            using var small = new Mat();
            Cv2.Resize(norm, small, new Size(nw, nh), 0, 0, InterpolationFlags.Area);

            var canvas = new Mat(TemplateH, TemplateW, MatType.CV_32F, Scalar.All(0));
            int y0 = (TemplateH - nh) / 2, x0 = (TemplateW - nw) / 2;
            using (var dst = new Mat(canvas, new Rect(x0, y0, nw, nh))) small.CopyTo(dst);
            return canvas;
        }

        /// <summary>两张同尺寸图的相关系数, 等价于 OpenCV 的 TM_CCOEFF_NORMED。</summary>
        static double Ncc(float[,] a, float[,] b)
        {
            int h = a.GetLength(0), w = a.GetLength(1);
            double sa = 0, sb = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) { sa += a[y, x]; sb += b[y, x]; }
            double ma = sa / (h * w), mb = sb / (h * w);
            double num = 0, da = 0, db = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double u = a[y, x] - ma, v = b[y, x] - mb;
                    num += u * v; da += u * u; db += v * v;
                }
            double den = Math.Sqrt(da * db);
            return den < 1e-12 ? 0.0 : num / den;
        }

        static bool IsBluePage(Mat bgr)
        {
            int h3 = Math.Max(1, bgr.Height / 3);
            using var top = new Mat(bgr, new Rect(0, 0, bgr.Width, h3));
            var m = Cv2.Mean(top);
            return m.Val0 > m.Val2 + 25 && m.Val0 > m.Val1 + 15;
        }

        static Rect? LocateAmountBlue(Mat bgr)
        {
            int W = bgr.Width, H = bgr.Height;
            int y0b = (int)(H * 0.05), y1b = (int)(H * 0.45);
            if (y1b - y0b < 8) return null;
            using var band = new Mat(bgr, new Rect(0, y0b, W, y1b - y0b));
            var ch = Cv2.Split(band);
            try
            {
                using var t = new Mat(); using var mn = new Mat(); using var mx = new Mat();
                Cv2.Min(ch[0], ch[1], t); Cv2.Min(t, ch[2], mn);
                Cv2.Max(ch[0], ch[1], t); Cv2.Max(t, ch[2], mx);
                using var diff = new Mat(); Cv2.Subtract(mx, mn, diff);
                using var bright = new Mat(); Cv2.Threshold(mn, bright, 175, 255, ThresholdTypes.Binary);
                using var flat = new Mat(); Cv2.Threshold(diff, flat, 44, 255, ThresholdTypes.BinaryInv);
                using var white = new Mat(); Cv2.BitwiseAnd(bright, flat, white);

                var all = Label(white, minArea: 30);
                if (all.Count > MaxComponents) return null;
                var comps = new List<Comp>();
                foreach (var c in all)
                {
                    if (c.H < 0.025 * H || c.H > 0.12 * H || c.W > 0.5 * W) continue;
                    comps.Add(new Comp { X = c.X, Y = c.Y + y0b, W = c.W, H = c.H, Area = c.Area });
                }
                if (comps.Count == 0) return null;
                int maxh = comps.Max(c => c.H);
                var amt = comps.Where(c => c.H >= 0.6 * maxh).ToList();
                if (amt.Count < 2) return null;
                double cy = Median(amt.Select(c => c.Y + c.H / 2.0));
                amt = amt.Where(c => Math.Abs((c.Y + c.H / 2.0) - cy) < 0.6 * maxh).ToList();
                if (amt.Count < 2) return null;
                int x0 = amt.Min(c => c.X), x1 = amt.Max(c => c.X + c.W);
                int y0 = amt.Min(c => c.Y), y1 = amt.Max(c => c.Y + c.H);
                if (x1 - x0 < 0.08 * W) return null;
                return Rect.FromLTRB(x0, y0, x1, y1);
            }
            finally { foreach (var c in ch) c.Dispose(); }
        }

        readonly struct Span
        {
            public readonly int Start, End;
            public Span(int s, int e) { Start = s; End = e; }
        }

        static List<Span> Spans(bool[] row)
        {
            var outp = new List<Span>();
            int s = -1;
            for (int i = 0; i <= row.Length; i++)
            {
                bool on = i < row.Length && row[i];
                if (on && s < 0) s = i;
                else if (!on && s >= 0) { outp.Add(new Span(s, i)); s = -1; }
            }
            return outp;
        }

        static List<Comp> Label(Mat bin, int minArea)
        {
            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(bin, labels, stats, centroids,
                        PixelConnectivity.Connectivity8, MatType.CV_32S);
            var outp = new List<Comp>(Math.Max(0, n - 1));
            for (int i = 1; i < n; i++)
            {
                int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                if (area < minArea) continue;
                outp.Add(new Comp
                {
                    X = stats.At<int>(i, (int)ConnectedComponentsTypes.Left),
                    Y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top),
                    W = stats.At<int>(i, (int)ConnectedComponentsTypes.Width),
                    H = stats.At<int>(i, (int)ConnectedComponentsTypes.Height),
                    Area = area,
                    Label = i,
                });
            }
            return outp;
        }

        static float[,] ToFloats(Mat m)
        {
            m.GetRectangularArray(out float[,] a);
            return a;
        }

        static int[,] ToInts(Mat m)
        {
            m.GetRectangularArray(out int[,] a);
            return a;
        }

        static double Median(IEnumerable<double> xs)
        {
            var v = xs.OrderBy(k => k).ToList();
            if (v.Count == 0) return 0;
            return v.Count % 2 == 1 ? v[v.Count / 2]
                                    : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2.0;
        }
    }
}
