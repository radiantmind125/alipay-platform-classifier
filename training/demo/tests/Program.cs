using System;
using OpenCvSharp;
using Ssp;

// TextRegion 的两个新谓词 InRect / OnSegment 的验证。
// 断言不过就打印 FAIL 并最终返回非零退出码 —— 不许出现"打印一行就算测过"。

static class Program
{
    static int passed = 0;
    static int failed = 0;

    static void Check(string name, bool actual, bool expect)
    {
        if (actual == expect) { passed++; }
        else { failed++; Console.WriteLine($"  FAIL  {name}: 得到 {actual}, 应当 {expect}"); }
    }

    static void CheckInt(string name, int actual, int expect)
    {
        if (actual == expect) { passed++; }
        else { failed++; Console.WriteLine($"  FAIL  {name}: 得到 {actual}, 应当 {expect}"); }
    }

    /// 覆盖比例 = 交集面积 / 文字框面积, 和 InRect 里的算法一致
    static double Ratio(Rect text, Rect region)
    {
        var hit = Rect.Intersect(text, region);
        long area = (long)text.Width * text.Height;
        return area <= 0 ? 0 : (double)(hit.Width * hit.Height) / area;
    }

    static int Main(string[] args)
    {
        // ★ 自检: 带 --selftest 跑, 会故意插一条失败的断言。
        //   用来证明"失败时真的会返回非零" —— 不然这个壳子就是个只会打印的摆设。
        //   期望: dotnet run -- --selftest  之后  echo $LASTEXITCODE  是 1
        bool selftest = Array.IndexOf(args, "--selftest") >= 0;
        if (selftest)
        {
            Console.WriteLine("=== 自检: 下面这条一定要失败 ===");
            Check("自检 故意失败的断言", true, false);
        }

        Console.WriteLine("=== A. 先确认 OpenCvSharp 的语义和我假设的一致 ===");
        {
            var r = new Rect(10, 20, 30, 40);

            // A1 rect + Point 是不是平移
            var moved = r + new Point(0, 7);
            CheckInt("A1 加 Point 后 X 不变", moved.X, 10);
            CheckInt("A1 加 Point 后 Y 下移 7", moved.Y, 27);
            CheckInt("A1 加 Point 后 宽不变", moved.Width, 30);
            CheckInt("A1 加 Point 后 高不变", moved.Height, 40);

            // A2 不相交时 Intersect 返回什么
            var far = new Rect(1000, 1000, 5, 5);
            var none = Rect.Intersect(r, far);
            CheckInt("A2 不相交 -> 宽 0", none.Width, 0);
            CheckInt("A2 不相交 -> 高 0", none.Height, 0);

            // A3 Inflate 是两边各扩
            var inf = Rect.Inflate(r, 3, 5);
            CheckInt("A3 Inflate X 左移 3", inf.X, 7);
            CheckInt("A3 Inflate Y 上移 5", inf.Y, 15);
            CheckInt("A3 Inflate 宽 +2*3", inf.Width, 36);
            CheckInt("A3 Inflate 高 +2*5", inf.Height, 50);

            // A4 Contains 边界: 右下角算不算在内
            Check("A4 左上角在内", r.Contains(new Point(10, 20)), true);
            Check("A4 右下角不在内", r.Contains(new Point(40, 60)), false);
        }

        Console.WriteLine("=== B. InRect ===");
        {
            var region = new Rect(100, 200, 300, 100);   // x 100-400, y 200-300

            // B1 完全在内
            Check("B1 完全在内", TextRegion.InRect(new Rect(150, 220, 100, 40), region), true);

            // B2 完全在外
            Check("B2 完全在外", TextRegion.InRect(new Rect(150, 500, 100, 40), region), false);

            // B3 恰好压在 coverRatio 线上: 文字高 100, 落在里面 60 -> 0.60, 判据是 >=
            Check("B3 恰好 0.60 -> true",
                  TextRegion.InRect(new Rect(150, 240, 100, 100), region, 0, 0.60), true);

            // B4 落在里面 59 -> 0.59 < 0.60
            Check("B4 0.59 -> false",
                  TextRegion.InRect(new Rect(150, 241, 100, 100), region, 0, 0.60), false);

            // B5 空框
            Check("B5 零宽框", TextRegion.InRect(new Rect(150, 220, 0, 40), region), false);
            Check("B5 零高框", TextRegion.InRect(new Rect(150, 220, 100, 0), region), false);

            // B6 yOffset 动的是 region 不是 text
            //    文字在 y 420-460, region 原本 200-300, 下移 220 -> 420-520, 应当罩住
            var below = new Rect(150, 420, 100, 40);
            Check("B6 不偏移 -> false", TextRegion.InRect(below, region), false);
            Check("B6 region 下移 220 -> true", TextRegion.InRect(below, region, 220), true);

            // B7 负 yOffset = 往上
            var above = new Rect(150, 20, 100, 40);
            Check("B7 region 上移 200 -> true", TextRegion.InRect(above, region, -200), true);
        }

        Console.WriteLine("=== C. OnSegment ===");
        {
            var text = new Rect(100, 200, 200, 40);      // x 100-300, y 200-240

            // C1 水平线从框中间穿过
            Check("C1 水平线穿过",
                  TextRegion.OnSegment(text, new Point(0, 220), new Point(500, 220)), true);

            // C2 线在框上方, 无容差
            Check("C2 上方 10 像素 tol=0",
                  TextRegion.OnSegment(text, new Point(0, 190), new Point(500, 190)), false);

            // C3 上方 5 像素, tol=7 -> 够得着
            Check("C3 上方 5 像素 tol=7",
                  TextRegion.OnSegment(text, new Point(0, 195), new Point(500, 195), 0, 7), true);

            // C4 斜线穿过
            Check("C4 斜线穿过",
                  TextRegion.OnSegment(text, new Point(50, 180), new Point(350, 260)), true);

            // C5 竖线穿过
            Check("C5 竖线穿过",
                  TextRegion.OnSegment(text, new Point(200, 100), new Point(200, 400)), true);

            // C6 线完全在左边
            Check("C6 线在左边",
                  TextRegion.OnSegment(text, new Point(0, 100), new Point(0, 400)), false);

            // C7 退化成一个点, 落在框内
            Check("C7 点在框内",
                  TextRegion.OnSegment(text, new Point(200, 220), new Point(200, 220)), true);
            Check("C7 点在框外",
                  TextRegion.OnSegment(text, new Point(900, 900), new Point(900, 900)), false);

            // C8 yOffset 动的是线段
            Check("C8 线上移前不中",
                  TextRegion.OnSegment(text, new Point(0, 400), new Point(500, 400)), false);
            Check("C8 线上移 180 后中",
                  TextRegion.OnSegment(text, new Point(0, 400), new Point(500, 400), -180), true);

            // C9 共线: 线正好压在上边沿
            Check("C9 压在上边沿",
                  TextRegion.OnSegment(text, new Point(0, 200), new Point(500, 200)), true);

            // C10 空框
            Check("C10 空框", TextRegion.OnSegment(new Rect(100, 200, 0, 40),
                  new Point(0, 220), new Point(500, 220)), false);

            // C11 线段够不到(线段在框右侧终止)
            Check("C11 线段止于框左侧",
                  TextRegion.OnSegment(text, new Point(0, 220), new Point(50, 220)), false);
        }

        Console.WriteLine("=== D. 拼音并行那个真实场景(注释里的说法到底成不成立) ===");
        {
            // 类注释里的实测值: 拼音行 453-467, 正文行 468-496
            var textOnly = new Rect(0, 468, 500, 29);        // 正文行, 中心 482.5
            var merged   = new Rect(0, 453, 500, 44);        // 并进拼音后, 中心 475

            // 造两个相邻字段格子。分界线取 478 —— 正好卡在
            // "正文行中心 482" 和 "并入拼音后中心 475" 之间, 这样才验得出中心法的翻转。
            // ★ 中心只上移 7 像素, 所以只有分界线恰好落在这 7 像素里才会翻。
            //   这不是"一定会错", 是"可能会错"; 下面 D1 验的就是这种情形。
            var prevField   = new Rect(0, 440, 500, 38);     // y 440-478
            var targetField = new Rect(0, 478, 500, 32);     // y 478-510

            int cTextOnly = textOnly.Y + textOnly.Height / 2;
            int cMerged   = merged.Y + merged.Height / 2;
            Console.WriteLine($"  正文行中心 y={cTextOnly}, 并入拼音后中心 y={cMerged} (上移 {cTextOnly - cMerged})");

            // 按中心判: 并进来之后中心跑到上一个格子里 -> 会判成【错的字段】
            bool centreHitsPrev   = prevField.Contains(new Point(250, cMerged));
            bool centreHitsTarget = targetField.Contains(new Point(250, cMerged));
            Console.WriteLine($"  按中心: 落在上一格={centreHitsPrev}, 落在目标格={centreHitsTarget}");
            Check("D1 按中心会误判到上一个字段", centreHitsPrev, true);
            Check("D1 按中心认不出目标字段", centreHitsTarget, false);

            // 按面积占比: 目标格
            var hit = Rect.Intersect(merged, targetField);
            double ratio = (double)(hit.Width * hit.Height) / (merged.Width * merged.Height);
            Console.WriteLine($"  按面积: 并入后落在目标格的比例 = {ratio:F3}");

            bool areaTarget = TextRegion.InRect(merged, targetField);
            bool areaPrev   = TextRegion.InRect(merged, prevField);
            Console.WriteLine($"  按面积(0.6): 目标格={areaTarget}, 上一格={areaPrev}");

            // ★ 关键: 面积法在这个场景下【也认不出目标格】(比例只有 0.47),
            //   但它【不会误判到上一个格子】—— 这才是它真正的好处。
            Check("D2 面积法不会误判到上一个字段", areaPrev, false);
            Check("D2 面积法此时也认不出目标字段(诚实记录)", areaTarget, false);

            // 没并进拼音时, 两种都能正确认出目标格
            Check("D3 未并入时 面积法认得出", TextRegion.InRect(textOnly, targetField), true);
            Check("D3 未并入时 中心法也认得出",
                  targetField.Contains(new Point(250, cTextOnly)), true);

            // 实测比例 0.432 —— 想靠放低阈值把并入的行认出来, 得放到 0.43 以下,
            // 那对"区分相邻字段"来说已经太松了(半个框在别的格子里也算中)。
            Check("D4 阈值 0.40 能认出并入的行",
                  TextRegion.InRect(merged, targetField, 0, 0.40), true);
            Check("D4 但 0.50 就认不出了(所以放阈值不是办法)",
                  TextRegion.InRect(merged, targetField, 0, 0.50), false);

            // ★★ 正确的办法不是放阈值, 是先把拼音行滤掉再判。
            //    造一张真有拼音版面的图, 走 FindTextRows(dropAnnotationRows: true),
            //    看滤完之后正常阈值能不能认出目标字段。
            // ★ 这里必须画成**一个个字**而不是整条实心黑杠。
            //   取字用的是"和局部底色的差", 是不分正负的: 整条实心黑杠会把局部底色
            //   估计整个拉黑, 于是两行之间那条细白缝也被算成"和底色不同"=有字,
            //   两行就粘成一行了。真截图上字是稀疏笔画、底色是大片白, 不会这样
            //   (类注释里 011.jpg 实测就是分开的两行)。
            using var page = new Mat(600, 500, MatType.CV_8UC1, Scalar.All(255));
            for (int i = 0; i < 12; i++)
            {
                int x = 60 + i * 30;
                page.Rectangle(new Rect(x + 3, 453, 16, 11), Scalar.All(0), -1);   // 拼音, 矮
                page.Rectangle(new Rect(x,     468, 22, 29), Scalar.All(0), -1);   // 正文, 高
            }

            var rowsAll  = TextRegion.FindTextRows(page, dropAnnotationRows: false);
            var rowsKept = TextRegion.FindTextRows(page, dropAnnotationRows: true);
            Console.WriteLine($"  切行: 不滤 {rowsAll.Count} 行, 滤掉注音行后 {rowsKept.Count} 行");
            CheckInt("D5 不滤时切出 2 行", rowsAll.Count, 2);
            CheckInt("D5 滤完剩 1 行", rowsKept.Count, 1);

            if (rowsKept.Count == 1)
            {
                var kept = rowsKept[0];
                Console.WriteLine($"  滤完剩下的那行: y={kept.Y}..{kept.Y + kept.Height}");
                // 行框是整幅宽的, 判字段格子要先按 x 收窄(见 InRect 注释里的提醒)
                var narrowed = new Rect(60, kept.Y, 380, kept.Height);
                Check("D6 滤完之后 用默认 0.6 就能认出目标字段",
                      TextRegion.InRect(narrowed, targetField), true);
            }
            else { failed++; Console.WriteLine("  FAIL  D6 前置条件不满足, 跳过"); }

            // ★★ D7 OnSegment 对"并行"到底免不免疫 —— 注释里是这么写的, 验一下。
            //    线段法没有分母, 框往上长了, 原来那条基线仍然在框里穿过。
            var baseline = 482;    // 正文行的基线
            bool segTextOnly = TextRegion.OnSegment(textOnly,
                                   new Point(0, baseline), new Point(500, baseline));
            bool segMerged   = TextRegion.OnSegment(merged,
                                   new Point(0, baseline), new Point(500, baseline));
            Console.WriteLine($"  线段法: 未并入={segTextOnly}, 并入后={segMerged}");
            Check("D7 线段法 未并入时命中", segTextOnly, true);
            Check("D7 线段法 并入后仍然命中(这才是它比面积法强的地方)", segMerged, true);

            // 对照: 同一场景下面积法是掉线的
            Check("D7 对照 面积法并入后掉线", TextRegion.InRect(merged, targetField), false);
        }

        Console.WriteLine("=== E. 判和取的一致性: InRect 为真 <=> ByRect 能把它裁进来 ===");
        {
            using var img = new Mat(600, 500, MatType.CV_8UC1, Scalar.All(255));
            var region = new Rect(100, 200, 300, 100);

            // 完全在内的文字
            var inside = new Rect(150, 220, 100, 40);
            bool pred = TextRegion.InRect(inside, region, 0, 1.0);   // 要求 100% 在内
            using var got = TextRegion.ByRect(img, region, 0);
            bool cropCovers = got != null
                && Rect.Intersect(inside, new Rect(region.X, region.Y, got.Width, got.Height))
                       == inside;
            Console.WriteLine($"  InRect(100%)={pred}, ByRect 完整罩住={cropCovers}");
            Check("E1 判为真时取也能完整取到", pred && cropCovers, true);

            // 带 yOffset 时两边口径一致
            var below = new Rect(150, 420, 100, 40);
            bool pred2 = TextRegion.InRect(below, region, 220, 1.0);
            using var got2 = TextRegion.ByRect(img, region, 220);
            bool covers2 = got2 != null
                && Rect.Intersect(below, new Rect(region.X, region.Y + 220, got2.Width, got2.Height))
                       == below;
            Console.WriteLine($"  偏移 220: InRect={pred2}, ByRect 罩住={covers2}");
            Check("E2 偏移后判取仍一致", pred2 && covers2, true);
        }

        Console.WriteLine();
        Console.WriteLine("=== F. 带拼音时把格子顶撑开(经理说的那个数值) ===");
        {
            // 2026-09-16 在 800 张带拼音白底回单上量到:
            //   正文字高 中位 26, 框顶被拼音顶高 中位 12  ->  12/26 = 0.46
            Check("F1 倍数就是量出来的那个",
                  Math.Abs(TextRegion.AnnotationShiftRatio - 0.46) < 1e-9, true);
            CheckInt("F2 字高 26 撑 12 像素", TextRegion.AnnotationShift(26), 12);
            CheckInt("F3 字高 29 撑 13(011.jpg 量到 15, 在 10~16 区间里)",
                     TextRegion.AnnotationShift(29), 13);
            CheckInt("F4 字高 40 撑 18(高分辨率机型)", TextRegion.AnnotationShift(40), 18);

            // ★ 数不合理时宁可不动, 也不要按瞎算的数动
            CheckInt("F5 高度 0 不动", TextRegion.AnnotationShift(0), 0);
            CheckInt("F6 高度为负不动", TextRegion.AnnotationShift(-5), 0);
            CheckInt("F7 高度离谱不动", TextRegion.AnnotationShift(5000), 0);
            CheckInt("F8 NaN 不动", TextRegion.AnnotationShift(double.NaN), 0);

            // ★★ 撑开的是**顶**: 底不动, 高度变大
            var field = new Rect(10, 100, 200, 30);
            var wide = TextRegion.ExpandForAnnotation(field, 26);
            CheckInt("F9 顶上移了 12", wide.Y, 88);
            CheckInt("F10 底没动", wide.Y + wide.Height, field.Y + field.Height);
            CheckInt("F11 高度变大了 12", wide.Height, 42);
            CheckInt("F12 左右不动", wide.X, field.X);
            Check("F13 高度不合理时原样返回",
                  TextRegion.ExpandForAnnotation(field, 0) == field, true);

            // ★★★★★ 真实版面下的对比。
            //   拿真图量到的典型值: 汉字行 26 高, 拼音把框顶高 12 -> 合并框 38 高。
            //   字段格子和汉字行对齐(模板来自普通回单, 两边行位置基本一样 ——
            //   实测行距比值 0.98)。
            var chars = new Rect(0, 500, 500, 26);              // 汉字行
            var merged = new Rect(0, 488, 500, 38);             // 并入拼音后, 往上长了 12
            var box = TextRegion.ExpandForAnnotation(chars, 26);

            double covPlain = Ratio(merged, chars);
            double covWide = Ratio(merged, box);
            Console.WriteLine($"  不撑: 覆盖={covPlain:F3}, 判={TextRegion.InRect(merged, chars)}");
            Console.WriteLine($"  撑开: 覆盖={covWide:F3}, 判={TextRegion.InRect(merged, box)}");

            Check("F14 撑开之后覆盖比例明显变高", covWide > covPlain + 0.2, true);
            Check("F15 ★ 撑开之后判得中", TextRegion.InRect(merged, box), true);

            // ★★★ 全部 9068 张真图 105,514 行上量到的三种做法
            //     (记在这里免得以后有人再试"挪"):
            //       不动        通过率 88.5%   覆盖中位 0.656
            //       整个挪上去  通过率 86.1%   覆盖中位 0.647   <- 比不动还差
            //       顶往上撑开  通过率 99.6%   覆盖中位 0.958
            //
            // ★ 上面那个 merged 是"拼音正好顶高 12"的**边界情形**: 挪 12 之后格子顶
            //   刚好落在合并框顶上, 仍然整个在框内, 所以覆盖不变(0.684), 看不出差别。
            //   真图上顶高量是 10~16 而偏移是固定的 12, **有一半的行顶高不到 12**,
            //   这时候挪上去就有一截跑到框外 —— 下面用这种行验。
            int up = TextRegion.AnnotationShift(26);          // 12
            var mergedSmall = new Rect(0, 492, 500, 34);      // 这一行拼音只顶高 8
            double covSmallPlain = Ratio(mergedSmall, chars);
            var movedSmall = new Rect(chars.X, chars.Y - up, chars.Width, chars.Height);
            double covSmallMoved = Ratio(mergedSmall, movedSmall);
            var wideSmall = TextRegion.ExpandForAnnotation(chars, 26);
            double covSmallWide = Ratio(mergedSmall, wideSmall);
            Console.WriteLine($"  顶高只有 8 的行: 不动={covSmallPlain:F3}, "
                              + $"挪={covSmallMoved:F3}, 撑={covSmallWide:F3}");
            Check("F16 顶高小于偏移量时, 整个挪上去反而更差",
                  covSmallMoved < covSmallPlain, true);
            Check("F17 ★ 同一行撑开仍然最好", covSmallWide >= covSmallPlain, true);
        }

        Console.WriteLine();
        Console.WriteLine($"通过 {passed}, 失败 {failed}");
        return failed == 0 ? 0 : 1;
    }
}
