# 常态表重新标定 —— 服务器操作手册

## 为什么要做这件事

`FontCheck` 和 `ChevronCheck` 判的是**绝对像素尺寸**(金额数字多高、返回箭头多大)。
**支付宝改一次版, 这两张表就整体失效**, 而且失效的方式很难看:
要么整批真图被判成异常, 要么整批假图漏掉。

`MinusCheck` 不受影响 —— 它判的是**比值**(负号宽 / 数字宽), 字号一起变, 比值不变。
**所以只有前两个需要按月重标。**

> 目前仓库里 `FontCheck.cs` 那张表是**临时的**: 标定自一批 2026-07 的图, 12,000 张,
> 而且不是线上那批数据。**上线前必须按本文重新生成一次。**

---

## 第 0 步 环境

```powershell
cd E:\SSP_Work\alipay-platform-classifier
git pull
```

确认依赖(脚本只用 cv2 和 numpy, 不需要 torch):

```powershell
python -c "import cv2, numpy; print('cv2', cv2.__version__, '| numpy', numpy.__version__)"
```

确认图库目录。**先看这个变量指到的地方在不在**, 不在就用下面那句把候选目录列出来:

```powershell
$data = "D:\download2\OtherImages"
if (Test-Path $data) { "OK  $data" } else { "不在, 看看下面哪个是:"; Get-ChildItem D:\download2 -Directory | Select-Object FullName }
```

如果上面提示"不在", 把 `$data` 改成列出来的那个目录, 再往下走。

准备输出目录:

```powershell
if (-not (Test-Path D:\probe)) { New-Item -ItemType Directory D:\probe | Out-Null }
```

---

## 第 1 步 小样本试跑(5,000 张, 几分钟)

**先别跑全量。** 这一步是看定位有没有跑偏, 跑偏了全量就是白跑。

```powershell
python training\font_scan.py $data --limit 5000 --min-group 50 --out D:\probe\font_try.csv
```

> 试跑要带 `--min-group 50`。默认是 300, 而 5,000 张里能量到的大概只有 2,900 张,
> 摊到十几个分辨率上每组两三百张, **默认值下大部分组会被整组跳过, 你会看到一张空表**。
> 调小只是为了看清楚表的形状, **正式标定(第 2 步)不要带这个参数** —— 样本少定出来的常态不可信。

### 看三件事

1. **`判不了的占 xx%`** —— 应该在 **10%~15%**(本机 12.2%, 服务器 12.4%)。
   - 明显更高(比如 50%+): 目录不对, 或者这批图不是账单详情页。**停下来, 别往下跑。**
2. **分辨率表里的 `众数占比`** —— 服务器实测 **73%~98%**。
   - 明显更低(比如 30%): 该分辨率的"常态"根本立不住, 这条线在那个机型上没用。
3. **每个分辨率的 `张数`** —— 少于 `--min-group`(默认 300)的分辨率会被**整组跳过**。
   跳过 = **不判**, 不是判为正常。

> `Premature end of JPEG file` 是 OpenCV 对图库里截断 JPEG 的提示, **不是错误**, 忽略即可。
> **不要给 python 加 `2>&1`** —— PowerShell 5.1 会把原生程序 stderr 的每一行包成
> `NativeCommandError` 显示成红色报错, 看着像崩了其实没有。不加就正常打屏。

---

## 第 2 步 主跑 + 顺便生成 C# 表

一次跑完, 不用跑两遍。5 万张大概要**半小时到一个多小时**, 取决于磁盘。

```powershell
python training\font_scan.py $data --limit 50000 --out D:\probe\font.csv --emit-table | Tee-Object D:\probe\font_log.txt
```

> `Tee-Object` 会同时打屏和存文件。如果 `font_log.txt` 里中文是乱码,
> 改成 `| Out-File D:\probe\font_log.txt -Encoding utf8`(PowerShell 5.1 用 `>` 会写成 UTF-16, 别用 `>`)。

### ★ 最重要的一栏: 按月常态高

```
按月常态高(各月一样 = 字号没变过; 不一样 = app 改过版, 要分月各判各的)
        分辨率     202607     202608     202609
1179x2556          72(1078)   72(940)    72(311)
```

- **各月数字一样** -> 字号没变过, 一张表可以通用, 往下走。
- **某个月不一样** -> **支付宝改过版**。那就**必须分月各判各的**(第 3 步),
  而且不能拿一张混合出来的表去判任何一个月。

### 阈值对照表

输出的是一张**对照表**, 不是一个判定:

```
    字号倍数     比值倍数            报出       万分之
        1.03         1.08         2/8,680         2.30
```

**报出的全部是真图, 所以这一栏就是误报率。** 选哪个操作点是经理定的。

### 报出的例子

后面会列出报出的文件名。**挑几张打开看看** —— 这是关键的一步:

- 如果都是**正常页面**(转账详情页、系统大字体用户) -> 这条线在真实流量里没抓到东西。
- 如果有**金额明显被放大**的 -> **这是我们第一批"金额字体过大"的样本**,
  留下来, 后面标定和验证都要用。

> 我们**手上一张这种假图都没有**。经理说他见过, 但没给过图。
> 所以这一步不只是标定, 也是**第一次看这种手法在真实流量里到底有没有**。

---

## 第 2b 步 把报出来的图拼成一张, 人工看

**不用重新扫。** `--replay` 直接读第 2 步存下的 CSV, 秒级出结果:

```powershell
python training\font_scan.py $data --replay D:\probe\font.csv --sheet D:\probe\font_hits.png
```

出来的 `font_hits.png` 是把每张报出来的**金额行**裁下来上下叠在一起,
每条上面标着 `实测高/常态高=倍数  r=金额比正文  文件名`。**打开看一眼。**

`--replay` 还能用来换参数重算, 同样是秒级, 不用重扫:

```powershell
python training\font_scan.py $data --replay D:\probe\font.csv --size-mult 1.05 --ratio-mult 1.12
```

## 第 3 步 如果第 2 步发现改过版

分月各判各的。例如只用八月之后的数据:

```powershell
python training\font_scan.py $data --since 20260801 --out D:\probe\font_0801.csv --emit-table | Tee-Object D:\probe\font_0801_log.txt
```

`--since / --until` 都是 `YYYYMMDD` 八位, 取的是文件名里的时间戳。
**标定用的图和被判的图必须同期**, 这是这条线的硬要求。

---

## 第 4 步 把表贴回 C#

第 2 步(或第 3 步)的输出末尾会有一段:

```
// 贴进 FontCheck.cs 的 NormalDigitHeight。标定自 ..., 日期 ...~..., 共 N 张
        static readonly Dictionary<(int, int), int> NormalDigitHeight = new()
        {
            { (1179, 2556), 72 },   // n=1,078
            ...
        };
        public const double NormalAmountToBody = 2.41;
```

把这两段整个替换掉 `training/demo/FontCheck.cs` 里对应的部分,
**并把数据来源和日期范围写进提交说明**, 以后才查得到这张表是哪批数据标出来的。

---

## 第 5 步 返回箭头那条同样验一遍

`ChevronCheck.cs` 里的 `56x33` 和那张分辨率白名单也是硬编码的, 同样的毛病。

```powershell
python training\chevron_scan.py $data --limit 50000 --emit-table | Tee-Object D:\probe\chev_log.txt
```

★ 这条的 `--emit-table` 会**顺便验证 ChevronCheck.cs 赖以成立的那个前提**:
它假设**所有苹果机型共用同一个箭头尺寸**(iOS 导航图标按点数渲染)。

- 如果输出里只有 `// 常态 56x33 覆盖 N/N 个分辨率` -> 前提成立, 直接贴。
- 如果输出里出现 `★ 下列分辨率的常态和上面不一样` -> **前提不成立**,
  那些分辨率要么从白名单去掉, 要么 `ChevronCheck` 得改成按分辨率查表。

---

## 一次跑完的清单

```powershell
cd E:\SSP_Work\alipay-platform-classifier
git pull
$data = "D:\download2\OtherImages"
if (-not (Test-Path D:\probe)) { New-Item -ItemType Directory D:\probe | Out-Null }

python training\font_scan.py $data --limit 5000 --min-group 50 --out D:\probe\font_try.csv
python training\font_scan.py $data --limit 50000 --out D:\probe\font.csv --emit-table | Tee-Object D:\probe\font_log.txt
python training\font_scan.py $data --replay D:\probe\font.csv --sheet D:\probe\font_hits.png
python training\chevron_scan.py $data --limit 50000 --out D:\probe\chev.csv --emit-table | Tee-Object D:\probe\chev_log.txt
```

跑完把 `font.csv` / `font_log.txt` / `font_hits.png` / `chev.csv` / `chev_log.txt` 留着,
里面有全部结论。**其中 `font.csv` 最要紧** —— 有了它, 之后换阈值、看分月、出拼图
都能用 `--replay` 秒级重做, 不用再动图库。

---

## 2026-09-04 第一次在服务器上跑出来的结果(留个底, 下次比对用)

图库 **873,045 个文件**。抽 5,000 张, 量到 4,381 张, 判不了 12.4%。

- **常态表和本机七月标的那张一个不差**: 14 个分辨率全部一致, 0 个不一致。
  服务器上还多出三个: `1200x2652->73`, `1256x2760->76`, `720x1612->43`。
- **按月常态: 七月和八月完全一样**(72/66/78/71/79 五个分辨率两个月都没变) ——
  **支付宝没改过金额字号**, "会随版本漂移"这个担心在这个跨度上没有发生。
- **但这一批里没有九月的数据**, 而经理给的假图是 09-02, 九月这一档还没验过。
- `金额/正文` 常态中位 **2.433**(本机量的是 2.414)。
- 阈值 `1.03 / 1.08` 报出 **0/3,073**; 但 n 太小, 95% 上界约 9.8/万, **不能说成零误报**。
