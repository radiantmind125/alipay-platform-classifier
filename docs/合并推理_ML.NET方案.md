# 蓝图推理合并 + ML.NET / ONNX 导出方案

> 目标(经理):最终**跑一个入口就能把一张蓝图的所有结果推理出来**(字段 + 设备 + 真伪),并且**导出成 ML.NET 能加载的 ONNX**。

---

## 一、一个要先说清楚的结论:是"一个入口",不是"一个 ONNX 文件搞定一切"

"合并推理"最实际的做法,就是你俩说的**多个模型串在一起 + 一个统一入口**。原因很直接:

- 一张完整的图要跑:**神经网络**(设备状态栏 CNN、字段 OCR 检测/识别,这些能导 ONNX)+ **一堆非网络步骤**(分辨率查表、抠图、文本解析、真伪融合)。
- 严格讲,规则和控制流本身其实能塞进 ONNX(图里有 If/Loop/Scan 这些),但**真正卡死"单图端到端"的是三处没有对应 ONNX 算子的步骤**:
  1. OCR 检测的后处理——`findContours` 连通域/轮廓提取 + 文本框外扩(`unclip`/pyclipper),没有对应 ONNX 算子;
  2. 识别结果的"类别号 → 字符"字典映射;
  3. 文本 → 结构化字段(金额/收款方/时间…)的字符串/正则解析。
- 这三处导不进图,所以"一个 ONNX 文件端到端"不现实。**标准做法是:对外一个入口(一个 .NET 推理类 / 一个模型包),内部按顺序调用几个 ONNX 模型 + 这些非图步骤。** 经理那边"只跑一个"完全满足。

这也正是群里已经对齐的方向(mate:"多个模型串起来";经理:"你们看怎么合并方便")。

---

## 二、总体架构

```
                 一张原图(经理下载的蓝图)
                          │
        ┌─────────────────┴──────────────────┐
        │                                     │
   设备分支(我方,极省 CPU)              字段分支(mate,OCR)
   1. EXIF 摆正                          1. 检测模型 det.onnx → 框
   2. 先查分辨率(零解码)：               2. 后处理解码 + 抠出每个字段
      命中iPhone表→苹果                  3. 识别模型 rec.onnx → 文本(CTC解码)
      短边∈{720,1080,1440}→安卓          4. 得到 金额/收款方/时间/状态/付款方式
   3. 判不了的：裁顶部状态栏→
      statusbar.onnx → P(苹果)
        │                                     │
        └─────────────────┬──────────────────┘
                          │
                真伪融合(规则,我方)
              勾/分辨率一致性/翻拍/EXIF自相矛盾…
                          │
                  一个合并 JSON(见第五节)
```

要点:**设备分支不依赖 OCR**,单独就能出结果,而且几乎不吃 CPU;两条分支并行,最后用规则融合成一个结果。这样即使 OCR 那边慢,设备识别也不受拖累。

---

## 三、设备半边(我方)——ONNX 已就绪,ML.NET 原生可跑

已实测导出并在 onnxruntime 跑通(与 PyTorch 输出零误差):

| 项 | 值 |
|---|---|
| 文件 | `statusbar.onnx`(`export.py --mlnet` 生成) |
| 算子 | `Conv / Gemm / GlobalAveragePool / Relu / Softmax / Flatten` —— 全是最基础算子,ML.NET(OnnxTransformer)原生支持,无自定义算子 |
| opset | 13(ir_version 7,兼容性最好);17 也可 |
| 输入 | `strip`,`float32[batch,3,64,512]`(通道在前 CHW) |
| 输出 | `prob`,`float32[batch,2]`,已含 Softmax;**`prob[1]` = P(苹果)**,`prob[0]` = P(安卓) |

导出命令:
```powershell
python training\export.py --checkpoint training\runs\statusbar_v2\best.pt --out training\runs\statusbar_v2\statusbar.onnx --mlnet
```

### 3.0 ML.NET 加载两条硬约束(实测经验)

- **动态 batch 在 ML.NET 默认按 1 处理。** 单张推理(batch=1)直接加载就能跑;若要批量(batch>1),必须用带 `shapeDictionary` 的 `ApplyOnnxModel` 重载显式固定 batch 维,否则报形状不匹配。线上是单张流式,batch=1 即可。
- **锁死 onnxruntime 版本。** ML.NET 自带的 onnxruntime 往往比我们导出/验证用的旧;在 `.csproj` 里显式 `PackageReference` 固定 `Microsoft.ML.OnnxRuntime` 版本,保证行为一致(opset 13 从 ORT 1.6 起就支持,无兼容性风险)。

### 3.1 关键:预处理必须在 C# 端 1:1 复现(这里最容易掉点)

ONNX 里只有网络本身。下面这几步在喂进去之前要在 C# 做,**顺序和数值必须和训练完全一致**,否则 train/serve 偏差会直接掉点。逐条按坑写:

1. **EXIF 摆正,恰好一次,且在裁剪之前**:覆盖全部 8 种朝向。注意——**ImageSharp `Image.Load` 默认已自动按 EXIF 摆正**,就别再手动转一次(会造成双重旋转);**System.Drawing 不自动**,要手动读 `0x0112` 再转。截图大多没有 Orientation,但**翻拍图有**,这步对真伪判定也有用。
2. **裁顶部状态栏**:取最上面 8% 的整宽长条(`round(height * 0.08)`,至少 1 像素)。
3. **缩放到 (宽=512, 高=64)**(别把宽高传反)。**这一步是最大掉点风险**:
   - 这是大幅**下采样**(约 1080→512、~190→64)喂给一个 64px 的小网络,判别信号恰恰是状态栏那点**抗锯齿细节**;
   - 训练用的是 PIL 的 resize——**卷积式、下采样时按缩放因子放大卷积核做抗锯齿、三次核系数 a=−0.5**;
   - 而 System.Drawing/GDI+ 的 HighQualityBicubic、OpenCVSharp 的 INTER_CUBIC 是**固定 4×4 邻域、不抗锯齿、a=−0.75**,ML.NET 自带 ResizeImages 只近似双线性——**这几种和 PIL 差得远(实测像素能差 43–45/255),足以在"缩放过的刘海苹果 / 稀疏安卓栏"边界上把判定翻掉**;
   - 所以 C# 必须用**能匹配 PIL 的高质量缩放**:优先 **SkiaSharp 高质量缩放**,或自己写"缩放因子放大核 + a=−0.5"的抗锯齿卷积。
4. **转 uint8**:缩放结果**先 clip 到 [0,255] 再四舍五入成 uint8**(bicubic 会过冲,不 clip 会在高对比边缘分叉)。
5. **防 BGR 当 RGB**:System.Drawing / OpenCVSharp 内存里是 **BGR(A)** 序,取字节前要转成 **RGB**(否则减 mean 之后整体偏色、大幅掉点)。
6. **归一化**(逐通道,RGB 顺序):先 `/255` 到 0~1;再 `(x-mean)/std`,`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`;排成 CHW,组成 `float[1,3,64,512]`。

**验收(务必做)**:拿几十张真实截图 + 翻拍图,分别跑 Python 的 `preprocess_original` 和 C# 版,**逐像素 maxdiff ≤ 1**;更稳妥是直接比端到端 ONNX 输出,要求 **argmax 一致、P(苹果) 差 < 1e-2**。过了这关才算复现对。

### 3.2 分辨率规则(在 CNN 之前,C# 里几行 if 就行,省掉大部分 CNN 调用)

- 尺寸(宽,高)命中 **iPhone 分辨率表** → 直接判**苹果**(不用跑 CNN);
- 短边 ∈ {720,1080,1440} → 直接判**安卓**;
- 都不命中,才走上面的状态栏 CNN。

iPhone 分辨率表见 `src/alipay_platform/metadata_seed.py` 的 `IPHONE_RESOLUTIONS`(13 个竖屏机型),我把这张表整理成一个 C# 常量给 mate/经理。

### 3.3 C# 参考(设备分支,示意)

```csharp
// 依赖:Microsoft.ML.OnnxRuntime(ML.NET 底层同款推理引擎)
// 1) 先走分辨率规则
if (IsIphoneResolution(w, h))        return Device.Apple;      // 命中 iPhone 表
if (AndroidPanel.Contains(Math.Min(w, h))) return Device.Android; // 720/1080/1440

// 2) 判不了 → 状态栏 CNN
using var img = ExifTranspose(Image.Load(path));                            // EXIF 恰好一次、在裁剪前
var strip = Crop(img, 0, 0, img.Width, (int)Math.Round(img.Height * 0.08)); // 顶部 8%
var small = ResizePilLike(strip, 512, 64);   // ★ 必须匹配 PIL 抗锯齿(SkiaSharp 高质量/自写 a=-0.5),别用 GDI/ImageSharp 默认双三次
float[] input = ToChwImagenetNorm(small);    // clip→round→uint8→/255,减均值除方差(RGB序),CHW,得到 [1,3,64,512]

using var session = new InferenceSession("statusbar.onnx");
var tensor = new DenseTensor<float>(input, new[] { 1, 3, 64, 512 });
using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("strip", tensor) });
var prob = results.First().AsEnumerable<float>().ToArray(); // [P(安卓), P(苹果)]
var device = prob[1] >= 0.5 ? Device.Apple : Device.Android;
var confidence = Math.Max(prob[0], prob[1]);
```

> ML.NET 管道写法(`Transforms.ApplyOnnxModel`)也一样能跑,只是顶部 8% 裁剪不是 ML.NET 内置的图像变换,所以裁剪那一步仍需在 C# 手动做;网络调用交给 ML.NET/onnxruntime 都行。

---

## 四、字段半边(mate)——需要 mate 配合的点

mate 的 OCR 用的是 PaddleOCR。要并进来,需要:

1. 用 **`paddle2onnx`** 把**检测模型**和**识别模型**分别导出成 ONNX(PaddleOCR 官方支持);cls 方向分类若用到也一并导。
2. **检测后处理**(DB 解码出文字框)、**抠图**、**识别 CTC 解码**这些留在 C# 代码里(它们不是网络)。
3. 产出统一到第五节的字段(金额/收款方/时间/转账状态/付款方式)。

> 提醒:OCR 是目前 CPU 的大头。合并本身不降 CPU;设备这条已经做到几乎不占 CPU,不会给合并添负担。

---

## 五、统一输出(建议 schema,和 mate 对齐后定稿)

一张图出一个对象:

```json
{
  "file": "xxx.jpg",
  "device": { "type": "苹果", "confidence": 0.99, "source": "分辨率" },
  "fields": {
    "金额": "¥199.92",
    "收款方": "星(*星)",
    "时间": "00:37",
    "转账状态": "转账成功",
    "付款方式": "工商银行储蓄卡(3228)"
  },
  "fraud": { "verdict": "通过", "score": 0.08, "reasons": [] }
}
```

- `device.type` ∈ 苹果/安卓/不确定;`source` ∈ 分辨率/状态栏模型。
- `fraud.verdict` ∈ 通过/复核/拒绝(融合规则:勾、分辨率一致性、翻拍、EXIF 自相矛盾等,多个信号一起判)。

---

## 六、ML.NET 对外形态(两选一,都满足"跑一个")

- **方式 A(推荐):一个 .NET 推理类/服务** `InferBlueprint(image) -> 合并结果`。内部顺序:设备分支 → 字段分支 → 融合。经理只调这一个方法。最直接、最好维护。
- **方式 B:一个 ML.NET 管道(.zip)**,把几个 `ApplyOnnxModel` 串起来,规则用 `CustomMapping`。经理 `Load` 一个 .zip 即可。可行,但自定义规则要随 .zip 保存,得把逻辑写成实现 `CustomMappingFactory<,>`、带 `[CustomMappingFactory("契约名")]` 特性的**具名类**(用匿名 lambda 且 contractName 为空时 `Model.Save` 会直接抛),加载端还要保证这个程序集随应用一起部署。相比之下 `ApplyOnnxModel` 的存取不需要这些,所以**先做方式 A 更干净**,稳定后再评估打包成 B。

两种对经理都是"加载/调用一个东西"。建议先按方式 A 落地,后面要更"打包"再包成方式 B。

---

## 七、分工与对齐清单(发给 mate)

- **我方**:设备识别 `statusbar.onnx`(已就绪)+ 分辨率规则 C# 常量 + 预处理规范 + 真伪融合规则。
- **mate**:字段检测/识别 `det.onnx / rec.onnx`(paddle2onnx 导出)+ OCR 前后处理。
- **一起对齐**:
  1. 入口形态:方式 A(一个 C# 推理类)先落地,同意否?
  2. 输入:两条分支都吃**同一张原图**(不要各自再存中间图)。
  3. 输出:统一成第五节的 JSON 字段名。
  4. 由谁来写那个总入口(建议:总入口 + 融合我来搭,字段部分调用你的 ONNX + 后处理)。
