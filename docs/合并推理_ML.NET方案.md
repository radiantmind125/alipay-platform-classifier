# 蓝图推理合并 + ML.NET / ONNX 导出方案

> 目标(经理):最终**跑一个入口就能把一张蓝图的所有结果推理出来**(字段 + 设备 + 真伪),并且**导出成 ML.NET 能加载的 ONNX**。

---

## 一、一个要先说清楚的结论:是"一个入口",不是"一个 ONNX 文件搞定一切"

"合并推理"最实际的做法,就是你俩说的**多个模型串在一起 + 一个统一入口**。原因很直接:

- 一张完整的图,里面有三种不同性质的计算:
  1. **神经网络**(设备的状态栏 CNN、字段的 OCR 检测/识别)——这些能导 ONNX;
  2. **纯规则**(分辨率查表判设备、"没有勾"等真伪信号融合)——这些是 if/else,不是网络,塞不进一个计算图;
  3. **OCR 的前/后处理**(检测框解码、抠图、CTC 文本解码)——变长、有控制流,也塞不进单个静态图。
- 所以"一个 ONNX 文件端到端"在工程上不现实。**能做、且是标准做法的是:对外一个入口(一个 .NET 推理类 / 一个模型包),内部按顺序调用几个 ONNX 模型 + 少量规则。** 经理那边"只跑一个"这个诉求完全满足。

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

### 3.1 关键:预处理必须在 C# 端 1:1 复现(否则会掉点)

ONNX 里只有网络本身,**下面这几步喂进去之前要在 C# 做,顺序和参数必须完全一致**(与训练一致,否则 train/serve 偏差):

1. **EXIF 摆正**(按方向标签旋正,很多手机截图带方向)。
2. **裁顶部状态栏**:取图像**最上面 8%** 的整宽长条(`height * 0.08`,四舍五入,至少 1 像素)。
3. **缩放到 512×64**(宽512、高64),**插值用 BICUBIC(双三次)**——训练用的就是这个,别用双线性。
4. **归一化**(逐通道,RGB 顺序):
   - 先 `/255` 到 0~1;
   - 再 `(x - mean) / std`,`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`;
   - 排成 CHW,组成 `float[1,3,64,512]`。

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
using var img = ExifTranspose(Image.Load(path));
var strip = Crop(img, 0, 0, img.Width, (int)Math.Round(img.Height * 0.08)); // 顶部 8%
strip.Mutate(x => x.Resize(512, 64, KnownResamplers.Bicubic));              // 512x64 双三次
float[] input = ToChwImagenetNorm(strip);   // /255,减均值除方差,CHW,得到 [1,3,64,512]

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
- **方式 B:一个 ML.NET 管道(.zip)**,把几个 `ApplyOnnxModel` 串起来,规则用 `CustomMapping`。经理 `Load` 一个 .zip 即可。可行,但自定义规则(分辨率查表、CTC 解码、融合)在 .zip 里序列化/还原较麻烦,不如方式 A 干净。

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
