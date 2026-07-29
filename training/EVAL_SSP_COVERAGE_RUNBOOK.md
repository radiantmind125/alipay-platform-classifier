# SSP 能力边界测试 · AI 生成召回 + 篡改盲区 · 服务器一步一步

目的:把 SSP 唯一没验证的核心能力查清楚。
- **实验 A(重点)**:SSP 抓不抓 **AI 生成**假图?—— 而且要用**训练没用过的生成器**,才算数。
- **实验 B(搭车)**:用数据证明 SSP **看不到局部改金额**(经理 03/04 那种),坐实 Head B 的必要。

命令按 cmd 写。前置:`cd /d D:\alipay-platform-classifier` 然后 `git pull`。

---

## 实验 A · AI 生成召回(用 held-out 生成器)

**思路**:训练时的 AI 假图是用 `sd-vae-ft-mse` 这个 VAE 造的。现在换一个**没在训练里出现过的 VAE**(SDXL 的)去造假图,再看 SSP 抓不抓。抓得到 = 它学的是"通用生成指纹",不是死记一个生成器 —— 这点很关键,因为现在几乎所有 AI 图(SD / SDXL / Flux 等)都经过 VAE 解码器,"能识别 VAE 解码指纹"约等于"能识别大多数 AI 生成图"。

**源图用 held-out 的真图**(`D:\ssp_test\gen10k` 那一万张,训练没碰过),避免任何偏差。

### A1 · 造 300 张 held-out VAE 假图(GPU,约 10-20 分钟)
```
cd /d D:\alipay-platform-classifier
python training\gen_ai_fakes.py --genuine-roots D:\ssp_test\gen10k --out D:\ssp_test\aigen_sdxl --n 300 --methods vae --models madebyollin/sdxl-vae-fp16-fix --device cuda
```
**你应看到**:`造了 ~300 张 AI 假图 -> D:\ssp_test\aigen_sdxl`。

**必做质检**:打开 `D:\ssp_test\aigen_sdxl` 里随便 2-3 张看一眼 —— 应该是**能认出的支付宝截图**(可能略糊/略软),**不是纯黑或雪花噪声**。若是黑图/噪声,说明这个 VAE 在 fp16 下崩了(NaN),换下面的备用模型重跑 A1:
- 备用1:`--models stabilityai/sd-vae-ft-ema`(稳,但和训练同族,held-out 弱一点)
- 备用2:`--models stabilityai/sdxl-vae`(SDXL 官方,但 fp16 可能 NaN,出黑图就别用)
> 训练用的是 `sd-vae-ft-mse`,**绝对不要**用它做这个测试(那是循环论证)。

### A2 · 让 SSP 预测这 300 张(GPU 1-2 分 / CPU ~10 分)
```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input D:\ssp_test\aigen_sdxl --output_dir D:\ssp_test\aigen_sdxl_out --device cuda
```

### A3 · 算召回
```
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\ssp_test\aigen_sdxl_out\summary.csv --kind fake
```
**怎么读 `[假图召回] Reject 率`**:
- **> 90%**:SSP 能泛化到没见过的 VAE → 有力证据它是真的"AI 生成检测器",不是只会死记一个生成器。可以对经理说 SSP 覆盖 AI 生成 + 翻拍。
- **50–90%**:部分泛化 → 有点用但不牢,得说清楚是"部分覆盖"。
- **< 50%**:**不泛化** → 它其实主要靠翻拍/死记训练那个生成器 → "AI 检测"这块很弱,是关键短板,必须让经理知道。

---

## 实验 B · 篡改盲区(证明 SSP 看不到改金额)

**思路**:造一批**只在本地改了金额、没翻拍没重生成**的假图(干干净净存成图)。SSP 只看最平那块背景,看不到文字区的改动 → 预计**大部分会被判真**(召回很低)。这就把"SSP 抓不到 03/04"从"我说的"变成"数据证明的"。

### B1 · 造 300 张干净改字假图(约几分钟)
engine_b_tamper 的金额定位是在**白图(账单详情)**上调的,所以源用白池:
```
cd /d D:\alipay-platform-classifier
python training\engine_b_tamper.py --src-root D:\download2\TempFakeImages --out D:\ssp_test\tamper_clean --n 300
```
**你应看到**:`D:\ssp_test\tamper_clean` 里有一批改了金额的图(约 300 张;有些图定位不到金额会跳过,少一点正常)。

### B2 · 让 SSP 预测
```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input D:\ssp_test\tamper_clean --output_dir D:\ssp_test\tamper_clean_out --device cuda
```

### B3 · 算召回(预计很低)
```
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\ssp_test\tamper_clean_out\summary.csv --kind fake
```
**预期**:`Reject 率` 很低(接近 0)→ 即"这些改过金额的假图,SSP 基本全放过了" → 盲区坐实,Head B 必要性用数据说话。
> 若召回意外偏高,多半是存图时整图重压缩带进了别的痕迹,不是它看到了改动 —— 结论不变(它不是靠"看到改字"抓的)。

---

## 发我什么

1. 实验 A 的 A3 整段(尤其 `[假图召回] Reject 率`)+ 你用的是哪个 VAE 模型(主用还是备用)。
2. (可选)从 `D:\ssp_test\aigen_sdxl` 挑 1-2 张贴我,让我确认假图质量正常。
3. 实验 B 的 B3 整段。

我据此:给出 SSP 的**真实覆盖结论**(翻拍✓ / AI生成? / 改字✗),更新给经理的口径,并定后面的路线(去污retrain / 阈值标定 / 吞吐 / 融合)。

---

## (可选)实验 A 的加强版:img2img(更强的 held-out,机制不同)
VAE 往返是"再解码一次";img2img 是"用扩散模型轻改一遍",机制更不一样、更接近真实 AI 洗白。代价:模型大(~5GB)、慢。想做再跑:
```
python training\gen_ai_fakes.py --genuine-roots D:\ssp_test\gen10k --out D:\ssp_test\aigen_i2i --n 200 --methods img2img --models stable-diffusion-v1-5/stable-diffusion-v1-5 --strength 0.4 --cap 1024 --device cuda
# 注: stabilityai/stable-diffusion-2-1 已 gated/401; 用社区 stable-diffusion-v1-5/stable-diffusion-v1-5 或 Lykon/dreamshaper-8
```
(若该模型 id 下载失败,换一个当前可用的 SD1.x/2.x 模型;别用已下架的 runwayml/stable-diffusion-v1-5。)之后照 A2/A3 换路径预测+算召回。
