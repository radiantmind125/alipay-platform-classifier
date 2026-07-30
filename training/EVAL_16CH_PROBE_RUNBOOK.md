# 跨解码器架构泛化探针 · 16通道 / TAESD · 服务器一步一步

**（豆包/千问的开源代理测试 —— 不需要中国 API / 密钥 / VPN）**

---

## 这是在测什么(先读这段)

v3 训练时只见过 **SD 家族的 4 通道 VAE** 解码器指纹。这个探针换上**架构差异很大的解码器**造假图, 看 v3 还抓不抓:

- **ostris/vae-kl-f8-d16** = **16 通道** KL VAE(Flux / SD3 那一类的潜空间, 和训练的 4 通道完全不同)。
- **TAESD / TAESDXL** = **蒸馏微型解码器**(结构和标准 VAE 差最远)。

豆包和千问也都是潜扩散、都靠 VAE 解码器出图。所以:

- 这些架构差很大的解码器 v3 **都能抓** → **强证据 v3 也能抓豆包/千问**, 可以先给经理一个真结论, 等你拿到 App/API 再正式确认。
- **抓不到** → 说明指纹是"认架构"的, 覆盖豆包/千问需要在训练里加多样解码器(这也直接指导 v4)。

**核心思路和 truthscan 一样**: 训练见的生成器越多样, 越能抓没见过的; 这里用架构差异证明泛化的边界在哪。

---

## 前置

1. `cd /d D:\alipay-platform-classifier`
2. `git pull`  (拉到刚加的 `--vae-class` / `--vae-subfolder` 支持)
3. 确认 v3 最优模型路径: `D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v3\Net_epoch_best.pth`
   - **若你的最优模型在别的目录(如 `snapshot\aigen`), 把下面所有命令里的这个路径换成你的。**
4. GPU + 之前装过的 torch / diffusers(跑过 gen_ai_fakes 就有)。

---

## 步骤 0 · 10 秒地基核实(必做, 护住所有已报数字)

```
findstr /n "patch_list" D:\SSP\utils\patch.py D:\SSP-AI-Generated-Image-Detection-main\utils\patch.py
```

两个文件都必须是 `new_img = patch_list[-1]`。

- 若有一个是 `[0]` → **停, 告诉我**。(我们报给经理的召回/误杀都是按 `[-1]` 跑的; 若预测端是 `[0]`, 数字得重算。)

---

## 步骤 1 · 造探针假图

三个生成器**分开跑**(各自独立, 某个下不了不影响其他; 也方便分别读召回)。共用参数:

- `--source-split holdout`: 只用哈希切出的 held-out 源图(尽量避开训练造假图用过的源, 防"记住版面"虚高召回)。
- `--cap 0`: 原生分辨率, 别缩, 保住指纹。
- `--dtype bf16`: 16 通道 / 微型 VAE 在 fp16 容易 NaN 出黑图, bf16 稳。

### 1a · ostris 16 通道 KL VAE(主测, 必跑)

```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\probe\ostris16 --n 300 --methods vae --models ostris/vae-kl-f8-d16 --vae-class kl --cap 0 --dtype bf16 --source-split holdout --seed 777 --device cuda
```

应看到: `造了 ~300 张 AI 假图 -> D:\probe\ostris16`。

**质检**: 打开 `D:\probe\ostris16` 里 2-3 张 —— 应是**能认出的支付宝截图**(略糊/略软), 不是纯黑/雪花噪声。

- 若是黑图: bf16 没生效或模型异常 → 告诉我。
- 若报下载失败 / 401: ostris 仓库拉不到 → 告诉我, 我给替代的 16 通道模型(或用 HF token 走 Flux/SD3 子目录 `--vae-subfolder vae`)。

### 1b · TAESD 蒸馏微型解码器(加测, 架构差最远)

```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\probe\taesd --n 300 --methods vae --models madebyollin/taesd --vae-class tiny --cap 0 --dtype bf16 --source-split holdout --seed 778 --device cuda
```

(TAESD 是近似解码器, 出图会更糊一点, 正常 —— 只要认得出是截图就行。)

### 1c · TAESDXL(加测)

```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\probe\taesdxl --n 300 --methods vae --models madebyollin/taesdxl --vae-class tiny --cap 0 --dtype bf16 --source-split holdout --seed 779 --device cuda
```

> **时间紧就先只跑 1a**(ostris 16 通道已经是很强的架构差异测试)。1b/1c 是加分项, 越多样结论越硬。

---

## 步骤 2 · v3 逐个打分

(模型路径按你的实际最优模型; 下面用 aigen_v3)

```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v3\Net_epoch_best.pth --input D:\probe\ostris16 --output_dir D:\probe\ostris16_out --device cuda
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v3\Net_epoch_best.pth --input D:\probe\taesd --output_dir D:\probe\taesd_out --device cuda
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v3\Net_epoch_best.pth --input D:\probe\taesdxl --output_dir D:\probe\taesdxl_out --device cuda
```

(只跑了 1a 就只跑 ostris16 那一条。)

---

## 步骤 3 · 逐个算召回

```
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\ostris16_out\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd_out\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesdxl_out\summary.csv --kind fake
```

看每段的 `[假图召回] Reject 率`。

---

## 步骤 4 · 真图误杀对照(确认这批没引入误杀)

复用 v3 的 held-out 真图 `val\nature`:

```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v3\Net_epoch_best.pth --input D:\ssp_aigen\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_out --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\valnat_out\summary.csv --kind genuine
```

(若 `D:\ssp_aigen\...\val\nature` 不在了, 告诉我, 换别的 held-out 真图目录。)

---

## 怎么读结果

| 结果 | 含义 | 给经理的口径 |
|---|---|---|
| 三个解码器召回都高(>80%)+ 真图误杀低 | v3 跨架构泛化, 学的是通用生成指纹 | 换了架构差异很大的生成器都能抓, 极可能覆盖豆包千问, 等样本正式确认 |
| 部分高部分低 | 部分泛化 | 说清哪种架构抓得到、哪种抓不到(抓不到那种是豆包千问潜在盲区) |
| 都低 | 泛化弱, 认架构 | 覆盖豆包千问需在训练里加多样解码器; 正好指导 v4 + 拿到样本后重训 |

---

## 发我什么

1. 步骤 0 的 `findstr` 输出。
2. 步骤 3 的三段(每段 `Reject 率`) + 步骤 4 的 `Pass率 / 误杀率`。
3. 每个 probe 目录挑 1 张贴我, 确认质量正常。

我据此: 给覆盖结论 + 更新经理口径 + 定下一步(泛化好 → 转 v4 格式鲁棒 + 部署; 有盲区 → 定训练要加哪类解码器)。
