# SSP v4 重训 · 加 16 通道生成器 + 格式增广 · 服务器一步一步

**目标(全程在控 不用豆包千问接口)**
1. 把 ostris 16 通道 VAE 加进训练 —— 千问那类新模型同族 把对它们的判定从"部分进复核"锐化到硬拦。这就是经理说的不断新增生成器 用手上能拿到的最接近千问的开源模型执行。
2. 加 JPEG 和 模糊 训练增广 —— 让压缩过的假图仍被抓 而且压缩/缩放的真图更不会误杀。
3. **硬门槛: v4 必须不比 v3 差**(召回不降 误杀不升)才采用 否则不换。

---

## v3 基线(对照用 别丢)

| 测试集 | v3 召回 Reject | v3 漏检 |
|---|---|---|
| ostris16(16通道) | 76.0% | 0.0% |
| taesd | 93.7% | 0.3% |
| taesdxl | 92.3% | 0.0% |
| val/nature 真图 | 误杀 0.1% | Pass 99.6% |

v4 每项召回要 **>=** 上面 且 误杀要 **<=** 0.1%。尤其误杀不能升。

---

## 前置
```
cd /d D:\alipay-platform-classifier
git pull
```
(拉到新的 patch 规则 jpg_qual 会自动拓宽)

---

## 1. 造 v4 训练假图(4 个 VAE 家族 含 16 通道)(GPU 约 1 小时)
```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\ai_fakes_v4 --n 3000 --methods vae --models stabilityai/sd-vae-ft-mse stabilityai/sd-vae-ft-ema madebyollin/sdxl-vae-fp16-fix ostris/vae-kl-f8-d16 --vae-class kl --cap 0 --dtype bf16 --source-split train --seed 0 --device cuda
```
- 4 个 VAE 各 3000 张 共约 12000 张。`--source-split train` 只用 80% 的源图 留 holdout 给探针不污染。
- `--dtype bf16` 四个都稳(sdxl 和 16 通道在 fp16 会 NaN bf16 不会)。
- **质检**: 每个 tag 各开 1 张 应是能认出的截图 不是黑图。

## 2. 造 held-out 机制假图(img2img 只进 val 测泛化)(GPU)
```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\ai_fakes_v4 --n 800 --methods img2img --models stable-diffusion-v1-5/stable-diffusion-v1-5 --strength 0.4 --cap 1024 --source-split train --seed 1 --device cuda
```
- 追加到同一目录 manifest 会续写。img2img 是没训过的生成机制 留 val 专门测泛化。
- 下不了就换 `Lykon/dreamshaper-8`。

## 3. 组装数据集(img2img 留 val)
```
python training\build_aigen_dataset.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --aigen-root D:\ai_fakes_v4 --holdout-tag stable-diffusion-v1-5 --out D:\ssp_aigen_v4 --n-nature 9000 --val-frac 0.15
```
- 看打印 `train/ai` 应有 mse ema sdxl ostris 四种 `val/ai` 里应能看到 img2img 那个。
- train/ai 和 val/ai 都要非空。

## 4. 轻度质量对齐(保指纹 防 JPEG 捷径)
```
python training\reencode_uniform.py --root D:\ssp_aigen_v4\imagenet_ai_0419_sdv4 --max-side 0 --q 95
```

## 5. 打补丁(含新的 JPEG 增广质量区间)
```
python training\patch_ssp_repo.py --repo D:\SSP-AI-Generated-Image-Detection-main
```
- 会把 jpg_qual 从 [90,100] 拓到 [40,95](配下一步 --jpg_prob 才生效)。同时确认 choices scipy patch.py 都达标。

## 6. 训练 v4(开 JPEG 和 模糊 增广)(GPU 数小时)
```
cd /d D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_aigen_v4 --gpu_id 0 --save_path .\snapshot\aigen_v4\ --jpg_prob 0.5 --blur_prob 0.1
```
- `--jpg_prob 0.5` 一半样本随机 JPEG 压缩(质量 40 到 95)→ 学会被压缩后仍认指纹。
- `--blur_prob 0.1` 轻微模糊 → 抗软化 部分覆盖缩放敏感。
- 产出 `snapshot\aigen_v4\Net_epoch_best.pth`。

---

## 7. 验证 v4 对比 v3(关键门槛)

设一个变量少打字:
```
set V4=D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v4\Net_epoch_best.pth
```

### 7a 跨架构泛化 + 召回(复用探针那批)
```
cd /d D:\SSP
python predict_all_models.py --model_root %V4% --input D:\probe\ostris16 --output_dir D:\probe\ostris16_v4 --device cuda
python predict_all_models.py --model_root %V4% --input D:\probe\taesd --output_dir D:\probe\taesd_v4 --device cuda
python predict_all_models.py --model_root %V4% --input D:\probe\taesdxl --output_dir D:\probe\taesdxl_v4 --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\ostris16_v4\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd_v4\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesdxl_v4\summary.csv --kind fake
```

### 7b held-out img2img 召回(测泛化到没训过的机制)
```
cd /d D:\SSP
python predict_all_models.py --model_root %V4% --input D:\ssp_aigen_v4\imagenet_ai_0419_sdv4\val\ai --output_dir D:\probe\i2i_v4 --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\i2i_v4\summary.csv --kind fake
```

### 7c 误杀 battery(必须不比 v3 差)
```
cd /d D:\SSP
python predict_all_models.py --model_root %V4% --input D:\ssp_aigen_v4\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_v4 --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\valnat_v4\summary.csv --kind genuine
```
再测压缩过和缩放过的真图不误杀:
```
python training\recompress_dir.py --src D:\ssp_aigen_v4\imagenet_ai_0419_sdv4\val\nature --out D:\probe\valnat_wechat --q 65 --double
python training\recompress_dir.py --src D:\ssp_aigen_v4\imagenet_ai_0419_sdv4\val\nature --out D:\probe\valnat_resize --q 90 --max-side 1024
cd /d D:\SSP
python predict_all_models.py --model_root %V4% --input D:\probe\valnat_wechat --output_dir D:\probe\valnat_wechat_v4 --device cuda
python predict_all_models.py --model_root %V4% --input D:\probe\valnat_resize --output_dir D:\probe\valnat_resize_v4 --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\valnat_wechat_v4\summary.csv --kind genuine
python training\eval_summary.py D:\probe\valnat_resize_v4\summary.csv --kind genuine
```

### 7d 压缩过的假图召回(新增 测压缩没把指纹洗掉)
```
python training\recompress_dir.py --src D:\probe\ostris16 --out D:\probe\ostris16_wechat --q 65 --double
cd /d D:\SSP
python predict_all_models.py --model_root %V4% --input D:\probe\ostris16_wechat --output_dir D:\probe\ostris16_wechat_v4 --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\ostris16_wechat_v4\summary.csv --kind fake
```

---

## 采用判据(重要)
- v4 每项召回 **>=** v3 且 每项误杀 **<=** v3(尤其 7c 误杀不能升)→ **采用 v4** 把预测用的模型路径换成 aigen_v4。
- 只要有一项明显退步 尤其误杀升了 → **不采用** 把数据发我 我调增广强度或配比再来一版。
- 对比时看关键: ostris16 的召回应该比 v3 的 76% 更高(因为这次训练里放了 16 通道)误杀应仍是 0 附近。

## 发我什么
7a 7b 7d 的召回各段 + 7c 的误杀各段 + 造图时每个 tag 挑 1 张。我做 v3 对 v4 的对照结论 定采用不采用 + 更新给经理的口径。
