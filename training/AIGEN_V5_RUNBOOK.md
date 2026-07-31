# SSP v5 · 把真 Qwen + Flux 加进训练 · 一步步(PowerShell)

**为什么**: 探针发现 **v4 对真 Qwen 漏检 70%**(median 0.51 —— 看得到痕迹但没学到)。Flux 也漏 14.7%。
**做法**: probe-then-add(同 ostris 76%→95%)—— 把真 Qwen + Flux 的假图加进训练。预期真 Qwen 三成→九成以上。
**省事点**: 复用 v4 的假图池 只新造 Qwen + Flux 两个生成器。

## v4 基线(对照 别丢)
| 测试集 | v4 |
|---|---|
| **真 Qwen** | 漏 70.2% (median 0.51) ← 要修 |
| Flux | 漏 14.7% |
| TAESD3 / SD3 族 | 漏 0.3%(已覆盖 不用加) |
| ostris / taesd / taesdxl 干净 | 95 / 92.7 / 93.3 |
| img2img held-out | 88.1 |
| 误杀 原图/微信/缩放 | 0.1 / 0.1 / 0 |

---

## 前置
```
cd D:\alipay-platform-classifier
git pull
python -c "import diffusers; print(diffusers.__version__)"
```
diffusers <0.34 → `pip install -U diffusers`(Qwen 需要)。

## 0 确认 v4 假图池还在(重要 别跳)
```
(Get-ChildItem D:\ai_fakes_v4\aivae_sd-vae-ft-mse_*.jpg -ErrorAction SilentlyContinue | Measure-Object).Count
```
- 应约 3000。**若是 0** = `D:\ai_fakes_v4` 被删/清过 → **停 告诉我**。这时不能只追加 Qwen/Flux(会丢掉 mse/ema/sdxl/ostris 老生成器 v5 就废了)要重造全部。

## 1 造真 Qwen 训练假图(追加到 v4 假图池)
```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\ai_fakes_v4 --n 3000 --methods vae --models Qwen/Qwen-Image --vae-subfolder vae --vae-class qwen --cap 1024 --dtype bf16 --source-split train --seed 811 --device cuda
```
- 追加到 `D:\ai_fakes_v4`(v4 的已建数据集 `D:\ssp_aigen_v4` 在别的目录 不受影响)。
- `--source-split train`: 用训练源 和探针的 holdout 源不相交 → 探针 `D:\probe\qwen` 仍是干净测试集。
- 开跑 20 秒 QC 开头几张(同探针)是能认出的截图。

## 2 造 Flux 训练假图(追加)
```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\ai_fakes_v4 --n 3000 --methods vae --models diffusers/FLUX.1-vae --vae-class kl --cap 0 --dtype bf16 --source-split train --seed 812 --device cuda
```

## 3 重建数据集(现在含 Qwen + Flux)
```
python training\build_aigen_dataset.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --aigen-root D:\ai_fakes_v4 --holdout-tag stable-diffusion-v1-5 --out D:\ssp_aigen_v5 --n-nature 18000 --val-frac 0.15
```
- 看打印 `train/ai 生成器` 应新增 `Qwen-Image` 和 `FLUX.1-vae`。train/ai val/ai 都非空。
- 数据集比 v4 大(多了两个生成器)训练会久一点 正常。

## 4 质量对齐
```
python training\reencode_uniform.py --root D:\ssp_aigen_v5\imagenet_ai_0419_sdv4 --max-side 0 --q 95
```

## 5 打补丁(先去掉 v4b 的双压增广 回到 v4 的增广)
```
Copy-Item D:\SSP-AI-Generated-Image-Detection-main\utils\tdataloader.py.bak D:\SSP-AI-Generated-Image-Detection-main\utils\tdataloader.py -Force
python training\patch_ssp_repo.py --repo D:\SSP-AI-Generated-Image-Detection-main
findstr /n "v4b" D:\SSP-AI-Generated-Image-Detection-main\utils\tdataloader.py
```
- 前两行: 还原 tdataloader 原始版(去掉 v4b 跑过的双压)+ 重打 scipy 修复 + jpg_qual + 选块 但**不再加双压**(规则已撤)。
- 第三行 `findstr` **应无输出(空)**。若打印出带 v4b 的行 = 双压没去掉 → 告诉我。
- 若提示 .bak 不存在, 告诉我(我给手动去掉双压两行的办法)。

## 6 训练 v5(用 v4 的增广)
```
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_aigen_v5 --gpu_id 0 --save_path .\snapshot\aigen_v5\ --jpg_prob 0.5 --blur_prob 0.1
```
- `--jpg_prob 0.5`(v4 的 不是 v4b 的 0.6)。产出 `snapshot\aigen_v5\Net_epoch_best.pth`。

---

## 7 验证 v5 对比 v4(重点看真 Qwen 有没有修好 + 别的别退步)
```
$v5 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v5\Net_epoch_best.pth"
```

### 7a 真 Qwen + Flux(关键 应大幅提升)
```
cd D:\SSP
python predict_all_models.py --model_root $v5 --input D:\probe\qwen --output_dir D:\probe\qwen_v5 --device cuda
python predict_all_models.py --model_root $v5 --input D:\probe\flux --output_dir D:\probe\flux_v5 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\qwen_v5\summary.csv --kind fake
python training\eval_summary.py D:\probe\flux_v5\summary.csv --kind fake
```
真 Qwen 应从漏 70% 降到漏≈0(reject 大涨)。

### 7b 老生成器别退步(复用探针 对比 v4: ostris95 / taesd92.7 / taesdxl93.3 / taesd3 95.3 / img2img88.1)
```
cd D:\SSP
python predict_all_models.py --model_root $v5 --input D:\probe\ostris16 --output_dir D:\probe\ostris16_v5 --device cuda
python predict_all_models.py --model_root $v5 --input D:\probe\taesd --output_dir D:\probe\taesd_v5 --device cuda
python predict_all_models.py --model_root $v5 --input D:\probe\taesdxl --output_dir D:\probe\taesdxl_v5 --device cuda
python predict_all_models.py --model_root $v5 --input D:\probe\taesd3 --output_dir D:\probe\taesd3_v5 --device cuda
python predict_all_models.py --model_root $v5 --input D:\probe\i2i_holdout --output_dir D:\probe\i2i_v5 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\ostris16_v5\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd_v5\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesdxl_v5\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd3_v5\summary.csv --kind fake
python training\eval_summary.py D:\probe\i2i_v5\summary.csv --kind fake
```

### 7c 误杀别升(对比 v4 的 0.1 / 0.1 / 0)
```
cd D:\SSP
python predict_all_models.py --model_root $v5 --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_v5 --device cuda
python predict_all_models.py --model_root $v5 --input D:\probe\valnat_wechat --output_dir D:\probe\valnat_wechat_v5 --device cuda
python predict_all_models.py --model_root $v5 --input D:\probe\valnat_resize --output_dir D:\probe\valnat_resize_v5 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\valnat_v5\summary.csv --kind genuine
python training\eval_summary.py D:\probe\valnat_wechat_v5\summary.csv --kind genuine
python training\eval_summary.py D:\probe\valnat_resize_v5\summary.csv --kind genuine
```

---

## 采用判据
- **真 Qwen 召回大幅提升(漏检降到≈0)且 7b 老生成器不退步 7c 误杀不升** → 采用 v5 把预测模型换成 aigen_v5。
- 老生成器或误杀退步 → 发我 我调配比(n-nature / 各生成器数量)。

## 发我什么
7a 两段(真 Qwen 是重点)+ 7b 五段 + 7c 三段。我做 v4 对 v5 对照 定采用 + 更新经理口径。
