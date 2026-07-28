# SSP 重训为「AI 生成检测器」· 服务器一步一步(修掉第一版 AI 召回=0 的坑)

主目标(经理定):**检测 AI 生成假图**。第一版 AI 召回 0% 不是方向错, 是三个坑:
1. **过度压缩把 AI 指纹洗掉了**(reencode 到 768)→ 本轮 AI 假图**原生分辨率**, reencode 只做轻度质量对齐。
2. **val 全翻拍零 AI** → 本轮 ai 主体就是 AI 假图, 且**留一个生成器只进 val** 测泛化。
3. (选块盲区 = flattest-patch) → 先修 1+2 重训测一版; 若 held-out 召回还低, 再改选块(stage 2)。

多样性是关键(truthscan 抓没见过的 AI 靠的是多生成器, 不是模型大): 用**多个 VAE 混造**, 留一个不训只测。

命令 cmd 风格。前置 `cd /d D:\alipay-platform-classifier` 然后 `git pull`。

---

## 1. 造多样 AI 假图(原生分辨率, 多生成器)(GPU, 30-60 分钟)
```
cd /d D:\alipay-platform-classifier
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\ai_fakes --n 3500 --methods vae --models stabilityai/sd-vae-ft-mse stabilityai/sd-vae-ft-ema madebyollin/sdxl-vae-fp16-fix --cap 0 --device cuda
```
- `--cap 0` = **原生分辨率**(保住指纹, 别缩)。`--n 3500` 是每个模型的源图数 → 3 个 VAE ≈ 1 万张 AI 假图。
- 产出 `D:\ai_fakes\*.jpg` + `manifest.csv`(记了每张的生成器)。
- **质检**: 打开几张看是能认出的截图(略糊/略软), 不是黑图/噪声。某个 VAE 出黑图(fp16 NaN)就从 `--models` 去掉它。
- 想更多样(不同机制): 再补一批 img2img(需联网下 SD 模型, 慢):
  `... --out D:\ai_fakes --methods img2img --models stabilityai/stable-diffusion-2-1 --strength 0.4 --cap 0`
  (追加到同目录; manifest 会续写。img2img 也可当另一个 held-out 候选。)

---

## 2. 组装数据集(AI 假图为主 + held-out 生成器进 val)(5-15 分钟)
```
python training\build_aigen_dataset.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --aigen-root D:\ai_fakes --holdout-tag sdxl-vae-fp16-fix --out D:\ssp_aigen --n-nature 9000 --val-frac 0.15
```
- `--holdout-tag sdxl-vae-fp16-fix`: 这个生成器的假图**只进 val**, 训练完全没见过 → 测泛化。
- 看打印: `train/ai` 只应有 sd-vae-ft-mse/ema; `val/ai 各生成器` 里要能看到 `sdxl-vae-fp16-fix xN`。
- val/nature、val/ai 都要非空。

---

## 3. 轻度质量对齐(保指纹, 别再洗掉)(5-10 分钟)
```
python training\reencode_uniform.py --root D:\ssp_aigen\imagenet_ai_0419_sdv4 --max-side 0 --q 95
```
- **`--max-side 0` = 不缩尺寸**(关键, 上次就是缩到 768 把指纹洗了); `--q 95` 高质量, 只把真假两类压缩对齐防"JPEG=假"泄漏。

---

## 4. 训练(GPU, 与上次同脚本)
```
cd /d D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_aigen --gpu_id 0 --save_path .\snapshot\aigen\
```
- 这次 val/ai 有 AI 假图 → "最优轮"是按 **AI 检测** 选的(修了坑3)。产出 `snapshot\aigen\Net_epoch_best.pth`。

---

## 5. 老实报"没见过的生成器"召回(这才是真识别率)
把 held-out 生成器的假图单独拷出来测(PowerShell):
```
Remove-Item D:\aigen_test\heldout -Recurse -Force -ErrorAction SilentlyContinue; mkdir D:\aigen_test\heldout
Copy-Item D:\ssp_aigen\imagenet_ai_0419_sdv4\val\ai\aivae_sdxl-vae-fp16-fix_*.jpg -Destination D:\aigen_test\heldout
```
预测 + 算召回:
```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen\Net_epoch_best.pth --input D:\aigen_test\heldout --output_dir D:\aigen_test\heldout_out --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\aigen_test\heldout_out\summary.csv --kind fake
```
再测真图误杀(specificity):
```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen\Net_epoch_best.pth --input D:\ssp_aigen\imagenet_ai_0419_sdv4\val\nature --output_dir D:\aigen_test\valnat_out --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\aigen_test\valnat_out\summary.csv --kind genuine
```

**怎么读**:
- **held-out 召回高(>80%)+ 真图误杀低** = 成了, 它真在抓 AI 生成、且能泛化到没训过的生成器。可以对经理说 AI 检测跑通。
- **held-out 召回还低** = 坑1+2 里"选块"是主因 → 进 stage 2: 改 SSP 选块(看整图/细节区非最平块)。我来改。
- 发我: held-out 召回整段 + 真图误杀整段(+ 造假图时挑 1-2 张贴我看质量)。

---

## 诚实口径
- 召回按 **held-out 生成器** 报 = 对"没见过的 AI"的真实识别率; 别用训练见过的生成器报(那乐观)。
- 合成 AI 假图训练无妨, 但真实世界 AI 欺诈可能更多样 → 上线后 review-only 收真样本持续校准。
