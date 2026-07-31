# 真实生成器覆盖探针 · 真Qwen + Flux + SD3族 · 一步步(PowerShell)

**目标**: 经理要"不断新增生成器"。用真实现代生成器的 VAE 测 v4 覆盖 —— **全开源 不用 App/API/VPN**:
- **真 Qwen-Image VAE** —— 经理点名的千问 开源权重 直接测**真指纹**(不是代理)。
- **Flux VAE** —— 顶级现代生成器 16 通道。
- **SD3 族(TAESD3)** —— SD3 潜空间的蒸馏 VAE 免 token。

probe-then-add(同 ostris): v4 抓到就确认覆盖; 漏的加进下一版训练。诚实预期: 都是 16 通道族(v4 靠 ostris 已覆盖)→ 多半已抓 主要是**确认真生成器 + 履行经理"多加"指令 + 拿到真 Qwen 的硬证据**。

---

## 前置
```
cd D:\alipay-platform-classifier
git pull
python -c "import diffusers; print(diffusers.__version__)"
```
- 真 Qwen VAE 需 **diffusers >= 0.34**。若打印 <0.34 → `pip install -U diffusers`(其他两个不挑版本)。
- 全程无需 HF token(Flux 走免 token 的 `diffusers/FLUX.1-vae`; Qwen/TAESD3 本就开源非门控)。

---

## 1a 真 Qwen-Image VAE(经理点名 重点)
```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\probe\qwen --n 300 --methods vae --models Qwen/Qwen-Image --vae-subfolder vae --vae-class qwen --cap 1024 --dtype bf16 --source-split holdout --seed 801 --device cuda
```
- 只下 VAE 子目录(约 254MB)。`--cap 1024` 省显存(Qwen 是视频 VAE 5D 较吃内存)显存够可改 `--cap 0`。
- **质检(重要 这是新代码路径 先看再等)**: 开跑 20 秒后 到 `D:\probe\qwen` 开头几张看一眼 —— 必须是**能认出的支付宝截图**。若发黑/发灰/发白/颜色反常 = 解码输出范围不对 → **Ctrl-C 停 告诉我 我修 别等 300 张跑完**。正常再让它跑完。

## 1b Flux VAE
```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\probe\flux --n 300 --methods vae --models diffusers/FLUX.1-vae --vae-class kl --cap 0 --dtype bf16 --source-split holdout --seed 802 --device cuda
```
- 用免 token 的官方镜像 `diffusers/FLUX.1-vae`(FLUX.1-schnell 本体是 gated 要 token 不用它)。质检同上(开几张看是截图)。

## 1c SD3 族(TAESD3)
```
python training\gen_ai_fakes.py --genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages --out D:\probe\taesd3 --n 300 --methods vae --models madebyollin/taesd3 --vae-class tiny --cap 0 --dtype bf16 --source-split holdout --seed 803 --device cuda
```
- SD3 本体 gated 用免 token 的 SD3 潜空间蒸馏 VAE `taesd3`。出图会更糊点 正常(蒸馏)。

---

## 2 v4 逐个打分
```
$v4 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v4\Net_epoch_best.pth"
cd D:\SSP
python predict_all_models.py --model_root $v4 --input D:\probe\qwen --output_dir D:\probe\qwen_out --device cuda
python predict_all_models.py --model_root $v4 --input D:\probe\flux --output_dir D:\probe\flux_out --device cuda
python predict_all_models.py --model_root $v4 --input D:\probe\taesd3 --output_dir D:\probe\taesd3_out --device cuda
```

## 3 算召回
```
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\qwen_out\summary.csv --kind fake
python training\eval_summary.py D:\probe\flux_out\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd3_out\summary.csv --kind fake
```

---

## 怎么读
- **真 Qwen 召回高 漏检≈0** = v4 直接覆盖真千问 → 可给经理**硬证据(是真 Qwen 不是代理)**。
- Flux / SD3 族 高 = 覆盖顶级现代生成器。
- 谁低(漏检高)= 那个加进下一版训练(probe-then-add 同 ostris 76%→95%)。
- 参照 v4 基线: ostris 16 通道 95% / taesd taesdxl 92-93%。这几个同为 16 通道族 预期相近。

## 发我什么
三段召回 + 每个目录 1 张样图(尤其 **Qwen 那张我要确认新代码质量对不对**)。我据此定覆盖结论 + 要不要把哪个加进训练 + 更新经理口径。
