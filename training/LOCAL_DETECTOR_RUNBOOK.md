# 局部篡改检测(Head B)· 训练一个专门看"小块"的模型 · 一步步(PowerShell)

## 为什么要它(前面用数据证明的)
| 方案 | 局部改金额的召回 | 真图误杀 |
|---|---|---|
| v6 整图打分 | **0%**(300张全漏, median 0.002 = 和真图一样) | 0.1% |
| v6 + 分块取max(不重训) | 0.1%误杀预算下 **13.7%**(2%预算也才到23%) | 0.2% |
| **本方案(专门训练)** | 目标大幅提升 | 目标 ≤0.2% |

**根因**: v6 是按"整图里选最富纹理的那一块"训练的, 孤立的小块对它是**分布外**输入, 判别力弱 ->
分块取 max 只是把噪声和信号一起放大(真图也有块打到 1.000)。
**正路**: 直接拿**小块**训练一个判别器, 让它在小块上就分得开; 推理时分块取 max 才成立。

## 训练数据怎么来(免费且干净)
`gen_local_ai_edit.py --save-crops` 会输出**配对裁块**:
- `ai/crop_000123.jpg` = 改动区(被 AI 重绘过)的裁块
- `nature/crop_000123.jpg` = **同一张原图 同一位置** 的裁块
两者**内容完全一样, 只差有没有被 AI 动过** -> 这是最干净的正负样本, 而且标注是免费的(位置我们自己定的)。

---

## 1 造配对训练块(GPU, 几十分钟)
```
cd D:\alipay-platform-classifier
git pull
python training\gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit_train --save-crops D:\localcrops --n 3000 --mode local --device cuda
```
- `--n 3000` 目标 3000 对(定位不到金额的会跳过, 实际少一些)。
- 产出 `D:\localcrops\ai\*.jpg` 和 `D:\localcrops\nature\*.jpg`, **一一配对**。
- **质检**: 打开一对同名的(ai 和 nature 各一张)—— 应该看着几乎一样(都是金额那块), 这就对了。

## 2 组装成训练目录(官方 loader 的固定结构)
```
python training\build_crop_dataset.py --crops D:\localcrops --out D:\ssp_local --val-frac 0.15
```
看打印: train/ai train/nature val/ai val/nature 四个都要非空, 且 ai 与 nature 数量接近。

## 3 训练(GPU, 比整图训练快很多 —— 图小)
```
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_local --gpu_id 0 --save_path .\snapshot\localdet\ --jpg_prob 0.5 --blur_prob 0.1
```
产出 `snapshot\localdet\Net_epoch_best.pth`。

## 4 用分块方式测整张图(这才是上线的用法)
```
cd D:\alipay-platform-classifier
$ld = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet\Net_epoch_best.pth"
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld --input D:\probe\localedit --output_dir D:\probe\localedit_ld --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_ld --limit 1000 --device cuda
python training\sweep_thresholds.py --fake D:\probe\localedit_ld\summary.csv --genuine D:\probe\valnat_ld\summary.csv
```
**关键看 sweep 的"误杀 0.1% 预算下召回"** —— 对照 v6 分块的 **13.7%**:
- 大幅超过(比如 >60%) → 成了, 局部篡改这条线可以上, 和 v6 一起用(v6 抓整图AI, 它抓局部改动)。
- 只是略好 → 说明小块里的 VAE 指纹本身就弱, 要换思路(传统取证特征: ELA/噪声不一致/JPEG ghost, 这类对复制粘贴改字也管用)。
- 没提升 → 发我数据, 我重新设计。

## 5(重要)也要测传统非AI篡改
经理最初的 03/04 是**直接改数字**(不一定用AI)。造一批用 engine_b_tamper:
```
python training\engine_b_tamper.py --src-root D:\download2\TempFakeImages --out D:\probe\tamper_classic --n 300
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld --input D:\probe\tamper_classic --output_dir D:\probe\tamper_classic_ld --device cuda
python training\eval_summary.py D:\probe\tamper_classic_ld\summary.csv --kind fake
```
(本模型是按"AI 重绘的块"训练的, 对复制粘贴/重打字未必有效 —— 测了才知道, 这决定要不要再补取证特征那条线。)

## 发我什么
1 的一对样图 + 2 的打印 + **4 的 sweep 输出(重点)** + 5 的召回。我据此定这条线成不成、要不要换方法、以及怎么和 v6 一起配置上线。
