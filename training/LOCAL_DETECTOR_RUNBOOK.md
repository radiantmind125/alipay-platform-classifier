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

## 1 造配对训练块 · **用三个不同的生成器**(GPU, 约 1 小时)

**为什么要三个**: 千问那次的教训 —— 只按一个生成器训, 换个生成器就抓不到(当时漏了七成)。
真骗子可能用 PS 生成填充 / SDXL 局部重绘 / 别的工具, 不会正好是我们训练用的那个。
所以三个不同架构的 VAE 各造一批, 让它学**通用的"这块被 AI 动过"**而不是某一个模型的指纹。

```
cd D:\alipay-platform-classifier
git pull
python training\gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit_train --save-crops D:\localcrops --n 1500 --mode local --model stabilityai/sd-vae-ft-mse --device cuda
python training\gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit_train --save-crops D:\localcrops --n 1500 --mode local --model madebyollin/sdxl-vae-fp16-fix --seed 11 --device cuda
python training\gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit_train --save-crops D:\localcrops --n 1500 --mode local --model ostris/vae-kl-f8-d16 --dtype bf16 --seed 22 --device cuda
```
- 三条命令**累积**到同一个 `D:\localcrops`(文件名带模型标记, 不会互相覆盖)。合计约 4500 对。
- 不同 `--seed` = 用不同的源图, 增加多样性。
- 第三个是 16 通道 VAE, **必须 `--dtype bf16`**(fp16 会出黑图)。
- **质检**: 打开一对同名的(`D:\localcrops\ai\crop_sd-vae-ft-mse_000001.jpg` 和 `nature\` 里同名那张)——
  应该看着几乎一样(都是金额那块), 这就对了。再确认三个模型的文件都在:
```
Get-ChildItem D:\localcrops\ai | Group-Object {($_.BaseName -split '_')[1]} | Select-Object Name,Count
```
  应看到三行(sd-vae-ft-mse / sdxl-vae-fp16-fix / vae-kl-f8-d16)每行约 1500。

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

### ⚠️ 前 5 轮就要判断学没学动(别等 30 轮)
**学得动的样子**: loss 从 0.69 往下掉(<0.6), Accuracy 稳步上升(>0.65 并继续涨)。
**学不动的样子(立刻停, 告诉我)**:
```
loss 一直在 0.63~0.72 之间晃(0.693 就是瞎猜)
Accuracy 卡在 0.5~0.6
ai accu 和 nature accu **反向摆动**(比如 0.87/0.14 -> 0.20/0.96 -> 0.31/0.81)
```
最后那条是典型信号: 模型分不开, 只是在来回挪判定线。
**2026-08-04 第一次跑就是这样, 原因是裁块外扩到 128 -> 裁块里八成是没被动过的真像素 ->
patch_img 选中的 32x32 多半落在真像素上 -> "ai" 标签一半是错的 -> 学不动。**
已修: `--crop-min` 默认改成 0 = **紧贴改动区裁**(裁块内每个像素都被 AI 动过)。
**所以第 1 步必须用新代码重新造裁块**(旧的 D:\localcrops 要删掉重来)。

## 4 用分块方式测整张图(这才是上线的用法)

**注意用和 v6 完全一样的口径去比**(--roi-amount + --require-located), 否则不是同一把尺子:
```
cd D:\alipay-platform-classifier
$ld = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet\Net_epoch_best.pth"
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld --input D:\probe\localedit --output_dir D:\probe\localedit_ld --roi-amount --roi-top 0.6 --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_ld --roi-amount --roi-top 0.6 --device cuda
python training\sweep_thresholds.py --fake D:\probe\localedit_ld\summary.csv --genuine D:\probe\valnat_ld\summary.csv --require-located
```
**对照基线(v6 同口径, 全量2700真图 + 仅定位成功)**:
| 误杀预算 | v6 基线 | 本模型 |
|---|---|---|
| 0.1% | **33.3%** | ? |
| 1.0% | **66.0%** | ? |

- 明显超过基线 → 成了, 这条线可以上(v6 抓整图AI, 它抓局部改金额)。
- 只是持平/略好 → 说明小块里的指纹本身就弱, 要换思路(传统取证特征: ELA/噪声不一致/JPEG ghost —— 这类对**复制粘贴改字**也管用, 正好补第5步那块)。
- 反而更差 → 发我数据(附训练日志最后几轮的 val accuracy), 我看是过拟合还是别的。

**重要**: `D:\probe\localedit` 是用 `sd-vae-ft-mse` 造的(第一次那批), 而本模型**训练里见过这个生成器** —— 所以这个数字偏乐观。
更诚实的做法是另造一批**训练没用过的**生成器的局部编辑图来测:
```
python training\gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit_heldout --n 300 --mode local --model stabilityai/sd-vae-ft-ema --seed 99 --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld --input D:\probe\localedit_heldout --output_dir D:\probe\localedit_heldout_ld --roi-amount --roi-top 0.6 --device cuda
python training\sweep_thresholds.py --fake D:\probe\localedit_heldout_ld\summary.csv --genuine D:\probe\valnat_ld\summary.csv --require-located
```
(`sd-vae-ft-ema` 没进训练 -> 这个数字才代表"没见过的编辑工具"的真实识别率。)

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

---

# 附: v2 —— 把蓝图也训进来(白图版已验证成功后做)

**为什么**: 只训白图的模型**迁移不到蓝图**(实测: 白图 92.2% vs 蓝图只有 22.0% @0.1%误杀)。
白字蓝底和深字白底长得太不一样, 对模型是分布外输入。
**收益**: 开了蓝图定位器后, 真图能出信号的比例从 **952/2700(35%) 升到 2253/2700(83%)** —— 局部覆盖翻一倍还多。

## v2-1 造蓝图配对裁块(3 个生成器, 累积到同一个 crops 目录)
**必须用 `--tag` 区分**, 否则文件名会和白图那批撞车被覆盖(白图用的也是同样的模型名):
```
python training\gen_local_ai_edit.py --src-root D:\download\TempFakeImages --out D:\probe\le_blue_train --save-crops D:\localcrops --n 1500 --mode local --page-type blue --model stabilityai/sd-vae-ft-mse --tag blue-mse --seed 101 --device cuda
python training\gen_local_ai_edit.py --src-root D:\download\TempFakeImages --out D:\probe\le_blue_train --save-crops D:\localcrops --n 1500 --mode local --page-type blue --model madebyollin/sdxl-vae-fp16-fix --tag blue-sdxl --seed 111 --device cuda
python training\gen_local_ai_edit.py --src-root D:\download\TempFakeImages --out D:\probe\le_blue_train --save-crops D:\localcrops --n 1500 --mode local --page-type blue --model ostris/vae-kl-f8-d16 --tag blue-ostris --dtype bf16 --seed 121 --device cuda
```
- 源用 `D:\download\TempFakeImages`(蓝图在这个池子里多)。
- **校验**(应看到 6 组: 3 个白图 tag + 3 个 blue- 开头的):
```
Get-ChildItem D:\localcrops\ai | Group-Object {($_.BaseName -split '_')[1]} | Select-Object Name,Count
```
- **质检**: 开一对 `crop_blue-mse_*`(ai 和 nature 同名), 应该是蓝底上的白色金额, 两张几乎一样。

## v2-2 重建数据集 + 重训(白图蓝图一起训一个模型)
```
python training\build_crop_dataset.py --crops D:\localcrops --out D:\ssp_local2 --val-frac 0.15
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_local2 --gpu_id 0 --save_path .\snapshot\localdet2\ --jpg_prob 0.5 --blur_prob 0.1
```
(一个模型同时管两种页型, 部署简单; 前 5 轮同样看学没学动: loss 掉到 0.6 以下 + Accuracy 往上走。)

## v2-3 验证(白图不能退步, 蓝图要大幅提升)
```
cd D:\alipay-platform-classifier
$ld2 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet2\Net_epoch_best.pth"
:: 真图基线(两种页型都出信号)
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld2 --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_ld2 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
:: 白图(对照 92.5%)
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld2 --input D:\probe\localedit_heldout --output_dir D:\probe\le_white_ld2 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
:: 蓝图(对照 22.0%)
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld2 --input D:\probe\localedit_blue --output_dir D:\probe\le_blue_ld2 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
python training\sweep_thresholds.py --fake D:\probe\le_white_ld2\summary.csv --genuine D:\probe\valnat_ld2\summary.csv --require-located
python training\sweep_thresholds.py --fake D:\probe\le_blue_ld2\summary.csv --genuine D:\probe\valnat_ld2\summary.csv --require-located
```
**判据**: 白图不明显退步(≥85%) 且 蓝图大幅提升(目标 ≥70%)→ 采用 localdet2, 覆盖从 35% 升到 83%。
若蓝图上去了但白图掉了 → 说明一个模型吃不下两种页型, 那就**分开训两个**(推理时按页型分派, 代码已支持)。
