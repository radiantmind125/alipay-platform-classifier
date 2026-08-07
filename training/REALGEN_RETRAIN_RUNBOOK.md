# 把**真实国内服务**(豆包 Seedream + 阿里万相)加进训练 · 两条线一起

## 为什么做
现在两条线对真实服务都能**覆盖**(漏检 0%), 但**置信度不够**:
| | 线路A 漏检 | 线路A 高置信硬拦 | 线路B 召回 |
|---|---|---|---|
| 千问(**已训**) | 0% | **98%** | — |
| 豆包 Seedream | 0% | 63-72% | 94.5% |
| **万相 Wan** | 0% | **2%** ⚠️ | 80.9% |
- **万相只有 2% 硬拦**: review-only 下没事(全都会被人看到), **但以后若加"分数≥0.90 自动拒"的档位, 万相假图会整批溜过去。**
- 对照千问: **进过训练就是 98%**。所以把真实服务喂进去, 效果是明确的。
- 这也正是经理反复说的"不断新增生成器"。

## 关键省时技巧: 一次调用喂两条线
整张重绘图**本身**就是线路A 的训练样本; 把它的**金额区贴回原图**又得到线路B 的配对裁块。
所以用 `--send full --save-full --save-crops` **一次 API 调用同时产出两条线的数据**, 省一半时间。

⚠️ **万相单张 60-90 秒, 豆包约 30-60 秒** —— 这是整个流程最慢的部分, 建议**挂着跑**(晚上/后台)。

---

## 1 造真实服务数据(慢, 挂着跑)

```
cd D:\alipay-platform-classifier
git pull
$env:DMX_KEY = "sk-你自己的key"      # 只在本机设, 别写进任何文件
```

> ⚠ **key 绝对不能写进这个仓库 —— 仓库是公开的。**
> 这里以前真的贴过一个 key(commit b457f79), 已经泄露过一次, 那个 key 必须作废重申请。
> 正确做法: 每次开 PowerShell 临时 `$env:DMX_KEY = "..."`, 或者放系统环境变量, **不进版本库**。

**万相(优先, 因为它最弱)**:
```
python training\gen_api_local_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\api_local_wan --save-crops D:\localcrops --save-full D:\api_full --n 400 --model wan2.7-image --send full --tag apiwan --seed 201
```
**豆包**(整图会被风控拒一部分, 脚本会跳过继续; 源图备货 6 倍够用):
```
python training\gen_api_local_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\api_local_seed --save-crops D:\localcrops --save-full D:\api_full --n 400 --model doubao-seedream-4-5-251128 --send full --tag apiseed --seed 202
```
- `--send full` 是必须的(要拿到整张重绘图喂线路A)。
- `--tag` 是必须的(和已有的 6 组裁块区分, 否则覆盖)。
- 中断了直接重跑, 会断点续跑。**每个跑到 200-300 张也够用**, 不必强求 400。

**校验**(应看到原来的 6 组 + 新的 apiwan / apiseed):
```
Get-ChildItem D:\localcrops\ai | Group-Object {($_.BaseName -split '_')[1]} | Select-Object Name,Count
(Get-ChildItem D:\api_full -Filter *.jpg | Measure-Object).Count
```

## 2 重训线路B(局部改金额)
```
python training\build_crop_dataset.py --crops D:\localcrops --out D:\ssp_local3 --val-frac 0.15
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_local3 --gpu_id 0 --save_path .\snapshot\localdet3\ --jpg_prob 0.5 --blur_prob 0.1
```
(前 5 轮照例看学没学动: loss 掉到 0.6 以下 + Accuracy 往上走。)

## 3 重训线路A(整图 AI 生成)
把真实服务的整图重绘并进 v5 的 ai 类:
```
cd D:\alipay-platform-classifier
Copy-Item D:\api_full\*.jpg -Destination D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\train\ai\
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_aigen_v5 --gpu_id 0 --save_path .\snapshot\aigen_v7\ --jpg_prob 0.5 --blur_prob 0.1
```
(真实服务样本只占几个百分点, 但它们是"最像真实威胁"的那批, 通常足以把置信度拉起来 —— 千问就是这么从 0 漏但低置信变成 98% 硬拦的。)

---

## 4 验证(两条线都要, 且**老的不能退步**)

**线路A(v7)**:
```
cd D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth --input D:\probe\wan_full --output_dir D:\probe\wan_full_v7 --device cuda
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth --input D:\probe\qwen --output_dir D:\probe\qwen_v7 --device cuda
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_v7 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\wan_full_v7\summary.csv --kind fake
python training\eval_summary.py D:\probe\qwen_v7\summary.csv --kind fake
python training\eval_summary.py D:\probe\valnat_v7\summary.csv --kind genuine
```
**判据**: 万相硬拦从 **2%** 大幅上升; 千问保持 ~98%; **真图误杀不超过 0.1%**。

**线路B(localdet3)**:
```
$ld3 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet3\Net_epoch_best.pth"
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld3 --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_ld3 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld3 --input D:\probe\wan_local --output_dir D:\probe\wan_local_ld3 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld3 --input D:\probe\localedit_seedream --output_dir D:\probe\seed_local_ld3 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld3 --input D:\probe\localedit_heldout --output_dir D:\probe\white_ld3 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $ld3 --input D:\probe\localedit_blue --output_dir D:\probe\blue_ld3 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
python training\sweep_thresholds.py --require-located --fake D:\probe\wan_local_ld3\summary.csv   --genuine D:\probe\valnat_ld3\summary.csv
python training\sweep_thresholds.py --require-located --fake D:\probe\seed_local_ld3\summary.csv  --genuine D:\probe\valnat_ld3\summary.csv
python training\sweep_thresholds.py --require-located --fake D:\probe\white_ld3\summary.csv       --genuine D:\probe\valnat_ld3\summary.csv
python training\sweep_thresholds.py --require-located --fake D:\probe\blue_ld3\summary.csv        --genuine D:\probe\valnat_ld3\summary.csv
```
**判据(@0.5% 误杀, 对照现有)**: 万相 80.9% → 目标 ≥90%; 豆包 94.5% 不退步; 白图 95.9% / 蓝图 99.3% 不退步。

> ⚠️ **测试集就是训练集的风险**: `D:\probe\wan_local` 和 `D:\probe\localedit_seedream` 是**之前**用不同 seed 造的, 而这次训练数据用 `--seed 201/202` + 不同源图池, 源图重叠有限但**不能保证零重叠**。若要完全干净的数字, 再用**第三个 seed** 另造一小批测试集。

## 发我什么
1 的分组计数 + 3/4 的训练日志尾巴 + 4 的所有 eval/sweep 输出。我做前后对照, 定采用不采用。
