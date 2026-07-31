# SSP v4b · 抗压缩硬化 · 只重训 不重造数据(PowerShell)

**目标**: 修 v4 的压缩弱点(7d: 压缩过的假图 reject 只有 6.3% 漏检 4.7%)。
**做法**: JPEG 增广加"一半概率再压一次"(双压) 让模型见过微信多次转发那种重压缩的样子 学会压缩后仍认指纹。
**省事点**: **复用 v4 的数据集 `D:\ssp_aigen_v4` 不用重造** —— 只改增广重训。

---

## v4 基线(对照 别丢)
| 测试集 | v4 召回/误杀 |
|---|---|
| ostris16 干净 | 95.0% reject / 0% miss |
| taesd / taesdxl 干净 | 92.7% / 93.3% |
| img2img held-out | 88.1% / 0% miss |
| 误杀 原图/微信/缩放 | 0.1% / 0.1% / 0% |
| **压缩过的 ostris 假图(7d)** | **reject 6.3% / 漏 4.7%** ← 要提升的 |

---

## 1. 拉最新 + 打补丁(含新的双压增广)
```
cd D:\alipay-platform-classifier
git pull
python training\patch_ssp_repo.py --repo D:\SSP-AI-Generated-Image-Detection-main
```
- 新规则会给 `data_augment` 的 JPEG 加"一半概率再压一次"。打印里应看到 `已改 utils/tdataloader.py`(或已达标)。

## 2. 重训(复用 v4 数据 + 更强压缩增广)(GPU 数小时)
```
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_aigen_v4 --gpu_id 0 --save_path .\snapshot\aigen_v4b\ --jpg_prob 0.6 --blur_prob 0.1
```
- `--jpg_prob 0.5→0.6` 更多样本走压缩增广 双压逻辑已在补丁里。
- 产出 `snapshot\aigen_v4b\Net_epoch_best.pth`。

---

## 3. 验证 v4b 对比 v4(重点看压缩假图有没有提升 别的别退步)
```
$v4b = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v4b\Net_epoch_best.pth"
```

### 3a 压缩过的假图召回(关键 应比 v4 的 6.3%/漏4.7% 明显好)
先造压缩版假图(三种解码器都压一下):
```
cd D:\alipay-platform-classifier
python training\recompress_dir.py --src D:\probe\ostris16 --out D:\probe\ostris16_wechat --q 65 --double
python training\recompress_dir.py --src D:\probe\taesd --out D:\probe\taesd_wechat --q 65 --double
python training\recompress_dir.py --src D:\probe\taesdxl --out D:\probe\taesdxl_wechat --q 65 --double
cd D:\SSP
python predict_all_models.py --model_root $v4b --input D:\probe\ostris16_wechat --output_dir D:\probe\ostris16_wechat_v4b --device cuda
python predict_all_models.py --model_root $v4b --input D:\probe\taesd_wechat --output_dir D:\probe\taesd_wechat_v4b --device cuda
python predict_all_models.py --model_root $v4b --input D:\probe\taesdxl_wechat --output_dir D:\probe\taesdxl_wechat_v4b --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\ostris16_wechat_v4b\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd_wechat_v4b\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesdxl_wechat_v4b\summary.csv --kind fake
```

### 3b 干净假图召回别退步(复用探针 对比 v4 的 95 / 92.7 / 93.3)
```
cd D:\SSP
python predict_all_models.py --model_root $v4b --input D:\probe\ostris16 --output_dir D:\probe\ostris16_v4b --device cuda
python predict_all_models.py --model_root $v4b --input D:\probe\taesd --output_dir D:\probe\taesd_v4b --device cuda
python predict_all_models.py --model_root $v4b --input D:\probe\taesdxl --output_dir D:\probe\taesdxl_v4b --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\ostris16_v4b\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd_v4b\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesdxl_v4b\summary.csv --kind fake
```

### 3c 误杀别升(对比 v4 的 0.1 / 0.1 / 0)
```
cd D:\SSP
python predict_all_models.py --model_root $v4b --input D:\ssp_aigen_v4\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_v4b --device cuda
python predict_all_models.py --model_root $v4b --input D:\probe\valnat_wechat --output_dir D:\probe\valnat_wechat_v4b --device cuda
python predict_all_models.py --model_root $v4b --input D:\probe\valnat_resize --output_dir D:\probe\valnat_resize_v4b --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\valnat_v4b\summary.csv --kind genuine
python training\eval_summary.py D:\probe\valnat_wechat_v4b\summary.csv --kind genuine
python training\eval_summary.py D:\probe\valnat_resize_v4b\summary.csv --kind genuine
```

### 3d img2img 别退步(对比 v4 的 88.1%)
```
cd D:\SSP
python predict_all_models.py --model_root $v4b --input D:\probe\i2i_holdout --output_dir D:\probe\i2i_v4b --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\i2i_v4b\summary.csv --kind fake
```

---

## 采用判据
- **3a 压缩假图召回明显提升**(漏检降下来 reject 升上去)**且 3b 3c 3d 不退步**(尤其 3c 误杀不升)→ 采用 v4b。
- 若压缩提升了但干净召回或误杀退步 → 发我 我调 jpg_prob 或双压概率再来一版。
- 诚实预期: 压缩能提升但压得特别狠时高频指纹真会被抹掉 属于这类检测器的硬限 尽量提不追求满分。

## 发我什么
3a 三段 + 3b 三段 + 3d 一段 召回 + 3c 三段误杀。我做 v4 对 v4b 的对照 定采用不采用 + 更新经理口径。
