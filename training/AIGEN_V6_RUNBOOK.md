# SSP v6 · 加缩放增广 · 修 v5 的 resize 误杀 · 只重训(PowerShell)

**为什么**: v5 把真 Qwen 修好了(70%漏→0%漏 很好)但引入一个问题 —— **缩放过的真图误杀从 0% 蹦到 13.8%**(median 0.16→0.80)。加 Qwen 后 模型把"缩放/降采样"本身误当假图信号。
**做法**: 加**缩放增广** —— 训练时随机把两类图都降采样 → 教模型"缩放不是假"。修误杀 同时保住 Qwen 的战果。
**省事**: **复用 v5 数据集 `D:\ssp_aigen_v5` 只重训**(改的是增广不是数据)。

## v5 基线(对照 别丢)
| 测试集 | v5 |
|---|---|
| 真 Qwen | 97.3% reject 漏 0%(**要保住**)|
| Flux | 86.3% 漏 0% |
| ostris / taesd / taesdxl / taesd3 / img2img | 93.7 / 90.3 / 78.5 / 94.7 / 94.5(漏≈0)|
| 误杀 原图 / 微信 | 0 / 0.1 |
| **误杀 缩放** | **13.8%(median 0.80)← 要修回≈0** |

---

## 前置
```
cd D:\alipay-platform-classifier
git pull
```

## 1 打补丁(加缩放增广)
```
python training\patch_ssp_repo.py --repo D:\SSP-AI-Generated-Image-Detection-main
findstr /n "v6" D:\SSP-AI-Generated-Image-Detection-main\utils\tdataloader.py
```
- 第二行 `findstr` **应打印出带 v6 缩放增广的那行**(确认加上了)。若为空 = 没打上 → 告诉我。
- 不用还原 .bak(v5 已经把 tdataloader 弄成 v4 态了 这次只在其上加缩放增广)。

## 2 重训 v6(复用 v5 数据 + 缩放增广)(GPU 数小时)
```
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_aigen_v5 --gpu_id 0 --save_path .\snapshot\aigen_v6\ --jpg_prob 0.5 --blur_prob 0.1
```
- 缩放增广硬编码在 data_augment 里 训练自动生效 不用额外参数。产出 `snapshot\aigen_v6\Net_epoch_best.pth`。

---

## 3 验证 v6(重点: 缩放误杀降回≈0 且 Qwen 保住 别的别退步)
```
$v6 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v6\Net_epoch_best.pth"
```

### 3a 缩放误杀(关键 应从 13.8% 降回≈0)
```
cd D:\SSP
python predict_all_models.py --model_root $v6 --input D:\probe\valnat_resize --output_dir D:\probe\valnat_resize_v6 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\valnat_resize_v6\summary.csv --kind genuine
```

### 3b 真 Qwen + Flux 保住(应仍漏≈0)
```
cd D:\SSP
python predict_all_models.py --model_root $v6 --input D:\probe\qwen --output_dir D:\probe\qwen_v6 --device cuda
python predict_all_models.py --model_root $v6 --input D:\probe\flux --output_dir D:\probe\flux_v6 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\qwen_v6\summary.csv --kind fake
python training\eval_summary.py D:\probe\flux_v6\summary.csv --kind fake
```

### 3c 原图 / 微信 误杀别升
```
cd D:\SSP
python predict_all_models.py --model_root $v6 --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_v6 --device cuda
python predict_all_models.py --model_root $v6 --input D:\probe\valnat_wechat --output_dir D:\probe\valnat_wechat_v6 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\valnat_v6\summary.csv --kind genuine
python training\eval_summary.py D:\probe\valnat_wechat_v6\summary.csv --kind genuine
```

### 3d 老生成器别退步
```
cd D:\SSP
python predict_all_models.py --model_root $v6 --input D:\probe\ostris16 --output_dir D:\probe\ostris16_v6 --device cuda
python predict_all_models.py --model_root $v6 --input D:\probe\taesd --output_dir D:\probe\taesd_v6 --device cuda
python predict_all_models.py --model_root $v6 --input D:\probe\taesdxl --output_dir D:\probe\taesdxl_v6 --device cuda
python predict_all_models.py --model_root $v6 --input D:\probe\taesd3 --output_dir D:\probe\taesd3_v6 --device cuda
python predict_all_models.py --model_root $v6 --input D:\probe\i2i_holdout --output_dir D:\probe\i2i_v6 --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\ostris16_v6\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd_v6\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesdxl_v6\summary.csv --kind fake
python training\eval_summary.py D:\probe\taesd3_v6\summary.csv --kind fake
python training\eval_summary.py D:\probe\i2i_v6\summary.csv --kind fake
```

---

## 采用判据
- **缩放误杀降回≈0(3a)且 真 Qwen 保住漏≈0(3b)且 原图/微信误杀不升(3c)且 老生成器不明显退步(3d)** → 采用 v6 换预测模型为 aigen_v6。这版就是可上线的。
- 若缩放误杀降了但 Qwen 掉了 → 缩放增广和 Qwen 有冲突 发我 我调(降增广强度 或 把 Qwen 改成原生分辨率重造去掉降采样耦合)。

## 发我什么
3a(重点)+ 3b + 3c + 3d 各段。我做 v5 对 v6 对照 定采用 + 更新经理口径(Qwen 修好 + 误杀干净 = 完整好消息)。
