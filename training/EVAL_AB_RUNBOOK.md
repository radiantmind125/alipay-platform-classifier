# SSP 评估 (a) 真图特异性 + (b) 真造假召回 · 服务器一步一步手册

目标:把之前"50 真图 / 13 假图"的小样本,换成**更大、且无泄漏**的两个数:
- **(a) 特异性**:真图里有多少被正确 Pass(= 1 − 误杀率)。经理最在意误杀。
- **(b) 召回**:真翻拍里有多少被正确 Reject。

> 关键:两处泄漏必须避开,否则数字会骗人。
> 1. 训练的 `nature` 是从两个图库按 seed 抽的约 1 万张真图 → "随手取真图"会撞上训练集,Pass 率虚高。
> 2. 13 张 gold 已进训练 ai 类 → 拿它们报召回是"训练集成绩",乐观。真正 held-out 的是我挖的 40 张。
>
> 下面的 `sample_eval_sets.py` 用**磁盘上真实落地的训练 nature 文件名**当排除集,两个坑一起挡掉。

命令按 **cmd**(不是 PowerShell)写,单行可直接复制。真图拷贝全交给 python 脚本,不用 PowerShell。

---

## 0. 前置(1 分钟)

```
cd /d D:\alipay-platform-classifier
git pull
```
拿到最新的 `training\sample_eval_sets.py`、`training\eval_summary.py`、`training\data\recapture_eval_ext.txt`。

确认三样在:
- 模型:`D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth`
- 训练数据集根:`D:\ssp_alipay\imagenet_ai_0419_sdv4`(里面有 train/val 的 nature、ai)
- 预测脚本:`D:\SSP\predict_all_models.py`(带 networks\ 目录,从 D:\SSP 里跑)

> 标签约定:模型是 nature=1 / ai=0 训的 → predict 默认 `--ai_label 0` 正好,**不用加任何 flag**。
> 设备:有 GPU 就 `--device cuda`(推理快很多;没 GPU 会自动切 CPU,只是慢)。

---

## A0. 先拿一个"零成本、干净"的特异性数(2 分钟)

`val\nature`(约 1500 张)是训练时切出来、**没参与反传**的真图,已经在磁盘上、已统一到 768/q90。
直接拿它当第一版特异性,不用拷贝:

```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input D:\ssp_alipay\imagenet_ai_0419_sdv4\val\nature --output_dir D:\ssp_test\valnat_out --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\ssp_test\valnat_out\summary.csv --kind genuine
```
看 `[真图特异性] Pass 率`。这是 held-out(只被用于选最优轮次)、同分布(768/q90)的特异性 —— 干净但偏"实验室"。
真正贴近线上的、原生分辨率的大样本在下面 A2。

---

## A1. 造两个无泄漏测试集(3-5 分钟,一条命令同时出 a 和 b)

```
cd /d D:\alipay-platform-classifier
python training\sample_eval_sets.py --dataset-root D:\ssp_alipay\imagenet_ai_0419_sdv4 --genuine-roots D:\download2\TempFakeImages --n-genuine 1200 --genuine-out D:\ssp_test\gen_clean --recap-list training\data\recapture_eval_ext.txt --recap-src-root D:\download\TempFakeImages --realfake-out D:\ssp_test\realfake2
```
它会打印:
- `训练 nature 已用真图(stem)数` —— 应是上万(说明排除集读到了)。
- `(b) 真造假测试集 ... 拷 N 张(泄漏跳过 x, 找不到 y)` —— N 应接近 40。
- `(a) 无泄漏真图测试集 ... 拷 1200 张(训练已用跳过 ..., 翻拍跳过 ..., 非干净截图跳过 ...)`。

> 想测白图(账单详情)以外也覆盖蓝图,把 `--genuine-roots` 换成
> `D:\download2\TempFakeImages D:\download\TempFakeImages`(两个都给)。

---

## A2. (a) 原生大样本真图特异性(GPU 几分钟 / CPU 20-40 分钟)

```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input D:\ssp_test\gen_clean --output_dir D:\ssp_test\gen_clean_out --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\ssp_test\gen_clean_out\summary.csv --kind genuine
```
看 `[真图特异性] Pass 率` 和 `误杀(Reject)率`。**这是要报给经理的特异性主数**(原生分辨率、无泄漏、1200 张)。

---

## B. (b) 真造假召回(held-out 40 张真翻拍,1-2 分钟)

```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input D:\ssp_test\realfake2 --output_dir D:\ssp_test\realfake2_out --device cuda
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\ssp_test\realfake2_out\summary.csv --kind fake
```
看 `[假图召回] Reject 率`。**这是真正 held-out 的召回主数**(比之前 13 张可信,且不含训练样本)。

---

## 结果怎么读 / 怎么报(重要)

- 报三个数:**A2 特异性(真图 Pass 率)**、**B 召回(真翻拍 Reject 率)**、外加 A0 做旁证。
- 之前的"49/50 Pass、13/13 Reject"里,13 张有一半进过训练 → 别再单独引用那个召回;用 B。
- **诚实边界一句话**:这套只覆盖"整图渲染真假 / 翻拍";改金额、改负号那种局部篡改(经理 03/04)
  SSP 结构上看不到,归 Head B。评估集里的真造假目前**全是翻拍**一种类型,AI 生成/局部篡改暂无真标签。
- 若特异性偏低(误杀高):多半是原生分辨率与训练 768 的尺度差 → 可在 predict 前把测试图也统一 768 再看,
  或后续训练加原生分辨率增广。先把数记下来再调。

---

## 排错

- **`没有找到模型`**:`--model_root` 要指到 `Net_epoch_best.pth` 这个文件本身(不是目录)。
- **`未检测到 CUDA,自动切换到 CPU`**:正常,只是慢;想快就确认 `torch.cuda.is_available()` 为 True。
- **(b) 拷 0 张 / 找不到很多**:`--recap-src-root` 指错了。40 张翻拍在 `D:\download\TempFakeImages`(蓝图)。
- **(a) 训练已用跳过 0**:`--dataset-root` 指错,没读到 nature → 排除失效。确认路径到 `imagenet_ai_0419_sdv4`。
- **中文乱码**:脚本已 reconfigure UTF-8;若控制台仍乱码不影响数字,看英文/数字部分即可。
- 每张图 predict 取 16 个随机 patch 取平均,分数每次跑会有极小抖动,属正常。
