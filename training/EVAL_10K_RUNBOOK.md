# 10K 真图特异性 + 运行时间 + 误杀核查 · 服务器一步一步

目标三件事:
1. 用 **1 万张**无泄漏真图测特异性(比 1200 更可信的误杀率)。
2. 知道**运行时间**(取决于 predict 是否真用 GPU)。
3. 把被判"假"的**误杀图**挑出来逐张看(区分真误杀 / 图库里混进来没标注的真假图)。

命令按 **cmd** 写,单行可直接复制。每步都写了「你应看到 / 耗时 / 出错怎么办 / 发我什么」。

---

## 步骤 0 · 准备(1 分钟)

```
cd /d D:\alipay-platform-classifier
git pull
```
**你应看到**:拉到 `training\collect_flagged.py` 等最新脚本。

四个路径确认在:
- 模型 `D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth`
- 数据集根 `D:\ssp_alipay\imagenet_ai_0419_sdv4`
- 预测脚本 `D:\SSP\predict_all_models.py`
- 真图库 `D:\download2\TempFakeImages`(白图)、`D:\download\TempFakeImages`(蓝图)

---

## 步骤 1 · 先看已有的误杀(不用重跑,10 秒)

你手上已经有 1200(A2)和 1500(A0)的结果。直接把被判假的图拷出来:
```
python training\collect_flagged.py D:\ssp_test\gen_clean_out\summary.csv --out D:\ssp_test\fp_A2
python training\collect_flagged.py D:\ssp_test\valnat_out\summary.csv   --out D:\ssp_test\fp_A0
```
**你应看到**:各「命中 1 张 … 拷出 1 张」。`D:\ssp_test\fp_A2`、`fp_A0` 里各有 1 张图,名字像 `1.000_xxxx.png`(分数前缀)。

**发我什么**:打开这两个文件夹,把里面的图**贴给我**。我逐张判:真误杀,还是图库里混进来没标注的真假图。

**出错怎么办**:若报「找不到 summary.csv」,说明那两次 predict 的结果没留;跳过这步,等步骤 6 在 10K 上看。

---

## 步骤 2 · 查设备:predict 到底用不用 GPU(30 秒)

```
python -c "import torch; print('CUDA', torch.cuda.is_available())"
```
- 打印 **`CUDA True`** → GPU 路径,10K 大约 **20-40 分钟**。
- 打印 **`CUDA False`** → CPU 路径,10K 大约 **1-2 小时**(还是能跑,只是慢)。

**出错怎么办**:若「python 找不到」,用你平时跑 predict 的那个 python(同一个环境)。

---

## 步骤 3 · 造 1 万张无泄漏真图(约 5-10 分钟)

```
python training\sample_eval_sets.py --dataset-root D:\ssp_alipay\imagenet_ai_0419_sdv4 --genuine-roots D:\download2\TempFakeImages --n-genuine 10000 --genuine-out D:\ssp_test\gen10k --recap-list training\data\recapture_eval_ext.txt --recap-src-root D:\download\TempFakeImages --realfake-out D:\ssp_test\realfake2 --force
```
**你应看到**三行关键输出:
- `训练 nature 已用真图(stem)数: N` —— N 应约 1 万。**记下**。
- `(b) 真造假测试集 … 拷 40 张`。
- `(a) 无泄漏真图测试集 … 拷 10000 张(训练已用跳过 X, …)` —— X 应 > 0。**记下**。

> 这两个数(N 约一万、X>0)就是"排除闸真的把训练图剔掉了"的书面证据,顺手也交了差。
> `--force` 是因为 `realfake2` 已存在(只是把那 40 张重拷一遍,无妨)。

**耗时**:拷 1 万张原生图,约 5-10 分钟。

**出错怎么办**:
- `只凑到 x/10000 … 已中止` → 白图不够,蓝图也加进来:把 `--genuine-roots D:\download2\TempFakeImages` 换成 `--genuine-roots D:\download2\TempFakeImages D:\download\TempFakeImages`,重跑。
- `排除集为空 … 已中止` → `--dataset-root` 指错了,确认到 `imagenet_ai_0419_sdv4`。

---

## 步骤 4 · 预测 1 万张(GPU 20-40 分 / CPU 1-2 时)

```
cd /d D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input D:\ssp_test\gen10k --output_dir D:\ssp_test\gen10k_out --device cuda
```
**怎么估时间**:它会一行行打印 `[i/10000] 文件名 … ai_score=… 决策`。开跑后**看 1 分钟**,记下 i,`ETA ≈ 10000 ÷ (这一分钟的 i) 分钟`。太慢(CPU)可以 Ctrl-C,改小 `--n-genuine` 重来,或放后台/过夜跑。

**想要精确总时间**(PowerShell 里跑):
```
Measure-Command { python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input D:\ssp_test\gen10k --output_dir D:\ssp_test\gen10k_out --device cuda }
```
跑完打印 `TotalMinutes`。

**你应看到**末尾三行:`图片数量: 10000`、`模型数量: 1`、`汇总结果: …\gen10k_out\summary.csv`。这三行对了,summary.csv 才存在。

**发我什么**:你实测的耗时(1 分钟 i 值,或 Measure-Command 的 TotalMinutes)。

---

## 步骤 5 · 算特异性(1 秒)

```
cd /d D:\alipay-platform-classifier
python training\eval_summary.py D:\ssp_test\gen10k_out\summary.csv --kind genuine
```
**你应看到**:`[真图特异性] Pass 率=99.x%  |  误杀(Reject)率=0.x%`。

**发我什么**:整段输出(文件/张数/ai_score/决策/按阈值/特异性 那几行)。

**出错怎么办**:`FileNotFoundError` → 步骤 4 的 predict 没写出 summary.csv,回看步骤 4 末尾三行。

---

## 步骤 6 · 把 1 万张里的误杀图挑出来核查

```
python training\collect_flagged.py D:\ssp_test\gen10k_out\summary.csv --out D:\ssp_test\gen10k_flagged
```
**你应看到**:`命中 N 张 … 拷出 N 张 -> D:\ssp_test\gen10k_flagged`(N 约十来张)。文件夹里图按分数命名,还有一份 `flagged_list.csv`。

**发我什么**:`D:\ssp_test\gen10k_flagged` 里的图**贴给我**(或先发 `flagged_list.csv`)。我逐张判真误杀 / 混进来的真假图。

**想连边缘复核区一起看**(0.60-0.90):
```
python training\collect_flagged.py D:\ssp_test\gen10k_out\summary.csv --out D:\ssp_test\gen10k_flagged --min 0.60 --force
```

---

## 最后 · 一次性发我这些

1. 步骤 1 的 `fp_A2` / `fp_A0` 两张图。
2. 步骤 3 的 `N`(已用真图数)和 `X`(训练已用跳过)。
3. 步骤 4 的实测耗时。
4. 步骤 5 的特异性整段。
5. 步骤 6 的 `gen10k_flagged` 里的图(或 flagged_list.csv)。

我据此:核对每张误杀是真误杀还是白捡的真假图 → 修正真实误杀率 + 把混进来的真假图加进召回集 → 写经理版更新。
