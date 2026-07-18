# GPU 服务器运行手册（数据准备用 CPU，训练用 GPU，上线用 CPU）

全流程：**免费自举标注 -> 防泄漏切分 -> GPU 训练极小 CNN -> 导出 ONNX -> CPU 上线**。
数据准备只需 numpy/PIL，训练需 torch(GPU)，上线需 onnxruntime(CPU)。

> 记号：下例假设服务器是 Windows（D: 盘 / PowerShell），全量图片池在
> `D:\download\raw_images`。Linux 把路径换成正斜杠即可。

---

## 0. 前提
- Python 3.10–3.12；NVIDIA 驱动 + CUDA；一块 GPU（训练用）。
- 全量图片池路径（几十万张）。
- 把 `platform-classifier/` 整个目录拷到服务器（含 `src/ scripts/ training/ gold/ requirements.txt`）。
  不要拷 `runs/`（在服务器重新生成）。

## 1. 装环境
```powershell
cd platform-classifier
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 数据准备 + 上线依赖（轻量）
pip install -r requirements.txt          # numpy, Pillow
pip install -r training\requirements-train.txt   # onnx, onnxruntime（先不装 torch）
# 训练依赖：按 pytorch.org 选匹配 CUDA 的命令，例如：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # 应为 True
```
> 若中文输出乱码：`$env:PYTHONIOENCODING="utf-8"`。

**开跑前先预检**（避免跑几小时才发现环境问题）：
```powershell
python scripts\preflight.py --input D:\download\raw_images
```
必需项应全部 `[OK ]`（torch/onnxruntime 是训练/上线用，数据准备阶段可暂缺）。

## 2. 一次性数据准备（CPU，numpy/PIL）

### 2a. 全量提取“状态栏条 + 元数据 + dHash”（最耗时的一步，**多进程并行**）
10 万+ 图逐张解码很慢，用分片多开进程（每片一个进程，`$N` 取 CPU 核数，8~16）：
```powershell
$N = 16
$procs = 0..($N-1) | ForEach-Object {
  Start-Process python -PassThru -ArgumentList `
    "scripts\inspect_samples.py --input D:\download\raw_images --output runs\pool_full --strip-fraction 0.08 --shard-index $_ --shard-count $N"
}
$procs | Wait-Process          # 等所有分片跑完
# 合并各分片清单为一个 inspect.jsonl（状态栏条已同写 runs\pool_full\status_bar\，文件名唯一无冲突）：
Get-Content runs\pool_full\inspect.shard-*.jsonl | Set-Content runs\pool_full\inspect.jsonl
```
- 产出 `runs\pool_full\inspect.jsonl` + `runs\pool_full\status_bar\*.png`。
- **`--strip-fraction 0.08` 必须与 `training/preprocess.py` 的 `STATUS_STRIP_FRACTION` 一致**，否则训练/上线预处理不一致。
- 损坏图自动跳过并记录；某个分片挂了，单独重跑那个 `--shard-index` 即可。
- 单机小样本可直接不分片：`python scripts\inspect_samples.py --input <目录> --output runs\pool_full --strip-fraction 0.08`。

### 2b. 免费自举标注（分辨率规则，~79% 高精度）
```powershell
python scripts\bootstrap_labels.py --jsonl runs\pool_full\inspect.jsonl `
    --gold gold\gold_seed_v2.json --output runs\pool_full\bootstrap_labels.jsonl
```
产出 `bootstrap_labels.jsonl`（ios/android/abstain）+ 覆盖率 + 金标精度。

### 2c. P0 翻拍门（阻塞项，务必先过）
```powershell
python scripts\photo_scan.py --inspect runs\pool_full\inspect.jsonl `
    --bootstrap runs\pool_full\bootstrap_labels.jsonl --image-root D:\download\raw_images
```
- 看 **“自举已标注里的毒样本率”**——应接近 0（相机照片尺寸不命中手机分辨率，会自动 abstain）。
- 对“判为翻拍”的子集**人工抽查**，并按真实翻拍标定像素噪声阈值（样本里没有真翻拍，标定不了）。

### 2d. 数据体检基线（漂移监控用）
```powershell
python scripts\hygiene_report.py --inspect runs\pool_full\inspect.jsonl
```

### 2e. 防泄漏切分（整组分配 + 时间/对抗留出）
```powershell
python scripts\build_splits.py --inspect runs\pool_full\inspect.jsonl `
    --bootstrap runs\pool_full\bootstrap_labels.jsonl --gold gold\gold_seed_v2.json `
    --crops runs\pool_full\status_bar --out-dir training\data --temporal-holdout 0.15
```
产出 `training\data\{train,val,test_temporal,test_adversarial,predict}.jsonl`；
末尾会打印“跨切分文件=0、train∩temporal 组交集=0”，**必须都为 0**。

## 3. GPU 训练
```powershell
python training\train.py --train training\data\train.jsonl --val training\data\val.jsonl `
    --out training\runs\statusbar_v1 --epochs 20 --batch-size 256 --workers 8 --device cuda
```
- 极小模型（<15 万参数），几十万条也就几分钟一轮。
- 按**平衡准确率**在金标上选 `best.pt`；关注 `test_adversarial` 上的表现（真·头号指标），不看训练准确率。

## 4. 导出 ONNX + CPU 上线
```powershell
# 默认导 FP32（模型很小，INT8 收益有限且可能移动边界；确需再加 --int8）
python training\export.py --checkpoint training\runs\statusbar_v1\best.pt `
    --out training\runs\statusbar_v1\model_fp32.onnx

# CPU 推理（只需 numpy/PIL/onnxruntime，无需 torch）：
python training\infer_cpu.py --onnx training\runs\statusbar_v1\model_fp32.onnx `
    --input D:\download\raw_images --out runs\pool_full\cnn_device.jsonl
```

## 5. 出全量设备标签 + 假图裁决（Tier-0 + Tier-1 合并）
```powershell
python scripts\merge_device_labels.py `
    --bootstrap runs\pool_full\bootstrap_labels.jsonl `
    --cnn runs\pool_full\cnn_device.jsonl `
    --inspect runs\pool_full\inspect.jsonl `
    --out runs\pool_full\final_device.jsonl
    # 有 mate 检测器“有无勾”结果时再加 --checkmark <他的jsonl>，假图信号才完整
```
产出 `final_device.jsonl`：每图 `device`(ios/android/uncertain) + `source` + `confidence` +
`fraud_score`/`fraud_verdict`(pass/review/reject)。Tier-0 非 abstain 直接用分辨率；abstain 用 CNN。

## 6. 注意事项 / 坑
- **预处理一致**：2a 的裁条比例、`preprocess.STATUS_STRIP_FRACTION`、上线端必须都是 0.08。
- **P0 翻拍门先过**再信任自举标签。
- **金标太小**：现在只有 32 张，只够验证简单情形。要信任“难例/假图”指标，需用状态栏联系表
  （`scripts/make_contact_sheet.py`）把金标扩到 ~2000 张（约一人周）。
- **头号指标是 `test_adversarial`**（缩放 iPhone / 非常规安卓），不是训练/自举分布准确率。
- **dHash 只做“几乎一模一样”的去重（阈值 0~1）**；真·交易族分组要用 mate 的 OCR 字段（金额+时间+收款人）。
- **生产是 CPU**：只上 ONNX（默认 FP32）；GPU 只用于训练/批处理。

## 7. 可选增强
- **门控自训练**（脚本已备 `training/self_train.py`，CNN v0 训好后再跑）：
  ```powershell
  python training\self_train.py --train training\data\train.jsonl --val training\data\val.jsonl `
      --predict training\data\predict.jsonl --adversarial training\data\test_adversarial.jsonl `
      --out training\runs\statusbar_selftrain --rounds 4 --device cuda
  ```
  只对模棱两可桶打伪标签，两类均衡录入，每轮在对抗集把关、回退即停回滚。
- 扩金标到 ~2000（`scripts\make_contact_sheet.py` 出联系表批量打标）。
- 假图融合评分已在 `fusion.py`/`merge_device_labels.py`；接入 mate 的“有无勾”后信号才完整。

## 8. 常见问题 / 排错
- **`torch.cuda.is_available()` 为 False**：装了 CPU 版 torch。先卸载 `pip uninstall torch torchvision`，
  再按 pytorch.org 选“对应 CUDA 版本”的命令重装（`nvidia-smi` 看驱动 CUDA 版本）。
- **中文乱码 / `UnicodeEncodeError`**：`$env:PYTHONIOENCODING="utf-8"`。
- **`ModuleNotFoundError: alipay_platform`**：在 `platform-classifier` 目录下跑，且已 `pip install -r requirements.txt`。
- **2a 太慢**：加大 `$N`（到 CPU 核数）；用固态盘；先只跑 1 个分片验证再放量。
- **`build_splits` 报“跨切分/组交集”不为 0**：不应发生；若发生，检查 `inspect.jsonl` 是否有重复文件行（合并分片时重复 `Get-Content`）。
- **训练 val 抖动大**：金标只有 32 张，正常；别据此调参，尽快扩金标。看 `test_adversarial` 才准。
- **导出 ONNX 后精度掉**：只有加了 `--int8` 才会明显；默认 FP32 不会。INT8 必须在 `test_adversarial`+金标上复验。
- **磁盘**：状态栏条 PNG 很小（每张几 KB），10 万张约几百 MB；`runs\` 可放大盘。
