# 状态栏 tiny-CNN 训练包（GPU 训练，CPU 上线）

把安卓/苹果分类从“numpy 逻辑回归”升级为“状态栏条上的极小 CNN”。GPU 上用数万级免费自举
标签训练，导出 ONNX 在 CPU 上线。numpy 逻辑回归只是之前无 GPU 环境下的产物。

## 一条铁律（防循环）
标签来自分辨率，所以**模型绝不能看到分辨率**。做法：只喂“状态栏条缩放到固定 512x64”的画布
（`preprocess.py`），绝对分辨率被抹掉。**头号指标是“分辨率对抗测试集”**（缩放过的 iPhone、
非常规安卓宽度）上的准确率，而不是自举分布上的留出准确率。

## 两层上线（PaddleOCR 不进热路径）
- **Tier-0（近乎免费，解决 ~79%）**：读文件头分辨率做自举 + EXIF/翻拍门；不调用模型。
- **Tier-1（只处理 ~21% 弃权 + 抽检）**：`preprocess` -> tiny-CNN（512x64 条）。
  设备判定 = argmax；与分辨率/EXIF 先验不一致 -> `inconsistent`，交欺诈评分。
- **假图/欺诈是独立的融合评分**（勾缺失 + OS-vs-分辨率不一致 + 翻拍 + iPhone分辨率-vs-EXIF 矛盾），
  不是 CNN 的一类，也不是单条“没有勾就是假”。

## 流程
```bash
# 1) 先做数据准备（numpy，任意机器）：自举标注 + 缓存状态栏条（见上层脚本），再：
python training/prepare_dataset.py            # 产出 train/val/predict 清单（防泄漏去重）

# 2) GPU 训练：
pip install -r training/requirements-train.txt
python training/train.py --train training/data/train.jsonl --val training/data/val.jsonl \
    --out training/runs/statusbar_v1 --epochs 20 --batch-size 256 --device cuda

# 3) 门控自训练（可选，3-5 轮）：只对 predict.jsonl（模棱两可）打伪标签，
#    仅当校准置信度>0.95 且与分辨率先验一致才采纳，每轮用金标+对抗集把关，可回滚。

# 4) 导出（默认 FP32；确需再 INT8）：
python training/export.py --checkpoint training/runs/statusbar_v1/best.pt \
    --out training/runs/statusbar_v1/model_fp32.onnx

# 5) CPU 推理（只需 numpy/PIL/onnxruntime）：
python training/infer_cpu.py --onnx training/runs/statusbar_v1/model_fp32.onnx --input <图或目录>
```

## 训练要点
- 类别平衡靠 `WeightedRandomSampler`，**不写死类别权重**；标签平滑 0.05 抵御 ~1% 自举噪声；
  先验(≈60/40)在推理端用 `--prior-logit` 调整，不烘焙进权重（分布会漂移）。
- 温和增强（亮度/对比度/轻微横向缩放），**禁止**强 JPEG/模糊（会抹掉判别用的抗锯齿）。
- 按“平衡准确率 + 对抗集准确率”选 best，不看自举留出准确率。

## 规模化前必做（P0 阻塞项）
- **重新测量整池“翻拍图”占比**：iPhone 翻拍安卓屏会破坏“iPhone 分辨率=iOS”。样本里是 0%，
  大池不会是 0。多信号翻拍检测（EXIF 拍摄字段 / 非截图长宽比 / 摩尔纹·边框）先量化再决定是否信任自举。
- **冻结并扩充 iPhone 分辨率表**（含全部在售机型、灵动岛/Pro Max、横竖两向）。
- **按组切分**：pHash 近重复 + 交易族键（金额+时间+收款人+订单号，来自 mate 的 OCR）+ 设备指纹，
  再加“最近 N 周时间测试集”和“分辨率对抗测试集”。
- **金标扩到 ~2000 张**（用状态栏联系表，约一人周），当前 32 张只够验证简单情形。
