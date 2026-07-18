# 支付宝截图设备分类器（安卓 / 苹果）+ 假图融合评分

把支付宝**转账成功**截图判为 **`android` / `ios`**（不确定时 `uncertain`），并给出**假图融合评分**。
两个核心约束：**提高识别率** + **减少 CPU**（生产是 CPU 服务器，GPU 只用于训练）。

## 方法（两层，服务端 GPU 训练 / 生产 CPU 部署）
- **判别信号 = 系统画的状态栏**。App 头部（勾 + “转账成功” + 金额）是 App 内置字体自绘，两个平台
  一样，不可判别（经理的“勾对齐”判据实测不成立）。安卓状态栏“吵”（网速/双卡/运营商/电量%/一排图标），
  iOS“干净”（定位箭头/电池胶囊/灵动岛）。
- **Tier-0（免费，~79%）**：分辨率规则自举——iPhone 精确分辨率→苹果；短边∈{720,1080,1440}→安卓；
  其余→abstain。人工核对 12/12 精确；毒样本率 0（相机照片尺寸不命中手机分辨率，自动 abstain）。
- **Tier-1（~21% 模棱两可）**：状态栏条上的**极小 CNN**（固定 512×64，抹掉分辨率，防“循环学习”）。
  服务端 GPU 训练，导出 ONNX 在 CPU 上跑。
- **假图/欺诈 = 独立融合评分**（勾缺失 + 设备-vs-分辨率不一致 + 翻拍 + iPhone分辨率-vs-EXIF 矛盾），
  非 CNN 的类别，中间带交人工。

## 目录与职责
```
src/alipay_platform/            # 纯 numpy/PIL 核心（可在任意机器跑/测）
  platform_labels.py   固定标签 + 权重类别守卫
  metadata_seed.py     冻结的 iPhone 分辨率表 + 零解码元数据标注函数
  bootstrap.py         分辨率两侧规则（iPhone→ios / 720·1080·1440→android）
  regions.py           几何裁剪（状态栏条 / 时钟 / 对勾）
  strip_features.py    状态栏条的 20 维便宜特征（给 numpy 逻辑回归基线）
  photo_detector.py    翻拍图检测（EXIF + 尺寸 + 平坦区噪声）——P0 门
  hashing.py           dHash（去重）
  grouping.py          并查集 + LSH 去重 + 文件名时间戳解析（防泄漏切分）
  labeling.py          弃权式投票聚合（银标）
  fusion.py            设备合并(Tier0+Tier1) + 假图融合评分
scripts/                        # 数据准备 / 分析（numpy/PIL）
  inspect_samples.py   全量提取状态栏条 + 元数据 + dHash
  bootstrap_labels.py  出自举标签 + 金标精度
  photo_scan.py        P0 翻拍门：量化毒样本率
  hygiene_report.py    数据体检基线
  build_splits.py      防泄漏整组切分 + 时间/对抗留出
  train_strip_classifier.py  numpy 逻辑回归基线（无 GPU 时用）
  merge_device_labels.py     合并出每图最终设备标签 + 假图裁决 -> final_device.jsonl
  make_contact_sheet.py / header_sheet.py / measure_check_alignment.py  # 打标/核对/验证工具
training/                       # GPU 训练包（需 torch）+ CPU 上线（onnxruntime）
  preprocess.py        训练/上线共用预处理（原图->状态栏条->512×64->归一化）
  model.py             StatusBarNet（深度可分离，<15 万参数）
  dataset.py           条数据集 + 温和增强
  prepare_dataset.py / (build_splits) 产出 train/val/predict 清单
  train.py             GPU 训练（均衡采样 + 标签平滑 + 金标选 best）
  self_train.py        门控自训练（伪标签，金标/对抗集把关，可回滚）
  export.py            导出 ONNX（默认 FP32，可选 INT8）
  infer_cpu.py         CPU 推理（onnxruntime，先验调整 + 一致性交叉校验）
gold/                           # 人工金标（不可变，只进 val/test）
RUNBOOK.md                      # 服务器全流程运行手册
```

## 怎么跑
- 单元测试（numpy/PIL，无需 torch）：`python run_tests.py`（68 通过）。
- 全流程（服务器）：见 **RUNBOOK.md**。
- 协作分工：见 **COORDINATION.md**。

## 一条铁律
标签来自分辨率，所以模型**绝不能看到分辨率**——只喂固定 512×64 的状态栏条。头号指标是
**对抗测试集**（缩放 iPhone / 非常规安卓）的准确率，不是自举分布上的准确率。

## 现状
数据准备、自举标注、P0 翻拍门、防泄漏切分、numpy 基线、GPU 训练包、门控自训练、ONNX 导出、
CPU 推理、设备合并、假图融合——**均已实现并通过单元测试**（GPU 训练本身在服务器上跑）。
待办：在服务器按 RUNBOOK 训 CNN；金标扩到 ~2000（联系表）；接入 mate 的 OCR 字段与“有无勾”。
