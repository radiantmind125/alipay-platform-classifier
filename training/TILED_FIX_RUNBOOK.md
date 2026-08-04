# 分块打分 · 试补"只改一小块"的盲区 · 不重训(PowerShell)

## 背景(已实测坐实的问题)
| 批次 | 同 VAE 同源图 只差改动面积 | median | 拦截 | **漏检** |
|---|---|---|---|---|
| 只改金额区(~1% 像素) | | 0.002 | 0.0% | **100%** ❌ |
| 整图重绘(100% 像素) | | 0.996 | 98.7% | 0% ✅ |

**根因**: SSP 在**整张图**里随机取 64 个 32x32 小块, 只留**纹理最丰富的一个**。收据上最丰富的通常是底部彩色缩略图, 金额那块几乎选不中 → 改动看不见。

## 这个方案
把图切成 3x6 共 18 块(带 15% 重叠), **每块单独打分**, 取**最高分**当整图分数。
块内选"最富纹理块"只能在这块里选 → 改动所在的块就有机会被看见。**复用现有 v6 不用重训。**

**代价/风险**: 打分机会变多 → **误杀可能上升**。按经理"绝对不能误杀"的红线, **必须同时在真图上测**, 通不过就不能用。

---

## 前置
```
cd D:\alipay-platform-classifier
git pull
$v6 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v6\Net_epoch_best.pth"
```

## 1 局部改金额假图(关键: 现在能不能抓到?)
```
python training\predict_tiled.py --ssp-repo D:\SSP --model $v6 --input D:\probe\localedit --output_dir D:\probe\localedit_tiled --device cuda
python training\eval_summary.py D:\probe\localedit_tiled\summary.csv --kind fake
```
对照: 原来整图打分是 **100% 漏检**。这里漏检降下来才算有效。

## 2 真图误杀(红线: 必须仍然极低)
```
python training\predict_tiled.py --ssp-repo D:\SSP --model $v6 --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --output_dir D:\probe\valnat_tiled --device cuda
python training\eval_summary.py D:\probe\valnat_tiled\summary.csv --kind genuine
```
对照: 整图打分时误杀 0.1%。**这里明显变高就不能用**(或要提高阈值)。

## 3 压缩/缩放真图也别误杀(同红线)
```
python training\predict_tiled.py --ssp-repo D:\SSP --model $v6 --input D:\probe\valnat_wechat --output_dir D:\probe\valnat_wechat_tiled --device cuda
python training\predict_tiled.py --ssp-repo D:\SSP --model $v6 --input D:\probe\valnat_resize --output_dir D:\probe\valnat_resize_tiled --device cuda
python training\eval_summary.py D:\probe\valnat_wechat_tiled\summary.csv --kind genuine
python training\eval_summary.py D:\probe\valnat_resize_tiled\summary.csv --kind genuine
```

## 4 整图AI假图别退步(原来 0 漏检, 分块后应该更好或持平)
```
python training\predict_tiled.py --ssp-repo D:\SSP --model $v6 --input D:\probe\qwen --output_dir D:\probe\qwen_tiled --device cuda
python training\eval_summary.py D:\probe\qwen_tiled\summary.csv --kind fake
```

---

## 判读
- **局部漏检大降 + 真图误杀仍≈0** → 方案成立, 盲区用推理期改动就补上了(不用重训), 直接进部署。
- **局部漏检降了但真图误杀上去了** → 有信号但阈值要重调。CSV 里存了 `tile_max/tile_top3/tile_mean` 三种聚合, 发我数据我来定阈值/聚合方式(可能用 top3 更稳)。
- **局部漏检还是很高** → 分块也救不了(说明小块里的指纹本身太弱)→ 只能上专门的局部篡改检测(Head B 线), 我据此规划。

## 发我什么
1-4 每段 `eval_summary` 输出。我做对照结论 + 定阈值 + 决定是否进部署 + 给经理的口径。
