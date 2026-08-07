# 自动拒绝阈值标定 · 大样本真图误杀 · 一步步

## 为什么做这个(经理改了目标)
经理最新口径: **"就是要做到自动的 · 误杀要尽可能低"** / **"主要任务就是用AI 提高识别率 减少人工"**。
之前我们按 **review-only**(只标记交人工)设计, 那时只要**漏检低**就行;
**现在要自动拒 → "高置信硬拦率"才是能减人工的那个数, 而误杀必须压到能自动执行的水平。**

**问题**: 现在的误杀是在 **2700 张**真图上量的 → 最细只能分辨到 **约 0.04%(1 张)**。
自动拒一般要**万分之一**量级的把握, 2700 张根本量不出来。
而且真图分数尾巴已经摸到 **0.992**, 和假图分布是重叠的 —— **安全线只能实测, 不能靠外推**。

**所以**: 用 **2-3 万张**真图(图库有 13.8 万+)重新标定, 找出"高于这个分数就可以直接拒"的线。

---

## 1 抽一大批真图(只用真截图, 排除相机/翻拍)
```
cd D:\alipay-platform-classifier
python training\sample_genuine_pool.py --roots D:\download\TempFakeImages D:\download2\TempFakeImages --out D:\probe\genuine_20k --n 20000 --seed 4242
```
**注意**: 要和训练用的真图**尽量不重叠**(seed 与训练不同; 训练 nature 用的是 seed 0 那批)。
完全零重叠做不到(同一个池子), 但 2 万张里大部分是训练没见过的, 足够看尾巴形状。

## 2 用 v7 打分(线路A)
```
cd D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth --input D:\probe\genuine_20k --output_dir D:\probe\genuine_20k_v7 --device cuda
```
(2 万张比较慢, 挂着跑。)

## 3 算"自动拒"能到什么水平
```
cd D:\alipay-platform-classifier
python training\autoreject_threshold.py --genuine D:\probe\genuine_20k_v7\summary.csv --fake D:\probe\wan_full_v7\summary.csv D:\probe\qwen_v7\summary.csv
```
输出会给出: 在若干个"误杀上限"(1/1000, 1/5000, 1/10000, 0)下, **对应的分数线** 和 **各假图集在该线上的自动拒比例**。

---

## 怎么给经理解释
拿到结果后, 结论会长这样:
> "分数高于 **X** 就直接拒, 两万张真图里只有 **N** 张会被误拒(约万分之几);
> 这条线能自动处理掉 **Y%** 的假图, 剩下的进人工复核。"

**如果 Y 很高(比如 >80%)** → 人工量能大幅下降, 正是经理要的。
**如果为了压误杀导致 Y 很低** → 说明现在的模型还不足以支撑全自动, 要么接受一部分人工, 要么继续提升模型。这也要如实说。

## 注意
- 线路B(局部改金额)同理要单独标一次, 但**先做线路A** —— 它是主力且样本更充足。
- 真图池子里可能**本来就混着假图**(我们查过, 有几张在每次测试里都打高分)。若发现分数线卡在少数几张上, **要人工看一眼那几张是不是真的真图** —— 如果它们其实是漏网的假图, 真实误杀比测出来的还低。
