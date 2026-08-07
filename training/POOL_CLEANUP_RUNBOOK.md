# 用两条线交叉验证批量清理真图池 · 一步步

## 为什么做
标定自动拒阈值时发现: **"真图"池里混着不少真的假图**(2 万张里分数最高的 20 张, **14 张查实是假的**, 其中 11 张右下角直接印着「豆包AI生成」)。
影响有两层:
1. **我们所有的误杀数字都被高估了** —— 模型判对了, 是标注错了;
2. **训练集里也可能混着同样的假图**(nature 类里混进假图 = 教模型"这种假图是真的"), 这会**直接压低模型上限**。

**但逐张人工看两万张不现实。** 所以用一条实测有效的规则批量筛:

> **两条线(整图AI 的 v7 + 局部改金额 的 localdet3)同时打高分的图 = 高置信假图。**

依据: 这是**两个独立训练、架构和训练数据都不同**的模型; 模型各自的癖好不会跨模型复现。
实测两条线各自的榜首里**有 6 张是同一批文件**, 而人工抽查确认那批 14/20 是假的。

---

## 1 交叉筛(秒级, 只读已有的两个 summary)
```
cd D:\alipay-platform-classifier
git pull
python training\cross_flag.py --a D:\probe\genuine_20k_v7\summary.csv --b D:\probe\genuine_20k_ld3\summary.csv --out D:\probe\cross_flagged.txt
```
**先看它打印的"巧合基准"那一行**:
- 实际交叉命中数 **远大于**(比如 >3 倍)巧合基准 → 规则成立, 这批可以当高置信假图;
- 没明显超过 → 交叉命中只是碰巧, 别批量处理, 老实人工抽查。

## 2 抽查验证规则(必做, 5 分钟)
把交叉命中的图拷出来看几张:
```
New-Item -ItemType Directory -Force D:\probe\cross_check | Out-Null
Get-Content D:\probe\cross_flagged.txt | Select-Object -First 12 | ForEach-Object {
  $f = Get-ChildItem D:\probe\genuine_20k -Filter $_ -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($f) { Copy-Item $f.FullName -Destination D:\probe\cross_check }
}
explorer D:\probe\cross_check
```
**看两个角**(水印位置不固定, 我们踩过这个坑): 右下角的「豆包AI生成」, 左上角的「AI生成」。
再看金额是不是 **N99.9X**(299.99 / 999.95 / 499.98 这种)。
**12 张里多数能确认是假 → 规则可靠, 继续第 3 步; 多数看着是真图 → 停下告诉我, 阈值要调。**

## 3 用清理后的池子重新标定自动拒
```
python training\autoreject_threshold.py --genuine D:\probe\genuine_20k_v7\summary.csv --fake D:\probe\wan_full_v7\summary.csv D:\probe\qwen_v7\summary.csv --exclude D:\probe\cross_flagged.txt
python training\autoreject_threshold.py --col tile_top3 --require-located --genuine D:\probe\genuine_20k_ld3\summary.csv --fake D:\probe\wan_ld3\summary.csv D:\probe\seed_ld3\summary.csv D:\probe\white_ld3\summary.csv D:\probe\blue_ld3\summary.csv --exclude D:\probe\cross_flagged.txt
```
两条线的严预算(1/5000、1/10000)覆盖率应该会明显上去 —— 因为卡住阈值的本来就是这些假图。

---

## 4(更大的收益)把训练集也清一遍
训练集的 nature 类是从同一个池子抽的 → **大概率也混着这种假图**。
混进去的后果: 模型被教"带豆包水印的图是真图", 直接压低上限。

```
:: 给训练集的 nature 打分(两条线)
cd D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v7\Net_epoch_best.pth --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\train\nature --output_dir D:\probe\trainnat_v7 --device cuda
cd D:\alipay-platform-classifier
python training\predict_tiled.py --ssp-repo D:\SSP --model D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet3\Net_epoch_best.pth --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\train\nature --output_dir D:\probe\trainnat_ld3 --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda

:: 交叉筛出训练集里的假图
python training\cross_flag.py --a D:\probe\trainnat_v7\summary.csv --b D:\probe\trainnat_ld3\summary.csv --out D:\probe\trainnat_bad.txt
```
抽查几张确认后, 把这些文件从 `train\nature` 挪走(**别删, 挪到别处留证**), 然后重训 v8 / localdet4。
**预期: 训练标签变干净 → 模型上限提高 → 严预算下的自动拒覆盖率进一步上升。**

## 发我什么
第 1 步的完整输出(尤其巧合基准那行)+ 第 2 步抽查的结论(12 张里几张是假的)。
我据此判断规则可不可靠, 再决定要不要走第 4 步的重训。
