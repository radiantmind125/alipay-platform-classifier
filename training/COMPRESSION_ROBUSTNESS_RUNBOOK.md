# 线路B 抗压缩测试 · 我们报的 95% 在被压缩过的图上还成不成立?

## 为什么必须测(这是目前最可能推翻现有数字的一件事)
- 线路B 现在所有数字(白图 95.9% / 蓝图 99.3% / 真Seedream 94.5%)**都是在没压缩的干净图上测的**。
- 但这行的图**天天在被转发和重新压缩**。
- 而且我们**已经知道线路A 栽在这上面**: v6 对整图AI 干净时 95% 硬拦, 微信双压后掉到 **6.3%**(高频指纹被压掉了)。
- **线路B 读的是同一类高频指纹, 而且只从一小块金额区里读 —— 信号更少, 理论上更脆。**
→ 所以: 如果线路B 也塌, 那"95%"只对原始上传成立, 必须如实修正口径。

## 怎么测(两个问题都要答)
1. **压缩过的假图还抓得到吗?**(召回会不会塌)
2. **压缩过的真图会不会变成误杀?**(阈值是按未压缩真图标定的, 压缩会不会把真图分数顶上去)

**关键: 用固定阈值 0.9330 评估, 不重新标定。** 上线阈值是定死的; 重新标定会把真实退化掩盖掉。

---

## 前置
```
cd D:\alipay-platform-classifier
git pull
$ld2 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\localdet\Net_epoch_best.pth"
```

## 1 造压缩版(两档: 轻度 + 微信式重压)
```
:: 轻度(单次 q80)
python training\recompress_dir.py --src D:\probe\localedit_heldout   --out D:\probe\cz_white_light  --q 80
python training\recompress_dir.py --src D:\probe\localedit_blue      --out D:\probe\cz_blue_light   --q 80
python training\recompress_dir.py --src D:\probe\localedit_seedream  --out D:\probe\cz_seed_light   --q 80
python training\recompress_dir.py --src D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --out D:\probe\cz_nat_light --q 80

:: 微信式重压(双次 q65) —— 和当初测线路A 同样的口径
python training\recompress_dir.py --src D:\probe\localedit_heldout   --out D:\probe\cz_white_heavy  --q 65 --double
python training\recompress_dir.py --src D:\probe\localedit_blue      --out D:\probe\cz_blue_heavy   --q 65 --double
python training\recompress_dir.py --src D:\probe\localedit_seedream  --out D:\probe\cz_seed_heavy   --q 65 --double
python training\recompress_dir.py --src D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\val\nature --out D:\probe\cz_nat_heavy --q 65 --double
```
(真图那两条最慢, 2700 张。)

## 2 全部打分(同上线配置)
```
cd D:\SSP
foreach ($d in @("cz_white_light","cz_blue_light","cz_seed_light","cz_nat_light","cz_white_heavy","cz_blue_heavy","cz_seed_heavy","cz_nat_heavy")) {
  python D:\alipay-platform-classifier\training\predict_tiled.py --ssp-repo D:\SSP --model $ld2 --input "D:\probe\$d" --output_dir "D:\probe\${d}_out" --roi-amount --amount-pad 0 --blue-locator --roi-top 0.6 --device cuda
}
```
(若 foreach 不方便, 就把 8 条 predict 命令逐条跑。)

## 3 按**固定阈值 0.9330** 看退化(重点)
```
cd D:\alipay-platform-classifier
:: 轻度压缩
python training\sweep_thresholds.py --at 0.9330 --require-located --fake D:\probe\cz_white_light_out\summary.csv --genuine D:\probe\cz_nat_light_out\summary.csv
python training\sweep_thresholds.py --at 0.9330 --require-located --fake D:\probe\cz_blue_light_out\summary.csv  --genuine D:\probe\cz_nat_light_out\summary.csv
python training\sweep_thresholds.py --at 0.9330 --require-located --fake D:\probe\cz_seed_light_out\summary.csv  --genuine D:\probe\cz_nat_light_out\summary.csv
:: 重度压缩
python training\sweep_thresholds.py --at 0.9330 --require-located --fake D:\probe\cz_white_heavy_out\summary.csv --genuine D:\probe\cz_nat_heavy_out\summary.csv
python training\sweep_thresholds.py --at 0.9330 --require-located --fake D:\probe\cz_blue_heavy_out\summary.csv  --genuine D:\probe\cz_nat_heavy_out\summary.csv
python training\sweep_thresholds.py --at 0.9330 --require-located --fake D:\probe\cz_seed_heavy_out\summary.csv  --genuine D:\probe\cz_nat_heavy_out\summary.csv
```
只看 **tile_max** 那一行(上线用的就是它)。

---

## 对照基线(未压缩, 固定阈值 0.9330)
| | 召回 | 误杀 |
|---|---|---|
| 白图 | 95.9% | 0.5% |
| 蓝图 | 99.3% | 0.5% |
| 真 Seedream | 94.5% | 0.5% |

## 怎么判
| 结果 | 含义 | 怎么办 |
|---|---|---|
| 召回掉一点(>80%)+ 误杀没涨 | 抗压缩还行 | 现有口径基本成立, 补一句"压缩后略降" |
| **召回大塌**(像线路A 那样掉到个位数) | 和线路A 同一个硬限 | **必须修正对外口径**: 95% 只对原始上传成立; 并考虑把压缩图加进训练试一版 |
| **误杀涨上去** | 压缩把真图分数顶高了 | 阈值要按"压缩后的真图"重新标定, 或按图是否被压缩走不同阈值 |

## 发我什么
第 3 步的六段输出(每段的 tile_max 行)。我做对照结论 + 决定要不要修口径 / 要不要拿压缩图重训。
