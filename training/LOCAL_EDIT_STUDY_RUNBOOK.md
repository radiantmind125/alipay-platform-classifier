# 局部 AI 改金额 · 结构盲区研究 · 一步步(PowerShell)

**这是部署前最后一个没验证的大问题。**

## 要回答什么
真实欺诈里最省事的做法**不是**整张重新生成, 而是**拿真截图只把金额那几个数字改掉**。
而 SSP 的判法是: 把图切成很多 32x32 小块, **只挑纹理最丰富的一块**去看指纹。
**金额那一小块很可能不是被挑中的那块 → 改动看不见 → 整张判成真图(漏检)。**
这是**结构性**弱点(不是训练不够, 加数据未必能修), 所以必须实测。

- **抓到** = v6 连只改一小块都能发现 → 部署前一个大信心 + 覆盖最常见的欺诈手法。
- **漏掉** = 本项目最重要的发现 → 说明 SSP 只擅长"整张AI生成", "局部AI改字"要另配检测(把改动区找出来的那种), 得让经理知道。

## 怎么造(已写好脚本)
`gen_local_ai_edit.py` 定位金额区 → **只把那一小块过 VAE 重绘**(盖上 AI 指纹) → 羽化贴回, 其余像素**原封不动**。
产出 = 99% 真像素 + 1% AI 像素 = 真实的"局部 AI 编辑"假图。
**同时造一个对照组**(整图重绘), 这样能干净地把差异归因到"改动面积小"而不是别的。

---

## 前置
```
cd D:\alipay-platform-classifier
git pull
```

## 1 造局部改金额假图(主测)
金额定位是按**白底账单详情**页调的, 所以源用白池:
```
python training\gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\localedit --n 300 --mode local --device cuda
```
- 打印会报"造了 N 张 / 跳过 M 张(多为定位不到金额)"。跳过一些正常(不是每张都是白底金额页)。
- **必做质检**: 开 2-3 张 `D:\probe\localedit` —— 应该看着**就是一张正常收据**, 金额那块可能略微发软/边缘略不同, 但**不该有明显拼接方块或色差**。若金额块明显是个补丁 → 告诉我(我调羽化)。

## 2 造对照组(整图重绘)
```
python training\gen_local_ai_edit.py --src-root D:\download2\TempFakeImages --out D:\probe\fulledit --n 300 --mode full --device cuda
```

## 3 v6 打分
```
$v6 = "D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v6\Net_epoch_best.pth"
cd D:\SSP
python predict_all_models.py --model_root $v6 --input D:\probe\localedit --output_dir D:\probe\localedit_out --device cuda
python predict_all_models.py --model_root $v6 --input D:\probe\fulledit --output_dir D:\probe\fulledit_out --device cuda
```

## 4 算召回
```
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\probe\localedit_out\summary.csv --kind fake
python training\eval_summary.py D:\probe\fulledit_out\summary.csv --kind fake
```

---

## 怎么读
| 结果 | 含义 | 下一步 |
|---|---|---|
| local 漏检低(<20%) | v6 连局部改字都能抓 | 大好消息 直接部署 报经理 |
| local 漏检高 + full 漏检低 | **盲区坐实**: 整张生成能抓 只改一小块抓不到 | 需另配局部篡改检测(Head B 那条线) 要报经理 |
| 两个漏检都高 | 造图有问题(不是盲区) | 发我 我查造图质量 |

**对照很关键**: full 抓得到而 local 抓不到, 才能证明是"改动面积"导致的, 不是造图方法本身的问题。

## 发我什么
两段 `eval_summary` + 1-2 张 `D:\probe\localedit` 样图(我确认金额块重绘得自然、没有假拼接痕迹)。
我据此定: 直接部署 / 还是要补局部篡改检测 + 给经理的口径。
