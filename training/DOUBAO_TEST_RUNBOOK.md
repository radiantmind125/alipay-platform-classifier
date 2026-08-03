# 豆包(及国内其他 AI 模型)覆盖测试 · 一步步(PowerShell)

**为什么**: 经理要重点覆盖**国内几个 AI 模型**(豆包 即梦 文心 可灵 千问)。千问已覆盖(开源直接测/训)。
**豆包等是闭源**(没有开源权重)→ 只能用 App 手动造 或 走接口。这份先把**豆包**测出来。
**按 Qwen 教训**: 不能假设 v6 已覆盖豆包 —— 当初也以为覆盖千问 结果差 70%。必须真测。

**思路**: 手动造一小批豆包假图 → 去水印 → 用 v6 测 → 看抓不抓。抓=覆盖(报经理); 漏=加进训练(像当初千问)。

---

## 1 手动造豆包假图(App, 约 15 分钟)
- 开 豆包 App 或 doubao.com 的**图像编辑**功能。
- 上传 20-30 张真收据截图 → 让它**中性重绘**(提示: 重绘这张图 保持内容不变)→ 存到 `D:\doubao_raw`。
- **别让它改金额**(可能被当欺诈拒)。中性重绘就行 —— 只要出图带豆包的生成指纹。
- 20-30 张就够看信号。

## 2 去水印
```
cd D:\alipay-platform-classifier
python training\strip_watermark.py --src D:\doubao_raw --out D:\doubao_clean --mode mask-corner --corner br --corner-frac 0.16
```
- 开一张 `D:\doubao_clean` 确认 "AI生成" 水印没了。位置不对就调 `--corner`(br/bl/tr/tl) / `--corner-frac`。
- 去水印是为了让模型学真指纹 不是学"有水印=假"(骗子也会把水印裁掉)。

## 3 用 v6 测
```
cd D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v6\Net_epoch_best.pth --input D:\doubao_clean --output_dir D:\doubao_out --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\doubao_out\summary.csv --kind fake
```

## 怎么读
- **拦截高 漏检≈0** → v6 已覆盖豆包 → 可给经理报"豆包也能抓"。
- **漏检高** → 豆包架构可能和训练的不一样(像当初千问那样)→ 要把豆包加进训练。这需要更多样本 → 走 Volcengine 接口批量造(那时我写自动生成器 同 --vae-class 那套)。
- 发我: `eval_summary` 那段 + 1-2 张 `D:\doubao_clean` 样图(我确认造图质量 + 定要不要加训练)。

---

## 其他国内模型(即梦 文心 可灵)
- 同样办法: App 手动造小批 → 去水印 → v6 测 → 漏就加训练。
- 即梦是字节的 和豆包同族(Volcengine); 文心是百度; 可灵是快手(视频)。
- **要批量/自动**(经理要覆盖好几个 手动会很累)→ 走各家接口(豆包即梦=Volcengine 方舟; 文心=百度千帆; ...)。**建议把接口访问开出来 我就能像千问那样自动造图 + 加训练**, 不用一张张手动搞。
