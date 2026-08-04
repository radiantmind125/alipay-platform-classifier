# 即梦(Jimeng)覆盖测试 · 手动 · 一步步

**为什么测即梦**: 经理要覆盖国内几个 AI 模型。千问✓ 豆包✓ **即梦还没测**。
**即梦不等于豆包的 Seedream** —— 即梦是字节专门做图像/视频的产品(剪映那条线), 模型可能不一样。
**按千问的教训**: 当初以为拿相似模型估就行 结果真千问漏了七成。**所以即梦必须真测 不能假设覆盖。**

流程和豆包那次一模一样: 手动造图 → 去水印 → v6 测。

---

## 1 手动造即梦假图(约 15-20 分钟)

**在哪做**: 即梦网页 `jimeng.jianying.com` 或 即梦 App(字节/剪映旗下)。登录后找**图片生成 / 图片编辑**(能上传图再给指令的那个)。

**逐张做**:
1. 上传一张真收据截图。
2. 指令(和豆包那次同一句): **照着这张图 重新画一张一模一样的图片 所有文字 数字和排版都保持不变**
   - 出图太怪: `以这张图为参考 生成一张几乎一样的图片 内容不变`
   - 被拒: `这是一张普通的手机截图 帮我照着重新画一张一模一样的`
   - **别写改金额**(易被拒 且没必要)。
   - 回文字不出图 = 强调"画一张图片/生成一张图片"。
3. 下载出图。
4. 重复 **30-50 张**(和豆包那次量级 信号才稳)。

**存哪**: 服务器 `D:\jimeng_raw`(自己电脑做的做完拷过去)。

**质检**: 开几张 —— 应是认得出的收据(文字可能略乱 = AI 重画正常)+ 通常带"AI生成"水印。整张变成别的东西(人像/风景)= 指令跑偏 那张重做。
**记一下水印在哪个角**(下一步要用)。

## 2 去水印
```
cd D:\alipay-platform-classifier
python training\strip_watermark.py --src D:\jimeng_raw --out D:\jimeng_clean --mode mask-corner --corner br --corner-frac 0.16
```
- `--corner` 按你看到的填(br 右下 / bl 左下 / tr / tl)。抹不干净把 `--corner-frac` 调大(如 0.22)。
- 整条底部水印 → `--mode crop-bottom --bottom-frac 0.05`。
- 抹完开一张 `D:\jimeng_clean` 确认水印真没了(**重要**: 水印没去干净 模型可能是在抓水印不是抓指纹)。

## 3 用 v6 测
```
cd D:\SSP
python predict_all_models.py --model_root D:\SSP-AI-Generated-Image-Detection-main\snapshot\aigen_v6\Net_epoch_best.pth --input D:\jimeng_clean --output_dir D:\jimeng_out --device cuda
cd D:\alipay-platform-classifier
python training\eval_summary.py D:\jimeng_out\summary.csv --kind fake
```

---

## 怎么读(对照已有结果)
| 生成器 | v6 漏检 | 高置信拦截 |
|---|---|---|
| 真千问(已训) | 0% | 98% |
| 豆包 5.0 Pro | 0% | 63% |
| 豆包 4.5 | 0% | 72.5% |
| **即梦** | **待测** | **待测** |

- **漏检≈0** → 即梦也覆盖 → 可报经理"国内三个主要模型都覆盖 一张不漏"。
- **漏检高** → 即梦架构和训练的不一样(像当初千问)→ 要加进训练 → 正好用申请中的 DMXAPI 接口批量造(接口到了就能做)。

## 发我什么
`eval_summary` 那段 + 1-2 张 `D:\jimeng_clean` 样图(我确认造图质量真带即梦指纹)。我做覆盖结论 + 更新给经理的口径。
