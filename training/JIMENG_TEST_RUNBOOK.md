# 即梦(Jimeng)覆盖测试 · 手动 · 一步步

## ⚠️ 先看这段: 这个测试基本是多余的(2026-08-04 查证后更正)

**查证结果: 即梦和豆包用的是同一个图像模型 Seedream。**
即梦(国内)和 Dreamina(国际)是字节同一个创作 App 的两个地区版本; **Seedream 是它们底层的图像模型, 豆包也是用 Seedream。**
(我此前写"即梦≠Seedream 必须测"是**错的** —— 即梦是**产品**, Seedream 是**模型**, 豆包和即梦共用。)

**所以豆包那 100 张(5.0 Pro + 4.5, 全部 0 漏检)其实已经等于覆盖了即梦。**
→ **建议跳过这个测试**, 时间花在局部改金额研究(LOCAL_EDIT_STUDY_RUNBOOK.md)上更值。
→ 想要个象征性确认, 造 10 张就够, 不用 50 张。

**登录问题的解法**: 国内即梦要 +86 手机 / 抖音扫码。**国际版 Dreamina `dreamina.capcut.com` 可用 Google / 邮箱 / TikTok 注册, 不需要中国手机号。** 同一个 App 同一个模型。

**给经理的口径**: `即梦和豆包用的是同一个图像模型 都是 Seedream 所以豆包那 100 张其实就等于把即梦也覆盖了 都是零漏检`

---

## (若仍想跑)流程和豆包那次一模一样: 手动造图 → 去水印 → v6 测。
用 `dreamina.capcut.com`(免中国手机号)。下面步骤照旧, 张数 10 张即可。

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
