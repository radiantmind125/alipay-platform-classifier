# 参照实现 `reference_py/` · 怎么用

这一份 **不是要你们跑的东西**, 是**对数用的**。

你们按 `ONNX_PORT_SPEC.md` 在 .NET 里实现一遍; 中间对不上的时候,
拿这份 Python 跑同一张图, 逐级比对, 看是哪一步岔了。

## 依赖

```
pip install numpy opencv-python pillow onnxruntime
```

**不需要 torch**, 也不需要任何模型训练框架 —— 归一化和网络都已经固化在 ONNX 图里了。
(仓库里另有走 torch 的老路径, 但那是命令行入口, 这份参照实现从不调用它。)

## 跑一张图

```
python reference_py/ssp_score_one.py ^
    --a-onnx aigen_v7.onnx ^
    --b-onnx localdet9.onnx ^
    --pretty ^
    --input 某张图.png
```

输出一段 JSON 到 **stdout**; 末尾那段汇总走 **stderr**, 所以汇总不会混进 JSON 里。

★ **但 stdout 不是纯 JSON**: **第一次跑**会先往 stdout 打一行
`(没找到配置, 已写一份默认的到 ...)`, 之后才是 JSON。
所以 `> out.json` 之后**逐行严格 parse 会在第一次运行就炸**。
要么跳过不以 `{` 开头的行, 要么先空跑一次让配置落盘再正式对数。

常用开关: `--quiet` 只出汇总不打每张 JSON; `--no-summary` 不要汇总;
`--input` 给目录就是整个目录跑一遍。

## 七个文件一个都不能少

```
ssp_score_one.py      入口
  ├── patch_select.py     取块(跨语言可复现的那套)
  ├── locate_blue.py      金额定位
  ├── predict_tiled.py    线路B 的切块和打分
  │     └── engine_b_tamper.py
  └── ssp_decide.py       判定规则(阈值在这里)
        └── watermark_scan.py   线路C 元数据
```

只拷 `ssp_score_one.py` 是跑不起来的。

## 关于 `ssp_config.json`

**第一次跑会自动在 `reference_py/` 里生成一份**, 里面是默认阈值
(`0.8031 / 0.6497 / 0.9811 / 0.7549`)和 `patch_mode: lcg` —— 与我们这边完全一致,
不用改。看到"自动生成了一份配置"的提示是正常的。

★ 配置里还有几个 `.pth` 训练模型路径, 在你们机器上多半不存在 —— **不用管**,
这条路走的是 ONNX, 从头到尾不加载 `.pth`。

## 一条容易漏的判定规则

**`.jpeg` 结尾的图不自动拒, 转人工复核。**

★ **但这条豁免管不到线路C**: 判定时**线路C 命中会短路**, 在扩展名检查之前就返回了。
也就是说**带 AIGC 元数据的 `.jpeg` 照样自动拒**。别把豁免实现成全局开关。

原因: 实测 `.jpeg` 占比不到 1%, 却在高分区占了 5% 以上 —— 它们基本都是过了修图 App
(醒图、美图秀秀那类)的图, 修图痕迹本身就会把分数抬高。
**我们那个每万张 2.7 张的误杀率, 是在"`.jpeg` 已经免自动拒"的前提下标定的**,
不实现这条, 实际误杀率会更高。

★ 这条**只认 `.jpeg` 不认 `.jpg`**, 看着像笔误其实不是:
数据上 `.jpg` 的表现跟 `.png` 一样正常, 只有 `.jpeg` 是修图 App 存出来的。

★ 一致性向量里那 7 张**全是 PNG**, 所以**这条规则对照图覆盖不到**, 只能靠你们自己实现。
