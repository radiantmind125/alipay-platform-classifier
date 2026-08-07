# 接下来怎么走 · 一步步(2026-08-07 版)

顺序是排过的, **别跳步** —— 第 2 步的产物是第 3 步的输入, 第 4 步会让第 3 步的数字作废(所以先冻结再重训)。

---

## 0 先做这个 · API key 已泄露, 去作废重申请

`REALGEN_RETRAIN_RUNBOOK.md` 第 27 行以前直接贴着真 key, commit `b457f79`, **已经推到公开 GitHub 上了**。
文件里已经改成占位符(commit `79d0037`), **但历史里还在, 删了不等于没泄露。**

1. 去 DMXAPI 后台**把那个 key 作废, 重新申请一个**。
2. 顺手看一眼用量记录, 有没有不是自己发的调用。
3. 新 key **只**这样用: `$env:DMX_KEY = "sk-新的"`, **不要写进任何文件**。

> 改写 git 历史(filter-repo + force push)也能做, 但**挽回不了已经泄露的事实**, 还会打乱别人的克隆。
> **作废重发才是真正的处置。** 要不要改历史, 你定。

---

## 1 建一份"证据独立"的排除名单

**为什么要重建**: 标误杀率就是在数模型判错了多少真图。**按模型自己的高分去剔图 = 把模型的错误从错误统计里删掉**,
测出来的误杀会假性偏低, 阈值就会定得过激进, 上线赔钱。
**所以剔图的证据必须跟模型分数无关。** 水印是生成器自己盖的, 正好符合。

```powershell
cd D:\alipay-platform-classifier
git pull

# 1a 先自检(必跑) —— 确认阈值在你的数据上也成立
python training\watermark_scan.py --selftest --fakes D:\probe\top_suspect --genuine D:\probe\genuine_20k
```

**看这两个数**:
- 阈值 **0.30** 那一行, **真图命中率(误报)必须接近 0**(本地实测 0.067%, 且命中的两张裁开看确实都带水印);
- 假图命中率**不用追求高** —— 去了水印的本来就抓不到, 这条判据要的是"删对"不是"删干净"。

**误报明显不接近 0 就停下告诉我**, 别往下走。

```powershell
# 1b 扫真图池, 出独立证据的排除名单
python training\watermark_scan.py --input D:\probe\genuine_20k --out D:\probe\exclude_evidence.txt
(Get-Content D:\probe\exclude_evidence.txt).Count
```

预期命中 **十几张**(2 万张 x 0.067% ≈ 13)。**先把命中的图拷出来肉眼扫一遍**, 确认都带水印再用。

---

## 2 用保守口径重标阈值, 并跑一次敏感性对照

**保守口径(拿去上线、拿去报经理的就是这个)**:
```powershell
python training\autoreject_threshold.py --genuine D:\probe\genuine_20k_v7\summary.csv --fake D:\probe\wan_full_v7\summary.csv D:\probe\qwen_v7\summary.csv --exclude D:\probe\exclude_evidence.txt

python training\autoreject_threshold.py --col tile_top3 --require-located --genuine D:\probe\genuine_20k_ld3\summary.csv --fake D:\probe\wan_ld3\summary.csv D:\probe\seed_ld3\summary.csv D:\probe\white_ld3\summary.csv D:\probe\blue_ld3\summary.csv --exclude D:\probe\exclude_evidence.txt
```

**乐观口径(只当敏感性看, 不作为上线依据)**: 把 `exclude_evidence.txt` 和 `cross_flagged.txt` 合并再跑一遍。
```powershell
Get-Content D:\probe\exclude_evidence.txt, D:\probe\cross_flagged.txt | Sort-Object -Unique |
  Set-Content -Encoding utf8 D:\probe\exclude_optimistic.txt
```

**怎么读**: **1/5000 和 1/10000 两档, 两个口径应该差不多** —— 那这个争论就不重要, 用保守的报数即可。
**只有零误杀档会差很多**, 因为那一档按定义就是由**单张**图决定的。

---

## 3 冻结上线配置

**建议上 1/10000 这一档, 不要上零误杀档。**
理由: 零误杀档的分数线**由单张图决定**, 那张图是真是假我们争论了半天也没定论(见 §178)。
**上 1/10000, 这个争论就绕过去了**, 而覆盖率几乎不损失。

保守起步也可以先上 **1/1000**, 跑一两周看真实复核量, 再往 1/5000 收。

把最终两张表填进 `DEPLOY_SPEC.md`, **并写清楚这是在"剔掉 N 张有水印实证的图"之后标的**。

---

## 4 清训练集 + 重训(最大的一块收益)

**注意: 这一步做完, 第 2 步的所有阈值都要重标。所以先把第 3 步冻结好, 留一个已知可用的基线。**

**只能按水印清, 不能按模型分数清** —— 原因见 `POOL_CLEANUP_RUNBOOK.md` 第 4 步那段。

```powershell
python training\watermark_scan.py --input D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\train\nature --out D:\probe\trainnat_watermarked.txt

# 挪走(别删, 留证)
New-Item -ItemType Directory -Force D:\probe\trainnat_removed | Out-Null
Get-Content D:\probe\trainnat_watermarked.txt | ForEach-Object {
  $p = Join-Path D:\ssp_aigen_v5\imagenet_ai_0419_sdv4\train\nature $_
  if (Test-Path $p) { Move-Item $p D:\probe\trainnat_removed }
}
```

然后重训 v8(线路A)和 localdet4(线路B)。

**★重训后的验收闸门(必须过, 不过就回滚)**:
1. **误杀不许变差** —— 在**同一批**真图上, 同一档预算下的分数线不许比 v7/localdet3 更高;
2. **老生成器不许退步** —— 白图/蓝图/豆包/万相四个测试集逐个比, 任何一个明显掉了就是坏了;
3. **聚合方式要重新横比**(max / top3 / mean) —— **聚合方式是模型的属性不是通用规律**, 上一版换模型就从 max 变成了 top3。

---

## 5 经理点名的两件事: 千问 + 抖音

**先说清楚现状**(报经理别说满):
- **线路A 的千问是用 Qwen-Image 的 VAE 编解码往返造的样本训的, 不是真千问改图服务出的图。**
  指纹是真的会转移, 但真服务还会再压一道加水印, 指纹会削弱 —— **万相就是这样, 0% 漏检但硬拦只有 2%, 补训才到 94%。**
- **线路B 从来没有千问的训练样本, 也没有千问的测试集** -> **对千问改金额是零测量。这是真缺口。**
- **抖音/即梦和豆包同属 Seedream 系, 我们已经训了豆包**, 大概率覆盖, **但要测不要假设**
  (仓库里就记着教训: 「千问那次拿相似模型估, 结果真模型漏了七成」)。

**第一步: 先探哪些接口真的能用**(key 换成新的之后再跑)
```powershell
$env:DMX_KEY = "sk-新的"
python training\probe_models.py --src D:\download\TempFakeImages\<随便一张真图>.jpg --out D:\probe\modelprobe
```
之前判定"qwen-image-edit 不通"**只试过豆包那个端点**; 万相走的是 DashScope 的 `/v1/responses`,
**千问同属阿里 DashScope, 很可能走的是那条路, 我们从来没试过。** 这个脚本两条端点都试。

**通了之后**: 用 `gen_api_local_edit.py --prompt-mode local-amount` 造一批千问改金额的样本 ->
先用现有 localdet3 测漏多少 -> 漏得多再加进训练。**即梦同样走一遍。**

---

## 每一步要发我什么
- 第 1a 步: 自检那张表(尤其**真图误报**那一列)
- 第 2 步: 两个口径的两张表
- 第 4 步: 重训后的三条验收闸门
- 第 5 步: probe_models 的输出(哪些模型 x 哪个端点通了)
