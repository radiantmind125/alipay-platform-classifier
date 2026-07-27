# 官方 SSP 支付宝重训 · 服务器 GPU 详细手册（一步一步）

目标：在服务器 GPU 上重训**官方 ResNet50 SSP**，产出 `Net_epoch_best.pth`，能被经理的
`predict_ssp.py` / `predict_all_models.py` 直接加载。全程命令按 Windows（D 盘）写，单行可直接复制。

> 先读一遍"诚实前提"（第 9 节）——SSP 抓的是"整图渲染真假/翻拍"，抓不到局部改金额（那靠 Head B）。

---

## 0. 前置检查（5 分钟）

1. 确认有 GPU：
   ```
   nvidia-smi
   ```
   要能看到显卡。看不到就没法训（官方 train_val.py 只有 GPU 路径）。

2. 确认三样东西在服务器上，记下路径：
   - SSP 源码目录，例如 `D:\SSP-AI-Generated-Image-Detection-main`
   - 真图：`D:\download\TempFakeImages`（蓝图）、`D:\download2\TempFakeImages`（白图）
   - platform-classifier 仓库（含 training\ 脚本）。已拉到 `D:\alipay-platform-classifier`。
     每次开工前 `git pull` 拿最新脚本 + gold 标签：
     ```
     cd /d D:\alipay-platform-classifier
     git pull
     ```

3. 之后所有 `python training\...` 命令都在 `D:\alipay-platform-classifier` 目录下运行：
   ```
   cd /d D:\alipay-platform-classifier
   ```

---

## 1. 装环境（10-20 分钟，看网速）

用带 GPU 的 torch（**别装 CPU 版**，否则训练会报 CUDA 不可用）：
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy opencv-python scipy diffusers transformers accelerate
```
验证 GPU torch：
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
必须打印 `True`。打印 `False` 就是装成 CPU 版了，重装 cu121 那行。
（cu121 是 CUDA 12.1；显卡驱动老的话换 cu118。）

---

## 2. 给 SSP 仓库打补丁（1 分钟）

修官方仓库两个会直接崩的坑（choices 验证目录、scipy 导入路径），带 .bak 备份：
```
python training\patch_ssp_repo.py --repo D:\SSP-AI-Generated-Image-Detection-main
```
看到"已改 options.py / utils/tdataloader.py"即成功。（那条"没找到待替换串"是备用规则的正常提示，忽略。）

---

## 3. 造假图 ai 类（这是数据瓶颈——我们自己造，不问经理）

SSP 只能从"整图指纹"类假图学：**翻拍** 和 **AI 生成**。局部改金额它看不到（那是 Head B 的活）。

**3a. AI 生成假图（SSP 的主 ai 类，GPU，约 10-20 分钟）——用 vae（稳、快、只下 335MB）**
```
python training\gen_ai_fakes.py --genuine-roots D:\download\TempFakeImages D:\download2\TempFakeImages --out D:\ssp_ai_raw\aigen --n 4000 --methods vae --models stabilityai/sd-vae-ft-mse --device cuda
```
产出：`D:\ssp_ai_raw\aigen\*.jpg`（约 4000 张）。
> 注意:**不要用 `runwayml/stable-diffusion-v1-5` 做 img2img —— 该模型已被官方从 HuggingFace 下架(404)。**
> img2img 非必需;vae 往返已足够给 SSP 盖上生成指纹。真想加 img2img,自己换一个当前可用的 SD 模型 id。
> vae 输出会偏小偏糊,没关系:下面第 5 步会把两类统一到同一尺寸/质量,消除"低清=假"的泄漏。

> 合成翻拍 + 真翻拍 + redteam 由下一步的 build_ssp_dataset 自动加进 ai，不用手动跑。

**3b.（可选，给 Head B 用，不进 SSP）局部篡改假图**
```
python training\engine_b_tamper.py --src-root D:\download2\TempFakeImages --out D:\headb\tamper --n 6000 --save-mask
```
这批是给 Head B（字段/字形取证）训练/评估用的，**不要**放进 SSP 的 ai 目录。

---

## 4. 组装官方 nature/ai 数据集（10-20 分钟）

nature=真图（自动排除相机/翻拍），ai=合成翻拍+真翻拍+redteam+第3a步的 AI 生成：
```
python training\build_ssp_dataset.py --genuine-roots D:\download\TempFakeImages D:\download2\TempFakeImages --gold training\data\recapture_gold.jsonl --gold-img-root D:\download\TempFakeImages --extra-ai D:\ssp_ai_raw\aigen --out D:\ssp_alipay --n-nature 10000 --n-recap-synth 3000 --val-frac 0.15
```
（`--gold training\data\recapture_gold.jsonl` 是随仓库来的真翻拍标签，指向的图在 `D:\download\TempFakeImages`。
若你另有 redteam 攻击图，加 `--redteam-dir <那个目录>`；没有就不加。）
产出布局：`D:\ssp_alipay\imagenet_ai_0419_sdv4\{train,val}\{nature,ai}`。

---

## 5. 统一尺寸+压缩（防泄漏，5-10 分钟）—— 别跳过

否则 SSP 会学成"JPEG=假"或"低清=假"。把 nature 和 ai 两类**统一到同一最长边(默认768)和同一 JPEG 质量**：
```
python training\reencode_uniform.py --root D:\ssp_alipay\imagenet_ai_0419_sdv4 --q 90 --max-side 768
```
（这一步同时解决 vae 假图偏小偏糊的问题:真图也缩到 768,两类清晰度对齐,模型学的是真伪不是清晰度。）

---

## 6. 开训前自检（1 分钟）

看四个目录都非空（**val\ai 为空训练会崩/NaN**）：
```
python -c "import os,glob;[print(s,c,len(glob.glob(rf'D:\ssp_alipay\imagenet_ai_0419_sdv4\{s}\{c}\*.jpg'))) for s in ['train','val'] for c in ['nature','ai']]"
```
四行数字都 > 0。val\ai 至少几十张。若 val\ai=0，回第 4 步（gold 真翻拍应已分一半进 val）。

---

## 7. 训练（GPU，约 30-90 分钟，看显卡和数据量）

```
cd /d D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_alipay --gpu_id 0 --save_path .\snapshot\alipay\
```
- 每轮会打印 `ai accu` / `nature accu` / `Accuracy`，共 30 轮。
- 卡在 DataLoader 不动（Windows 常见）：把 `utils\tdataloader.py` 里两处 `num_workers=4` 改成 `num_workers=0`，重跑。
- 显存不够（CUDA out of memory）：`options.py` 里 `batchsize` 64 改 32 或 16。
- 卡在下 resnet50 权重：联网即可；离线要预放 torchvision 的 resnet50 到 `C:\Users\<你>\.cache\torch\hub\checkpoints\`。
产出：`D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth`。

---

## 8. 推理测试（drop-in 经理脚本）

标签约定已对齐 nature=1/ai=0，**无需任何 flag**：
```
python D:\SSP\predict_ssp.py --model D:\SSP-AI-Generated-Image-Detection-main\snapshot\alipay\Net_epoch_best.pth --input <一批待测图目录> --device cpu
```
输出每图 `ai_score` + 决策（Reject≥0.9 / ManualReview≥0.6 / Pass）。

---

## 9. 诚实前提 / 汇报口径（很重要）

- **正样本约 99% 是我们自造**（合成翻拍 + AI 生成），所以整体分数偏乐观。**必须单独在真造假图上报召回**：`training\data\recapture_gold.jsonl` 里的真翻拍 + 拼音假页 + redteam，跟合成分开报。
- **SSP 抓"整图渲染真假 + 翻拍"**；**改金额/负号那种局部篡改（经理 03/04 例子）SSP 结构上看不到**，要靠 Head B 字段/字形取证（我在本地建，随后接进同一融合分）。
- 图库挖掘倾向：真实欺诈以"模板改字"为主 → SSP 主要靠"真渲染 vs 假渲染管线"差异，**Head B 才是抓字段级欺诈的主力**。两者一起进融合层，别指望单个 SSP 包打。
- 一句话对经理：SSP 已能在我们自己的支付宝数据上重训跑通、直接进你的 predict 脚本;字段级篡改另有配套检测在做。
