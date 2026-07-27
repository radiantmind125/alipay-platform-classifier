# 官方 SSP 支付宝重训 · 服务器 GPU 手册

经理要的"重训那几个评估模型"的可执行步骤。重训**官方 ResNet50 SSP**(能直接被经理的
predict_ssp.py / predict_all_models.py 加载),不是我的 SSP-tiny(那只是 CPU 实验)。

## 0. 前提
- 有 GPU 的服务器。真图在 `D:\download\TempFakeImages`(蓝图)+ `D:\download2\TempFakeImages`(白图)。
- SSP 源码在 `D:\SSP-AI-Generated-Image-Detection-main`(或 `D:\SSP`)。
- platform-classifier 仓库在服务器上(含 training/ 脚本)。

## 1. 环境
```
pip install torch torchvision pillow numpy opencv-python diffusers transformers accelerate
# 打过补丁后 scipy 不必钉;若不打补丁则必须 pip install scipy==1.10.1
```
离线服务器:先把 torchvision resnet50 权重预放 `~/.cache/torch/hub/checkpoints/`(首训 pretrained=True 要下)。

## 2. 给 SSP 仓库打补丁(修三个坑)
```
python training/patch_ssp_repo.py --repo D:\SSP-AI-Generated-Image-Detection-main
```
改:choices=[2,2,2,2,1,2,2,2](只在 sdv4 槽训+验,不然验证会找不到其余 7 个 GenImage 目录崩)、
scipy 导入路径(现代 scipy 可用)。带 .bak 备份。

## 3. 造假图(ai 类)—— 真图海量、假图为零,自己造,不问经理
```
# a) 模板/字段类假图(贴近真实欺诈,主力):改金额/负号/字体
python training/engine_b_tamper.py --src-root D:\download2\TempFakeImages ^
    --out D:\ssp_ai_raw\tamper --n 6000 --save-mask
# b) AI 生成类(扩散指纹,补充多样性):VAE 往返 + img2img
python training/gen_ai_fakes.py --genuine-roots D:\download\TempFakeImages D:\download2\TempFakeImages ^
    --out D:\ssp_ai_raw\aigen --n 4000 --methods vae img2img ^
    --models stabilityai/sd-vae-ft-mse runwayml/stable-diffusion-v1-5 --device cuda
```

## 4. 组装官方 nature/ai 数据集
```
python training/build_ssp_dataset.py ^
    --genuine-roots D:\download\TempFakeImages D:\download2\TempFakeImages ^
    --gold platform-classifier/gold/recapture_gold.jsonl --gold-img-root D:\download\TempFakeImages ^
    --redteam-dir redteam_prod\attacked --extra-ai D:\ssp_ai_raw\tamper ^
    --out D:\ssp_alipay --n-nature 10000 --n-recap-synth 3000 --val-frac 0.15
# 再把 gen_ai_fakes 的 aigen 也拷进 D:\ssp_alipay\imagenet_ai_0419_sdv4\train\ai\
# 检查:train/{nature,ai} 与 val/{nature,ai} 都非空(val/ai 空会 NaN/崩)
```

## 5. 训练(GPU)
```
cd D:\SSP-AI-Generated-Image-Detection-main
python train_val.py --image_root D:\ssp_alipay --gpu_id 0 --save_path .\snapshot\alipay\
# Windows 若卡在 DataLoader,把 tdataloader.py 的 num_workers=4 改 0
```
出 `.\snapshot\alipay\Net_epoch_best.pth`。

## 6. 推理(drop-in 经理脚本,标签约定已对齐 nature=1/ai=0,无需 --ai_label)
```
python D:\SSP\predict_ssp.py --model .\snapshot\alipay\Net_epoch_best.pth --input <待测图> --device cpu
```

## 7. 诚实汇报(必须)
- 正样本约 99% 是**自造**(合成翻拍/篡改/AI 生成),周一数字偏乐观。
- **单独在真造假图上报召回**:gold/recapture_gold.jsonl 里的真翻拍 + 拼音假页 + redteam,别和合成混report。
- SSP 抓的是"整图渲染真假";**局部改金额(03/04 负号)要靠 Head B 字形/模板取证**,SSP 结构上看不到。
- 若真实欺诈以"模板改字"为主(图库挖掘倾向如此),SSP 主要靠"真渲染 vs 假渲染管线"差异,
  Head B 才是抓字段级的主力 —— 两者一起进融合。
