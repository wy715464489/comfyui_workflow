# LoRA 训练指南（kohya_ss）

> 训练角色专属 LoRA 是实现跨图一致性的**最强方案**。本文档覆盖从数据集准备到训练完成的完整流程，针对 Mac M4（MPS）和 GTX 1080（CUDA）两套环境。

---

## 目录

1. [为什么需要训练 LoRA](#一为什么需要训练-lora)
2. [环境准备](#二环境准备)
3. [数据集准备](#三数据集准备)
4. [训练配置](#四训练配置)
5. [开始训练](#五开始训练)
6. [训练结果验证](#六训练结果验证)
7. [在工作流中使用 LoRA](#七在工作流中使用-lora)
8. [常见问题](#八常见问题)

---

## 一、为什么需要训练 LoRA

| 方法 | 一致性强度 | 训练成本 | 使用门槛 |
|---|---|---|---|
| 固定 seed + 提示词 | ⭐⭐ | 无 | 极低 |
| IP-Adapter | ⭐⭐⭐⭐ | 无 | 低 |
| ControlNet Pose | ⭐⭐⭐（结构一致） | 无 | 中 |
| **角色 LoRA** | ⭐⭐⭐⭐⭐ | 需要数据集 + 训练 | 中 |

**LoRA 的优势**：
- 模型直接"记住"角色的外观特征（发色、服装、脸型等）
- 只需提示词中包含触发词（trigger word）即可稳定召唤角色
- 与 IP-Adapter 叠加使用，可获得极强的一致性

---

## 二、环境准备

### 安装 kohya_ss

kohya_ss 是业界最主流的 LoRA 训练工具，支持 SD1.5 / SDXL / Flux。

```bash
# 克隆仓库
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# 创建专用 conda 环境（与 ComfyUI 隔离，避免依赖冲突）
conda create -n kohya python=3.10 -y
conda activate kohya
```

#### Mac M4（MPS）

```bash
# PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# kohya_ss 依赖
pip install -r requirements_mac.txt   # 若存在
# 或
pip install -r requirements.txt

# 安装 accelerate 并配置（Mac 选 no distributed，fp16 选 no，使用 mps）
pip install accelerate
accelerate config
# 配置选项：
#   compute environment: This machine
#   distributed type: No distributed training
#   machine rank: 0
#   num processes: 1
#   mixed precision: no  (Mac MPS 暂不稳定支持 fp16)
```

#### GTX 1080（CUDA，Windows/Linux）

```bash
# CUDA 11.8 版 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# kohya 依赖
pip install -r requirements.txt

# xformers（显存优化，GTX 1080 必装）
pip install xformers

# accelerate 配置
accelerate config
# 配置选项：
#   distributed type: No distributed training
#   mixed precision: fp16
```

### 安装自动标注工具（可选但推荐）

WD14 Tagger 自动为训练图生成 Booru 风格标注：

```bash
pip install huggingface_hub
# 模型会在第一次使用时自动下载
```

---

## 三、数据集准备

### 图像要求

| 项目 | 要求 |
|---|---|
| 数量 | **20~50 张**（SD1.5）；SDXL 建议 30~80 张 |
| 分辨率 | 512×512 或 768×768（kohya_ss 自动裁剪） |
| 格式 | PNG / JPG |
| 内容 | 目标角色的多角度、多表情、多姿势图（越多样越好） |
| 背景 | 建议去背（白底）或背景简单，减少背景被学习 |
| 质量 | 清晰、无噪点、角色主体突出 |

### 目录结构

```
dataset/
└── train/
    └── 30_hero_warrior/         # 格式：{重复次数}_{触发词}
        ├── hero_001.png
        ├── hero_001.txt          # 对应的标注文件（同名 .txt）
        ├── hero_002.png
        ├── hero_002.txt
        └── ...
```

**目录命名规则**：`{重复次数}_{触发词}`
- **重复次数**：每张图在一个 epoch 内被重复使用的次数。图少时（<20张）设 5~10；图多时（>50张）设 1~3
- **触发词**：模型将学会用该词召唤此角色，如 `hero_warrior`、`yui_chan`

### 生成标注文件

**方式 1：手动编写（最准确）**

每张图对应一个 `.txt` 文件，内容为该图的描述（不含触发词，触发词由目录名自动添加）：

```
# hero_001.txt
1girl, standing, white background, full body, silver armor, red hair, blue eyes, smile
```

**方式 2：WD14 Tagger 自动标注**

```bash
# 安装
pip install onnxruntime  # CPU 推理
# 或 pip install onnxruntime-gpu  # GPU 推理

# 使用 kohya_ss 内置脚本标注
python finetune/tag_images_by_wd14_tagger.py \
  --batch_size 4 \
  --caption_extension ".txt" \
  --model_dir "path/to/wd14_tagger_model" \
  "dataset/train/30_hero_warrior/"
```

或使用 ComfyUI-Manager 内置的 WD14 Tagger 节点在 ComfyUI 界面中批量标注。

### 标注质量优化技巧

1. **删除背景描述**：如果训练角色 LoRA，从标注中删除 `background` 相关词（防止学到背景）
2. **保留特征词**：`red hair`, `blue eyes` 等角色特征词**保留**（让模型知道特征在哪）
3. **删除姿势词**：`standing`, `sitting` 等**可删除**（LoRA 应学角色外观，不学姿势）
4. **统一格式**：标注用逗号分隔 tag，不用自然语言句子

---

## 四、训练配置

### 推荐训练参数（SD1.5 角色 LoRA）

```toml
# config_character_lora.toml

[general]
enable_bucket = true
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 1024

[dataset]
# 训练数据集根目录
train_data_dir = "dataset/train"
resolution = 512

[training]
pretrained_model_name_or_path = "/path/to/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
output_dir = "output/lora"
output_name = "hero_warrior_v1"
save_model_as = "safetensors"

# 训练步数（图片数 × 重复次数 × epoch数 / batch_size）
# 例：30张 × 5重复 × 10epoch / 1batch = 1500步
max_train_steps = 1500

learning_rate = 1e-4
unet_lr = 1e-4
text_encoder_lr = 5e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

train_batch_size = 1

# LoRA 配置
network_module = "networks.lora"
network_dim = 32          # LoRA rank，越大模型越大（16~64，推荐32）
network_alpha = 16        # network_alpha / network_dim = 实际 LoRA 强度比例

# 保存
save_every_n_steps = 500
mixed_precision = "fp16"   # Mac 改为 "no"
```

### Mac M4 特殊配置

```toml
# Mac MPS 不稳定支持 fp16，需要修改：
mixed_precision = "no"          # 关闭混合精度
full_fp16 = false
xformers = false                # Mac 不支持 xformers
cache_latents = true            # 缓存 latent，减少显存压力
```

### GTX 1080 特殊配置

```toml
# 8GB 显存优化：
mixed_precision = "fp16"
xformers = true                 # 开启，节省显存约 30%
gradient_checkpointing = true   # 梯度检查点，节省显存（速度慢 20%）
train_batch_size = 1
cache_latents = true
```

---

## 五、开始训练

### 使用 kohya_ss GUI（推荐新手）

```bash
cd kohya_ss
conda activate kohya
python kohya_gui.py
# 浏览器访问 http://127.0.0.1:7860
# 选择 LoRA → 填写配置 → 开始训练
```

### 使用命令行

```bash
cd kohya_ss
conda activate kohya

accelerate launch train_network.py \
  --pretrained_model_name_or_path="/path/to/v1-5-pruned-emaonly-fp16.safetensors" \
  --train_data_dir="dataset/train" \
  --output_dir="output/lora" \
  --output_name="hero_warrior_v1" \
  --save_model_as=safetensors \
  --resolution=512 \
  --enable_bucket \
  --train_batch_size=1 \
  --max_train_steps=1500 \
  --learning_rate=1e-4 \
  --unet_lr=1e-4 \
  --text_encoder_lr=5e-5 \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16 \
  --lr_scheduler=cosine_with_restarts \
  --lr_warmup_steps=100 \
  --save_every_n_steps=500 \
  --mixed_precision=fp16 \
  --cache_latents \
  --xformers
# Mac 删除 --xformers，修改 --mixed_precision=no
```

### 预计训练时间

| 设备 | 步数 | 预计时间 |
|---|---|---|
| Mac M4 (MPS) | 1500步 | 约 30~60 分钟 |
| GTX 1080 8GB | 1500步 | 约 15~30 分钟 |
| RTX 3090 | 1500步 | 约 5~10 分钟 |

---

## 六、训练结果验证

训练完成后，LoRA 文件在 `output/lora/hero_warrior_v1.safetensors`（约 40~80MB）。

**步骤 1**：将 LoRA 复制到 ComfyUI

```bash
cp output/lora/hero_warrior_v1.safetensors \
   /path/to/ComfyUI/models/loras/
```

**步骤 2**：在 ComfyUI 中测试

1. 打开 ComfyUI 界面
2. 加载 `workflows/character/minimal_txt2img.json` 或 `base_character.json`
3. 在 `LoRA Loader` 节点选择 `hero_warrior_v1.safetensors`，strength 设为 0.8
4. 提示词中加入触发词（目录名中的词，如 `hero_warrior`）
5. 生成并对比：**有 LoRA** vs **无 LoRA** 的差异

**评估标准**：
- ✅ 触发词起效：提示词中加入触发词后，角色外观显著向训练图靠拢
- ✅ 可控性：改变其他提示词（姿势、背景），角色外观保持稳定
- ❌ 过拟合：只能生成一种固定外观，无法响应姿势/表情变化 → 减少训练步数

**对比测试脚本**：

```bash
# 不带 LoRA（strength=0）
python scripts/test_single.py \
  --prompt "hero_warrior, 1girl, red hair, blue eyes, silver armor, white background, full body"

# 带 LoRA（在 base_character.json 中开启，strength=0.8）
# 编辑 characters.json 后运行：
python scripts/batch_generate.py --ids hero_001
```

---

## 七、在工作流中使用 LoRA

### 在 ComfyUI 界面

1. 打开 `base_character.json`
2. 找到 `LoRA Loader` 节点
3. 设置 `lora_name` = `hero_warrior_v1.safetensors`
4. 设置 `strength_model` = 0.8，`strength_clip` = 0.8
5. 在正向提示词中加入触发词：`hero_warrior`

### 在批量脚本中

编辑 `scripts/config/characters.json`：

```json
{
  "id": "hero_001",
  "lora": "hero_warrior_v1.safetensors",
  "lora_weight": 0.8,
  "base_prompt": "hero_warrior, 1girl, red hair, blue eyes, silver armor, ...",
  ...
}
```

### LoRA + IP-Adapter 叠加使用

**最佳组合配置**：

```
CheckpointLoader → IPAdapterUnifiedLoader → LoraLoader → KSampler
```

- LoRA weight: **0.6~0.8**（角色外观的基础锚定）
- IP-Adapter weight: **0.4~0.6**（参考图的外观微调）

当两者叠加时，通常需要**适当降低**两者的权重，避免过度约束导致图像僵硬。

---

## 八、常见问题

**Q：训练完 LoRA 后触发词不起效？**

A：检查目录名格式是否正确（`{重复次数}_{触发词}`），确认触发词在提示词中出现。

**Q：LoRA 导致图像出现大量噪点/崩坏？**

A：LoRA 权重过高或训练步数过多。尝试将 strength 从 0.8 降到 0.5，或使用中间保存点（`--save_every_n_steps`）。

**Q：Mac M4 训练时显存报错？**

A：在配置中添加 `--cache_latents`（提前编码所有图像到 Latent）和 `--gradient_checkpointing`（梯度检查点）。

**Q：20 张图够训练吗？**

A：SD1.5 基础可行。提高重复次数（设为 `10_触发词`）和 epoch 数补偿数据量不足。但更多高质量图像总是更好。

**Q：如何训练 SDXL LoRA？**

A：将 `pretrained_model_name_or_path` 替换为 SDXL checkpoint，分辨率改为 1024，其他参数类似。kohya_ss 支持 SDXL LoRA，命令不变。

**Q：IP-Adapter 和 LoRA 哪个优先？**

A：先跑 IP-Adapter（快，无需训练）验证概念，满意后再训练 LoRA 固化外观。两者可以叠加使用。

---

> 📖 相关文档：[工作流原理指南](./workflow_guide.md) | [README](../README.md)
>
> 🔗 参考资源：
> - [kohya_ss 官方文档](https://github.com/bmaltais/kohya_ss)
> - [LoRA 训练最佳实践（Civitai 指南）](https://civitai.com/articles/124/lora-training-guide)
> - [WD14 Tagger 自动标注](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)
