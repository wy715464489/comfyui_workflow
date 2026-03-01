# ComfyUI 工作流原理与使用指南

> 面向 2D 游戏资源生成场景，基于 SD1.5 + IP-Adapter + LoRA 技术栈。

---

## 目录

1. [扩散模型基础原理](#一扩散模型基础原理)
2. [工作流节点详解](#二工作流节点详解)
3. [工作流数据流全景图](#三工作流数据流全景图)
4. [正向与负向提示词写法](#四正向与负向提示词写法)
5. [UI 界面使用步骤](#五ui-界面使用步骤)
6. [Python 脚本使用步骤](#六python-脚本使用步骤)
7. [一致性生成技巧](#七一致性生成技巧)

---

## 一、扩散模型基础原理

### Latent Diffusion Model（潜空间扩散模型）

Stable Diffusion 是一种 **Latent Diffusion Model**。生成图像的核心流程分为三个阶段：

```
[文本提示词]
     │
     ▼ CLIP Text Encoder（文本编码器）
[文本条件向量 Conditioning]
     │
     ▼
[纯噪声 Latent]──────────────────────┐
     │                              │
     ▼ U-Net 去噪（KSampler 控制）   │ 反复迭代 N 步
[逐步去噪的 Latent]◄─────────────────┘
     │
     ▼ VAE Decoder（解码器）
[最终图像 512×768 px]
```

**关键概念：**

- **Latent Space（潜空间）**：图像被压缩为 1/8 尺寸的低维向量（512×768 图 → 64×96 的 Latent）。所有去噪计算在潜空间进行，速度快、显存占用低。
- **U-Net**：去噪网络，接受"带噪 Latent + 时间步 + 文本条件"作为输入，预测噪声并逐步去除。
- **CFG（Classifier-Free Guidance）**：同时做有条件预测和无条件预测，用 CFG Scale 控制二者差值，值越高越"听从"提示词，但过高会导致过饱和/失真。
- **Sampler（采样器）**：控制每一步去噪的算法。`dpmpp_2m karras` 是质量与速度的最佳平衡。

### IP-Adapter 的作用

IP-Adapter 在 U-Net 的 Cross-Attention 层中注入参考图的**视觉特征**：

```
[参考图] → CLIP Vision Encoder → [图像特征向量]
                                         │
                                         ▼ 注入 U-Net Cross-Attention
[U-Net 去噪] ← 文本条件 + 图像条件（双重引导）
```

效果：生成图会在保持提示词语义的同时，向参考图的外观风格靠拢（颜色、风格、人物外观）。

### LoRA 的作用

LoRA（Low-Rank Adaptation）是 U-Net 权重的**低秩微调增量**，以小文件（几十 MB）的形式附加在基础模型上：

```
基础模型权重 W + LoRA 增量 ΔW = 微调后的权重
```

当针对特定角色训练 LoRA 后，模型会"记住"该角色的外观，只需提示词中包含触发词（如 `hero_warrior`）即可召唤。

---

## 二、工作流节点详解

### 1. CheckpointLoaderSimple（基础模型加载）

```
输出：MODEL（U-Net）、CLIP（文本编码器）、VAE（图像编/解码器）
```

**作用**：加载 `.safetensors` 格式的基础模型（checkpoint），提取其中三个核心组件：
- **MODEL**：U-Net 去噪网络，是生成的"大脑"
- **CLIP**：文本编码器，将提示词转为向量
- **VAE**：变分自编码器，负责 Latent ↔ 图像的转换

**参数**：
- `ckpt_name`：模型文件名，需在 `ComfyUI/models/checkpoints/` 目录下

**选择指南**：
| 模型 | 适用场景 |
|---|---|
| `v1-5-pruned-emaonly-fp16.safetensors` | SD1.5 通用基底，快速验证 |
| Animagine XL 3.1 | 动漫风格角色/背景（SDXL） |
| Flat2D AnimeMix | 2D 图标、扁平插画（SD1.5） |

---

### 2. IPAdapterUnifiedLoader（IP-Adapter 统一加载器）

```
输入：MODEL（来自 CheckpointLoader）
输出：MODEL（已嵌入 IP-Adapter hook）、IPADAPTER（适配器权重对象）
```

**作用**：根据 `preset` 自动定位并加载对应的：
1. IP-Adapter 适配器权重（`models/ipadapter/` 目录）
2. CLIP Vision 图像编码器（`models/clip_vision/` 目录）

将两者注入 MODEL，后续 `IPAdapter` 节点用于实际的特征注入。

**参数 - preset（预设）**：

| preset | 适配器文件（正则匹配） | CLIP Vision 文件 | 特点 |
|---|---|---|---|
| `STANDARD (medium strength)` | `ip-adapter_sd15.safetensors` | `ipadapter_sd15.safetensors` | 平衡，默认推荐 |
| `PLUS (high strength)` | `ip-adapter-plus_sd15.safetensors` | `ipadapter_sd15.safetensors` | 更强外观贴合 |
| `PLUS FACE (portraits)` | `ip-adapter-plus-face_sd15.safetensors` | `ipadapter_sd15.safetensors` | 专注面部一致性 |
| `VIT-G (medium strength)` | `ip-adapter_sd15_vit-G.safetensors` | ViT-bigG-14 | 使用更大编码器 |

> ⚠️ **文件命名规则**：CLIP Vision 文件名必须匹配正则 `ipadapter.*sd15` 或 `ViT.H.14.*s32B.b79K`。
> 推荐命名：`ipadapter_sd15.safetensors`

---

### 3. LoraLoader（LoRA 加载器）

```
输入：MODEL、CLIP
输出：MODEL（加权叠加 LoRA）、CLIP（加权叠加 LoRA）
```

**作用**：将 LoRA 权重增量叠加到 MODEL 和 CLIP 上，改变模型对特定概念的"偏好"。

**参数**：
- `lora_name`：LoRA 文件名（`ComfyUI/models/loras/` 目录）
- `strength_model`：对 U-Net 的影响强度（0~1，推荐 0.6~0.9）
- `strength_clip`：对文本编码器的影响强度（通常与 model 保持一致）

> 💡 **暂无 LoRA？** 将两个 strength 都设为 `0`，节点会直接透传 MODEL 和 CLIP，不影响效果。

---

### 4. CLIPTextEncode（文本提示词编码）

```
输入：CLIP（文本编码器）
输出：CONDITIONING（条件向量）
```

**作用**：将文本提示词通过 CLIP 编码为向量，作为 U-Net 去噪的条件信号。工作流中有**两个**该节点：
- `Positive Prompt`：正向提示词，引导生成方向
- `Negative Prompt`：负向提示词，排除不想要的内容

> 详见第四章：[正向与负向提示词写法](#四正向与负向提示词写法)

---

### 5. EmptyLatentImage（空白潜空间）

```
输入：无
输出：LATENT（全噪声的潜空间）
```

**作用**：创建指定尺寸的纯噪声 Latent，作为去噪的起点。

**参数**：
- `width`、`height`：图像尺寸（像素）
- `batch_size`：同时生成的图像数量

**尺寸建议**：

| 用途 | 推荐尺寸 | 说明 |
|---|---|---|
| 角色立绘全身（竖版）| **512 × 1024** | SD1.5 全身图推荐尺寸，1:2 比例 |
| 场景背景（横版） | 768 × 512 | 横版场景 |
| UI 图标（方形） | 512 × 512 | 图标/道具 |
| 角色上半身特写 | 512 × 768 | 3:4 比例，构图紧凑 |

> ⚠️ **SD1.5 分辨率安全区（重要）**：
> - SD1.5 训练分辨率为 512×512，总像素数超过 ~786k（512×1536 以上）会严重崩坏
> - **全身图上限：512×1024**（524k 像素），超过此高度模型会重复生成角色
> - 如需更大尺寸，先以 512×1024 生成，再用 **Upscaler** 放大（4× → 2048×4096）
> - SDXL 对高尺寸更友好，可使用 768×1152 生成全身图

> 💡 **全身图被裁剪？三步修复：**
> 1. 分辨率改为 512×1024（不要用 512×768）
> 2. 正向提示词加：`(full body:1.5), whole body, head to toe, from far away, character sheet`
> 3. 负向提示词加：`cropped, out of frame, cut off, partial body, missing legs, missing feet`
> 4. IP-Adapter 参数：`start_at=0.2`，`weight_type="prompt is more important"`（防止参考图构图污染）

---

### 6. IPAdapter（IP-Adapter 特征注入）

```
输入：MODEL（来自 IPAdapterUnifiedLoader）、IPADAPTER、IMAGE（参考图）
输出：MODEL（已注入图像特征的 U-Net）
```

**作用**：将参考图的视觉特征通过 CLIP Vision 编码后，注入 U-Net 的 Cross-Attention 层，影响去噪过程中对外观的偏好。

**参数**：
- `weight`：注入强度（0~1，推荐 0.5~0.7）
  - 0.3 以下：轻微风格参考，提示词主导
  - 0.5~0.7：平衡，外观+提示词双重引导（**推荐**）
  - 0.9 以上：强烈贴合参考图，可能限制多样性
- `start_at`、`end_at`：在哪些去噪步骤中生效（0.0~1.0）
  - `start_at=0.0`（默认）：从第一步开始注入，参考图影响构图
  - **`start_at=0.2`（推荐全身图）**：前 20% 步骤由纯文本建立构图，避免参考图把"特写"构图污染全身图
- `weight_type`：注入方式
  - `standard`：均匀注入（默认）
  - `prompt is more important`：前期减弱 IP 影响，让提示词先建立结构
  - `style transfer`：偏向风格迁移

---

### 7. KSampler（核心采样器）

```
输入：MODEL、正向 CONDITIONING、负向 CONDITIONING、LATENT
输出：LATENT（去噪后）
```

**作用**：这是整个工作流的**核心**。它在 N 步内将纯噪声 Latent 逐步去噪，得到符合提示词条件的图像 Latent。

**参数详解**：

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `seed` | 任意整数 | 随机种子，**相同 seed 得到相同构图**（复现关键） |
| `steps` | 20 | 去噪步骤数。20 步是质量/速度最佳平衡；越高越精细但收益递减 |
| `cfg` | 7.0 | CFG Scale。7 是通用值；写实风格可降至 5~6；动漫风格 7~9 |
| `sampler_name` | `dpmpp_2m` | 采样算法。`dpmpp_2m` 是最常用的高质量快速算法 |
| `scheduler` | `karras` | 噪声调度方案。`karras` 配合 `dpmpp_2m` 效果最佳 |
| `denoise` | 1.0 | 去噪强度。1.0 = 从纯噪声开始（txt2img）；<1 = img2img 修改原图 |

**seed 的重要性**：
- **固定 seed** = 可复现的构图布局（同一角色不同动作时的背景/构图一致）
- **随机 seed**（-1）= 每次生成不同结果

---

### 8. VAEDecode（潜空间解码）

```
输入：LATENT（去噪后）、VAE
输出：IMAGE（像素图像）
```

**作用**：将去噪后的潜空间向量解码为实际像素图像（64×96 Latent → 512×768 PNG）。

> 不同 VAE 解码器会影响图像的色彩饱和度和细节锐度。SD1.5 原版 VAE 偏灰，可替换为 `vae-ft-mse-840000-ema-pruned.safetensors` 改善色彩。

---

### 9. LoadImage（加载参考图）

```
输入：无（从 ComfyUI/input/ 目录读取文件）
输出：IMAGE、MASK
```

**作用**：从 `ComfyUI/input/` 目录加载参考图，传给 IP-Adapter 作为外观参考。

**参数**：
- `image`：文件名（如 `hero_ref.png`），文件需预先放入 `ComfyUI/input/` 目录

---

### 10. SaveImage（保存图像）

```
输入：IMAGE
输出：无（写入文件）
```

**作用**：将生成的图像保存到 `ComfyUI/output/` 目录。

**参数**：
- `filename_prefix`：文件名前缀，ComfyUI 会自动追加序号（如 `character/hero_00001_.png`）
- 支持子目录：`filename_prefix` 中含 `/` 会自动创建子目录

---

## 三、工作流数据流全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                      base_character.json 工作流                   │
│                                                                   │
│  [CheckpointLoaderSimple]                                         │
│   ├─ MODEL ─────────► [IPAdapterUnifiedLoader]                   │
│   │                    ├─ MODEL ──► [LoraLoader]                 │
│   │                    │            ├─ MODEL ──────────────────┐  │
│   │                    │            └─ CLIP ─► [CLIPTextEncode] │  │
│   │                    └─ IPADAPTER ─────────► [IPAdapter]      │  │
│   ├─ CLIP ──────────────────────────────────► [CLIPTextEncode]  │  │
│   │                                            ├─ CONDITIONING(+)► │
│   └─ VAE ──────────────────────────────────────────────────┐    │  │
│                                                             │    │  │
│  [CLIPTextEncode (负向)]──── CONDITIONING(-)────────────┐  │    │  │
│                                                          │  │    │  │
│  [EmptyLatentImage]─────────────── LATENT ──────────┐   │  │    │  │
│                                                      │   │  │    │  │
│  [LoadImage]──► IMAGE ──► [IPAdapter]                │   │  │    │  │
│                            └─ MODEL ──► [KSampler] ◄─┤◄──┤  │    │  │
│                                         ├─ (MODEL)    │   │  │    │  │
│                                         │  (COND+)    │   │  │    │  │
│                                         │  (COND-)    │   │  │    │  │
│                                         │  (LATENT) ◄─┘   │  │    │  │
│                                         └─ LATENT ──► [VAEDecode]◄┘  │
│                                                       └─ IMAGE ► [SaveImage] │
└─────────────────────────────────────────────────────────────────┘
```

**简化流程（6步）**：

```
① 加载模型     CheckpointLoader → MODEL + CLIP + VAE
② 注入IP特征   IPAdapterUnifiedLoader + LoadImage → 图像条件 MODEL
③ 加载LoRA     LoraLoader → 角色专属 MODEL
④ 编码提示词   CLIPTextEncode × 2 → 正向/负向条件
⑤ 去噪生成     KSampler（N步扩散）→ 去噪 Latent
⑥ 解码输出     VAEDecode → 像素图像 → SaveImage
```

---

## 四、正向与负向提示词写法

### 提示词权重语法

ComfyUI 使用括号语法调整单个词的权重：

| 语法 | 含义 | 示例 |
|---|---|---|
| `(word:1.3)` | 增强权重到 1.3× | `(red hair:1.3)` |
| `(word:0.7)` | 降低权重到 0.7× | `(armor:0.7)` |
| `[word]` | 降低权重约 0.91× | `[background]` |
| `word, word` | 逗号分隔，等权重 | `1girl, warrior` |
| `(word)` | 增强约 1.1× | `(masterpiece)` |
| `((word))` | 增强约 1.21× | `((masterpiece))` |

### 正向提示词结构（推荐顺序）

```
[质量词] [画风词] [主体描述] [外观细节] [动作/姿态] [背景] [构图词]
```

**示例：角色立绘**

```
masterpiece, best quality, highres,
anime style, 2D game character,
1girl, hero warrior,
red hair, blue eyes, silver fantasy armor, white cape,
standing, arms crossed, confident pose,
white background, full body,
clean line art, flat shading
```

**常用质量词**（正向）：

| 分类 | 常用词 |
|---|---|
| 通用质量 | `masterpiece`, `best quality`, `highres`, `ultra-detailed` |
| 动漫风格 | `anime style`, `anime coloring`, `clean line art` |
| 2D 游戏 | `2D game character`, `game sprite`, `flat shading`, `cel shading` |
| 像素风 | `pixel art`, `16-bit`, `pixel perfect` |
| 构图 | `full body`, `upper body`, `portrait`, `white background` |

### 负向提示词写法

负向提示词告诉模型**不要**生成什么。

**通用基础负向词（建议所有工作流使用）**：

```
lowres, bad anatomy, bad hands, extra limbs, missing limbs,
deformed, ugly, blurry, watermark, signature, text,
worst quality, low quality, normal quality, jpeg artifacts
```

**角色立绘专用负向词**：

```
3D, realistic, photo, photography,
extra fingers, fused fingers, malformed hands,
disconnected limbs, floating limbs,
cropped, out of frame, poorly drawn face,
mutation, disfigured
```

**提示词权重技巧**：

1. **强调角色核心特征**：`(red hair:1.4), (blue eyes:1.3)` — 防止颜色被随机化
2. **固定背景**：`(white background:1.5), simple background` — 游戏资源通常需要抠图
3. **强化画风**：`(anime style:1.2), (2D game art:1.2)` — 防止偏向写实

### 提示词模板（游戏资源推荐）

**角色立绘模板**：
```
masterpiece, best quality, (anime style:1.2), (2D game character:1.1),
{角色描述：如 1girl, warrior, red hair, blue eyes},
{外观：如 silver armor, white cape},
{动作：如 standing, full body},
(white background:1.5), simple background,
clean line art, game sprite
```

**场景背景模板**：
```
masterpiece, best quality, (2D game background:1.2),
{场景：如 fantasy forest, ancient ruins},
{时间：如 daytime, golden hour},
no characters, environment only,
(game art style:1.2), painterly
```

**UI 图标模板**：
```
masterpiece, best quality, (game icon:1.3), (item card:1.2),
{物品：如 iron sword, magic potion},
(white background:1.5), transparent background,
(flat design:1.2), simple, clean
```

---

## 五、UI 界面使用步骤

### 前置条件

- ComfyUI 已启动：`cd ComfyUI && python main.py --port 8188`
- 浏览器打开：`http://127.0.0.1:8188`
- 参考图已放入：`ComfyUI/input/`（如 `hero_ref.png`）

### 操作步骤

**步骤 1：加载工作流**

将 `workflows/character/base_character.json` 直接拖入浏览器窗口，或点击左侧菜单 **Load** 选择文件。

**步骤 2：检查节点参数**

点击各节点，确认以下参数：

| 节点 | 参数 | 默认值 | 可能需要修改 |
|---|---|---|---|
| `CheckpointLoaderSimple` | ckpt_name | `v1-5-pruned-emaonly-fp16.safetensors` | 改为你的模型 |
| `IP-Adapter Unified Loader` | preset | `STANDARD (medium strength)` | 有 PLUS 模型则改 |
| `LoRA Loader` | lora_name / strength | `hero_lora_v1.safetensors` / 0.8 | 改为你的 LoRA；无 LoRA 设 strength=0 |
| `Positive Prompt` | text | 默认角色描述 | **必须修改**为你的角色 |
| `Negative Prompt` | text | 通用负向词 | 按需调整 |
| `EmptyLatentImage` | width/height | 512 × 768 | 按输出规格修改 |
| `Load Reference Image` | image | `hero_ref.png` | 改为你的参考图文件名 |
| `KSampler` | seed / steps / cfg | 42 / 20 / 7.0 | seed 固定或设 -1 |
| `SaveImage` | filename_prefix | `character/hero` | 按角色名修改 |

**步骤 3：运行生成**

点击右下角 **Queue Prompt**（或 `Ctrl+Enter`），等待生成完成。

**步骤 4：查看结果**

结果保存在 `ComfyUI/output/` 目录，也可在浏览器右上角 **History** 中查看。

### 调参建议

| 目标 | 调整方式 |
|---|---|
| 提高与参考图外观相似度 | 增大 IP-Adapter `weight`（0.7~0.9） |
| 提高提示词控制力 | 降低 IP-Adapter `weight`（0.3~0.5）；增大 CFG（8~9） |
| 稳定复现同一构图 | 固定 seed 值 |
| 探索多样结果 | seed 设为 -1（随机） |
| 减少肢体变形 | 增加负向词：`bad anatomy, extra limbs, deformed` |
| 更干净的背景 | 正向加 `(white background:1.5), simple background`；负向加 `complex background, gradient background` |

---

## 六、Python 脚本使用步骤

### 单张验证（`scripts/test_single.py`）

适合快速验证环境是否正常：

```bash
# 确保 ComfyUI 已启动，然后：
conda activate torch312
cd /path/to/comfyui_workflow

# 使用默认参数（SD1.5，512×768，seed=42）
python scripts/test_single.py

# 自定义参数
python scripts/test_single.py \
  --ckpt v1-5-pruned-emaonly-fp16.safetensors \
  --prompt "2D game character, male warrior, blue armor" \
  --seed 123 \
  --width 512 \
  --height 768

# 指定 ComfyUI 地址（远程机器）
python scripts/test_single.py --url http://192.168.1.100:8188
```

### 批量生成（`scripts/batch_generate.py`）

**步骤 1**：编辑 `scripts/config/characters.json`：

```json
[
  {
    "id": "hero_001",
    "name": "主角勇士",
    "workflow": "workflows/character/base_character.json",
    "ckpt": "v1-5-pruned-emaonly-fp16.safetensors",
    "lora": "hero_lora_v1.safetensors",
    "lora_weight": 0.8,
    "reference_image": "hero_ref.png",
    "base_prompt": "1girl, hero warrior, red hair, blue eyes, silver armor, white background, full body, anime style, masterpiece",
    "negative_prompt": "lowres, bad anatomy, bad hands, watermark, 3D, realistic",
    "seed": 42,
    "steps": 20,
    "cfg": 7.0,
    "width": 512,
    "height": 768
  }
]
```

**步骤 2**：运行批量生成：

```bash
# 生成所有角色
python scripts/batch_generate.py

# 只生成指定角色
python scripts/batch_generate.py --ids hero_001

# 指定 ComfyUI 地址
python scripts/batch_generate.py --url http://127.0.0.1:8188

# 增加超时时间（高分辨率/多步骤时）
python scripts/batch_generate.py --timeout 600
```

**步骤 3**：查看结果，在 `output/<character_id>/` 目录下。

### 代码调用（`scripts/utils/`）

```python
from scripts.utils import load_workflow, apply_config, run_workflow

# 加载工作流 JSON
wf = load_workflow("workflows/character/base_character.json")

# 通过节点 title 替换参数
wf = apply_config(wf, {
    "CheckpointLoaderSimple": {"param_index": 0, "value": "v1-5-pruned-emaonly-fp16.safetensors"},
    "Positive Prompt": {"param_index": 0, "value": "1girl, hero warrior, red hair"},
    "KSampler": {"param_index": 0, "value": 42},   # seed
})

# 提交并等待结果，自动下载到本地
paths = run_workflow(wf, save_dir="output/hero_001", output_prefix="hero")
print(paths)  # ['output/hero_001/hero_00001_.png', ...]
```

---

## 七、一致性生成技巧

### 技巧 1：固定种子 + 参考图

同一角色的所有变体（不同动作、不同场景），使用：
- **相同 seed**：保持整体构图布局相似
- **相同参考图**：IP-Adapter 保持外观一致
- **相同 LoRA**：角色特征稳定不漂移

### 技巧 2：提示词分层管理

将提示词拆分为「固定层」+ 「变量层」：

```python
BASE_PROMPT = "masterpiece, best quality, anime style, 2D game character, 1girl, hero warrior, red hair, blue eyes, silver armor, white background, simple background"

# 变量层：动作
ACTION_PROMPTS = {
    "idle":    "standing, relaxed pose, full body",
    "attack":  "action pose, sword raised, dynamic angle",
    "walk":    "walking, one foot forward, full body",
    "victory": "arms raised, cheering, happy expression",
}

# 组合
full_prompt = f"{BASE_PROMPT}, {ACTION_PROMPTS['attack']}"
```

### 技巧 3：IP-Adapter weight 渐进调优

```
weight=0.3  →  外观仅作参考，构图由提示词主导
weight=0.5  →  推荐起始值，外观+提示词平衡
weight=0.7  →  外观贴合参考图，适合角色一致性
weight=0.9  →  强烈贴合，但可能失去多样性
```

### 技巧 4：批量测试种子

运行 `test_single.py` 时用不同 seed 生成 5~10 张，从中挑选构图最好的 seed，之后固定使用：

```bash
for seed in 42 123 456 789 1024; do
  python scripts/test_single.py --seed $seed
done
```

### 技巧 5：LoRA 权重调节

LoRA 权重过高会导致角色外观"僵硬"，失去提示词的灵活控制：

```
weight=0.4~0.6  →  轻度风格引导
weight=0.7~0.8  →  推荐，角色识别度高+有弹性
weight=0.9~1.0  →  强角色 LoRA，适合写实角色
```

---

> 📖 延伸阅读：[LoRA 训练指南](./lora_training_guide.md) | [README](../README.md)
