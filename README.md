# ComfyUI 2D 游戏资源一致性生成工作流

利用 ComfyUI + IP-Adapter + LoRA + ControlNet，批量生成**风格统一**的 2D 游戏资源（角色立绘、场景背景、UI 图标）。

---

## 目录

- [本地安装 ComfyUI](#一本地安装-comfyui)
- [安装必需自定义节点](#二安装必需自定义节点)
- [下载模型文件](#三下载模型文件)
- [在 ComfyUI 界面使用工作流](#四在-comfyui-界面使用工作流)
- [使用 Python 脚本批量生成](#五使用-python-脚本批量生成)
- [项目结构](#项目结构)

---

## 一、本地安装 ComfyUI

### Mac（Apple Silicon M 系列）

```bash
# 1. 克隆 ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖（MPS 后端，无需 CUDA）
pip install torch torchvision torchaudio
pip install -r requirements.txt

# 4. 启动（MPS 加速）
python main.py --use-mps-device
```

浏览器访问 `http://127.0.0.1:8188`

### Windows / Linux（NVIDIA GPU，GTX 1080 等）

```bash
# 1. 克隆 ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. 创建虚拟环境
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# 3. 安装 PyTorch（CUDA 11.8 示例，按实际 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装依赖
pip install -r requirements.txt

# 5. 启动（GTX 1080 建议加 --lowvram 节省显存）
python main.py --lowvram
```

> **GTX 1080（8GB VRAM）提示**：运行 SDXL 工作流时追加 `--lowvram`；若仍 OOM，改用 SD1.5 系列工作流。

---

## 二、安装必需自定义节点

### 方法 A：通过 ComfyUI-Manager（推荐）

1. 进入 ComfyUI `custom_nodes` 目录，克隆 Manager：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   ```
2. 重启 ComfyUI，界面右下角出现 **Manager** 按钮。
3. 点击 Manager → **Install Missing Custom Nodes**，自动检测并安装工作流所需节点。

### 方法 B：手动克隆

```bash
cd ComfyUI/custom_nodes

# IP-Adapter（外观一致性核心）
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# ControlNet 预处理（Pose、Lineart 等）
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

# 安装各节点的 Python 依赖
pip install -r ComfyUI_IPAdapter_plus/requirements.txt
pip install -r comfyui_controlnet_aux/requirements.txt
```

重启 ComfyUI 生效。

---

## 三、下载模型文件

将模型放到 ComfyUI 对应目录（`ComfyUI/models/`）：

| 模型 | 目录 | 推荐来源 |
|---|---|---|
| **Animagine XL 3.1**（角色/背景） | `models/checkpoints/` | [Hugging Face](https://huggingface.co/cagliostrolab/animagine-xl-3.1) |
| **Flat2D AnimeMix**（图标/像素） | `models/checkpoints/` | [Civitai](https://civitai.com/models/35960) |
| **IP-Adapter Plus SDXL** | `models/ipadapter/` | [Hugging Face](https://huggingface.co/h94/IP-Adapter) → `sdxl/` |
| **CLIP Vision ViT-H** | `models/clip_vision/` | 同上，`models/image_encoder/` 下的 `model.safetensors` |
| **ControlNet OpenPose XL** | `models/controlnet/` | [Hugging Face](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0) |

> **IP-Adapter 文件对照**：
> - `ip-adapter-plus_sdxl_vit-h.safetensors` → `models/ipadapter/`
> - `clip_vision_g.safetensors`（ViT-bigG）→ `models/clip_vision/`

角色专属 LoRA（可选，效果最佳）放入 `models/loras/`。若暂无 LoRA，可将 `characters.json` 中 `lora` 字段留空，工作流会跳过 LoRA 节点。

---

## 四、在 ComfyUI 界面使用工作流

### 加载工作流

1. 打开 `http://127.0.0.1:8188`
2. 将本项目 `workflows/character/base_character.json` **直接拖入浏览器窗口**，或点击 **Load** 按钮选择文件

### 调整参数

加载后，依次检查并修改以下节点（节点右上角显示 title）：

| 节点 title | 需要修改的参数 | 说明 |
|---|---|---|
| `CheckpointLoaderSimple` | 模型文件名 | 选择你已下载的 checkpoint |
| `LoRA Loader` | LoRA 文件名、权重（0~1） | 没有 LoRA 可将权重设为 0 |
| `Positive Prompt` | 正向提示词 | 描述角色外观、动作、风格 |
| `Negative Prompt` | 负向提示词 | 排除不想要的元素 |
| `KSampler` | seed（种子）、steps | 固定 seed 可复现结果 |
| `Load Reference Image` | 参考图文件名 | 将参考图放入 `ComfyUI/input/` 目录 |
| `IP-Adapter` | weight（建议 0.5~0.7） | 越高越贴近参考图外观 |

### 放置参考图

将角色参考图复制到 **`ComfyUI/input/`** 目录，然后在 `Load Reference Image` 节点输入文件名（如 `hero_ref.png`）。

### 生成

点击 **Queue Prompt**（或 `Ctrl+Enter`）开始生成，结果保存至 `ComfyUI/output/`。

---

## 五、使用 Python 脚本批量生成

适用于需要批量生成多个角色/多个动作变体的场景。

### 前置条件

- ComfyUI 已在本地启动（默认 `http://127.0.0.1:8188`）
- 本项目与 ComfyUI 使用**同一 Python 环境**，或已单独安装依赖（无额外第三方依赖，仅用标准库）

### 1. 配置角色

编辑 `scripts/config/characters.json`，每条记录对应一个角色：

```json
{
  "id": "hero_001",
  "name": "主角",
  "workflow": "workflows/character/base_character.json",
  "ckpt": "animagineXL31.safetensors",
  "lora": "hero_lora_v1.safetensors",
  "lora_weight": 0.8,
  "reference_image": "hero_ref.png",
  "base_prompt": "1girl, hero, red hair, blue eyes, fantasy armor, white background, full body",
  "negative_prompt": "lowres, bad anatomy, watermark",
  "seed": 42,
  "steps": 20,
  "cfg": 7.0,
  "width": 768,
  "height": 1024
}
```

> `reference_image` 填写放在 `ComfyUI/input/` 目录下的文件名。

### 2. 运行批量生成

```bash
# 进入本项目根目录
cd /path/to/comfyui_workflow

# 生成所有角色
python scripts/batch_generate.py

# 只生成指定角色（按 id）
python scripts/batch_generate.py --ids hero_001

# 指定 ComfyUI 地址（远程机器）
python scripts/batch_generate.py --url http://192.168.1.100:8188

# 设置超时（秒）
python scripts/batch_generate.py --timeout 600
```

生成结果保存在 `output/<character_id>/` 目录下。

### 3. 在代码中调用

```python
from scripts.utils import load_workflow, apply_config, run_workflow

# 加载工作流模板
wf = load_workflow("workflows/character/base_character.json")

# 替换参数
wf = apply_config(wf, {
    "CheckpointLoaderSimple": {"param_index": 0, "value": "animagineXL31.safetensors"},
    "Positive Prompt":        {"param_index": 0, "value": "1girl, hero, red hair"},
    "KSampler":               {"param_index": 0, "value": 42},   # seed
})

# 提交并下载结果
paths = run_workflow(wf, save_dir="output/hero_001", output_prefix="hero")
print(paths)  # ['output/hero_001/hero_ComfyUI_00001_.png', ...]
```

---

## 项目结构

```
comfyui_workflow/
├── workflows/
│   ├── character/
│   │   └── base_character.json   # 角色立绘：IP-Adapter + LoRA + 二次精修
│   ├── background/               # 场景背景工作流（待添加）
│   └── ui/                       # UI 图标工作流（待添加）
├── scripts/
│   ├── utils/
│   │   ├── comfy_api.py          # ComfyUI API 封装
│   │   └── workflow_template.py  # 工作流参数替换工具
│   ├── config/
│   │   └── characters.json       # 角色配置
│   └── batch_generate.py         # 批量生成入口
├── references/
│   ├── characters/               # 角色参考图（用于 IP-Adapter / LoRA 训练）
│   └── style_samples/            # 风格参考图
├── output/                       # 生成结果（已 .gitignore）
└── .github/
    └── copilot-instructions.md   # Copilot 上下文说明
```

---

## 一致性原理说明

| 技术 | 作用 | 何时使用 |
|---|---|---|
| **角色 LoRA** | 最强一致性，训练后固定角色外观 | 已有 20–50 张角色参考图时 |
| **IP-Adapter** | 无需训练，快速注入参考图外观 | 快速原型 / 参考图少于 20 张 |
| **ControlNet Pose** | 固定骨骼结构，换动作不变形 | 角色多动作 Sprite Sheet |
| **固定 seed + Prompt 模板** | 最简单，但跨图一致性有限 | 同一张图的细节变体 |

两种推荐组合：
- **快速出图**：IP-Adapter（weight 0.6）+ 固定 seed + 风格 LoRA
- **高质量批量**：角色专属 LoRA + IP-Adapter Face ID + ControlNet Pose
