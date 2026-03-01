# ComfyUI 2D 游戏资源一致性生成工作流

利用 ComfyUI + IP-Adapter + LoRA + ControlNet，批量生成**风格统一**的 2D 游戏资源（角色立绘、场景背景、UI 图标）。

> ✅ **已验证环境**：Mac M4 + macOS + Miniconda（torch312 环境）+ ComfyUI 0.9.2 + MPS 加速

---

## 目录

- [本地安装 ComfyUI](#一本地安装-comfyui)
- [安装必需自定义节点](#二安装必需自定义节点)
- [下载模型文件](#三下载模型文件)
- [验证单张图生成](#四验证单张图生成)
- [在 ComfyUI 界面使用工作流](#五在-comfyui-界面使用工作流)
- [使用 Python 脚本批量生成](#六使用-python-脚本批量生成)
- [项目结构](#项目结构)

---

## 一、本地安装 ComfyUI

### Mac（Apple Silicon，M 系列）

**推荐使用 Conda 管理环境**（与系统 Python 隔离）：

```bash
# 1. 安装 Miniconda（如已安装跳过）
brew install --cask miniconda

# 2. 创建专用环境（Python 3.12 + PyTorch for MPS）
conda create -n torch312 python=3.12 -y
conda activate torch312
pip install torch torchvision torchaudio   # 自动选择 MPS 支持版本

# 3. 克隆 ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# 4. 启动（MPS 自动识别，无需额外参数）
python main.py --port 8188
```

浏览器访问 `http://127.0.0.1:8188`

> **注意**：新版 ComfyUI（0.9+）已移除 `--use-mps-device` 参数，直接 `python main.py` 即可自动使用 MPS。

### Windows / Linux（NVIDIA GPU，GTX 1080 等）

```bash
# 1. 创建 Conda 环境
conda create -n comfyui python=3.12 -y
conda activate comfyui

# 2. 安装 PyTorch（CUDA 11.8 示例，按实际 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 克隆并安装 ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# 4. 启动（GTX 1080 建议加 --lowvram 节省显存）
python main.py --lowvram
```

> **GTX 1080（8GB VRAM）提示**：运行 SDXL 工作流时追加 `--lowvram`；若仍 OOM，改用 SD1.5 系列工作流。

---

## 二、安装必需自定义节点

```bash
cd ComfyUI/custom_nodes

# 1. ComfyUI-Manager（节点管理，必装）
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
pip install -r ComfyUI-Manager/requirements.txt

# 2. IP-Adapter（外观一致性核心，无额外依赖）
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# 3. （可选）ControlNet 预处理（Pose、Lineart 等）
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
pip install -r comfyui_controlnet_aux/requirements.txt
```

重启 ComfyUI 生效。启动日志中出现以下内容说明节点加载成功：

```
Import times for custom nodes:
   0.0 seconds: ComfyUI_IPAdapter_plus
   0.1 seconds: ComfyUI-Manager
```

> 也可以在 ComfyUI 界面点击右下角 **Manager → Install Missing Custom Nodes** 自动安装工作流缺少的节点。

---

## 三、下载模型文件

### 基础模型（checkpoint）

将 `.safetensors` 文件放入 `ComfyUI/models/checkpoints/`：

| 模型 | 用途 | 下载 |
|---|---|---|
| `v1-5-pruned-emaonly-fp16.safetensors` | SD1.5 基础，快速验证 | [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| Animagine XL 3.1 | 动漫风角色/背景（SDXL） | [Hugging Face](https://huggingface.co/cagliostrolab/animagine-xl-3.1) |
| Flat2D AnimeMix | 2D 图标/道具（SD1.5） | [Civitai](https://civitai.com/models/35960) |

### IP-Adapter 模型（一致性核心）

使用 `huggingface_hub` 下载（自动处理断点续传）。中国用户可通过 `HF_ENDPOINT=https://hf-mirror.com` 加速：

```python
import os, shutil
# 中国用户取消下一行注释
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

COMFYUI = "/path/to/ComfyUI"   # 替换为你的 ComfyUI 路径

# IP-Adapter SD1.5 适配器权重（~43MB）
# 文件名必须为 ip-adapter_sd15.safetensors（匹配 STANDARD preset 的正则）
path = hf_hub_download(repo_id="h94/IP-Adapter",
                       filename="models/ip-adapter_sd15.safetensors",
                       local_dir="/tmp/ipadapter")
shutil.copy2(path, f"{COMFYUI}/models/ipadapter/ip-adapter_sd15.safetensors")

# CLIP Vision ViT-H（~2.3GB，图像编码器）
# 文件名必须含 "ipadapter" 和 "sd15"（如 ipadapter_sd15.safetensors）才能被 IPAdapterUnifiedLoader 识别
path = hf_hub_download(repo_id="h94/IP-Adapter",
                       filename="models/image_encoder/model.safetensors",
                       local_dir="/tmp/ipadapter_hf")
shutil.copy2(path, f"{COMFYUI}/models/clip_vision/ipadapter_sd15.safetensors")
```

> ⚠️ **注意文件命名规则**（IPAdapter_plus 通过正则自动匹配）：
> - `models/ipadapter/` 中的文件名需匹配 `ip.adapter.sd15` → 使用 `ip-adapter_sd15.safetensors`
> - `models/clip_vision/` 中的文件名需匹配 `ipadapter.*sd15` 或 `ViT.H.14.*s32B.b79K` → 使用 `ipadapter_sd15.safetensors`

### LoRA（可选，效果最佳）

角色专属 LoRA 放入 `ComfyUI/models/loras/`。若暂无 LoRA，将 `characters.json` 中 `lora_weight` 设为 `0` 即可跳过。

---

## 四、验证单张图生成

ComfyUI 启动后，运行以下脚本验证环境是否正常（仅需 SD1.5 checkpoint，无需 IP-Adapter）：

```bash
# 在本项目根目录运行
python scripts/test_single.py
```

或直接用 Python：

```python
import json, urllib.request, urllib.parse, time

URL = "http://127.0.0.1:8188"

workflow = {
  "1": {"class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"}},
  "2": {"class_type": "CLIPTextEncode",
        "inputs": {"text": "2D game character, female warrior, anime style", "clip": ["1", 1]}},
  "3": {"class_type": "CLIPTextEncode",
        "inputs": {"text": "lowres, bad anatomy", "clip": ["1", 1]}},
  "4": {"class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 768, "batch_size": 1}},
  "5": {"class_type": "KSampler",
        "inputs": {"seed": 42, "steps": 20, "cfg": 7.0,
                   "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                   "model": ["1",0], "positive": ["2",0], "negative": ["3",0], "latent_image": ["4",0]}},
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5",0], "vae": ["1",2]}},
  "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test_character", "images": ["6",0]}}
}

data = json.dumps({"prompt": workflow}).encode()
req = urllib.request.Request(f"{URL}/prompt", data=data, headers={"Content-Type": "application/json"})
prompt_id = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
print(f"提交成功，prompt_id: {prompt_id}，等待生成（约 30-120s）...")

for _ in range(60):
    time.sleep(3)
    history = json.loads(urllib.request.urlopen(
        f"{URL}/history/{urllib.parse.quote(prompt_id)}").read())
    if prompt_id in history:
        for out in history[prompt_id]["outputs"].values():
            for img in out.get("images", []):
                print(f"✅ 生成成功：ComfyUI/output/{img['filename']}")
        break
```

> ✅ **已验证**：Mac M4 + MPS + SD1.5，约 **30s** 生成一张 512×768 图像。

---

## 五、在 ComfyUI 界面使用工作流（含 IP-Adapter）

### 加载工作流

1. 打开 `http://127.0.0.1:8188`
2. 将本项目 `workflows/character/base_character.json` **直接拖入浏览器窗口**，或点击 **Load** 按钮选择文件

### 调整参数

加载后，依次检查并修改以下节点（节点右上角显示 title）：

| 节点 title | 需要修改的参数 | 说明 |
|---|---|---|
| `CheckpointLoaderSimple` | 模型文件名 | 选择你已下载的 checkpoint |
| `IP-Adapter Unified Loader` | preset | `STANDARD` 配 `ip-adapter_sd15.safetensors`；`PLUS` 需下载 plus 版本 |
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

## 六、使用 Python 脚本批量生成

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
