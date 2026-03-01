# Copilot Instructions

## 项目目标

利用 ComfyUI 生成具备**跨图一致性**的 2D 游戏资源：角色立绘（Sprite）、2D 场景/背景、UI 图标与道具图。  
核心挑战：同一角色在不同动作/方向下外观、色调、风格必须统一。

## 目录结构

```
comfyui_workflow/
├── workflows/
│   ├── character/          # 角色相关工作流（立绘、多动作、Sprite Sheet）
│   ├── background/         # 场景/背景工作流（含视差分层）
│   └── ui/                 # 图标、道具、UI 元素工作流
├── scripts/
│   ├── utils/
│   │   ├── comfy_api.py        # ComfyUI HTTP/WebSocket API 封装
│   │   └── workflow_template.py # 工作流 JSON 参数替换工具
│   ├── config/
│   │   ├── characters.json     # 角色配置（参考图、LoRA、Prompt 模板）
│   │   └── items.json          # 道具配置
│   └── batch_generate.py       # 批量生成主脚本
├── references/
│   ├── characters/         # 角色参考图（用于 IP-Adapter 或 LoRA 训练）
│   └── style_samples/      # 风格参考图
└── output/                 # 生成结果（已 .gitignore）
```

## 一致性技术栈

项目同时使用以下技术叠加保证跨图一致性（优先级从高到低）：

1. **角色/风格 LoRA**（最稳定）  
   针对特定角色或美术风格训练的 LoRA，所有相关图片以固定权重加载同一 LoRA。  
   训练工具：kohya_ss，约需 20–50 张参考图。

2. **IP-Adapter**（快速外观对齐，无需训练）  
   将参考图外观特征注入采样过程。节点包：`ComfyUI_IPAdapter_plus`。  
   面部一致性使用 `IP-Adapter Face ID` 变体。

3. **ControlNet**（结构约束）  
   - OpenPose：角色不同动作保持骨骼比例一致  
   - Lineart / Canny：根据草图生成，保持轮廓一致  
   - Tile：高清放大时保持细节

4. **固定 Prompt 模板 + 种子管理**  
   每类资源维护基础 Prompt 模板，风格词固定，仅变更动作/场景关键词。

## 推荐模型配置

| 资源类型 | 推荐基模 | 备注 |
|---|---|---|
| 角色立绘（动漫/卡通风） | Animagine XL 3.1（SDXL） | Mac M4 / GTX 1080 均可跑 fp16 |
| 像素风 Sprite | Anything v5 或 Flat2D AnimeMix（SD1.5） | 1080 8GB 显存友好 |
| 场景背景 | BluePencil XL（SDXL） | 需开启 Tiled VAE |
| UI 图标/道具 | Flat2D AnimeMix（SD1.5） | 小尺寸批量生成首选 |

> **GTX 1080（8GB VRAM）**：SDXL 需 xformers + Tiled VAE；SD1.5 全工作流无压力。  
> **Mac M4**：使用 MPS 后端，避免使用 Flux（兼容性差且显存不足）。

## 工作流 JSON 约定

- 文件保存为 `.json`，顶层结构：`{ "nodes": [...], "links": [...], "groups": [...] }`
- 节点字段：`id`（整数，全局唯一，不复用）、`type`（节点类名）、`widgets_values`（参数值数组）
- `links` 格式：`[link_id, from_node_id, from_slot, to_node_id, to_slot, type]`
- **不提交含绝对路径的字段**（`ckpt_name`、`lora_name`、`image` 等），统一使用相对于 ComfyUI models 目录的路径
- 工作流中需要批量替换的参数，在文件名或注释中注明参数位置（见 `workflow_template.py`）

## 批量生成脚本约定

`scripts/utils/comfy_api.py` 提供：
- `queue_prompt(workflow, client_id)` → 返回 `prompt_id`
- `poll_until_done(prompt_id, timeout)` → 阻塞直到完成，返回 history 记录
- `download_outputs(history, save_dir)` → 下载所有输出图像

`scripts/utils/workflow_template.py` 提供：
- `load_workflow(path)` → 读取工作流 JSON
- `set_node_value(workflow, node_title, param_index, value)` → 按节点 title 定位并替换 widget 值
- `apply_config(workflow, config_dict)` → 批量应用配置字典

`scripts/config/characters.json` 每条记录格式：
```json
{
  "id": "hero_001",
  "name": "主角",
  "workflow": "workflows/character/base_character.json",
  "ckpt": "animagineXL31.safetensors",
  "lora": "hero_lora_v1.safetensors",
  "lora_weight": 0.8,
  "reference_image": "references/characters/hero_ref.png",
  "base_prompt": "1girl, hero, red hair, blue eyes, fantasy armor",
  "negative_prompt": "lowres, bad anatomy, watermark",
  "seed": 42
}
```

## 必需的 ComfyUI 自定义节点

| 节点包 | 用途 |
|---|---|
| `ComfyUI-Manager` | 节点管理，必装 |
| `ComfyUI_IPAdapter_plus` | IP-Adapter 外观一致性 |
| `comfyui_controlnet_aux` | ControlNet 预处理（OpenPose、Lineart 等） |
| `ComfyUI-Easy-Use` 或 `efficiency-nodes-comfyui` | 简化常用节点连接 |
| `ComfyUI-VideoHelperSuite` | 如需生成动画帧序列 |

## 模型下载规范

**所有模型下载必须使用中国镜像站**，禁止直连 huggingface.co（网络不通）。

### 优先镜像站

| 镜像站 | 地址 | 适用 |
|---|---|---|
| HF Mirror | `https://hf-mirror.com` | HuggingFace 全部模型 |
| ModelScope | `https://modelscope.cn` | 国内原生模型（Kolors、CogVideo 等） |

### 下载命令格式

```bash
# 单文件下载（必须加 -L 跟随重定向，--retry 3 自动重试）
curl -L --retry 3 \
  "https://hf-mirror.com/{owner}/{repo}/resolve/main/{filename}" \
  -o "/Users/zero/data/comfyui/ComfyUI/models/{type}/{filename}"

# 示例：下载 SDXL Base 1.0
curl -L --retry 3 \
  "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors" \
  -o "/Users/zero/data/comfyui/ComfyUI/models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors"
```

> 禁止使用 `wget`（Mac 未安装）；禁止直连 `huggingface.co`；禁止使用需科学上网的工具。

### 镜像路径映射

- 原始：`https://huggingface.co/{owner}/{repo}/resolve/main/{file}`
- 替换：`https://hf-mirror.com/{owner}/{repo}/resolve/main/{file}`

### 模型存放路径

| 模型类型 | ComfyUI 目录 |
|---|---|
| 大模型（checkpoint） | `models/checkpoints/` |
| LoRA | `models/loras/` |
| ControlNet | `models/controlnet/` |
| IP-Adapter | `models/ipadapter/` |
| VAE | `models/vae/` |
| CLIP Vision | `models/clip_vision/` |
| Diffusers 格式（Kolors 等） | `models/diffusers/{model_name}/` |

---

## 验证脚本规范

### 目录要求

**所有验证/测试脚本必须写入 `test/` 目录**（项目根目录下），禁止写入 `/tmp/` 或其他临时目录。

```
comfyui_workflow/
└── test/
    ├── test_character.py       # 角色立绘工作流验证
    ├── test_scene.py           # 场景工作流验证
    ├── test_api.py             # ComfyUI API 连通性验证
    └── README.md               # 测试说明
```

### 编写要求

1. **命名**：`test_{模块}.py`，可直接 `python test/test_xxx.py` 运行，无需参数
2. **输出格式**：使用 `✅` / `❌` / `⏰` 前缀，便于快速识别结果
3. **超时**：单张图最长 300s（SDXL），批量工作流（视差层等）最长 600s
4. **API 转换**：工作流为 UI 格式（`nodes[]`+`links[]`），提交前需用 `build_api_prompt()` 转为 API 格式
5. **验证内容**：检查输出文件名、用 PIL 读取尺寸、确认文件大小 >100KB
6. **依赖限制**：只使用标准库（`json`、`urllib.request`、`time`、`os`）和 `PIL`，禁止 `requests` 等需安装的包

### 验证脚本模板

```python
# test/test_xxx.py
import json, urllib.request, time, os
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "/Users/zero/data/comfyui/ComfyUI/output"

def build_api_prompt(wf_ui: dict) -> dict:
    """将 ComfyUI UI 格式工作流转换为 API prompt 格式"""
    prompt = {}
    node_outputs = {}
    widget_map = {
        "CheckpointLoaderSimple": ["ckpt_name"],
        "CLIPTextEncode": ["text"],
        "EmptyLatentImage": ["width", "height", "batch_size"],
        "KSampler": ["seed", "control_after_generate", "steps", "cfg",
                     "sampler_name", "scheduler", "denoise"],
        "SaveImage": ["filename_prefix"],
        "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    }
    for node in wf_ui["nodes"]:
        nid = str(node["id"])
        prompt[nid] = {"class_type": node["type"], "inputs": {}}
        for i, out in enumerate(node.get("outputs", [])):
            for link_id in out.get("links", []):
                node_outputs[link_id] = [nid, out.get("slot_index", i)]
    for node in wf_ui["nodes"]:
        nid = str(node["id"])
        for inp in node.get("inputs", []):
            if inp.get("link") is not None and inp["link"] in node_outputs:
                prompt[nid]["inputs"][inp["name"]] = node_outputs[inp["link"]]
        for k, v in zip(widget_map.get(node["type"], []), node.get("widgets_values", [])):
            prompt[nid]["inputs"][k] = v
    return prompt

def submit_workflow(wf_path: str) -> str | None:
    with open(wf_path) as f:
        prompt = build_api_prompt(json.load(f))
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt", data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if "error" in resp:
        print(f"❌ 提交失败: {resp['error']}")
        return None
    return resp["prompt_id"]

def wait_for_result(prompt_id: str, label: str, timeout: int = 300) -> bool:
    for i in range(timeout // 5):
        time.sleep(5)
        history = json.loads(
            urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}").read()
        )
        if prompt_id not in history:
            continue
        if history[prompt_id].get("status", {}).get("status_str") == "error":
            print(f"❌ {label}: 生成失败")
            return False
        images = [
            img["filename"]
            for out in history[prompt_id].get("outputs", {}).values()
            for img in out.get("images", [])
        ]
        if images:
            for fname in images:
                fpath = os.path.join(OUTPUT_DIR, fname)
                img = Image.open(fpath)
                size_kb = os.path.getsize(fpath) // 1024
                print(f"✅ {label}: {fname} {img.size[0]}x{img.size[1]} {size_kb}KB ({(i+1)*5}s)")
            return True
    print(f"⏰ {label}: 超时 {timeout}s")
    return False

if __name__ == "__main__":
    pid = submit_workflow("workflows/xxx/xxx.json")
    if pid:
        wait_for_result(pid, "xxx 工作流")
```

---

## 环境

- Python 3.12，conda 环境名：`torch312`
- 实际 Python 路径：`/Users/zero/miniconda3/envs/torch312/bin/python`
- pip 路径：`/Users/zero/miniconda3/envs/torch312/bin/pip`（**不要用系统 pip**）
- ComfyUI 路径：`/Users/zero/data/comfyui/ComfyUI`
- ComfyUI 启动命令：
  ```bash
  cd /Users/zero/data/comfyui/ComfyUI && \
  nohup /Users/zero/miniconda3/envs/torch312/bin/python main.py --port 8188 \
    > /tmp/comfyui.log 2>&1 &
  ```
- `output/` 目录已加入 `.gitignore`，生成结果不提交
- 进程管理：用 `ps aux | grep "python.*main.py" | grep -v grep` 找 PID，`kill <PID>` 停止（**禁用 `pkill`/`killall`**）
