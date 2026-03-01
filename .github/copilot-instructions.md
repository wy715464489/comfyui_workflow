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

## 环境

- Python 3.10+，与 ComfyUI 共用同一虚拟环境或 conda 环境
- 新增依赖后执行 `pip install -r requirements.txt`
- `output/` 目录已加入 `.gitignore`，生成结果不提交
