"""
工作流 JSON 参数替换工具。

ComfyUI 工作流 JSON 的节点参数存储在 widgets_values 数组中，
本模块通过节点 title 定位节点，按参数索引替换对应值，
避免手动编辑 JSON 时误改节点结构。
"""

import json
import copy
from pathlib import Path
from typing import Any


def load_workflow(path: str | Path) -> dict:
    """加载工作流 JSON 文件，返回 dict。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_workflow(workflow: dict, path: str | Path) -> None:
    """将工作流 dict 写入 JSON 文件。"""
    Path(path).write_text(json.dumps(workflow, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def find_node_by_title(workflow: dict, title: str) -> dict | None:
    """按节点 title 查找节点，找不到返回 None。"""
    for node in workflow.get("nodes", []):
        if node.get("title") == title:
            return node
    return None


def find_node_by_type(workflow: dict, node_type: str) -> dict | None:
    """按节点 type（类名）查找第一个匹配节点，找不到返回 None。"""
    for node in workflow.get("nodes", []):
        if node.get("type") == node_type:
            return node
    return None


def set_node_value(workflow: dict, title: str, param_index: int, value: Any) -> None:
    """
    按节点 title 定位节点，将 widgets_values[param_index] 设置为 value。
    找不到节点时抛出 KeyError。
    """
    node = find_node_by_title(workflow, title)
    if node is None:
        raise KeyError(f"找不到 title='{title}' 的节点")
    widgets = node.setdefault("widgets_values", [])
    # 按需扩展列表
    while len(widgets) <= param_index:
        widgets.append(None)
    widgets[param_index] = value


def apply_config(workflow: dict, config: dict) -> dict:
    """
    批量应用配置字典到工作流（深拷贝，不修改原始 workflow）。

    config 格式示例：
    {
        "CheckpointLoaderSimple": {"param_index": 0, "value": "animagineXL31.safetensors"},
        "LoRA Loader":            {"param_index": 1, "value": "hero_lora_v1.safetensors"},
        "Positive Prompt":        {"param_index": 0, "value": "1girl, hero, red hair"},
        "Negative Prompt":        {"param_index": 0, "value": "lowres, bad anatomy"},
        "KSampler":               [
            {"param_index": 0, "value": 42},    # seed
            {"param_index": 4, "value": 20},    # steps
        ],
    }

    键为节点 title；值可以是单个 dict（单参数）或 list of dict（多参数）。
    """
    wf = copy.deepcopy(workflow)
    for title, spec in config.items():
        specs = spec if isinstance(spec, list) else [spec]
        for item in specs:
            set_node_value(wf, title, item["param_index"], item["value"])
    return wf


def set_checkpoint(workflow: dict, ckpt_name: str,
                   node_title: str = "CheckpointLoaderSimple") -> None:
    """快捷方法：设置 CheckpointLoader 的模型文件名（param_index=0）。"""
    set_node_value(workflow, node_title, 0, ckpt_name)


def set_lora(workflow: dict, lora_name: str, lora_weight: float = 0.8,
             node_title: str = "LoRA Loader") -> None:
    """快捷方法：设置 LoRA 文件名（param_index=1）和权重（param_index=2、3）。"""
    set_node_value(workflow, node_title, 1, lora_name)
    set_node_value(workflow, node_title, 2, lora_weight)  # model_weight
    set_node_value(workflow, node_title, 3, lora_weight)  # clip_weight


def set_prompts(workflow: dict, positive: str, negative: str,
                pos_title: str = "Positive Prompt",
                neg_title: str = "Negative Prompt") -> None:
    """快捷方法：设置正向/负向提示词（param_index=0）。"""
    set_node_value(workflow, pos_title, 0, positive)
    set_node_value(workflow, neg_title, 0, negative)


def set_seed(workflow: dict, seed: int,
             node_title: str = "KSampler") -> None:
    """快捷方法：设置 KSampler 种子（param_index=0）。"""
    set_node_value(workflow, node_title, 0, seed)


def set_reference_image(workflow: dict, image_path: str,
                         node_title: str = "Load Reference Image") -> None:
    """快捷方法：设置 IP-Adapter 参考图路径（param_index=0）。"""
    set_node_value(workflow, node_title, 0, image_path)
