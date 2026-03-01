"""
批量生成主脚本。

读取 scripts/config/characters.json，逐条加载工作流、替换参数、提交到 ComfyUI 队列，
并将结果图像保存到 output/<character_id>/ 目录。

用法：
    # 生成所有角色
    python scripts/batch_generate.py

    # 只生成指定角色
    python scripts/batch_generate.py --ids hero_001 mage_001

    # 指定 ComfyUI 地址
    python scripts/batch_generate.py --url http://192.168.1.100:8188
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便直接运行此脚本
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import scripts.utils.comfy_api as api
import scripts.utils.workflow_template as tmpl


def build_config(char: dict) -> dict:
    """将角色配置 dict 转换为 apply_config 所需的格式。"""
    return {
        "CheckpointLoaderSimple": {"param_index": 0, "value": char["ckpt"]},
        "LoRA Loader": [
            {"param_index": 1, "value": char["lora"]},
            {"param_index": 2, "value": char["lora_weight"]},
            {"param_index": 3, "value": char["lora_weight"]},
        ],
        "Positive Prompt": {"param_index": 0, "value": char["base_prompt"]},
        "Negative Prompt": {"param_index": 0, "value": char["negative_prompt"]},
        "KSampler": [
            {"param_index": 0, "value": char["seed"]},
            {"param_index": 4, "value": char.get("steps", 20)},
            {"param_index": 5, "value": char.get("cfg", 7.0)},
        ],
        "Load Reference Image": {"param_index": 0, "value": char["reference_image"]},
    }


def generate_character(char: dict, output_root: Path, timeout: int = 300) -> None:
    print(f"\n→ 生成角色：{char['name']} ({char['id']})")

    workflow_path = ROOT / char["workflow"]
    if not workflow_path.exists():
        print(f"  [跳过] 工作流文件不存在：{workflow_path}")
        return

    workflow = tmpl.load_workflow(workflow_path)
    workflow = tmpl.apply_config(workflow, build_config(char))

    save_dir = output_root / char["id"]
    api.run_workflow(workflow, save_dir=save_dir,
                     output_prefix=char["id"], timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI 批量角色生成脚本")
    parser.add_argument("--ids", nargs="*", help="只生成指定 id 的角色，默认全部")
    parser.add_argument("--url", default="http://127.0.0.1:8188",
                        help="ComfyUI 服务地址")
    parser.add_argument("--timeout", type=int, default=300,
                        help="单个工作流超时秒数")
    args = parser.parse_args()

    # 覆盖 API 模块中的地址
    api.COMFYUI_URL = args.url

    config_path = ROOT / "scripts" / "config" / "characters.json"
    characters = json.loads(config_path.read_text(encoding="utf-8"))

    if args.ids:
        characters = [c for c in characters if c["id"] in args.ids]
        if not characters:
            print("未找到匹配的角色 id，退出。")
            return

    output_root = ROOT / "output"
    for char in characters:
        try:
            generate_character(char, output_root, timeout=args.timeout)
        except Exception as e:
            print(f"  [错误] {char['id']}: {e}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
