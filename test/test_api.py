#!/usr/bin/env python3
"""
test/test_api.py
验证 ComfyUI API 基本连通性、模型可用性、节点注册情况
用法：python test/test_api.py
"""
import json, urllib.request, sys

COMFY_URL = "http://127.0.0.1:8188"

REQUIRED_CHECKPOINTS = [
    "sd_xl_base_1.0_0.9vae.safetensors",
]

REQUIRED_NODES = [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "EmptyLatentImage",
]


def check(label: str, ok: bool, detail: str = ""):
    icon = "✅" if ok else "❌"
    msg = f"{icon} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return ok


if __name__ == "__main__":
    results = []

    # 1. 连通性
    try:
        data = json.loads(urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=5).read())
        device = data.get("devices", [{}])[0].get("name", "unknown")
        results.append(check("ComfyUI 连通", True, f"设备: {device}"))
    except Exception as e:
        results.append(check("ComfyUI 连通", False, str(e)))
        print("\n❌ ComfyUI 未运行，后续检查跳过")
        print("启动命令：")
        print("  cd /Users/zero/data/comfyui/ComfyUI && \\")
        print("  nohup /Users/zero/miniconda3/envs/torch312/bin/python main.py --port 8188 > /tmp/comfyui.log 2>&1 &")
        sys.exit(1)

    # 2. 检查必需 checkpoint 模型
    ckpt_info = json.loads(urllib.request.urlopen(
        f"{COMFY_URL}/object_info/CheckpointLoaderSimple"
    ).read())
    available_ckpts = ckpt_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    for ckpt in REQUIRED_CHECKPOINTS:
        results.append(check(f"模型: {ckpt}", ckpt in available_ckpts))

    # 3. 检查必需节点
    all_nodes = json.loads(urllib.request.urlopen(f"{COMFY_URL}/object_info").read())
    for node in REQUIRED_NODES:
        results.append(check(f"节点: {node}", node in all_nodes))

    # 4. 汇总
    print("\n--- API 验证汇总 ---")
    passed = sum(1 for ok in results if ok)
    total = len(results)
    print(f"{'✅' if passed == total else '⚠️'} {passed}/{total} 项通过")
    if passed < total:
        sys.exit(1)
