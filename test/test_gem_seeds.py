#!/usr/bin/env python3
"""
test/test_gem_seeds.py
批量测试不同 seed 对宝石单一生成的效果，找到稳定生成单颗宝石的 seed。
用法：python test/test_gem_seeds.py
"""
import json, urllib.request, time, os, copy
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "/Users/zero/data/comfyui/ComfyUI/output"
GEM_WF = "workflows/ui/gem_icon.json"

# 测试的 seed 列表
TEST_SEEDS = [42, 77, 256, 512, 888]


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


def submit_with_seed(wf_path: str, seed: int) -> str | None:
    """提交工作流，覆盖 seed"""
    with open(wf_path) as f:
        wf_ui = json.load(f)
    # 找到 KSampler 节点并覆盖 seed
    for node in wf_ui["nodes"]:
        if node["type"] == "KSampler":
            node["widgets_values"][0] = seed
            node["widgets_values"][1] = "fixed"
    prompt = build_api_prompt(wf_ui)
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt", data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None
    if "error" in resp:
        print(f"❌ 提交失败: {resp['error']}")
        return None
    return resp["prompt_id"]


def wait_for_image(prompt_id: str, label: str, timeout: int = 300) -> str | None:
    """等待生成完成，返回文件路径"""
    for _ in range(timeout // 5):
        time.sleep(5)
        history = json.loads(
            urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}").read()
        )
        if prompt_id not in history:
            continue
        if history[prompt_id].get("status", {}).get("status_str") == "error":
            print(f"❌ {label}: 生成失败")
            return None
        images = [
            img["filename"]
            for out in history[prompt_id].get("outputs", {}).values()
            for img in out.get("images", [])
        ]
        if images:
            return os.path.join(OUTPUT_DIR, images[0])
    print(f"⏰ {label}: 超时")
    return None


if __name__ == "__main__":
    # 检查 ComfyUI
    try:
        urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3)
    except Exception:
        print("❌ ComfyUI 未运行")
        exit(1)

    print(f"🔍 批量测试宝石工作流 seeds: {TEST_SEEDS}\n")
    results = []

    for seed in TEST_SEEDS:
        label = f"gem seed={seed}"
        print(f"📤 提交: {label}")
        pid = submit_with_seed(GEM_WF, seed)
        if not pid:
            results.append((seed, None))
            continue
        fpath = wait_for_image(pid, label)
        if fpath:
            img = Image.open(fpath)
            size_kb = os.path.getsize(fpath) // 1024
            fname = os.path.basename(fpath)
            print(f"  ✅ {fname} {img.size[0]}x{img.size[1]} {size_kb}KB → {fpath}")
            results.append((seed, fpath))
        else:
            results.append((seed, None))

    print("\n--- 汇总（请目视检查 output 目录中的图片，选择只出现一颗宝石的 seed）---")
    for seed, fpath in results:
        status = f"→ {os.path.basename(fpath)}" if fpath else "❌ 失败"
        print(f"  seed={seed}: {status}")

    valid_seeds = [s for s, f in results if f]
    if valid_seeds:
        print(f"\n建议将 gem_icon.json 的 seed 更新为上述列表中视觉效果最好的值。")
        print(f"目视检查路径: {OUTPUT_DIR}/gem_icon_*.png")
