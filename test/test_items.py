#!/usr/bin/env python3
"""
test/test_items.py
验证道具/宝石/装备工作流（道具图标、宝石图标、武器、防具）
用法：python test/test_items.py
"""
import json, urllib.request, time, os
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "/Users/zero/data/comfyui/ComfyUI/output"

# (工作流路径, 显示标签, 超时秒, 期望尺寸)
WORKFLOWS = [
    ("workflows/ui/item_icon.json",       "道具图标 (1024×1024)",  300, (1024, 1024)),
    ("workflows/ui/gem_icon.json",        "宝石图标 (1024×1024)",  300, (1024, 1024)),
    ("workflows/ui/equipment_weapon.json","武器装备 (1024×1024)",  300, (1024, 1024)),
    ("workflows/ui/equipment_armor.json", "防具装备 (1024×1024)",  300, (1024, 1024)),
]


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
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None
    if "error" in resp:
        print(f"❌ 提交失败: {resp['error']}")
        if "node_errors" in resp:
            for nid, err in resp["node_errors"].items():
                print(f"   节点 {nid}: {err}")
        return None
    return resp["prompt_id"]


def wait_for_result(prompt_id: str, label: str,
                    timeout: int = 300, expected_size: tuple = None) -> bool:
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
            ok = True
            for fname in images:
                fpath = os.path.join(OUTPUT_DIR, fname)
                img = Image.open(fpath)
                size_kb = os.path.getsize(fpath) // 1024
                # 验证尺寸
                if expected_size and img.size != expected_size:
                    print(f"⚠️  {label}: 尺寸不符 {img.size} (期望 {expected_size})")
                    ok = False
                # 验证文件大小
                if size_kb < 100:
                    print(f"⚠️  {label}: 文件过小 {size_kb}KB")
                    ok = False
                if ok:
                    print(f"✅ {label}: {fname} {img.size[0]}x{img.size[1]} {size_kb}KB ({(i+1)*5}s)")
            return ok
    print(f"⏰ {label}: 超时 {timeout}s")
    return False


if __name__ == "__main__":
    # 检查 ComfyUI 是否运行
    try:
        urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3)
    except Exception:
        print("❌ ComfyUI 未运行，请先启动：")
        print("   cd /Users/zero/data/comfyui/ComfyUI && \\")
        print("   nohup /Users/zero/miniconda3/envs/torch312/bin/python main.py --port 8188 > /tmp/comfyui.log 2>&1 &")
        exit(1)

    results = []
    for wf_path, label, timeout, expected_size in WORKFLOWS:
        print(f"\n📤 提交: {label}")
        pid = submit_workflow(wf_path)
        if pid:
            ok = wait_for_result(pid, label, timeout, expected_size)
            results.append((label, ok))
        else:
            results.append((label, False))

    print("\n--- 验证汇总 ---")
    for label, ok in results:
        print(f"{'✅' if ok else '❌'} {label}")
    passed = sum(1 for _, ok in results if ok)
    if passed == len(results):
        print(f"\n✅ 所有道具/装备工作流验证通过 ({passed}/{len(results)})")
    else:
        print(f"\n❌ {len(results)-passed} 项失败，请检查上方日志")
        exit(1)
