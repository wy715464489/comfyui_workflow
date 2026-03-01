#!/usr/bin/env python3
"""
test/test_scene.py
验证场景背景工作流（室外 + 室内 + 视差三层）
用法：python test/test_scene.py
"""
import json, urllib.request, time, os
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "/Users/zero/data/comfyui/ComfyUI/output"

# 需要验证的工作流（path, label, timeout秒）
WORKFLOWS = [
    ("workflows/background/scene_outdoor.json",        "室外场景 (1216×832)",      300),
    ("workflows/background/scene_indoor.json",         "室内/地牢 (1216×832)",     300),
    ("workflows/background/scene_parallax_layers.json","视差三层 sky+mid+fore",    600),
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
                assert size_kb > 100, f"图片过小（{size_kb}KB），可能生成失败"
                print(f"✅ {label}: {fname} {img.size[0]}x{img.size[1]} {size_kb}KB ({(i+1)*5}s)")
            return True
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
    for wf_path, label, timeout in WORKFLOWS:
        print(f"\n📤 提交: {label}")
        pid = submit_workflow(wf_path)
        if pid:
            ok = wait_for_result(pid, label, timeout)
            results.append((label, ok))
        else:
            results.append((label, False))

    print("\n--- 验证汇总 ---")
    for label, ok in results:
        print(f"{'✅' if ok else '❌'} {label}")
    if all(ok for _, ok in results):
        print("\n✅ 所有场景工作流验证通过")
    else:
        print("\n❌ 存在失败项，请检查上方日志")
        exit(1)
