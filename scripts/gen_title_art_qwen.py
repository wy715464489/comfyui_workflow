#!/usr/bin/env python3
"""
千问图像（Qwen-Image）生成「万灵绘」标题艺术字
用法: python3 gen_title_art_qwen.py [--seed SEED] [--output 文件名]
前提: ComfyUI 已运行，已安装 ComfyUI-GGUF 插件，模型文件已就位
"""
import urllib.request, json, time, os, argparse

API = "http://127.0.0.1:8188"
WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "../workflows/ui/qwen_image_title_art.json")
UNITY_OUT = "/Users/zero/data/project/lore_of_myriad_beings/Assets/Art/UI/Generated"


def load_workflow():
    with open(os.path.realpath(WORKFLOW_PATH), "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 转为 API prompt 格式（节点 id → inputs）
    prompt = {}
    for node in raw["nodes"]:
        node_id = str(node["id"])
        inputs = {}

        # 处理 widget 值
        wv = node.get("widgets_values", [])
        node_type = node["type"]

        if node_type == "UnetLoaderGGUF":
            inputs["unet_name"] = wv[0]
        elif node_type == "CLIPLoader":
            inputs["clip_name"] = wv[0]
            inputs["type"] = wv[1]
            inputs["device"] = wv[2] if len(wv) > 2 else "default"
        elif node_type == "VAELoader":
            inputs["vae_name"] = wv[0]
        elif node_type == "EmptyLatentImage":
            inputs["width"] = wv[0]
            inputs["height"] = wv[1]
            inputs["batch_size"] = wv[2]
        elif node_type == "CLIPTextEncode":
            inputs["text"] = wv[0]
        elif node_type == "KSampler":
            inputs["seed"] = wv[0]
            inputs["control_after_generate"] = wv[1]
            inputs["steps"] = wv[2]
            inputs["cfg"] = wv[3]
            inputs["sampler_name"] = wv[4]
            inputs["scheduler"] = wv[5]
            inputs["denoise"] = wv[6]
        elif node_type == "SaveImage":
            inputs["filename_prefix"] = wv[0]

        # 处理连线输入
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                # 找到连线源节点
                link_id = inp["link"]
                for link in raw["links"]:
                    if link[0] == link_id:
                        src_node_id = str(link[1])
                        src_slot = link[2]
                        inputs[inp["name"]] = [src_node_id, src_slot]
                        break

        prompt[node_id] = {
            "class_type": node_type,
            "inputs": inputs,
        }
    return prompt


def queue_prompt(prompt):
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{API}/prompt", data=data,
        headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req).read())


def wait_result(prompt_id, timeout=600):
    print(f"⏳ 等待生成完成（最长 {timeout}s）...")
    for i in range(timeout):
        time.sleep(1)
        h = json.loads(urllib.request.urlopen(f"{API}/history/{prompt_id}").read())
        if prompt_id in h:
            return h[prompt_id]["outputs"]
        if i % 10 == 0:
            print(f"   已等待 {i}s...")
    return None


def save_image(outputs, filename):
    os.makedirs(UNITY_OUT, exist_ok=True)
    for node_id, out in outputs.items():
        if "images" in out:
            img = out["images"][0]
            url = f"{API}/view?filename={img['filename']}&subfolder={img.get('subfolder', '')}&type=output"
            dest = os.path.join(UNITY_OUT, filename)
            urllib.request.urlretrieve(url, dest)
            print(f"✅ 已保存: {dest}")
            return dest
    return None


def main():
    parser = argparse.ArgumentParser(description="千问图像生成万灵绘标题艺术字")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    parser.add_argument("--output", default="title_qwen.png", help="输出文件名（默认title_qwen.png）")
    parser.add_argument("--steps", type=int, default=20, help="推理步数（默认20）")
    parser.add_argument("--cfg", type=float, default=4.5, help="CFG引导强度（默认4.5）")
    args = parser.parse_args()

    # 加载并修改工作流
    prompt = load_workflow()

    # 修改 KSampler 参数
    for node_id, node in prompt.items():
        if node["class_type"] == "KSampler":
            node["inputs"]["seed"] = args.seed
            node["inputs"]["steps"] = args.steps
            node["inputs"]["cfg"] = args.cfg

    # 检查 ComfyUI 是否运行
    try:
        urllib.request.urlopen(f"{API}/system_stats", timeout=5)
    except Exception:
        print(f"❌ 无法连接 ComfyUI API ({API})")
        print("   请先启动 ComfyUI: cd /Users/zero/data/comfyui/ComfyUI && python main.py")
        return

    print(f"🎨 开始生成「万灵绘」标题艺术字")
    print(f"   种子: {args.seed} | 步数: {args.steps} | CFG: {args.cfg}")

    result = queue_prompt(prompt)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(f"❌ 提交失败: {result}")
        return

    print(f"📤 已提交，prompt_id: {prompt_id}")
    outputs = wait_result(prompt_id)

    if outputs:
        save_image(outputs, args.output)
    else:
        print("❌ 生成超时或失败")


if __name__ == "__main__":
    main()
