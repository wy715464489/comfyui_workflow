"""
单张图生成验证脚本。
用法：python scripts/test_single.py [--url http://127.0.0.1:8188] [--ckpt v1-5-pruned-emaonly-fp16.safetensors]

确保 ComfyUI 已启动后运行。
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="ComfyUI 单张图生成验证")
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--ckpt", default="v1-5-pruned-emaonly-fp16.safetensors")
    parser.add_argument("--prompt", default="2D game character, female warrior, red hair, fantasy armor, white background, full body, anime style, masterpiece")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()

    # 检查 ComfyUI 是否在线
    try:
        stats = json.loads(urllib.request.urlopen(f"{args.url}/system_stats", timeout=5).read())
        device = stats["devices"][0]["type"] if stats.get("devices") else "unknown"
        print(f"✅ ComfyUI {stats['system']['comfyui_version']} 已就绪，设备：{device}")
    except Exception as e:
        print(f"❌ 无法连接 ComfyUI（{args.url}）：{e}")
        print("请先启动 ComfyUI：python main.py --port 8188")
        sys.exit(1)

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": args.ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": args.prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "lowres, bad anatomy, bad hands, watermark, signature, extra limbs, deformed",
                         "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": args.width, "height": args.height, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"seed": args.seed, "steps": args.steps, "cfg": 7.0,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                         "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "test_character", "images": ["6", 0]}},
    }

    print(f"📤 提交工作流（模型：{args.ckpt}，{args.width}×{args.height}，seed={args.seed}）...")
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{args.url}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"❌ 提交失败：{e.code} {e.reason}\n{body}")
        sys.exit(1)

    prompt_id = resp["prompt_id"]
    print(f"   prompt_id: {prompt_id}")
    print("⏳ 等待生成中", end="", flush=True)

    start = time.time()
    for _ in range(200):
        time.sleep(2)
        print(".", end="", flush=True)
        history = json.loads(
            urllib.request.urlopen(f"{args.url}/history/{urllib.parse.quote(prompt_id)}").read()
        )
        if prompt_id in history:
            elapsed = time.time() - start
            record = history[prompt_id]
            if record.get("status", {}).get("status_str") == "error":
                print(f"\n❌ 生成失败：{record['status']}")
                sys.exit(1)
            print(f"\n✅ 生成完成（耗时 {elapsed:.0f}s）")
            for out in record.get("outputs", {}).values():
                for img in out.get("images", []):
                    print(f"   📁 ComfyUI/output/{img['filename']}")
            sys.exit(0)

    print("\n❌ 超时（400s），请检查 ComfyUI 日志")
    sys.exit(1)


if __name__ == "__main__":
    main()
