#!/usr/bin/env python3
"""
SD3.5 Medium 批量生成万灵绘主菜单 UI 素材
用法: python3 gen_main_menu_ui.py
"""
import urllib.request, json, time, os

API = "http://127.0.0.1:8188"
UNITY_OUT = "/Users/zero/data/project/lore_of_myriad_beings/Assets/Art/UI/Generated"
NEG = "blurry, low quality, ugly, multiple elements, text, watermark, modern, western, realistic photo, 3d render, oversaturated"

TASKS = [
    ("btn_normal.png",
     "single game UI button sprite, ancient chinese jade stone rectangle with gold dragon border ornament, dark background, centered isolated element, highly detailed texture, fantasy RPG button, 2D game asset",
     512, 512, 1001, 4.5),
    ("btn_hover.png",
     "single game UI button sprite glowing, ancient chinese jade stone rectangle with golden phoenix border, bright luminous glow effect, fantasy RPG button hover state, 2D game asset, highly detailed",
     512, 512, 1002, 4.5),
    ("btn_pressed.png",
     "single game UI button sprite pressed, ancient chinese dark jade stone rectangle with bronze dragon border, inner shadow depth effect, darker color scheme, fantasy RPG button active state, 2D game asset",
     512, 512, 1003, 4.5),
    ("btn_disabled.png",
     "single game UI button sprite disabled, grey stone rectangle with faded bronze border, desaturated muted tones, no glow, fantasy RPG button disabled state, 2D game asset",
     512, 512, 1004, 4.0),
    ("panel_background.png",
     "ancient chinese game UI panel frame, large ornate dark panel with gold dragon cloud border corners, inner dark semi-transparent area, traditional chinese lacquer aesthetic, fantasy RPG game UI, high detail",
     1024, 1024, 1005, 4.5),
    ("title_banner.png",
     "ancient chinese game title banner, wide horizontal decorative banner, golden dragon motifs, red and gold color scheme, misty ink wash atmosphere, shanhaijing mythology style, game UI title decoration",
     1024, 256, 1006, 4.5),
    ("title_bg.png",
     "ancient chinese wide horizontal background, misty mountain ink wash painting, celestial light rays, dark navy with gold accents, fantasy game title background, shanhaijing mythological atmosphere",
     1024, 256, 1007, 4.0),
    ("panel_bg.png",
     "ancient chinese square panel background, dark navy with subtle cloud and mountain ink patterns, gold trim corners, traditional chinese game UI background, clean design",
     512, 512, 1008, 4.0),
    ("divider.png",
     "ancient chinese horizontal ornamental line, single thin gold and jade decorative line with cloud motif at center, traditional UI separator element",
     512, 64, 1009, 4.5),
]

def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{API}/prompt", data=data,
                                  headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def wait_result(prompt_id, timeout=300):
    for _ in range(timeout):
        time.sleep(1)
        h = json.loads(urllib.request.urlopen(f"{API}/history/{prompt_id}").read())
        if prompt_id in h:
            return h[prompt_id]["outputs"]
    return None

def save_image(outputs, filename):
    for node_id, out in outputs.items():
        if "images" in out:
            img = out["images"][0]
            url = f"{API}/view?filename={img['filename']}&subfolder={img.get('subfolder','')}&type=output"
            data = urllib.request.urlopen(url).read()
            path = os.path.join(UNITY_OUT, filename)
            with open(path, "wb") as f:
                f.write(data)
            print(f"  ✅ 保存: {filename} ({len(data)//1024}KB)")
            return True
    return False

def make_workflow(positive, negative, width, height, seed, cfg):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "sd3.5_medium.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "TripleCLIPLoader",
              "inputs": {"clip_name1": "text_encoders/clip_g.safetensors",
                         "clip_name2": "text_encoders/clip_l.safetensors",
                         "clip_name3": "text_encoders/t5xxl_fp16.safetensors"}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["2", 0]}},
        "5": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
                         "latent_image": ["5", 0], "seed": seed, "steps": 28,
                         "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAELoader",
              "inputs": {"vae_name": "sd3_vae.safetensors"}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["7", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "sd35_main_menu"}}
    }

os.makedirs(UNITY_OUT, exist_ok=True)
print(f"SD3.5 Medium — 生成 {len(TASKS)} 张主菜单 UI 素材\n")

for filename, positive, w, h, seed, cfg in TASKS:
    print(f"生成: {filename} ({w}x{h})")
    wf = make_workflow(positive, NEG, w, h, seed, cfg)
    result = queue_prompt(wf)
    pid = result.get("prompt_id")
    if not pid:
        print(f"  ❌ 队列失败: {result}")
        continue
    print(f"  等待生成 (id={pid[:8]})...")
    outputs = wait_result(pid, timeout=300)
    if outputs:
        save_image(outputs, filename)
    else:
        print(f"  ❌ 超时/失败")

print("\n✅ 全部完成！素材已保存到 Unity 项目")
