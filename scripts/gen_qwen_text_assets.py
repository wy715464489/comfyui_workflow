#!/usr/bin/env python3
"""
万灵绘 · 千问Image 艺术字批量生成脚本
使用 Qwen-Image (ComfyUI-GGUF) 生成所有游戏 UI 文字图，背景透明。

生成清单（黑底原始图 → 去背景 RGBA）:
  title_art.png              — 「万灵绘」标题艺术字
  btn_start_art.png          — 「开始游戏」
  btn_load_art.png           — 「加载游戏」
  btn_settings_art.png       — 「设  置」
  btn_about_art.png          — 「关  于」
  btn_quit_art.png           — 「退  出」

后续由 gen_ui_assets.py 从 *_art.png 生成4态按钮图（无 TTF 字体）。

用法:
  python3 gen_qwen_text_assets.py            # 全部生成
  python3 gen_qwen_text_assets.py --title    # 仅标题
  python3 gen_qwen_text_assets.py --buttons  # 仅按钮
  python3 gen_qwen_text_assets.py --seed 42  # 指定随机种子
"""
import argparse, json, os, time, urllib.request
from PIL import Image

API = "http://127.0.0.1:8188"
WORKFLOW = os.path.join(os.path.dirname(__file__), "../workflows/ui/qwen_image_title_art.json")
OUT_DIR  = "/Users/zero/data/project/lore_of_myriad_beings/Assets/Art/UI/Generated"

# ── 去背景（感知亮度法）────────────────────────────────────────────────────────
def remove_dark_bg(img: Image.Image, threshold: int = 50, feather: int = 40) -> Image.Image:
    img = img.convert("RGBA")
    data = list(img.getdata())
    new_data = []
    for rv, gv, bv, av in data:
        lum = int(0.299 * rv + 0.587 * gv + 0.114 * bv)
        if lum < threshold:
            new_data.append((rv, gv, bv, 0))
        elif lum < threshold + feather:
            alpha = int(255 * (lum - threshold) / feather)
            new_data.append((rv, gv, bv, alpha))
        else:
            new_data.append((rv, gv, bv, av))
    result = Image.new("RGBA", img.size)
    result.putdata(new_data)
    return result


# ── 资产定义 ──────────────────────────────────────────────────────────────────
#  (output_name, positive_prompt, width, height)
ASSETS = [
    (
        "title_art",
        '「万灵绘」三字，中国传统篆书书法大字，金色笔墨，纯黑色背景，山海经神话国风游戏标题，'
        '每个字笔画清晰，墨迹精细，祥云纹饰，史诗感，高对比度，无其他文字',
        '游戏标题艺术字，纯黑背景，金色大字，第一个字是"万"第二个字是"灵"第三个字是"绘"，'
        '中国古典书法，水墨笔触，山海经风格',
        800, 512,
    ),
    (
        "btn_start_art",
        '「开始游戏」四字横排，中国古风游戏按钮文字，金色隶书，纯黑色背景，'
        '笔画清晰，高对比度，无其他文字，横幅构图',
        '游戏按钮文字，纯黑背景，四个汉字"开始游戏"，金色古典字体，水平排列',
        512, 128,
    ),
    (
        "btn_load_art",
        '「加载游戏」四字横排，中国古风游戏按钮文字，金色隶书，纯黑色背景，'
        '笔画清晰，高对比度，无其他文字，横幅构图',
        '游戏按钮文字，纯黑背景，四个汉字"加载游戏"，金色古典字体，水平排列',
        512, 128,
    ),
    (
        "btn_settings_art",
        '「设置」两字横排，中国古风游戏按钮文字，金色隶书，纯黑色背景，'
        '笔画清晰，高对比度，无其他文字，横幅构图',
        '游戏按钮文字，纯黑背景，两个汉字"设置"，金色古典字体，水平排列',
        512, 128,
    ),
    (
        "btn_about_art",
        '「关于」两字横排，中国古风游戏按钮文字，金色隶书，纯黑色背景，'
        '笔画清晰，高对比度，无其他文字，横幅构图',
        '游戏按钮文字，纯黑背景，两个汉字"关于"，金色古典字体，水平排列',
        512, 128,
    ),
    (
        "btn_quit_art",
        '「退出」两字横排，中国古风游戏按钮文字，金色隶书，纯黑色背景，'
        '笔画清晰，高对比度，无其他文字，横幅构图',
        '游戏按钮文字，纯黑背景，两个汉字"退出"，金色古典字体，水平排列',
        512, 128,
    ),
]

NEG_PROMPT = (
    "blurry, low quality, watermark, latin characters, modern font, "
    "white background, western style, extra text, extra characters, wrong strokes"
)


# ── ComfyUI API ───────────────────────────────────────────────────────────────
def build_prompt(positive: str, width: int, height: int, seed: int) -> dict:
    """直接构造 API prompt（不依赖工作流 JSON 解析）"""
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "qwen-image-Q4_K_M.gguf"}
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["2", 0]}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEG_PROMPT, "clip": ["2", 0]}
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["5", 0],
                "negative":     ["6", 0],
                "latent_image": ["4", 0],
                "seed":         seed,
                "steps":        20,
                "cfg":          4.5,
                "sampler_name": "euler",
                "scheduler":    "simple",
                "denoise":      1.0,
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "qwen_text"}
        }
    }


def queue_prompt(prompt: dict) -> str:
    data = json.dumps({"prompt": prompt}).encode()
    req  = urllib.request.Request(f"{API}/prompt", data=data,
                                  headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["prompt_id"]


def wait_result(prompt_id: str, timeout: int = 1800) -> dict | None:
    """等待单个 prompt 完成，返回 outputs"""
    for i in range(timeout):
        time.sleep(1)
        h = json.loads(urllib.request.urlopen(f"{API}/history/{prompt_id}").read())
        if prompt_id in h:
            return h[prompt_id]["outputs"]
        if i > 0 and i % 30 == 0:
            print(f"      已等待 {i}s...")
    return None


def download_first_image(outputs: dict, dest_path: str):
    for node_id, out in outputs.items():
        if "images" in out:
            img_info = out["images"][0]
            url = (f"{API}/view?filename={img_info['filename']}"
                   f"&subfolder={img_info.get('subfolder','')}&type=output")
            urllib.request.urlretrieve(url, dest_path)
            return dest_path
    return None


# ── 主流程 ────────────────────────────────────────────────────────────────────
def generate_asset(name: str, positive: str, secondary_prompt: str,
                   width: int, height: int, seed: int, attempts: int = 1):
    """生成单个资产：原始图 → 去背景 → 保存"""
    raw_path = os.path.join(OUT_DIR, f"{name}_raw.png")
    art_path = os.path.join(OUT_DIR, f"{name}.png")

    print(f"\n🎨 生成 [{name}]  {width}×{height}  seed={seed}")
    print(f"   提示词: {positive[:60]}...")

    for attempt in range(attempts):
        cur_seed = seed + attempt * 1000
        if attempt > 0:
            print(f"   ⚠️  重试 #{attempt+1}（seed={cur_seed}）")

        prompt = build_prompt(positive, width, height, cur_seed)
        pid = queue_prompt(prompt)
        print(f"   📤 已提交 prompt_id={pid}")

        outputs = wait_result(pid)
        if not outputs:
            print(f"   ❌ 超时，跳过")
            continue

        raw = download_first_image(outputs, raw_path)
        if not raw:
            print(f"   ❌ 无输出图像")
            continue

        print(f"   📥 已下载原始图: {raw_path}")

        # 去黑背景
        src = Image.open(raw_path).convert("RGB")
        result = remove_dark_bg(src, threshold=50, feather=45)
        result.save(art_path, "PNG")
        print(f"   ✅ 去背景完成: {art_path}  {result.size} RGBA")
        return True

    print(f"   ❌ 全部尝试失败: {name}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title",   action="store_true")
    parser.add_argument("--buttons", action="store_true")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--attempts", type=int, default=1,
                        help="每张图最多尝试次数（字形不对时自动重试）")
    args = parser.parse_args()

    # 检查 ComfyUI
    try:
        urllib.request.urlopen(f"{API}/system_stats", timeout=5)
        print("✅ ComfyUI 连接正常")
    except Exception:
        print(f"❌ 无法连接 ComfyUI ({API})")
        print("   请先启动: cd /Users/zero/data/comfyui/ComfyUI && python main.py")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    do_all    = not args.title and not args.buttons
    do_title  = args.title   or do_all
    do_btns   = args.buttons or do_all

    tasks = []
    if do_title:
        tasks.append(ASSETS[0])        # title_art
    if do_btns:
        tasks.extend(ASSETS[1:])       # btn_*_art

    total = len(tasks)
    ok = 0
    for i, (name, pos, _sec, w, h) in enumerate(tasks, 1):
        print(f"\n[{i}/{total}] ═══════════════════════════════════════")
        if generate_asset(name, pos, _sec, w, h, args.seed, args.attempts):
            ok += 1

    print(f"\n🎉 完成 {ok}/{total}")
    print("   下一步: python3 gen_ui_assets.py --from-qwen")


if __name__ == "__main__":
    main()
