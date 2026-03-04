#!/usr/bin/env python3
"""
万灵绘 UI 图像后处理脚本（无 TTF 版本）
功能：
  1. 处理标题图 title_art.png（千问Image生成的去背景图，直接使用）
  2. 从千问Image生成的按钮基础图生成 4 种状态（normal/hover/pressed/disabled）

前提：gen_qwen_text_assets.py 已完成生成，输出 *_art.png 到 Generated/ 目录

用法:
  python3 gen_ui_assets.py              # 全部处理
  python3 gen_ui_assets.py --title      # 仅处理标题
  python3 gen_ui_assets.py --buttons    # 仅生成按钮四态
"""
import argparse, os
from PIL import Image, ImageFilter

# ── 路径配置 ─────────────────────────────────────────────────────────────────
PROJECT = "/Users/zero/data/project/lore_of_myriad_beings"
OUT_DIR = f"{PROJECT}/Assets/Art/UI/Generated"

# 按钮最终输出尺寸（Unity MakeSpriteBtn 对应 480×96）
BTN_W, BTN_H = 480, 96

BUTTONS = [
    "btn_start",
    "btn_load",
    "btn_settings",
    "btn_about",
    "btn_quit",
]


# ════════════════════════════════════════════════════════════════════════════
# 通用工具
# ════════════════════════════════════════════════════════════════════════════

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """调整 RGBA 图像 RGB 通道亮度，保留 Alpha"""
    r, g, b, a = img.split()
    r = r.point(lambda x: min(255, int(x * factor)))
    g = g.point(lambda x: min(255, int(x * factor)))
    b = b.point(lambda x: min(255, int(x * factor)))
    return Image.merge("RGBA", (r, g, b, a))


def adjust_alpha(img: Image.Image, factor: float) -> Image.Image:
    """整体缩放 Alpha 通道（0.0–1.0）"""
    r, g, b, a = img.split()
    a = a.point(lambda x: int(x * factor))
    return Image.merge("RGBA", (r, g, b, a))


def to_grayscale_rgba(img: Image.Image) -> Image.Image:
    """将 RGBA 图像的 RGB 通道转为灰度（保留 Alpha）"""
    r, g, b, a = img.split()
    gray = r.point(lambda x: 0)  # 占位
    # 感知灰度
    data = [
        (int(0.299 * rv + 0.587 * gv + 0.114 * bv),) * 3
        for rv, gv, bv in zip(r.getdata(), g.getdata(), b.getdata())
    ]
    g_r = r.copy(); g_r.putdata([d[0] for d in data])
    g_g = g.copy(); g_g.putdata([d[0] for d in data])
    g_b = b.copy(); g_b.putdata([d[0] for d in data])
    return Image.merge("RGBA", (g_r, g_g, g_b, a))


def add_glow(img: Image.Image, radius: float = 4, strength: float = 0.5) -> Image.Image:
    """添加外发光效果（模糊叠加）"""
    bright = adjust_brightness(img, 1.3)
    glow   = bright.filter(ImageFilter.GaussianBlur(radius=radius))
    glow   = adjust_alpha(glow, strength)
    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result = Image.alpha_composite(result, glow)
    result = Image.alpha_composite(result, img)
    return result


# ════════════════════════════════════════════════════════════════════════════
# 1. 标题艺术字（直接使用千问生成的去背景图）
# ════════════════════════════════════════════════════════════════════════════

def process_title():
    """验证并整理标题图（千问Image已输出透明背景，无需TTF叠加）"""
    src = f"{OUT_DIR}/title_art.png"
    if not os.path.exists(src):
        # 兼容旧命名：尝试 title_qwen.png
        alt = f"{OUT_DIR}/title_qwen.png"
        if os.path.exists(alt):
            print(f"   ⚠️  未找到 title_art.png，请先运行 gen_qwen_text_assets.py 生成")
            print(f"   临时使用旧文件 title_qwen.png（含黑色背景，不推荐）")
        else:
            print("   ❌ 未找到标题图，请先运行 gen_qwen_text_assets.py")
        return

    img = Image.open(src).convert("RGBA")
    print(f"   ✅ 标题图: {src}  {img.size}  模式=RGBA")
    # 验证透明度
    corners = [img.getpixel((0, 0)), img.getpixel((img.width - 1, img.height - 1))]
    alphas  = [c[3] for c in corners]
    print(f"   四角透明度: {alphas}  {'✅ 背景透明' if all(a < 30 for a in alphas) else '⚠️ 背景可能不透明'}")


# ════════════════════════════════════════════════════════════════════════════
# 2. 按钮四态生成（基于千问生成的 btn_*_art.png，无 TTF）
# ════════════════════════════════════════════════════════════════════════════

def make_states(base: Image.Image) -> dict:
    """从基础图（normal）生成4种状态"""
    base = base.convert("RGBA")

    # normal：原始去背景图
    normal = base.copy()

    # hover：提亮 + 外发光
    hover = add_glow(adjust_brightness(base, 1.2), radius=4, strength=0.6)

    # pressed：压暗 + 轻微缩小感（pad 2px 内缩）
    pressed_base = adjust_brightness(base, 0.7)
    pressed = Image.new("RGBA", base.size, (0, 0, 0, 0))
    inner = pressed_base.crop((2, 2, base.width - 2, base.height - 2))
    inner = inner.resize((base.width - 4, base.height - 4), Image.LANCZOS)
    pressed.paste(inner, (3, 3))  # 右下偏移1px营造下沉感

    # disabled：去饱和 + alpha 降为 55%
    disabled = adjust_alpha(to_grayscale_rgba(base), 0.55)

    return {"normal": normal, "hover": hover, "pressed": pressed, "disabled": disabled}


def generate_buttons():
    """为每个按钮生成 normal/hover/pressed/disabled 四种状态图"""
    print("\n🎨 从千问Image艺术字生成按钮四态...")
    missing = []

    for name in BUTTONS:
        art_path = f"{OUT_DIR}/{name}_art.png"
        if not os.path.exists(art_path):
            missing.append(name)
            print(f"   ⚠️  缺少源图: {art_path}")
            continue

        # 加载源图并缩放到按钮尺寸
        src = Image.open(art_path).convert("RGBA")
        # 保持比例居中裁剪/缩放至 BTN_W×BTN_H
        src_ratio = src.width / src.height
        tgt_ratio = BTN_W / BTN_H
        if src_ratio > tgt_ratio:
            # 图像更宽：按高缩放
            new_h = BTN_H
            new_w = int(src.width * BTN_H / src.height)
        else:
            new_w = BTN_W
            new_h = int(src.height * BTN_W / src.width)
        src = src.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (BTN_W, BTN_H), (0, 0, 0, 0))
        paste_x = (BTN_W - new_w) // 2
        paste_y = (BTN_H - new_h) // 2
        canvas.paste(src, (paste_x, paste_y))
        base = canvas

        states = make_states(base)
        for state, img in states.items():
            out = f"{OUT_DIR}/{name}_{state}.png"
            img.save(out, "PNG")
        print(f"   ✅ {name}: normal/hover/pressed/disabled")

    if missing:
        print(f"\n   ⚠️  以下按钮缺少源图（需先运行 gen_qwen_text_assets.py --buttons）:")
        for m in missing:
            print(f"      - {m}_art.png")
    else:
        print(f"   输出目录: {OUT_DIR}")


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title",   action="store_true", help="仅处理标题")
    parser.add_argument("--buttons", action="store_true", help="仅生成按钮四态")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    do_all = not args.title and not args.buttons

    if args.title or do_all:
        print("📖 检查标题艺术字...")
        process_title()

    if args.buttons or do_all:
        generate_buttons()

    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()
