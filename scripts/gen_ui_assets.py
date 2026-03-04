#!/usr/bin/env python3
"""
万灵绘 UI 图像生成与处理脚本
功能：
  1. 处理 Qwen-Image 生成的标题图（去黑背景 + 叠加正确"万灵绘"文字）
  2. 生成古风按钮图（5个按钮 × 4种状态 = 20张图）

用法:
  python3 gen_ui_assets.py              # 全部生成
  python3 gen_ui_assets.py --title      # 仅处理标题
  python3 gen_ui_assets.py --buttons    # 仅生成按钮
"""
import argparse, os, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── 路径配置 ─────────────────────────────────────────────────────────────────
PROJECT = "/Users/zero/data/project/lore_of_myriad_beings"
OUT_DIR = f"{PROJECT}/Assets/Art/UI/Generated"
FONT_PATH = f"{PROJECT}/Assets/Fonts/SourceHanSansCN-Regular.otf"

# ── 颜色 ─────────────────────────────────────────────────────────────────────
GOLD        = (242, 200,  65, 255)   # 金色
GOLD_BRIGHT = (255, 235, 120, 255)   # 高亮金
GOLD_DARK   = (160, 120,  30, 255)   # 暗金（pressed）
GRAY        = (140, 130, 100, 180)   # 禁用灰
INK         = (  8,   4,   2, 230)   # 描边墨色
WHITE       = (255, 255, 255, 255)


# ════════════════════════════════════════════════════════════════════════════
# 1. 标题艺术字处理
# ════════════════════════════════════════════════════════════════════════════

def remove_dark_bg(img: Image.Image, threshold: int = 60, feather: int = 30) -> Image.Image:
    """将黑色/深色背景去除，变为透明（基于感知亮度）"""
    img = img.convert("RGBA")
    r, g, b, a = img.split()
    data = list(zip(r.getdata(), g.getdata(), b.getdata(), a.getdata()))

    new_data = []
    for rv, gv, bv, av in data:
        # 感知亮度
        lum = int(0.299 * rv + 0.587 * gv + 0.114 * bv)
        if lum < threshold:
            # 纯黑区域：全透明
            new_data.append((rv, gv, bv, 0))
        elif lum < threshold + feather:
            # 过渡区：渐变透明
            alpha = int(255 * (lum - threshold) / feather)
            new_data.append((rv, gv, bv, alpha))
        else:
            new_data.append((rv, gv, bv, av))

    result = Image.new("RGBA", img.size)
    result.putdata(new_data)
    return result


def draw_outlined_text(draw: ImageDraw.Draw, xy, text: str, font: ImageFont.FreeTypeFont,
                       fill, outline, outline_width: int = 8):
    """绘制带描边的文字"""
    x, y = xy
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx * dx + dy * dy <= outline_width * outline_width:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)


def process_title():
    """处理标题图：去黑背景 + 叠加正确的万灵绘文字"""
    src = f"{OUT_DIR}/title_qwen.png"
    dst = f"{OUT_DIR}/title_art.png"

    print("📖 处理标题艺术字...")

    # 1. 加载 Qwen 生成的艺术纹理并去背景
    if os.path.exists(src):
        base = Image.open(src).convert("RGB").resize((800, 800), Image.LANCZOS)
        base = remove_dark_bg(base, threshold=55, feather=40)
        print(f"   去黑背景完成 ({base.size})")
    else:
        # 无源图：创建空白画布
        base = Image.new("RGBA", (800, 800), (0, 0, 0, 0))
        print("   无源图，使用空白画布")

    # 2. 在去背景的纹理上叠加"万灵绘"文字
    # 创建文字图层
    text_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    font_main = ImageFont.truetype(FONT_PATH, 220)

    # 居中计算
    W, H = base.size
    bbox = draw.textbbox((0, 0), "万灵绘", font=font_main)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (W - tw) // 2
    ty = (H - th) // 2

    # 金色渐变描边文字
    draw_outlined_text(draw, (tx, ty), "万灵绘", font_main,
                       fill=GOLD, outline=INK, outline_width=10)

    # 添加金色光晕（高斯模糊的亮色层）
    glow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    draw_outlined_text(gd, (tx, ty), "万灵绘", font_main,
                       fill=GOLD_BRIGHT, outline=(0, 0, 0, 0), outline_width=0)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=12))

    # 合成：基础纹理 + 光晕 + 文字
    result = Image.new("RGBA", base.size, (0, 0, 0, 0))
    result = Image.alpha_composite(result, base)
    result = Image.alpha_composite(result, glow_layer)
    result = Image.alpha_composite(result, text_layer)

    result.save(dst, "PNG")
    print(f"   ✅ 已保存: {dst} ({result.size})")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 2. 古风按钮图生成
# ════════════════════════════════════════════════════════════════════════════

BUTTONS = [
    ("btn_start",    "开始游戏"),
    ("btn_load",     "加载游戏"),
    ("btn_settings", "设    置"),
    ("btn_about",    "关    于"),
    ("btn_quit",     "退    出"),
]

BTN_W, BTN_H = 480, 96  # 按钮尺寸（3:1横向）


def draw_button_base(draw: ImageDraw.Draw, w: int, h: int, text: str,
                     font: ImageFont.FreeTypeFont,
                     text_color, border_color, border_alpha: int = 180):
    """绘制古风按钮底图（透明背景 + 细描边 + 文字）"""
    # 细装饰横线（上下）
    line_y_top = h // 4
    line_y_bot = h * 3 // 4
    line_color = border_color[:3] + (border_alpha,)
    draw.line([(20, line_y_top), (w - 20, line_y_top)], fill=line_color, width=1)
    draw.line([(20, line_y_bot), (w - 20, line_y_bot)], fill=line_color, width=1)

    # 左右菱形装饰点
    dia = 5
    for px in [16, w - 16]:
        draw.polygon([
            (px, h // 2 - dia), (px + dia, h // 2),
            (px, h // 2 + dia), (px - dia, h // 2)
        ], fill=line_color)

    # 居中文字
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = (h - th) // 2 - 2
    draw_outlined_text(draw, (tx, ty), text, font,
                       fill=text_color, outline=INK, outline_width=5)


def make_button_state(text: str, text_color, border_color,
                      bg_color=(0, 0, 0, 0), blur: float = 0.0,
                      brightness: float = 1.0) -> Image.Image:
    """生成单个按钮状态图"""
    img = Image.new("RGBA", (BTN_W, BTN_H), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 44)
    draw_button_base(draw, BTN_W, BTN_H, text, font, text_color, border_color)

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    if brightness != 1.0:
        r, g, b, a = img.split()
        r = r.point(lambda x: min(255, int(x * brightness)))
        g = g.point(lambda x: min(255, int(x * brightness)))
        b = b.point(lambda x: min(255, int(x * brightness)))
        img = Image.merge("RGBA", (r, g, b, a))
    return img


def generate_buttons():
    """为每个按钮生成 normal / hover / pressed / disabled 四种状态"""
    print("\n🎨 生成古风按钮图...")

    for name, text in BUTTONS:
        # normal：金色文字，透明背景
        n = make_button_state(text, text_color=GOLD, border_color=GOLD)
        n.save(f"{OUT_DIR}/{name}_normal.png", "PNG")

        # hover：更亮金色 + 微微发光
        h = make_button_state(text, text_color=GOLD_BRIGHT, border_color=GOLD_BRIGHT,
                              brightness=1.15)
        # 添加整体辉光
        glow = h.filter(ImageFilter.GaussianBlur(radius=3))
        h_final = Image.alpha_composite(glow, h)
        h_final.save(f"{OUT_DIR}/{name}_hover.png", "PNG")

        # pressed：暗金 + 轻微下沉感（向右下偏移1px重绘）
        p = make_button_state(text, text_color=GOLD_DARK, border_color=GOLD_DARK)
        # 轻微整体变暗
        p = p.point(lambda x: int(x * 0.80))
        p.save(f"{OUT_DIR}/{name}_pressed.png", "PNG")

        # disabled：灰色，低透明度
        d = make_button_state(text, text_color=GRAY, border_color=GRAY)
        d.save(f"{OUT_DIR}/{name}_disabled.png", "PNG")

        print(f"   ✅ {name}: normal/hover/pressed/disabled")

    print(f"   输出目录: {OUT_DIR}")


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title",   action="store_true", help="仅处理标题")
    parser.add_argument("--buttons", action="store_true", help="仅生成按钮")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.title or (not args.title and not args.buttons):
        process_title()

    if args.buttons or (not args.title and not args.buttons):
        generate_buttons()

    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()
