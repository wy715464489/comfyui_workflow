#!/bin/bash
# 千问图像（Qwen-Image）模型下载脚本
# 适配 Mac M4 Pro 24GB 统一内存 | 使用 aria2c 多线程下载
# 主模型：Q4_K_M (~8GB) + 文本编码器 fp8 (~7GB) + VAE (~250MB) ≈ 15GB 总计

TOKEN="${HF_TOKEN:-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}"
COMFYUI="/Users/zero/data/comfyui/ComfyUI"

MODELS_UNET="$COMFYUI/models/diffusion_models"
MODELS_CLIP="$COMFYUI/models/text_encoders"
MODELS_VAE="$COMFYUI/models/vae"

mkdir -p "$MODELS_UNET" "$MODELS_CLIP" "$MODELS_VAE"

aria2_dl() {
    local URL="$1"
    local DEST_DIR="$2"
    local FILENAME="$3"
    local TARGET="$DEST_DIR/$FILENAME"

    if [ -f "$TARGET" ] && [ "$(stat -f%z "$TARGET" 2>/dev/null)" -gt 100000000 ]; then
        echo "✅ 已存在: $FILENAME ($(du -sh "$TARGET" | cut -f1))"
        return 0
    fi

    echo "⬇️  下载: $FILENAME ..."
    aria2c \
        --header="Authorization: Bearer $TOKEN" \
        --dir="$DEST_DIR" \
        --out="$FILENAME" \
        --continue=true \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=10M \
        --max-concurrent-downloads=1 \
        --retry-wait=5 \
        --max-tries=10 \
        --console-log-level=notice \
        "$URL"

    if [ $? -eq 0 ]; then
        echo "✅ 完成: $FILENAME ($(du -sh "$TARGET" | cut -f1))"
    else
        echo "❌ 失败: $FILENAME"
        return 1
    fi
}

echo "=== 千问图像 Qwen-Image 模型下载 ==="
echo "目标路径: $COMFYUI"
echo ""

# 1. 主模型（GGUF Q4_K_M，~8GB，适合24GB内存）
echo "【1/3】主模型 GGUF Q4_K_M (~8GB)"
aria2_dl \
    "https://huggingface.co/city96/Qwen-Image-gguf/resolve/main/qwen-image-Q4_K_M.gguf" \
    "$MODELS_UNET" \
    "qwen-image-Q4_K_M.gguf"

# 2. 文本编码器（fp8量化，~7GB，节省内存）
echo ""
echo "【2/3】文本编码器 Qwen2.5-VL-7B fp8 (~7GB)"
aria2_dl \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$MODELS_CLIP" \
    "qwen_2.5_vl_7b_fp8_scaled.safetensors"

# 3. VAE（~250MB）
echo ""
echo "【3/3】VAE (~250MB)"
aria2_dl \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
    "$MODELS_VAE" \
    "qwen_image_vae.safetensors"

echo ""
echo "🎉 下载完成！"
echo ""
echo "📋 后续步骤："
echo "  1. 在 ComfyUI Manager 中安装插件: ComfyUI-GGUF (city96/ComfyUI-GGUF)"
echo "  2. 重启 ComfyUI"
echo "  3. 加载工作流: workflows/ui/qwen_image_title_art.json"
echo ""
du -sh "$MODELS_UNET"/qwen-image-Q4_K_M.gguf "$MODELS_CLIP"/qwen_2.5_vl_7b_fp8_scaled.safetensors "$MODELS_VAE"/qwen_image_vae.safetensors 2>/dev/null
