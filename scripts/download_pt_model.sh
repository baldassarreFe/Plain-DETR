#! /usr/bin/env bash

set -e

mkdir -p ./pt_models

function download_if_missing {
    local dest="$1"
    local url="$2"
    if [ -f "${dest}" ]; then
        echo "Already exists: ${dest}, skipping."
    else
        echo "Downloading ${dest}..."
        wget -O "${dest}" "${url}"
    fi
}

# ---------------------------------------------------------------------------
# SwinV2 pre-trained models
# ---------------------------------------------------------------------------

# MIM pre-trained model, swinv2-small
download_if_missing ./pt_models/swinv2_small_1k_500k_mim_pt.pth \
    "https://huggingface.co/zdaxie/SimMIM/resolve/main/simmim_swinv2_pretrain_models/swinv2_small_1k_500k.pth?download=true"

# Supervised pre-trained model, swinv2-small
download_if_missing ./pt_models/swinv2_small_patch4_window16_256_sup_pt.pth \
    "https://huggingface.co/microsoft/swinv2-small-patch4-window16-256/blob/main/pytorch_model.bin?download=true"

# ---------------------------------------------------------------------------
# DINOv3 pre-trained models (from timm on HuggingFace)
# ---------------------------------------------------------------------------

# DINOv3 ViT-S/16 (384-dim, 12 blocks, 21M params)
download_if_missing ./pt_models/dinov3_vit_small.safetensors \
    "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/resolve/main/model.safetensors"

# DINOv3 ViT-B/16 (768-dim, 12 blocks, 86M params)
download_if_missing ./pt_models/dinov3_vit_base.safetensors \
    "https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/resolve/main/model.safetensors"

# DINOv3 ViT-L/16 (1024-dim, 24 blocks, 300M params)
download_if_missing ./pt_models/dinov3_vit_large.safetensors \
    "https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m/resolve/main/model.safetensors"
