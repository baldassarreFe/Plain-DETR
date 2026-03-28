#! /usr/bin/env bash

mkdir -p ./pt_models

# MIM pre-trained model, swinv2-small
echo "Downloading MIM pre-trained model..."
wget -O ./pt_models/swinv2_small_1k_500k_mim_pt.pth \
"https://huggingface.co/zdaxie/SimMIM/resolve/main/simmim_swinv2_pretrain_models/swinv2_small_1k_500k.pth?download=true"

# Supervised pre-trained model, swinv2-small
echo "Downloading supervised pre-trained model..."
wget -O ./pt_models/swinv2_small_patch4_window16_256_sup_pt.pth \
"https://huggingface.co/microsoft/swinv2-small-patch4-window16-256/blob/main/pytorch_model.bin?download=true"