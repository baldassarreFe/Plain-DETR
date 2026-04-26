#!/usr/bin/env bash

set -x

FILE_NAME=$(basename "$0")
EXP_DIR=./exps/${FILE_NAME%.*}
PY_ARGS=("${@:1}")

# Build repeated --args.train_min_sizes flags for cyclopts list parsing
TRAIN_SIZES=()
for s in 900 960 1020 1080 1140 1200 1260 1320 1380 1440 1500; do
    TRAIN_SIZES+=(--args.train_min_sizes "$s")
done

python -u -m plain_detr.main \
    --args.output_dir "${EXP_DIR}" \
    --args.with_box_refine \
    --args.two_stage \
    --args.mixed_selection \
    --args.look_forward_twice \
    --args.num_queries_one2one 300 \
    --args.num_queries_one2many 1500 \
    --args.k_one2many 6 \
    --args.lambda_one2many 1.0 \
    --args.dropout 0.0 \
    --args.norm_type pre_norm \
    --args.backbone dinov3_vit_base \
    --args.drop_path_rate 0.1 \
    --args.num_feature_levels 1 \
    --args.decoder_type global_rpe_decomp \
    --args.decoder_rpe_type linear \
    --args.proposal_feature_levels 4 \
    --args.proposal_in_stride 16 \
    --args.pretrained_backbone_path ./pt_models/dinov3_vit_base.safetensors \
    --args.epochs 24 \
    --args.lr_drop 22 \
    --args.warmup 1000 \
    --args.lr 2e-4 \
    --args.use_layerwise_decay \
    --args.lr_decay_rate 0.9 \
    --args.weight_decay 0.05 \
    --args.wd_norm_mult 0.0 \
    --args.dataset_name zod \
    --args.batch_size 2 \
    --args.amp_dtype bf16 \
    --args.max_size 2500 \
    "${TRAIN_SIZES[@]}" \
    --args.val_min_size 1500 \
    --args.crop_scales 800 --args.crop_scales 1000 --args.crop_scales 1200 \
    --args.crop_min_size 700 \
    --args.crop_max_size 1200 \
    --args.topk 300 \
    --args.max_dets 1 --args.max_dets 10 --args.max_dets 300 \
    "${PY_ARGS[@]}"
