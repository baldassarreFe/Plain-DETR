#!/usr/bin/env bash

set -x

FILE_NAME=$(basename "$0")
EXP_DIR=./exps/${FILE_NAME%.*}
PY_ARGS=("${@:1}")

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
    --args.backbone swin_v2_small_window12to16_2global \
    --args.drop_path_rate 0.1 \
    --args.upsample_backbone_output \
    --args.upsample_stride 16 \
    --args.num_feature_levels 1 \
    --args.decoder_type global_rpe_decomp \
    --args.decoder_rpe_type linear \
    --args.proposal_feature_levels 4 \
    --args.proposal_in_stride 16 \
    --args.pretrained_backbone_path ./pt_models/swinv2_small_1k_500k_mim_pt.pth \
    --args.epochs 12 \
    --args.lr_drop 11 \
    --args.warmup 1000 \
    --args.lr 2e-4 \
    --args.use_layerwise_decay \
    --args.lr_decay_rate 0.9 \
    --args.weight_decay 0.05 \
    --args.wd_norm_mult 0.0 \
    "${PY_ARGS[@]}"
