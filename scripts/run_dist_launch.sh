#! /usr/bin/env bash
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

set -x

GPUS="${1:?}"
RUN_COMMAND=("${@:2}")
GPUS_PER_NODE=${GPUS_PER_NODE:-$(( GPUS < 8 ? GPUS : 8 ))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-0}
NNODES=$(( GPUS / GPUS_PER_NODE ))

torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    "${RUN_COMMAND[@]}"
