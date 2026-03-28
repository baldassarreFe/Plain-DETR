#! /usr/bin/env bash
# --------------------------------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------------------------------------------------------------------------
# Modified from https://github.com/open-mmlab/mmdetection/blob/3b53fe15d87860c6941f3dda63c0f27422da6266/tools/slurm_train.sh
# --------------------------------------------------------------------------------------------------------------------------

set -x

PARTITION="${1:?}"
JOB_NAME="${2:?}"
GPUS="${3:?}"
GPUS_PER_NODE=${GPUS_PER_NODE:-$(( GPUS < 8 ? GPUS : 8 ))}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
IFS=' ' read -ra SRUN_ARGS <<< "${SRUN_ARGS:-}"
RUN_COMMAND=("${@:4}")

srun -p "${PARTITION}" \
    --job-name="${JOB_NAME}" \
    --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --kill-on-bad-exit=1 \
    "${SRUN_ARGS[@]}" \
    "${RUN_COMMAND[@]}"

