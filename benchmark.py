# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Benchmark inference speed of Deformable DETR.
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from main import get_args_parser as get_main_args_parser
from plain_detr.datasets import build_dataset
from plain_detr.models import build_model
from plain_detr.util.misc import nested_tensor_from_tensor_list

logger = logging.getLogger(__name__)


def get_benckmark_arg_parser():
    parser = argparse.ArgumentParser("Benchmark inference speed of Deformable DETR.")
    parser.add_argument("--num_iters", type=int, default=300, help="total iters to benchmark speed")
    parser.add_argument(
        "--warm_iters",
        type=int,
        default=5,
        help="ignore first several iters that are very slow",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference")
    parser.add_argument("--resume", type=str, help="load the pre-trained checkpoint")
    return parser


@torch.no_grad()
def measure_average_inference_time(model, inputs, num_iters=100, warm_iters=5):
    ts = []
    for iter_ in range(num_iters):
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
            ts.append(t)
    logger.debug(f"{ts}")
    return sum(ts) / len(ts)


def benchmark():
    args, _ = get_benckmark_arg_parser().parse_known_args()
    main_args = get_main_args_parser().parse_args(_)
    # Derive dataset paths from --data_dir when not explicitly provided
    if main_args.coco_path is None:
        main_args.coco_path = str(Path(main_args.data_dir) / "coco")
    if main_args.coco_panoptic_path is None:
        main_args.coco_panoptic_path = str(Path(main_args.data_dir) / "coco")
    assert 0 <= args.warm_iters < args.num_iters
    assert args.batch_size > 0
    assert args.resume is None or Path(args.resume).exists()
    dataset = build_dataset("val", main_args)
    model, _, _ = build_model(main_args)
    model.cuda()
    model.eval()
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"])
    inputs = nested_tensor_from_tensor_list([dataset.__getitem__(0)[0].cuda() for _ in range(args.batch_size)])
    t = measure_average_inference_time(model, inputs, args.num_iters, args.warm_iters)
    return 1.0 / t * args.batch_size


if __name__ == "__main__":
    fps = benchmark()
    logger.info(f"Inference Speed: {fps:.1f} FPS")
