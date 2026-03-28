# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Benchmark inference speed of Deformable DETR.
"""

from __future__ import annotations

import logging
import time

import cyclopts
import torch
from pydantic import BaseModel

from plain_detr.datasets import build_dataset
from plain_detr.main import Config  # noqa: TC001 -- cyclopts needs this at runtime
from plain_detr.models.detr import build as build_model
from plain_detr.util.misc import nested_tensor_from_tensor_list

logger = logging.getLogger(__name__)


class BenchmarkConfig(BaseModel):
    """Benchmark inference speed of Deformable DETR."""

    num_iters: int = 300
    """Total iterations to benchmark speed."""
    warm_iters: int = 5
    """Ignore first several slow iterations."""


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


def benchmark(bench: BenchmarkConfig, cfg: Config):
    assert 0 <= bench.warm_iters < bench.num_iters
    assert cfg.batch_size > 0
    assert cfg.resume is None or cfg.resume.exists()
    dataset = build_dataset("val", cfg)
    model, _, _ = build_model(cfg)
    model.cuda()
    model.eval()
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"])
    inputs = nested_tensor_from_tensor_list([dataset.__getitem__(0)[0].cuda() for _ in range(cfg.batch_size)])
    t = measure_average_inference_time(model, inputs, bench.num_iters, bench.warm_iters)
    return 1.0 / t * cfg.batch_size


if __name__ == "__main__":
    fps = cyclopts.run(benchmark)
    logger.info(f"Inference Speed: {fps:.1f} FPS")
