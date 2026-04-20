# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from . import coco, coco_panoptic, zod

if TYPE_CHECKING:
    import torch.utils.data

    from plain_detr.main import Config


def get_num_classes(dataset_name: str) -> int:
    """Return the number of classes for a dataset (max category id + 1)."""
    lookup = {
        "coco": coco.NUM_CLASSES,
        "coco_panoptic": coco_panoptic.NUM_CLASSES,
        "zod": zod.NUM_CLASSES,
    }
    if dataset_name not in lookup:
        raise ValueError(f"unknown dataset_name {dataset_name!r}")
    return lookup[dataset_name]


def build_dataset(image_set: str, args: Config) -> tuple[torch.utils.data.Dataset, int]:
    """Build a dataset and return ``(dataset, num_classes)``."""
    # coco and coco_panoptic share the same root directory.
    if args.dataset_name == "coco":
        return coco.build(image_set, args, args.data_dir / "coco"), coco.NUM_CLASSES
    if args.dataset_name == "coco_panoptic":
        return coco_panoptic.build(image_set, args, args.data_dir / "coco"), coco_panoptic.NUM_CLASSES
    if args.dataset_name == "zod":
        return zod.build(image_set, args, args.data_dir / "zod"), zod.NUM_CLASSES
    raise ValueError(f"unknown dataset_name {args.dataset_name!r}")
