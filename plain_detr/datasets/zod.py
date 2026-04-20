"""ZOD (Zenseact Open Dataset) data loading for object detection.

ZOD annotations are pre-converted to COCO format by ``scripts/convert_zod_to_coco.py``.
The image ``file_name`` field contains the path relative to the dataset root
(e.g. ``single_frames/000042/camera_front_blur/000042_india_….jpg``), so the
image folder passed to :class:`CocoDetection` is the dataset root itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plain_detr.util.misc import get_local_rank, get_local_size

from .coco import CocoDetection, make_coco_transforms

if TYPE_CHECKING:
    from pathlib import Path

    from plain_detr.main import Config

# Number of object classes (max category id + 1).
# ZOD uses contiguous ids 1-10.
NUM_CLASSES = 11

# ZOD category names, indexed by label id (0-10).
# Matches the category mapping in scripts/convert_zod_to_coco.py.
CATEGORY_NAMES: list[str] = [
    "N/A",  # 0  - background / unused
    "Vehicle",  # 1
    "VulnerableVehicle",  # 2
    "Pedestrian",  # 3
    "Animal",  # 4
    "PoleObject",  # 5
    "TrafficSign",  # 6
    "TrafficSignal",  # 7
    "TrafficGuide",  # 8
    "TrafficBeacon",  # 9
    "DynamicBarrier",  # 10
]


def build(image_set: str, args: Config, root: Path) -> CocoDetection:
    if image_set == "train":
        img_folder = root / "single_frames"
        ann_file = root / "annotations" / "instances_train.json"
    elif image_set == "val":
        img_folder = root / "single_frames"
        ann_file = root / "annotations" / "instances_val.json"
    else:
        raise ValueError(f"unknown image_set {image_set!r}")

    if args.do_segmentation:
        raise ValueError("ZOD doesn't have segmentation annotations")

    return CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_seg_masks=False,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
    )
