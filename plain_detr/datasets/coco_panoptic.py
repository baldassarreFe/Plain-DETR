# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from panopticapi.utils import rgb2id
from PIL import Image
from torch.utils.data import Dataset

from plain_detr.util.box_ops import seg_masks_to_boxes

from .coco import make_coco_transforms

if TYPE_CHECKING:
    from plain_detr.datasets import transforms as T
    from plain_detr.main import Config

# Panoptic COCO uses category ids up to 200 (80 thing + 53 stuff + gaps).
NUM_CLASSES = 250


class CocoPanoptic(Dataset):
    def __init__(
        self,
        img_folder: str | Path,
        ann_folder: str | Path,
        ann_file: str | Path,
        transforms: T.Compose | None = None,
        return_seg_masks: bool = True,
    ) -> None:
        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["id"])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco["images"], self.coco["annotations"]):
                assert img["file_name"][:-4] == ann["file_name"][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_seg_masks = return_seg_masks

    def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, dict[str, Any]]:
        ann_info = self.coco["annotations"][idx] if "annotations" in self.coco else self.coco["images"][idx]
        img_path = Path(self.img_folder) / ann_info["file_name"].replace(".png", ".jpg")
        ann_path = Path(self.ann_folder) / ann_info["file_name"]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor(
                [ann["category_id"] for ann in ann_info["segments_info"]],
                dtype=torch.int64,
            )

        target = {}
        target["image_id"] = torch.tensor([ann_info["image_id"] if "image_id" in ann_info else ann_info["id"]])
        if self.return_seg_masks:
            target["seg_masks"] = masks
        target["labels"] = labels

        target["boxes"] = seg_masks_to_boxes(masks)

        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ["iscrowd", "area"]:
                target[name] = torch.tensor([ann[name] for ann in ann_info["segments_info"]])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.coco["images"])

    def get_height_and_width(self, idx: int) -> tuple[int, int]:
        img_info = self.coco["images"][idx]
        height = img_info["height"]
        width = img_info["width"]
        return height, width


def build(image_set: str, args: Config, root: Path) -> CocoPanoptic:
    if image_set == "train":
        img_folder = root / "train2017"
        ann_folder = root / "panoptic_train2017"
        ann_file = root / "annotations" / "panoptic_train2017.json"
    elif image_set == "val":
        img_folder = root / "val2017"
        ann_folder = root / "panoptic_val2017"
        ann_file = root / "annotations" / "panoptic_val2017.json"
    else:
        raise ValueError(f"unknown image set {image_set!r}")

    return CocoPanoptic(
        img_folder,
        ann_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_seg_masks=args.do_segmentation,
    )
