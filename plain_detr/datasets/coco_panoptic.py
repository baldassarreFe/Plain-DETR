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

# COCO panoptic category names, indexed by label id (0-249).
# IDs 1-90 are the 80 "thing" categories (same as COCO detection).
# IDs 92-200 are the 53 "stuff" categories.
# All other indices are unused ("N/A").
# Source: https://github.com/cocodataset/panopticapi (panoptic_coco_categories.json)
CATEGORY_NAMES: list[str] = [
    "N/A",  # 0
    "person",  # 1
    "bicycle",  # 2
    "car",  # 3
    "motorcycle",  # 4
    "airplane",  # 5
    "bus",  # 6
    "train",  # 7
    "truck",  # 8
    "boat",  # 9
    "traffic light",  # 10
    "fire hydrant",  # 11
    "N/A",  # 12
    "stop sign",  # 13
    "parking meter",  # 14
    "bench",  # 15
    "bird",  # 16
    "cat",  # 17
    "dog",  # 18
    "horse",  # 19
    "sheep",  # 20
    "cow",  # 21
    "elephant",  # 22
    "bear",  # 23
    "zebra",  # 24
    "giraffe",  # 25
    "N/A",  # 26
    "backpack",  # 27
    "umbrella",  # 28
    "N/A",  # 29
    "N/A",  # 30
    "handbag",  # 31
    "tie",  # 32
    "suitcase",  # 33
    "frisbee",  # 34
    "skis",  # 35
    "snowboard",  # 36
    "sports ball",  # 37
    "kite",  # 38
    "baseball bat",  # 39
    "baseball glove",  # 40
    "skateboard",  # 41
    "surfboard",  # 42
    "tennis racket",  # 43
    "bottle",  # 44
    "N/A",  # 45
    "wine glass",  # 46
    "cup",  # 47
    "fork",  # 48
    "knife",  # 49
    "spoon",  # 50
    "bowl",  # 51
    "banana",  # 52
    "apple",  # 53
    "sandwich",  # 54
    "orange",  # 55
    "broccoli",  # 56
    "carrot",  # 57
    "hot dog",  # 58
    "pizza",  # 59
    "donut",  # 60
    "cake",  # 61
    "chair",  # 62
    "couch",  # 63
    "potted plant",  # 64
    "bed",  # 65
    "N/A",  # 66
    "dining table",  # 67
    "N/A",  # 68
    "N/A",  # 69
    "toilet",  # 70
    "N/A",  # 71
    "tv",  # 72
    "laptop",  # 73
    "mouse",  # 74
    "remote",  # 75
    "keyboard",  # 76
    "cell phone",  # 77
    "microwave",  # 78
    "oven",  # 79
    "toaster",  # 80
    "sink",  # 81
    "refrigerator",  # 82
    "N/A",  # 83
    "book",  # 84
    "clock",  # 85
    "vase",  # 86
    "scissors",  # 87
    "teddy bear",  # 88
    "hair drier",  # 89
    "toothbrush",  # 90
    "N/A",  # 91
    "banner",  # 92
    "blanket",  # 93
    "N/A",  # 94
    "bridge",  # 95
    "N/A",  # 96
    "N/A",  # 97
    "N/A",  # 98
    "N/A",  # 99
    "cardboard",  # 100
    "N/A",  # 101
    "N/A",  # 102
    "N/A",  # 103
    "N/A",  # 104
    "N/A",  # 105
    "N/A",  # 106
    "counter",  # 107
    "N/A",  # 108
    "curtain",  # 109
    "N/A",  # 110
    "N/A",  # 111
    "door-stuff",  # 112
    "N/A",  # 113
    "N/A",  # 114
    "N/A",  # 115
    "N/A",  # 116
    "N/A",  # 117
    "floor-wood",  # 118
    "flower",  # 119
    "N/A",  # 120
    "N/A",  # 121
    "fruit",  # 122
    "N/A",  # 123
    "N/A",  # 124
    "gravel",  # 125
    "N/A",  # 126
    "N/A",  # 127
    "house",  # 128
    "N/A",  # 129
    "light",  # 130
    "N/A",  # 131
    "N/A",  # 132
    "mirror-stuff",  # 133
    "N/A",  # 134
    "N/A",  # 135
    "N/A",  # 136
    "N/A",  # 137
    "net",  # 138
    "N/A",  # 139
    "N/A",  # 140
    "pillow",  # 141
    "N/A",  # 142
    "N/A",  # 143
    "platform",  # 144
    "playingfield",  # 145
    "N/A",  # 146
    "railroad",  # 147
    "river",  # 148
    "road",  # 149
    "N/A",  # 150
    "roof",  # 151
    "N/A",  # 152
    "N/A",  # 153
    "sand",  # 154
    "sea",  # 155
    "shelf",  # 156
    "N/A",  # 157
    "N/A",  # 158
    "snow",  # 159
    "N/A",  # 160
    "stairs",  # 161
    "N/A",  # 162
    "N/A",  # 163
    "N/A",  # 164
    "N/A",  # 165
    "tent",  # 166
    "N/A",  # 167
    "towel",  # 168
    "N/A",  # 169
    "N/A",  # 170
    "wall-brick",  # 171
    "N/A",  # 172
    "N/A",  # 173
    "N/A",  # 174
    "wall-stone",  # 175
    "wall-tile",  # 176
    "wall-wood",  # 177
    "water-other",  # 178
    "N/A",  # 179
    "window-blind",  # 180
    "window-other",  # 181
    "N/A",  # 182
    "N/A",  # 183
    "tree-merged",  # 184
    "fence-merged",  # 185
    "ceiling-merged",  # 186
    "sky-other-merged",  # 187
    "cabinet-merged",  # 188
    "table-merged",  # 189
    "floor-other-merged",  # 190
    "pavement-merged",  # 191
    "mountain-merged",  # 192
    "grass-merged",  # 193
    "dirt-merged",  # 194
    "paper-merged",  # 195
    "food-other-merged",  # 196
    "building-other-merged",  # 197
    "rock-merged",  # 198
    "wall-other-merged",  # 199
    "rug-merged",  # 200
] + ["N/A"] * 49  # 201-249 are unused


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
