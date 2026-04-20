# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""COCO dataset which returns image_id for evaluation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import PIL.Image
import torch
import tqdm
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from plain_detr.datasets import transforms as T
from plain_detr.util.misc import get_local_rank, get_local_size

if TYPE_CHECKING:
    from PIL.Image import Image as PilImage

    from plain_detr.main import Config

# COCO uses non-contiguous category ids with a max of 90, so num_classes = 91.
NUM_CLASSES = 91


class CocoDetection(Dataset):
    """COCO detection dataset with optional image caching."""

    def __init__(
        self,
        img_folder: str | Path,
        ann_file: str | Path,
        transforms: T.Compose,
        return_seg_masks: bool,
        cache_mode: bool = False,
        local_rank: int = 0,
        local_size: int = 1,
    ) -> None:
        self.root = Path(img_folder)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.return_seg_masks = return_seg_masks

        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        self.cache: dict[str, bytes] = {}
        if cache_mode:
            self._cache_images()

    # -- Caching ----------------------------------------------------------------

    def _cache_images(self) -> None:
        """Pre-load image bytes into memory, sharded across local ranks."""
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]["file_name"]
            with open(self.root / path, "rb") as f:
                self.cache[path] = f.read()

    def _load_image(self, path: str) -> PilImage:
        """Load an image from disk or cache, returning an RGB PIL image."""
        if self.cache_mode:
            if path not in self.cache:
                with open(self.root / path, "rb") as f:
                    self.cache[path] = f.read()
            return PIL.Image.open(BytesIO(self.cache[path])).convert("RGB")
        return PIL.Image.open(self.root / path).convert("RGB")

    # -- Dataset interface ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = self._load_image(path)
        w, h = img.size

        target = prepare_coco_target(img_id, target, h, w, self.return_seg_masks)
        img, target = self.transforms(img, target)
        return img, target


def _convert_polys_to_seg_mask(segmentations: list, height: int, width: int) -> torch.Tensor:
    """Decode COCO polygon segmentations into a ``(N, H, W)`` binary mask tensor."""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def prepare_coco_target(
    image_id: int,
    annotations: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    return_seg_masks: bool,
) -> dict[str, Any]:
    """Convert raw COCO annotations into a training-ready target dict.

    Filters crowd annotations, converts ``[x, y, w, h]`` boxes to
    ``[x1, y1, x2, y2]``, clamps to image bounds, removes degenerate boxes,
    and optionally decodes polygon segmentations into binary masks.
    """
    anno = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=image_width)
    boxes[:, 1::2].clamp_(min=0, max=image_height)

    classes = torch.tensor([obj["category_id"] for obj in anno], dtype=torch.int64)

    if return_seg_masks:
        segmentations = [obj["segmentation"] for obj in anno]
        masks = _convert_polys_to_seg_mask(segmentations, image_height, image_width)

    keypoints = None
    if anno and "keypoints" in anno[0]:
        keypoints = torch.as_tensor([obj["keypoints"] for obj in anno], dtype=torch.float32)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]
    if return_seg_masks:
        masks = masks[keep]
    if keypoints is not None:
        keypoints = keypoints[keep]

    target: dict[str, Any] = {
        "boxes": boxes,
        "labels": classes,
        "image_id": torch.tensor([image_id]),
        "orig_size": torch.as_tensor([image_height, image_width]),
        "size": torch.as_tensor([image_height, image_width]),
    }
    if return_seg_masks:
        target["seg_masks"] = masks
    if keypoints is not None:
        target["keypoints"] = keypoints

    # for conversion to coco api
    area = torch.tensor([obj["area"] for obj in anno])
    iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]

    return target


def make_coco_transforms(image_set: str, args: Config) -> T.Compose:
    """Build the train or val augmentation pipeline (resize, crop, flip, normalize)."""
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], args.reparam),
        ]
    )

    if image_set == "train":
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    # Single scale, no randomness
    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set: str, args: Config, root: Path) -> CocoDetection:
    """Construct a :class:`CocoDetection` for the given split (``'train'`` or ``'val'``)."""
    if image_set == "train":
        img_folder = root / "train2017"
        ann_file = root / "annotations" / "instances_train2017.json"
    elif image_set == "val":
        img_folder = root / "val2017"
        ann_file = root / "annotations" / "instances_val2017.json"
    else:
        raise ValueError(f"unknown image set {image_set!r}")
    return CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_seg_masks=args.do_segmentation,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
    )
