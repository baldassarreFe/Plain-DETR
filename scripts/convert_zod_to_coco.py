#!/usr/bin/env python3
"""Convert ZOD (Zenseact Open Dataset) object detection annotations to COCO format.

ZOD annotations use quadrilateral bounding boxes (4 corner points per object).
This script converts them to axis-aligned bounding boxes in COCO format.

Usage:
    python scripts/convert_zod_to_coco.py \
        --zod-root path/to/zod \
        --output-dir path/to/zod/annotations \
        --val-fraction 0.1 \
        --seed 42

Output:
    <output-dir>/instances_train.json
    <output-dir>/instances_val.json
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ZOD class -> (COCO-style category id, supercategory)
# We keep the ZOD class hierarchy. "type" subfield is stored as an attribute but
# does NOT create separate categories — mirrors how ZOD itself treats them.
ZOD_CATEGORIES: list[dict] = [
    {"id": 1, "name": "Vehicle", "supercategory": "vehicle"},
    {"id": 2, "name": "VulnerableVehicle", "supercategory": "vehicle"},
    {"id": 3, "name": "Pedestrian", "supercategory": "person"},
    {"id": 4, "name": "Animal", "supercategory": "animal"},
    {"id": 5, "name": "PoleObject", "supercategory": "infrastructure"},
    {"id": 6, "name": "TrafficSign", "supercategory": "traffic"},
    {"id": 7, "name": "TrafficSignal", "supercategory": "traffic"},
    {"id": 8, "name": "TrafficGuide", "supercategory": "traffic"},
    {"id": 9, "name": "TrafficBeacon", "supercategory": "traffic"},
    {"id": 10, "name": "DynamicBarrier", "supercategory": "infrastructure"},
]

# Lookup: class name -> category id
_CLASS_TO_ID: dict[str, int] = {cat["name"]: cat["id"] for cat in ZOD_CATEGORIES}

# Classes to skip (not meaningful for object detection)
_SKIP_CLASSES: set[str] = {"Inconclusive"}


def quadrilateral_to_aabb(coords: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert 4-corner quadrilateral to axis-aligned bounding box.

    Args:
        coords: List of 4 [x, y] points.

    Returns:
        (x, y, width, height) in COCO format (top-left corner + size).
    """
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def parse_frame(
    frame_dir: Path,
    image_id: int,
    annotation_id_start: int,
) -> tuple[dict | None, list[dict], int]:
    """Parse a single ZOD frame directory.

    Returns:
        (image_info, list_of_annotations, next_annotation_id)
        image_info is None if the frame has no image or annotation file.
    """
    ann_file = frame_dir / "annotations" / "object_detection.json"
    img_dir = frame_dir / "camera_front_blur"

    if not ann_file.exists() or not img_dir.exists():
        return None, [], annotation_id_start

    # Find the image file
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        return None, [], annotation_id_start
    img_file = img_files[0]

    # Build image info — ZOD images are consistently 3848x2168 but we record
    # the actual filename so consumers can resolve it.
    image_info = {
        "id": image_id,
        "file_name": f"{frame_dir.name}/camera_front_blur/{img_file.name}",
        "width": 3848,
        "height": 2168,
    }

    # Parse annotations
    with open(ann_file) as f:
        raw_anns = json.load(f)

    annotations: list[dict] = []
    ann_id = annotation_id_start

    for raw in raw_anns:
        props = raw["properties"]
        cls_name = props["class"]

        if cls_name in _SKIP_CLASSES:
            continue

        cat_id = _CLASS_TO_ID.get(cls_name)
        if cat_id is None:
            logger.warning(f"Unknown class '{cls_name}' in frame {frame_dir.name}, skipping")
            continue

        coords = raw["geometry"]["coordinates"]
        if len(coords) != 4:
            logger.warning(f"Expected 4 points, got {len(coords)} in frame {frame_dir.name}, skipping")
            continue

        x, y, w, h = quadrilateral_to_aabb(coords)

        # Skip degenerate boxes
        if w < 1 or h < 1:
            continue

        ann_dict = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
        }

        # Store useful ZOD-specific attributes
        subtype = props.get("type")
        if subtype and subtype != "N/A":
            ann_dict["zod_subtype"] = subtype

        occlusion = props.get("occlusion_ratio")
        if occlusion and occlusion != "N/A":
            ann_dict["zod_occlusion"] = occlusion

        annotations.append(ann_dict)
        ann_id += 1

    return image_info, annotations, ann_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ZOD to COCO format")
    parser.add_argument("--zod-root", type=Path, required=True, help="Root of the ZOD dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write COCO JSON files")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of frames for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    frames_dir = args.zod_root / "single_frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Expected single_frames directory at {frames_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all frame directories
    frame_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(frame_dirs)} frame directories")

    # Split into train/val
    rng = random.Random(args.seed)
    shuffled = list(frame_dirs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_dirs = set(d.name for d in shuffled[:n_val])

    logger.info(f"Split: {len(frame_dirs) - n_val} train, {n_val} val")

    # Process all frames
    train_images: list[dict] = []
    train_annotations: list[dict] = []
    val_images: list[dict] = []
    val_annotations: list[dict] = []

    image_id = 1
    ann_id = 1
    skipped = 0

    for i, frame_dir in enumerate(frame_dirs):
        if (i + 1) % 10000 == 0:
            logger.info(f"Processing frame {i + 1}/{len(frame_dirs)}...")

        img_info, anns, ann_id = parse_frame(frame_dir, image_id, ann_id)

        if img_info is None:
            skipped += 1
            continue

        if frame_dir.name in val_dirs:
            val_images.append(img_info)
            val_annotations.extend(anns)
        else:
            train_images.append(img_info)
            train_annotations.extend(anns)

        image_id += 1

    logger.info(f"Skipped {skipped} frames (missing image or annotations)")
    logger.info(f"Train: {len(train_images)} images, {len(train_annotations)} annotations")
    logger.info(f"Val: {len(val_images)} images, {len(val_annotations)} annotations")

    # Build COCO dicts
    info = {
        "description": "Zenseact Open Dataset (ZOD) - Object Detection",
        "url": "https://zod.zenseact.com",
        "version": "1.0",
        "year": 2023,
        "contributor": "Zenseact",
    }

    for split_name, images, annotations in [
        ("train", train_images, train_annotations),
        ("val", val_images, val_annotations),
    ]:
        coco_dict = {
            "info": info,
            "licenses": [],
            "categories": ZOD_CATEGORIES,
            "images": images,
            "annotations": annotations,
        }
        out_path = args.output_dir / f"instances_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(coco_dict, f)
        logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
