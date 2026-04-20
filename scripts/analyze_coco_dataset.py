#!/usr/bin/env python3
"""Analyze a COCO-format dataset and print statistics useful for comparison.

Reports:
  - Dataset overview (images, annotations, categories)
  - Per-category annotation counts and percentage
  - Image resolution statistics
  - Bounding box area statistics (absolute and relative to image area)
  - Box aspect ratio statistics
  - Annotations per image statistics
  - Small / medium / large object distribution (COCO absolute thresholds: 32^2, 96^2)
  - Small / medium / large object distribution (relative thresholds: 10%, 40%)
  - Per-category relative-area breakdown

Usage:
    python scripts/analyze_coco_dataset.py \
        --annotation-file /path/to/instances_train.json \
        --name "COCO train2017"

    # Compare two datasets side by side:
    python scripts/analyze_coco_dataset.py \
        --annotation-file /path/to/coco/instances_train2017.json /path/to/zod/instances_train.json \
        --name "COCO train2017" "ZOD train"
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# COCO absolute-area thresholds (px^2)
SMALL_ABS_THRESHOLD = 32**2  # 1024
LARGE_ABS_THRESHOLD = 96**2  # 9216

# Relative-area thresholds (fraction of image area)
SMALL_REL_THRESHOLD = 0.10  # 10%
LARGE_REL_THRESHOLD = 0.40  # 40%


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total > 0 else "  N/A"


def compute_stats(annotation_file: Path, name: str) -> dict:
    """Load a COCO annotation file and return raw DataFrames + metadata."""
    logger.info(f"Loading {annotation_file}...")
    with open(annotation_file) as f:
        data = json.load(f)

    categories = data["categories"]
    images = data["images"]
    annotations = data["annotations"]

    cat_id_to_name: dict[int, str] = {cat["id"]: cat["name"] for cat in categories}

    # --- images dataframe ---
    img_df = pd.DataFrame(images)[["id", "width", "height"]]
    img_df = img_df.rename(columns={"id": "image_id"})
    img_df["image_area"] = img_df["width"] * img_df["height"]

    # --- annotations dataframe ---
    ann_records: list[dict] = []
    for ann in annotations:
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name is None:
            continue
        bbox = ann["bbox"]  # [x, y, w, h]
        bw, bh = bbox[2], bbox[3]
        area = ann.get("area", bw * bh)
        ann_records.append(
            {
                "image_id": ann["image_id"],
                "category": cat_name,
                "box_w": bw,
                "box_h": bh,
                "area": area,
            }
        )

    ann_df = pd.DataFrame(ann_records)

    # merge image dims onto annotations
    ann_df = ann_df.merge(img_df[["image_id", "image_area"]], on="image_id", how="left")
    ann_df["relative_area"] = ann_df["area"] / ann_df["image_area"]
    ann_df["aspect_ratio"] = np.where(ann_df["box_h"] > 0, ann_df["box_w"] / ann_df["box_h"], np.nan)

    # annotations per image (include images with 0 annotations)
    ann_per_img = ann_df.groupby("image_id").size().reindex(img_df["image_id"], fill_value=0)

    return {
        "name": name,
        "categories": categories,
        "img_df": img_df,
        "ann_df": ann_df,
        "ann_per_img": ann_per_img,
    }


def _summary_block(label: str, arr: np.ndarray, unit: str = "") -> str:
    """Compact distribution summary using numpy percentiles."""
    if len(arr) == 0:
        return f"  {label}: (no data)"
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    fmt = lambda v: f"{v:,.1f}"  # noqa: E731
    return (
        f"  {label}:\n"
        f"    mean={fmt(arr.mean())}{unit}  median={fmt(p50)}{unit}\n"
        f"    p5={fmt(p5)}{unit}  p25={fmt(p25)}{unit}  p75={fmt(p75)}{unit}  p95={fmt(p95)}{unit}\n"
        f"    min={fmt(arr.min())}{unit}  max={fmt(arr.max())}{unit}"
    )


def print_report(stats: dict) -> None:
    """Print a formatted report for a single dataset."""
    name = stats["name"]
    img_df: pd.DataFrame = stats["img_df"]
    ann_df: pd.DataFrame = stats["ann_df"]
    ann_per_img: pd.Series = stats["ann_per_img"]
    categories = stats["categories"]

    n_images = len(img_df)
    n_anns = len(ann_df)
    n_cats = len(categories)
    n_empty = int((ann_per_img == 0).sum())

    print(f"\n{'=' * 70}")
    print(f"  Dataset: {name}")
    print(f"{'=' * 70}")

    # ---- Overview ----
    print("\n--- Overview ---")
    print(f"  Images:      {n_images:,}")
    print(f"  Annotations: {n_anns:,}")
    print(f"  Categories:  {n_cats}")
    print(f"  Empty images (no annotations): {n_empty:,}")
    if n_images > 0:
        print(f"  Avg annotations/image: {n_anns / n_images:.1f}")

    # ---- Per-category counts ----
    print("\n--- Per-category annotation counts ---")
    cat_counts = ann_df["category"].value_counts()
    for cat_name, count in cat_counts.items():
        print(f"  {cat_name:<25s} {count:>8,}  ({_pct(count, n_anns)})")

    # ---- Image resolution ----
    print("\n--- Image resolution ---")
    print(_summary_block("Width", img_df["width"].to_numpy(dtype=np.float64), "px"))
    print(_summary_block("Height", img_df["height"].to_numpy(dtype=np.float64), "px"))

    # ---- Bounding box statistics ----
    print("\n--- Bounding box statistics ---")
    print(_summary_block("Area (px^2)", ann_df["area"].to_numpy()))
    print(_summary_block("Relative area (% of image)", ann_df["relative_area"].to_numpy() * 100, "%"))
    print(_summary_block("Width (px)", ann_df["box_w"].to_numpy()))
    print(_summary_block("Height (px)", ann_df["box_h"].to_numpy()))
    ar = ann_df["aspect_ratio"].dropna().to_numpy()
    print(_summary_block("Aspect ratio (w/h)", ar))

    # ---- Absolute size distribution (COCO thresholds) ----
    print("\n--- Size distribution (COCO absolute thresholds) ---")
    areas = ann_df["area"].to_numpy()
    n_small = int((areas < SMALL_ABS_THRESHOLD).sum())
    n_medium = int(((areas >= SMALL_ABS_THRESHOLD) & (areas < LARGE_ABS_THRESHOLD)).sum())
    n_large = int((areas >= LARGE_ABS_THRESHOLD).sum())
    total = n_small + n_medium + n_large
    if total > 0:
        print(f"  Small  (area < {SMALL_ABS_THRESHOLD:,}px^2):   {n_small:>8,}  ({_pct(n_small, total)})")
        print(
            f"  Medium ({SMALL_ABS_THRESHOLD:,} <= area < {LARGE_ABS_THRESHOLD:,}):  {n_medium:>8,}  ({_pct(n_medium, total)})"
        )
        print(f"  Large  (area >= {LARGE_ABS_THRESHOLD:,}px^2):  {n_large:>8,}  ({_pct(n_large, total)})")

    # ---- Relative size distribution ----
    print("\n--- Size distribution (relative thresholds) ---")
    rel = ann_df["relative_area"].to_numpy()
    n_rel_small = int((rel < SMALL_REL_THRESHOLD).sum())
    n_rel_medium = int(((rel >= SMALL_REL_THRESHOLD) & (rel < LARGE_REL_THRESHOLD)).sum())
    n_rel_large = int((rel >= LARGE_REL_THRESHOLD).sum())
    total_rel = n_rel_small + n_rel_medium + n_rel_large
    if total_rel > 0:
        print(f"  Small  (rel_area < {SMALL_REL_THRESHOLD:.0%}):  {n_rel_small:>8,}  ({_pct(n_rel_small, total_rel)})")
        print(
            f"  Medium ({SMALL_REL_THRESHOLD:.0%} <= rel_area < {LARGE_REL_THRESHOLD:.0%}): {n_rel_medium:>8,}  ({_pct(n_rel_medium, total_rel)})"
        )
        print(f"  Large  (rel_area >= {LARGE_REL_THRESHOLD:.0%}):  {n_rel_large:>8,}  ({_pct(n_rel_large, total_rel)})")

    # ---- Per-category relative-area breakdown ----
    print("\n--- Per-category relative-area breakdown ---")
    print(f"  {'Category':<25s} {'Small (<10%)':<16s} {'Medium':<16s} {'Large (>=40%)':<16s} {'Median rel%':>11s}")
    for cat_name in cat_counts.index:
        cat_rel = ann_df.loc[ann_df["category"] == cat_name, "relative_area"].to_numpy()
        cs = int((cat_rel < SMALL_REL_THRESHOLD).sum())
        cm = int(((cat_rel >= SMALL_REL_THRESHOLD) & (cat_rel < LARGE_REL_THRESHOLD)).sum())
        cl = int((cat_rel >= LARGE_REL_THRESHOLD).sum())
        ct = cs + cm + cl
        med_pct = float(np.median(cat_rel) * 100)
        print(
            f"  {cat_name:<25s} {cs:>7,} ({_pct(cs, ct)}) {cm:>7,} ({_pct(cm, ct)}) {cl:>7,} ({_pct(cl, ct)}) {med_pct:>10.2f}%"
        )

    # ---- Annotations per image ----
    print("\n--- Annotations per image ---")
    print(_summary_block("Count", ann_per_img.to_numpy(dtype=np.float64)))

    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Analyze COCO-format dataset(s)")
    parser.add_argument(
        "--annotation-file",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to COCO annotation JSON file(s)",
    )
    parser.add_argument(
        "--name",
        type=str,
        nargs="+",
        required=True,
        help="Name(s) for the dataset(s), one per annotation file",
    )
    args = parser.parse_args()

    if len(args.annotation_file) != len(args.name):
        raise ValueError(
            f"Number of annotation files ({len(args.annotation_file)}) must match number of names ({len(args.name)})"
        )

    all_stats: list[dict] = []
    for ann_file, name in zip(args.annotation_file, args.name):
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        all_stats.append(compute_stats(ann_file, name))

    for stats in all_stats:
        print_report(stats)


if __name__ == "__main__":
    main()
