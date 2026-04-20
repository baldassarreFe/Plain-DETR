# COCO vs ZOD dataset comparison

Analysis run on the full datasets using `scripts/analyze_coco_dataset.py`.

ZOD annotations were converted to COCO format using `scripts/convert_zod_to_coco.py`
(90/10 train/val split, seed 42). The converted annotations live at
`zod/annotations/instances_{train,val}.json`.

## Overview

| Metric                     | COCO train    | COCO val  | ZOD train     | ZOD val   |
|----------------------------|---------------|-----------|---------------|-----------|
| Images                     | 118,287       | 5,000     | 90,000        | 10,000    |
| Annotations                | 860,001       | 36,781    | 6,182,825     | 681,113   |
| Categories                 | 80            | 80        | 10            | 10        |
| Empty images               | 1,021         | 48        | 0             | 0         |
| Avg annotations/image      | 7.3           | 7.4       | 68.7          | 68.1      |

## Image resolution

| Metric     | COCO train        | ZOD train        |
|------------|-------------------|------------------|
| Width      | mean=578, variable| 3848 (fixed)     |
| Height     | mean=484, variable| 2168 (fixed)     |
| Megapixels | ~0.28 MP          | 8.34 MP          |

All ZOD images are exactly 3848x2168 (from a single front-facing camera).
COCO images have varying resolutions sourced from Flickr.

## Categories

### COCO (80 categories, top 10 shown)

| Category       | Count   | Pct    |
|----------------|---------|--------|
| person         | 262,465 | 30.5%  |
| car            | 43,867  | 5.1%   |
| chair          | 38,491  | 4.5%   |
| book           | 24,715  | 2.9%   |
| bottle         | 24,342  | 2.8%   |
| cup            | 20,650  | 2.4%   |
| dining table   | 15,714  | 1.8%   |
| bowl           | 14,358  | 1.7%   |
| traffic light  | 12,884  | 1.5%   |
| handbag        | 12,354  | 1.4%   |

COCO is heavily dominated by "person" (30.5%).

### ZOD (10 categories, all shown)

| Category         | Count     | Pct    |
|------------------|-----------|--------|
| Vehicle          | 1,551,811 | 25.1%  |
| TrafficSign      | 1,548,733 | 25.0%  |
| PoleObject       | 1,432,215 | 23.2%  |
| TrafficGuide     | 567,758   | 9.2%   |
| TrafficSignal    | 523,900   | 8.5%   |
| Pedestrian       | 323,288   | 5.2%   |
| VulnerableVehicle| 206,775   | 3.3%   |
| TrafficBeacon    | 21,797    | 0.4%   |
| DynamicBarrier   | 3,486     | 0.1%   |
| Animal           | 3,062     | 0.0%   |

ZOD's top 3 categories are nearly equally distributed (~25% each).
Vehicle has subtypes: Car, Van, Truck, Bus, Trailer, HeavyEquip, TramTrain.
VulnerableVehicle has subtypes: Bicycle, Motorcycle, Stroller, Wheelchair, PersonalTransporter.

## Bounding box statistics (train splits)

| Metric                       | COCO train       | ZOD train       |
|------------------------------|------------------|-----------------|
| **Area (px^2)**              |                  |                 |
| mean                         | 12,026           | 12,492          |
| median                       | 1,697            | 766             |
| p5 / p95                     | 60 / 61,255      | 52 / 29,013     |
| **Relative area (% of img)**|                  |                 |
| mean                         | 4.3%             | 0.1%            |
| median                       | 0.6%             | ~0.01%          |
| **Width (px)**               |                  |                 |
| mean / median                | 104 / 54         | 49 / 19         |
| **Height (px)**              |                  |                 |
| mean / median                | 107 / 62         | 89 / 41         |
| **Aspect ratio (w/h)**       |                  |                 |
| mean / median                | 1.2 / 0.9        | 0.9 / 0.7       |

ZOD objects are smaller in both absolute and relative terms. The median absolute area
is ~2x smaller, and the median relative area is ~60x smaller (because ZOD images are
30x larger in pixel count).

## Size distribution (COCO thresholds: small < 32^2, large >= 96^2)

| Size bucket | COCO train         | ZOD train          |
|-------------|--------------------|--------------------|
| Small       | 356,340 (41.4%)    | 3,452,215 (55.8%)  |
| Medium      | 295,163 (34.3%)    | 1,976,254 (32.0%)  |
| Large       | 208,498 (24.2%)    | 754,356 (12.2%)    |

ZOD has a much heavier tail of small objects—typical for driving scenes where many
objects (signs, poles, distant vehicles) are far from the camera.

## Annotations per image

| Metric     | COCO train | ZOD train  |
|------------|------------|------------|
| mean       | 7.3        | 68.7       |
| median     | 4          | 60         |
| p5 / p95   | 1 / 22     | 10 / 158   |
| max        | 93         | 446        |

ZOD images are ~10x more densely annotated.

## Key takeaways

- ZOD is a **driving-domain** dataset (10 classes) vs COCO's **general-purpose** 80 classes.
- ZOD has **~7x more annotations** than COCO despite fewer images, due to dense urban driving scenes.
- ZOD images are **fixed 3848x2168** (8.3 MP) vs COCO's variable ~0.28 MP — this may require
  adjusting input resolution / training augmentation strategies.
- ZOD is **dominated by small objects** (55.8% vs 41.4%) — detection performance on small objects
  will be critical.
- ZOD's bounding boxes are originally **quadrilateral** (4 oriented corner points); the COCO
  conversion uses axis-aligned bounding boxes (AABB) derived from the min/max of the corners.
  A small number of annotations with non-standard point counts (3, 5, 6, etc.) were skipped.
- ZOD is supported via a dedicated loader in `plain_detr/datasets/zod.py` that reuses
  `CocoDetection` with ZOD-specific paths (`single_frames/` image root, `instances_train.json` /
  `instances_val.json` annotation files). Select it with `--args.dataset_name zod`.
