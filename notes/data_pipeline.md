# Data Transforms Pipeline

## Overview

All transforms operate on `(image, target)` pairs using a custom `Compose` class
(`plain_detr/datasets/transforms.py`) — not `torchvision.transforms.Compose` — so
geometric transforms are applied consistently to both the image and annotations.

## Pipeline Construction

The factory function `make_coco_transforms(image_set, args)` in
`plain_detr/datasets/coco.py` builds different pipelines for training vs. evaluation.

Both pipelines share a common normalization suffix:

```python
normalize = Compose([
    ToTensor(),                                              # PIL -> float32 tensor [0, 1]
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], # ImageNet stats
              args.reparam),                                  # box coord mode
])
```

`Normalize` (`transforms.py`) does double duty: it normalizes pixel values **and**
converts boxes from `xyxy` to `cxcywh` format. If `reparam=False` (default), boxes are
also divided by image dimensions to produce relative [0, 1] coordinates; if `reparam=True`,
they stay absolute.

## Training Pipeline

```
RandomHorizontalFlip(p=0.5)
  -> RandomSelect(p=0.5):
      Branch A: RandomResize(scales=[480,512,...,800], max_size=1333)
      Branch B: RandomResize([400,500,600])
                -> RandomSizeCrop(384, 600)
                -> RandomResize(scales=[480,512,...,800], max_size=1333)
  -> ToTensor + Normalize
```

1. **RandomHorizontalFlip** — 50% chance of flipping image and mirroring boxes.
2. **RandomSelect** — 50/50 between:
   - **Branch A**: simple multi-scale resize (min-side randomly chosen from 11 values in
     480-800, step 32; max-side capped at 1333).
   - **Branch B**: resize to a small scale (400/500/600), random crop with dimensions in
     [384, 600], then resize to the final multi-scale range. This simulates a "zoom-in"
     augmentation.
3. **Normalize** — ToTensor + ImageNet normalization + box format conversion.

No color jitter, cutout, mixup, or other photometric augmentations are used.
`RandomErasing` is defined in `transforms.py` but not included in any pipeline.

## Evaluation Pipeline

```
RandomResize([800], max_size=1333)   # deterministic (single-element list)
  -> ToTensor + Normalize
```

Despite the class name "RandomResize", with a single-element list this is **deterministic**
— always resizes so the shorter side is 800 pixels (or smaller if the longer side would
exceed 1333). No flip, no crop, no randomness.

## Invocation Flow

The full call chain at `__getitem__` time (`datasets/coco.py`):

1. **Load**: `CocoDetection.__getitem__` loads the PIL image from disk (or from the
   in-memory cache when `cache_mode` is enabled) and fetches raw COCO annotations via
   `pycocotools`.
2. **Prepare**: `prepare_coco_target()` (`coco.py`) parses raw annotations — converts
   `[x, y, w, h]` to `xyxy`, filters crowd annotations, removes degenerate boxes,
   optionally decodes polygon masks.
3. **Transform**: the augmentation + normalization pipeline is applied. Image enters as PIL,
   exits as a normalized tensor. Target dict is mutated accordingly.
4. **Batch**: the DataLoader's custom `collate_fn` (`util/misc.py:263`) pads images of
   different sizes to the batch maximum and wraps them in a `NestedTensor` with a padding
   mask.

## Training vs. Evaluation Summary

| Aspect             | Training                | Eval      |
|--------------------|-------------------------|-----------|
| Horizontal flip    | p=0.5                   | No        |
| Resize scales      | Random from [480..800]  | Fixed 800 |
| Crop augmentation  | 50% chance              | Never     |
| max_size           | 1333                    | 1333      |
| Normalize          | Same                    | Same      |

## All Transform Classes and Functions

### Standalone Functions (`transforms.py`)

| Function                      | Description                                                                                                     |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `crop(image, target, region)` | Crops image and adjusts all target fields. Boxes shifted, clamped, area recomputed. Objects with area <= 1.0 removed. Seg masks sliced. |
| `hflip(image, target)`        | Horizontal flip. Swaps box x-coordinates. Flips seg masks along last dimension.                                 |
| `resize(image, target, size, max_size)` | Aspect-ratio-preserving resize. Scales boxes and areas by width/height ratios. Interpolates seg masks with nearest-neighbor. |
| `pad(image, target, padding)` | Pads image on bottom-right. Pads seg masks correspondingly. Updates `target["size"]`.                           |

### Transform Classes (`transforms.py`)

| Class                 | Parameters                              | Description                                                                                 |
|-----------------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| `RandomCrop`          | `size: (h, w)`                          | Random crop of fixed size. Delegates to `crop()`.                                           |
| `RandomSizeCrop`      | `min_size: int, max_size: int`          | Random crop with random dimensions in [min_size, min(img_dim, max_size)]. Delegates to `crop()`. |
| `CenterCrop`          | `size: (h, w)`                          | Center crop of fixed size. Delegates to `crop()`.                                           |
| `RandomHorizontalFlip`| `p=0.5`                                 | Flips horizontally with probability `p`. Delegates to `hflip()`.                            |
| `RandomResize`        | `sizes: list[int], max_size: int\|None` | Randomly picks min-side from `sizes`, resizes keeping aspect ratio. Delegates to `resize()`. |
| `RandomPad`           | `max_pad: int`                          | Pads bottom-right by random amounts in [0, max_pad]. Delegates to `pad()`.                  |
| `RandomSelect`        | `transforms1, transforms2, p=0.5`      | With probability `p` applies `transforms1`, otherwise `transforms2`.                        |
| `ToTensor`            | (none)                                  | PIL Image to float32 tensor via `torchvision.transforms.functional.to_tensor`.              |
| `RandomErasing`       | forwarded to `T.RandomErasing`          | Wraps torchvision RandomErasing. **Not used in any pipeline.**                              |
| `Normalize`           | `mean, std, reparam=False`              | ImageNet normalization + box format conversion (xyxy -> cxcywh, optional coord normalization). |
| `Compose`             | `transforms: list`                      | Sequentially applies transforms, threading `(image, target)` through each.                  |

### Pre-processing (`coco.py`)

| Function                     | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `prepare_coco_target()`      | Parses raw COCO annotations into structured target dict. Filters crowd annotations, converts bbox format, removes degenerate boxes, optionally decodes polygon segmentations. Applied **before** any augmentation. |

## Design Notes

- **`crop()` filters objects with area <= 1.0** (`transforms.py`), not zero — prevents
  near-degenerate boxes from persisting.
- **`RandomPad` and `CenterCrop`** are defined but unused in any pipeline.
- **`reparam` flag** is a global switch that affects not just `Normalize` but also the loss
  function, matcher, decoder, and postprocessor throughout the codebase.
- **Panoptic** (`coco_panoptic.py`) reuses `make_coco_transforms` identically; the only
  difference is how annotations are loaded (from PNG masks instead of polygon annotations).
- **`CocoDetection`** (`datasets/coco.py`) extends `torch.utils.data.Dataset` directly
  and includes optional image caching (enabled via `cache_mode`).
