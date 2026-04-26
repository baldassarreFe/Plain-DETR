# First ZOD training run: DINOv3 ViT-Base + BoxRPE

Date: 2026-04-25

## Setup

- Config: `configs/dinov3_vit_base_boxrpe.sh` (COCO defaults)
- Dataset: ZOD (`--args.dataset_name zod`)
- Hardware: 8x H100 80GB
- Global batch size: 16 (batch_size=2 per GPU)
- Precision: bf16
- Epochs: 12, LR drop at epoch 11
- Output: `exps/dinov3_vit_base_boxrpe_zod/`

All hyperparameters were transferred directly from the COCO config with no
ZOD-specific tuning. The only changes were `dataset_name`, `data_dir`,
`batch_size`, and `amp_dtype`.

## Results

### AP progression

| Epoch | AP   | AP50 | AP75 | APs  | APm  | APl  | Train Loss |
|-------|------|------|------|------|------|------|------------|
| 0     | 3.7  | 10.6 | 1.9  | 0.6  | 5.4  | 14.9 | 32.29      |
| 1     | 5.3  | 13.5 | 3.2  | 0.7  | 8.1  | 20.6 | 28.50      |
| 2     | 6.2  | 15.9 | 3.8  | 1.0  | 10.4 | 24.3 | 27.70      |
| 3     | 6.7  | 16.8 | 4.4  | 1.1  | 11.8 | 25.3 | 27.32      |
| 4     | 7.3  | 18.1 | 4.8  | 1.3  | 13.1 | 26.7 | 27.01      |
| 5     | 7.4  | 18.1 | 4.9  | 1.3  | 13.1 | 27.5 | 26.82      |
| 6     | 7.7  | 18.8 | 5.2  | 1.4  | 13.6 | 27.5 | 26.62      |
| 7     | 7.6  | 18.9 | 5.0  | 1.3  | 13.8 | 28.7 | 26.35      |
| 8     | 7.8  | 19.2 | 5.3  | 1.5  | 13.6 | 27.6 | 26.13      |
| 9     | 8.1  | 19.6 | 5.4  | 1.6  | 13.9 | 28.9 | 26.28      |
| 10    | 8.4  | 19.9 | 6.0  | 1.7  | 14.2 | 29.8 | 26.02      |
| 11    | 9.8  | 22.8 | 7.1  | 2.2  | 17.1 | 33.3 | 24.81      |

LR drop at epoch 11 gave +1.4 AP (8.4 -> 9.8).

### Comparison with COCO runs

| Experiment                    | Dataset | Backbone         | Final AP |
|-------------------------------|---------|------------------|----------|
| `swinv2_small_mim_pt_boxrpe`  | COCO    | SwinV2-Small     | 49.8     |
| `dinov3_vit_small_boxrpe`     | COCO    | DINOv3 ViT-Small | 49.7     |
| `dinov3_vit_base_boxrpe`      | COCO    | DINOv3 ViT-Base  | 53.3     |
| `dinov3_vit_base_boxrpe_zod`  | ZOD     | DINOv3 ViT-Base  | 9.8      |

## Root cause analysis

### 1. Resolution is catastrophically low (critical)

ZOD images are 3848x2168 (8.3 MP). The COCO transforms resize them to
1332x751 (max_size=1333 clamps). After the ViT stride-16 backbone, the
feature map is 83x47.

66% of all annotations (PoleObject, TrafficSign, TrafficSignal,
TrafficGuide, TrafficBeacon) have median dimensions that are sub-pixel
in the feature map after resize. Concrete examples after resize to 1332x751:

| Category       | Median W (px) | Median H (px) | Feature cells W | Feature cells H |
|----------------|---------------|---------------|-----------------|-----------------|
| TrafficSign    | 6.5           | 6.2           | 0.41            | 0.39            |
| TrafficSignal  | 4.4           | 9.1           | 0.28            | 0.57            |
| PoleObject     | 2.6           | 43.0          | 0.17            | 2.68            |
| TrafficGuide   | 2.7           | 11.6          | 0.17            | 0.72            |

These objects are essentially invisible to the model.

Additionally, the training scales [480..800] with max_size=1333 lose augmentation
diversity for ZOD's aspect ratio: the top 3 scales (736, 768, 800) all produce
the same 1332x751 output because max_size clamps them.

### 2. topk=100 and maxDets=[1,10,100] are too low

ZOD has a median of 60 objects/image and up to 446. At eval, topk=100 caps
predictions at 100 per image, and COCOeval's default maxDets=[1,10,100]
limits AP/AR computation at 100 detections.

### 3. Crop augmentation is mismatched

The 50% crop branch first downscales to 400-600px short side, then crops
a 384-600px patch. For a 3848x2168 image this represents ~10-15% of the
original through a tiny window. Many objects are partially cropped and the
scale distribution within the crop is unnatural.

### 4. Training schedule is short

12 epochs with ~7x more annotations per image than COCO. The model sees
more diverse objects per epoch, but gains were still consistent at epoch 10
before the LR drop.

### 5. Proposal size priors are too large

The finest proposal level (stride 8) initializes boxes at 0.05 normalized
width/height = ~67x38 px. Median ZOD object after resize is ~23x28 px. The
smallest proposals are already 2-3x larger than most objects. Higher
resolution partially mitigates this.

### 6. No per-class AP logging

With 500:1 class imbalance (Vehicle vs Animal), aggregate AP may mask
class-specific failures. Lower priority than resolution.

## Plan for next run

Config: `configs/dinov3_vit_base_boxrpe_zod.sh`

Key changes from the COCO baseline:

| Parameter        | COCO (baseline)          | ZOD (next run)                     |
|------------------|--------------------------|------------------------------------|
| `max_size`       | 1333                     | 2500                               |
| `train_min_sizes`| [480..800, step 32]      | [900..1500, step 60]               |
| `val_min_size`   | 800                      | 1500                               |
| `crop_scales`    | [400, 500, 600]          | [800, 1000, 1200]                  |
| `crop_min_size`  | 384                      | 700                                |
| `crop_max_size`  | 600                      | 1200                               |
| `topk`           | 100                      | 300                                |
| `max_dets`       | [1, 10, 100]             | [1, 10, 300]                       |
| `epochs`         | 12                       | 24                                 |
| `lr_drop`        | 11                       | 22                                 |

Expected impact:
- Resolution increase should produce the largest AP gain by making 66% of
  currently-invisible objects detectable.
- topk/maxDets increase removes the eval ceiling on dense images.
- Longer schedule takes more time to converge
