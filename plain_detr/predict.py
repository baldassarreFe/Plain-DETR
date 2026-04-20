"""Run inference with a trained Plain-DETR model.

CLI usage::

    python -m plain_detr.predict \
        --checkpoint exps/swinv2_small_mim_pt_boxrpe/checkpoint.epoch_11.pth \
        --threshold 0.5 \
        image1.jpg image2.png dir/image3.jpg

Programmatic usage::

    from plain_detr.predict import load_model, predict_image

    model, postprocess, config = load_model("checkpoint.epoch_11.pth")
    result = predict_image(model, postprocess, config, image)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path  # noqa: TC003 -- cyclopts resolves annotations at runtime
from typing import TYPE_CHECKING

import cyclopts
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch
import torchvision.transforms.functional as F

from plain_detr.datasets import get_category_names, get_num_classes
from plain_detr.models.detr import PostProcess
from plain_detr.models.detr import build as build_model
from plain_detr.util.misc import nested_tensor_from_tensor_list

if TYPE_CHECKING:
    from plain_detr.main import Config

logger = logging.getLogger(__name__)

# Distinct colours for up to 20 categories, then cycling.
_PALETTE: list[str] = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
]


def _color_for_label(label: int) -> str:
    return _PALETTE[label % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Preprocessing - mirrors the val transform in datasets/coco.py
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_resize_shape(
    image_width: int,
    image_height: int,
    target_min_size: int,
    max_size: int,
) -> tuple[int, int]:
    """Compute (new_h, new_w) preserving aspect ratio.

    The shortest side is scaled to *target_min_size* unless the longest side
    would exceed *max_size*, in which case it is clamped.
    """
    min_original = float(min(image_width, image_height))
    max_original = float(max(image_width, image_height))

    # Clamp so the longest side doesn't exceed max_size.
    effective_size = target_min_size
    if max_original / min_original * target_min_size > max_size:
        effective_size = int(round(max_size * min_original / max_original))

    if image_width < image_height:
        new_w = effective_size
        new_h = min(int(effective_size * image_height / image_width), max_size)
    else:
        new_h = effective_size
        new_w = min(int(effective_size * image_width / image_height), max_size)

    return new_h, new_w


def preprocess(
    image: PIL.Image.Image,
    min_size: int = 800,
    max_size: int = 1333,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize and normalize an image for inference.

    Returns the preprocessed tensor ``[C, H, W]`` and the original image size
    ``(width, height)``.
    """
    original_size = image.size  # (w, h)

    new_h, new_w = _get_resize_shape(image.size[0], image.size[1], min_size, max_size)
    resized = image.resize((new_w, new_h), resample=PIL.Image.Resampling.BILINEAR)
    tensor = F.to_tensor(resized)
    tensor = F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return tensor, original_size


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _try_load_font(size: int = 14) -> PIL.ImageFont.ImageFont | PIL.ImageFont.FreeTypeFont:
    """Try to load a TrueType font, fall back to the default bitmap font."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        try:
            return PIL.ImageFont.truetype(path, size)
        except OSError:
            continue
    return PIL.ImageFont.load_default()


def draw_predictions(
    image: PIL.Image.Image,
    scores: torch.Tensor,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    category_names: list[str],
) -> PIL.Image.Image:
    """Draw bounding boxes, labels, and scores on *image* (modified in place and returned).

    Args:
        image: RGB PIL image.
        scores: ``[N]`` confidence scores.
        labels: ``[N]`` integer class labels.
        boxes: ``[N, 4]`` boxes in ``(x1, y1, x2, y2)`` absolute coordinates.
        category_names: list mapping label index to human-readable name.

    Returns:
        The annotated image.
    """
    draw = PIL.ImageDraw.Draw(image)
    font = _try_load_font()

    for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        x1, y1, x2, y2 = box
        color = _color_for_label(label)

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label text
        name = category_names[label] if label < len(category_names) else f"class_{label}"
        text = f"{name} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Background rectangle for readability
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), text, fill="white", font=font)

    return image


def prediction_output_path(image_path: Path) -> Path:
    """Derive the output path from an input image path.

    ``photo.jpg`` becomes ``photo.pred.jpg``.
    """
    return image_path.with_suffix(f".pred{image_path.suffix}")


# ---------------------------------------------------------------------------
# Reusable API
# ---------------------------------------------------------------------------


def load_model(checkpoint: Path, topk: int = 100) -> tuple[torch.nn.Module, PostProcess, Config]:
    """Load a trained Plain-DETR model from a checkpoint.

    Args:
        checkpoint: Path to a ``checkpoint.epoch_N.pth`` file.
        topk: Number of top-scoring predictions the post-processor keeps.

    Returns:
        A tuple of ``(model, postprocess, config)`` ready for inference.
        The model is already on CUDA in eval mode.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    logger.info(f"Loading checkpoint from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if "args" not in ckpt:
        raise ValueError("Checkpoint does not contain 'args' - cannot reconstruct model config")

    config: Config = ckpt["args"]
    config.device = "cuda"
    config.eval = True

    num_classes = get_num_classes(config.dataset_name)
    logger.info("Building model ...")
    model, _criterion, _postprocessors = build_model(config, num_classes)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.warning(f"Missing keys when loading state dict: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading state dict: {unexpected}")
    del ckpt

    model.to(torch.device("cuda"))
    model.eval()

    postprocess = PostProcess(topk=topk, reparam=config.reparam)

    return model, postprocess, config


def predict_image(
    model: torch.nn.Module,
    postprocess: PostProcess,
    config: Config,
    image: PIL.Image.Image,
    threshold: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Run inference on a single PIL image.

    Args:
        model: Plain-DETR model (on CUDA, eval mode).
        postprocess: ``PostProcess`` instance matching the model.
        config: Model config (needed for ``reparam`` flag).
        image: RGB PIL image.
        threshold: Minimum confidence score to keep a detection.

    Returns:
        A dict with keys ``scores``, ``labels``, ``boxes`` (all CPU tensors,
        filtered by *threshold*).  Boxes are in ``(x1, y1, x2, y2)`` absolute
        pixel coordinates of the original image.
    """
    device = torch.device("cuda")
    tensor, original_size = preprocess(image)
    samples = nested_tensor_from_tensor_list([tensor]).to(device)

    with torch.no_grad():
        outputs = model(samples)

    orig_w, orig_h = original_size
    if config.reparam:
        resized_h, resized_w = tensor.shape[1], tensor.shape[2]
        target_sizes = torch.tensor([[resized_h, resized_w]], device=device)
        original_target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        results = postprocess(outputs, target_sizes, original_target_sizes)
    else:
        target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        results = postprocess(outputs, target_sizes)

    result = results[0]
    scores = result["scores"].cpu()
    labels = result["labels"].cpu()
    boxes = result["boxes"].cpu()

    keep = scores >= threshold
    return {
        "scores": scores[keep],
        "labels": labels[keep],
        "boxes": boxes[keep],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    *image_paths: Path,
    checkpoint: Path,
    threshold: float = 0.3,
    topk: int = 100,
) -> None:
    """Run Plain-DETR inference on one or more images.

    Output images are saved alongside the originals with a ``.pred`` suffix
    inserted before the extension (e.g. ``photo.jpg`` -> ``photo.pred.jpg``).

    Args:
        checkpoint: Path to a ``checkpoint.epoch_N.pth`` file.
        images: One or more input image paths.
        threshold: Minimum confidence score to draw a detection.
        topk: Number of top-scoring predictions to consider.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    if not image_paths:
        raise ValueError("No images provided")

    model, postprocess, config = load_model(checkpoint, topk=topk)
    category_names = get_category_names(config.dataset_name)

    for image_path in image_paths:
        if not image_path.exists():
            logger.error(f"Image not found, skipping: {image_path}")
            continue

        logger.info(f"Processing {image_path}")
        with open(image_path, "rb") as f:
            pil_image = PIL.Image.open(f).convert("RGB")
        result = predict_image(model, postprocess, config, pil_image, threshold=threshold)

        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]

        logger.info(f"  {len(scores)} detections above threshold {threshold}")
        for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
            name = category_names[label] if label < len(category_names) else f"class_{label}"
            logger.info(f"    {name}: {score:.3f}  box={[f'{c:.1f}' for c in box]}")

        annotated = draw_predictions(pil_image, scores, labels, boxes, category_names)
        out_path = prediction_output_path(image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(out_path)
        logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    cyclopts.run(main)
