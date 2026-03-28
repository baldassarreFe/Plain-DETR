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

from __future__ import annotations

import copy
import datetime
import gc
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Literal

import cyclopts
import numpy as np
import torch
import wandb
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator
from torch import distributed as dist
from torch.utils.data import DataLoader

import plain_detr.datasets as datasets
import plain_detr.datasets.samplers as samplers
import plain_detr.util.misc as utils
from plain_detr.datasets import build_dataset, get_coco_api_from_dataset
from plain_detr.engine import evaluate, train_one_epoch
from plain_detr.models.detr import build as build_model

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Plain-DETR / Deformable DETR configuration."""

    # -- Training hyperparameters ------------------------------------------------
    lr: PositiveFloat = 2e-4
    """Learning rate."""
    lr_backbone_names: list[str] = ["backbone.0"]
    """Backbone parameter group names."""
    lr_backbone: PositiveFloat = 2e-5
    """Learning rate for backbone parameters."""
    lr_linear_proj_names: list[str] = ["reference_points", "sampling_offsets"]
    """Linear projection parameter group names."""
    lr_linear_proj_mult: NonNegativeFloat = 0.1
    """Learning rate multiplier for linear projection parameters."""
    batch_size: PositiveInt = 1
    """Batch size per GPU."""
    weight_decay: NonNegativeFloat = 1e-4
    """Weight decay."""
    epochs: PositiveInt = 50
    """Total training epochs."""
    lr_drop: PositiveInt = 40
    """Epoch to drop learning rate."""
    lr_drop_epochs: list[int] | None = None
    """Specific epochs to drop learning rate."""
    clip_max_norm: NonNegativeFloat = 0.1
    """Gradient clipping max norm (0 to disable)."""
    warmup: NonNegativeInt = 0
    """Warmup iterations."""
    sgd: bool = False
    """Use SGD optimizer instead of AdamW."""

    # -- Modern DETR tricks ------------------------------------------------------
    with_box_refine: bool = False
    """Enable iterative box refinement."""
    two_stage: bool = False
    """Enable two-stage detection."""
    mixed_selection: bool = False
    """Enable mixed selection (DINO trick)."""
    look_forward_twice: bool = False
    """Enable look-forward-twice (DINO trick)."""
    k_one2many: NonNegativeInt = 5
    """One-to-many matching multiplier (0 to disable)."""
    lambda_one2many: NonNegativeFloat = 1.0
    """One-to-many loss weight."""
    num_queries_one2one: PositiveInt = 300
    """Number of query slots for one-to-one matching."""
    num_queries_one2many: NonNegativeInt = 0
    """Number of query slots for one-to-many matching."""
    reparam: bool = False
    """Use absolute coordinates & reparameterization for bounding boxes."""

    # -- Model parameters --------------------------------------------------------
    frozen_weights: Path | None = None
    """Path to pretrained model. If set, only the mask head will be trained."""

    # -- Backbone ----------------------------------------------------------------
    backbone: str = "resnet50"
    """Name of the convolutional backbone to use."""
    dilation: bool = False
    """Replace stride with dilation in the last convolutional block (DC5)."""
    position_embedding: Literal["sine", "learned", "sine_unnorm"] = "sine"
    """Type of positional embedding to use on top of the image features."""
    position_embedding_scale: PositiveFloat = float(2 * np.pi)
    """position / size * scale."""
    num_feature_levels: PositiveInt = 1
    """Number of feature levels."""
    pretrained_backbone_path: Path = Path("swin_tiny_patch4_window7_224.pkl")
    """Path to pretrained backbone weights."""
    drop_path_rate: NonNegativeFloat = 0.1
    """Drop path rate."""
    upsample_backbone_output: bool = False
    """Upsample backbone output feature to target stride."""
    upsample_stride: PositiveInt = 16
    """Target stride for upsampling backbone output feature."""

    # -- Transformer -------------------------------------------------------------
    dec_layers: PositiveInt = 6
    """Number of decoding layers in the transformer."""
    dim_feedforward: PositiveInt = 2048
    """Intermediate size of the feedforward layers in the transformer blocks."""
    hidden_dim: PositiveInt = 256
    """Size of the embeddings (dimension of the transformer)."""
    dropout: NonNegativeFloat = 0.1
    """Dropout applied in the transformer."""
    nheads: PositiveInt = 8
    """Number of attention heads inside the transformer's attentions."""
    norm_type: Literal["pre_norm", "post_norm"] = "pre_norm"
    """Normalization type."""
    proposal_feature_levels: PositiveInt = 1
    """Number of proposal feature levels."""
    proposal_in_stride: PositiveInt = 8
    """Proposal input stride."""
    proposal_tgt_strides: list[int] = [8, 16, 32, 64]
    """Proposal target strides."""
    decoder_type: str = "deform"
    """Decoder type."""
    decoder_use_checkpoint: bool = False
    """Use gradient checkpointing in decoder."""
    decoder_rpe_hidden_dim: PositiveInt = 512
    """Decoder RPE hidden dimension."""
    decoder_rpe_type: str = "linear"
    """Decoder RPE type."""
    wd_norm_names: list[str] = [
        # -- Shared (both Swin and DINOv3) --
        "norm",  # LayerNorm / RMSNorm in any backbone or decoder
        "bias",  # all bias terms
        # -- Swin-specific --
        "rpb_mlp",  # relative position bias MLP
        "cpb_mlp",  # continuous position bias MLP
        "logit_scale",  # per-head log scale in SwinV2 attention
        "relative_position_bias_table",  # learned relative position bias table
        # -- DINOv3-specific --
        "gamma_1",  # LayerScale parameter (post self-attention)
        "gamma_2",  # LayerScale parameter (post FFN)
        "cls_token",  # class token embedding
        "reg_token",  # register token embedding
        # -- Decoder / head --
        "level_embed",  # multi-scale level embedding
        "reference_points",  # reference point projection
        "sampling_offsets",  # deformable attention sampling offsets
        "rel_pos",  # decoder relative position parameters
    ]
    """Parameter names that receive reduced weight decay (wd_norm_mult)."""
    wd_norm_mult: NonNegativeFloat = 1.0
    """Weight decay multiplier for norm parameters."""
    use_layerwise_decay: bool = False
    """Use layer-wise learning rate decay."""
    lr_decay_rate: PositiveFloat = 1.0
    """Learning rate decay rate per layer."""

    # -- Segmentation ------------------------------------------------------------
    masks: bool = False
    """Train segmentation head."""

    # -- Loss --------------------------------------------------------------------
    aux_loss: bool = True
    """Enable auxiliary decoding losses (loss at each layer)."""

    # -- Matcher -----------------------------------------------------------------
    set_cost_class: NonNegativeFloat = 2
    """Class coefficient in the matching cost."""
    set_cost_bbox: NonNegativeFloat = 5
    """L1 box coefficient in the matching cost."""
    set_cost_giou: NonNegativeFloat = 2
    """GIoU box coefficient in the matching cost."""

    # -- Loss coefficients -------------------------------------------------------
    mask_loss_coef: NonNegativeFloat = 1
    """Mask loss coefficient."""
    dice_loss_coef: NonNegativeFloat = 1
    """Dice loss coefficient."""
    cls_loss_coef: NonNegativeFloat = 2
    """Classification loss coefficient."""
    bbox_loss_coef: NonNegativeFloat = 5
    """Bounding box loss coefficient."""
    giou_loss_coef: NonNegativeFloat = 2
    """GIoU loss coefficient."""
    focal_alpha: NonNegativeFloat = 0.25
    """Focal loss alpha."""

    # -- Dataset -----------------------------------------------------------------
    data_dir: Path = Path("data")
    """Root directory for datasets."""
    dataset_file: str = "coco"
    """Dataset name."""
    coco_path: Path = Path("coco")
    """Path to COCO dataset, absolute or relative to data_dir."""
    coco_panoptic_path: Path = Path("coco")
    """Path to COCO panoptic annotations, absolute or relative to data_dir."""
    remove_difficult: bool = False
    """Remove difficult examples."""

    # -- Runtime -----------------------------------------------------------------
    output_dir: Path = Path("output")
    """Path where to save checkpoints and logs."""
    device: str = "cuda"
    """Device to use for training / testing."""
    seed: NonNegativeInt = 42
    """Random seed."""
    resume: Path | None = None
    """Resume from checkpoint."""
    auto_resume: bool = True
    """Automatically resume from latest checkpoint."""
    start_epoch: NonNegativeInt = 0
    """Start epoch."""
    num_workers: NonNegativeInt = 8
    """Number of data loading workers."""
    cache_mode: bool = False
    """Whether to cache images on memory."""

    # -- Evaluation --------------------------------------------------------------
    eval: bool = False
    """Evaluation-only mode."""
    topk: PositiveInt = 100
    """Top-k predictions for evaluation."""

    # -- Training technologies ---------------------------------------------------
    amp_dtype: Literal["fp32", "fp16", "bf16"] = "fp32"
    """AMP dtype for mixed precision training: fp32 (disabled), fp16, or bf16."""
    use_checkpoint: bool = False
    """Use gradient checkpointing in backbone."""
    gc_collect_interval: PositiveInt = 500
    """Run gc.collect() every N training iterations."""

    # -- Logging -----------------------------------------------------------------
    use_wandb: bool = False
    """Enable Weights & Biases logging."""
    wandb_entity: str | None = None
    """W&B entity."""
    wandb_name: str | None = None
    """W&B run name."""

    # -- Distributed (set at runtime by init_distributed_mode) -------------------
    distributed: bool = False
    """Whether to use distributed training (set automatically)."""
    rank: NonNegativeInt = 0
    """Process rank (set automatically)."""
    world_size: PositiveInt = 1
    """Number of processes (set automatically)."""
    gpu: NonNegativeInt = 0
    """Local GPU ID (set automatically)."""
    dist_url: str = "env://"
    """URL for distributed training setup."""
    dist_backend: str = "nccl"
    """Distributed backend."""

    @model_validator(mode="after")
    def _check_frozen_weights_requires_masks(self) -> Config:
        if self.frozen_weights is not None and not self.masks:
            raise ValueError("frozen_weights requires masks=True (frozen training is meant for segmentation only)")
        return self


def main(args: Config):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logger.info(f"git:\n  {utils.get_sha()}\n")
    logger.info(f"Args:\n{args.model_dump_json(indent=2)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    if utils.is_main_process():
        with open(args.output_dir / "args.json", "w") as f:
            f.write(args.model_dump_json(indent=2))

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- data loading/transform libs rely on the global seed
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    logger.info(f"Model:\n{model}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")

    # Only fp16 needs gradient scaling
    amp_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    logger.info(f"AMP dtype: {amp_dtype}, GradScaler enabled: {scaler.is_enabled()}")

    # Datasets
    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)
    logger.info(f"Training dataset size: {len(dataset_train)}")
    logger.info(f"Validation dataset size: {len(dataset_val)}")

    # DataLoaders
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    logger.info(f"DataLoader train iters per epoch: {len(data_loader_train)}")
    logger.info(f"DataLoader val iters: {len(data_loader_val)}")

    # Optimizer — use the unwrapped model for param groups (DDP wrapping comes later)
    model_without_ddp = model
    param_dicts = utils.get_param_groups(model_without_ddp, args)
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    for i, pd in enumerate(param_dicts):
        logger.debug(f"Group-{i}: lr {pd['lr']} wd {pd['weight_decay']}, params {len(pd['params'])}")
        logger.debug(f"{json.dumps(pd['names'], indent=2)}")

    # LR scheduler
    drop_iter = args.lr_drop * len(data_loader_train)

    def lr_func(cur_iter):
        return (
            cur_iter / args.warmup if args.warmup and cur_iter < args.warmup else (0.1 if cur_iter > drop_iter else 1)
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu", weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    if args.use_wandb and dist.get_rank() == 0:
        wandb.init(
            entity=args.wandb_entity,
            project="Plain-DETR",
            id=args.wandb_name,  # set id as wandb_name for resume
            name=args.wandb_name,
        )

    output_dir = args.output_dir
    if args.auto_resume:
        resume_from = utils.find_latest_checkpoint(output_dir)
        if resume_from is not None:
            logger.info(f"Use autoresume, overwrite args.resume with {resume_from}")
            args.resume = resume_from
        else:
            logger.warning(f"Use autoresume, but can not find checkpoint in {output_dir}")
    if args.resume is not None and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if len(missing_keys) > 0:
            logger.warning(f"Missing Keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected Keys: {unexpected_keys}")
        if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]

            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # For LambdaLR, the lambda funcs are not stored in state_dict, see
            # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR.state_dict
            logger.warning(
                "lr scheduler has been resumed from checkpoint, but the lambda funcs are not"
                " stored in state_dict. The new lr schedule would override the resumed lr schedule."
            )
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint["epoch"] + 1

            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                args=args,
                model=model,
                criterion=criterion,
                postprocessors=postprocessors,
                data_loader=data_loader_val,
                base_ds=base_ds,
                step=args.start_epoch * len(data_loader_train),
            )

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            args=args,
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            data_loader=data_loader_val,
            base_ds=base_ds,
            step=args.start_epoch * len(data_loader_train),
        )
        if utils.is_main_process():
            torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        if utils.is_main_process():
            areaRngLbl = ["", "50", "75", "s", "m", "l"]
            msg = "copypaste: "
            for label in areaRngLbl:
                msg += f"AP{label} "
            for ap in coco_evaluator.coco_eval["bbox"].stats[: len(areaRngLbl)]:
                msg += "{:.3f} ".format(ap)
            logger.info(msg)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        gc.collect()
        gc.disable()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args=args,
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        gc.enable()

        checkpoint_data = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if utils.is_main_process():
            torch.save(checkpoint_data, output_dir / f"checkpoint.epoch_{epoch}.pth")
        del checkpoint_data

        test_stats, coco_evaluator = evaluate(
            args=args,
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            data_loader=data_loader_val,
            base_ds=base_ds,
            step=(epoch + 1) * len(data_loader_train),
        )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        del train_stats, test_stats, log_stats

        if utils.is_main_process():
            (output_dir / "eval").mkdir(exist_ok=True)
            if "bbox" in coco_evaluator.coco_eval:
                filenames = ["latest.pth"]
                if epoch % 50 == 0:
                    filenames.append(f"{epoch:03}.pth")
                for name in filenames:
                    torch.save(
                        coco_evaluator.coco_eval["bbox"].eval,
                        output_dir / "eval" / name,
                    )

            areaRngLbl = ["", "50", "75", "s", "m", "l"]
            msg = "copypaste: "
            for label in areaRngLbl:
                msg += f"AP{label} "
            for ap in coco_evaluator.coco_eval["bbox"].stats[: len(areaRngLbl)]:
                msg += "{:.3f} ".format(ap)
            logger.info(msg)
        del coco_evaluator

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    cyclopts.run(main)
