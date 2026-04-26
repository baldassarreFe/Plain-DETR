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

"""
Train and eval functions used in main.py
"""

from __future__ import annotations

import gc
import logging
import math
import sys
from typing import TYPE_CHECKING

import torch
import wandb
from torch import distributed as dist

import plain_detr.util.misc as utils
from plain_detr.datasets.coco_eval import CocoEvaluator
from plain_detr.datasets.data_prefetcher import data_prefetcher
from plain_detr.datasets.panoptic_eval import PanopticEvaluator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from plain_detr.main import Config
    from plain_detr.models.detr import SetCriterion

logger = logging.getLogger(__name__)


def train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many):
    # Compute num_boxes once with a single all_reduce, then pass to both criterion calls
    num_boxes = sum(len(t["labels"]) for t in targets)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    if utils.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(num_boxes)
    num_boxes = torch.clamp(num_boxes / utils.get_world_size(), min=1).item()

    # one-to-one-loss
    loss_dict = criterion(outputs, targets, num_boxes=num_boxes)
    # Create one2many targets by repeating boxes and labels, sharing all other fields
    multi_targets = [
        {**t, "boxes": t["boxes"].repeat(k_one2many, 1), "labels": t["labels"].repeat(k_one2many)} for t in targets
    ]

    outputs_one2many = dict()
    outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
    outputs_one2many["pred_boxes"] = outputs["pred_boxes_one2many"]
    outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]
    if "pred_boxes_old_one2many" in outputs.keys():
        outputs_one2many["pred_boxes_old"] = outputs["pred_boxes_old_one2many"]
        outputs_one2many["pred_deltas"] = outputs["pred_deltas_one2many"]

    # one-to-many loss: scale num_boxes by k_one2many since targets were repeated
    loss_dict_one2many = criterion(outputs_one2many, multi_targets, num_boxes=num_boxes * k_one2many)
    for key, value in loss_dict_one2many.items():
        if key + "_one2many" in loss_dict.keys():
            loss_dict[key + "_one2many"] += value * lambda_one2many
        else:
            loss_dict[key + "_one2many"] = value * lambda_one2many
    return loss_dict


def train_one_epoch(
    *,
    args: Config,
    model: torch.nn.Module,
    criterion: SetCriterion,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    amp_dtype: torch.dtype,
    scaler: torch.amp.grad_scaler.GradScaler,
):
    device = torch.device(args.device)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for idx in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(samples)

            if args.k_one2many > 0:
                loss_dict = train_hybrid(outputs, targets, args.k_one2many, criterion, args.lambda_one2many)
            else:
                loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            logger.error(f"{loss_dict_reduced}")
            sys.exit(1)

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        if args.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        else:
            grad_total_norm = torch.nn.utils.clip_grad._get_total_norm(
                [p.grad for p in model.parameters() if p.grad is not None]
            )
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(
            loss=loss_value,
            **loss_dict_reduced_scaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
        lr_scheduler.step()

        if args.use_wandb and idx % print_freq == 0 and dist.get_rank() == 0:
            log_data = dict(
                loss=loss_value,
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=grad_total_norm,
                **loss_dict_reduced_scaled,
            )
            log_data = {"train/" + k: v for k, v in log_data.items()}
            wandb.log(data=log_data, step=(epoch * len(data_loader) + idx))

        del outputs, loss_dict, losses, loss_dict_reduced, loss_dict_reduced_scaled
        del losses_reduced_scaled, grad_total_norm
        if (idx + 1) % args.gc_collect_interval == 0:
            gc.collect()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    *,
    args: Config,
    model,
    criterion: SetCriterion,
    postprocessors,
    data_loader: DataLoader,
    base_ds,
    step: int,
) -> tuple[dict, CocoEvaluator]:
    device = torch.device(args.device)
    output_dir = args.output_dir

    # (hack) disable the one-to-many branch queries
    # save them frist
    save_num_queries = model.module.num_queries
    save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
    model.module.num_queries = model.module.num_queries_one2one
    model.module.transformer.two_stage_num_proposals = model.module.num_queries

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, max_dets=args.max_dets)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=(output_dir / "panoptic_eval").as_posix(),
        )

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _idx in metric_logger.log_every(range(len(data_loader)), 10, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        if args.reparam:
            results = postprocessors["bbox"](outputs, target_sizes, orig_target_sizes)
        else:
            results = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if "bbox" in postprocessors.keys():
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    if "segm" in postprocessors.keys():
        stats["coco_eval_seg_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]
    if args.use_wandb and utils.get_rank() == 0:
        log_data = {
            "bbox/AP": stats["coco_eval_bbox"][0],
            "bbox/AP50": stats["coco_eval_bbox"][1],
            "bbox/AP75": stats["coco_eval_bbox"][2],
            "bbox/APs": stats["coco_eval_bbox"][3],
            "bbox/APm": stats["coco_eval_bbox"][4],
            "bbox/APl": stats["coco_eval_bbox"][5],
        }
        for k, v in stats.items():
            if k not in ["coco_eval_bbox", "coco_eval_seg_masks"]:
                log_data["val/" + k] = v
        wandb.log(data=log_data, step=step)

    # recover the model parameters for next training epoch
    model.module.num_queries = save_num_queries
    model.module.transformer.two_stage_num_proposals = save_two_stage_num_proposals
    return stats, coco_evaluator
