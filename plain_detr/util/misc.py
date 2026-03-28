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
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from __future__ import annotations

import copy
import datetime
import logging
import os
import re
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from plain_detr.main import Config

logger = logging.getLogger(__name__)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt: str = fmt

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data: Any) -> list[Any]:
    """Gather arbitrary picklable *data* from every rank and return a list (one element per rank)."""
    world_size = get_world_size()
    if world_size <= 1:
        return [data]
    output: list[Any] = [None] * world_size
    dist.all_gather_object(output, data)
    return output


@torch.no_grad()
def reduce_dict(input_dict: dict[str, Tensor], average: bool = True) -> dict[str, Tensor]:
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size <= 1:
        return input_dict
    # sort the keys so that they are consistent across processes
    names = sorted(input_dict.keys())
    values = torch.stack([input_dict[k] for k in names], dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.AVG if average else dist.ReduceOp.SUM)
    return {k: v for k, v in zip(names, values)}


class MetricLogger(object):
    def __init__(self, delimiter: str = "\t") -> None:
        self.meters: defaultdict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter: str = delimiter

    def update(self, **kwargs: float | int | Tensor) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> SmoothedValue:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable: Sequence[Any], print_freq: int, header: str | None = None) -> Iterator[Any]:
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def get_sha() -> str:
    cwd = Path(__file__).resolve().parent

    def _run(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch: list[Any]) -> tuple[Any, ...]:
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list: list[list[int]]) -> list[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: Sequence[Tensor]) -> NestedTensor:
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors: Tensor, mask: Tensor | None) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device, non_blocking: bool = False) -> NestedTensor:
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        cast_mask = None if self.mask is None else self.mask.to(device, non_blocking=non_blocking)
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.tensors.record_stream(stream)
        if self.mask is not None:
            self.mask.record_stream(stream)

    def decompose(self) -> tuple[Tensor, Tensor | None]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_local_size() -> int:
    return int(os.environ["LOCAL_SIZE"]) if is_dist_avail_and_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"]) if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode(args: Config) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.dist_url = "env://"
        os.environ["LOCAL_SIZE"] = str(torch.cuda.device_count())
    elif "SLURM_PROCID" in os.environ:
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput("scontrol show hostname {} | head -n1".format(node_list))
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
        os.environ["LOCAL_SIZE"] = str(num_gpus)
        args.dist_url = "env://"
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        logger.info("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    logger.info(f"| distributed init (rank {args.rank}): {args.dist_url}")
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()


@torch.no_grad()
def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)) -> list[Tensor]:
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input: Tensor,
    size: list[int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def find_latest_checkpoint(path: str | Path, ext: str = "pth") -> Path | None:
    path = Path(path)
    if not path.exists():
        return None
    default = path / f"checkpoint.{ext}"
    if default.exists():
        return default

    checkpoints = [ckpt for ckpt in path.glob(f"*.{ext}") if ckpt.name != "eval.pth"]
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        match = re.search(r"epoch_(\d+)", checkpoint.name)
        if match is None:
            continue
        count = int(match.group(1))
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def match_name_keywords(n: str, name_keywords: list[str]) -> bool:
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_swin_layer_id(var_name: str, backbone_type: str) -> int:
    maps: dict[str, dict[str, Any]] = {
        "tiny": dict(num_max_layer=12, layers_per_stage=[2, 2, 6, 2]),
        "small": dict(num_max_layer=24, layers_per_stage=[2, 2, 18, 2]),
        "base": dict(num_max_layer=24, layers_per_stage=[2, 2, 18, 2]),
        "large": dict(num_max_layer=24, layers_per_stage=[2, 2, 18, 2]),
    }
    map_type = None
    for k in maps.keys():
        if k in backbone_type:
            map_type = k
            break
    assert map_type is not None, f"Unsupported backbone type {backbone_type}"

    # hack for UpSampleWrapper
    if var_name.startswith("backbone.0.net."):
        num_max_layer = maps[map_type]["num_max_layer"]
        layers_per_stage = maps[map_type]["layers_per_stage"]
        if var_name.startswith("backbone.0.net.body.patch_embed"):
            layer_id = 0
        elif var_name.startswith("backbone.0.net.body.layers"):
            if var_name.split(".")[6] == "blocks":
                stage_id = int(var_name.split(".")[5])
                layer_id = int(var_name.split(".")[7]) + sum(layers_per_stage[:stage_id])
                layer_id = layer_id + 1
            elif var_name.split(".")[6] == "downsample":
                stage_id = int(var_name.split(".")[5])
                layer_id = sum(layers_per_stage[: stage_id + 1])
                layer_id = layer_id
        elif var_name.startswith("backbone.0.net.body.norm"):
            layer_id = num_max_layer + 1
        else:
            layer_id = num_max_layer + 1
        return num_max_layer + 1 - layer_id

    num_max_layer = maps[map_type]["num_max_layer"]
    layers_per_stage = maps[map_type]["layers_per_stage"]
    if var_name.startswith("backbone.0.body.patch_embed"):
        layer_id = 0
    elif var_name.startswith("backbone.0.body.layers"):
        if var_name.split(".")[5] == "blocks":
            stage_id = int(var_name.split(".")[4])
            layer_id = int(var_name.split(".")[6]) + sum(layers_per_stage[:stage_id])
            layer_id = layer_id + 1
        elif var_name.split(".")[5] == "downsample":
            stage_id = int(var_name.split(".")[4])
            layer_id = sum(layers_per_stage[: stage_id + 1])
            layer_id = layer_id
    elif var_name.startswith("backbone.0.body.norm"):
        layer_id = num_max_layer + 1
    else:
        layer_id = num_max_layer + 1
    return num_max_layer + 1 - layer_id


def get_dinov3_layer_id(var_name: str, backbone_type: str) -> int:
    """Return a *reverse* layer ID for DINOv3 ViT params (deeper → smaller ID → higher LR).

    DINOv3 ViT is a flat stack of transformer blocks — no hierarchical stages.  The
    naming inside Plain-DETR is ``backbone.0.body.model.<param>``.

    Layer ID assignment (forward order, 0-indexed):
      - ``patch_embed.*``, ``cls_token``, ``reg_token`` → 0
      - ``blocks.{i}.*``                                → i + 1
      - ``norm.*`` (final norm)                         → depth + 1
      - anything else                                   → depth + 1  (full LR)

    Returned value is ``depth + 1 - forward_id`` so that the deepest layers get the
    smallest reverse ID (= highest learning rate under exponential decay).
    """
    from plain_detr.models.backbone import DINOV3_VARIANTS

    _, _, depth = DINOV3_VARIANTS[backbone_type]
    num_max_layer = depth  # forward IDs go from 0 to depth+1

    # Strip the outer prefix to get the timm-internal parameter name.
    prefix = "backbone.0.body.model."
    if var_name.startswith(prefix):
        inner = var_name[len(prefix) :]
    else:
        # Fallback: parameter is outside the backbone (e.g. position encoding) → full LR.
        return 0

    if inner.startswith("patch_embed") or inner in ("cls_token", "reg_token"):
        layer_id = 0
    elif inner.startswith("blocks."):
        # inner looks like "blocks.3.norm1.weight" → block index is the second part
        block_idx = int(inner.split(".")[1])
        layer_id = block_idx + 1
    elif inner.startswith("norm"):
        layer_id = num_max_layer + 1
    else:
        layer_id = num_max_layer + 1

    return num_max_layer + 1 - layer_id


def get_param_groups(model: nn.Module, args: Config) -> list[dict[str, Any]]:
    # sanity check: a variable could not match backbone_names and linear_proj_names at the same time
    for n, p in model.named_parameters():
        if match_name_keywords(n, args.lr_backbone_names) and match_name_keywords(n, args.lr_linear_proj_names):
            raise ValueError

    if args.use_layerwise_decay:
        return _get_param_groups_layerwise_decay(model, args)
    else:
        return _get_param_groups_simple(model, args)


def _get_param_groups_layerwise_decay(model: nn.Module, args: Config) -> list[dict[str, Any]]:
    parameter_groups: dict[str, dict[str, Any]] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names):
            if match_name_keywords(n, args.wd_norm_names):
                group_name = "wd_mult"
                weight_decay = args.weight_decay * args.wd_norm_mult
            else:
                group_name = "wd"
                weight_decay = args.weight_decay
            if "swin" in args.backbone:
                layer_id = get_swin_layer_id(n, args.backbone)
            elif "dinov3" in args.backbone:
                layer_id = get_dinov3_layer_id(n, args.backbone)
            else:
                raise NotImplementedError(f"Layerwise decay not implemented for backbone {args.backbone!r}")
            group_name = f"layer_{layer_id}_{group_name}"

            if group_name not in parameter_groups:
                scale = args.lr_decay_rate**layer_id

                parameter_groups[group_name] = {
                    "params": [],
                    "names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * args.lr,
                    "weight_decay": weight_decay,
                }
            parameter_groups[group_name]["params"].append(p)
            parameter_groups[group_name]["names"].append(n)
        elif not match_name_keywords(n, args.lr_backbone_names) and match_name_keywords(n, args.lr_linear_proj_names):
            if match_name_keywords(n, args.wd_norm_names):
                group_name = "wd_mult"
                weight_decay = args.weight_decay * args.wd_norm_mult
            else:
                group_name = "wd"
                weight_decay = args.weight_decay
            group_name = f"head_linear_proj_{group_name}"

            if group_name not in parameter_groups:
                scale = args.lr_linear_proj_mult

                parameter_groups[group_name] = {
                    "params": [],
                    "names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * args.lr,
                    "weight_decay": weight_decay,
                }
            parameter_groups[group_name]["params"].append(p)
            parameter_groups[group_name]["names"].append(n)
        elif not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(
            n, args.lr_linear_proj_names
        ):
            if match_name_keywords(n, args.wd_norm_names):
                group_name = "wd_mult"
                weight_decay = args.weight_decay * args.wd_norm_mult
            else:
                group_name = "wd"
                weight_decay = args.weight_decay
            group_name = f"head_{group_name}"

            if group_name not in parameter_groups:
                scale = 1.0

                parameter_groups[group_name] = {
                    "params": [],
                    "names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * args.lr,
                    "weight_decay": weight_decay,
                }
            parameter_groups[group_name]["params"].append(p)
            parameter_groups[group_name]["names"].append(n)
        else:
            raise ValueError

    return list(parameter_groups.values())


def _get_param_groups_simple(model: nn.Module, args: Config) -> list[dict[str, Any]]:
    # Build (condition, lr, wd) specs for each param group
    groups_spec: list[tuple[Callable[[str], bool], float, float]] = [
        # Backbone params, default wd
        (
            lambda n: (
                match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
            ),
            args.lr_backbone,
            args.weight_decay,
        ),
        # Backbone norm params, wd multiplied
        (
            lambda n: (
                match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
            ),
            args.lr_backbone,
            args.weight_decay * args.wd_norm_mult,
        ),
        # Linear projection params, default wd
        (
            lambda n: (
                not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
            ),
            args.lr * args.lr_linear_proj_mult,
            args.weight_decay,
        ),
        # Linear projection norm params, wd multiplied
        (
            lambda n: (
                not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
            ),
            args.lr * args.lr_linear_proj_mult,
            args.weight_decay * args.wd_norm_mult,
        ),
        # Head params, default wd
        (
            lambda n: (
                not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
            ),
            args.lr,
            args.weight_decay,
        ),
        # Head norm params, wd multiplied
        (
            lambda n: (
                not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
            ),
            args.lr,
            args.weight_decay * args.wd_norm_mult,
        ),
    ]

    param_dicts: list[dict[str, Any]] = []
    for condition, lr, wd in groups_spec:
        names: list[str] = []
        params: list[nn.Parameter] = []
        for n, p in model.named_parameters():
            if p.requires_grad and condition(n):
                names.append(n)
                params.append(p)
        param_dicts.append({"params": params, "names": names, "lr": lr, "weight_decay": wd})

    return param_dicts


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation: str) -> Callable[..., Tensor]:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
