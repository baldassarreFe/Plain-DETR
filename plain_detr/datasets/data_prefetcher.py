# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch


def to_cuda(samples, targets, device):
    """Transfer samples and targets to the given device with non-blocking copies.

    Requires the source tensors to reside in pinned (page-locked) memory
    (``pin_memory=True`` on the DataLoader) for the non-blocking transfer to
    actually be asynchronous.
    """
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets


class data_prefetcher:
    """Overlap host-to-device data transfers with GPU computation.

    Wraps a DataLoader iterator and uses a secondary CUDA stream to
    asynchronously copy the *next* batch to the GPU while the *current* batch
    is being processed on the main stream.  This hides the H2D transfer
    latency behind compute.

    The timeline looks like::

        Main stream:  [...backward/optim...]  [sync]  [forward ...]
        Side stream:  [H2D copy next batch]-----'

    Three mechanisms are needed for correctness:

    1. **Side stream** -- ``preload()`` issues the ``non_blocking`` H2D copy on
       a dedicated CUDA stream so it can run concurrently with main-stream
       kernels.
    2. **wait_stream** -- ``next()`` calls
       ``current_stream.wait_stream(side_stream)`` before returning data to the
       caller, ensuring the transfer has finished before any main-stream kernel
       reads the tensors.
    3. **record_stream** -- ``next()`` calls ``tensor.record_stream(current)``
       on every returned tensor.  Because the tensors were *allocated* on the
       side stream, the CUDA caching allocator would otherwise be free to
       recycle their memory as soon as the side stream finishes -- even if the
       main stream is still using them.  ``record_stream`` tells the allocator
       to wait until the main stream is done too.

    When ``prefetch=False`` the class falls back to synchronous transfer on the
    main stream (equivalent to calling ``to_cuda`` inline in the training
    loop).

    Note: the DataLoader **must** use ``pin_memory=True`` for the
    ``non_blocking`` transfers to actually be asynchronous.

    Adapted from the NVIDIA APEX data prefetcher example.
    """

    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        """Fetch the next batch from the DataLoader and start an async H2D copy on the side stream."""
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)

    def next(self):
        """Return the prefetched batch (blocking until the transfer is done) and kick off the next prefetch."""
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
