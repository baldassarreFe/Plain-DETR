# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import json
from pathlib import Path

import plain_detr.util.misc as utils

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass


class PanopticEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        self.output_dir = Path(output_dir)
        if utils.is_main_process():
            self.output_dir.mkdir(exist_ok=True)
        self.predictions = []

    def update(self, predictions):
        for p in predictions:
            with open(self.output_dir / p["file_name"], "wb") as f:
                f.write(p.pop("png_string"))

        self.predictions += predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        if utils.is_main_process():
            json_data = {"annotations": self.predictions}
            predictions_json = self.output_dir / "predictions.json"
            with open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
            return pq_compute(
                self.gt_json,
                predictions_json,
                gt_folder=self.gt_folder,
                pred_folder=self.output_dir,
            )
        return None
