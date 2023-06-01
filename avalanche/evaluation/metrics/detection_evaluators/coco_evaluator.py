# The coco evaluator code has been adapted from:
# https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py
# which has been distributed under the following license:
################################################################################
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################

# For the Avalanche adaptation:
################################################################################
# Copyright (c) 2022 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-03-2022                                                             #
# Author: Lorenzo Pellegrini                                                   #
#                                                                              #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################


import copy
import io
from collections import OrderedDict
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from avalanche.evaluation.metrics.detection import (
    DetectionEvaluator,
    TCommonDetectionOutput,
)


COCO_STATS_DET_ORDER = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR@1",
    "AR@10",
    "AR@100",
    "ARs@100",
    "ARm@100",
    "ARl@100",
)

COCO_STATS_KPS_ORDER = (
    "AP",
    "AP50",
    "AP75",
    "APm",
    "APl",
    "AR",
    "AR50",
    "AR75",
    "ARm",
    "ARl",
)


class CocoEvaluator(DetectionEvaluator[Dict[str, COCOeval], TCommonDetectionOutput]):
    """
    Defines an evaluator for the COCO dataset.

    This evaluator is usually used through a metric returned by
    :func:`make_coco_metrics`.

    This mostly acts a wrapper around :class:`COCOEval` from the `pycocotools`
    library.
    """

    def __init__(self, coco_gt: COCO, iou_types: List[str]):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval: Dict[str, COCOeval] = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids: List[int] = []
        self.eval_imgs: Dict[str, List[np.ndarray]] = {k: [] for k in iou_types}

    def update(self, predictions: TCommonDetectionOutput):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)

            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

        if dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def evaluate(
        self,
    ) -> Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, COCOeval]]]]:
        main_process = self.synchronize_between_processes()

        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

        result_dict: Dict[str, Dict[str, Union[int, float]]] = OrderedDict()
        if main_process:
            for iou, eval_data in self.coco_eval.items():
                result_dict[iou] = OrderedDict()
                with redirect_stdout(io.StringIO()):
                    eval_data.summarize()
                metrics_stats = eval_data.stats
                if iou == "segm" or iou == "bbox":
                    for metric_name, metric_value in zip(
                        COCO_STATS_DET_ORDER, metrics_stats
                    ):
                        result_dict[iou][metric_name] = metric_value
                elif iou == "keypoints":
                    for metric_name, metric_value in zip(
                        COCO_STATS_KPS_ORDER, metrics_stats
                    ):
                        result_dict[iou][metric_name] = metric_value

        if dist.is_initialized():
            dist.barrier()

        if main_process:
            return result_dict, self.coco_eval
        else:
            return None

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def merge(
    img_ids: List[int], eval_imgs: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids_np = np.array(merged_img_ids)
    merged_eval_imgs_np = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids_np, idx = np.unique(merged_img_ids_np, return_index=True)
    merged_eval_imgs_np = merged_eval_imgs_np[..., idx]

    return merged_img_ids_np, merged_eval_imgs_np


def create_common_coco_eval(
    coco_eval: COCOeval, img_ids: List[int], eval_imgs: List[np.ndarray]
):
    img_ids_np, eval_imgs_np = merge(img_ids, eval_imgs)
    img_ids_lst = list(img_ids_np)
    eval_imgs_list = list(eval_imgs_np.flatten())

    coco_eval.evalImgs = eval_imgs_list
    coco_eval.params.imgIds = img_ids_lst
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return (
        imgs.params.imgIds,
        np.asarray(imgs.evalImgs).reshape(
            -1, len(imgs.params.areaRng), len(imgs.params.imgIds)
        ),
    )


__all__ = ["CocoEvaluator"]
