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
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist
from lvis import LVISEval, LVISResults, LVIS

from avalanche.evaluation.metrics.detection import (
    DetectionEvaluator,
    TCommonDetectionOutput,
)


class LvisEvaluator(DetectionEvaluator[Dict[str, LVISEval], TCommonDetectionOutput]):
    """
    Defines an evaluator for the LVIS dataset.

    This evaluator is usually used through a metric returned by
    :func:`make_lvis_metrics`.

    This mostly acts a wrapper around :class:`LVISEval` from the `lvis` library.
    """

    def __init__(self, lvis_gt: LVIS, iou_types: List[str]):
        assert isinstance(iou_types, (list, tuple))
        self.lvis_gt = lvis_gt

        self.iou_types = iou_types
        self.img_ids: List[int] = []
        self.predictions: List[Dict[str, Any]] = []
        self.lvis_eval_per_iou: Dict[str, LVISEval] = dict()

    def update(self, predictions: TCommonDetectionOutput):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare_for_lvis_detection(predictions)
        self.predictions.extend(results)

    def synchronize_between_processes(self):
        if dist.is_initialized():
            # Bypass NCCL (which forces CUDA-only sync)
            if dist.get_backend() == "nccl":
                group = dist.new_group(backend="gloo")
            else:
                group = dist.group.WORLD

            my_rank = dist.get_rank()
            is_main_rank = my_rank == 0
            if is_main_rank:
                output: List[Dict[str, Any]] = [
                    None
                ] * dist.get_world_size()  # type: ignore
                dist.gather_object(
                    self.predictions,
                    output,
                    dst=0,
                    group=group,
                )
                return list(itertools.chain.from_iterable(output)), True
            else:
                dist.gather_object(
                    self.predictions,
                    None,
                    dst=0,
                    group=group,
                )
                return None, False
        else:
            return self.predictions, True

    def evaluate(
        self, max_dets_per_image=None
    ) -> Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, LVISEval]]]]:
        all_preds, main_process = self.synchronize_between_processes()
        if main_process:
            if max_dets_per_image is None:
                max_dets_per_image = 300

            eval_imgs = [lvis_res["image_id"] for lvis_res in all_preds]

            gt_subset = LvisEvaluator._make_lvis_subset(self.lvis_gt, eval_imgs)

            for iou_type in self.iou_types:
                print("Evaluating for iou", iou_type)
                if iou_type == "segm":
                    # See:
                    # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/lvis_evaluation.html
                    lvis_results = copy.deepcopy(all_preds)
                    for c in lvis_results:
                        c.pop("bbox", None)
                else:
                    lvis_results = all_preds

                lvis_results = LVISResults(
                    gt_subset, lvis_results, max_dets=max_dets_per_image
                )
                lvis_eval = LVISEval(gt_subset, lvis_results, iou_type)
                lvis_eval.params.img_ids = list(set(eval_imgs))
                lvis_eval.run()
                self.lvis_eval_per_iou[iou_type] = lvis_eval
        else:
            self.lvis_eval_per_iou = None  # type: ignore

        if dist.is_initialized():
            dist.barrier()

        result_dict: Dict[str, Dict[str, Union[int, float]]] = dict()
        if self.lvis_eval_per_iou is not None:
            for iou, eval_data in self.lvis_eval_per_iou.items():
                result_dict[iou] = dict()
                for key in eval_data.results:
                    value = eval_data.results[key]
                    result_dict[iou][key] = value

        if main_process:
            return result_dict, self.lvis_eval_per_iou
        else:
            return None

    def summarize(self):
        if self.lvis_eval_per_iou is not None:
            for iou_type, lvis_eval in self.lvis_eval_per_iou.items():
                print(f"IoU metric: {iou_type}")
                lvis_eval.print_results()

    def prepare_for_lvis_detection(self, predictions: TCommonDetectionOutput):
        lvis_results: List[Dict[str, Any]] = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            has_mask = "mask" in prediction
            has_bbox = "boxes" in prediction
            has_keypoint = "keypoints" in prediction

            if has_bbox:
                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()

            if has_mask:
                masks = prediction["masks"]
                masks = masks > 0.5
                rles = [
                    mask_util.encode(
                        np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                    )[0]
                    for mask in masks
                ]
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")

            if has_keypoint:
                keypoints = prediction["keypoints"]
                keypoints = keypoints.flatten(start_dim=1).tolist()

            for pred_idx in range(len(labels)):
                lvis_pred = {
                    "image_id": original_id,
                    "category_id": labels[pred_idx],
                    "score": scores[pred_idx],
                }

                if has_bbox:
                    lvis_pred["bbox"] = boxes[pred_idx]

                if has_mask:
                    lvis_pred["segmentation"] = rles[pred_idx]

                if has_keypoint:
                    lvis_pred["keypoints"] = keypoints[pred_idx]

                lvis_results.append(lvis_pred)

        return lvis_results

    @staticmethod
    def _make_lvis_subset(lvis_gt, img_ids):
        img_ids = set(img_ids)

        subset = dict()
        subset["categories"] = list(lvis_gt.dataset["categories"])

        subset_imgs = []
        for img in lvis_gt.dataset["images"]:
            if img["id"] in img_ids:
                subset_imgs.append(img)
        subset["images"] = subset_imgs

        subset_anns = []
        for ann in lvis_gt.dataset["annotations"]:
            if ann["image_id"] in img_ids:
                subset_anns.append(ann)
        subset["annotations"] = subset_anns

        return DictLVIS(subset)


class DictLVIS(LVIS):
    """
    Child class of LVIS that allows for the creation of LVIS objects from
    a dictionary.
    """

    def __init__(self, annotation_dict):
        """Class for reading and visualizing annotations.
        Args:
            annotation_dict (dict): annotations
        """
        self.logger = logging.getLogger(__name__)
        self.dataset = annotation_dict

        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        self._create_index()


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


__all__ = ["LvisEvaluator"]
