import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
# from lvis_api import LVIS
from lvis import LVIS
from pycocotools.coco import COCO
from torch.utils.data import Subset

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset, \
    AvalancheConcatDataset
from examples.tvdetection.coco_eval import CocoEvaluator
from examples.tvdetection.coco_utils import CocoDetection, convert_to_coco_api
from examples.tvdetection.lvis_eval import LvisEvaluator
from examples.tvdetection.utils import MetricLogger, SmoothedValue, reduce_dict


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # Avalanche: added "*_"
    for images, targets, *_ in metric_logger.log_every(
            data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def evaluate_coco(
        model, data_loader, device, coco, metric_logger, cpu_device, iou_types):
    coco_evaluator = CocoEvaluator(coco, iou_types)
    header = "Test:"

    for images, targets, *_ in metric_logger.log_every(
            data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output
               for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator


def evaluate_lvis(
        model, data_loader, device, lvis: LVIS, metric_logger, cpu_device,
        iou_types):

    # Lorenzo: implemented by taking inspiration from COCO code
    lvis_evaluator = LvisEvaluator(lvis, iou_types)
    header = "Test:"

    for images, targets, *_ in metric_logger.log_every(
            data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output
               for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        lvis_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    lvis_eval_per_iou = lvis_evaluator.evaluate()
    lvis_evaluator.summarize()

    return lvis_eval_per_iou


@torch.inference_mode()
def evaluate(model, data_loader, device):
    # Lorenzo: splitted evaluation in "evaluate_coco" and "evaluate_lvis"
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")

    det_api = get_detection_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    if isinstance(det_api, COCO):
        result = evaluate_coco(
            model, data_loader, device, det_api, metric_logger,
            cpu_device, iou_types)
    else:
        result = evaluate_lvis(
            model, data_loader, device, det_api, metric_logger,
            cpu_device, iou_types)

    torch.set_num_threads(n_threads)
    return result


def get_detection_api_from_dataset(dataset):
    # Lorenzo: adapted to support LVIS and AvalancheDataset
    for _ in range(10):
        if isinstance(dataset, CocoDetection):
            break
        elif hasattr(dataset, 'lvis_api'):
            break
        elif isinstance(dataset, Subset):
            dataset = dataset.dataset
        elif isinstance(dataset, AvalancheSubset):
            dataset = dataset._original_dataset
        elif isinstance(dataset, AvalancheConcatDataset):
            dataset = dataset._dataset_list[0]
        elif isinstance(dataset, AvalancheDataset):
            dataset = dataset._dataset

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if hasattr(dataset, 'lvis_api'):
        return dataset.lvis_api
    return convert_to_coco_api(dataset)
