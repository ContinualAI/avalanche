################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-02-2022                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example shows how to run object detection/segmentation tasks.
This example will use a toy benchmark based on the LVIS dataset in which the
stream of experiences is obtained by splitting the dataset in equal parts.
"""

import logging
from pathlib import Path
from typing import Union

from avalanche.benchmarks import StreamUserDef
from avalanche.benchmarks.datasets import LvisDataset
from avalanche.benchmarks.scenarios.detection_scenario import \
    DetectionCLScenario
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset
from avalanche.training.supervised.naive_object_detection import \
    ObjectDetectionTemplate

from avalanche.evaluation.metrics import make_lvis_metrics, timing_metrics, \
    loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin
import argparse
import torch
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
from examples.detection_examples_utils import split_detection_benchmark

logging.basicConfig(level=logging.NOTSET)


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    train_transform = ToTensor()
    test_transform = ToTensor()
    # ---------

    # --- SCENARIO CREATION
    torch.random.manual_seed(1234)
    n_exps = 100  # Keep it high to run a short exp
    benchmark = split_lvis(
        n_experiences=n_exps,
        train_transform=train_transform,
        eval_transform=test_transform)
    # ---------

    # MODEL CREATION
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # Just tune the box predictor
    for p in model.parameters():
        p.requires_grad = False

    # Replace the classifier with a new one, that has "num_classes" outputs
    num_classes = benchmark.n_classes + 1  # N classes + background
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    # Define the optimizer and the scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    train_mb_size = 5
    warmup_factor = 1.0 / 1000
    warmup_iters = min(
        1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ObjectDetectionTemplate(
        model=model,
        optimizer=optimizer,
        train_mb_size=train_mb_size,
        train_epochs=1,
        eval_mb_size=train_mb_size,
        device=device,
        plugins=[
            LRSchedulerPlugin(lr_scheduler, step_granularity='iteration',
                              first_exp_only=True, first_epoch_only=True)
        ],
        evaluator=EvaluationPlugin(
            timing_metrics(epoch=True),
            loss_metrics(epoch_running=True),
            make_lvis_metrics(),
            loggers=[InteractiveLogger()])
    )

    # TRAINING LOOP
    print("Starting experiment...")
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print('Train dataset contains', len(experience.dataset), 'instances')

        cl_strategy.train(experience, num_workers=8)
        print("Training completed")

        cl_strategy.eval(benchmark.test_stream, num_workers=8)
        print('Evaluation completed')


def split_lvis(
        n_experiences: int, train_transform=None, eval_transform=None,
        shuffle=True, root_path: Union[str, Path] = None):
    """
    Creates the example Split LVIS benchmark.

    This is a toy benchmark created only to show how a detection benchmark can
    be created. It was not meant to be used for research purposes!

    :param n_experiences: The number of train experiences to create.
    :param train_transform: The train transformation.
    :param eval_transform: The eval transformation.
    :param shuffle: If True, the dataset will be split randomly
    :param root_path: The root path of the dataset. Defaults to None,
        which means that the default path will be used.
    :return: A :class:`DetectionScenario` instance.
    """

    train_dataset = LvisDataset(root=root_path, train=True)
    val_dataset = LvisDataset(root=root_path, train=False)

    all_cat_ids = set(train_dataset.lvis_api.get_cat_ids())
    all_cat_ids.union(val_dataset.lvis_api.get_cat_ids())

    return split_detection_benchmark(
        n_experiences=n_experiences,
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        n_classes=len(all_cat_ids),
        train_transform=train_transform,
        eval_transform=eval_transform,
        shuffle=shuffle
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
