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
Main script for the CLVision challenge - object detection tracks.
https://sites.google.com/view/clvision2022/challenge?authuser=0
"""
import logging
import sys

from avalanche.training.supervised.naive_object_detection import \
    NaiveObjectDetection
from examples.detection import split_lvis

sys.path.append('.')

from avalanche.evaluation.metrics import LvisMetrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin
import argparse
import torch
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
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
    n_exps = 200  # Keep it high to run a short exp
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

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = benchmark.n_classes + 1  # N classes + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
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

    cl_strategy = NaiveObjectDetection(
        model=model,
        optimizer=optimizer,
        train_mb_size=train_mb_size,
        train_epochs=1,
        eval_mb_size=train_mb_size,
        device=device,
        plugins=[
            LRSchedulerPlugin(lr_scheduler)
        ],
        evaluator=EvaluationPlugin(
            timing_metrics(epoch=True),
            LvisMetrics(save_folder='./model_outputs'),
            loggers=[InteractiveLogger()])
    )

    print('Learnable parameters:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    # TRAINING LOOP
    print("Starting experiment...")
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print('Dataset contains', len(experience.dataset), 'instances')

        cl_strategy.train(experience, num_workers=4)
        print("Training completed")

        print('Test set size', len(benchmark.test_stream[0].dataset))
        cl_strategy.eval(benchmark.test_stream[0], num_workers=4)


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
