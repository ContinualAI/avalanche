################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-17-2022                                                             #
# Author: Jia Shi, Zhiqiu Lin                                                  #
# E-mail: jiashi@andrew.cmu.edu, zl279@cornell.edu                             #
# Website: https://clear-benchmark.github.io                                   #
################################################################################

"""
Example: Training and evaluating on CLEAR benchmark (RGB images)
"""

import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.supervised import Naive
from avalanche.benchmarks.classic.clear import CLEAR, CLEARMetric


# For CLEAR dataset setup
DATASET_NAME = "clear100_cvpr2022"
NUM_CLASSES = {"clear10": 11, "clear100_cvpr2022": 100}
assert DATASET_NAME in NUM_CLASSES.keys()

# please refer to paper for discussion on streaming v.s. iid protocol
EVALUATION_PROTOCOL = "streaming"  # trainset = testset per timestamp
# EVALUATION_PROTOCOL = "iid"  # 7:3 trainset_size:testset_size

# For saving the datasets/models/results/log files
ROOT = Path("..")
DATA_ROOT = ROOT / DATASET_NAME
MODEL_ROOT = ROOT / "models"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# Define hyperparameters/scheduler/augmentation
HPARAM = {
    "batch_size": 256,
    "num_epoch": 100,
    "step_scheduler_decay": 30,
    "scheduler_step": 0.1,
    "start_lr": 0.01,
    "weight_decay": 1e-5,
    "momentum": 0.9,
}


def make_scheduler(optimizer, step_size, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    return scheduler


def main():
    model = torchvision.models.resnet18(pretrained=False)

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )

    # log to Tensorboard
    tb_logger = TensorboardLogger(ROOT)

    # log to text file
    text_logger = TextLogger(open(ROOT / "log.txt", "w+"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, 
                         stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, 
                     stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(
            num_classes=NUM_CLASSES[DATASET_NAME], save_image=False, 
            stream=True
        ),
        disk_usage_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loggers=[interactive_logger, text_logger, tb_logger],
    )

    if EVALUATION_PROTOCOL == "streaming":
        seed = None
    else:
        seed = 0

    benchmark = CLEAR(
        data_name=DATASET_NAME,
        evaluation_protocol=EVALUATION_PROTOCOL,
        feature_type=None,
        seed=seed,
        train_transform=train_transform,
        eval_transform=test_transform,
        dataset_root=DATA_ROOT,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=HPARAM["start_lr"],
        weight_decay=HPARAM["weight_decay"],
        momentum=HPARAM["momentum"],
    )
    scheduler = make_scheduler(
        optimizer,
        HPARAM["step_scheduler_decay"],
        HPARAM["scheduler_step"],
    )

    plugin_list = [LRSchedulerPlugin(scheduler)]
    cl_strategy = Naive(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        train_mb_size=HPARAM["batch_size"],
        train_epochs=HPARAM["num_epoch"],
        eval_mb_size=HPARAM["batch_size"],
        evaluator=eval_plugin,
        device=device,
        plugins=plugin_list,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    print("Current protocol : ", EVALUATION_PROTOCOL)
    for index, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        res = cl_strategy.train(experience)
        torch.save(
            model.state_dict(), 
            str(MODEL_ROOT / f"model{str(index).zfill(2)}.pth")
        )
        print("Training completed")
        print(
            "Computing accuracy on the whole test set with"
            f" {EVALUATION_PROTOCOL} evaluation protocol"
        )
        results.append(cl_strategy.eval(benchmark.test_stream))
    # generate accuracy matrix
    num_timestamp = len(results)
    accuracy_matrix = np.zeros((num_timestamp, num_timestamp))
    for train_idx in range(num_timestamp):
        for test_idx in range(num_timestamp):
            accuracy_matrix[train_idx][test_idx] = results[train_idx][
                f"Top1_Acc_Stream/eval_phase/test_stream/Task00{test_idx}"
            ]
    print("Accuracy_matrix : ")
    print(accuracy_matrix)
    metric = CLEARMetric().get_metrics(accuracy_matrix)
    print(metric)

    metric_log = open(ROOT / "metric_log.txt", "w+")
    metric_log.write(f"Protocol: {EVALUATION_PROTOCOL} " f"Seed: {seed} ")
    json.dump(accuracy_matrix.tolist(), metric_log, indent=6)
    json.dump(metric, metric_log, indent=6)
    metric_log.close()


if __name__ == "__main__":
    main()
