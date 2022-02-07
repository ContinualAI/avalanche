################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-12-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to enable distributed training in Avalanche.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import SplitMNIST
from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_helper import hash_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive, ClassBalancedBuffer
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin


def main(args):
    DistributedHelper.init_distributed(random_seed=4321, use_cuda=args.use_cuda)
    rank = DistributedHelper.rank
    world_size = DistributedHelper.world_size
    device = DistributedHelper.make_device()
    print(f'Current process rank: {rank}/{world_size}')
    print('Will use device:', device)

    if not DistributedHelper.is_main_process:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose([
        RandomCrop(28, padding=4),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # ---------

    # --- SCENARIO CREATION
    scenario = SplitMNIST(
        5,
        train_transform=train_transform,
        eval_transform=test_transform)

    benchmark_checksum = hash_benchmark(scenario)
    DistributedHelper.check_equal_objects(benchmark_checksum)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes)

    # CREATE THE STRATEGY INSTANCE (NAIVE)

    loggers = []
    if DistributedHelper.is_main_process:
        distr_str = 'single_process'
        approach_str = 'naive'

        if DistributedHelper.is_distributed:
            distr_str = 'distributed'

        if args.use_replay:
            approach_str = 'replay'

        loggers.append(TensorboardLogger(
            tb_log_dir=f'./tb_data/{distr_str}_{approach_str}'))

    my_evaluator = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=loggers,
        suppress_warnings=True
    )

    # Adapt the minibatch size
    mb_size = 192 // DistributedHelper.world_size

    plugins = []
    if args.use_replay:
        class_balanced_policy = ClassBalancedBuffer(1500)
        plugins.append(ReplayPlugin(
            1500,
            storage_policy=class_balanced_policy))

    cl_strategy = Naive(
        model, SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(), train_mb_size=mb_size, train_epochs=4,
        eval_mb_size=mb_size, plugins=plugins,
        device=device, evaluator=my_evaluator)

    start_time = time.time()

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience, num_workers=4)

        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=4))

    print('Training+eval took', time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_replay', action='store_true')
    main(parser.parse_args())
