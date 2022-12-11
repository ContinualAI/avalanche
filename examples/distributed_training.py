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


import argparse
import os
import sys
import time

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import SplitMNIST
from avalanche.distributed import DistributedHelper
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive, ClassBalancedBuffer
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, \
    LRSchedulerPlugin

OVERALL_MB_SIZE = 192


def main(args):
    # >> Notes on enabling distributed training support in Avalanche <<
    #
    # There are only a few changes to be made when enabling distributed
    # training in Avalanche. These are all shown in this example. To recap:
    #
    # 1. Wrap the main code in a function. Call that function from
    #    within a "if __name__ == '__main__':" section.
    # 2. Add a call to `init_distributed` at the beginning of the main function.
    #    Obtain the device object using `make_device`.
    # 3. (Optional, recommended) Suppress the output for non-main processes.
    # 4. (If needed) Avalanche classic benchmarks already have proper ways
    #    to ensure that dataset files are not downloaded and written
    #    concurrently. If you need to dynamically download a custom dataset or
    #    create other working files, do it in the main process only (the one
    #    with rank 0).
    # 5. Loggers cannot be created in non-main processes. Make sure you create
    #    them in the main process only. Metrics should be instantiated as usual.
    # 6. IMPORTANT! Scale your minibatch size by the number of processes used.
    #
    # Notice that these changes do not impact your ability to run the same
    # script in the classic single-process fashion.
    #
    # You can check how to run this script in a distributed way by looking at
    # the `run_distributed_training_example.sh` script in the `examples` folder.
    print('Starting experiment', args.exp_name)

    DistributedHelper.init_distributed(random_seed=4321, use_cuda=args.use_cuda)
    rank = DistributedHelper.rank
    world_size = DistributedHelper.world_size
    device = DistributedHelper.make_device()
    print(f'Current process rank: {rank}/{world_size}, '
          f'will use device: {device}')

    if not DistributedHelper.is_main_process:
        # Suppress the output of non-main processes
        # This prevents the output from being duplicated in the console
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
        n_experiences=5,
        train_transform=train_transform,
        eval_transform=test_transform)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    loggers = []
    if DistributedHelper.is_main_process:
        # Loggers should be created in the main process only
        loggers.append(TensorboardLogger(
            tb_log_dir=f'./distributed_training_logs/{args.exp_name}'))

    # Metrics should be created as usual, with no differences between main and
    # non-main processes.
    my_evaluator = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=loggers
    )

    # Adapt the minibatch size
    mb_size = OVERALL_MB_SIZE // DistributedHelper.world_size

    plugins = []
    if args.use_replay:
        class_balanced_policy = ClassBalancedBuffer(1500)
        plugins.append(ReplayPlugin(
            1500,
            storage_policy=class_balanced_policy))

    if args.use_scheduler:
        plugins.append(
            LRSchedulerPlugin(
                ReduceLROnPlateau(optimizer), step_granularity='iteration',
                metric='train_loss'
            )
        )

    cl_strategy = Naive(
        model, optimizer,
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
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--exp_name', default='dist_exp')
    main(parser.parse_args())
