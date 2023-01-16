################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-12-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a deterministic version of the script with the same name found in the
examples folder.

Used in unit tests.

Adapted from the one used for unit testing the checkpointing functionality.
"""


import argparse
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from avalanche.benchmarks import CLExperience, \
    SplitCIFAR100, SplitMNIST, SplitFMNIST, SplitCIFAR10
from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_consistency_verification import \
    hash_benchmark, hash_model
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    class_accuracy_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger
from avalanche.models import SimpleMLP, as_multitask
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, CWRStarPlugin, \
    ReplayPlugin, GDumbPlugin, LwFPlugin, SynapticIntelligencePlugin, \
    EWCPlugin, LRSchedulerPlugin, SupervisedPlugin
from tests.unit_tests_utils import get_fast_benchmark

OVERALL_MB_SIZE = 192
BENCHMARK_HASH = \
    '8ac6f78597e6f7279c601f1f75113aec6c56abd1518e3386a6729c7be9262cdd'
MODEL_HASH = \
    'cbb45bc281908892402fda9794e82d71c3593631f76229f1f396fa7a936affaa'


class CheckModelAlignedPlugin(SupervisedPlugin):

    supports_distributed = True
    
    def after_update(self, strategy, *args, **kwargs):
        DistributedHelper.check_equal_objects(
            hash_model(strategy.model, include_buffers=True))


def main(args):
    torch.use_deterministic_algorithms(True)

    is_dist = DistributedHelper.init_distributed(
        random_seed=4321, use_cuda=args.cuda
    )

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

    # --- SCENARIO CREATION
    use_tasks = 'si' not in args.plugins and 'cwr' not in args.plugins \
        and args.benchmark != 'Stream51'
    input_size = 32*32*3

    if args.benchmark == 'TestBenchmark':
        input_size = 28 * 28 * 1
        scenario = get_fast_benchmark(
            use_task_labels=use_tasks,
            n_features=input_size,
            n_samples_per_class=256,
            seed=1337
        )
        
        if use_tasks:
            # print(hash_benchmark(scenario, num_workers=4))
            assert hash_benchmark(scenario, num_workers=4) == BENCHMARK_HASH
            print('Benchmark hash is correct.')
    elif args.benchmark == 'SplitMNIST':
        scenario = SplitMNIST(n_experiences=5, return_task_id=True)
        input_size = 28*28*1
    elif args.benchmark == 'SplitFMNIST':
        scenario = SplitFMNIST(n_experiences=5, return_task_id=True)
        input_size = 28*28*1
    elif args.benchmark == 'SplitCifar100':
        scenario = SplitCIFAR100(n_experiences=5, return_task_id=use_tasks)
    elif args.benchmark == 'SplitCifar10':
        scenario = SplitCIFAR10(n_experiences=5, return_task_id=use_tasks)
    else:
        raise ValueError('Unrecognized benchmark name from CLI.')
    train_stream: Sequence[CLExperience] = scenario.train_stream
    test_stream: Sequence[CLExperience] = scenario.test_stream

    print('Testing using the', args.benchmark, 'benchmark')
    # ---------

    # MODEL CREATION
    if use_tasks:
        model = SimpleMLP(input_size=input_size,
                          num_classes=scenario.n_classes // 5)
        model = as_multitask(model, 'classifier')
        if args.benchmark == 'TestBenchmark' and use_tasks:
            # print(hash_model(model))
            assert hash_model(model) == MODEL_HASH
            print('Model hash is correct.')
    else:
        model = SimpleMLP(input_size=input_size, num_classes=scenario.n_classes)

    DistributedHelper.check_equal_objects(
        hash_model(model, include_buffers=True))
    DistributedHelper.check_equal_objects(
        hash_benchmark(scenario, num_workers=4))
    
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    # CREATE THE STRATEGY INSTANCE (NAIVE)

    # Adapt the minibatch size
    mb_size = OVERALL_MB_SIZE // DistributedHelper.world_size

    plugins = [
        CheckModelAlignedPlugin()
    ]

    cli_plugins = []
    cli_plugin_names = '_'.join(args.plugins)
    for cli_plugin in args.plugins:
        if cli_plugin == 'cwr':
            plugin_instance = CWRStarPlugin(
                model, freeze_remaining_model=True)
        elif cli_plugin == 'replay':
            plugin_instance = ReplayPlugin(mem_size=500)
        elif cli_plugin == 'gdumb':
            plugin_instance = GDumbPlugin(mem_size=500)
        elif cli_plugin == 'lwf':
            plugin_instance = LwFPlugin()
        elif cli_plugin == 'si':
            plugin_instance = SynapticIntelligencePlugin(0.001)
        elif cli_plugin == 'ewc':
            plugin_instance = EWCPlugin(0.001)
        elif cli_plugin == 'reduce_on_plateau':
            plugin_instance = LRSchedulerPlugin(
                ReduceLROnPlateau(optimizer), step_granularity='iteration',
                metric='train_loss'
            )
        else:
            raise ValueError('Unrecognized plugin name from CLI.')
        print('Adding plugin', plugin_instance)
        cli_plugins.append(plugin_instance)
    plugins += cli_plugins

    loggers = []
    if DistributedHelper.is_main_process:
        use_cuda_str = 'cuda' if args.cuda else 'cpu'
        is_dist_str = 'distributed' if is_dist else 'single'
        eval_every = f'peval{args.eval_every}'

        log_location: Path = Path('logs') / \
            (f'distributed_{args.benchmark}_' + 
             f'{use_cuda_str}_{is_dist_str}_{eval_every}_{cli_plugin_names}')

        #  Loggers should be created in the main process only
        os.makedirs(log_location, exist_ok=True)
        loggers = [
            TextLogger(open(log_location / 'log.txt', 'w')),
            InteractiveLogger(),
            TensorboardLogger(log_location)
        ]

        if args.wandb:
            loggers.append(WandBLogger(
                project_name='AvalancheDistributedTraining',
                run_name=f'distributed_{args.benchmark}_'
                         f'{use_cuda_str}_{is_dist_str}_'
                         f'{eval_every}_{cli_plugin_names}'
            ))
        Path(args.log_metrics_to).mkdir(parents=True, exist_ok=True)
    
    # Metrics should be created as usual, with no differences between main and
    # non-main processes.
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True,
                         experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True,
                     experience=True, stream=True),
        class_accuracy_metrics(
            stream=True
        ),
        loggers=loggers
    )

    cl_strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=mb_size,
        train_epochs=2,
        eval_mb_size=mb_size,
        eval_every=args.eval_every,
        peval_mode=args.eval_every_mode,
        device=device,
        plugins=plugins,
        evaluator=evaluation_plugin
    )

    start_time = time.time()

    # TRAINING LOOP

    for experience in train_stream:
        cl_strategy.train(
            experience,
            num_workers=8,
            drop_last=True,
            shuffle=False)

        metrics = cl_strategy.eval(
            test_stream,
            num_workers=8, 
            drop_last=True,
            shuffle=False)

        if DistributedHelper.is_main_process:
            with open(Path(args.log_metrics_to) /
                      f'metrics_exp'
                      f'{experience.current_experience}.pkl', 'wb') as f:
                pickle.dump(metrics, f)

    print('Training+eval took', time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cuda',
        default=False,
        action='store_true',
        help="If set, use GPUs."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default='SplitCifar100',
        help="The benchmark to use."
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=-1,
        help="Evaluation frequency."
    )
    parser.add_argument(
        "--eval_every_mode",
        type=str,
        default="epoch",
        help="Periodic evaluation mode (epoch, experience, iteration)."
    )
    parser.add_argument(
        "--log_metrics_to",
        type=str,
        default='./metrics'
    )
    parser.add_argument(
        "--wandb",
        action='store_true'
    )
    parser.add_argument(
        "--plugins",
        nargs='*',
        required=False,
        default=[]
    )
    main(parser.parse_args())
