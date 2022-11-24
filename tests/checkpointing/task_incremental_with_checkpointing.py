################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-09-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
This is a deterministic version of the script with the same name found in the
examples folder.

Used in unit tests.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import CLExperience, SplitCIFAR100, CLStream51, \
    SplitOmniglot, SplitMNIST, SplitFMNIST
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    class_accuracy_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger
from avalanche.models import SimpleMLP, as_multitask
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin, CWRStarPlugin, \
    ReplayPlugin, GDumbPlugin, LwFPlugin, SynapticIntelligencePlugin, EWCPlugin
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage
from avalanche.training.supervised import Naive
from tests.unit_tests_utils import get_fast_benchmark


def main(args):
    # FIRST CHANGE: SET THE RANDOM SEEDS
    # In fact, you should to this no matter the checkpointing functionality.
    # Remember to load checkpoints by setting the same random seed used when
    # creating them...
    RNGManager.set_random_seeds(1234)

    # Determinism is not strictly required, but it is here added to show that
    # checkpointing doesn't alter the results of an experiment.
    # If you don't want to use deterministic algorithms, you can remove this.
    # Using `use_deterministic_algorithms` may require setting the
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable.
    torch.use_deterministic_algorithms(True)

    # Nothing new here...
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print('Using device', device)

    # Code used to select the benchmark: not checkpoint-related
    use_tasks = 'si' not in args.plugins and 'cwr' not in args.plugins \
        and args.benchmark != 'Stream51'
    input_size = 32*32*3
    # CL Benchmark Creation
    if args.benchmark == 'TestBenchmark':
        input_size = 28 * 28 * 1
        scenario = get_fast_benchmark(
            use_task_labels=True,
            n_features=input_size,
            n_samples_per_class=256,
            seed=1337
        )
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
    elif args.benchmark == 'Stream51':
        scenario = CLStream51()
        scenario.n_classes = 51
        input_size = 224*224*3
    elif args.benchmark == 'SplitOmniglot':
        scenario = SplitOmniglot(n_experiences=4, return_task_id=use_tasks)
        input_size = 105*105*1
    else:
        raise ValueError('Unrecognized benchmark name from CLI.')
    train_stream: Sequence[CLExperience] = scenario.train_stream
    test_stream: Sequence[CLExperience] = scenario.test_stream

    # Define the model (and load initial weights if necessary)
    # Again, not checkpoint-related
    if use_tasks:
        model = SimpleMLP(input_size=input_size,
                          num_classes=scenario.n_classes // 5)
        model = as_multitask(model, 'classifier')
    else:
        model = SimpleMLP(input_size=input_size, num_classes=scenario.n_classes)

    # Prepare for training & testing: not checkpoint-related
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    # SECOND CHANGE: INSTANTIATE THE CHECKPOINT PLUGIN
    # FileSystemCheckpointStorage is a good default choice.
    # The provided directory should point to the SPECIFIC experiment: do not
    # re-use the same folder for different experiments/runs.
    # Obvious noob advice: do not use a runtime-computed timestamp for the
    # directory name, or you will end up by NOT loading the previous
    # checkpoint ;)
    # Please notice the `map_location`: you should set the current device there.
    # That will take care of loading the checkpoint on the correct device, even
    # if it was previously produced on a cuda device with a different id. It
    # can also be used to resume a cuda checkpoint from cuda to CPU.
    # However, it will not work when loading a CPU checkpoint to cuda...
    # In brief: CUDA -> CPU (OK), CUDA:0 -> CUDA:1 (OK), CPU -> CUDA (NO!)
    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
            directory='./checkpoints/task_incremental',
        ),
        map_location=device
    )

    # THIRD CHANGE: LOAD THE CHECKPOINT IF EXISTS
    # IF THE CHECKPOINT EXISTS, SKIP THE CREATION OF THE STRATEGY!
    # OTHERWISE, CREATE THE STRATEGY AS USUAL.
    # NOTE: add the checkpoint plugin to the list of strategy plugins!

    # Load checkpoint (if exists)
    strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()

    # Create the CL strategy (if not already loaded from checkpoint)
    if strategy is None:
        # Add the checkpoint plugin to the list of plugins!
        plugins = [
            checkpoint_plugin
        ]

        # Create other plugins
        # ...
        cli_plugins = []
        cli_plugin_names = '_'.join(args.plugins)
        for cli_plugin in args.plugins:
            if cli_plugin == 'cwr':
                plugin_instance = CWRStarPlugin(
                    model, freeze_remaining_model=False)
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
            else:
                raise ValueError('Unrecognized plugin name from CLI.')
            print('Adding plugin', plugin_instance)
            cli_plugins.append(plugin_instance)
        plugins += cli_plugins

        # Create loggers (as usual)
        os.makedirs(f'./logs/checkpointing_{args.checkpoint_at}',
                    exist_ok=True)

        loggers = [
            TextLogger(
                open(f'./logs/checkpointing_'
                     f'{args.checkpoint_at}/log.txt', 'w')),
            InteractiveLogger(),
            TensorboardLogger(f'./logs/checkpointing_{args.checkpoint_at}')
        ]

        if args.wandb:
            loggers.append(WandBLogger(
                project_name='AvalancheCheckpointing',
                run_name=f'checkpointing_{args.benchmark}_'
                         f'{args.checkpoint_at}_'
                         f'{cli_plugin_names}'
            ))

        # Create the evaluation plugin (when not using the default one)
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

        # Create the strategy
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=128,
            train_epochs=1,
            eval_mb_size=128,
            device=device,
            plugins=plugins,
            evaluator=evaluation_plugin
        )

    # Train and test loop: as usual
    # Notice the if checking "checkpoint_at", which here is only used to
    # demonstrate the checkpoint functionality. In your code, you may want
    # to add a similar check based on received early termination signals.
    # These signals may include keyboard interrupts, SLURM interrupts, etc.
    # Just keep in mind that the checkpoint is saved AFTER each eval phase.
    # If you terminate the process before the end of the eval phase,
    # all the work done between the previous checkpoint and the current moment
    # is lost.
    for train_task in train_stream[initial_exp:]:
        strategy.train(train_task, num_workers=2, persistent_workers=True)
        metrics = strategy.eval(test_stream, num_workers=2)

        Path(args.log_metrics_to).mkdir(parents=True, exist_ok=True)
        with open(Path(args.log_metrics_to) /
                  f'metrics_exp{train_task.current_experience}.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        if train_task.current_experience == args.checkpoint_at:
            print('Exiting early')
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default='SplitCifar100',
        help="The benchmark to use."
    )
    parser.add_argument(
        "--checkpoint_at",
        type=int,
        default=-1
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
