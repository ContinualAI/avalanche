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
Example on how to use the checkpoint plugin.

This is basically a vanilla Avalanche main script, but with the replay
functionality enabled. Proper comments are provided to point out the changes
required to use the checkpoint plugin.
"""

import argparse
import os
from typing import Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import CLExperience, SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    class_accuracy_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger
from avalanche.models import SimpleMLP, as_multitask
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage
from avalanche.training.supervised import Naive


def main(args):
    # FIRST CHANGE: SET THE RANDOM SEEDS
    # In fact, you should to this no matter the checkpointing functionality.
    # Remember to load checkpoints by setting the same random seed used when
    # creating them...
    RNGManager.set_random_seeds(1234)

    # Nothing new here...
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print('Using device', device)

    # CL Benchmark Creation
    n_experiences = 5
    scenario = SplitMNIST(n_experiences=n_experiences,
                          return_task_id=True)
    input_size = 28*28*1

    train_stream: Sequence[CLExperience] = scenario.train_stream
    test_stream: Sequence[CLExperience] = scenario.test_stream

    # Define the model (and load initial weights if necessary)
    # Again, not checkpoint-related
    model = SimpleMLP(input_size=input_size,
                      num_classes=scenario.n_classes // n_experiences)
    model = as_multitask(model, 'classifier')

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

    # THIRD CHANGE: LOAD THE CHECKPOINT IF IT EXISTS
    # IF THE CHECKPOINT EXISTS, SKIP THE CREATION OF THE STRATEGY!
    # OTHERWISE, CREATE THE STRATEGY AS USUAL.
    # NOTE: add the checkpoint plugin to the list of strategy plugins!

    # Load checkpoint (if exists)
    strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()

    # Create the CL strategy (if not already loaded from checkpoint)
    if strategy is None:
        # Add the checkpoint plugin to the list of plugins!
        plugins = [
            checkpoint_plugin,
            ReplayPlugin(mem_size=500),
            # ...
        ]

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
                run_name=f'checkpointing_{args.checkpoint_at}'
            ))

        # Create the evaluation plugin (as usual)
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

        # Create the strategy (as usual)
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=128,
            train_epochs=2,
            eval_mb_size=128,
            device=device,
            plugins=plugins,
            evaluator=evaluation_plugin
        )

    # Train and test loop, as usual.
    # Notice the "if" checking "checkpoint_at", which here is only used to
    # demonstrate the checkpoint functionality. In your code, you may want
    # to add a similar check based on received early termination signals.
    # These signals may include keyboard interrupts, SLURM interrupts, etc.
    # Just keep in mind that the checkpoint is saved AFTER each eval phase.
    # If you terminate the process before the end of the eval phase,
    # all the work done between the previous checkpoint and the current moment
    # is lost.
    for train_task in train_stream[initial_exp:]:
        strategy.train(train_task, num_workers=10, persistent_workers=True)
        strategy.eval(test_stream, num_workers=10)

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
        "--checkpoint_at",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--wandb",
        action='store_true'
    )
    main(parser.parse_args())
