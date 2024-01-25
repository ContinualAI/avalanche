################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-09-2022                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
Example on how to use the checkpoint plugin.

This is basically a vanilla Avalanche main script, but with the checkpointing
functionality enabled. Proper comments are provided to point out the changes
required to use the checkpoint plugin.
"""
import argparse

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger

from avalanche.models import SimpleMLP
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive


def main_with_checkpointing(args):
    # STEP 1: SET THE RANDOM SEEDS to guarantee reproducibility
    RNGManager.set_random_seeds(1234)

    # Nothing new here...
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    print("Using device", device)

    # CL Benchmark Creation (as usual)
    benchmark = SplitMNIST(5)
    model = SimpleMLP(input_size=28 * 28, num_classes=10)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Create the evaluation plugin (as usual)
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True), loggers=[InteractiveLogger()]
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
        evaluator=evaluation_plugin,
    )

    # STEP 2: TRY TO LOAD THE LAST CHECKPOINT
    # if the checkpoint exists, load it into the newly created strategy
    # the method also loads the experience counter, so we know where to
    # resume training
    fname = "./checkpoint.pkl"  # name of the checkpoint file
    strategy, initial_exp = maybe_load_checkpoint(strategy, fname)

    # STEP 3: USE THE "initial_exp" to resume training
    for train_exp in benchmark.train_stream[initial_exp:]:
        strategy.train(train_exp, num_workers=4, persistent_workers=True)
        strategy.eval(benchmark.test_stream, num_workers=4)

        # STEP 4: SAVE the checkpoint after training on each experience.
        save_checkpoint(strategy, fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    main_with_checkpointing(parser.parse_args())
