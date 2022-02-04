################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-06-2021                                                             #
# Author: Timm Hess                                                            #
# E-mail: hess@ccc.cs.uni-frankfurt.de                                         #
# Website: avalanche.continualai.org                                           #
################################################################################


"""
This example makes use of the Endless-Continual-Learning-Simulator's derived 
dataset scenario, trained with a Naive strategy. The default scenario is an
incremental classification learning setting ('scenario = Classes'), however
also the 'Illumination' and 'Weather' learning scenarios can be selected.
"""


import argparse

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


from avalanche.benchmarks.classic import EndlessCLSim
from avalanche.models import SimpleCNN
from avalanche.training.supervised import Naive


def main(args):
    # Config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # Model
    model = SimpleCNN(num_classes=5)

    # CL Benchmark Creation
    scenario = EndlessCLSim(
        scenario=args.scenario,  # "Classes", "Illumination", "Weather"
        sequence_order=None,
        task_order=None,
        semseg=args.semseg,
        dataset_root=args.dataset_root,
    )

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Prepare for training & testing
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    # Choose a CL strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=64,
        train_epochs=3,
        eval_mb_size=128,
        device=device,
    )

    # Train and test loop
    for train_task in train_stream:
        strategy.train(train_task, num_worker=0)
        strategy.eval(test_stream)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="Classes",
        help="Select scenario: Classes, Illumination, Weather",
    )
    parser.add_argument("--semseg", action="store_true", default=False)
    parser.add_argument("--dataset_root", type=str, default=None)
    args = parser.parse_args()
    main(args)

    print("Done..")
