################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 27-01-2022                                                             #
# Author(s): Hamed Hemati                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This simple profiler measures the amount of time required to finish an
experience in an online manner using different strategies and options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import expanduser
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import time

from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies.strategy_wrappers_online import OnlineNaive
from avalanche.training.strategies.strategy_wrappers import Naive
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset
from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


class SimpleOnlineStrategy:
    """
    The most basic form of an online strategy without any callbacks
    that simply receives an experience and iterates through its sample
    one by one and updates the model correspondingly.
    """
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        self.device = device

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, experience):
        self.model.train()
        dataloader = DataLoader(experience.dataset, batch_size=1)
        pbar = tqdm(dataloader)
        for (x, y, _) in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"Loss: {loss.item()}")


def run_simple_online(experience, device):
    """
    Runs simple naive strategy for one experience.
    """

    model = SimpleMLP(num_classes=10)
    cl_strategy = SimpleOnlineStrategy(model, device)

    start = time.time()
    print("Running SimpleOnlineStrategy ...")
    cl_strategy.train(experience)
    end = time.time()
    duration = end - start

    return duration


def run_base_online(
        experience,
        device,
        use_interactive_logger: bool = False
):
    """
    Runs OnlineNaive for one experience.
    """

    # Create list of loggers to be used
    loggers = []
    if use_interactive_logger:
        interactive_logger = InteractiveLogger()
        loggers.append(interactive_logger)

    # Evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=loggers,
    )

    # Model
    model = SimpleMLP(num_classes=10)

    # Create OnlineNaive strategy
    cl_strategy = OnlineNaive(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        num_passes=1,
        train_mb_size=1,
        device=device,
        evaluator=eval_plugin,
    )

    start = time.time()
    print("Running OnlineNaive ...")
    cl_strategy.train(experience)
    end = time.time()
    duration = end - start

    return duration


def run_base(
        experience,
        device,
        use_interactive_logger: bool = False
):
    """
        Runs Naive (from BaseStrategy) for one experience.
    """

    def create_sub_experience_list(experience):
        """Creates a list of sub-experiences from an experience.
        It returns a list of experiences, where each experience is
        a subset of the original experience.

        :param experience: single Experience.

        :return: list of Experience.
        """

        # Shuffle the indices
        indices = torch.randperm(len(experience.dataset))
        num_sub_exps = len(indices)
        mb_size = 1
        sub_experience_list = []
        for subexp_id in range(num_sub_exps):
            subexp_indices = indices[
                             subexp_id * mb_size: (subexp_id + 1) * mb_size]
            sub_experience = copy.copy(experience)
            subexp_ds = AvalancheSubset(
                sub_experience.dataset, indices=subexp_indices
            )
            sub_experience.dataset = subexp_ds
            sub_experience_list.append(sub_experience)

        return sub_experience_list

    # Create list of loggers to be used
    loggers = []
    if use_interactive_logger:
        interactive_logger = InteractiveLogger()
        loggers.append(interactive_logger)

    # Evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=loggers,
    )

    # Model
    model = SimpleMLP(num_classes=10)

    # Create OnlineNaive strategy
    cl_strategy = Naive(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        train_mb_size=1,
        device=device,
        evaluator=eval_plugin,
    )

    start = time.time()
    sub_experience_list = create_sub_experience_list(experience)

    # !!! This is only for profiling purpose. This method may not work
    # in practice for dynamic modules since the model adaptation step
    # can go wrong.

    # Train for each sub-experience
    print("Running OnlineNaive ...")
    for i, sub_experience in enumerate(sub_experience_list):
        experience = sub_experience
        cl_strategy.train(experience)
    end = time.time()
    duration = end - start

    return duration


def main(args):
    # Compute device
    device = "cuda" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    print("Using ", device)

    # Benchmark
    benchmark = SplitMNIST(
        n_experiences=5,
        dataset_root=expanduser("~") + "/.avalanche/data/mnist/",
    )

    # Train for the first experience only
    experience_0 = benchmark.train_stream[0]

    # Run tests
    dur_simple = run_simple_online(experience_0, device)
    dur_base_online = run_base_online(experience_0, device,
                                      use_interactive_logger=False)
    dur_base_online_intlog = run_base_online(experience_0, device,
                                             use_interactive_logger=True)
    dur_base = run_base(experience_0, device,
                        use_interactive_logger=False)
    dur_base_intlog = run_base(experience_0, device,
                               use_interactive_logger=True)

    print(f"Duration for SimpleOnlineStrategy: ", dur_simple)
    print(f"Duration for BaseOnlineStrategy: ", dur_base_online)
    print(f"Duration for BaseOnlineStrategy+IntLogger: ",
          dur_base_online_intlog)
    print(f"Duration for BaseStrategy: ", dur_base)
    print(f"Duration for BaseStrategy+IntLogger: ",
          dur_base_intlog)


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
