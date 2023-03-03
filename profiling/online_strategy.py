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

from os.path import expanduser
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import cProfile
import pstats

from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.scenarios.online_scenario import \
    fixed_size_experience_split
from avalanche.models import SimpleMLP
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
from avalanche.benchmarks.scenarios import OnlineCLScenario
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


##################################################
#   Online naive strategy without Avalanche
##################################################
def profile_online_naive_no_avl(benchmark, device):
    """
    Online naive strategy without Avalanche.
    """
    print("=" * 30)
    print("Profiling online naive strategy without Avalanche ...")

    experience_0 = benchmark.train_stream[0]

    with cProfile.Profile() as pr:
        # Initialize model, optimizer and criterion
        model = SimpleMLP(num_classes=10)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Iterate over the dataset and train the model
        dataloader = DataLoader(experience_0.dataset, batch_size=1)
        pbar = tqdm(dataloader)
        for (x, y, _) in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():0.4f}")

    stats = pstats.Stats(pr)
    stats.sort_stats('tottime').print_stats(15)


##################################################
#   Online naive strategy without Avalanche using lazy stream
##################################################
def profile_online_naive_lazy_stream(benchmark, device):
    """
    Online naive strategy without Avalanche using lazy stream.
    """

    print("=" * 30)
    print("Profiling online naive strategy  using lazy streams (no AVL) ...")

    experience_0 = benchmark.train_stream[0]

    def load_all_data(data):
        return next(iter(DataLoader(data, len(data))))

    with cProfile.Profile() as pr:
        model = SimpleMLP(num_classes=10).to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for exp in tqdm(fixed_size_experience_split(experience_0, 1)):
            x, y, _ = load_all_data(exp.dataset)
            x, y = x.to(device), torch.tensor([y]).to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    stats = pstats.Stats(pr)
    stats.sort_stats('tottime').print_stats(15)


##################################################
#        Online strategy using Avalanche
##################################################
def profile_online_avl(
        benchmark,
        device,
        strategy="naive",
        use_interactive_logger: bool = True
):
    """
    Online strategy using Avalanche.
    """
    print("=" * 30)
    print(f"Profiling online {strategy} strategy using Avalanche ...")

    experience_0 = benchmark.train_stream[0]

    # Create list of loggers to be used
    loggers = []
    if use_interactive_logger:
        interactive_logger = InteractiveLogger()
        loggers.append(interactive_logger)

    # Evaluation plugin
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers,
    )

    with cProfile.Profile() as pr:
        # Model
        model = SimpleMLP(num_classes=10)

        plugins = []
        if strategy == "er":
            # CREATE THE STRATEGY INSTANCE (ONLINE-REPLAY)
            storage_policy = ReservoirSamplingBuffer(max_size=100)
            replay_plugin = ReplayPlugin(mem_size=100, batch_size=1,
                                         storage_policy=storage_policy)
            plugins.append(replay_plugin)

        # Create OnlineNaive strategy
        cl_strategy = OnlineNaive(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            CrossEntropyLoss(),
            train_passes=1,
            train_mb_size=1,
            device=device,
            evaluator=eval_plugin,
            plugins=plugins
        )
        online_cl_scenario = OnlineCLScenario(benchmark.streams.values(),
                                              experience_0)

        # Train on the first experience only
        cl_strategy.train(online_cl_scenario.train_stream)

    stats = pstats.Stats(pr)
    stats.sort_stats('tottime').print_stats(40)


def main(args):
    # Compute device
    device = "cuda" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    print("Using ", device)

    # Benchmark
    benchmark = SplitMNIST(
        n_experiences=5,
        dataset_root=expanduser("~") + "/.avalanche/data/mnist/",
    )

    # Profilers:
    profile_online_naive_no_avl(benchmark, device)

    profile_online_naive_lazy_stream(benchmark, device)

    profile_online_avl(benchmark, device, strategy="naive")

    profile_online_avl(benchmark, device, strategy="er")


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
