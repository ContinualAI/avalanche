"""
This example tests RWalk on Split MNIST and Permuted MNIST.
"""

import torch
import argparse
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from avalanche.benchmarks import PermutedMNIST, nc_benchmark
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.training.supervised import Naive
from avalanche.training.plugins import RWalkPlugin
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    bwt_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    model = SimpleMLP(hidden_size=args.hs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print(f"Using device: {device}")

    # create benchmark
    if args.scenario == "pmnist":
        benchmark = PermutedMNIST(n_experiences=args.permutations)
    elif args.scenario == "smnist":
        mnist_train = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
            transform=ToTensor(),
        )
        mnist_test = MNIST(
            root=default_dataset_location("mnist"),
            train=False,
            download=True,
            transform=ToTensor(),
        )
        benchmark = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False, seed=1234
        )
    else:
        raise ValueError("Wrong scenario name. Allowed pmnist, smnist.")

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    tensorboard_logger = TensorboardLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[interactive_logger, tensorboard_logger],
    )

    # create strategy
    strategy = Naive(
        model,
        optimizer,
        criterion,
        train_epochs=args.epochs,
        device=device,
        train_mb_size=args.minibatch_size,
        evaluator=eval_plugin,
        plugins=[
            RWalkPlugin(
                ewc_lambda=args.ewc_lambda,
                ewc_alpha=args.ewc_alpha,
                delta_t=args.delta_t,
            )
        ],
    )

    # train on the selected benchmark with the chosen strategy
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)

        strategy.train(experience)
        print("End training on experience", experience.current_experience)
        print("Computing accuracy on the test set")
        results.append(strategy.eval(benchmark.test_stream[:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["pmnist", "smnist"],
        default="smnist",
        help="Choose between Permuted MNIST, Split MNIST.",
    )
    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=0.1,
        help="Penalty hyperparameter for RWalk",
    )
    parser.add_argument(
        "--ewc_alpha",
        type=float,
        default=0.9,
        help="EWC++ alpha term.",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=10,
        help="Number of iterations after which the RWalk scores are updated.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hs", type=int, default=256, help="MLP hidden size.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=64, help="Minibatch size."
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5,
        help="Number of experiences in Permuted MNIST.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Specify GPU id to use. Use CPU if -1.",
    )
    args = parser.parse_args()

    main(args)
