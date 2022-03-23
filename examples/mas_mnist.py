import torch
import argparse
from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks import PermutedMNIST
from avalanche.benchmarks import RotatedMNIST
from avalanche.training.supervised import MAS
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


"""
This example tests Memory Aware Synapses (MAS) on either Split MNIST,
Rotated MNIST or Permuted MNIST.
"""


def main(args):
    model = SimpleMLP(hidden_size=args.hs)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print(f"Using device: {device}")

    # create split scenario
    if args.scenario == "pmnist":
        scenario = PermutedMNIST(n_experiences=args.experiences)
    elif args.scenario == "smnist":
        scenario = SplitMNIST(n_experiences=args.experiences)
    elif args.scenario == "rmnist":
        scenario = RotatedMNIST(n_experiences=args.experiences)
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # Create strategy
    strategy = MAS(
        model,
        optimizer,
        criterion,
        lambda_reg=args.mas_lambda,
        alpha=args.alpha,
        verbose=args.verbose,
        train_epochs=args.epochs,
        device=device,
        train_mb_size=args.minibatch_size,
        evaluator=eval_plugin,
    )

    # train on the selected scenario with the chosen strategy
    print("Starting experiment...")
    results = []
    for train_batch_info in scenario.train_stream:
        print(
            "Start training on experience ",
            train_batch_info.current_experience
        )

        strategy.train(train_batch_info, num_workers=0)
        print(
            "End training on experience ", train_batch_info.current_experience
        )
        print("Computing accuracy on the test set")
        results.append(strategy.eval(scenario.test_stream[:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Strategy parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hyperparameter for influence update"
    )
    parser.add_argument(
        "--mas_lambda",
        type=float,
        default=1.0,
        help="Hyperparameter to weight penalty term"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Shows a progress bar during the estimate of the influence"
    )
    # Benchmark paramaters
    parser.add_argument(
        "--scenario",
        type=str,
        default="smnist",
        help="The benchmark scenario to use"
    )
    parser.add_argument(
        "--experiences",
        type=int,
        default=5,
        help="Number of experiences in the benchmark"
    )
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--hs", type=int, default=256, help="MLP hidden size.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=128, help="Minibatch size."
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Specify GPU id to use. Use CPU if -1.",
    )
    args = parser.parse_args()

    main(args)
