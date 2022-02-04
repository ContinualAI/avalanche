import torch
import argparse
from avalanche.benchmarks import PermutedMNIST
from avalanche.training.supervised import LFL
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


"""
This example tests Less-Forgetful Learning on Permuted MNIST.
The performance with default arguments should give an accuracy
of more than 88% on all tasks.
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

    # create Permuted MNIST scenario
    scenario = PermutedMNIST(n_experiences=4)

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # create strategy
    assert (
        len(args.lambda_e) == 1 or len(args.lambda_e) == 5
    ), "Lambda_e must be a non-empty list."
    lambda_e = args.lambda_e[0] if len(args.lambda_e) == 1 else args.lambda_e

    strategy = LFL(
        model,
        optimizer,
        criterion,
        lambda_e=lambda_e,
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
            "Start training on experience ", train_batch_info.current_experience
        )

        strategy.train(train_batch_info, num_workers=0)
        print(
            "End training on experience ", train_batch_info.current_experience
        )
        print("Computing accuracy on the test set")
        results.append(strategy.eval(scenario.test_stream[:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambda_e",
        nargs="+",
        type=float,
        default=[0.0001],
        help="Penalty hyperparameter for LFL. It can be either"
        "a list with multiple elements (one lambda_e per "
        "experience) or a list of one element (same "
        "lambda_e for all experiences).",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--hs", type=int, default=256, help="MLP hidden size.")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs."
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
