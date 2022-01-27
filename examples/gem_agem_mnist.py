import torch
import argparse
from avalanche.benchmarks import PermutedMNIST, SplitMNIST
from avalanche.training.supervised import GEM, AGEM
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

"""
This example tests both GEM and A-GEM on Split MNIST and Permuted MNIST.
GEM is a streaming strategy, that is it uses only 1 training epochs.
A-GEM may use a larger number of epochs.
Both GEM and A-GEM work with small mini batches (usually with 10 patterns).

Warning1: This implementation of GEM and A-GEM does not use task vectors.
Warning2: GEM is much slower than A-GEM.

Results (learning rate is always 0.1):

GEM-PMNIST (5 experiences):
Hidden size 512. 1 training epoch. 512 patterns per experience, 0.5 memory 
strength. Average Accuracy over all experiences at the end of training on the 
last experience: 92.6%

GEM-SMNIST:
Patterns per experience: 256, Memory strength: 0.5, hidden size: 256
Average Accuracy over all experiences at the end of training on the last 
experience: 93.3%

AGEM-PMNIST (5 experiences):
Patterns per experience = sample size: 256. 256 hidden size, 1 training epoch.
Average Accuracy over all experiences at the end of training on the last 
experience: 83.5%

AGEM-SMNIST:
Patterns per experience = sample size: 256, 512, 1024. Performance on previous 
tasks remains very bad in terms of forgetting. Training epochs do not change 
result.
Hidden size 256.
Results for 1024 patterns per experience and sample size, 1 training epoch.
Average Accuracy over all experiences at the end of training on the last 
experience: 67.0%

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

    # create scenario
    if args.scenario == "pmnist":
        scenario = PermutedMNIST(n_experiences=args.permutations)
    elif args.scenario == "smnist":
        scenario = SplitMNIST(n_experiences=5, return_task_id=False)
    else:
        raise ValueError("Wrong scenario name. Allowed pmnist, smnist.")

    # choose some metrics and evaluation method
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
    if args.strategy == "gem":
        strategy = GEM(
            model,
            optimizer,
            criterion,
            args.patterns_per_exp,
            args.memory_strength,
            train_epochs=args.epochs,
            device=device,
            train_mb_size=10,
            evaluator=eval_plugin,
        )
    elif args.strategy == "agem":
        strategy = AGEM(
            model,
            optimizer,
            criterion,
            args.patterns_per_exp,
            args.sample_size,
            train_epochs=args.epochs,
            device=device,
            train_mb_size=10,
            evaluator=eval_plugin,
        )
    else:
        raise ValueError("Wrong strategy name. Allowed gem, agem.")
    # train on the selected scenario with the chosen strategy
    print("Starting experiment...")
    results = []
    for experience in scenario.train_stream:
        print("Start training on experience ", experience.current_experience)

        strategy.train(experience)
        print("End training on experience ", experience.current_experience)
        print("Computing accuracy on the test set")
        results.append(strategy.eval(scenario.test_stream[:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["gem", "agem"],
        default="gem",
        help="Choose between GEM and A-GEM",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["pmnist", "smnist"],
        default="smnist",
        help="Choose between Permuted MNIST, Split MNIST.",
    )
    parser.add_argument(
        "--patterns_per_exp",
        type=int,
        default=256,
        help="Patterns to store in the memory for each" " experience",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=256,
        help="Number of patterns to sample from memory when \
                        projecting gradient. A-GEM only.",
    )
    parser.add_argument(
        "--memory_strength",
        type=float,
        default=0.5,
        help="Offset to add to the projection direction. " "GEM only.",
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument("--hs", type=int, default=256, help="MLP hidden size.")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
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
