"""
This example trains a simple MLP on Permuted MNIST with Improved Elastic Weight
Consolidation (IEWC). IEWC uses the same penalty as EWC but normalizes each
per-sample term of the importance by the squared norm of the loss gradient with
respect to the model output (https://arxiv.org/abs/2611.16039).
"""

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import IEWC
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # Config
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    # model
    model = SimpleMLP(num_classes=10)

    # CL Benchmark Creation
    benchmark = PermutedMNIST(n_experiences=5, seed=1234)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    # choose some metrics and evaluation method
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[InteractiveLogger()],
    )

    # Choose a CL strategy
    strategy = IEWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ewc_lambda=args.ewc_lambda,
        kappa=1e-2,
        train_mb_size=128,
        train_epochs=1,
        eval_mb_size=512,
        device=device,
        evaluator=eval_plugin,
    )

    # train and test loop
    for train_task in train_stream:
        strategy.train(train_task)
        strategy.eval(test_stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument("--ewc_lambda", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
