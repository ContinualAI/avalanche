import argparse
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10
from avalanche.benchmarks.classic.ccifar100 import SplitCIFAR100
from avalanche.models import SimpleCNN
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

from avalanche.training.plugins import HuberEWCPlugin, EWCPlugin


def _resolve_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(benchmark_name: str, use_huber: bool, device, mb_size: int, epochs: int):
    if benchmark_name == "cifar10":
        bench = SplitCIFAR10(n_experiences=5, seed=42)
        num_classes = 10
    elif benchmark_name == "cifar100":
        bench = SplitCIFAR100(n_experiences=10, seed=42)
        num_classes = 100
    else:
        raise ValueError("benchmark_name must be 'cifar10' or 'cifar100'")

    # SimpleCNN has built-ins for CIFAR
    model = SimpleCNN(num_classes=num_classes)
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )

    plugin = (
        HuberEWCPlugin(ewc_lambda=50.0, beta=0.5)
        if use_huber
        else EWCPlugin(ewc_lambda=50.0)
    )

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=mb_size,
        train_epochs=epochs,
        eval_mb_size=mb_size,
        device=device,
        plugins=[plugin],
        evaluator=eval_plugin,
    )

    for exp in bench.train_stream:
        print(f"Start of experience {exp.current_experience}")
        strategy.train(exp, num_workers=2, pin_memory=False)
        print("End of experience", exp.current_experience)
        print("Evaluation on test stream")
        strategy.eval(bench.test_stream, num_workers=2, pin_memory=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, default="cifar10", choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--use_huber", type=int, default=1)
    parser.add_argument("--train_mb_size", type=int, default=128)
    parser.add_argument("--train_epochs", type=int, default=2)
    args = parser.parse_args()

    device = _resolve_device()
    run(
        args.benchmark,
        bool(args.use_huber),
        device,
        args.train_mb_size,
        args.train_epochs,
    )
