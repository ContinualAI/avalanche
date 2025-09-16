import argparse
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from avalanche.benchmarks.classic.cmnist import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

from avalanche.training.plugins import HuberEWCPlugin
from avalanche.training.plugins.pseudo_annotation import PseudoAnnotationPlugin


def _resolve_device(preference: str) -> torch.device:
    pref = preference.lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps":
        return (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    if pref == "cuda":
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    return torch.device("cpu")


def main(args: argparse.Namespace):
    if args.seed is not None:
        torch.manual_seed(int(args.seed))

    device = _resolve_device(args.device)

    benchmark = SplitMNIST(n_experiences=args.n_experiences, seed=args.seed or 42)

    model = SimpleMLP(num_classes=benchmark.n_classes, input_size=28 * 28)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )

    huber_plugin = HuberEWCPlugin(ewc_lambda=args.ewc_lambda, beta=args.beta)
    pseudo_plugin = PseudoAnnotationPlugin(confidence_thresh=args.confidence_thresh)

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=args.train_mb_size,
        train_epochs=args.train_epochs,
        eval_mb_size=args.eval_mb_size,
        device=device,
        plugins=[huber_plugin, pseudo_plugin],
        evaluator=eval_plugin,
    )

    for experience in benchmark.train_stream:
        print(f"Start of experience {experience.current_experience}")
        print("Current classes:", experience.classes_in_this_experience)
        strategy.train(
            experience,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )
        print("End of experience", experience.current_experience)
        print("Evaluation on test stream")
        strategy.eval(
            benchmark.test_stream,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument("--train_mb_size", type=int, default=64)
    parser.add_argument("--eval_mb_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--confidence_thresh", type=float, default=0.95)
    parser.add_argument("--n_experiences", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)
    args = parser.parse_args()
    main(args)
