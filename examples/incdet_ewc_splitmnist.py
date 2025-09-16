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


def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    device = _resolve_device()
    benchmark = SplitMNIST(n_experiences=2, seed=42)

    model = SimpleMLP(num_classes=benchmark.n_classes, input_size=28 * 28)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )

    huber_plugin = HuberEWCPlugin(ewc_lambda=1000.0, beta=0.5)
    pseudo_plugin = PseudoAnnotationPlugin(confidence_thresh=0.95)

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=64,
        train_epochs=1,
        eval_mb_size=64,
        device=device,
        plugins=[huber_plugin, pseudo_plugin],
        evaluator=eval_plugin,
    )

    for experience in benchmark.train_stream:
        print(f"Start of experience {experience.current_experience}")
        print("Current classes:", experience.classes_in_this_experience)
        strategy.train(experience, num_workers=0, pin_memory=False)
        print("End of experience", experience.current_experience)
        print("Evaluation on test stream")
        strategy.eval(benchmark.test_stream, num_workers=0, pin_memory=False)


if __name__ == "__main__":
    main()
