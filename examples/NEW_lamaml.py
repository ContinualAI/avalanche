import torch
from os.path import expanduser

from avalanche.models import MTSimpleMLP
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging import InteractiveLogger
from avalanche.training.templates.common_templates import (
    SupervisedMetaLearningTemplate
)
from avalanche.training.plugins.NEW_lamaml import LaMAMLPlugin


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scenario = SplitMNIST(
        n_experiences=5,
        dataset_root=expanduser("~") + "/.avalanche/data/mnist/",
        return_task_id=True
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    model = MTSimpleMLP(hidden_size=128)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # LaMAML plugin
    lamaml_plugin = LaMAMLPlugin(
        n_inner_updates=5,
        second_order=True,
        grad_clip_norm=1.0,
        learn_lr=True,
        lr_alpha=0.25,
        sync_update=False,
        alpha_init=0.1,
    )

    # create strategy
    strategy = SupervisedMetaLearningTemplate(
        model,
        optimizer,
        criterion,
        train_epochs=1,
        device=device,
        train_mb_size=32,
        evaluator=eval_plugin,
        plugins=[lamaml_plugin]
    )

    # train on the selected scenario with the chosen strategy
    for experience in scenario.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(scenario.test_stream[:])


if __name__ == "__main__":
    main()
