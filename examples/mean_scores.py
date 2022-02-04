from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
from datetime import datetime

import torch
import torch.optim.lr_scheduler
from torch.optim import Adam
from avalanche.benchmarks import SplitMNIST
from avalanche.evaluation.metrics.mean_scores import mean_scores_metrics
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import TensorboardLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(cuda: int):
    # --- CONFIG
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    )

    # --- SCENARIO CREATION
    scenario = SplitMNIST(n_experiences=5, seed=42)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes)

    # choose some metrics and evaluation method
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True),
        mean_scores_metrics(on_train=True, on_eval=True),
        loggers=[
            TensorboardLogger(f"tb_data/{datetime.now()}"),
            InteractiveLogger(),
        ],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        Adam(model.parameters()),
        train_mb_size=128,
        train_epochs=2,
        eval_mb_size=128,
        device=device,
        plugins=[ReplayPlugin(mem_size=100)],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    for i, experience in enumerate(scenario.train_stream, 1):
        cl_strategy.train(experience)
        cl_strategy.eval(scenario.test_stream[:i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args.cuda)
