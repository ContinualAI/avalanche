"""
This is a simple example on how to use the Dataset inspection plugins.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
from datetime import datetime

import torch
import torch.optim.lr_scheduler
from torch.optim import Adam
from torchvision.transforms import (
    Compose,
    RandomCrop,
    ToTensor,
    RandomHorizontalFlip,
    Normalize,
)

from avalanche.benchmarks import SplitCIFAR10
from avalanche.evaluation.metric_utils import (
    repartition_bar_chart_image_creator,
)
from avalanche.evaluation.metrics.labels_repartition import (
    labels_repartition_metrics,
)

from avalanche.evaluation.metrics.images_samples import images_samples_metrics
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
    scenario = SplitCIFAR10(n_experiences=2, seed=42)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes, input_size=196608 // 64)

    # choose some metrics and evaluation method
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True),
        images_samples_metrics(
            on_train=True,
            on_eval=True,
            n_cols=10,
            n_rows=10,
        ),
        labels_repartition_metrics(
            # image_creator=repartition_bar_chart_image_creator,
            on_train=True,
            on_eval=True,
        ),
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
        train_epochs=1,
        eval_mb_size=128,
        device=device,
        plugins=[ReplayPlugin(mem_size=1_000)],
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
