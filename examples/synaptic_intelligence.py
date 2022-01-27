################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 26-01-2021                                                            #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Synaptic Intelligence Plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

from avalanche.benchmarks import SplitCIFAR10
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.models.mobilenetv1 import MobilenetV1
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised.strategy_wrappers import SynapticIntelligence
from avalanche.training.utils import adapt_classification_layer


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose(
        [Resize(224), ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_transform = transforms.Compose(
        [Resize(224), ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # ---------

    # --- SCENARIO CREATION
    scenario = SplitCIFAR10(
        5, train_transform=train_transform, eval_transform=test_transform
    )
    # ---------

    # MODEL CREATION
    model = MobilenetV1()
    adapt_classification_layer(model, scenario.n_classes, bias=False)

    # DEFINE THE EVALUATION PLUGIN AND LOGGER

    my_logger = TensorboardLogger(
        tb_log_dir="logs", tb_log_exp_name="logging_example"
    )

    # print to stdout
    interactive_logger = InteractiveLogger()

    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[my_logger, interactive_logger],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE with the Synaptic Intelligence plugin)
    cl_strategy = SynapticIntelligence(
        model,
        Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        si_lambda=0.0001,
        train_mb_size=128,
        train_epochs=4,
        eval_mb_size=128,
        device=device,
        evaluator=evaluation_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(scenario.test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
