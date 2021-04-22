################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-04-2020                                                             #
# Author(s): Matthias De Lange                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the CoPE plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.optim.lr_scheduler
from torchvision.transforms import transforms

from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import CoPEPlugin
from avalanche.evaluation.metrics import ExperienceForgetting, \
    accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    """
    Last Avalanche version reference performance (online):
        Top1_Acc_Stream/eval_phase/test_stream = 0.9421
    """
    # --- DEFAULT PARAMS ONLINE DATA INCREMENTAL LEARNING
    nb_tasks = 5  # Can still design the data stream based on tasks
    epochs = 1  # All data is only seen once: Online
    batch_size = 10  # Only process small amount of data at a time
    return_task_id = False  # Data incremental (task-agnostic/task-free)
    # TODO use data_incremental_generator, now experience=task

    # --- CONFIG
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")
    # ---------

    # --- SCENARIO CREATION
    scenario = SplitMNIST(nb_tasks, return_task_id=return_task_id,
                          fixed_class_order=[i for i in range(10)])
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=args.featsize,
                      hidden_size=400, hidden_layers=2)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        ExperienceForgetting(),
        loggers=[interactive_logger])

    # CoPE PLUGIN
    cope = CoPEPlugin(mem_size=2000, p_size=args.featsize,
                      n_classes=scenario.n_classes)

    # CREATE THE STRATEGY INSTANCE (NAIVE) WITH CoPE PLUGIN
    cl_strategy = Naive(model, torch.optim.SGD(model.parameters(), lr=0.01),
                        cope.loss,  # CoPE PPP-Loss
                        train_mb_size=batch_size, train_epochs=epochs,
                        eval_mb_size=100, device=device,
                        plugins=[cope],
                        evaluator=eval_plugin
                        )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(scenario.test_stream))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    parser.add_argument('--featsize', type=int, default=100,
                        help='Feature size for the embedding.')
    args = parser.parse_args()
    main(args)
