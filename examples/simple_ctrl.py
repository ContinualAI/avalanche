################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
In this simple example we show all the different ways you can use MNIST with
Avalanche.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
from pprint import pprint

import torch
import argparse

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST, \
    SplitMNIST
from avalanche.benchmarks.classic.ctrl import CTrL
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.training import EvaluationPlugin
from avalanche.training.strategies import Naive


def main(args):
    # Device config
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # model
    # model = SimpleMLP(num_classes=11, input_size=3*32*32)
    model = SimpleCNN(num_classes=10)
    model_last = deepcopy(model)

    scenario = CTrL(stream_name='s_plus')

    # Than we can extract the parallel train and test streams
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Continual learning strategy with default logger

    logger = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True,
                         stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True,
                     stream=True),
        loggers=[InteractiveLogger()])
    cl_strategy = Naive(
        model, optimizer, criterion, train_mb_size=32, train_epochs=200,
        eval_mb_size=128, device=device, evaluator=logger)

    # train and test loop
    assert logger == cl_strategy.evaluator
    for tid, train_task in enumerate(train_stream):
        cl_strategy.train(train_task)
        cl_strategy.eval(test_stream)

    transfer_mat = []
    for tid in range(len(train_stream)):
        transfer_mat.append(
            logger.all_metric_results[f'Top1_Acc_Exp/eval_phase/test_stream/'
                                      f'Task00{tid}/Exp00{tid}'][1])
    # cl_strategy.evaluator.all_metric_results[
    #     f'Top1_Acc_Exp/eval_phase/test_stream/Task00{tid}/Exp00{tid}'][1])

    print(torch.tensor(transfer_mat))
    # pprint(logger.all_metric_results)
    pprint(logger.last_metric_results)

    optimizer = SGD(model_last.parameters(), lr=0.001, momentum=0.9)
    cl_strategy = Naive(
        model_last, optimizer, criterion, train_mb_size=32, train_epochs=200,
        eval_mb_size=32, device=device)
    cl_strategy.train(train_task)
    res = cl_strategy.eval([test_stream[-1]])
    print()
    print(res)

    print(transfer_mat[-1][-1] - res['Top1_Acc_Exp/eval_phase/test_stream/'
                                     'Task005/Exp-01'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist_type', type=str, default='split',
                        choices=['rotated', 'permuted', 'split'],
                        help='Choose between MNIST variations: '
                             'rotated, permuted or split.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()

    main(args)
