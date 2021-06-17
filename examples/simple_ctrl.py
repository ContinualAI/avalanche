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
In this simple example we show a simple way to use the ctrl benchmark using the
s+ stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
from copy import deepcopy

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic.ctrl import CTrL
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleCNN
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.strategies import Naive


class EarlyStopper(StrategyPlugin):
    def __init__(self, patience: int, val_stream_name: str,
                 metric_name: str = 'Top1_Acc_Stream', mode: str = 'max'):
        """
        Simple plugin stopping the training when the accuracy on the
        corresponding validation task stopped progressing for a few epochs.

        :param patience: Number of epochs to wait before stopping the training.
        :param val_stream_name: Name of the validation stream to search in the
        metrics. The corresponding stream will be used to keep track of the
        evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
        reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
        given metric should me maximized (resp. minimized).
        """
        super().__init__()
        self.val_stream_name = val_stream_name
        self.patience = patience
        self.metric_name = metric_name
        self.metric_key = f'{self.metric_name}/eval_phase/' \
                          f'{self.val_stream_name}'
        if mode not in ('max', 'min'):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == 'max' else operator.lt

        self.best = None  # Contains the best val acc on the current experience
        self.best_epoch = None

    def before_training(self, strategy, **kwargs):
        self.best = None
        self.best_epoch = None

    def after_training_epoch(self, strategy, **kwargs):
        self._update_best(strategy)
        if strategy.epoch - self.best_epoch >= self.patience:
            strategy.stop_training()

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        val_acc = res.get(self.metric_key)
        if self.best is None or self.operator(val_acc, self.best):
            self.best = val_acc
            self.best_epoch = strategy.epoch


def main(args):
    # Device config
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # Intialize the model, stream and training strategy
    model = SimpleCNN(num_classes=10)
    if args.stream != 's_long':
        model_init = deepcopy(model)

    scenario = CTrL(stream_name=args.stream, save_to_disk=args.save,
                    path=args.path, seed=10)

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream
    val_stream = scenario.val_stream

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    logger = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True,
                         stream=True),
        loggers=[InteractiveLogger()])

    cl_strategy = Naive(
        model, optimizer, criterion, train_mb_size=32, device=device,
        train_epochs=args.max_epochs, eval_mb_size=128, evaluator=logger,
        plugins=[EarlyStopper(50, 'val_stream')], eval_every=5
    )

    # train and test loop
    for train_task, val_task in zip(train_stream, val_stream):
        cl_strategy.train(train_task, eval_streams=[val_task])
        cl_strategy.eval(test_stream)

    transfer_mat = []
    for tid in range(len(train_stream)):
        transfer_mat.append(
            logger.all_metric_results[f'Top1_Acc_Exp/eval_phase/test_stream/'
                                      f'Task00{tid}/Exp00{tid}'][1])

    if args.stream == 's_long':
        res = logger.last_metric_results["Top1_Acc_Stream/eval_phase/" \
                                         "test_stream"]
        print(f'Average accuracy on S_long : {res}')
    else:
        optimizer = SGD(model_init.parameters(), lr=0.001, momentum=0.9)
        cl_strategy = Naive(
            model_init, optimizer, criterion, train_mb_size=32, device=device,
            train_epochs=args.max_epochs, eval_mb_size=128,
            plugins=[EarlyStopper(50, 'val_stream')], eval_every=5
        )

        cl_strategy.train(train_stream[-1])
        res = cl_strategy.eval([test_stream[-1]])

        acc_last_stream = transfer_mat[-1][-1]
        acc_last_only = res[
            'Top1_Acc_Exp/eval_phase/test_stream/Task005/Exp-01']
        transfer_value = acc_last_stream - acc_last_only

        print(f'Accuracy on probe task after training on the whole '
              f'stream: {acc_last_stream}')
        print(f'Accuracy on probe task after trained '
              f'independently: {acc_last_only}')
        print(f'T({args.stream})={transfer_value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=str, default='s_plus',
                        choices=['s_plus', 's_minus', 's_in', 's_out', 's_pl',
                                 's_long'],
                        help='Select the CTrL Stream to train on: [s_plus], '
                             's_minus, s_in, s_out, s_pl or s_long.')
    parser.add_argument('--save', type=bool, default=False,
                        help='Whether to save the generated experiences to'
                             ' disk or load them all in memory.')
    parser.add_argument('--path', type=str,
                        help='Path used to save the generated stream.')
    parser.add_argument('--max-epochs', type=int, default=200,
                        help='The maximum number of training epochs for each '
                             'task. Default to 200.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()
    main(args)
