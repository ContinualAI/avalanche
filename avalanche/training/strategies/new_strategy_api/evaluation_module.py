#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-09-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

import warnings
from typing import Optional

import torch
from torch import Tensor

from avalanche.benchmarks.scenarios import IStepInfo
from avalanche.evaluation.metrics import ACC
from avalanche.evaluation import EvalProtocol
from .cl_strategy import StrategySkeleton
from .strategy_flow import TrainingFlow, TestingFlow


class EvaluationModule(StrategySkeleton):
    """
    An evaluation module that can be plugged in a strategy.

    Instances of this class should be used as strategy submodules.

    This module obtains relevant data from the training and testing loops of the
    main strategy by using the integrated callbacks systems.

    Internally, the evaluation module tries to use the "evaluation_protocol"
    namespace value. If found and not None, the evaluation protocol
    (usually an instance of :class:`EvalProtocol`), is used to compute the
    required metrics. The "evaluation_protocol" is usually a field of the main
    strategy.

    For an example on how to use it, see :class:`DeepLearningStrategy` and
    :class:`Naive`.

    Beware that, while most of the required callbacks are automatically managed
    by the :class:`DeepLearningStrategy` class, some callbacks such as
    "after_training_iteration" and "after_test_iteration" must be called
    by the implementing strategy subclasses. For an example of a vanilla
    training/testing epoch, see :class:`Naive`.
    """
    def __init__(self):
        super().__init__()

        # Training
        self._training_dataset_size = None
        self._training_accuracy = None
        self._training_correct_count = 0
        self._training_average_loss = 0
        self._training_total_iterations = 0
        self._train_current_task_id = None
        self._training_ok = False

        # Testing
        self._test_dataset_size = None
        # self._test_accuracy = None
        # self._test_correct_count = 0
        self._test_average_loss = 0
        self._test_total_iterations = 0
        self._test_current_task_id = None
        self._test_true_y = None
        self._test_predicted_y = None
        self._test_protocol_results = None
        self._test_ok = False

    def get_train_result(self):
        return self._training_average_loss, self._training_accuracy

    def get_test_result(self):
        return self._test_protocol_results

    @TrainingFlow
    def before_training(self, step_info: IStepInfo = None):
        if step_info is None:
            self._training_ok = False
            self.__missing('step_info')
            return

        _, task_id = step_info.current_training_set()
        self._training_dataset_size = None
        self._train_current_task_id = task_id
        self._training_accuracy = None
        self._training_correct_count = 0
        self._training_average_loss = 0
        self._training_ok = True

    @TrainingFlow
    def make_train_dataloader(self, train_dataset=None):
        if not self._training_ok:
            return

        if train_dataset is None:
            self._training_ok = False
            self.__missing('train_dataset')
            return
        self._training_dataset_size = len(train_dataset)

    @TrainingFlow
    def after_training_iteration(
            self, evaluation_protocol: Optional[EvalProtocol] = None,
            epoch: int = None, iteration: int = None, train_mb_y: Tensor = None,
            logits: Tensor = None, loss: Tensor = None, **kwargs):
        if not self._training_ok:
            return

        self._training_total_iterations += 1

        if evaluation_protocol is None:
            self._training_ok = False
            self.__no_evaluation_protocol()
            return

        if iteration is None or train_mb_y is None or logits is None \
                or loss is None:
            self._training_ok = False
            self.__missing('iteration', 'train_mb_y', 'logits', 'loss')
            return

        _, predicted_labels = torch.max(logits, 1)
        correct_predictions = torch.eq(predicted_labels,
                                       train_mb_y).sum().item()
        self._training_correct_count += correct_predictions

        torch.eq(predicted_labels, train_mb_y)

        self._training_average_loss += loss.item()
        den = ((iteration + 1) * train_mb_y.shape[0] +
               epoch * self._training_dataset_size)
        self._training_average_loss /= den

        self._training_accuracy = self._training_correct_count / den

        if iteration % 100 == 0:
            print(
                '[Evaluation] ==>>> it: {}, avg. loss: {:.6f}, '
                'running train acc: {:.3f}'.format(
                    iteration, self._training_average_loss,
                    self._training_accuracy))

            evaluation_protocol.update_tb_train(
                self._training_average_loss, self._training_accuracy,
                self._training_total_iterations, torch.unique(train_mb_y),
                self._train_current_task_id)

    @TestingFlow
    def before_testing(self):
        self._test_protocol_results = dict()

    @TestingFlow
    def before_step_testing(self, step_info: IStepInfo = None,
                            step_id: int = None):
        if step_info is None or step_id is None:
            self.__missing('step_info', 'step_id')
            return

        _, task_id = step_info.step_specific_test_set(step_id)
        self._test_dataset_size = None
        self._test_current_task_id = task_id
        # self._test_accuracy = None
        # self._test_correct_count = 0
        self._test_average_loss = 0
        self._test_true_y = []
        self._test_predicted_y = []
        self._test_ok = True

    @TestingFlow
    def make_test_dataloader(self, test_dataset=None):
        if not self._test_ok:
            return

        if test_dataset is None:
            self._test_ok = False
            self.__missing('test_dataset')
            return
        self._test_dataset_size = len(test_dataset)

    @TestingFlow
    def after_test_iteration(
            self, evaluation_protocol: Optional[EvalProtocol] = None,
            iteration: int = None, test_mb_y: Tensor = None,
            test_logits: Tensor = None, test_loss: Tensor = None):

        if not self._test_ok:
            return

        self._test_total_iterations += 1

        if evaluation_protocol is None:
            self._test_ok = False
            self.__no_evaluation_protocol()
            return

        if iteration is None or test_mb_y is None or test_logits is None or \
                test_loss is None:
            self._test_ok = False
            self.__missing('iteration', 'test_mb_y', 'test_logits', 'test_loss')
            return

        _, predicted_labels = torch.max(test_logits, 1)
        self._test_true_y.append(test_mb_y.numpy())
        self._test_predicted_y.append(predicted_labels.numpy())

        self._test_average_loss += test_loss.item()

    @TestingFlow
    def after_step_testing(self,
                           evaluation_protocol: Optional[EvalProtocol] = None):
        if not self._test_ok:
            return

        self._test_average_loss /= self._test_dataset_size

        if evaluation_protocol is None:
            self._test_ok = False
            self.__no_evaluation_protocol()
            return

        results = evaluation_protocol.get_results(
            self._test_true_y, self._test_predicted_y,
            self._train_current_task_id, self._test_current_task_id)
        acc, accs = results[ACC]

        print("[Evaluation] Task {0}: Avg Loss {1}; Avg Acc {2}"
              .format(self._test_current_task_id, self._test_average_loss, acc))

        self._test_protocol_results[self._test_current_task_id] = \
            (self._test_average_loss, acc, accs, results)

    @TestingFlow
    def after_testing(self, evaluation_protocol: Optional[EvalProtocol] = None,
                      step_info: IStepInfo = None):
        if not self._test_ok:
            return

        if evaluation_protocol is None:
            self._test_ok = False
            self.__no_evaluation_protocol()
            return

        if step_info is None:
            self._test_ok = False
            self.__missing('step_info')
            return

        evaluation_protocol.update_tb_test(self._test_protocol_results,
                                           step_info.current_step)

    @staticmethod
    def __missing(*elements):
        str_warning = ''
        for element in elements:
            str_warning += element
            str_warning += ', '
        str_warning = str_warning[:-2]
        str_warning += ' must be in the global namespace for the evaluation ' \
                       'module to work'
        warnings.warn(str_warning)

    @staticmethod
    def __no_evaluation_protocol():
        warnings.warn('No evaluation protocol has been set')


__all__ = ['EvaluationModule']
