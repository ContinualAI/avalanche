#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import TypeVar
from torch import Tensor

TElement = TypeVar('TElement')


def _detach_tensor(tensor: TElement) -> TElement:
    if isinstance(tensor, Tensor):
        return tensor.detach().cpu()

    return tensor


class EvalData:
    """
    The base evaluation data class.

    This class is the root class for all the strategy-related events that can be
    captured in order to compute relevant metrics.

    This class only defines a step counter, the training step ID and the
    training task label.
    """

    def __init__(self,
                 step_counter: int,
                 training_step_id: int,
                 training_task_label: int):
        self.train_phase: bool = True
        """
        If True, this event refers to the training phase.
        """

        self.step_counter: int = step_counter
        """
        The number of training steps encountered so far.
        """

        self.training_step_id: int = training_step_id
        """
        The training step ID. May be different from the step counter.
        """

        self.training_task_label: int = training_task_label
        """
        The training task label.
        """

    @property
    def test_phase(self):
        """
        If True, this event refers to the test phase.
        """
        return not self.train_phase


class EvalTestData(EvalData):
    """
    The base evaluation data class for test phase related events.

    This class is the root class for all the strategy-related events of the
    test phase.

    This class contains all the fields of the EvalData base class, the
    test step ID and the test task label.
    """

    def __init__(self,
                 step_counter: int,
                 training_step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        super().__init__(step_counter, training_step_id, training_task_label)
        self.train_phase: bool = False
        """
        If True, this event refers to the training phase.
        """

        self.test_step_id: int = test_step_id
        """
        The test step ID.
        """
        self.test_task_label: int = test_task_label
        """
        The test task label.
        """


class OnTrainPhaseStart(EvalData):
    """
    Evaluation data sent to metrics when a training phase is about to start.

    Beware that a training phase may involve running the training procedure
    on multiple training steps.

    The step_counter here refers to the counter as it is before starting the
    training phase.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int):
        super().__init__(step_counter, step_id, training_task_label)


class OnTestPhaseStart(EvalTestData):
    """
    Evaluation data sent to metrics when a test phase is about to start.

    Beware that a test phase usually involves running the test procedure
    on multiple test steps.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)


class OnTrainPhaseEnd(EvalData):
    """
    Evaluation data sent to metrics when a training phase is ended.

    This means that all training steps have completed and the strategy is
    about to switch to the test phase.

    The step_counter here refers to the counter as it is after all training
    steps have completed.
    """

    # TODO: step_id, training_task_label == last step id, last train task label

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int):
        super().__init__(step_counter, step_id, training_task_label)


class OnTestPhaseEnd(EvalTestData):
    """
    Evaluation data sent to metrics when a test phase is ended.

    This means that all test steps have completed and the strategy is
    about to switch to the training phase.
    """
    # TODO: test_step_id, test_task_label == last step id, last test task label

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)


class OnTrainStepStart(EvalData):
    """
    Evaluation data sent to metrics when a training step is about to start.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int):
        super().__init__(step_counter, step_id, training_task_label)


class OnTestStepStart(EvalTestData):
    """
    Evaluation data sent to metrics when a test step is about to start.

    Beware that this type of events also cover the "test epoch start"
    checkpoint, as a test step only involves running a single epoch on the test
    dataset.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)


class OnTrainStepEnd(EvalData):
    """
    Evaluation data sent to metrics when a training step completes.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int):
        super().__init__(step_counter, step_id, training_task_label)


class OnTestStepEnd(EvalTestData):
    """
    Evaluation data sent to metrics when a test step completes.

    Beware that this type of events also cover the "test epoch end"
    checkpoint, as a test step only involves running a single epoch on the test
    dataset.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)


class OnTrainEpochStart(EvalData):
    """
    Evaluation data sent to metrics when a training epoch is about to start.

    Beware that the equivalent "Test" event doesn't exist.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 epoch: int):
        super().__init__(step_counter, step_id, training_task_label)
        self.epoch: int = epoch
        """
        The epoch that is about to start (first epoch = 0).
        """


class OnTrainEpochEnd(EvalData):
    """
    Evaluation data sent to metrics when a training epoch completes.

    Beware that the equivalent "Test" event doesn't exist.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 epoch: int):
        super().__init__(step_counter, step_id, training_task_label)
        self.epoch: int = epoch
        """
        The epoch that just completed (first epoch = 0).
        """


class OnTrainIterationStart(EvalData):
    """
    Evaluation data sent to metrics when a training iteration (on a minibatch)
    is about to start.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 epoch: int,
                 iteration: int,
                 ground_truth: Tensor):
        super().__init__(step_counter, step_id, training_task_label)
        self.epoch: int = epoch
        """
        The current epoch.
        """

        self.iteration: int = iteration
        """
        The iteration that is about to start (first iteration = 0).
        """

        self.ground_truth: Tensor = _detach_tensor(ground_truth)
        """
        A Tensor describing the ground truth for the current minibatch.
        """


class OnTestIterationStart(EvalTestData):
    """
    Evaluation data sent to metrics when a test iteration (on a minibatch)
    is about to start.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int,
                 iteration: int,
                 ground_truth: Tensor):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)
        self.iteration: int = iteration
        """
        The iteration that is about to start (first iteration = 0).
        """

        self.ground_truth: Tensor = _detach_tensor(ground_truth)
        """
        A Tensor describing the ground truth for the current minibatch.
        """


class OnTrainIterationEnd(EvalData):
    """
    Evaluation data sent to metrics when a training iteration (on a minibatch)
    completes.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 epoch: int,
                 iteration: int,
                 ground_truth: Tensor,
                 prediction_logits: Tensor,
                 loss: Tensor):
        super().__init__(step_counter, step_id, training_task_label)
        self.epoch: int = epoch
        """
        The current epoch.
        """

        self.iteration: int = iteration
        """
        The iteration that just completed (first iteration = 0).
        """

        self.ground_truth: Tensor = _detach_tensor(ground_truth)
        """
        A Tensor describing the ground truth for the current minibatch.
        """

        self.prediction_logits: Tensor = _detach_tensor(prediction_logits)
        """
        A Tensor describing the prediction logits for the current minibatch.
        """

        self.loss: Tensor = _detach_tensor(loss)
        """
        A Tensor describing the loss for the current minibatch.
        
        Metrics should be able to handle different reduction types.
        """


class OnTestIterationEnd(EvalTestData):
    """
    Evaluation data sent to metrics when a test iteration (on a minibatch)
    completes.
    """

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int,
                 iteration: int,
                 ground_truth: Tensor,
                 prediction_logits: Tensor,
                 loss: Tensor):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)
        self.iteration: int = iteration
        """
        The iteration that just completed (first iteration = 0).
        """

        self.ground_truth: Tensor = _detach_tensor(ground_truth)
        """
        A Tensor describing the ground truth for the current minibatch.
        """

        self.prediction_logits: Tensor = _detach_tensor(prediction_logits)
        """
        A Tensor describing the prediction logits for the current minibatch.
        """

        self.loss: Tensor = _detach_tensor(loss)
        """
        A Tensor describing the loss for the current minibatch.

        Metrics should be able to handle different reduction types.
        """


__all__ = ['EvalData',
           'EvalTestData',
           'OnTrainPhaseStart',
           'OnTestPhaseStart',
           'OnTrainPhaseEnd',
           'OnTestPhaseEnd',
           'OnTrainStepStart',
           'OnTestStepStart',
           'OnTrainStepEnd',
           'OnTestStepEnd',
           'OnTrainEpochStart',
           'OnTrainEpochEnd',
           'OnTrainIterationStart',
           'OnTestIterationStart',
           'OnTrainIterationEnd',
           'OnTestIterationEnd']
