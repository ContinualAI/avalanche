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

from abc import abstractmethod, ABC
from functools import partial
from numbers import Number
from typing import Tuple, TypeVar, Union, Generic, Any, Callable, Optional

import torch
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from torch import Tensor

from .evaluation_data import EvalData, EvalTestData, OnTrainIteration, \
    OnTrainEpochStart, OnTestIteration, OnTrainEpochEnd, OnTrainStepStart, \
    OnTrainStepEnd, OnTestStepStart, OnTestStepEnd, OnTrainPhaseStart, \
    OnTestPhaseStart, OnTrainPhaseEnd, OnTestPhaseEnd
from .raw_accumulators import AverageAccumulator, SumAccumulator, \
    TensorAccumulator

TMetricType = TypeVar('TMetricType')
TNumericalType = TypeVar('TNumericalType', bound=Number)


class MetricUnit(Generic[TMetricType], ABC):
    """
    The base class of all Metric Units.

    A metric unit is a simple element that analyzes incoming training and test
    events to compute some value. This value, obtained using the value field,
    can be used to create a valid Metric.

    When creating a new metric, it is a good practice to begin by creating the
    corresponding units. The unit will serve as the actual metric computation
    utility while the Metric class should be used to handle the packaging of the
    metric value(s), which usually involves:

    - Aggregating values of different units
    - Setting the metric name
    - Define alternative representations (image, readable stdout string, etc.)

    The usual procedure involves the accumulation and aggregation of some values
    at fixed checkpoints during the train and test phases. For instance this
    is needed when computing metrics like the epoch accuracy, average loss,
    confusion matrix, etc., which requires accumulating values across epoch
    iterations. A specific child class is available to handle this specific kind
    of  "accumulate and aggregate" metric units:
    :class:`AbstractMetricAggregator`, whose use is strongly recommended when
    possible.

    Beware that this class should be directly subclasses only if the
    :class:`AbstractMetricAggregator` (and its subclasses) can't handle the
    specific type of unit you're trying to implement.
    """

    def __init__(self, allowed_events: Tuple[type, ...] = None):
        """
        Creates a Metric Unit.

        This constructor allows the definition of a list of allowed events.
        Events are checked against this list before calling the abstract
        "_on_update" method and non-matching events will be silently discarded.

        :param allowed_events: A tuple describing the allowed events.
            Defaults to None, which means that all events will be dispatched
            to the "_on_update".
        """
        self._last_train_step: Optional[int] = None
        """
        This field can be used to obtain the last training step ID.
        """

        self._last_test_step: Optional[int] = None
        """
        This field can be used to obtain the last test step ID.
        """

        self._is_training: bool = True
        """
        This field can be used to check if the strategy is currently in
        the training phase.
        """

        self._allowed_events: Optional[Tuple[type, ...]] = allowed_events
        """
        A tuple describing the allowed events.
        """

    def __call__(self, eval_data: EvalData) -> None:
        self._is_training = not isinstance(eval_data, EvalTestData)

        if isinstance(eval_data, EvalTestData):
            # Test phase
            self._last_test_step = eval_data.test_step_id
        else:
            # Training phase
            self._last_train_step = eval_data.training_step_id

        if self._check_event_allowed(eval_data):
            self._on_update(eval_data)

    @property
    @abstractmethod
    def value(self) -> TMetricType:
        """
        The value of this unit, as a property.

        The value and type of this property depends on the specific unit.

        :return: The value computed from this unit.
        """
        pass

    @abstractmethod
    def _on_update(self, eval_data: EvalData):
        """
        The callback method used to receive training and test events.

        Subclasses must implement this method.
        """
        pass

    def _check_event_allowed(self, eval_data: EvalData, fail=False) -> bool:
        if eval_data is None:
            raise ValueError('Evaluation data can\'t be None')
        return self._check_event_type_allowed(type(eval_data), fail=fail)

    def _check_event_type_allowed(self, event_type: type, fail=False) -> bool:
        if not issubclass(event_type, EvalData):
            raise ValueError('Evaluation data of type ' + str(event_type) +
                             'is not supported. Must be a EvalData instance.')

        if self._allowed_events is not None:
            if not issubclass(event_type, self._allowed_events):
                if fail:
                    raise ValueError('Evaluation data of type ' +
                                     str(event_type) +
                                     ' is not supported by this metric unit.')
                else:
                    return False

        return True


def _identity(val):
    """
    For internal use only.
    """
    return val


class AbstractMetricAggregator(Generic[TMetricType],
                               MetricUnit[TMetricType], ABC):
    """
    Base class for most standard Metric Units.

    This base class defines methods that can be overridden to accumulate
    and aggregate values across (from coarse to fine-grained) "phases",
    "steps", "epochs", "iterations".

    Implementing all the "_consolidate_*" and "_accumulate_*" methods is not
    mandatory as they default to np-op.

    Also, this class serves as a filter to automatically discard or allow train
    or test related events. This feature can be used to create metric units that
    selectively operate on the train phase, the test phase or both.

    When creating a metric unit that has to accumulate values across iterations
    of a single epoch, consider using the far simpler to use
    :class:`IterationsAggregator` or even :class:`EpochAverage` adapter
    subclasses instead.

    Subclasses must implement the proper paired "_consolidate_*" and
    "_accumulate_*" methods. Subclasses must also implement the "value"
    property to define the unit value.
    """
    def __init__(self,
                 on_train: bool = True,
                 on_test: bool = True,
                 initial_accumulator_value: Any = None,
                 initial_accumulator_creator: Callable[[], Any] = None):
        """
        Creates an instance of the metric aggregator.

        :param on_train: If True, training related events will be considered.
            Defaults to True.
        :param on_test: If True, test related events will be considered.
            Defaults to True.
        :param initial_accumulator_value: The initial value of accumulators.
            Defaults to None. Not compatible with "initial_accumulator_creator".
        :param initial_accumulator_creator: The factory for the initial value
            of accumulators. Defaults to None, which means that the value
            of the "initial_accumulator_value" parameter will be used.
        """

        if (not on_train) and (not on_test):
            raise ValueError('on_train and on_test can\'t be both False at the'
                             'same time.')

        super().__init__()

        self._on_train = on_train
        """
        If True, training events will be considered.
        """

        self._on_test = on_test
        """
        If True, test events will be considered.
        """

        if initial_accumulator_value is not None and \
                initial_accumulator_creator is not None:
            raise ValueError(
                'initial_accumulator_value and initial_accumulator_creator'
                ' can\'t be set at the same time.')

        callable_initializer = initial_accumulator_creator
        if initial_accumulator_value is not None:
            callable_initializer = partial(_identity, initial_accumulator_value)

        self._initial_value_factory: Callable[[], Any] = callable_initializer
        """
        Defines the factory for the initial value for the accumulator variables.
        """

        self._train_iterations_accumulator = self._initial_value_factory()
        """
        Value accumulated across all iterations (of the current train epoch)
        """

        self._train_epochs_accumulator = self._initial_value_factory()
        """
        Value accumulated across all epochs (of the current train step)
        """

        self._train_steps_accumulator = self._initial_value_factory()
        """
        Value accumulated across all train steps (of a single training phase).
        """

        self._train_phases_accumulator = self._initial_value_factory()
        """
        Value accumulated across all training phases.
        """

        self._test_iterations_accumulator = self._initial_value_factory()
        """
        Value accumulated across all iterations (of a single test epoch).
        """

        self._test_steps_accumulator = self._initial_value_factory()
        """
        Value accumulated across all test steps (of a single test phase).
        """

        self._test_phases_accumulator = self._initial_value_factory()
        """
        Value accumulated across all test phases.
        """

    @property
    def train_iterations_value(self) -> Any:
        return self._consolidate_iterations_value(
            self._train_iterations_accumulator, True)

    @property
    def test_iterations_value(self) -> Any:
        return self._consolidate_iterations_value(
            self._test_iterations_accumulator, False)

    @property
    def train_epochs_value(self) -> Any:
        return self._consolidate_epochs_value(
            self._train_epochs_accumulator)

    @property
    def train_steps_value(self) -> Any:
        return self._consolidate_steps_value(
            self._train_steps_accumulator, True)

    @property
    def test_steps_value(self) -> Any:
        return self._consolidate_steps_value(
            self._test_steps_accumulator, False)

    @property
    def train_phases_value(self) -> Any:
        return self._consolidate_steps_value(
            self._train_phases_accumulator, True)

    @property
    def test_phases_value(self) -> Any:
        return self._consolidate_steps_value(
            self._test_phases_accumulator, False)

    def _consolidate_iterations_value(
            self, accumulated, is_train_phase: bool) -> Any:
        return None

    def _consolidate_epochs_value(self, accumulated) -> Any:
        return None

    def _consolidate_steps_value(
            self, accumulated, is_train_phase: bool) -> Any:
        return None

    def _consolidate_phases_value(
            self, accumulated, is_train_phase: bool) -> Any:
        return None

    def _accumulate_iteration(
           self, eval_data: Union[OnTrainIteration, OnTestIteration],
           accumulated, is_train_phase: bool) -> Any:
        return None

    def _accumulate_train_epoch(
           self, eval_data: OnTrainEpochEnd, accumulated) -> Any:
        return None

    def _accumulate_step(
           self, eval_data: Union[OnTrainStepEnd, OnTestStepEnd],
           accumulated, is_train_phase: bool) -> Any:
        return None

    def _accumulate_phase(
           self, eval_data: Union[OnTrainPhaseEnd, OnTestPhaseEnd],
           accumulated, is_train_phase: bool) -> Any:
        return None

    def _on_update(self, eval_data: EvalData):
        # This fairly complex block of ifs does a simple thing:
        # 1) each time a iteration/epoch/step/phase starts, sets the proper
        #   "accumulated" field to its default value.
        # 2) each time a iteration/epoch/step/phase ends, accumulates values by
        #   updating the "accumulated" field.

        is_train = eval_data.train_phase
        if (is_train and not self._on_train) or \
                ((not is_train) and not self._on_test):
            return

        if isinstance(eval_data, (OnTrainIteration, OnTestIteration)):
            # An iteration ended: accumulate the iteration data
            if is_train:
                self._train_iterations_accumulator = self._accumulate_iteration(
                    eval_data, self._train_iterations_accumulator, is_train)
            else:
                self._test_iterations_accumulator = self._accumulate_iteration(
                    eval_data, self._test_iterations_accumulator, is_train)
        elif isinstance(eval_data, (OnTrainEpochStart, OnTestStepStart)):
            # A new epoch is starting: reset the iterations accumulator
            # Only training applies, as testing involves only one epoch per step
            if is_train:
                self._train_iterations_accumulator = \
                    self._initial_value_factory()
            else:
                self._test_iterations_accumulator = \
                    self._initial_value_factory()
        elif isinstance(eval_data, OnTrainEpochEnd):
            # Accumulate epochs data
            # Only training applies, as testing involves only one epoch per step
            self._train_epochs_accumulator = self._accumulate_train_epoch(
                eval_data, self._train_epochs_accumulator)
        elif isinstance(eval_data, OnTrainStepStart):
            # A new step is starting: reset the epochs accumulator
            self._train_epochs_accumulator = self._initial_value_factory()
        elif isinstance(eval_data, (OnTrainStepEnd, OnTestStepEnd)):
            # Accumulate steps data
            if is_train:
                self._train_steps_accumulator = self._accumulate_step(
                    eval_data, self._train_steps_accumulator, is_train)
            else:
                self._test_steps_accumulator = self._accumulate_step(
                    eval_data, self._test_steps_accumulator, is_train)
        elif isinstance(eval_data, (OnTrainPhaseStart, OnTestPhaseStart)):
            # A new phase is starting: reset the steps accumulator
            if is_train:
                self._train_steps_accumulator = self._initial_value_factory()
            else:
                self._test_steps_accumulator = self._initial_value_factory()
        elif isinstance(eval_data, (OnTrainPhaseEnd, OnTestPhaseEnd)):
            # Accumulate phases data
            if is_train:
                self._train_phases_accumulator = self._accumulate_phase(
                    eval_data, self._train_phases_accumulator, is_train)
            else:
                self._test_phases_accumulator = self._accumulate_phase(
                    eval_data, self._test_phases_accumulator, is_train)

    @property
    @abstractmethod
    def value(self) -> TMetricType:
        pass


class IterationsAggregator(Generic[TMetricType],
                           AbstractMetricAggregator[TMetricType], ABC):
    """
    Base class of Metric Units that accumulate values over the iterations of
    an epoch. Most notable child classes are:

    - :class:`EpochAverage`, which serves as a base class for units that average
        over iteration values;
    - :class:`ConfusionMatrixUnit`, which computes the Confusion Matrix of
        a single epoch;

    The accumulation and aggregation procedures are handled by child classes
    using the `_accumulate_iteration` and `_consolidate_iterations_value`
    abstract methods.
    """
    def __init__(self,
                 on_train_epochs: bool = True,
                 on_test_epochs: bool = True,
                 initial_accumulator_value: Any = None,
                 initial_accumulator_creator: Callable[[], Any] = None):
        super().__init__(
            on_train=on_train_epochs,
            on_test=on_test_epochs,
            initial_accumulator_value=initial_accumulator_value,
            initial_accumulator_creator=initial_accumulator_creator)

    @abstractmethod
    def _consolidate_iterations_value(
            self,
            accumulated,
            is_train_phase: bool):
        pass

    @abstractmethod
    def _accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated,
            is_train_phase: bool):
        pass

    @property
    def value(self):
        use_train = (self._is_training and self._on_train) or \
                    (not self._is_training and not self._on_test)
        if use_train:
            return self.train_iterations_value
        else:
            return self.test_iterations_value


class EpochPatternsCounterUnit(IterationsAggregator[int]):
    """
    A simple metric unit that can be used to count the number of seen patterns
    in an epoch.
    """
    def __init__(self,
                 on_train_epochs=True,
                 on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs,
                         initial_accumulator_creator=SumAccumulator)

    def _consolidate_iterations_value(self,
                                      accumulated: SumAccumulator,
                                      is_train_phase: bool):
        return accumulated.value

    def _accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated: SumAccumulator,
            is_train_phase: bool):
        return accumulated(len(eval_data.ground_truth))

    def __int__(self):
        return self.value


class EpochAverage(IterationsAggregator[float], ABC):
    """
    Base class of Metric Units that accumulate values over the iterations of
    an epoch in order to get an average value out of it.

    The accumulation procedure is handled by using a AverageAccumulator.

    The child classes must implement the _accumulate_iteration method to
    update the AverageAccumulator state.
    """
    def __init__(self, on_train_epochs=True, on_test_epochs=True):
        super().__init__(
            on_train_epochs=on_train_epochs,
            on_test_epochs=on_test_epochs,
            initial_accumulator_creator=AverageAccumulator)

    @abstractmethod
    def _accumulate_iteration(
            self, eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated: AverageAccumulator,
            is_train_phase: bool):
        pass

    def _consolidate_iterations_value(
            self, accumulated: AverageAccumulator, is_train_phase: bool):
        return accumulated.value

    def __float__(self):
        return self.value


class AverageLossUnit(EpochAverage):
    """
    A metric unit that can be used to compute the average epoch loss.
    """

    def __init__(self, on_train_epochs=True, on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs)

    def _accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated: AverageAccumulator, is_train_phase: bool) -> \
            Tuple[float, int]:
        # torch.mean manages different loss reduction types (sum, none, mean).
        weight = len(eval_data.ground_truth)
        mean_loss = torch.mean(eval_data.loss)

        return accumulated(mean_loss, weight=weight)


class AverageAccuracyUnit(EpochAverage):
    """
    A metric unit that can be used to compute the average epoch accuracy.
    """

    def __init__(self, on_train_epochs=True, on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs)

    def _accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated: AverageAccumulator, is_train_phase: bool) -> \
            Tuple[float, int]:
        pred_y: Tensor = torch.max(eval_data.prediction_logits, 1)[1]
        true_y: Tensor = eval_data.ground_truth

        return accumulated(torch.eq(pred_y, true_y))


class ConfusionMatrixUnit(IterationsAggregator[ndarray]):
    """
    A metric unit that can be used to compute the epoch confusion matrix.
    """

    def __init__(self, num_classes: int = None, normalize=None,
                 on_train_epochs=True, on_test_epochs=True):
        super().__init__(
            on_train_epochs=on_train_epochs,
            on_test_epochs=on_test_epochs,
            initial_accumulator_creator=lambda: (TensorAccumulator(),
                                                 TensorAccumulator()))
        self.num_classes = num_classes
        self.normalize = normalize

    def _consolidate_iterations_value(
            self,
            accumulated: Tuple[TensorAccumulator, TensorAccumulator],
            is_train_phase: bool):
        true_y = accumulated[0].value
        pred_y = accumulated[1].value

        if self.num_classes is None:
            labels = max(true_y.max(), pred_y.max())
        else:
            labels = list(range(self.num_classes))

        return confusion_matrix(
            true_y, pred_y, labels=labels, normalize=self.normalize)

    def _accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated: Tuple[TensorAccumulator, TensorAccumulator],
            is_train_phase: bool):
        pred_y: Tensor = torch.max(eval_data.prediction_logits, 1)[1]
        true_y: Tensor = eval_data.ground_truth

        accumulated[0](true_y)
        accumulated[1](pred_y)

        return accumulated


__all__ = [
    'MetricUnit',
    'AbstractMetricAggregator',
    'IterationsAggregator',
    'EpochPatternsCounterUnit',
    'EpochAverage',
    'AverageLossUnit',
    'AverageAccuracyUnit',
    'ConfusionMatrixUnit']
