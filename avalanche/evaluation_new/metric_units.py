from abc import abstractmethod, ABC
from numbers import Number
from typing import Tuple, TypeVar, \
    Union, Sequence, Generic, Any, Literal

import torch
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from torch import Tensor

from avalanche.evaluation.evaluation_data import EvalData, \
    EvalTestData, OnTrainIteration, OnTrainEpochStart, OnTestIteration, \
    OnTrainEpochEnd, OnTestEpochEnd, OnTestEpochStart, OnTrainStart, \
    OnTrainEnd, OnTestStart, OnTestEnd

TMetricType = TypeVar('TMetricType')
TNumericalType = TypeVar('TNumericalType', bound=Number)


class MetricUnit(Generic[TMetricType], ABC):

    def __init__(self, allowed_events: Tuple[type, ...] = None):
        self._last_train_step = None
        self._last_test_step = None
        self._last_train_epoch = -1
        self._is_training = True
        self._allowed_events = allowed_events

    def __call__(self, eval_data: EvalData) -> None:
        self._is_training = not isinstance(eval_data, EvalTestData)

        if isinstance(eval_data, EvalTestData):
            # Test phase
            self._last_test_step = eval_data.test_step_id
        else:
            # Training phase
            self._last_train_step = eval_data.training_step_id
            self._last_train_epoch = -1

            if isinstance(eval_data, (OnTrainIteration, OnTrainEpochStart)):
                self._last_train_epoch = -1

        if self._check_event_allowed(eval_data):
            self.on_update(eval_data)

    @property
    @abstractmethod
    def value(self) -> TMetricType:
        pass

    @abstractmethod
    def on_update(self, eval_data: EvalData):
        pass

    def _check_event_allowed(self, eval_data: EvalData, fail=False) -> bool:
        if eval_data is None:
            raise ValueError('Evaluation data can\'t be None')
        return self._check_event_type_allowed(type(eval_data), fail=fail)

    def _check_event_type_allowed(self, event_type: type, fail=False) -> bool:
        if not issubclass(event_type, EvalData):
            ValueError('Evaluation data of type ' + str(event_type) +
                       'is not supported. Must be a EvalData instance.')

        if self._allowed_events is not None:
            if not issubclass(event_type, self._allowed_events):
                if fail:
                    ValueError('Evaluation data of type ' + str(event_type) +
                               ' is not supported by this metric unit.')
                else:
                    return False
        return True


class AbstractMetricAggregator(Generic[TMetricType],
                               MetricUnit[TMetricType], ABC):

    def __init__(self,
                 iteration_aggregator: bool = False,
                 epoch_aggregator: bool = False,
                 step_aggregator: bool = True,
                 on_train: bool = True,
                 on_test: bool = True,
                 initial_accumulator_value: Any = None,
                 aggregator_value_property:
                 Literal['step', 'epoch', 'iteration'] = None):

        if (not on_train) and (not on_test):
            raise ValueError('on_train and on_test can\'t be both False at the'
                             'same time.')

        if (not iteration_aggregator) and (not epoch_aggregator) and \
                (not step_aggregator):
            raise ValueError('iteration_aggregator, epoch_aggregator and '
                             'step_aggregator can\'t be both False at the'
                             'same time.')

        supported_event_types = []
        if iteration_aggregator:
            if on_train:
                supported_event_types.append(OnTrainIteration)
            if on_test:
                supported_event_types.append(OnTestIteration)
        if epoch_aggregator:
            if on_train:
                supported_event_types.append(OnTrainEpochStart)
                supported_event_types.append(OnTrainEpochEnd)
            if on_test:
                supported_event_types.append(OnTestEpochStart)
                supported_event_types.append(OnTestEpochEnd)
        if step_aggregator:
            if on_train:
                supported_event_types.append(OnTrainStart)
                supported_event_types.append(OnTrainEnd)
            if on_test:
                supported_event_types.append(OnTestStart)
                supported_event_types.append(OnTestEnd)

        super().__init__(allowed_events=tuple(supported_event_types))

        self._iteration_aggregator = iteration_aggregator
        self._epoch_aggregator = epoch_aggregator
        self._step_aggregator = step_aggregator
        self._on_train = on_train
        self._on_test = on_test
        self._initial_value = initial_accumulator_value

        if aggregator_value_property is None:
            if self._step_aggregator:
                aggregator_value_property = 'step'
            elif self._epoch_aggregator:
                aggregator_value_property = 'epoch'
            elif self._iteration_aggregator:
                aggregator_value_property = 'iteration'
        self._value_property = aggregator_value_property
        """
        Defines the value returned from the "value" property.

        Valid values: 'step', 'epoch', 'iteration'
        """

        if (self._value_property == 'step' and not step_aggregator) or \
                (self._value_property == 'epoch' and not epoch_aggregator) or \
                (self._value_property == 'iteration' and
                 not iteration_aggregator):
            raise ValueError('Incompatible aggregator_value and *_aggregator'
                             'parameters')

        self.train_iterations_accumulated = self._initial_value
        """
        Value accumulated across all iterations (of the current train epoch)
        """

        self.train_epochs_accumulated = self._initial_value
        """
        Value accumulated across all epochs (of the current train step)
        """

        self.train_steps_accumulated = self._initial_value
        """
        Value accumulated across all steps (train phase)
        """

        self.test_iterations_accumulated = self._initial_value
        """
        Value accumulated across all iterations (of the current test epoch)
        """

        self.test_epochs_accumulated = self._initial_value
        """
        Value accumulated across all epochs (of the current test step)
        
        Usually unused, as each test step usually involves running one epoch.
        """

        self.test_steps_accumulated = self._initial_value
        """
        Value accumulated across all steps (test phase)
        """

        self.train_iterations_aggregated = None
        self.train_epochs_aggregated = None
        self.train_steps_aggregated = None
        self.test_iterations_aggregated = None
        self.test_epochs_aggregated = None
        self.test_steps_aggregated = None

    def consolidate_iterations_value(
            self, accumulated, is_train_phase: bool) -> Any:
        return None

    def consolidate_epochs_value(
            self, accumulated, is_train_phase: bool) -> Any:
        return None

    def consolidate_steps_value(
            self, accumulated, is_train_phase: bool) -> Any:
        return None

    def accumulate_iteration(
            self, eval_data: EvalData,
            accumulated, is_train_phase: bool) -> Any:
        return None

    def accumulate_epoch(
            self, eval_data: EvalData,
            accumulated, is_train_phase: bool) -> Any:
        return None

    def accumulate_step(
            self, eval_data: EvalData,
            accumulated, is_train_phase: bool) -> Any:
        return None

    def on_update(self, eval_data: EvalData):
        # This fairly complex block of ifs does a simple thing:
        # 1) each time a iteration/epoch/step starts sets the proper
        #   "accumulated" field to its default value.
        # 2) each time a iteration/epoch/step ends
        #   a. consolidates and stores the final value in a "aggregated" field;
        #   b. accumulates values, thus updating the "accumulated" field.

        is_train = eval_data.train_phase
        if (is_train and not self._on_train) or \
                ((not is_train) and not self._on_test):
            return

        is_iteration_end = isinstance(
            eval_data, (OnTrainIteration, OnTestIteration))
        is_epoch_start = isinstance(
            eval_data, (OnTrainEpochStart, OnTestEpochStart))
        is_epoch_end = isinstance(
            eval_data, (OnTrainEpochEnd, OnTestEpochEnd))
        is_step_start = isinstance(
            eval_data, (OnTrainStart, OnTestStart))
        is_step_end = isinstance(
            eval_data, (OnTrainEnd, OnTestEnd))

        if is_iteration_end and self._iteration_aggregator:
            # An iteration ended: accumulate the iteration data
            if is_train:
                self.train_iterations_accumulated = self.accumulate_iteration(
                    eval_data, self.train_iterations_accumulated, is_train)
            else:
                self.test_iterations_accumulated = self.accumulate_iteration(
                    eval_data, self.test_iterations_accumulated, is_train)
        elif is_epoch_start and self._iteration_aggregator:
            # A new epoch is starting: reset the iterations accumulator
            if is_train:
                self.train_iterations_accumulated = self._initial_value
                self.train_iterations_aggregated = None
            else:
                self.test_iterations_accumulated = self._initial_value
                self.test_iterations_aggregated = None
        elif is_epoch_end:
            # An epoch ended ...
            # First: aggregate iterations data
            if self._iteration_aggregator:

                if is_train:
                    self.train_iterations_aggregated = \
                        self.consolidate_iterations_value(
                            self.train_iterations_accumulated, is_train)
                else:
                    self.test_iterations_aggregated = \
                        self.consolidate_iterations_value(
                            self.test_iterations_accumulated, is_train)

            # Second: accumulate epochs data
            if self._epoch_aggregator:
                if is_train:
                    self.train_epochs_accumulated = self.accumulate_epoch(
                        eval_data, self.train_epochs_accumulated, is_train)
                else:
                    self.test_epochs_accumulated = self.accumulate_epoch(
                        eval_data, self.test_epochs_accumulated, is_train)
        elif is_step_start and self._epoch_aggregator:
            # A new step is starting: reset the epochs accumulator
            if is_train:
                self.train_epochs_accumulated = self._initial_value
                self.train_epochs_aggregated = None
            else:
                self.test_epochs_accumulated = self._initial_value
                self.test_epochs_aggregated = None
        elif is_step_end:
            # A step ended ...
            # First: aggregate epochs data
            if self._epoch_aggregator:
                if is_train:
                    self.train_epochs_aggregated = \
                        self.consolidate_epochs_value(
                            self.train_epochs_accumulated, is_train)
                else:
                    self.test_epochs_aggregated = \
                        self.consolidate_epochs_value(
                            self.test_epochs_accumulated, is_train)
            # Second: accumulate steps data
            if self._step_aggregator:
                if is_train:
                    self.train_steps_accumulated = self.accumulate_step(
                        eval_data, self.train_steps_accumulated, is_train)
                else:
                    self.test_steps_accumulated = self.accumulate_step(
                        eval_data, self.test_steps_accumulated, is_train)

    @property
    def value(self) -> TMetricType:
        if self._value_property == 'step':
            if self._is_training:
                if self._on_train:
                    return self.train_steps_aggregated
                else:  # self._on_test must be True when _on_train is False
                    # This aggregator doesn't capture data from the training
                    # phase, which means that the last test aggregated data
                    # is returned.
                    return self.test_steps_aggregated
            else:
                if self._on_test:
                    return self.test_steps_aggregated
                else:
                    return self.train_steps_aggregated
        elif self._value_property == 'epoch':
            if self._is_training:
                if self._on_train:
                    return self.train_epochs_aggregated
                else:
                    return self.test_epochs_aggregated
            else:
                if self._on_test:
                    return self.test_epochs_aggregated
                else:
                    return self.train_epochs_aggregated

        elif self._value_property == 'iteration':
            if self._is_training:
                if self._on_train:
                    return self.train_iterations_aggregated
                else:
                    return self.test_iterations_aggregated
            else:
                if self._on_test:
                    return self.test_iterations_aggregated
                else:
                    return self.train_iterations_aggregated
        else:
            # This shouldn't happen as we already check that _value_property
            # has a valid value in the class constructor.
            raise AssertionError('Invalid value of _value_property field.'
                                 'This should never happen!')


class IterationsAggregator(Generic[TMetricType],
                           AbstractMetricAggregator[TMetricType], ABC):
    def __init__(self,
                 on_train_epochs: bool = True,
                 on_test_epochs: bool = True,
                 initial_accumulator_value: Any = None):
        super().__init__(
            iteration_aggregator=True, on_train=on_train_epochs,
            on_test=on_test_epochs,
            initial_accumulator_value=initial_accumulator_value)

    @abstractmethod
    def consolidate_iterations_value(
            self,
            accumulated,
            is_train_phase: bool):
        pass

    @abstractmethod
    def accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated,
            is_train_phase: bool):
        pass


class EpochPatternsCounterUnit(IterationsAggregator[int]):
    def __init__(self,
                 on_train_epochs=True,
                 on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs,
                         initial_accumulator_value=0)

    def consolidate_iterations_value(self, accumulated, is_train_phase):
        return accumulated

    def accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated,
            is_train_phase):
        return accumulated + len(eval_data.ground_truth)

    def __int__(self):
        return self.value


class EpochAverage(IterationsAggregator[float], ABC):
    def __init__(self, on_train_epochs=True, on_test_epochs=True):
        super().__init__(
            on_train_epochs=on_train_epochs, on_test_epochs=on_test_epochs,
            initial_accumulator_value=(0.0, 0))

    def consolidate_iterations_value(self, accumulated, is_train_phase):
        elements_sum: float = accumulated[0]
        elements_count: int = accumulated[1]
        if elements_count == 0:
            return 0.0
        return elements_sum / elements_count

    def accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated,
            is_train_phase):
        elements_sum: float = accumulated[0] + torch.mean(eval_data.loss).item()
        pattern_count: int = accumulated[1] + len(eval_data.ground_truth)
        return elements_sum, pattern_count

    @abstractmethod
    def partial_sum_and_count(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            is_train) -> Tuple[Union[float, int], int]:
        pass

    def __float__(self):
        return self.value


class AverageLossUnit(EpochAverage[float]):

    def __init__(self, on_train_epochs=True, on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs)

    def partial_sum_and_count(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            is_train) -> Tuple[float, int]:
        return torch.mean(eval_data.loss).item(), len(eval_data.ground_truth)


class AverageAccuracyUnit(EpochAverage[float]):
    def __init__(self, on_train_epochs=True, on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs)

    def partial_sum_and_count(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            is_train) -> Tuple[int, int]:
        pred_y, _ = torch.max(eval_data.prediction_logits, 1)
        true_y: Tensor = eval_data.ground_truth
        correct_patterns = int(torch.sum(torch.eq(pred_y, true_y)))
        return correct_patterns, len(true_y)


class ConfusionMatrixUnit(IterationsAggregator[ndarray]):
    def __init__(self, num_classes: int = None, normalize=None,
                 on_train_epochs=True, on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs,
                         initial_accumulator_value=None)
        self.num_classes = num_classes
        self.normalize = normalize

    def consolidate_iterations_value(
            self, accumulated: Sequence[Tensor], is_train_phase: bool):
        if accumulated is None:
            accumulated = [[], []]

        true_y = torch.cat(accumulated[0]).numpy()
        pred_y = torch.cat(accumulated[1]).numpy()

        if self.num_classes is None:
            labels = max(true_y.max(), pred_y.max())
        else:
            labels = list(range(self.num_classes))

        return confusion_matrix(
            true_y, pred_y, labels=labels, normalize=self.normalize)

    def accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated,
            is_train_phase):
        if accumulated is None:
            accumulated = [[], []]

        pred_y = torch.max(eval_data.prediction_logits, 1)
        true_y = eval_data.ground_truth

        accumulated[0].append(true_y)
        accumulated[1].append(pred_y)

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
