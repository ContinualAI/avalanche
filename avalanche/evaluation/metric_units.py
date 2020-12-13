from abc import abstractmethod, ABC
from numbers import Number
from typing_extensions import Literal
from typing import Tuple, TypeVar, Union, Sequence, Generic, Any

import torch
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from torch import Tensor

from .evaluation_data import EvalData, EvalTestData, OnTrainIteration, \
    OnTrainEpochStart, OnTestIteration, OnTrainEpochEnd, OnTrainStepStart, \
    OnTrainStepEnd, OnTestStepStart, OnTestStepEnd, OnTrainPhaseStart, \
    OnTestPhaseStart, OnTrainPhaseEnd, OnTestPhaseEnd

TMetricType = TypeVar('TMetricType')
TNumericalType = TypeVar('TNumericalType', bound=Number)


class MetricUnit(Generic[TMetricType], ABC):

    def __init__(self, allowed_events: Tuple[type, ...] = None):
        self._last_train_step = None
        self._last_test_step = None
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
                 aggregate_on: Literal['phase', 'step', 'epoch', 'iteration'],
                 on_train: bool = True,
                 on_test: bool = True,
                 initial_accumulator_value: Any = None):

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

        self._initial_value = initial_accumulator_value
        """
        Defines the initial value for the accumulator variables.
        """

        self._value_property = aggregate_on
        """
        Defines the value returned from the "value" property.

        Valid values: 'phase', 'step', 'epoch', 'iteration'.
        """

        self._train_iterations_accumulator = self._initial_value
        """
        Value accumulated across all iterations (of the current train epoch)
        """

        self._train_epochs_accumulator = self._initial_value
        """
        Value accumulated across all epochs (of the current train step)
        """

        self._train_steps_accumulator = self._initial_value
        """
        Value accumulated across all train steps (of a single training phase).
        """

        self._train_phases_accumulator = self._initial_value
        """
        Value accumulated across all training phases.
        """

        self._test_iterations_accumulator = self._initial_value
        """
        Value accumulated across all iterations (of a single test epoch).
        """

        self._test_steps_accumulator = self._initial_value
        """
        Value accumulated across all test steps (of a single test phase).
        """

        self._test_phases_accumulator = self._initial_value
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

    def on_update(self, eval_data: EvalData):
        # TODO: doc
        # This fairly complex block of ifs does a simple thing:
        # 1) each time a iteration/epoch/step starts, sets the proper
        #   "accumulated" field to its default value.
        # 2) each time a iteration/epoch/step ends, accumulates values by
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
                self._train_iterations_accumulator = self._initial_value
            else:
                self._test_iterations_accumulator = self._initial_value
        elif isinstance(eval_data, OnTrainEpochEnd):
            # Accumulate epochs data
            # Only training applies, as testing involves only one epoch per step
            self._train_epochs_accumulator = self._accumulate_train_epoch(
                eval_data, self._train_epochs_accumulator)
        elif isinstance(eval_data, OnTrainStepStart):
            # A new step is starting: reset the epochs accumulator
            self._train_epochs_accumulator = self._initial_value
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
                self._train_steps_accumulator = self._initial_value
            else:
                self._test_steps_accumulator = self._initial_value
        elif isinstance(eval_data, (OnTrainPhaseEnd, OnTestPhaseEnd)):
            # Accumulate phases data
            if is_train:
                self._train_phases_accumulator = self._accumulate_phase(
                    eval_data, self._train_phases_accumulator, is_train)
            else:
                self._test_phases_accumulator = self._accumulate_phase(
                    eval_data, self._test_phases_accumulator, is_train)

    @property
    def value(self) -> TMetricType:

        # During training phase and self._on_train -> aggregated train data
        # During training phase and not self._on_train -> aggregated test data
        # During test phase and self._on_test -> aggregated test data
        # During test phase and not self._on_test -> aggregated train data

        use_train = (self._is_training and self._on_train) or \
                    (not self._is_training and not self._on_test)
        if self._value_property == 'phase':
            if use_train:
                return self.train_phases_value
            else:
                return self.test_phases_value
        elif self._value_property == 'step':
            if use_train:
                return self.train_steps_value
            else:
                return self.test_steps_value
        elif self._value_property == 'epoch':
            if use_train:
                return self.train_epochs_value
            # There is no test_epochs_value

        elif self._value_property == 'iteration':
            if use_train:
                return self.train_iterations_value
            else:
                return self.test_iterations_value
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
            'iteration',
            on_train=on_train_epochs,
            on_test=on_test_epochs,
            initial_accumulator_value=initial_accumulator_value)

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


class EpochPatternsCounterUnit(IterationsAggregator[int]):
    def __init__(self,
                 on_train_epochs=True,
                 on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs,
                         initial_accumulator_value=0)

    def _consolidate_iterations_value(self, accumulated, is_train_phase):
        return accumulated

    def _accumulate_iteration(
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

    def _consolidate_iterations_value(self, accumulated, is_train_phase):
        elements_sum: float = accumulated[0]
        elements_count: int = accumulated[1]
        if elements_count == 0:
            return 0.0
        return elements_sum / elements_count

    def _accumulate_iteration(
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


class AverageLossUnit(EpochAverage):

    def __init__(self, on_train_epochs=True, on_test_epochs=True):
        super().__init__(on_train_epochs=on_train_epochs,
                         on_test_epochs=on_test_epochs)

    def partial_sum_and_count(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            is_train) -> Tuple[float, int]:
        return torch.mean(eval_data.loss).item(), len(eval_data.ground_truth)


class AverageAccuracyUnit(EpochAverage):
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

    def _consolidate_iterations_value(
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

    def _accumulate_iteration(
            self,
            eval_data: Union[OnTrainIteration, OnTestIteration],
            accumulated,
            is_train_phase):
        if accumulated is None:
            accumulated = [[], []]

        pred_y: Tensor = torch.max(eval_data.prediction_logits, 1)[1]
        true_y: Tensor = eval_data.ground_truth

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
