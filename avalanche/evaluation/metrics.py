import time
from typing import Callable, Union, Dict

import torch
from PIL.Image import Image
from numpy import ndarray

from .abstract_metric import AbstractMetric
from .evaluation_data import OnTrainEpochEnd, OnTestStepEnd,\
    OnTrainStepEnd, OnTrainEpochStart, OnTrainStepStart, OnTestStepStart
from .metric_definitions import MetricValue, PlotPosition, AlternativeValues
from .metric_units import AverageAccuracyUnit, ConfusionMatrixUnit, \
    AverageLossUnit
from .metric_utils import default_cm_image_creator, filter_accepted_events, \
    get_task_label


class Accuracy(AbstractMetric):
    """
    The average accuracy metric.

    This metric is computed separately for each task.

    The accuracy will be emitted after each epoch by aggregating minibatch
    values. Beware that the training accuracy is the "running" one.
    """
    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the Accuracy metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        # Create accuracy unit
        self.accuracy_unit = AverageAccuracyUnit(on_train_epochs=train,
                                                 on_test_epochs=test)

        # When computing the accuracy metric we need to get EpochEnd events
        # to check if the epoch ended. The actual element in charge of
        # accumulating the running accuracy is the accuracy_unit.
        on_events = filter_accepted_events(
            [OnTrainEpochEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._attach(self.accuracy_unit)\
            ._on(on_events, self.result_emitter)

    def result_emitter(self, eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]):
        # This simply queries accuracy_unit for the accuracy value and
        # emits that value by labeling it with the appropriate name.
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self.accuracy_unit.value

        metric_name = 'Top1_Task{:03}_{}'.format(task_label, phase_name)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(self, metric_name, metric_value,
                           PlotPosition.SPECIFIC, plot_x_position)


class Loss(AbstractMetric):
    """
    The average loss metric.

    This metric is computed separately for each task.

    The loss will be emitted after each epoch by aggregating minibatch
    values. Beware that the training loss is the "running" one.
    """
    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the Loss metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        # Create loss unit
        self.loss_unit = AverageLossUnit(on_train_epochs=train,
                                         on_test_epochs=test)

        # When computing the loss metric we need to get EpochEnd events
        # to check if the epoch ended. The actual element in charge of
        # accumulating the running loss is the loss_unit.
        on_events = filter_accepted_events(
            [OnTrainEpochEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._attach(self.loss_unit)._on(on_events, self.result_emitter)

    def result_emitter(self, eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]):
        # This simply queries loss_unit for the loss value and
        # emits that value by labeling it with the appropriate name.
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self.loss_unit.value

        metric_name = 'Loss_Task{:03}_{}'.format(task_label, phase_name)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(self, metric_name, metric_value,
                           PlotPosition.SPECIFIC, plot_x_position)


class ConfusionMatrix(AbstractMetric):
    """
    The Confusion Matrix metric.

    This matrix logs the Tensor and PIL Image representing the confusion
    matrix after each step.

    This metric is computed separately for each task.

    By default this metric computes the matrix on the test set only but this
    behaviour can be changed by passing train=True in the constructor, in which
    case also the training matrix is logged.
    """
    def __init__(self, *,
                 train: bool = False,
                 test: bool = True,
                 num_classes: int = None,
                 normalize: str = None,
                 save_image: bool = True,
                 image_creator: Callable[[ndarray], Image] =
                 default_cm_image_creator):
        """
        Creates an instance of the Confusion Matrix metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to False.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        :param num_classes: When not None, is used to properly define the
            amount of rows/columns in the confusion matrix. When None, the
            matrix will have many rows/columns as the maximum value of the
            predicted and true pattern labels. Defaults to None.
        :param normalize: Normalizes confusion matrix over the true (rows),
            predicted (columns) conditions or all the population. If None,
            confusion matrix will not be normalized. Valid values are: 'true',
            'pred' and 'all'.
        :param save_image: If True, a graphical representation of the confusion
            matrix will be logged, too. If False, only the Tensor representation
            will be logged. Defaults to True.
        :param image_creator: A callable that, given the tensor representation
            of the confusion matrix, returns a graphical representation of the
            matrix as a PIL Image. Defaults to `default_cm_image_creator`.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        self._save_image = save_image

        if image_creator is None:
            image_creator = default_cm_image_creator
        self._image_creator = image_creator

        # Create CM unit
        self._cm_unit = ConfusionMatrixUnit(
            num_classes=num_classes, normalize=normalize,
            on_train_epochs=train, on_test_epochs=test)

        on_events = filter_accepted_events(
            [OnTrainStepEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._attach(self._cm_unit)._on(on_events, self.result_emitter)

    def result_emitter(self, eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]):
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self._cm_unit.value

        metric_name = 'CM_Task{:03}_{}'.format(task_label, phase_name)
        plot_x_position = self._next_x_position(metric_name)

        metric_representation = MetricValue(
            self, metric_name, torch.as_tensor(metric_value),
            PlotPosition.SPECIFIC, plot_x_position)

        if self._save_image:
            cm_image = self._image_creator(metric_value)
            metric_representation = MetricValue(
                self, metric_name, AlternativeValues(cm_image, metric_value),
                PlotPosition.SPECIFIC, plot_x_position)

        return metric_representation


class CatastrophicForgetting(AbstractMetric):
    """
    The Catastrophic Forgetting metric.

    This metric is computed separately for each task as the difference between
    the accuracy result obtained after training on a task and the accuracy
    result obtained at the end of next steps on the same task.

    This metric is computed in the test phase only.
    """
    def __init__(self):
        """
        Creates an instance of the Catastrophic Forgetting metric.

        """
        super().__init__()

        self.best_accuracy: Dict[int, float] = dict()
        """
        The best accuracy of each task.
        """

        # Create accuracy unit
        self._accuracy_unit = AverageAccuracyUnit(
            on_train_epochs=False, on_test_epochs=True)

        # Attach callbacks
        self._attach(self._accuracy_unit)\
            ._on(OnTestStepEnd, self.result_emitter)

    def result_emitter(self, eval_data: OnTestStepEnd):
        train_task_label = eval_data.training_task_label
        test_task_label = eval_data.test_task_label
        accuracy_value = float(self._accuracy_unit)

        if test_task_label not in self.best_accuracy and \
                train_task_label == test_task_label:
            self.best_accuracy[test_task_label] = accuracy_value

        forgetting = 0.0

        if test_task_label in self.best_accuracy:
            forgetting = self.best_accuracy[test_task_label] - accuracy_value

        metric_name = 'Forgetting_Task{:03}_Test'.format(test_task_label)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(self, metric_name, forgetting,
                           PlotPosition.SPECIFIC, plot_x_position)


class EpochTime(AbstractMetric):
    """
    Time usage metric, measured in seconds.

    The time is measured between the start and end of an epoch.

    Beware that this metric logs a time value for each epoch! For the average
    epoch time use :class:`AverageEpochTime` instead, which logs the average
    the average epoch time for each step.

    By default this metric takes the time on training epochs only but this
    behaviour can be changed by passing test=True in the constructor.
    """

    def __init__(self, *, train=True, test=False):
        """
        Creates an instance of the Epoch Time metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the time will be taken on training epochs.
            Defaults to True.
        :param test: When True, the time will be taken on test epochs.
            Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        self._start_time = None

        on_start_events = filter_accepted_events(
            [OnTrainEpochStart, OnTestStepStart], train=train, test=test)

        on_end_events = filter_accepted_events(
            [OnTrainEpochEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._on(on_start_events, self.time_start)\
            ._on(on_end_events, self.result_emitter)

    def time_start(self, eval_data):
        # Epoch start
        self._start_time = time.perf_counter()

    def result_emitter(self, eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]):
        # Epoch end
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        elapsed_time = time.perf_counter() - self._start_time

        metric_name = 'EpochTime_Task{:03}_{}'.format(task_label, phase_name)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(self, metric_name, elapsed_time,
                           PlotPosition.SPECIFIC, plot_x_position)


class AverageEpochTime(AbstractMetric):
    """
    Time usage metric, measured in seconds.

    The time is measured as the average epoch time of a step.
    The average value is computed and emitted at the end of the train/test step.

    By default this metric takes the time of epochs in training steps only. This
    behaviour can be changed by passing test=True in the constructor.

    Consider that, when used on the test set, the epoch time is the same as the
    step time.
    """

    def __init__(self, *, train=True, test=False):
        """
        Creates an instance of the Average Epoch Time metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the time will be taken on training epochs.
            Defaults to True.
        :param test: When True, the time will be taken on test epochs.
            Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')
        self._epoch_start_time = None
        self._accumulated_time = 0.0
        self._n_epochs = 0

        on_step_start_events = filter_accepted_events(
            [OnTrainStepStart, OnTestStepStart], train=train, test=test)

        on_epoch_start_events = filter_accepted_events(
            [OnTrainEpochStart], train=train, test=test)

        on_epoch_end_events = filter_accepted_events(
            [OnTrainEpochEnd], train=train, test=test)

        on_step_end_events = filter_accepted_events(
            [OnTrainStepEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._on(on_step_start_events, self.step_start) \
            ._on(on_epoch_start_events, self.epoch_start) \
            ._on(on_epoch_end_events, self.epoch_end) \
            ._on(on_step_end_events, self.result_emitter)

    def step_start(self, eval_data):
        # Step start
        self._accumulated_time = 0.0
        self._n_epochs = 0
        # Used for timing during the test phase
        self._epoch_start_time = time.perf_counter()

    def epoch_start(self, eval_data):
        # Epoch start (training phase)
        self._epoch_start_time = time.perf_counter()

    def epoch_end(self, eval_data):
        # Epoch end  (training phase)
        self._accumulated_time = time.perf_counter() - self._epoch_start_time
        self._n_epochs += 1

    def result_emitter(self, eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]):
        # Epoch end
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)

        if self._n_epochs == 0:
            # Test phase
            self._n_epochs = 1
            self._accumulated_time = \
                time.perf_counter() - self._epoch_start_time

        average_epoch_time = self._accumulated_time / self._n_epochs

        metric_name = 'AvgEpochTime_Task{:03}_{}'.format(task_label, phase_name)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(
            self, metric_name, average_epoch_time,
            PlotPosition.SPECIFIC, plot_x_position)


__all__ = [
    'Accuracy',
    'Loss',
    'ConfusionMatrix',
    'CatastrophicForgetting',
    'EpochTime',
    'AverageEpochTime']
