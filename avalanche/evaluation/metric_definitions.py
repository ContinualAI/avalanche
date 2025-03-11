################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta, Andrea Cossu                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from abc import ABC, abstractmethod
from typing import (
    Generic,
    TypeVar,
    Optional,
    TYPE_CHECKING,
    List,
    Union,
    overload,
    Literal,
    Protocol,
)
from .metric_results import MetricValue, MetricType, AlternativeValues
from .metric_utils import (
    get_metric_name,
    generic_get_metric_name,
    default_metric_name_template,
)

if TYPE_CHECKING:
    from .metric_results import MetricResult
    from ..training.templates import SupervisedTemplate

TResult_co = TypeVar("TResult_co", covariant=True)
TMetric = TypeVar("TMetric", bound="Metric")


class Metric(Protocol[TResult_co]):
    """Standalone metric.

    A standalone metric exposes methods to reset its internal state and
    to emit a result. Emitting a result does not automatically cause
    a reset in the internal state.

    The specific metric implementation exposes ways to update the internal
    state. Usually, standalone metrics like :class:`Sum`, :class:`Mean`,
    :class:`Accuracy`, ... expose an `update` method.

    The `Metric` class can be used as a standalone metric by directly calling
    its methods.
    In order to automatically integrate the metric with the training and
    evaluation flows, you can use :class:`PluginMetric` class. The class
    receives events directly from the :class:`EvaluationPlugin` and can
    emits values on each callback. Usually, an instance of `Metric` is
    created within `PluginMetric`, which is then responsible for its
    update and results. See :class:`PluginMetric` for more details.
    """

    def result(self) -> Optional[TResult_co]:
        """
        Obtains the value of the metric.

        :return: The value of the metric.
        """
        pass

    def reset(self) -> None:
        """
        Resets the metric internal state.

        :return: None.
        """
        pass


class PluginMetric(Metric[TResult_co], ABC):
    """A metric that can be used together with :class:`EvaluationPlugin`.

    This class leaves the implementation of the `result` and `reset` methods
    to child classes while providing an empty implementation of the callbacks
    invoked by the :class:`EvaluationPlugin`. Subclasses should implement
    the `result`, `reset` and the desired callbacks to compute the specific
    metric.

    Remember to call the `super()` method when overriding
    `after_train_iteration` or `after_eval_iteration`.

    An instance of this class usually leverages a `Metric` instance to update,
    reset and emit metric results at appropriate times
    (during specific callbacks).
    """

    def __init__(self):
        """
        Creates an instance of a plugin metric.

        Child classes can safely invoke this (super) constructor as the first
        experience.
        """
        pass

    @abstractmethod
    def result(self) -> Optional[TResult_co]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def before_training(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_training_exp(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        pass

    def after_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        pass

    def before_training_epoch(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_training_iteration(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        pass

    def before_forward(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_forward(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_backward(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_backward(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_training_iteration(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        pass

    def before_update(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_update(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_training_epoch(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_training_exp(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_training(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_eval(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_eval_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        pass

    def after_eval_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        pass

    def before_eval_exp(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_eval(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_eval_iteration(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def before_eval_forward(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_eval_forward(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass


class GenericPluginMetric(PluginMetric[TResult_co], Generic[TResult_co, TMetric]):
    """
    This class provides a generic implementation of a Plugin Metric.
    The user can subclass this class to easily implement custom plugin
    metrics.
    """

    @overload
    def __init__(
        self,
        metric: TMetric,
        reset_at: Literal[
            "iteration", "epoch", "experience", "stream", "never"
        ] = "experience",
        emit_at: Literal["iteration", "epoch", "experience", "stream"] = "experience",
        mode: Literal["train"] = "train",
    ): ...

    @overload
    def __init__(
        self,
        metric: TMetric,
        reset_at: Literal["iteration", "experience", "stream", "never"] = "experience",
        emit_at: Literal["iteration", "experience", "stream"] = "experience",
        mode: Literal["eval"] = "eval",
    ): ...

    def __init__(
        self, metric: TMetric, reset_at="experience", emit_at="experience", mode="eval"
    ):
        super(GenericPluginMetric, self).__init__()
        assert mode in {"train", "eval"}
        if mode == "train":
            assert reset_at in {
                "iteration",
                "epoch",
                "experience",
                "stream",
                "never",
            }
            assert emit_at in {"iteration", "epoch", "experience", "stream"}
        else:
            assert reset_at in {"iteration", "experience", "stream", "never"}
            assert emit_at in {"iteration", "experience", "stream"}
        self._metric: TMetric = metric
        self._reset_at = reset_at
        self._emit_at = emit_at
        self._mode = mode

    def reset(self) -> None:
        self._metric.reset()

    def result(self):
        return self._metric.result()

    def update(self, strategy: "SupervisedTemplate"):
        pass

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result()
        add_exp = self._emit_at == "experience"
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def before_training(self, strategy: "SupervisedTemplate"):
        super().before_training(strategy)
        if self._reset_at == "stream" and self._mode == "train":
            self.reset()

    def before_training_exp(self, strategy: "SupervisedTemplate"):
        super().before_training_exp(strategy)
        if self._reset_at == "experience" and self._mode == "train":
            self.reset()

    def before_training_epoch(self, strategy: "SupervisedTemplate"):
        super().before_training_epoch(strategy)
        if self._reset_at == "epoch" and self._mode == "train":
            self.reset()

    def before_training_iteration(self, strategy: "SupervisedTemplate"):
        super().before_training_iteration(strategy)
        if self._reset_at == "iteration" and self._mode == "train":
            self.reset()

    def after_training_iteration(self, strategy: "SupervisedTemplate"):
        super().after_training_iteration(strategy)
        if self._mode == "train":
            self.update(strategy)
        if self._emit_at == "iteration" and self._mode == "train":
            return self._package_result(strategy)

    def after_training_epoch(self, strategy: "SupervisedTemplate"):
        super().after_training_epoch(strategy)
        if self._emit_at == "epoch" and self._mode == "train":
            return self._package_result(strategy)

    def after_training_exp(self, strategy: "SupervisedTemplate"):
        super().after_training_exp(strategy)
        if self._emit_at == "experience" and self._mode == "train":
            return self._package_result(strategy)

    def after_training(self, strategy: "SupervisedTemplate"):
        super().after_training(strategy)
        if self._emit_at == "stream" and self._mode == "train":
            return self._package_result(strategy)

    def before_eval(self, strategy: "SupervisedTemplate"):
        super().before_eval(strategy)
        if self._reset_at == "stream" and self._mode == "eval":
            self.reset()

    def before_eval_exp(self, strategy: "SupervisedTemplate"):
        super().before_eval_exp(strategy)
        if self._reset_at == "experience" and self._mode == "eval":
            self.reset()

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        super().after_eval_exp(strategy)
        if self._emit_at == "experience" and self._mode == "eval":
            return self._package_result(strategy)

    def after_eval(self, strategy: "SupervisedTemplate"):
        super().after_eval(strategy)
        if self._emit_at == "stream" and self._mode == "eval":
            return self._package_result(strategy)

    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        super().after_eval_iteration(strategy)
        if self._mode == "eval":
            self.update(strategy)
        if self._emit_at == "iteration" and self._mode == "eval":
            return self._package_result(strategy)

    def before_eval_iteration(self, strategy: "SupervisedTemplate"):
        super().before_eval_iteration(strategy)
        if self._reset_at == "iteration" and self._mode == "eval":
            self.reset()


class _ExtendedPluginMetricValue:
    """
    A data structure used to describe a metric value.

    Mainly used to compose the final "name" or "path" of a metric.

    For the moment, this class should be considered an internal utility. Use it
    at your own risk!
    """

    def __init__(
        self,
        *,
        metric_name: str,
        metric_value: Union[MetricType, AlternativeValues],
        phase_name: str,
        stream_name: Optional[str],
        experience_id: Optional[int],
        task_label: Optional[int],
        plot_position: Optional[int] = None,
        **other_info
    ):
        super().__init__()
        self.metric_name = metric_name
        """
        The name of metric, as a string (cannot be None).
        """

        self.metric_value = metric_value
        """
        The metric value name (cannot be None).
        """

        self.phase_name = phase_name
        """
        The phase name, as a str (cannot be None).
        """

        self.stream_name = stream_name
        """
        The stream name, as a str (can be None if stream-agnostic).
        """

        self.experience_id = experience_id
        """
        The experience id, as an int (can be None if experience-agnostic).
        """

        self.task_label = task_label
        """
        The task label, as an int (can be None if task-agnostic).
        """

        self.plot_position = plot_position
        """
        The x position of the value, as an int (cannot be None).
        """
        self.other_info = other_info
        """
        Additional info for this value, as a dictionary (may be empty).
        """


class _ExtendedGenericPluginMetric(
    GenericPluginMetric[List[_ExtendedPluginMetricValue], TMetric]
):
    """
    A generified version of :class:`GenericPluginMetric` which supports emitting
    multiple metrics from a single metric instance.
    Child classes need to emit metrics via `result()` as a list of
    :class:`ExtendedPluginMetricValue`.
    This is in contrast with :class:`GenericPluginMetric`, that expects a
    simpler dictionary "task_label -> value".

    The resulting metric name will be given by the implementation of the
    :meth:`metric_value_name` method.

    For the moment, this class should be considered an internal utility. Use it
    at your own risk!
    """

    def __init__(self, *args, **kwargs):
        """
        Creates an instance of an extended :class:`GenericPluginMetric`.

        :param args: The positional arguments to be passed to the
            :class:`GenericPluginMetric` constructor.

        :param kwargs: The named arguments to be passed to the
            :class:`GenericPluginMetric` constructor.
        """
        super().__init__(*args, **kwargs)

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        emitted_values = self.result()
        default_plot_x_position = strategy.clock.train_iterations

        metrics = []
        for m_value in emitted_values:
            if not isinstance(m_value, _ExtendedPluginMetricValue):
                raise RuntimeError(
                    "Emitted a value that is not of type " "ExtendedPluginMetricValue"
                )

            m_name = self.metric_value_name(m_value)
            x_pos = m_value.plot_position
            if x_pos is None:
                x_pos = default_plot_x_position
            metrics.append(MetricValue(self, m_name, m_value.metric_value, x_pos))

        return metrics

    def metric_value_name(self, m_value: _ExtendedPluginMetricValue) -> str:
        return generic_get_metric_name(default_metric_name_template, vars(m_value))


__all__ = [
    "Metric",
    "PluginMetric",
    "GenericPluginMetric",
    "_ExtendedPluginMetricValue",
    "_ExtendedGenericPluginMetric",
]
