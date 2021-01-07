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

from abc import ABC
from typing import Union, List, Tuple, Optional, Callable, Sequence, Dict, \
    TypeVar, Type

from .evaluation_data import EvalData
from .metric_definitions import Metric
from .metric_results import MetricResult, MetricValue

NMetricValues = Union[None, MetricValue, List[MetricValue]]
MetricCallbackType = Callable[[EvalData], NMetricValues]
WrapperCallbackType = Callable[[MetricResult], NMetricValues]
ListenersType = Union[Tuple[Optional[Type[EvalData]], MetricCallbackType],
                      Tuple['Metric', WrapperCallbackType]]
TAbstractMetric = TypeVar('TAbstractMetric', bound='AbstractMetric')


class SimpleCounter(object):

    def __init__(self, initial_value: int = 0):
        self.count: int = initial_value

    def __int__(self) -> int:
        return self.count

    def __call__(self) -> int:
        val = self.count
        self.count += 1
        return val


class AbstractMetric(ABC, Metric):
    """
    Base class for all Metric.

    This class exposes protected methods that can be used to link event
    types to callbacks and to keep track of the last "x" position of a metric
    value.
    """

    def __init__(self):
        """
        Creates an instance of AbstractMetric.

        This sets up the initial values of different internal utility fields
        that are used for keeping track of the binding between event types
        and callbacks as well as other fields that can be used by child classes.
        """

        self._listeners: List[Tuple[bool, ListenersType]] = []
        """
        A list of _listeners to notify on new events, as a list of tuples.
        The first element of the tuple is a boolean value (True if the listener
        is a metric added using "_use_metric", False if it is a regular
        callback). The second element is a two element tuple describing 1) the
        event type (None when listening on any event)  and 2) the listener to 
        notify.
        """

        self._metric_x_counters: Dict[str, SimpleCounter] = dict()
        """
        A dictionary that can be used to keep track of the next "x" position
        of the value of a metric. Usually used by calling the 
        "_next_x_position" method.
        """

    def _on(self: TAbstractMetric,
            event_types: Optional[Union[Type[EvalData],
                                        Sequence[Type[EvalData]]]],
            *listeners: MetricCallbackType) -> TAbstractMetric:
        """
        Registers the given listeners so that they will be invoked when certain
        training/test events occur.

        :param event_types: The events to listen for. Can be None, which
            means that the listeners will receive all events.
        :param listeners: The listeners to notify when one of the events occur.
        :return: Self.
        """
        if event_types is None or isinstance(event_types, type):
            event_types = [event_types]
        for listener in listeners:
            for event_type in event_types:
                event_type: Type[EvalData]
                self._listeners.append((False, (event_type, listener)))
        return self

    def _attach(self: TAbstractMetric, *listeners: MetricCallbackType) \
            -> TAbstractMetric:
        """
        Registers the given listeners so that they will be invoked when an
        event occurs.

        This is equivalent to "_on(None, listeners)".

        :param listeners: The listeners to notify when an event occurs.
        :return: Self.
        """
        return self._on(None, *listeners)

    def _use_metric(self: TAbstractMetric, metric: Metric,
                    *listeners: WrapperCallbackType) -> TAbstractMetric:
        """
        Registers the given listeners so that they will be invoked when the
        given metric emits a value.

        The metric value will not be directly sent to the Loggers but
        will be made available to listeners. This is useful when using existing
        metrics to compute the value of the current metric.

        :param metric: The metric to observe.
        :param listeners: The listeners to notify when the metric returns
            a metric value.
        :return: Self.
        """
        for listener in listeners:
            self._listeners.append((True, (metric, listener)))
        return self

    def __call__(self, eval_data: EvalData) -> MetricResult:
        """
        Used to feed the metric.

        This method is usually called by the evaluation plugin when an event
        occurs (epoch started, iteration ended, ...).

        When called, the metric will dispatch the event to the appropriate
        listener registered using "_on", "_attach" and "_use_metric".

        :param eval_data: The evaluation data received from the evaluation
            plugin.
        :return: The results of this metric. Can be None.
        """
        emitted_metrics = []
        listener_definition: ListenersType
        for is_metric, (type_or_metric, listener) in self._listeners:

            if is_metric:
                type_or_metric: Metric
                listener: WrapperCallbackType
                # This is a wrapped metric added via .use_metric(...)
                metric_result = type_or_metric(eval_data)
                if metric_result is not None:
                    listener(metric_result)
            else:
                type_or_metric: Type[EvalData]
                listener: MetricCallbackType
                # Standard event _listeners
                if type_or_metric is None or \
                        isinstance(eval_data, type_or_metric):
                    metric_result = listener(eval_data)
                    if metric_result is not None:
                        if isinstance(metric_result, Sequence):
                            emitted_metrics += list(metric_result)
                        else:
                            emitted_metrics.append(metric_result)
        return emitted_metrics

    def _next_x_position(self, metric_name: str, initial_x: int = 0) -> int:
        """
        Utility method that can be used to get the next "x" position of a
        metric value (given its name).

        :param metric_name: The metric value name.
        :param initial_x: The initial "x" value. Defaults to 0.
        :return: The next "x" value to use.
        """
        if metric_name not in self._metric_x_counters:
            self._metric_x_counters[metric_name] = SimpleCounter(initial_x)
        return self._metric_x_counters[metric_name]()


__all__ = [
    'AbstractMetric',
    'MetricCallbackType',
    'WrapperCallbackType',
    'ListenersType',
    'TAbstractMetric']
