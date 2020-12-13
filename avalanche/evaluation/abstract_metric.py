from abc import ABC
from typing import Union, List, Tuple, Optional, Callable, Sequence, Dict

from .evaluation_data import EvalData
from .metric_definitions import Metric, MetricResult
from .metric_units import MetricUnit


MetricCallbackType = Union[Callable[[EvalData], MetricResult], MetricUnit]
WrapperCallbackType = Callable[[MetricResult], MetricResult]
ListenersType = Union[Tuple[Optional[type], MetricCallbackType],
                      Tuple['Metric', WrapperCallbackType]]


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
    # TODO: doc

    def __init__(self):
        # TODO: doc
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

    def _on(self,
            event_types: Optional[Union[type, Sequence[type]]],
            *listeners: MetricCallbackType):
        # TODO: doc
        if event_types is None or isinstance(event_types, type):
            event_types = [event_types]
        for listener in listeners:
            for event_type in event_types:

                self._listeners.append((False, (event_type, listener)))
        return self

    def _attach(self, *listeners: MetricCallbackType):
        # TODO: doc
        return self._on(None, *listeners)

    def _use_metric(self, metric: Metric, *listeners: WrapperCallbackType):
        # TODO: doc
        for listener in listeners:
            self._listeners.append((True, (metric, listener)))
        return self

    def __call__(self, eval_data: EvalData) -> Union[None, List[MetricResult]]:
        # TODO: doc
        emitted_metrics = []
        for is_metric, listener_definition in self._listeners:
            listener = listener_definition[1]

            if is_metric:
                # This is a wrapped metric added via .use_metric(...)
                metric = listener_definition[0]
                metric_result = metric(eval_data)
                if metric_result is not None:
                    listener(metric_result)
            else:
                # Standard event _listeners
                event_type = listener_definition[0]
                if event_type is None or isinstance(eval_data, event_type):
                    metric_result = listener(eval_data)
                    if metric_result is not None:
                        emitted_metrics.append(metric_result)
        return emitted_metrics

    def _next_x_position(self, metric_name: str, initial_x:int = 0) -> int:
        if metric_name not in self._metric_x_counters:
            self._metric_x_counters[metric_name] = SimpleCounter(initial_x)
        return self._metric_x_counters[metric_name]()


__all__ = ['AbstractMetric']
