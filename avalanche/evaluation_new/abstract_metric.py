from abc import ABC
from typing import Union, List, Tuple, Optional, Callable, Sequence

from avalanche.evaluation.evaluation_data import EvalData, EvalTestData
from avalanche.evaluation.metric_result import MetricValue
from avalanche.evaluation.metric_units import MetricUnit

MetricResult = Optional[Union[List[MetricValue], MetricValue]]
MetricCallbackType = Union[Callable[[EvalData], MetricResult], MetricUnit]
WrapperCallbackType = Callable[[MetricResult], MetricResult]
ListenersType = Union[Tuple[Optional[type], MetricCallbackType],
                      Tuple['Metric', WrapperCallbackType]]


class Metric(ABC):
    # TODO: doc

    def __init__(self):
        # TODO: doc
        self.listeners: List[ListenersType] = []
        # TODO: field doc

    def on(self,
           event_types: Optional[Union[type, Sequence[type]]],
           *listeners: MetricCallbackType):
        # TODO: doc
        if event_types is None or isinstance(event_types, type):
            event_types = [event_types]
        for listener in listeners:
            for event_type in event_types:
                self.listeners.append((event_type, listener))
        return self

    def attach(self, *listeners: MetricCallbackType):
        # TODO: doc
        return self.on(None, *listeners)

    def use_metric(self, metric: 'Metric', *listeners: WrapperCallbackType):
        # TODO: doc
        for listener in listeners:
            self.listeners.append((metric, listener))
        return self

    def __call__(self, eval_data: EvalData) -> Union[None, List[MetricResult]]:
        # TODO: doc
        emitted_metrics = []
        for listener_definition in self.listeners:
            listener = listener_definition[1]

            if isinstance(listener_definition[0], Metric):
                # This is a wrapped metric added via .use_metric(...)
                metric = listener_definition[0]
                metric_result = metric(eval_data)
                if metric_result is not None:
                    listener(metric_result)
            else:
                # Standard event listeners
                event_type = listener_definition[0]
                if event_type is None or isinstance(eval_data, event_type):
                    metric_result = listener(eval_data)
                    if metric_result is not None:
                        emitted_metrics.append(metric_result)
        return emitted_metrics

    @staticmethod
    def get_task_label(eval_data: Union[EvalData, EvalTestData]) -> int:
        # TODO: doc

        if eval_data.test_phase:
            return eval_data.test_task_label

        return eval_data.training_task_label

    @staticmethod
    def filter_accepted_events(event_types: Union[type, Sequence[type]],
                               train=True, test=False) -> List[type]:
        # TODO: doc
        if isinstance(event_types, type):
            event_types = [event_types]

        accepted = []
        for event_type in event_types:
            if issubclass(event_type, EvalTestData) and test:
                accepted.append(event_type)
            elif issubclass(event_type, EvalData) and train:
                accepted.append(event_type)
        return accepted


__all__ = ['MetricResult', 'Metric']
