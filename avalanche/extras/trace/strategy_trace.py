from abc import ABC
from typing import TYPE_CHECKING, List, Type, Optional, Sequence

import warnings

from avalanche.extras.logging.strategy_logger import \
    StrategyLogger


if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricValue
    from avalanche.evaluation import PluginMetric


class StrategyTrace(StrategyLogger, ABC):
    
    def __init__(self):
        super().__init__(text_trace=None)

    def log_metric(self, metric_value: 'MetricValue'):
        super().log_metric(metric_value)

    @staticmethod
    def _find_metric(metric_values: List['MetricValue'],
                     search_metrics: Sequence[Type['PluginMetric']]):
        """
        Finds the required metric value from a list of candidates.

        :param metric_values: The list of metric values.
        :param search_metrics: The originating metric classes (by priority).
        :return: The best metric value (or None if no matching metric is found).
        """
        if len(search_metrics) == 0:
            raise ValueError('No supported metrics')

        found = None
        for s in search_metrics:
            for m in metric_values:
                if isinstance(m.origin, s):
                    found = m
                    break
            if found is not None:
                break

        return found


__all__ = [
    'StrategyTrace'
]
