from enum import Enum
from numbers import Number
from typing_extensions import Protocol
from typing import NamedTuple, Union, Optional, Tuple, List, Callable

from PIL.Image import Image
from torch import Tensor

from .evaluation_data import EvalData

MetricType = Union[float, int, Tensor, Image]
MetricResult = Optional[Union[List['MetricValue'], 'MetricValue']]


class Metric(Protocol):
    def __call__(self, eval_data: EvalData) -> Union[None, List[MetricResult]]:
        ...


class PlotPosition(Enum):
    SPECIFIC = 0
    ONE_SHOT = 1


class AlternativeValues:
    def __init__(self, *alternatives: MetricType):
        self.alternatives: Tuple[MetricType] = alternatives

    def best_supported_value(self, *supported_types: type) -> \
            Optional[MetricType]:
        for alternative in self.alternatives:
            if isinstance(alternative, supported_types):
                return alternative
        return None


class MetricValue(NamedTuple):
    origin: Metric
    name: str
    value: Union[MetricType, AlternativeValues]
    plot_position: PlotPosition
    x_plot: Number


__all__ = [
    'MetricResult',
    'Metric',
    'PlotPosition',
    'AlternativeValues',
    'MetricValue']
