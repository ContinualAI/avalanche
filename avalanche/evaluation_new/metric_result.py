from enum import Enum
from numbers import Number
from typing import NamedTuple, Union, List, Sequence, Optional, Tuple

from PIL.Image import Image
from torch import Tensor

MetricType = Union[float, int, Tensor, Image]


class PlotPosition(Enum):
    NEXT = 0
    SPECIFIC = 1
    ONE_SHOT = 2


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
    name: str
    value: Union[MetricType, AlternativeValues]
    plot_position: PlotPosition
    x_plot: Optional[Number] = None


__all__ = ['PlotPosition', 'AlternativeValues', 'MetricValue']
