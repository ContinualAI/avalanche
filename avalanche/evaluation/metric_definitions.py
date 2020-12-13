from numbers import Number
from typing_extensions import Protocol
from typing import NamedTuple, Union, Optional, Tuple, List

from PIL.Image import Image
from torch import Tensor

from .evaluation_data import EvalData

MetricType = Union[float, int, Tensor, Image]
MetricResult = Optional[Union[List['MetricValue'], 'MetricValue']]


class Metric(Protocol):
    """
    Protocol definition of a metric.

    A metric simply accepts an evaluation data object containing relevant
    information retrieved from the strategy and optionally outputs values
    to be logged.
    """
    def __call__(self, eval_data: EvalData) -> Union[None, List[MetricResult]]:
        ...


class AlternativeValues:
    """
    A container for alternative representations of the same metric value.
    """
    def __init__(self, *alternatives: MetricType):
        self.alternatives: Tuple[MetricType] = alternatives

    def best_supported_value(self, *supported_types: type) -> \
            Optional[MetricType]:
        """
        Retrieves a supported representation for this metric value.

        :param supported_types: A list of supported value types.
        :return: The best supported representation. Returns None if no supported
            representation is found.
        """
        for alternative in self.alternatives:
            if isinstance(alternative, supported_types):
                return alternative
        return None


class MetricValue(NamedTuple):
    """
    The result of a Metric.

    A result has a name, a value and a "x" position in which the metric value
    should be plotted.

    The "value" field can also be an instance of "AlternativeValues", in which
    case it means that alternative representations exist for this value. For
    instance, the Confusion Matrix can be represented both as a Tensor and as
    an Image. It's up to the Logger, according to its capabilities, decide which
    representation to use.
    """
    origin: Metric
    name: str
    value: Union[MetricType, AlternativeValues]
    x_plot: Number


__all__ = [
    'MetricResult',
    'Metric',
    'AlternativeValues',
    'MetricValue']
