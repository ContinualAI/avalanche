from enum import Enum, auto
from typing import Union, List, Tuple, Optional

from PIL.Image import Image
from torch import Tensor

from avalanche.evaluation import Metric

MetricType = Union[float, int, Tensor, Image]
MetricResult = Optional[List['MetricValue']]


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


class MetricTypes(Enum):
    OTHER = auto()
    """
    Value used to flag a metric type that doesn't fit in any of the other
    standard types.
    """

    ACCURACY = auto()
    """
    Used to flag accuracy values.
    """

    LOSS = auto()
    """
    Used to flag loss values.
    """

    FORGETTING = auto()
    """
    Used to flag values representing accuracy losses.
    """

    CONFUSION_MATRIX = auto()
    """
    Used to flag confusion matrices.
    """

    ELAPSED_TIME = auto()
    """
    Used to flag values describing an elapsed time (usually in seconds).
    """

    CPU_USAGE = auto()
    """
    Used to flag values describing the CPU usage.
    """

    GPU_USAGE = auto()
    """
    Used to flag values describing the GPU usage.
    """

    RAM_USAGE = auto()
    """
    Used to flag values describing the RAM usage (usually in MiB).
    """

    STORAGE_USAGE = auto()
    """
    Used to flag values describing the storage occupation (usually in MiB).
    """


class MetricValue(object):
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
    def __init__(self, origin: Metric, name: str, metric_type: MetricTypes,
                 value: Union[MetricType, AlternativeValues],
                 x_plot: int):
        """
        Creates an instance of MetricValue.

        :param origin: The originating Metric instance.
        :param name: The display name of this value.
        :param metric_type: The type of this metric value, as a element
            from the  MetricTypes enumeration.

        :param value:
        :param x_plot:
        """
        self.origin: Metric = origin
        self.name: str = name
        self.metric_type: MetricTypes = metric_type
        self.value: Union[MetricType, AlternativeValues] = value
        self.x_plot: int = x_plot