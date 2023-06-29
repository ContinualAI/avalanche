################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, Tuple, Union

from PIL.Image import Image
from matplotlib.figure import Figure
from torch import Tensor
from enum import Enum

if TYPE_CHECKING:
    from .metric_definitions import Metric

MetricResult = Optional[List["MetricValue"]]


class LoggingType(Enum):
    """A type for MetricValues.

    It can be used by MetricValues to choose how they want to be visualize.
    For example, a 2D tensor could be a line plot or be used to create a
    histogram.
    """

    ANY = 1  # generic type. The logger will use the value type to decide how
    # to serialize it.
    IMAGE = 2
    FIGURE = 3  # Matplotlib figure.
    HISTOGRAM = 4
    # you can add others here. All Tensorboard metrics are good candidates:
    # https://pytorch.org/docs/stable/tensorboard.html
    # just remember to add explicit support to the loggers once you add them.
    # If a metric is already printed correctly by the loggers (e.g. scalars)
    # there is no need to add it here.


@dataclass
class TensorImage:
    image: Tensor

    def __array__(self):
        return self.image.numpy()


MetricType = Union[float, int, str, Tensor, Image, TensorImage, Figure]


class AlternativeValues:
    """
    A container for alternative representations of the same metric value.
    """

    def __init__(self, *alternatives: MetricType):
        self.alternatives: Tuple[MetricType, ...] = alternatives

    def best_supported_value(self, *supported_types: type) -> Optional[MetricType]:
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

    def __init__(
        self,
        origin: "Metric",
        name: str,
        value: Union[MetricType, AlternativeValues],
        x_plot: int,
        logging_type: LoggingType = LoggingType.ANY,
    ):
        """
        Creates an instance of MetricValue.

        :param origin: The originating Metric instance.
        :param name: The display name of this value. This value roughly
            corresponds to the name of the plot in which the value should
            be logged.
        :param value: The value of the metric. Can be a scalar value,
            a PIL Image, or a Tensor. If more than a possible representation
            of the same value exist, an instance of :class:`AlternativeValues`
            can be passed. For instance, the Confusion Matrix can be represented
            both as an Image and a Tensor, in which case an instance of
            :class:`AlternativeValues` carrying both the Tensor and the Image
            is more appropriate. The Logger instance will then select the most
            appropriate way to log the metric according to its capabilities.
        :param x_plot: The position of the value. This value roughly corresponds
            to the x-axis position of the value in a plot. When logging a
            singleton value, pass 0 as a value for this parameter.
        :param logging_type: determines how the metric should be logged.
        """
        self.origin: "Metric" = origin
        self.name: str = name
        self.value: Union[MetricType, AlternativeValues] = value
        self.x_plot: int = x_plot
        self.logging_type = logging_type


__all__ = [
    "MetricType",
    "MetricResult",
    "AlternativeValues",
    "MetricValue",
    "TensorImage",
]
