################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 10-03-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import TypeVar, List, Callable

from avalanche.benchmarks import GenericExperience, Experience, TScenario, \
    TScenarioStream, GenericCLScenario, TStreamsUserDict, GenericScenarioStream
from avalanche.benchmarks.utils import AvalancheDataset

TDetectionExperience = TypeVar("TDetectionExperience",
                               bound=GenericExperience)


class DetectionCLScenario(GenericCLScenario[TDetectionExperience]):
    """
    Base implementation of a Continual Learning object detection benchmark.

    This is basically a wrapper for a :class:`GenericCLScenario`, with a
    different default experience factory.

    It is recommended to refer to :class:`GenericCLScenario` for more details.
    """
    def __init__(
            self,
            stream_definitions: TStreamsUserDict,
            n_classes: int = None,
            complete_test_set_only: bool = False,
            experience_factory: Callable[
                ["GenericScenarioStream", int], TDetectionExperience
            ] = None):
        """
        Creates an instance a Continual Learning object detection benchmark.

        :param stream_definitions: The definition of the streams. For a more
            precise description, please refer to :class:`GenericCLScenario`
        :param n_classes: The number of classes in the scenario. Defaults to
            None.
        :param complete_test_set_only: If True, the test stream will contain
            a single experience containing the complete test set. This also
            means that the definition for the test stream must contain the
            definition for a single experience.
        :param experience_factory: If not None, a callable that, given the
            benchmark instance and the experience ID, returns an experience
            instance. This parameter is usually used in subclasses (when
            invoking the super constructor) to specialize the experience class.
            Defaults to None, which means that the :class:`DetectionExperience`
            constructor will be used.
        """

        if experience_factory is None:
            experience_factory = DetectionExperience

        super(DetectionCLScenario, self).__init__(
            stream_definitions=stream_definitions,
            complete_test_set_only=complete_test_set_only,
            experience_factory=experience_factory
        )

        self.n_classes = n_classes
        """
        The number of classes in the scenario.
        """


class DetectionExperience(
    Experience[TScenario, TScenarioStream]
):
    """
   Definition of a learning experience based on a :class:`DetectionScenario`
   instance.

   This experience implementation uses the generic experience-patterns
   assignment defined in the :class:`DetectionScenario` instance. Instances of
   this class are usually obtained from an object detection benchmark stream.
   """
    def __init__(
        self: TDetectionExperience,
        origin_stream: TScenarioStream,
        current_experience: int,
    ):
        """
        Creates an instance of an experience given the stream from this
        experience was taken and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """
        self.origin_stream: TScenarioStream = origin_stream
        self.benchmark: TScenario = origin_stream.benchmark
        self.current_experience: int = current_experience

        self.dataset: AvalancheDataset = (
            origin_stream.benchmark.stream_definitions[
                origin_stream.name
            ].exps_data[current_experience]
        )

    def _get_stream_def(self):
        return self.benchmark.stream_definitions[self.origin_stream.name]

    @property
    def task_labels(self) -> List[int]:
        stream_def = self._get_stream_def()
        return list(stream_def.exps_task_labels[self.current_experience])

    @property
    def task_label(self) -> int:
        if len(self.task_labels) != 1:
            raise ValueError(
                "The task_label property can only be accessed "
                "when the experience contains a single task label"
            )

        return self.task_labels[0]


__all__ = [
    "TDetectionExperience",
    "DetectionCLScenario",
    "DetectionExperience"
]
