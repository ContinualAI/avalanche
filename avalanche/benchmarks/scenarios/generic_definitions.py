################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
from abc import abstractmethod
from enum import Enum

try:
    from typing import TypeVar, Tuple, List, Protocol, runtime_checkable, \
        Sequence, Any, Union, Iterable, Generic
except ImportError:
    from typing import TypeVar, Tuple, List, Sequence, Any, Union, Iterable, \
        Generic
    from typing_extensions import Protocol, runtime_checkable

from avalanche.benchmarks.utils import TransformationDataset


class DatasetPart(Enum):
    """An enumeration defining the different dataset parts"""
    CURRENT = 1  # Classes in this step only
    CUMULATIVE = 2  # Encountered classes (including classes in this step)
    OLD = 3  # Encountered classes (excluding classes in this step)
    FUTURE = 4  # Future classes
    COMPLETE = 5  # All classes (encountered + not seen yet)


class DatasetType(Enum):
    """An enumeration defining the different dataset types"""
    TRAIN = 1  # Training set
    VALIDATION = 2  # Validation (or test) set


TrainSet = TypeVar('TrainSet', bound=TransformationDataset)
TestSet = TypeVar('TestSet', bound=TransformationDataset)
TScenario = TypeVar('TScenario')
TStepInfo = TypeVar('TStepInfo', bound='IStepInfo')
TScenarioStream = TypeVar('TScenarioStream', bound='IScenarioStream')


@runtime_checkable
class IStepInfo(Protocol[TScenario, TScenarioStream]):
    """
    Definition of a learning step. A learning step contains a set of patterns
    which has become available at a particular time instant. The content and
    size of a Step is defined by the specific benchmark that creates the
    IStepInfo instance.

    For instance, a step of a New Classes scenario will contain all patterns
    belonging to a subset of classes of the original training set. A step of a
    New Instance scenario will contain patterns from previously seen classes.

    Steps of  Single Incremental Task (a.k.a. task-free) scenarios are usually
    called "batches" while in Multi Task scenarios a Step is usually associated
    to a "task". Finally, in a Multi Incremental Task scenario the Step may be
    composed by patterns from different tasks.
    """

    origin_stream: TScenarioStream
    """
    A reference to the original stream from which this step was obtained.
    """

    scenario: TScenario
    """
    A reference to the scenario.
    """

    current_step: int
    """
    The current step. This is an incremental, 0-indexed, value used to
    keep track of the position of current step in the original stream.
    
    Beware that this value only describes the step position in the original
    stream and may be unrelated to the order in which the strategy will
    receive steps
    """
    @property
    @abstractmethod
    def dataset(self) -> TransformationDataset:
        """
        The dataset containing the patterns available in this step.
        """
        ...

    @property
    @abstractmethod
    def task_label(self) -> int:
        """
        The task label. This value will never have value "None". However,
        for scenarios that don't produce task labels a placeholder value like 0
        is usually set.
        """
        ...


class IScenarioStream(Protocol[TScenario, TStepInfo]):
    """
    A scenario stream describes a sequence of incremental steps. Steps are
    described as :class:`IStepInfo` instances. They contain a set of patterns
    which has become available at a particular time instant along with any
    optional, scenario specific, metadata.

    Most scenario expose two different streams: the training stream and the test
    stream.
    """

    name: str
    """
    The name of the stream.
    """

    scenario: TScenario
    """
    A reference to the scenario this stream belongs to.
    """

    def __getitem__(self: TScenarioStream,
                    step_idx: Union[int, slice, Iterable[int]]) \
            -> Union[TStepInfo, TScenarioStream]:
        """
        Gets a step given its step index (or a stream slice given the step
        order).

        :param step_idx: An int describing the step index or an iterable/slice
            object describing a slice of this stream.
        :return: The step instance associated to the given step index or
            a sliced stream instance.
        """
        ...

    def __len__(self) -> int:
        ...


__all__ = ['DatasetPart', 'DatasetType', 'TrainSet',
           'TestSet', 'IStepInfo', 'TStepInfo', 'TScenario',
           'IScenarioStream', 'TScenarioStream']
