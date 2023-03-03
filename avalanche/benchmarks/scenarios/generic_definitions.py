################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from abc import abstractmethod
import warnings

from avalanche.benchmarks.utils.classification_dataset import \
    ClassificationDataset

try:
    from typing import (
        TypeVar,
        Tuple,
        List,
        Protocol,
        runtime_checkable,
        Sequence,
        Any,
        Union,
        Iterable,
        Generic,
        TYPE_CHECKING,
    )

    if TYPE_CHECKING:
        from .generic_scenario import CLScenario, CLStream, CLExperience
except ImportError:
    from typing import (
        TypeVar,
        Tuple,
        List,
        Sequence,
        Any,
        Union,
        Iterable,
        Generic,
    )
    from typing_extensions import Protocol, runtime_checkable

from avalanche.benchmarks.utils import make_classification_dataset


TCLScenario = TypeVar("TCLScenario", bound="CLScenario")
TCLExperience = TypeVar("TCLExperience", bound="CLExperience")
TCLStream = TypeVar("TCLStream", bound="CLStream")


@runtime_checkable
class ClassificationExperience(Protocol[TCLScenario, TCLStream]):
    """Definition of a classification experience.

    A classification experience contains a set of patterns
    which has become available at a particular time instant. The content and
    size of an Experience is defined by the specific benchmark that creates the
    IExperience instance.

    For instance, an experience of a New Classes scenario will contain all
    patterns belonging to a subset of classes of the original training set. An
    experience of a New Instance scenario will contain patterns from previously
    seen classes.

    Experiences of Single Incremental Task (a.k.a. task-free) scenarios are
    usually called "batches" while in Multi Task scenarios an Experience is
    usually associated to a "task". Finally, in a Multi Incremental Task
    scenario the Experience may be composed by patterns from different tasks.
    """

    origin_stream: TCLStream
    """
    A reference to the original stream from which this experience was obtained.
    """

    benchmark: TCLScenario
    """
    A reference to the benchmark.
    """

    current_experience: int
    """
    This is an incremental, 0-indexed, value used to keep track of the position 
    of current experience in the original stream.
    
    Beware that this value only describes the experience position in the 
    original stream and may be unrelated to the order in which the strategy will
    encounter experiences.
    """

    dataset: ClassificationDataset
    """
    The dataset containing the patterns available in this experience.
    """

    @property
    @abstractmethod
    def task_labels(self) -> List[int]:
        """
        This list will contain the unique task labels of the patterns contained
        in this experience. In the most common scenarios this will be a list
        with a single value. Note: for scenarios that don't produce task labels,
        a placeholder task label value like 0 is usually set to each pattern
        (see the description of the originating scenario for details).
        """
        ...

    @property
    @abstractmethod
    def task_label(self) -> int:
        """
        The task label. This value will never have value "None". However,
        for scenarios that don't produce task labels a placeholder value like 0
        is usually set. Beware that this field is meant as a shortcut to obtain
        a unique task label: it assumes that only patterns labeled with a
        single task label are present. If this experience contains patterns from
        multiple tasks, accessing this property will result in an exception.
        """
        ...

    @property
    def scenario(self) -> TCLScenario:
        """This property is DEPRECATED, use self.benchmark instead."""
        warnings.warn(
            "Using self.scenario is deprecated in Experience. "
            "Consider using self.benchmark instead.",
            stacklevel=2,
        )
        return self.benchmark


__all__ = [
    "ClassificationExperience",
    "TCLExperience",
    "TCLScenario",
    "TCLStream",
]
