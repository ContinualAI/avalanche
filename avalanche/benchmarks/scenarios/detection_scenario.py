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
import copy
import warnings
from abc import abstractmethod, ABC
from typing import TypeVar, List, Callable, Protocol, runtime_checkable, \
    Union, Iterable, Generic, Sequence, Optional, Mapping, Set

from avalanche.benchmarks import (
    TCLScenario,
    TCLStream,
    GenericCLScenario,
    TStreamsUserDict, TCLExperience, )
from avalanche.benchmarks.scenarios.classification_scenario import \
    _get_slice_ids
from avalanche.benchmarks.utils.dataset_utils import manage_advanced_indexing
from avalanche.benchmarks.utils.detection_dataset import DetectionDataset

TGenericCLDetectionScenario = TypeVar(
    "TGenericCLDetectionScenario", bound="DetectionCLScenario"
)
TGenericDetectionExperience = TypeVar(
    "TGenericDetectionExperience", bound="GenericDetectionExperience"
)
TGenericScenarioStream = TypeVar(
    "TGenericScenarioStream", bound="DetectionStream"
)


class DetectionCLScenario(GenericCLScenario[TCLExperience]):
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
            ["DetectionStream", int], TCLExperience
        ] = None,
    ):
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
            experience_factory = GenericDetectionExperience

        super(DetectionCLScenario, self).__init__(
            stream_definitions=stream_definitions,
            complete_test_set_only=complete_test_set_only,
            experience_factory=experience_factory,
        )

        self.n_classes = n_classes
        """
        The number of classes in the scenario.
        """

    @GenericCLScenario.classes_in_experience.getter
    def classes_in_experience(
            self,
    ) -> Mapping[str, Sequence[Optional[Set[int]]]]:
        """
        A dictionary mapping each stream (by name) to a list.

        Each element of the list is a set describing the classes included in
        that experience (identified by its index).

        In previous releases this field contained the list of sets for the
        training stream (that is, there was no way to obtain the list for other
        streams). That behavior is deprecated and support for that usage way
        will be removed in the future.
        """

        return _LazyStreamClassesInDetectionExps(self)


class _LazyStreamClassesInDetectionExps(Mapping[str,
                                                Sequence[Optional[Set[int]]]]):
    def __init__(self, benchmark: GenericCLScenario):
        self._benchmark = benchmark
        self._default_lcie = _LazyClassesInDetectionExps(
            benchmark, stream="train")

    def __len__(self):
        return len(self._benchmark.stream_definitions)

    def __getitem__(self, stream_name_or_exp_id):
        if isinstance(stream_name_or_exp_id, str):
            return _LazyClassesInDetectionExps(
                self._benchmark, stream=stream_name_or_exp_id
            )

        warnings.warn(
            "Using classes_in_experience[exp_id] is deprecated. "
            "Consider using classes_in_experience[stream_name][exp_id]"
            "instead.",
            stacklevel=2,
        )
        return self._default_lcie[stream_name_or_exp_id]

    def __iter__(self):
        yield from self._benchmark.stream_definitions.keys()


class _LazyClassesInDetectionExps(Sequence[Optional[Set[int]]]):
    def __init__(self, benchmark: GenericCLScenario, stream: str = "train"):
        self._benchmark = benchmark
        self._stream = stream

    def __len__(self):
        return len(self._benchmark.streams[self._stream])

    def __getitem__(self, exp_id) -> Set[int]:
        return manage_advanced_indexing(
            exp_id,
            self._get_single_exp_classes,
            len(self),
            _LazyClassesInDetectionExps._slice_collate,
        )

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )

    def _get_single_exp_classes(self, exp_id):
        b = self._benchmark.stream_definitions[self._stream]
        if not b.is_lazy and exp_id not in b.exps_data.targets_field_sequence:
            raise IndexError
        targets = b.exps_data.targets_field_sequence[exp_id]
        if targets is None:
            return None

        classes_in_exp = set()
        for target in targets:
            for label in target['labels']:
                classes_in_exp.add(int(label))
        return classes_in_exp

    @staticmethod
    def _slice_collate(*classes_in_exps: Optional[Set[int]]):
        if any(x is None for x in classes_in_exps):
            return None

        return [list(x) for x in classes_in_exps]


class DetectionScenarioStream(Protocol[TCLScenario, TCLExperience]):
    """
    A scenario stream describes a sequence of incremental experiences.
    Experiences are described as :class:`IExperience` instances. They contain a
    set of patterns which has become available at a particular time instant
    along with any optional, scenario specific, metadata.

    Most scenario expose two different streams: the training stream and the test
    stream.
    """

    name: str
    """
    The name of the stream.
    """

    benchmark: TCLScenario
    """
    A reference to the scenario this stream belongs to.
    """

    @property
    def scenario(self) -> TCLScenario:
        """This property is DEPRECATED, use self.benchmark instead."""
        warnings.warn(
            "Using self.scenario is deprecated ScenarioStream. "
            "Consider using self.benchmark instead.",
            stacklevel=2,
        )
        return self.benchmark

    def __getitem__(
        self: TCLStream, experience_idx: Union[int, slice, Iterable[int]]
    ) -> Union[TCLExperience, TCLStream]:
        """
        Gets an experience given its experience index (or a stream slice given
        the experience order).

        :param experience_idx: An int describing the experience index or an
            iterable/slice object describing a slice of this stream.
        :return: The Experience instance associated to the given experience
            index or a sliced stream instance.
        """
        ...

    def __len__(self) -> int:
        """
        Used to get the length of this stream (the amount of experiences).

        :return: The amount of experiences in this stream.
        """
        ...


class DetectionStream(
    Generic[TCLExperience, TGenericCLDetectionScenario],
    DetectionScenarioStream[
        TGenericCLDetectionScenario, TCLExperience
    ],
    Sequence[TCLExperience],
):
    def __init__(
        self: TGenericScenarioStream,
        name: str,
        benchmark: TGenericCLDetectionScenario,
        *,
        slice_ids: List[int] = None,
    ):
        super(DetectionStream, self).__init__()
        self.slice_ids: Optional[List[int]] = slice_ids
        """
        Describes which experiences are contained in the current stream slice. 
        Can be None, which means that this object is the original stream. """

        self.name: str = name
        """
        The name of the stream (for instance: "train", "test", "valid", ...).
        """

        self.benchmark = benchmark
        """
        A reference to the benchmark.
        """

    def __len__(self) -> int:
        """
        Gets the number of experiences this stream it's made of.

        :return: The number of experiences in this stream.
        """
        if self.slice_ids is None:
            return len(self.benchmark.stream_definitions[self.name].exps_data)
        else:
            return len(self.slice_ids)

    def __getitem__(
        self, exp_idx: Union[int, slice, Iterable[int]]
    ) -> Union[TCLExperience, TCLStream]:
        """
        Gets an experience given its experience index (or a stream slice given
        the experience order).

        :param exp_idx: An int describing the experience index or an
            iterable/slice object describing a slice of this stream.

        :return: The experience instance associated to the given experience
            index or a sliced stream instance.
        """
        if isinstance(exp_idx, int):
            if exp_idx < len(self):
                if self.slice_ids is None:
                    return self.benchmark.experience_factory(self, exp_idx)
                else:
                    return self.benchmark.experience_factory(
                        self, self.slice_ids[exp_idx]
                    )
            raise IndexError(
                "Experience index out of bounds" + str(int(exp_idx))
            )
        else:
            return self._create_slice(exp_idx)

    def _create_slice(
        self: TGenericScenarioStream,
        exps_slice: Union[int, slice, Iterable[int]],
    ) -> TCLStream:
        """
        Creates a sliced version of this stream.

        In its base version, a shallow copy of this stream is created and
        then its ``slice_ids`` field is adapted.

        :param exps_slice: The slice to use.
        :return: A sliced version of this stream.
        """
        stream_copy = copy.copy(self)
        slice_exps = _get_slice_ids(exps_slice, len(self))

        if self.slice_ids is None:
            stream_copy.slice_ids = slice_exps
        else:
            stream_copy.slice_ids = [self.slice_ids[x] for x in slice_exps]
        return stream_copy

    def drop_previous_experiences(self, to_exp: int) -> None:
        """
        Drop the reference to experiences up to a certain experience ID
        (inclusive).

        This means that any reference to experiences with ID [0, from_exp] will
        be released. By dropping the reference to previous experiences, the
        memory associated with them can be freed, especially the one occupied by
        the dataset. However, if external references to the experience or the
        dataset still exist, dropping previous experiences at the stream level
        will have little to no impact on the memory usage.

        To make sure that the underlying dataset can be freed, make sure that:
        - No reference to previous datasets or experiences are kept in you code;
        - The replay implementation doesn't keep a reference to previous
            datasets (in which case, is better to store a copy of the raw
            tensors instead);
        - The benchmark is being generated using a lazy initializer.

        By dropping previous experiences, those experiences will no longer be
        available in the stream. Trying to access them will result in an
        exception.

        :param to_exp: The ID of the last exp to drop (inclusive). Can be a
            negative number, in which case this method doesn't have any effect.
            Can be greater or equal to the stream length, in which case all
            currently loaded experiences will be dropped.
        :return: None
        """
        self.benchmark.stream_definitions[
            self.name
        ].exps_data.drop_previous_experiences(to_exp)


@runtime_checkable
class DetectionExperience(Protocol[TCLScenario, TCLStream]):
    """Definition of a detection experience.

    A classification detection contains a set of patterns
    which has become available at a particular time instant. The content and
    size of an Experience is defined by the specific benchmark that creates the
    IExperience instance.

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

    dataset: DetectionDataset
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


class AbstractDetectionExperience(
    DetectionExperience[TGenericCLDetectionScenario, TCLStream], ABC
):
    """
    Definition of a learning experience. A learning experience contains a set of
    patterns which has become available at a particular time instant. The
    content and size of an Experience is defined by the specific benchmark that
    creates the experience.

    For instance, an experience of a New Classes scenario will contain all
    patterns belonging to a subset of classes of the original training set. An
    experience of a New Instance scenario will contain patterns from previously
    seen classes.
    """

    def __init__(
        self,
        origin_stream: TCLStream,
        current_experience: int,
        classes_in_this_exp: Sequence[int],
        previous_classes: Sequence[int],
        classes_seen_so_far: Sequence[int],
        future_classes: Optional[Sequence[int]],
    ):
        """
        Creates an instance of the abstract experience given the benchmark
        stream, the current experience ID and data about the classes timeline.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        :param classes_in_this_exp: The list of classes in this experience.
        :param previous_classes: The list of classes in previous experiences.
        :param classes_seen_so_far: List of classes of current and previous
            experiences.
        :param future_classes: The list of classes of next experiences.
        """

        self.origin_stream: TCLStream = origin_stream

        # benchmark keeps a reference to the base benchmark
        self.benchmark: TCLScenario = origin_stream.benchmark

        # current_experience is usually an incremental, 0-indexed, value used to
        # keep track of the current batch/task.
        self.current_experience: int = current_experience

        self.classes_in_this_experience: Sequence[int] = classes_in_this_exp
        """ The list of classes in this experience """

        self.previous_classes: Sequence[int] = previous_classes
        """ The list of classes in previous experiences """

        self.classes_seen_so_far: Sequence[int] = classes_seen_so_far
        """ List of classes of current and previous experiences """

        self.future_classes: Optional[Sequence[int]] = future_classes
        """ The list of classes of next experiences """

    @property
    def task_label(self) -> int:
        """
        The task label. This value will never have value "None". However,
        for scenarios that don't produce task labels a placeholder value like 0
        is usually set. Beware that this field is meant as a shortcut to obtain
        a unique task label: it assumes that only patterns labeled with a
        single task label are present. If this experience contains patterns from
        multiple tasks, accessing this property will result in an exception.
        """
        if len(self.task_labels) != 1:
            raise ValueError(
                "The task_label property can only be accessed "
                "when the experience contains a single task label"
            )

        return self.task_labels[0]


class GenericDetectionExperience(
    AbstractDetectionExperience[
        TGenericCLDetectionScenario,
        DetectionStream[
            TGenericDetectionExperience, TGenericCLDetectionScenario
        ],
    ]
):
    """
    Definition of a learning experience based on a :class:`GenericCLScenario`
    instance.

    This experience implementation uses the generic experience-patterns
    assignment defined in the :class:`GenericCLScenario` instance. Instances of
    this class are usually obtained from a benchmark stream.
    """

    def __init__(
        self: TGenericDetectionExperience,
        origin_stream: DetectionStream[
            TGenericDetectionExperience, TGenericCLDetectionScenario
        ],
        current_experience: int,
    ):
        """
        Creates an instance of a generic experience given the stream from this
        experience was taken and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """
        self.dataset: DetectionDataset = (
            origin_stream.benchmark.stream_definitions[
                origin_stream.name
            ].exps_data[current_experience]
        )

        (
            classes_in_this_exp,
            previous_classes,
            classes_seen_so_far,
            future_classes,
        ) = origin_stream.benchmark.get_classes_timeline(
            current_experience, stream=origin_stream.name
        )

        super().__init__(
            origin_stream,
            current_experience,
            classes_in_this_exp,
            previous_classes,
            classes_seen_so_far,
            future_classes,
        )

    def _get_stream_def(self):
        return self.benchmark.stream_definitions[self.origin_stream.name]

    @property
    def task_labels(self) -> List[int]:
        stream_def = self._get_stream_def()
        return list(stream_def.exps_task_labels[self.current_experience])


__all__ = [
    'TGenericCLDetectionScenario',
    'TGenericDetectionExperience',
    'TGenericScenarioStream',
    'DetectionCLScenario',
    'DetectionStream',
    'AbstractDetectionExperience',
    'GenericDetectionExperience'
]
