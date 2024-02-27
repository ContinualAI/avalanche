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

from typing import (
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    List,
    Callable,
    Union,
    overload,
)
import warnings


from avalanche.benchmarks.scenarios.deprecated.dataset_scenario import (
    ClassesTimelineCLScenario,
    DatasetScenario,
    FactoryBasedStream,
    TStreamsUserDict,
)
from avalanche.benchmarks.scenarios.deprecated import AbstractClassTimelineExperience
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import manage_advanced_indexing
from avalanche.benchmarks.utils.detection_dataset import DetectionDataset

# --- Dataset ---
# From utils:
TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset", covariant=True)

# --- Scenario ---
# From dataset_scenario:
TDatasetScenario = TypeVar("TDatasetScenario", bound="DatasetScenario")
TDetectionScenario = TypeVar("TDetectionScenario", bound="DetectionScenario")

# --- Stream ---
# Defined here:
TDetectionStream = TypeVar("TDetectionStream", bound="DetectionStream")

# --- Experience ---
# From generic_scenario:
TDetectionExperience = TypeVar("TDetectionExperience", bound="DetectionExperience")


def _default_detection_stream_factory(stream_name: str, benchmark: "DetectionScenario"):
    return DetectionStream(name=stream_name, benchmark=benchmark)


def _default_detection_experience_factory(
    stream: "DetectionStream", experience_idx: int
):
    return DetectionExperience(origin_stream=stream, current_experience=experience_idx)


class DetectionScenario(
    ClassesTimelineCLScenario[TDetectionStream, TDetectionExperience, DetectionDataset]
):
    """
    Base implementation of a Continual Learning object detection benchmark.

    For more info, please refer to the base class :class:`DatasetScenario`.
    """

    def __init__(
        self: TDetectionScenario,
        stream_definitions: TStreamsUserDict,
        n_classes: Optional[int] = None,
        stream_factory: Callable[
            [str, TDetectionScenario], TDetectionStream
        ] = _default_detection_stream_factory,
        experience_factory: Callable[
            [TDetectionStream, int], TDetectionExperience
        ] = _default_detection_experience_factory,
        complete_test_set_only: bool = False,
    ):
        """
        Creates an instance a Continual Learning object detection benchmark.

        :param stream_definitions: The definition of the streams. For a more
            precise description, please refer to :class:`DatasetScenario`
        :param n_classes: The number of classes in the scenario. Defaults to
            None.
        :param stream_factory: A callable that, given the name of the
            stream and the benchmark instance, returns a stream instance.
            Defaults to the constructor of :class:`DetectionStream`.
        :param experience_factory: A callable that, given the
            stream instance and the experience ID, returns an experience
            instance.
            Defaults to the constructor of :class:`DetectionExperience`.
        :param complete_test_set_only: If True, the test stream will contain
            a single experience containing the complete test set. This also
            means that the definition for the test stream must contain the
            definition for a single experience.
        """

        super().__init__(
            stream_definitions=stream_definitions,
            stream_factory=stream_factory,
            experience_factory=experience_factory,
            complete_test_set_only=complete_test_set_only,
        )

        self.n_classes: Optional[int] = n_classes
        """
        The number of classes in the scenario.

        May be None if unknown.
        """

    @property
    def classes_in_experience(self):
        return _LazyStreamClassesInDetectionExps(self)


DetectionCLScenario = DetectionScenario


class DetectionStream(FactoryBasedStream[TDetectionExperience]):
    def __init__(
        self,
        name: str,
        benchmark: DetectionScenario,
        *,
        slice_ids: Optional[List[int]] = None,
        set_stream_info: bool = True
    ):
        self.benchmark: DetectionScenario = benchmark
        super().__init__(
            name=name,
            benchmark=benchmark,
            slice_ids=slice_ids,
            set_stream_info=set_stream_info,
        )


class DetectionExperience(AbstractClassTimelineExperience[DetectionDataset]):
    """
    Definition of a learning experience based on a :class:`DetectionScenario`
    instance.

    This experience implementation uses the generic experience-patterns
    assignment defined in the :class:`DetectionScenario` instance. Instances of
    this class are usually obtained from an object detection benchmark stream.
    """

    def __init__(
        self: TDetectionExperience,
        origin_stream: DetectionStream[TDetectionExperience],
        current_experience: int,
    ):
        """
        Creates an instance of an experience given the stream from this
        experience was taken and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """

        self._benchmark: DetectionScenario = origin_stream.benchmark

        dataset: DetectionDataset = origin_stream.benchmark.stream_definitions[
            origin_stream.name
        ].exps_data[current_experience]

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
            dataset,
            current_experience,
            classes_in_this_exp,
            previous_classes,
            classes_seen_so_far,
            future_classes,
        )

    @property  # type: ignore[override]
    def benchmark(self) -> DetectionScenario:
        bench = self._benchmark
        DetectionExperience._check_unset_attribute("benchmark", bench)
        return bench

    @benchmark.setter
    def benchmark(self, bench: DetectionScenario):
        self._benchmark = bench

    def _get_stream_def(self):
        return self._benchmark.stream_definitions[self.origin_stream.name]

    @property
    def task_labels(self) -> List[int]:
        stream_def = self._get_stream_def()
        return list(stream_def.exps_task_labels[self.current_experience])


GenericDetectionExperience = DetectionExperience


class _LazyStreamClassesInDetectionExps(Mapping[str, Sequence[Optional[Set[int]]]]):
    def __init__(self, benchmark: DetectionScenario):
        self._benchmark = benchmark
        self._default_lcie = _LazyClassesInDetectionExps(benchmark, stream="train")

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


LazyClassesInExpsRet = Union[Tuple[Optional[Set[int]], ...], Optional[Set[int]]]


class _LazyClassesInDetectionExps(Sequence[Optional[Set[int]]]):
    def __init__(self, benchmark: DetectionScenario, stream: str = "train"):
        self._benchmark = benchmark
        self._stream = stream

    def __len__(self):
        return len(self._benchmark.streams[self._stream])

    @overload
    def __getitem__(self, exp_id: int) -> Optional[Set[int]]: ...

    @overload
    def __getitem__(self, exp_id: slice) -> Tuple[Optional[Set[int]], ...]: ...

    def __getitem__(self, exp_id: Union[int, slice]) -> LazyClassesInExpsRet:
        indexing_collate = _LazyClassesInDetectionExps._slice_collate
        result = manage_advanced_indexing(
            exp_id, self._get_single_exp_classes, len(self), indexing_collate
        )
        return result

    def __str__(self):
        return "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"

    def _get_single_exp_classes(self, exp_id) -> Optional[Set[int]]:
        b = self._benchmark.stream_definitions[self._stream]
        if not b.is_lazy and exp_id not in b.exps_data.targets_field_sequence:
            raise IndexError
        targets = b.exps_data.targets_field_sequence[exp_id]
        if targets is None:
            return None

        classes_in_exp = set()
        for target in targets:
            for label in target["labels"]:
                classes_in_exp.add(int(label))
        return classes_in_exp

    @staticmethod
    def _slice_collate(
        classes_in_exps: Iterable[Optional[Iterable[int]]],
    ) -> Optional[Tuple[Set[int], ...]]:
        result: List[Set[int]] = []
        for x in classes_in_exps:
            if x is None:
                return None
            result.append(set(x))

        return tuple(result)


__all__ = [
    "DetectionScenario",
    "DetectionCLScenario",
    "DetectionStream",
    "GenericDetectionExperience",
    "DetectionExperience",
]
