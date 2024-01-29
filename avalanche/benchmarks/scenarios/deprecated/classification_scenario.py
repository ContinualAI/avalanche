from typing import (
    Callable,
    TypeVar,
    Union,
    Sequence,
    Optional,
    Iterable,
    List,
    Set,
    Tuple,
    Mapping,
    overload,
    TYPE_CHECKING,
)

import warnings

from avalanche.benchmarks.scenarios.deprecated import AbstractClassTimelineExperience

from avalanche.benchmarks.scenarios.deprecated.dataset_scenario import (
    ClassesTimelineCLScenario,
    FactoryBasedStream,
    TStreamsUserDict,
)


from avalanche.benchmarks.utils.dataset_utils import manage_advanced_indexing

if TYPE_CHECKING:
    from avalanche.benchmarks.utils.classification_dataset import (
        TaskAwareClassificationDataset,
    )
    from .dataset_scenario import DatasetScenario


# --- Dataset ---
# From utils:
TClassificationDataset = TypeVar(
    "TClassificationDataset", bound="TaskAwareClassificationDataset"
)

# --- Scenario ---
# From dataset_scenario:
TDatasetScenario = TypeVar("TDatasetScenario", bound="DatasetScenario")
TClassificationScenario = TypeVar(
    "TClassificationScenario", bound="ClassificationScenario"
)

# --- Stream ---
# Defined here:
TClassificationStream = TypeVar("TClassificationStream", bound="ClassificationStream")

# --- Experience ---
TClassificationExperience = TypeVar(
    "TClassificationExperience", bound="ClassificationExperience"
)


def _default_classification_stream_factory(
    stream_name: str, benchmark: "ClassificationScenario"
):
    return ClassificationStream(name=stream_name, benchmark=benchmark)


def _default_classification_experience_factory(
    stream: "ClassificationStream", experience_idx: int
):
    return ClassificationExperience(
        origin_stream=stream, current_experience=experience_idx
    )


class ClassificationScenario(
    ClassesTimelineCLScenario[
        TClassificationStream, TClassificationExperience, TClassificationDataset
    ]
):
    """
    Base implementation of a Continual Learning classification benchmark.

    For more info, please refer to the base class :class:`DatasetScenario`.
    """

    def __init__(
        self: TClassificationScenario,
        *,
        stream_definitions: TStreamsUserDict,
        stream_factory: Callable[
            [str, TClassificationScenario], TClassificationStream
        ] = _default_classification_stream_factory,
        experience_factory: Callable[
            [TClassificationStream, int], TClassificationExperience
        ] = _default_classification_experience_factory,
        complete_test_set_only: bool = False
    ):
        """
        Creates an instance a Continual Learning object classification
        benchmark.

        :param stream_definitions: The definition of the streams. For a more
            precise description, please refer to :class:`DatasetScenario`
        :param n_classes: The number of classes in the scenario. Defaults to
            None.
        :param stream_factory: A callable that, given the name of the
            stream and the benchmark instance, returns a stream instance.
            Defaults to the constructor of :class:`ClassificationStream`.
        :param experience_factory: A callable that, given the
            stream instance and the experience ID, returns an experience
            instance.
            Defaults to the constructor of :class:`ClassificationExperience`.
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

    @property
    def classes_in_experience(self):
        return _LazyStreamClassesInClassificationExps(self)


GenericCLScenario = ClassificationScenario


class ClassificationStream(FactoryBasedStream[TClassificationExperience]):
    def __init__(
        self,
        name: str,
        benchmark: ClassificationScenario,
        *,
        slice_ids: Optional[List[int]] = None,
        set_stream_info: bool = True
    ):
        self.benchmark: ClassificationScenario = benchmark
        super().__init__(
            name=name,
            benchmark=benchmark,
            slice_ids=slice_ids,
            set_stream_info=set_stream_info,
        )


class ClassificationExperience(AbstractClassTimelineExperience[TClassificationDataset]):
    """
    Definition of a learning experience based on a :class:`GenericCLScenario`
    instance.

    This experience implementation uses the generic experience-patterns
    assignment defined in the :class:`GenericCLScenario` instance. Instances of
    this class are usually obtained from a benchmark stream.
    """

    def __init__(
        self: TClassificationExperience,
        origin_stream: ClassificationStream[TClassificationExperience],
        current_experience: int,
    ):
        """
        Creates an instance of a generic experience given the stream from this
        experience was taken and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """

        self._benchmark: ClassificationScenario = origin_stream.benchmark

        dataset: TClassificationDataset = origin_stream.benchmark.stream_definitions[
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
    def benchmark(self) -> ClassificationScenario:
        bench = self._benchmark
        ClassificationExperience._check_unset_attribute("benchmark", bench)
        return bench

    @benchmark.setter
    def benchmark(self, bench: ClassificationScenario):
        self._benchmark = bench

    def _get_stream_def(self):
        return self.benchmark.stream_definitions[self.origin_stream.name]

    @property
    def task_labels(self) -> List[int]:
        with self.no_attribute_masking():  # Needed for "current_experience"
            stream_def = self._get_stream_def()
            return list(stream_def.exps_task_labels[self.current_experience])


GenericClassificationExperience = ClassificationExperience


class _LazyStreamClassesInClassificationExps(Mapping[str, Sequence[Set[int]]]):
    def __init__(self, benchmark: GenericCLScenario):
        self._benchmark = benchmark
        self._default_lcie = _LazyClassesInClassificationExps(benchmark, stream="train")

    def __len__(self):
        return len(self._benchmark.stream_definitions)

    def __getitem__(self, stream_name_or_exp_id):
        if isinstance(stream_name_or_exp_id, str):
            return _LazyClassesInClassificationExps(
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


class _LazyClassesInClassificationExps(Sequence[Optional[Set[int]]]):
    def __init__(self, benchmark: GenericCLScenario, stream: str = "train"):
        self._benchmark = benchmark
        self._stream = stream

    def __len__(self) -> int:
        return len(self._benchmark.streams[self._stream])

    @overload
    def __getitem__(self, exp_id: int) -> Optional[Set[int]]: ...

    @overload
    def __getitem__(self, exp_id: slice) -> Tuple[Optional[Set[int]], ...]: ...

    def __getitem__(self, exp_id: Union[int, slice]) -> LazyClassesInExpsRet:
        indexing_collate = _LazyClassesInClassificationExps._slice_collate
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

        return set(targets)

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
    "ClassificationScenario",
    "GenericCLScenario",
    "ClassificationStream",
    "ClassificationExperience",
    "GenericClassificationExperience",
]
