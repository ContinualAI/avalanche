from typing import (
    Callable,
    Generic,
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
)

import warnings

from avalanche.benchmarks.scenarios.generic_scenario import (
    AbstractClassTimelineExperience,
    CLStream,
    ClassificationExperienceProtocol,
    SettableGenericExperienceProtocol
)

from avalanche.benchmarks.scenarios.dataset_scenario import (
    DatasetScenario, ClassesTimelineCLScenario, FactoryBasedStream, TStreamsUserDict
)

from avalanche.benchmarks.utils import (
    AvalancheDataset
)
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)
from avalanche.benchmarks.utils.dataset_utils import manage_advanced_indexing


### Dataset ###
# From utils:
TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset", covariant=True)

### Scenario ###
# From dataset_scenario:
TDatasetScenario = TypeVar(
    "TDatasetScenario", bound="DatasetScenario"
)
TGenericCLScenario = TypeVar('TGenericCLScenario', bound='GenericCLScenario')

### Stream ###
# From generic_scenario:
TCLStream = TypeVar('TCLStream', bound='CLStream', covariant=True)
# Defined here:
TClassificationStream = TypeVar('TClassificationStream', bound='ClassificationStream')

### Experience ###
# From generic_scenario:
TSettableGenericExperience = TypeVar('TSettableGenericExperience', bound='SettableGenericExperienceProtocol')
TClassificationExperience = TypeVar('TClassificationExperience', bound='ClassificationExperienceProtocol')
TGenericClassificationExperience = TypeVar(
    "TGenericClassificationExperience", bound="GenericClassificationExperience"
)


# TODO: more appropriate name (like ClassificationScenario)
class GenericCLScenario(ClassesTimelineCLScenario[TGenericCLScenario, TCLStream, TClassificationExperience, ClassificationDataset]):
    """
    Base implementation of a Continual Learning classification benchmark.

    For more info, please refer to the base class :class:`DatasetScenario`.
    """
    
    def __init__(
        self: TGenericCLScenario,
        *,
        stream_definitions: TStreamsUserDict,
        stream_factory: Optional[Callable[[str, TGenericCLScenario], TCLStream]] = None,
        experience_factory: Optional[Callable[[TCLStream, int], TClassificationExperience]] = None,
        complete_test_set_only: bool = False):

        if stream_factory is None:
            stream_factory = ClassificationStream # type: ignore
        
        if experience_factory is None:
            experience_factory = GenericClassificationExperience # type: ignore

        # PyLance -_-
        assert stream_factory is not None
        assert experience_factory is not None

        super().__init__(
            stream_definitions=stream_definitions,
            stream_factory=stream_factory,
            experience_factory=experience_factory,
            complete_test_set_only=complete_test_set_only)

    @property
    def classes_in_experience(self):
        return LazyStreamClassesInExps(self)


class ClassificationStream(
    FactoryBasedStream[
        TGenericCLScenario, TGenericClassificationExperience
    ],
    Generic[TGenericCLScenario, TGenericClassificationExperience]
):
    def __init__(
        self,
        name: str,
        benchmark: TGenericCLScenario,
        *,
        slice_ids: Optional[List[int]] = None,
        set_stream_info: bool = True
    ):
        super().__init__(
            name=name,
            benchmark=benchmark,
            slice_ids=slice_ids,
            set_stream_info=set_stream_info)


class GenericClassificationExperience(
    AbstractClassTimelineExperience[
        TGenericCLScenario, ClassificationStream[
           TGenericCLScenario, TGenericClassificationExperience
       ], ClassificationDataset
    ],
    ClassificationExperienceProtocol[TGenericCLScenario, ClassificationStream[
        TGenericCLScenario, TGenericClassificationExperience
    ]]
):
    """
    Definition of a learning experience based on a :class:`GenericCLScenario`
    instance.

    This experience implementation uses the generic experience-patterns
    assignment defined in the :class:`GenericCLScenario` instance. Instances of
    this class are usually obtained from a benchmark stream.
    """

    def __init__(
        self: TGenericClassificationExperience,
        origin_stream: ClassificationStream[
            TGenericCLScenario, TGenericClassificationExperience
        ],
        current_experience: int
    ):
        """
        Creates an instance of a generic experience given the stream from this
        experience was taken and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """

        dataset: ClassificationDataset = (
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
            dataset,
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
        with self.no_attribute_masking(): # Needed for "current_experience"
            stream_def = self._get_stream_def()
            return list(stream_def.exps_task_labels[self.current_experience])


class LazyStreamClassesInExps(Mapping[str, Sequence[Set[int]]]):
    def __init__(self, benchmark: GenericCLScenario):
        self._benchmark = benchmark
        self._default_lcie = LazyClassesInExps(benchmark, stream="train")

    def __len__(self):
        return len(self._benchmark.stream_definitions)

    def __getitem__(self, stream_name_or_exp_id):
        if isinstance(stream_name_or_exp_id, str):
            return LazyClassesInExps(
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

class LazyClassesInExps(Sequence[Optional[Set[int]]]):
    def __init__(self, benchmark: GenericCLScenario, stream: str = "train"):
        self._benchmark = benchmark
        self._stream = stream

    def __len__(self) -> int:
        return len(self._benchmark.streams[self._stream])

    @overload
    def __getitem__(self, exp_id: int, /) -> Optional[Set[int]]:
        ...
    
    @overload
    def __getitem__(self, exp_id: slice, /) -> Tuple[Optional[Set[int]], ...]:
        ...
    
    def __getitem__(self, exp_id: Union[int, slice], /) -> LazyClassesInExpsRet:
        indexing_collate = LazyClassesInExps._slice_collate
        result =  manage_advanced_indexing(
            exp_id,
            self._get_single_exp_classes,
            len(self),
            indexing_collate
        )
        return result

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )

    def _get_single_exp_classes(self, exp_id) -> Optional[Set[int]]:
        b = self._benchmark.stream_definitions[self._stream]
        if not b.is_lazy and exp_id not in b.exps_data.targets_field_sequence:
            raise IndexError
        targets = b.exps_data.targets_field_sequence[exp_id]
        if targets is None:
            return None
        
        return set(targets)

    @staticmethod
    def _slice_collate(classes_in_exps: Iterable[Optional[Iterable[int]]]) -> Optional[Tuple[Set[int], ...]]:
        result: List[Set[int]] = []
        for x in classes_in_exps:
            if x is None:
                return None
            result.append(set(x))

        return tuple(result)


__all__ = [
    "GenericCLScenario",
    "ClassificationStream",
    "GenericClassificationExperience",
]
