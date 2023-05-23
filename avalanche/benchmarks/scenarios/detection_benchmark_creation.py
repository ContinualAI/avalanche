from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
from avalanche.benchmarks.scenarios.dataset_scenario import (
    DatasetScenario,
    TStreamsUserDict,
)
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    DatasetFactory,
    LazyStreamDefinition,
    create_lazy_generic_benchmark,
    create_multi_dataset_generic_benchmark,
)

from avalanche.benchmarks.utils.transform_groups import XTransform, YTransform
from avalanche.benchmarks.scenarios.detection_scenario import (
    DetectionExperience,
    DetectionScenario,
    DetectionStream,
)
from avalanche.benchmarks.utils.detection_dataset import (
    make_detection_dataset,
    SupportedDetectionDataset,
)


TDatasetScenario = TypeVar(
    'TDatasetScenario',
    bound='DatasetScenario')


def _make_detection_scenario(
    stream_definitions: TStreamsUserDict,
    complete_test_set_only: bool
) -> DetectionScenario[
        DetectionStream[
            DetectionExperience],
        DetectionExperience]:
    return DetectionScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only
    )


def create_multi_dataset_detection_benchmark(
    train_datasets: Sequence[SupportedDetectionDataset],
    test_datasets: Sequence[SupportedDetectionDataset],
    *,
    other_streams_datasets: Optional[
        Mapping[str, Sequence[SupportedDetectionDataset]]] = None,
    complete_test_set_only: bool = False,
    train_transform: XTransform = None,
    train_target_transform: YTransform = None,
    eval_transform: XTransform = None,
    eval_target_transform: YTransform = None,
    other_streams_transforms: Optional[
        Mapping[str, Tuple[XTransform, YTransform]]] = None,
    dataset_factory: DatasetFactory = make_detection_dataset,
    benchmark_factory: Callable[
        [
            TStreamsUserDict,
            bool
        ], TDatasetScenario
    ] = _make_detection_scenario  # type: ignore
) -> TDatasetScenario:
    """
    Creates a detection benchmark instance given a list of datasets.
    Each dataset will be considered as a separate experience.

    Contents of the datasets must already be set, including task labels.
    Transformations will be applied if defined.

    For additional info, please refer to
    :func:`create_multi_dataset_generic_benchmark`.
    """
    return create_multi_dataset_generic_benchmark(
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        other_streams_datasets=other_streams_datasets,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        other_streams_transforms=other_streams_transforms,
        dataset_factory=dataset_factory,
        benchmark_factory=benchmark_factory
    )


def create_lazy_detection_benchmark(
    train_generator: LazyStreamDefinition,
    test_generator: LazyStreamDefinition,
    *,
    other_streams_generators: Optional[Dict[str, LazyStreamDefinition]] = None,
    complete_test_set_only: bool = False,
    train_transform: XTransform = None,
    train_target_transform: YTransform = None,
    eval_transform: XTransform = None,
    eval_target_transform: YTransform = None,
    other_streams_transforms: Optional[
        Mapping[str, Tuple[XTransform, YTransform]]] = None,
    dataset_factory: DatasetFactory = make_detection_dataset,
    benchmark_factory: Callable[
        [
            TStreamsUserDict,
            bool
        ], TDatasetScenario
    ] = _make_detection_scenario  # type: ignore
) -> TDatasetScenario:
    """
    Creates a lazily-defined detection benchmark instance given a dataset
    generator for each stream.

    Generators must return properly initialized instances of
    :class:`AvalancheDataset` which will be used to create experiences.

    For additional info, please refer to :func:`create_lazy_generic_benchmark`.
    """
    return create_lazy_generic_benchmark(
        train_generator=train_generator,
        test_generator=test_generator,
        other_streams_generators=other_streams_generators,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        other_streams_transforms=other_streams_transforms,
        dataset_factory=dataset_factory,
        benchmark_factory=benchmark_factory
    )


__all__ = [
    'create_multi_dataset_detection_benchmark',
    'create_lazy_detection_benchmark'
]
