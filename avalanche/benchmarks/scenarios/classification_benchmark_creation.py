from typing import (
    Any,
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
    _make_classification_scenario,
    FileAndLabel,
    DatasetFactory,
    LazyStreamDefinition,
    create_generic_benchmark_from_filelists,
    create_generic_benchmark_from_paths,
    create_generic_benchmark_from_tensor_lists,
    create_lazy_generic_benchmark,
    create_multi_dataset_generic_benchmark,
)

from avalanche.benchmarks.utils.classification_dataset import (
    SupportedDataset,
    make_classification_dataset,
)
from avalanche.benchmarks.utils.transform_groups import XTransform, YTransform


TDatasetScenario = TypeVar(
    'TDatasetScenario',
    bound='DatasetScenario')


def create_multi_dataset_classification_benchmark(
    train_datasets: Sequence[SupportedDataset],
    test_datasets: Sequence[SupportedDataset],
    *,
    other_streams_datasets: Optional[
        Mapping[str, Sequence[SupportedDataset]]] = None,
    complete_test_set_only: bool = False,
    train_transform: XTransform = None,
    train_target_transform: YTransform = None,
    eval_transform: XTransform = None,
    eval_target_transform: YTransform = None,
    other_streams_transforms: Optional[
        Mapping[str, Tuple[XTransform, YTransform]]] = None,
    dataset_factory: DatasetFactory = make_classification_dataset,
    benchmark_factory: Callable[
        [
            TStreamsUserDict,
            bool
        ], TDatasetScenario
    ] = _make_classification_scenario  # type: ignore
) -> TDatasetScenario:
    """
    Creates a classification benchmark instance given a list of datasets.
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


def create_lazy_classification_benchmark(
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
    dataset_factory: DatasetFactory = make_classification_dataset,
    benchmark_factory: Callable[
        [
            TStreamsUserDict,
            bool
        ], TDatasetScenario
    ] = _make_classification_scenario  # type: ignore
) -> TDatasetScenario:
    """
    Creates a lazily-defined classification benchmark instance given a dataset
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


create_classification_benchmark_from_filelists = \
    create_generic_benchmark_from_filelists


def create_classification_benchmark_from_paths(
    train_lists_of_files: Sequence[Sequence[FileAndLabel]],
    test_lists_of_files: Sequence[Sequence[FileAndLabel]],
    *,
    other_streams_lists_of_files: Optional[Dict[
        str, Sequence[Sequence[FileAndLabel]]
    ]] = None,
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform: XTransform = None,
    train_target_transform: YTransform = None,
    eval_transform: XTransform = None,
    eval_target_transform: YTransform = None,
    other_streams_transforms: Optional[
        Mapping[str, Tuple[XTransform, YTransform]]] = None,
    dataset_factory: DatasetFactory = make_classification_dataset,
    benchmark_factory: Callable[
        [
            TStreamsUserDict,
            bool
        ], TDatasetScenario
    ] = _make_classification_scenario  # type: ignore
) -> TDatasetScenario:
    """
    Creates a classification benchmark instance given a sequence of lists of
    files. A separate dataset will be created for each list. Each of those
    datasets will be considered a separate experience.

    This is very similar to
    :func:`create_classification_benchmark_from_filelists`,
    with the main difference being that
    :func:`create_classification_benchmark_from_filelists` accepts, for each
    experience, a file list formatted in Caffe-style. On the contrary, this
    accepts a list of tuples where each tuple contains two elements: the full
    path to the pattern and its label. Optionally, the tuple may contain a third
    element describing the bounding box of the element to crop. This last
    bounding box may be useful when trying to extract the part of the image
    depicting the desired element.

    For additional info, please refer to
    :func:`create_generic_benchmark_from_paths`.
    """
    return create_generic_benchmark_from_paths(
        train_lists_of_files=train_lists_of_files,
        test_lists_of_files=test_lists_of_files,
        other_streams_lists_of_files=other_streams_lists_of_files,
        task_labels=task_labels,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        other_streams_transforms=other_streams_transforms,
        dataset_factory=dataset_factory,
        benchmark_factory=benchmark_factory
    )


def create_classification_benchmark_from_tensor_lists(
    train_tensors: Sequence[Sequence[Any]],
    test_tensors: Sequence[Sequence[Any]],
    *,
    other_streams_tensors: Optional[Dict[str, Sequence[Sequence[Any]]]] = None,
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform: XTransform = None,
    train_target_transform: YTransform = None,
    eval_transform: XTransform = None,
    eval_target_transform: YTransform = None,
    other_streams_transforms: Optional[
        Mapping[str, Tuple[XTransform, YTransform]]] = None,
    dataset_factory: DatasetFactory = make_classification_dataset,
    benchmark_factory: Callable[
        [
            TStreamsUserDict,
            bool
        ], TDatasetScenario
    ] = _make_classification_scenario  # type: ignore
) -> TDatasetScenario:
    """
    Creates a classification benchmark instance given lists of Tensors. A
    separate dataset will be created from each Tensor tuple (x, y, z, ...)
    and each of those training datasets will be considered a separate training
    experience. Using this helper function is the lowest-level way to create a
    Continual Learning benchmark. When possible, consider using higher level
    helpers.

    Experiences are defined by passing lists of tensors as the `train_tensors`,
    `test_tensors` (and `other_streams_tensors`) parameters. Those parameters
    must be lists containing lists of tensors, one list for each experience.
    Each tensor defines the value of a feature ("x", "y", "z", ...) for all
    patterns of that experience.

    By default the second tensor of each experience will be used to fill the
    `targets` value (label of each pattern).

    For additional info, please refer to
    :func:`create_generic_benchmark_from_tensor_lists`.
    """
    return create_generic_benchmark_from_tensor_lists(
        train_tensors=train_tensors,
        test_tensors=test_tensors,
        other_streams_tensors=other_streams_tensors,
        task_labels=task_labels,
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
    'create_multi_dataset_classification_benchmark',
    'create_lazy_classification_benchmark',
    'create_classification_benchmark_from_filelists',
    'create_classification_benchmark_from_paths',
    'create_classification_benchmark_from_tensor_lists'
]
