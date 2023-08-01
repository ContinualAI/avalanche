from collections import OrderedDict, defaultdict, deque
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from avalanche.benchmarks.scenarios.generic_scenario import CLScenario
from avalanche.benchmarks.utils.data import (
    _FlatDataWithTransform,
    AvalancheDataset,
)
from avalanche.benchmarks.utils.dataset_definitions import IDataset
from avalanche.benchmarks.utils.dataset_utils import find_list_from_index
from avalanche.benchmarks.utils.flat_data import FlatData

from torch.utils.data import Subset, ConcatDataset, Dataset

from avalanche.benchmarks.utils.transform_groups import EmptyTransformGroups
from avalanche.benchmarks.utils.transforms import TupleTransform
from torchvision.datasets.vision import StandardTransform


def dataset_list_from_benchmark(benchmark: CLScenario) -> List[AvalancheDataset]:
    """
    Traverse a benchmark and obtain the dataset of each experience.

    This will traverse all streams in alphabetical order.

    :param benchmark: The benchmark to traverse.
    :return: The list of datasets.
    """
    single_datasets = OrderedDict()
    for stream_name in sorted(benchmark.streams.keys()):
        stream = benchmark.streams[stream_name]
        for experience in stream:
            dataset: AvalancheDataset = experience.dataset
            if dataset not in single_datasets:
                single_datasets[dataset] = dataset

    return list(single_datasets.keys())


def flat_datasets_from_benchmark(
    benchmark: CLScenario, include_leaf_transforms: bool = True
):
    """
    Obtain a list of flattened datasets from a benchmark.

    In practice, this function will traverse all the
    datasets in the benchmark to find the leaf datasets.
    A dataset can be traversed and flattened to (one or more) leaf
    dataset(s) if all subset and dataset concatenations point to a
    single leaf dataset and if transformations are the same across
    all paths.

    Traversing the dataset means traversing :class:`AvalancheDataset`
    as well as PyTorch :class:`Subset` and :class:`ConcatDataset` to
    obtain the leaf datasets, the indices, and the transformations chain.

    Note: this means that datasets will be plain PyTorch datasets,
    not :class:`AvalancheDataset` (Avalanche datasets are traversed).

    In common benchmarks, this returns one dataset for the train
    and one dataset for test.

    :param benchmark: The benchmark to traverse.
    :param include_leaf_transforms: If True, include the transformations
        found in the leaf dataset in the transforms list. Defaults to True.
    :return: The list of leaf datasets. Each element in the list is
        a tuple `(dataset, indices, transforms)`.
    """
    single_datasets = dataset_list_from_benchmark(benchmark)
    leaves = leaf_datasets(
        AvalancheDataset(single_datasets),
        include_leaf_transforms=include_leaf_transforms,
    )

    result = []
    for dataset, indices_and_transforms in leaves.items():
        # Check that all transforms are the same
        first_transform = indices_and_transforms[0][1]
        same_transforms = all([first_transform == t for _, t in indices_and_transforms])

        if not same_transforms:
            for indices, transforms in indices_and_transforms:
                result.append((dataset, indices, transforms))
            continue

        flat_indices = [i for i, _ in indices_and_transforms]

        result.append((dataset, flat_indices, first_transform))
    return result


T = TypeVar("T")
Y = TypeVar("Y")
TraverseT = Union[Dataset, AvalancheDataset, FlatData, IDataset]


def _traverse_supported_dataset_with_intermediate(
    dataset: TraverseT,
    values_selector: Callable[
        [TraverseT, Optional[List[int]], Optional[T]], Optional[List[Y]]
    ],
    intermediate_selector: Optional[Callable[[TraverseT, Optional[T]], T]] = None,
    intermediate: Optional[T] = None,
    indices: Optional[List[int]] = None,
) -> List[Y]:
    """
    Traverse the given dataset by gathering required info.

    The given dataset is traversed by covering all sub-datasets
    contained in PyTorch :class:`Subset` and :class`ConcatDataset`
    as well as :class:`AvalancheDataset`.

    For each dataset, the `values_selector` will be called to gather
    the required information. The values returned by the given selector
    are then concatenated to create a final list of values.

    While traversing, the `intermediate_selector` (if provided)
    will be called to create a chain of intermediate values, which
    are passed to `values_selector`.

    :param dataset: The dataset to traverse.
    :param values_selector: A function that, given the dataset
        and the indices to consider (which may be None if the entire
        dataset must be considered), returns a list of selected values.
    :returns: The list of selected values.
    """

    if intermediate_selector is not None:
        intermediate = intermediate_selector(dataset, intermediate)

    leaf_result: Optional[List[Y]] = values_selector(dataset, indices, intermediate)

    if leaf_result is not None:
        if len(leaf_result) == 0:
            raise RuntimeError("Empty result")
        return leaf_result

    if isinstance(dataset, AvalancheDataset):
        return list(
            _traverse_supported_dataset_with_intermediate(
                dataset._flat_data,
                values_selector,
                intermediate_selector=intermediate_selector,
                indices=indices,
                intermediate=intermediate,
            )
        )

    if isinstance(dataset, Subset):
        if indices is None:
            indices = [dataset.indices[x] for x in range(len(dataset))]
        else:
            indices = [dataset.indices[x] for x in indices]

        return list(
            _traverse_supported_dataset_with_intermediate(
                dataset.dataset,
                values_selector,
                intermediate_selector=intermediate_selector,
                indices=indices,
                intermediate=intermediate,
            )
        )

    if isinstance(dataset, FlatData) and dataset._indices is not None:
        if indices is None:
            indices = [dataset._indices[x] for x in range(len(dataset))]
        else:
            indices = [dataset._indices[x] for x in indices]

    if isinstance(dataset, (ConcatDataset, FlatData)):
        result: List[Y] = []

        concatenated_datasets: Sequence[TraverseT]
        if isinstance(dataset, ConcatDataset):
            concatenated_datasets = dataset.datasets
        else:
            concatenated_datasets = dataset._datasets

        if indices is None:
            for c_dataset in concatenated_datasets:
                result += list(
                    _traverse_supported_dataset_with_intermediate(
                        c_dataset,
                        values_selector,
                        intermediate_selector=intermediate_selector,
                        indices=indices,
                        intermediate=intermediate,
                    )
                )
            if len(result) == 0:
                raise RuntimeError("Empty result")
            return result

        datasets_to_indexes = defaultdict(list)
        indexes_to_dataset = []
        datasets_len = []
        recursion_result = []

        all_size = 0
        for c_dataset in concatenated_datasets:
            len_dataset = len(c_dataset)
            datasets_len.append(len_dataset)
            all_size += len_dataset

        for subset_idx in indices:
            dataset_idx, pattern_idx = find_list_from_index(
                subset_idx, datasets_len, all_size
            )
            datasets_to_indexes[dataset_idx].append(pattern_idx)
            indexes_to_dataset.append(dataset_idx)

        for dataset_idx, c_dataset in enumerate(concatenated_datasets):
            recursion_result.append(
                deque(
                    _traverse_supported_dataset_with_intermediate(
                        c_dataset,
                        values_selector,
                        intermediate_selector=intermediate_selector,
                        indices=datasets_to_indexes[dataset_idx],
                        intermediate=intermediate,
                    )
                )
            )

        result = []
        for idx in range(len(indices)):
            dataset_idx = indexes_to_dataset[idx]
            result.append(recursion_result[dataset_idx].popleft())

        if len(result) == 0:
            raise RuntimeError("Empty result")
        return result

    raise ValueError("Error: can't find the needed data in the given dataset")


def _extract_transforms_from_standard_dataset(dataset):
    if hasattr(dataset, "transforms"):
        # Has torchvision >= v0.3.0 transforms
        # Ignore transform and target_transform
        transforms = getattr(dataset, "transforms")
        if isinstance(transforms, StandardTransform):
            if (
                transforms.transform is not None
                or transforms.target_transform is not None
            ):
                return TupleTransform(
                    [transforms.transform, transforms.target_transform]
                )
    elif hasattr(dataset, "transform") or hasattr(dataset, "target_transform"):
        return TupleTransform(
            [getattr(dataset, "transform"), getattr(dataset, "target_transform")]
        )

    return None


def leaf_datasets(dataset: TraverseT, include_leaf_transforms: bool = True):
    """
    Obtains the leaf datasets of a Dataset.

    This is a low level utility. For most use cases, it is better to use
    :func:`single_flat_dataset` or :func:`flat_datasets_from_benchmark`.

    :param dataset: The dataset to traverse.
    :param include_leaf_transforms: If True, include the transformations
        found in the leaf dataset in the transforms list. Defaults to True.
    :return: A dictionary mapping each leaf dataset to a list of tuples.
        Each tuple contains two elements: the index and the transformation
        applied to that exemplar.
    """

    def leaf_selector(subset, indices, transforms):
        if isinstance(subset, (AvalancheDataset, FlatData, Subset, ConcatDataset)):
            # Returning None => continue traversing
            return None

        if indices is None:
            indices = range(len(subset))

        if include_leaf_transforms:
            leaf_transforms = _extract_transforms_from_standard_dataset(subset)

            if leaf_transforms is not None:
                transforms = list(transforms) + [leaf_transforms]

        return [(subset, idx, transforms) for idx in indices]

    def transform_selector(subset, transforms):
        if isinstance(subset, _FlatDataWithTransform):
            if subset._frozen_transform_groups is not None and not isinstance(
                subset._frozen_transform_groups, EmptyTransformGroups
            ):
                transforms = list(transforms) + [
                    subset._frozen_transform_groups[
                        subset._frozen_transform_groups.current_group
                    ]
                ]
            if subset._transform_groups is not None and not isinstance(
                subset._transform_groups, EmptyTransformGroups
            ):
                transforms = list(transforms) + [
                    subset._transform_groups[subset._transform_groups.current_group]
                ]

        return transforms

    leaves = _traverse_supported_dataset_with_intermediate(
        dataset,
        leaf_selector,
        intermediate_selector=transform_selector,
        intermediate=[],
    )

    leaves_dict: Dict[Any, List[Tuple[int, Any]]] = defaultdict(list)
    for leaf_dataset, idx, transform in leaves:
        transform_reversed = list(reversed(transform))
        leaves_dict[leaf_dataset].append((idx, transform_reversed))

    return leaves_dict


def single_flat_dataset(dataset, include_leaf_transforms: bool = True):
    """
    Obtains the single leaf dataset of a Dataset.

    A dataset can be traversed and flattened to a single leaf dataset
    if all subset and dataset concatenations point to a single leaf
    dataset and if transformations are the same across all paths.

    :param dataset: The dataset to traverse.
    :param include_leaf_transforms: If True, include the transformations
        found in the leaf dataset in the transforms list. Defaults to True.
    :return: A tuple containing three elements: the dataset, the list of
        indices, and the list of transformations. If the dataset cannot
        be flattened to a single dataset, None is returned.
    """
    leaves_dict = leaf_datasets(
        dataset, include_leaf_transforms=include_leaf_transforms
    )
    if len(leaves_dict) != 1:
        return None

    # Obtain the single dataset element
    dataset = list(leaves_dict.keys())[0]
    indices_and_transforms = list(leaves_dict.values())[0]

    # Check that all transforms are the same
    first_transform = indices_and_transforms[0][1]
    same_transforms = all([first_transform == t for _, t in indices_and_transforms])

    if not same_transforms:
        return None

    flat_indices = [i for i, _ in indices_and_transforms]

    return dataset, flat_indices, first_transform


__all__ = [
    "dataset_list_from_benchmark",
    "flat_datasets_from_benchmark",
    "leaf_datasets",
    "single_flat_dataset",
]
