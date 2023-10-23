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

""" Common benchmarks/environments utils. """

from collections import OrderedDict, defaultdict, deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    Dict,
    SupportsInt,
)
import warnings

import torch
from torch import Tensor
from torch.utils.data import Subset, ConcatDataset

from .data import AvalancheDataset
from .data_attribute import DataAttribute
from .dataset_definitions import (
    ISupportedClassificationDataset,
)
from .dataset_utils import (
    SubSequence,
    find_list_from_index,
)
from .flat_data import ConstantSequence
from .transform_groups import (
    TransformGroupDef,
    TransformGroups,
    XTransform,
    YTransform,
)

if TYPE_CHECKING:
    from .classification_dataset import TaskAwareClassificationDataset

T_co = TypeVar("T_co", covariant=True)
TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")


def tensor_as_list(sequence) -> List:
    # Numpy: list(np.array([1, 2, 3])) returns [1, 2, 3]
    # whereas: list(torch.tensor([1, 2, 3])) returns ->
    # -> [tensor(1), tensor(2), tensor(3)]
    #
    # This is why we have to handle Tensor in a different way
    if isinstance(sequence, list):
        return sequence
    if not isinstance(sequence, Iterable):
        return [sequence]
    if isinstance(sequence, Tensor):
        return sequence.tolist()
    return list(sequence)


def _indexes_grouped_by_classes(
    targets: Sequence[int],
    patterns_indexes: Union[None, Sequence[int]],
    sort_indexes: bool = True,
    sort_classes: bool = True,
) -> Union[List[int], None]:
    result_per_class: Dict[int, List[int]] = OrderedDict()
    result: List[int] = []

    indexes_was_none = patterns_indexes is None

    if patterns_indexes is not None:
        patterns_indexes = tensor_as_list(patterns_indexes)
    else:
        patterns_indexes = list(range(len(targets)))

    targets = tensor_as_list(targets)

    # Consider that result_per_class is an OrderedDict
    # This means that, if sort_classes is True, the next for statement
    # will initialize "result_per_class" in sorted order which in turn means
    # that patterns will be ordered by ascending class ID.
    classes = torch.unique(torch.as_tensor(targets), sorted=sort_classes).tolist()

    for class_id in classes:
        result_per_class[class_id] = []

    # Stores each pattern index in the appropriate class list
    for idx in patterns_indexes:
        result_per_class[targets[idx]].append(idx)

    # Concatenate all the pattern indexes
    for class_id in classes:
        if sort_indexes:
            result_per_class[class_id].sort()
        result.extend(result_per_class[class_id])

    if result == patterns_indexes and indexes_was_none:
        # Result is [0, 1, 2, ..., N] and patterns_indexes was originally None
        # This means that the user tried to obtain a full Dataset
        # (indexes_was_none) only ordered according to the sort_indexes and
        # sort_classes parameters. However, sort_indexes+sort_classes returned
        # the plain pattern sequence as it already is. So the original Dataset
        # already satisfies the sort_indexes+sort_classes constraints.
        # By returning None, we communicate that the Dataset can be taken as-is.
        return None

    return result


def grouped_and_ordered_indexes(
    targets: Sequence[int],
    patterns_indexes: Union[None, Sequence[int]],
    bucket_classes: bool = True,
    sort_classes: bool = False,
    sort_indexes: bool = False,
) -> Union[List[int], None]:
    """
    Given the targets list of a dataset and the patterns to include, returns the
    pattern indexes sorted according to the ``bucket_classes``,
    ``sort_classes`` and ``sort_indexes`` parameters.

    :param targets: The list of pattern targets, as a list.
    :param patterns_indexes: A list of pattern indexes to include in the set.
        If None, all patterns will be included.
    :param bucket_classes: If True, pattern indexes will be returned so that
        patterns will be grouped by class. Defaults to True.
    :param sort_classes: If both ``bucket_classes`` and ``sort_classes`` are
        True, class groups will be sorted by class index. Ignored if
        ``bucket_classes`` is False. Defaults to False.
    :param sort_indexes: If True, patterns indexes will be sorted. When
        bucketing by class, patterns will be sorted inside their buckets.
        Defaults to False.

    :returns: The list of pattern indexes sorted according to the
        ``bucket_classes``, ``sort_classes`` and ``sort_indexes`` parameters or
        None if the patterns_indexes is None and the whole dataset can be taken
        using the existing patterns order.
    """
    if bucket_classes:
        return _indexes_grouped_by_classes(
            targets,
            patterns_indexes,
            sort_indexes=sort_indexes,
            sort_classes=sort_classes,
        )

    if patterns_indexes is None:
        # No grouping and sub-set creation required... just return None
        return None
    if not sort_indexes:
        # No sorting required, just return patterns_indexes
        return tensor_as_list(patterns_indexes)

    # We are here only because patterns_indexes != None and sort_indexes is True
    patterns_indexes = tensor_as_list(patterns_indexes)
    result = list(patterns_indexes)  # Make sure we're working on a copy
    result.sort()
    return result


def as_avalanche_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset:
    if isinstance(dataset, AvalancheDataset):
        return dataset
    return AvalancheDataset([dataset])


def as_classification_dataset(
    dataset: ISupportedClassificationDataset[T_co],
    transform_groups: Optional[TransformGroups] = None,
) -> "TaskAwareClassificationDataset":
    """Converts a dataset with a `targets` field into an Avalanche ClassificationDataset."""
    from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset

    if isinstance(dataset, ClassificationDataset):
        return dataset
    da = DataAttribute(dataset.targets, "targets")
    return ClassificationDataset(
        [dataset], transform_groups=transform_groups, data_attributes=[da]
    )


def as_taskaware_classification_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> "TaskAwareClassificationDataset":
    from avalanche.benchmarks.utils.classification_dataset import (
        TaskAwareClassificationDataset,
    )

    if isinstance(dataset, TaskAwareClassificationDataset):
        return dataset
    return TaskAwareClassificationDataset([dataset])


def _count_unique(*sequences: Sequence[SupportsInt]):
    uniques = set()

    for seq in sequences:
        for x in seq:
            uniques.add(int(x))

    return len(uniques)


def concat_datasets(datasets):
    """Concatenates a list of datasets."""
    if len(datasets) == 0:
        return AvalancheDataset([])
    res = datasets[0]
    if not isinstance(res, AvalancheDataset):
        res = AvalancheDataset([res])

    for d in datasets[1:]:
        if not isinstance(d, AvalancheDataset):
            d = AvalancheDataset([d])
        res = res.concat(d)
    return res


def find_common_transforms_group(
    datasets: Iterable[Any], default_group: str = "train"
) -> str:
    """
    Utility used to find the common transformations group across multiple
    datasets.

    To compute the common group, the current one is used. Objects which are not
    instances of :class:`AvalancheDataset` are ignored.
    If no common group is found, then the default one is returned.

    :param datasets: The list of datasets.
    :param default_group: The name of the default group.
    :returns: The name of the common group.
    """
    # Find common "current_group" or use "train"
    uniform_group: Optional[str] = None
    for d_set in datasets:
        if isinstance(d_set, AvalancheDataset):
            if uniform_group is None:
                uniform_group = d_set._flat_data._transform_groups.current_group
            else:
                if uniform_group != d_set._flat_data._transform_groups.current_group:
                    uniform_group = None
                    break

    if uniform_group is None:
        initial_transform_group = default_group
    else:
        initial_transform_group = uniform_group

    return initial_transform_group


Y = TypeVar("Y")
T = TypeVar("T")


def _traverse_supported_dataset(
    dataset: Y,
    values_selector: Callable[[Y, Optional[List[int]]], Optional[Sequence[T]]],
    indices: Optional[List[int]] = None,
) -> Sequence[T]:
    """
    Traverse the given dataset by gathering required info.

    The given dataset is traversed by covering all sub-datasets
    contained PyTorch :class:`Subset` and :class`ConcatDataset`.
    Beware that instances of :class:`AvalancheDataset` will not
    be traversed as those objects already have the proper data
    attribute fields populated with data from leaf datasets.

    For each dataset, the `values_selector` will be called to gather
    the required information. The values returned by the given selector
    are then concatenated to create a final list of values.

    :param dataset: The dataset to traverse.
    :param values_selector: A function that, given the dataset
        and the indices to consider (which may be None if the entire
        dataset must be considered), returns a list of selected values.
    :returns: The list of selected values.
    """
    initial_error = None
    try:
        result = values_selector(dataset, indices)
        if result is not None:
            return result
    except BaseException as e:
        initial_error = e

    if isinstance(dataset, Subset):
        if indices is None:
            indices = [dataset.indices[x] for x in range(len(dataset))]
        else:
            indices = [dataset.indices[x] for x in indices]

        return list(
            _traverse_supported_dataset(dataset.dataset, values_selector, indices)
        )

    if isinstance(dataset, ConcatDataset):
        result = []
        if indices is None:
            for c_dataset in dataset.datasets:
                result_data = _traverse_supported_dataset(c_dataset, values_selector)
                if isinstance(result_data, Tensor):
                    result += result_data.tolist()
                else:
                    result += list(result_data)
            return result

        datasets_to_indexes = defaultdict(list)
        indexes_to_dataset = []
        datasets_len = []
        recursion_result = []

        all_size = 0
        for c_dataset in dataset.datasets:
            len_dataset = len(c_dataset)
            datasets_len.append(len_dataset)
            all_size += len_dataset

        for subset_idx in indices:
            dataset_idx, pattern_idx = find_list_from_index(
                subset_idx, datasets_len, all_size
            )
            datasets_to_indexes[dataset_idx].append(pattern_idx)
            indexes_to_dataset.append(dataset_idx)

        for dataset_idx, c_dataset in enumerate(dataset.datasets):
            recursion_result.append(
                deque(
                    _traverse_supported_dataset(
                        c_dataset,
                        values_selector,
                        datasets_to_indexes[dataset_idx],
                    )
                )
            )

        result = []
        for idx in range(len(indices)):
            dataset_idx = indexes_to_dataset[idx]
            result.append(recursion_result[dataset_idx].popleft())

        return result

    if initial_error is not None:
        raise initial_error

    raise ValueError("Error: can't find the needed data in the given dataset")


def _init_task_labels(
    dataset, task_labels, check_shape=True
) -> Optional[DataAttribute[int]]:
    """
    Initializes the task label list (one for each pattern in the dataset).

    Precedence is given to the values contained in `task_labels` if passed.
    Otherwisem the elements will be retrieved from the dataset itself by
    traversing it and looking at the `targets_task_labels` field.

    :param dataset: The dataset for which the task labels list must be
        initialized. Ignored if `task_labels` is passed, but it may still be
        used if `check_shape` is true.
    :param task_labels: The task labels to use. May be None, in which case
        the labels will be retrieved from the dataset.
    :param check_shape: If True, will check if the length of the task labels
        list matches the dataset size. Ignored if the labels are retrieved
        from the dataset.
    :returns: A data attribute containing the task labels. May be None to
        signal that the dataset's `targets_task_labels` field should be used
        (because the dataset is a :class:`AvalancheDataset`).
    """
    if task_labels is not None:
        # task_labels has priority over the dataset fields
        if isinstance(task_labels, int):
            task_labels = ConstantSequence(task_labels, len(dataset))
        elif len(task_labels) != len(dataset) and check_shape:
            raise ValueError(
                "Invalid amount of task labels. It must be equal to the "
                "number of patterns in the dataset. Got {}, expected "
                "{}!".format(len(task_labels), len(dataset))
            )

        if isinstance(task_labels, ConstantSequence):
            tls = task_labels
        elif isinstance(task_labels, DataAttribute):
            tls = task_labels.data
        else:
            tls = SubSequence(task_labels, converter=int)
    else:
        task_labels = _traverse_supported_dataset(dataset, _select_task_labels)

        if task_labels is None:
            tls = None
        elif isinstance(task_labels, ConstantSequence):
            tls = task_labels
        elif isinstance(task_labels, DataAttribute):
            return DataAttribute(
                task_labels.data, "targets_task_labels", use_in_getitem=True
            )
        else:
            tls = SubSequence(task_labels, converter=int)

    if tls is None:
        return None
    return DataAttribute(tls, "targets_task_labels", use_in_getitem=True)


def _select_task_labels(
    dataset: Any, indices: Optional[List[int]]
) -> Optional[Sequence[SupportsInt]]:
    """
    Selector function to be passed to :func:`_traverse_supported_dataset`
    to obtain the `targets_task_labels` for the given dataset.

    :param dataset: the traversed dataset.
    :param indices: the indices describing the subset to consider.
    :returns: The list of task labels or None if not found.
    """
    found_task_labels: Optional[Sequence[SupportsInt]] = None
    if hasattr(dataset, "targets_task_labels"):
        found_task_labels = dataset.targets_task_labels

    if found_task_labels is None:
        if isinstance(dataset, (Subset, ConcatDataset)):
            return None  # Continue traversing

    if found_task_labels is None:
        if indices is None:
            return ConstantSequence(0, len(dataset))
        return ConstantSequence(0, len(indices))

    if indices is not None:
        found_task_labels = SubSequence(found_task_labels, indices=indices)

    return found_task_labels


def _init_transform_groups(
    transform_groups: Optional[Mapping[str, TransformGroupDef]],
    transform: Optional[XTransform],
    target_transform: Optional[YTransform],
    initial_transform_group: Optional[str],
    dataset,
) -> Optional[TransformGroups]:
    """
    Initializes the transform groups for the given dataset.

    This internal utility is commonly used to manage the transformation
    defintions coming from the user-facing API. The user may want to
    define transformations in a more classic (and simple) way by
    passing a single `transform`, or in a more elaborate way by
    passing a dictionary of groups (`transform_groups`).

    :param transform_groups: The transform groups to use as a dictionary
        (group_name -> group). Can be None. Mutually exclusive with
        `targets` and `target_transform`
    :param transform: The transformation for the X value. Can be None.
    :param target_transform: The transformation for the Y value. Can be None.
    :param initial_transform_group: The name of the initial group.
        If None, 'train' will be used.
    :param dataset: The avalanche dataset, used only to obtain the name of
        the initial transformations groups if `initial_transform_group` is
        None.
    :returns: a :class:`TransformGroups` instance if any transformation
        was passed, else None.
    """
    if transform_groups is not None and (
        transform is not None or target_transform is not None
    ):
        raise ValueError(
            "transform_groups can't be used with transform"
            "and target_transform values"
        )

    if transform_groups is not None:
        _check_groups_dict_format(transform_groups)

    if initial_transform_group is None:
        # Detect from the input dataset. If not an AvalancheDataset then
        # use 'train' as the initial transform group
        if (
            isinstance(dataset, AvalancheDataset)
            and dataset._flat_data._transform_groups is not None
        ):
            tgs = dataset._flat_data._transform_groups
            initial_transform_group = tgs.current_group
        else:
            initial_transform_group = "train"

    if transform_groups is None:
        if target_transform is None and transform is None:
            tgs = None
        else:
            tgs = TransformGroups(
                {
                    "train": (transform, target_transform),
                    "eval": (transform, target_transform),
                },
                current_group=initial_transform_group,
            )
    else:
        tgs = TransformGroups(transform_groups, current_group=initial_transform_group)
    return tgs


def _check_groups_dict_format(groups_dict):
    # The original groups_dict must be convertible to native Python dict
    groups_dict = dict(groups_dict)

    # Check if the format of the groups is correct
    for map_key in groups_dict:
        if not isinstance(map_key, str):
            raise ValueError(
                "Every group must be identified by a string."
                'Wrong key was: "' + str(map_key) + '"'
            )

    if "test" in groups_dict:
        warnings.warn(
            'A transformation group named "test" has been found. Beware '
            "that by default AvalancheDataset supports test transformations"
            ' through the "eval" group. Consider using that one!'
        )


def _split_user_def_task_label(
    datasets, task_labels: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]]
) -> List[Optional[Union[int, Sequence[int]]]]:
    """
    Given a datasets list and the user-defined list of task labels,
    returns the task labels list of each dataset.

    This internal utility is mainly used to manage the different ways
    in which the user can define the task labels:
    - As a single task label for all exemplars of all datasets
    - A single list of length equal to the sum of the lengths of all datasets
    - A list containing, for each dataset, one element between:
        - a list, defining the task labels of each exemplar of a that dataset
        - an int, defining the task label of all exemplars of a that dataset

    :param datasets: The list of datasets.
    :param task_labels: The user-defined task labels. Can be None, in which
        case a list of None will be returned.
    :returns: A list containing as many elements as the input `datasets`.
        Each element is either a list of task labels or None. If None
        (because `task_labels` is None), this means that the task labels
        should be retrieved by traversing each dataset.
    """
    t_labels = []
    idx_start = 0
    for dd_idx, dd in enumerate(datasets):
        end_idx = idx_start + len(dd)
        dataset_t_label: Optional[Union[int, Sequence[int]]]
        if task_labels is None:
            # No task label set
            dataset_t_label = None
        elif isinstance(task_labels, int):
            # Single integer (same label for all instances)
            dataset_t_label = task_labels
        elif isinstance(task_labels[0], int):
            # Single task labels sequence
            # (to be split across concatenated datasets)
            dataset_t_label = task_labels[idx_start:end_idx]  # type: ignore
        elif len(task_labels[dd_idx]) == len(dd):  # type: ignore
            # One sequence per dataset
            dataset_t_label = task_labels[dd_idx]
        else:
            raise ValueError("The task_labels parameter has an invalid format.")
        t_labels.append(dataset_t_label)

        idx_start = end_idx
    return t_labels


def _split_user_def_targets(
    datasets,
    targets: Optional[Union[Sequence[T], Sequence[Sequence[T]]]],
    single_element_checker: Callable[[Any], bool],
) -> List[Optional[Sequence[T]]]:
    """
    Given a datasets list and the user-defined list of targets,
    returns the targets list of each dataset.

    This internal utility is mainly used to manage the different ways
    in which the user can define the targets:
    - A single list of length equal to the sum of the lengths of all datasets
    - A list containing, for each dataset, a list, defining the targets
        of each exemplar of a that dataset

    :param datasets: The list of datasets.
    :param targets: The user-defined targets. Can be None, in which
        case a list of None will be returned.
    :returns: A list containing as many elements as the input `datasets`.
        Each element is either a list of targets or None. If None
        (because `targets` is None), this means that the targets
        should be retrieved by traversing each dataset.
    """
    t_labels = []
    idx_start = 0
    for dd_idx, dd in enumerate(datasets):
        end_idx = idx_start + len(dd)
        dataset_t_label: Optional[Sequence[T]]
        if targets is None:
            # No targets set
            dataset_t_label = None
        elif single_element_checker(targets[0]):
            # Single targets sequence
            # (to be split across concatenated datasets)
            dataset_t_label = targets[idx_start:end_idx]  # type: ignore
        elif len(targets[dd_idx]) == len(dd):  # type: ignore
            # One sequence per dataset
            dataset_t_label = targets[dd_idx]  # type: ignore
        else:
            raise ValueError("The targets parameter has an invalid format.")
        t_labels.append(dataset_t_label)

        idx_start = end_idx
    return t_labels


class TaskSet(Mapping[int, TAvalancheDataset], Generic[TAvalancheDataset]):
    """A lazy mapping for <task-label -> task dataset>.

    Given an `AvalancheClassificationDataset`, this class provides an
    iterator that splits the data into task subsets, returning tuples
    `<task_id, task_dataset>`.

    Usage:

    .. code-block:: python

        tset = TaskSet(data)
        for tid, tdata in tset:
            print(f"task {tid} has {len(tdata)} examples.")

    """

    def __init__(self, data: TAvalancheDataset):
        """Constructor.

        :param data: original data
        """
        super().__init__()
        self.data: TAvalancheDataset = data

    def __iter__(self) -> Iterator[int]:
        t_labels = self._get_task_labels_field()
        return iter(t_labels.uniques)

    def __getitem__(self, task_label: int):
        t_labels = self._get_task_labels_field()
        tl_idx = t_labels.val_to_idx[task_label]
        return self.data.subset(tl_idx)

    def __len__(self) -> int:
        t_labels = self._get_task_labels_field()
        return len(t_labels.uniques)

    def _get_task_labels_field(self) -> DataAttribute[int]:
        return self.data.targets_task_labels  # type: ignore


__all__ = [
    "tensor_as_list",
    "grouped_and_ordered_indexes",
    "as_avalanche_dataset",
    "as_classification_dataset",
    "as_taskaware_classification_dataset",
    "concat_datasets",
    "find_common_transforms_group",
    "TaskSet",
]
