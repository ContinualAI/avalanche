################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This module contains the implementation of the Avalanche Dataset,
which is the standard Avalanche implementation of a PyTorch dataset. Despite
being a child class of the PyTorch Dataset, the AvalancheDataset (and its
derivatives) is much more powerful as it offers many more features
out-of-the-box.
"""
import copy
from collections import OrderedDict, defaultdict, deque

from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset, Subset, ConcatDataset, TensorDataset

from avalanche.benchmarks.utils.dataset_definitions import IDataset
from .dataset_utils import (
    SequenceDataset,
    ClassificationSubset,
    LazyConcatIntTargets,
    find_list_from_index,
    ConstantSequence,
    LazyClassMapping,
    SubSequence,
    LazyConcatTargets,
)
from .dataset_definitions import (
    ITensorDataset,
    ClassificationDataset,
    IDatasetWithTargets,
    ISupportedClassificationDataset,
)

from typing import (
    List,
    Any,
    Sequence,
    Union,
    Optional,
    TypeVar,
    SupportsInt,
    Callable,
    Dict,
    Tuple,
    Collection,
)

from .transforms import AvalancheTransform, EmptyTransform

T_co = TypeVar("T_co", covariant=True)
TTargetType = TypeVar("TTargetType")

TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")


class DataAttribute:
    """Data attributes manage sample-wise information such as task or
    class labels.

    """
    def __init__(self, info: Union[Tensor, ConstantSequence]):
        self.info = info
        self._optimize_sequence()
        self._make_task_set_dict()

    def _optimize_sequence(self):
        if len(self.info) == 0 or isinstance(self.info, ConstantSequence):
            return
        if isinstance(self.info, list):
            return

        return list(self.info)

    def _make_task_set_dict(self) -> Dict[int, "AvalancheDataset"]:
        task_dict = _TaskSubsetDict()
        for task_id in sorted(self.tasks_pattern_indices.keys()):
            task_indices = self.tasks_pattern_indices[task_id]
            task_dict[task_id] = (self, task_indices)

        return task_dict


class AvalancheDataset(Dataset[T_co]):
    """Avalanche Dataset.

    This class extends pytorch Datasets with some additional functionality:
    - separate train/eval transformation groups
    - support for sample attributes such as class targets and task labels

    This dataset can also be used to apply several advanced operations involving
    transformations. For instance, it allows the user to add and replace
    transformations, freeze them so that they can't be changed, etc.

    This dataset also allows the user to keep distinct transformations groups.
    Simply put, a transformation group is a pair of transform+target_transform
    (exactly as in torchvision datasets). This dataset natively supports keeping
    two transformation groups: the first, 'train', contains transformations
    applied to training patterns. Those transformations usually involve some
    kind of data augmentation. The second one is 'eval', that will contain
    transformations applied to test patterns. Having both groups can be
    useful when, for instance, in need to test on the training data (as this
    process usually involves removing data augmentation operations). Switching
    between transformations can be easily achieved by using the
    :func:`train` and :func:`eval` methods.

    Moreover, arbitrary transformation groups can be added and used. For more
    info see the constructor and the :func:`with_transforms` method.

    This dataset will try to inherit the task labels from the input
    dataset. If none are available and none are given via the `task_labels`
    parameter, each pattern will be assigned a default task label "0".
    See the constructor for more details.
    """

    def __init__(
        self,
        dataset: IDataset,
        *,
        transform_groups: AvalancheTransform = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates a ``AvalancheDataset`` instance.

        :param dataset: Original dataset. Beware that
            AvalancheDataset will not overwrite transformations already
            applied by this dataset.
        :param transform_groups: Avalanche transform groups.
        """
        self._dataset = dataset
        """
        The original dataset.
        """

        self.transform_groups = transform_groups
        self.collate_fn = collate_fn

        self.collate_fn = self._initialize_collate_fn(
            dataset, collate_fn
        )
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """

    def __add__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([self, other])

    def __radd__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([other, self])

    def __getitem__(self, idx) -> Union[T_co, Sequence[T_co]]:
        element = self._dataset[idx]
        element = self.transform_groups(element)
        return element

    def __len__(self):
        return len(self._dataset)

    def train(self):
        """Returns a new dataset with the transformations of the 'train' group
        loaded.

        The current dataset will not be affected.

        :return: A new dataset with the training transformations loaded.
        """
        return self.with_transforms("train")

    def eval(self):
        """
        Returns a new dataset with the transformations of the 'eval' group
        loaded.

        Eval transformations usually don't contain augmentation procedures.
        This function may be useful when in need to test on training data
        (for instance, in order to run a validation pass).

        The current dataset will not be affected.

        :return: A new dataset with the eval transformations loaded.
        """
        return self.with_transforms("eval")

    def with_transforms(
        self: TAvalancheDataset, group_name: str
    ) -> TAvalancheDataset:
        """
        Returns a new dataset with the transformations of a different group
        loaded.

        The current dataset will not be affected.

        :param group_name: The name of the transformations group to use.
        :return: A new dataset with the new transformations.
        """
        datacopy = self._clone_dataset()
        datacopy.transform_groups.with_transform(group_name)
        if isinstance(self._dataset, AvalancheDataset):
            datacopy._dataset = datacopy._dataset.with_transform_group(group_name)
        return datacopy

    def _clone_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = copy.copy(self)
        dataset_copy.transform_groups = copy.copy(dataset_copy.transform_groups)
        return dataset_copy

    def _initialize_collate_fn(self, dataset, collate_fn):
        if collate_fn is not None:
            return collate_fn

        if hasattr(dataset, "collate_fn"):
            return getattr(dataset, "collate_fn")

        return default_collate


class AvalancheSubset(AvalancheDataset[T_co]):
    """Avalanche Dataset that behaves like a PyTorch
    :class:`torch.utils.data.Subset`.

    See :class:`AvalancheDataset` for more details.
    """
    def __init__(
        self,
        dataset: IDataset,
        indices: Sequence[int],
        *,
        transform_groups: AvalancheTransform = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates an ``AvalancheSubset`` instance.

        :param dataset: The whole dataset.
        :param indices: Indices in the whole set selected for subset. Can
            be None, which means that the whole dataset will be returned.
        """
        subset = Subset(dataset, indices=indices)
        super().__init__(subset, transform_groups, collate_fn)
        self._original_dataset = dataset
        self._indices = indices
        self._flatten_dataset()

    def _flatten_dataset(self):
        """Flattens this subset by borrowing indices from the original dataset
        (if it's an AvalancheSubset or PyTorch Subset)"""

        if isinstance(self._original_dataset, AvalancheSubset):
            # we need to take trasnforms and indices from parent
            self.transform_groups = copy.copy(self._original_dataset.transform_groups)

            grandparent_data = self._original_dataset._original_dataset
            self._original_dataset = grandparent_data

            parent_idxs = self._original_dataset._indices
            new_indices = [parent_idxs[x] for x in self._indices]
            self._indices = new_indices

            self._dataset = Subset(grandparent_data, indices=new_indices)


class AvalancheTensorDataset(AvalancheDataset[T_co]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset from PyTorch,
    this Dataset also supports transformations, slicing, advanced indexing,
    the targets field and all the other goodies listed in
    :class:`AvalancheDataset`.
    """

    def __init__(
        self,
        *dataset_tensors: Sequence[Tensor],
        transform_groups: AvalancheTransform = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """
        Creates a ``AvalancheTensorDataset`` instance.

        :param dataset_tensors: Sequences, Tensors or ndarrays representing the
            content of the dataset.
        """

        if len(dataset_tensors) < 1:
            raise ValueError("At least one sequence must be passed")

        super().__init__(
            TensorDataset(*dataset_tensors),
            transform_groups=transform_groups,
            collate_fn=collate_fn
        )


class AvalancheConcatDataset(AvalancheDataset[T_co]):
    """
    A Dataset that behaves like a PyTorch
    :class:`torch.utils.data.ConcatDataset`. However, this Dataset also supports
    transformations, slicing, advanced indexing and the targets field and all
    the other goodies listed in :class:`AvalancheDataset`.

    This dataset guarantees that the operations involving the transformations
    and transformations groups are consistent across the concatenated dataset
    (if they are subclasses of :class:`AvalancheDataset`).
    """

    def __init__(
        self,
        datasets: Collection[IDataset],
        *,
        transform_groups: AvalancheTransform = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates a ``AvalancheConcatDataset`` instance.

        :param datasets: A collection of datasets.
        """
        dataset_list = list(datasets)
        self.datasets = dataset_list
        self._flatten_dataset()

        self._datasets_lengths = [len(dataset) for dataset in dataset_list]
        self.cumulative_sizes = ConcatDataset.cumsum(dataset_list)
        self._total_length = sum(self._datasets_lengths)

        super().__init__(
            ClassificationDataset(),  # not used
            transform_groups=transform_groups,
            collate_fn=collate_fn
        )

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, idx: int):
        element = ConcatDataset.__getitem__(self, idx)
        element = self.transform_groups(element)
        return element

    def _clone_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = super()._clone_dataset()
        dataset_copy.datasets = list(dataset_copy.datasets)
        return dataset_copy

    def _flatten_dataset(self):
        # Flattens this subset by borrowing the list of concatenated datasets
        # from the original datasets (if they're 'AvalancheConcatSubset's or
        # PyTorch 'ConcatDataset's)

        flattened_list = []
        for dataset in self.datasets:
            if isinstance(dataset, AvalancheConcatDataset):
                if isinstance(dataset.transform_groups, EmptyTransform):
                    flattened_list.extend(dataset.datasets)
                else:
                    # Can't flatten as the dataset has custom transformations
                    flattened_list.append(dataset)
            elif isinstance(dataset, AvalancheSubset):
                flattened_list += self._flatten_subset_concat_branch(dataset)
            else:
                flattened_list.append(dataset)
        self.datasets = flattened_list

    def _flatten_subset_concat_branch(
        self, dataset: AvalancheSubset
    ) -> List[Dataset]:
        """Optimizes the dataset hierarchy in the corner case:

        self -> [Subset, Subset, ] -> ConcatDataset -> [Dataset]

        :param dataset: The dataset. This function returns [dataset] if the
            dataset is not a subset containing a concat dataset (or if other
            corner cases are encountered).
        :return: The flattened list of datasets to be concatenated.
        """
        if not isinstance(dataset._original_dataset, AvalancheConcatDataset):
            return [dataset]

        concat_dataset: AvalancheConcatDataset = dataset._original_dataset
        if concat_dataset._has_own_transformations():
            # The dataset has custom transforms -> do nothing
            return [dataset]

        result: List[AvalancheSubset] = []
        last_c_dataset = None
        last_c_idxs = []
        last_c_targets = []
        last_c_tasks = []
        for subset_idx, idx in enumerate(dataset._indices):
            dataset_idx, internal_idx = find_list_from_index(
                idx,
                concat_dataset._datasets_lengths,
                concat_dataset._total_length,
                cumulative_sizes=concat_dataset.cumulative_sizes,
            )

            if last_c_dataset is None:
                last_c_dataset = dataset_idx
            elif last_c_dataset != dataset_idx:
                # Consolidate current subset
                result.append(
                    AvalancheConcatDataset._make_similar_subset(
                        dataset,
                        concat_dataset.datasets[last_c_dataset],
                        last_c_idxs,
                        last_c_targets,
                        last_c_tasks,
                    )
                )

                # Switch to next dataset
                last_c_dataset = dataset_idx
                last_c_idxs = []
                last_c_targets = []
                last_c_tasks = []

            last_c_idxs.append(internal_idx)
            last_c_targets.append(dataset.targets[subset_idx])
            last_c_tasks.append(dataset.targets_task_labels[subset_idx])

        if last_c_dataset is not None:
            result.append(
                AvalancheConcatDataset._make_similar_subset(
                    dataset,
                    concat_dataset.datasets[last_c_dataset],
                    last_c_idxs,
                    last_c_targets,
                    last_c_tasks,
                )
            )

        return result

    @staticmethod
    def _make_similar_subset(subset, ref_dataset, indices, targets, tasks):
        t_groups = dict()
        f_groups = dict()
        AvalancheDataset._borrow_transformations(subset, t_groups, f_groups)

        collate_fn = None
        if hasattr(subset, "collate_fn"):
            collate_fn = subset.collate_fn

        result = AvalancheSubset(
            ref_dataset,
            indices=indices,
            class_mapping=subset._class_mapping,
            transform_groups=t_groups,
            initial_transform_group=subset.current_transform_group,
            task_labels=tasks,
            targets=targets,
            collate_fn=collate_fn,
        )

        result._frozen_transforms = f_groups
        return result


def concat_datasets_sequentially(
    train_dataset_list: Sequence[ISupportedClassificationDataset],
    test_dataset_list: Sequence[ISupportedClassificationDataset],
) -> Tuple[AvalancheConcatDataset, AvalancheConcatDataset, List[list]]:
    """
    Concatenates a list of datasets. This is completely different from
    :class:`ConcatDataset`, in which datasets are merged together without
    other processing. Instead, this function re-maps the datasets class IDs.
    For instance:
    let the dataset[0] contain patterns of 3 different classes,
    let the dataset[1] contain patterns of 2 different classes, then class IDs
    will be mapped as follows:

    dataset[0] class "0" -> new class ID is "0"

    dataset[0] class "1" -> new class ID is "1"

    dataset[0] class "2" -> new class ID is "2"

    dataset[1] class "0" -> new class ID is "3"

    dataset[1] class "1" -> new class ID is "4"

    ... -> ...

    dataset[-1] class "C-1" -> new class ID is "overall_n_classes-1"

    In contrast, using PyTorch ConcatDataset:

    dataset[0] class "0" -> ID is "0"

    dataset[0] class "1" -> ID is "1"

    dataset[0] class "2" -> ID is "2"

    dataset[1] class "0" -> ID is "0"

    dataset[1] class "1" -> ID is "1"

    Note: ``train_dataset_list`` and ``test_dataset_list`` must have the same
    number of datasets.

    :param train_dataset_list: A list of training datasets
    :param test_dataset_list: A list of test datasets

    :returns: A concatenated dataset.
    """
    remapped_train_datasets = []
    remapped_test_datasets = []
    next_remapped_idx = 0

    # Obtain the number of classes of each dataset
    classes_per_dataset = [
        _count_unique(
            train_dataset_list[dataset_idx].targets,
            test_dataset_list[dataset_idx].targets,
        )
        for dataset_idx in range(len(train_dataset_list))
    ]

    new_class_ids_per_dataset = []
    for dataset_idx in range(len(train_dataset_list)):

        # Get the train and test sets of the dataset
        train_set = train_dataset_list[dataset_idx]
        test_set = test_dataset_list[dataset_idx]

        # Get the classes in the dataset
        dataset_classes = set(map(int, train_set.targets))

        # The class IDs for this dataset will be in range
        # [n_classes_in_previous_datasets,
        #       n_classes_in_previous_datasets + classes_in_this_dataset)
        new_classes = list(
            range(
                next_remapped_idx,
                next_remapped_idx + classes_per_dataset[dataset_idx],
            )
        )
        new_class_ids_per_dataset.append(new_classes)

        # AvalancheSubset is used to apply the class IDs transformation.
        # Remember, the class_mapping parameter must be a list in which:
        # new_class_id = class_mapping[original_class_id]
        # Hence, a list of size equal to the maximum class index is created
        # Only elements corresponding to the present classes are remapped
        class_mapping = [-1] * (max(dataset_classes) + 1)
        j = 0
        for i in dataset_classes:
            class_mapping[i] = new_classes[j]
            j += 1

        # Create remapped datasets and append them to the final list
        remapped_train_datasets.append(
            AvalancheSubset(train_set, class_mapping=class_mapping)
        )
        remapped_test_datasets.append(
            AvalancheSubset(test_set, class_mapping=class_mapping)
        )
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return (
        AvalancheConcatDataset(remapped_train_datasets),
        AvalancheConcatDataset(remapped_test_datasets),
        new_class_ids_per_dataset,
    )


def as_avalanche_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, TTargetType]:
    if isinstance(dataset, AvalancheDataset):
        return dataset

    return AvalancheDataset(dataset)


def as_classification_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, int]:
    return as_avalanche_dataset(
        dataset
    )


def as_regression_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, Any]:
    return as_avalanche_dataset(
        dataset
    )


def as_segmentation_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, Any]:
    return as_avalanche_dataset(
        dataset
    )


def as_undefined_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, Any]:
    return as_avalanche_dataset(
        dataset
    )


def train_eval_avalanche_datasets(
    train_dataset: ISupportedClassificationDataset,
    test_dataset: ISupportedClassificationDataset,
    train_transformation,
    eval_transformation,
):
    train = AvalancheDataset(
        train_dataset,
        transform_groups=dict(
            train=(train_transformation, None), eval=(eval_transformation, None)
        ),
        initial_transform_group="train"
    )

    test = AvalancheDataset(
        test_dataset,
        transform_groups=dict(
            train=(train_transformation, None), eval=(eval_transformation, None)
        ),
        initial_transform_group="eval"
    )
    return train, test


def _traverse_supported_dataset(
    dataset, values_selector: Callable[[Dataset, List[int]], List], indices=None
) -> List:
    initial_error = None
    try:
        result = values_selector(dataset, indices)
        if result is not None:
            return result
    except BaseException as e:
        initial_error = e

    if isinstance(dataset, Subset):
        if indices is None:
            indices = range(len(dataset))
        indices = [dataset.indices[x] for x in indices]
        return list(
            _traverse_supported_dataset(
                dataset.dataset, values_selector, indices
            )
        )

    if isinstance(dataset, ConcatDataset):
        result = []
        if indices is None:
            for c_dataset in dataset.datasets:
                result += list(
                    _traverse_supported_dataset(
                        c_dataset, values_selector, indices
                    )
                )
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


def _count_unique(*sequences: Sequence[SupportsInt]):
    uniques = set()

    for seq in sequences:
        for x in seq:
            uniques.add(int(x))

    return len(uniques)


def _select_targets(dataset, indices):
    if hasattr(dataset, "targets"):
        # Standard supported dataset
        found_targets = dataset.targets
    elif hasattr(dataset, "tensors"):
        # Support for PyTorch TensorDataset
        if len(dataset.tensors) < 2:
            raise ValueError(
                "Tensor dataset has not enough tensors: "
                "at least 2 are required."
            )
        found_targets = dataset.tensors[1]
    else:
        raise ValueError(
            "Unsupported dataset: must have a valid targets field "
            "or has to be a Tensor Dataset with at least 2 "
            "Tensors"
        )

    if indices is not None:
        found_targets = SubSequence(found_targets, indices=indices)

    return found_targets


def _select_task_labels(dataset, indices):
    found_task_labels = None
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


def _make_target_from_supported_dataset(
    dataset: SupportedDataset, converter: Callable[[Any], TTargetType] = None
) -> Sequence[TTargetType]:
    if isinstance(dataset, AvalancheDataset):
        if converter is None:
            return dataset.targets
        elif (
            isinstance(dataset.targets, (SubSequence, LazyConcatTargets))
            and dataset.targets.converter == converter
        ):
            return dataset.targets
        elif isinstance(dataset.targets, LazyClassMapping) and converter == int:
            # LazyClassMapping already outputs int targets
            return dataset.targets

    targets = _traverse_supported_dataset(dataset, _select_targets)

    return SubSequence(targets, converter=converter)


def _make_task_labels_from_supported_dataset(
    dataset: SupportedDataset,
) -> Sequence[int]:
    if isinstance(dataset, AvalancheDataset):
        return dataset.targets_task_labels

    task_labels = _traverse_supported_dataset(dataset, _select_task_labels)

    return SubSequence(task_labels, converter=int)


__all__ = [
    "SupportedDataset",
    "AvalancheDataset",
    "AvalancheSubset",
    "AvalancheTensorDataset",
    "AvalancheConcatDataset",
    "concat_datasets_sequentially",
    "as_avalanche_dataset",
    "as_classification_dataset",
    "as_regression_dataset",
    "as_segmentation_dataset",
    "as_undefined_dataset",
    "train_eval_avalanche_datasets",
]
