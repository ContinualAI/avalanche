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
This module contains the implementation of the ``ClassificationDataset``,
which is the dataset used for supervised continual learning benchmarks.
ClassificationDatasets are ``AvalancheDatasets`` that manage class and task
labels automatically. Concatenation and subsampling operations are optimized
to be used frequently, as is common in replay strategies.
"""

from functools import partial
import torch
from torch.utils.data.dataset import Subset, ConcatDataset, TensorDataset

from avalanche.benchmarks.utils.utils import (
    TaskSet,
    _count_unique,
    find_common_transforms_group,
    _init_task_labels,
    _init_transform_groups,
    _split_user_def_targets,
    _split_user_def_task_label,
    _traverse_supported_dataset,
)

from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.transform_groups import (
    TransformGroupDef,
    DefaultTransformGroups,
    XTransform,
    YTransform,
)
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.dataset_utils import (
    SubSequence,
)
from avalanche.benchmarks.utils.flat_data import ConstantSequence
from avalanche.benchmarks.utils.dataset_definitions import (
    ISupportedClassificationDataset,
    ITensorDataset,
    IDatasetWithTargets,
)

from typing import (
    List,
    Any,
    Sequence,
    Union,
    Optional,
    TypeVar,
    Callable,
    Dict,
    Tuple,
    Mapping,
    overload,
)


T_co = TypeVar("T_co", covariant=True)
TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")
TTargetType = int

TClassificationDataset = TypeVar(
    "TClassificationDataset", bound="ClassificationDataset"
)


def lookup(indexable, idx):
    """
    A simple function that implements indexing into an indexable object.
    Together with 'partial' this allows us to circumvent lambda functions
    that cannot be pickled.
    """
    return indexable[idx]


class ClassificationDataset(AvalancheDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "targets" in self._data_attributes, (
            "The supervised version of the ClassificationDataset requires "
            + "the targets field"
        )

    @property
    def targets(self) -> DataAttribute[TTargetType]:
        return self._data_attributes["targets"]

    # TODO: this shouldn't be needed
    def subset(self, indices):
        data = super().subset(indices)
        return data.with_transforms(self._flat_data._transform_groups.current_group)

    # TODO: this shouldn't be needed
    def concat(self, other):
        data = super().concat(other)
        return data.with_transforms(self._flat_data._transform_groups.current_group)

    def __hash__(self):
        return id(self)


class TaskAwareClassificationDataset(AvalancheDataset[T_co]):
    @property
    def task_pattern_indices(self) -> Dict[int, Sequence[int]]:
        """A dictionary mapping task ids to their sample indices."""
        return self.targets_task_labels.val_to_idx  # type: ignore

    @property
    def task_set(self: TClassificationDataset) -> TaskSet[TClassificationDataset]:
        """Returns the datasets's ``TaskSet``, which is a mapping <task-id,
        task-dataset>."""
        return TaskSet(self)

    def subset(self, indices):
        data = super().subset(indices)
        return data.with_transforms(self._flat_data._transform_groups.current_group)

    def concat(self, other):
        data = super().concat(other)
        return data.with_transforms(self._flat_data._transform_groups.current_group)

    def __hash__(self):
        return id(self)


class TaskAwareSupervisedClassificationDataset(TaskAwareClassificationDataset[T_co]):
    # TODO: remove? ClassificationDataset should have targets
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "targets" in self._data_attributes, (
            "The supervised version of the ClassificationDataset requires "
            + "the targets field"
        )
        assert "targets_task_labels" in self._data_attributes, (
            "The supervised version of the ClassificationDataset requires "
            + "the targets_task_labels field"
        )

    @property
    def targets(self) -> DataAttribute[TTargetType]:
        return self._data_attributes["targets"]

    @property
    def targets_task_labels(self) -> DataAttribute[int]:
        return self._data_attributes["targets_task_labels"]


SupportedDataset = Union[
    IDatasetWithTargets,
    ITensorDataset,
    Subset,
    ConcatDataset,
    TaskAwareClassificationDataset,
]


@overload
def _make_taskaware_classification_dataset(
    dataset: TaskAwareSupervisedClassificationDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    ...


@overload
def _make_taskaware_classification_dataset(
    dataset: SupportedDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Union[int, Sequence[int]],
    targets: Sequence[TTargetType],
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    ...


@overload
def _make_taskaware_classification_dataset(
    dataset: SupportedDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareClassificationDataset:
    ...


def _make_taskaware_classification_dataset(
    dataset: SupportedDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> Union[TaskAwareClassificationDataset, TaskAwareSupervisedClassificationDataset]:
    """Avalanche Classification Dataset.

    Supervised continual learning benchmarks in Avalanche return instances of
    this dataset, but it can also be used in a completely standalone manner.

    This dataset applies input/target transformations, it supports
    slicing and advanced indexing and it also contains useful fields as
    `targets`, which contains the pattern labels, and `targets_task_labels`,
    which contains the pattern task labels. The `task_set` field can be used to
    obtain a the subset of patterns labeled with a given task label.

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
    parameter, each pattern will be assigned a default task label 0.

    Creates a ``AvalancheDataset`` instance.

    :param dataset: The dataset to decorate. Beware that
        AvalancheDataset will not overwrite transformations already
        applied by this dataset.
    :param transform: A function/transform that takes the X value of a
        pattern from the original dataset and returns a transformed version.
    :param target_transform: A function/transform that takes in the target
        and transforms it.
    :param transform_groups: A dictionary containing the transform groups.
        Transform groups are used to quickly switch between training and
        eval (test) transformations. This becomes useful when in need to
        test on the training dataset as test transformations usually don't
        contain random augmentations. ``AvalancheDataset`` natively supports
        the 'train' and 'eval' groups by calling the ``train()`` and
        ``eval()`` methods. When using custom groups one can use the
        ``with_transforms(group_name)`` method instead. Defaults to None,
        which means that the current transforms will be used to
        handle both 'train' and 'eval' groups (just like in standard
        ``torchvision`` datasets).
    :param initial_transform_group: The name of the initial transform group
        to be used. Defaults to None, which means that the current group of
        the input dataset will be used (if an AvalancheDataset). If the
        input dataset is not an AvalancheDataset, then 'train' will be
        used.
    :param task_labels: The task label of each instance. Must be a sequence
        of ints, one for each instance in the dataset. Alternatively can be
        a single int value, in which case that value will be used as the
        task label for all the instances. Defaults to None, which means that
        the dataset will try to obtain the task labels from the original
        dataset. If no task labels could be found, a default task label
        0 will be applied to all instances.
    :param targets: The label of each pattern. Defaults to None, which
        means that the targets will be retrieved from the dataset (if
        possible).
    :param collate_fn: The function to use when slicing to merge single
        patterns.This function is the function
        used in the data loading process, too. If None
        the constructor will check if a
        `collate_fn` field exists in the dataset. If no such field exists,
        the default collate function will be used.
    """

    is_supervised = isinstance(dataset, TaskAwareSupervisedClassificationDataset)

    transform_gs = _init_transform_groups(
        transform_groups,
        transform,
        target_transform,
        initial_transform_group,
        dataset,
    )
    targets_data: Optional[DataAttribute[TTargetType]] = _init_targets(dataset, targets)
    task_labels_data: Optional[DataAttribute[int]] = _init_task_labels(
        dataset, task_labels
    )

    das: List[DataAttribute] = []
    if targets_data is not None:
        das.append(targets_data)
    if task_labels_data is not None:
        das.append(task_labels_data)

        # Check if supervision data has been added
    is_supervised = is_supervised or (
        targets_data is not None and task_labels_data is not None
    )

    data: Union[
        TaskAwareClassificationDataset, TaskAwareSupervisedClassificationDataset
    ]
    if is_supervised:
        data = TaskAwareSupervisedClassificationDataset(
            [dataset],
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            collate_fn=collate_fn,
        )
    else:
        data = TaskAwareClassificationDataset(
            [dataset],
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            collate_fn=collate_fn,
        )

    if initial_transform_group is not None:
        return data.with_transforms(initial_transform_group)
    else:
        return data


def _init_targets(
    dataset, targets, check_shape=True
) -> Optional[DataAttribute[TTargetType]]:
    if targets is not None:
        # User defined targets always take precedence
        if isinstance(targets, int):
            targets = ConstantSequence(targets, len(dataset))
        elif len(targets) != len(dataset) and check_shape:
            raise ValueError(
                "Invalid amount of target labels. It must be equal to the "
                "number of patterns in the dataset. Got {}, expected "
                "{}!".format(len(targets), len(dataset))
            )
        return DataAttribute(targets, "targets")

    targets = _traverse_supported_dataset(dataset, _select_targets)

    if targets is not None:
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

    if targets is None:
        return None

    return DataAttribute(targets, "targets")


@overload
def _taskaware_classification_subset(
    dataset: TaskAwareSupervisedClassificationDataset,
    indices: Optional[Sequence[int]] = None,
    *,
    class_mapping: Optional[Sequence[int]] = None,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    ...


@overload
def _taskaware_classification_subset(
    dataset: SupportedDataset,
    indices: Optional[Sequence[int]] = None,
    *,
    class_mapping: Optional[Sequence[int]] = None,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Union[int, Sequence[int]],
    targets: Sequence[TTargetType],
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    ...


@overload
def _taskaware_classification_subset(
    dataset: SupportedDataset,
    indices: Optional[Sequence[int]] = None,
    *,
    class_mapping: Optional[Sequence[int]] = None,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareClassificationDataset:
    ...


def _taskaware_classification_subset(
    dataset: SupportedDataset,
    indices: Optional[Sequence[int]] = None,
    *,
    class_mapping: Optional[Sequence[int]] = None,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> Union[TaskAwareClassificationDataset, TaskAwareSupervisedClassificationDataset]:
    """Creates an ``AvalancheSubset`` instance.

    For simple subset operations you should use the method
    `dataset.subset(indices)`.
    Use this constructor only if you need to redefine transformation or
    class/task labels.

    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    This Dataset also supports transformations, slicing, advanced indexing,
    the targets field, class mapping and all the other goodies listed in
    :class:`AvalancheDataset`.

    :param dataset: The whole dataset.
    :param indices: Indices in the whole set selected for subset. Can
        be None, which means that the whole dataset will be returned.
    :param class_mapping: A list that, for each possible target (Y) value,
        contains its corresponding remapped value. Can be None.
        Beware that setting this parameter will force the final
        dataset type to be CLASSIFICATION or UNDEFINED.
    :param transform: A function/transform that takes the X value of a
        pattern from the original dataset and returns a transformed version.
    :param target_transform: A function/transform that takes in the target
        and transforms it.
    :param transform_groups: A dictionary containing the transform groups.
        Transform groups are used to quickly switch between training and
        eval (test) transformations. This becomes useful when in need to
        test on the training dataset as test transformations usually don't
        contain random augmentations. ``AvalancheDataset`` natively supports
        the 'train' and 'eval' groups by calling the ``train()`` and
        ``eval()`` methods. When using custom groups one can use the
        ``with_transforms(group_name)`` method instead. Defaults to None,
        which means that the current transforms will be used to
        handle both 'train' and 'eval' groups (just like in standard
        ``torchvision`` datasets).
    :param initial_transform_group: The name of the initial transform group
        to be used. Defaults to None, which means that the current group of
        the input dataset will be used (if an AvalancheDataset). If the
        input dataset is not an AvalancheDataset, then 'train' will be
        used.
    :param task_labels: The task label for each instance. Must be a sequence
        of ints, one for each instance in the dataset. This can either be a
        list of task labels for the original dataset or the list of task
        labels for the instances of the subset (an automatic detection will
        be made). In the unfortunate case in which the original dataset and
        the subset contain the same amount of instances, then this parameter
        is considered to contain the task labels of the subset.
        Alternatively can be a single int value, in which case
        that value will be used as the task label for all the instances.
        Defaults to None, which means that the dataset will try to
        obtain the task labels from the original dataset. If no task labels
        could be found, a default task label 0 will be applied to all
        instances.
    :param targets: The label of each pattern. Defaults to None, which
        means that the targets will be retrieved from the dataset (if
        possible). This can either be a list of target labels for the
        original dataset or the list of target labels for the instances of
        the subset (an automatic detection will be made). In the unfortunate
        case in which the original dataset and the subset contain the same
        amount of instances, then this parameter is considered to contain
        the target labels of the subset.
    :param collate_fn: The function to use when slicing to merge single
        patterns. This function is the function
        used in the data loading process, too. If None,
        the constructor will check if a
        `collate_fn` field exists in the dataset. If no such field exists,
        the default collate function will be used.
    """

    is_supervised = isinstance(dataset, TaskAwareSupervisedClassificationDataset)

    if isinstance(dataset, TaskAwareClassificationDataset):
        if (
            class_mapping is None
            and transform is None
            and target_transform is None
            and transform_groups is None
            and initial_transform_group is None
            and task_labels is None
            and targets is None
            and collate_fn is None
        ):
            return dataset.subset(indices)

    targets_data: Optional[DataAttribute[TTargetType]] = _init_targets(
        dataset, targets, check_shape=False
    )
    task_labels_data: Optional[DataAttribute[int]] = _init_task_labels(
        dataset, task_labels, check_shape=False
    )

    transform_gs = _init_transform_groups(
        transform_groups,
        transform,
        target_transform,
        initial_transform_group,
        dataset,
    )

    if initial_transform_group is not None and isinstance(dataset, AvalancheDataset):
        dataset = dataset.with_transforms(initial_transform_group)

    if class_mapping is not None:  # update targets
        if targets_data is None:
            tgs = [class_mapping[el] for el in dataset.targets]  # type: ignore
        else:
            tgs = [class_mapping[el] for el in targets_data]

        targets_data = DataAttribute(tgs, "targets")

    if class_mapping is not None:
        frozen_transform_groups = DefaultTransformGroups(
            (None, partial(lookup, class_mapping))
        )
    else:
        frozen_transform_groups = None

    das = []
    if targets_data is not None:
        das.append(targets_data)

    # Check if supervision data has been added
    is_supervised = is_supervised or (
        targets_data is not None and task_labels_data is not None
    )

    if task_labels_data is not None:
        # special treatment for task labels depending on length for
        # backward compatibility
        if len(task_labels_data) != len(dataset):
            # task labels are already subsampled
            dataset = TaskAwareClassificationDataset(
                [dataset],
                indices=list(indices) if indices is not None else None,
                data_attributes=das,
                transform_groups=transform_gs,
                frozen_transform_groups=frozen_transform_groups,
                collate_fn=collate_fn,
            )
            # now add task labels
            if is_supervised:
                return TaskAwareSupervisedClassificationDataset(
                    [dataset],
                    data_attributes=[dataset.targets, task_labels_data],  # type: ignore
                )
            else:
                return TaskAwareClassificationDataset(
                    [dataset],
                    data_attributes=[dataset.targets, task_labels_data],  # type: ignore
                )
        else:
            das.append(task_labels_data)

    if is_supervised:
        return TaskAwareSupervisedClassificationDataset(
            [dataset],
            indices=list(indices) if indices is not None else None,
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            frozen_transform_groups=frozen_transform_groups,
            collate_fn=collate_fn,
        )
    else:
        return TaskAwareClassificationDataset(
            [dataset],
            indices=list(indices) if indices is not None else None,
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            frozen_transform_groups=frozen_transform_groups,
            collate_fn=collate_fn,
        )


@overload
def _make_taskaware_tensor_classification_dataset(
    *dataset_tensors: Sequence,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Dict[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = "train",
    task_labels: Union[int, Sequence[int]],
    targets: Union[Sequence[TTargetType], int],
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    ...


@overload
def _make_taskaware_tensor_classification_dataset(
    *dataset_tensors: Sequence,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Dict[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = "train",
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Union[Sequence[TTargetType], int]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> Union[TaskAwareClassificationDataset, TaskAwareSupervisedClassificationDataset]:
    ...


def _make_taskaware_tensor_classification_dataset(
    *dataset_tensors: Sequence,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Dict[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = "train",
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Union[Sequence[TTargetType], int]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> Union[TaskAwareClassificationDataset, TaskAwareSupervisedClassificationDataset]:
    """Creates a ``AvalancheTensorDataset`` instance.

    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset from PyTorch,
    this Dataset also supports transformations, slicing, advanced indexing,
    the targets field and all the other goodies listed in
    :class:`AvalancheDataset`.

    :param dataset_tensors: Sequences, Tensors or ndarrays representing the
        content of the dataset.
    :param transform: A function/transform that takes in a single element
        from the first tensor and returns a transformed version.
    :param target_transform: A function/transform that takes a single
        element of the second tensor and transforms it.
    :param transform_groups: A dictionary containing the transform groups.
        Transform groups are used to quickly switch between training and
        eval (test) transformations. This becomes useful when in need to
        test on the training dataset as test transformations usually don't
        contain random augmentations. ``AvalancheDataset`` natively supports
        the 'train' and 'eval' groups by calling the ``train()`` and
        ``eval()`` methods. When using custom groups one can use the
        ``with_transforms(group_name)`` method instead. Defaults to None,
        which means that the current transforms will be used to
        handle both 'train' and 'eval' groups (just like in standard
        ``torchvision`` datasets).
    :param initial_transform_group: The name of the transform group
        to be used. Defaults to 'train'.
    :param task_labels: The task labels for each pattern. Must be a sequence
        of ints, one for each pattern in the dataset. Alternatively can be a
        single int value, in which case that value will be used as the task
        label for all the instances. Defaults to None, which means that a
        default task label 0 will be applied to all patterns.
    :param targets: The label of each pattern. Defaults to None, which
        means that the targets will be retrieved from the second tensor of
        the dataset. Otherwise, it can be a sequence of values containing
        as many elements as the number of patterns.
    :param collate_fn: The function to use when slicing to merge single
        patterns. In the future this function may become the function
        used in the data loading process, too.
    """
    if len(dataset_tensors) < 1:
        raise ValueError("At least one sequence must be passed")

    if targets is None:
        targets = dataset_tensors[1]
    elif isinstance(targets, int):
        targets = dataset_tensors[targets]
    tts = []
    for tt in dataset_tensors:  # TorchTensor requires a pytorch tensor
        if not hasattr(tt, "size"):
            tt = torch.tensor(tt)
        tts.append(tt)
    dataset = _TensorClassificationDataset(*tts)

    transform_gs = _init_transform_groups(
        transform_groups,
        transform,
        target_transform,
        initial_transform_group,
        dataset,
    )
    targets_data = _init_targets(dataset, targets)
    task_labels_data = _init_task_labels(dataset, task_labels)
    if initial_transform_group is not None and isinstance(dataset, AvalancheDataset):
        dataset = dataset.with_transforms(initial_transform_group)

    das = []
    for d in [targets_data, task_labels_data]:
        if d is not None:
            das.append(d)

    # Check if supervision data has been added
    is_supervised = targets_data is not None and task_labels_data is not None

    if is_supervised:
        return TaskAwareSupervisedClassificationDataset(
            [dataset],
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            collate_fn=collate_fn,
        )
    else:
        return TaskAwareClassificationDataset(
            [dataset],
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            collate_fn=collate_fn,
        )


class _TensorClassificationDataset(TensorDataset):
    """we want class labels to be integers, not tensors."""

    def __getitem__(self, item):
        elem = list(super().__getitem__(item))
        elem[1] = elem[1].item()
        return tuple(elem)


@overload
def _concat_taskaware_classification_datasets(
    datasets: Sequence[TaskAwareSupervisedClassificationDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
    targets: Optional[
        Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]]
    ] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    ...


@overload
def _concat_taskaware_classification_datasets(
    datasets: Sequence[SupportedDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Union[int, Sequence[int], Sequence[Sequence[int]]],
    targets: Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]],
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    ...


@overload
def _concat_taskaware_classification_datasets(
    datasets: Sequence[SupportedDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
    targets: Optional[
        Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]]
    ] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareClassificationDataset:
    ...


def _concat_taskaware_classification_datasets(
    datasets: Sequence[SupportedDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
    targets: Optional[
        Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]]
    ] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> Union[TaskAwareClassificationDataset, TaskAwareSupervisedClassificationDataset]:
    """Creates a ``AvalancheConcatDataset`` instance.

    For simple subset operations you should use the method
    `dataset.concat(other)` or
    `concat_datasets` from `avalanche.benchmarks.utils.utils`.
    Use this constructor only if you need to redefine transformation or
    class/task labels.

    A Dataset that behaves like a PyTorch
    :class:`torch.utils.data.ConcatDataset`. However, this Dataset also supports
    transformations, slicing, advanced indexing and the targets field and all
    the other goodies listed in :class:`AvalancheDataset`.

    This dataset guarantees that the operations involving the transformations
    and transformations groups are consistent across the concatenated dataset
    (if they are subclasses of :class:`AvalancheDataset`).

    :param datasets: A collection of datasets.
    :param transform: A function/transform that takes the X value of a
        pattern from the original dataset and returns a transformed version.
    :param target_transform: A function/transform that takes in the target
        and transforms it.
    :param transform_groups: A dictionary containing the transform groups.
        Transform groups are used to quickly switch between training and
        eval (test) transformations. This becomes useful when in need to
        test on the training dataset as test transformations usually don't
        contain random augmentations. ``AvalancheDataset`` natively supports
        the 'train' and 'eval' groups by calling the ``train()`` and
        ``eval()`` methods. When using custom groups one can use the
        ``with_transforms(group_name)`` method instead. Defaults to None,
        which means that the current transforms will be used to
        handle both 'train' and 'eval' groups (just like in standard
        ``torchvision`` datasets).
    :param initial_transform_group: The name of the initial transform group
        to be used. Defaults to None, which means that if all
        AvalancheDatasets in the input datasets list agree on a common
        group (the "current group" is the same for all datasets), then that
        group will be used as the initial one. If the list of input datasets
        does not contain an AvalancheDataset or if the AvalancheDatasets
        do not agree on a common group, then 'train' will be used.
    :param targets: The label of each pattern. Can either be a sequence of
        labels or, alternatively, a sequence containing sequences of labels
        (one for each dataset to be concatenated). Defaults to None, which
        means that the targets will be retrieved from the datasets (if
        possible).
    :param task_labels: The task labels for each pattern. Must be a sequence
        of ints, one for each pattern in the dataset. Alternatively, task
        labels can be expressed as a sequence containing sequences of ints
        (one for each dataset to be concatenated) or even a single int,
        in which case that value will be used as the task label for all
        instances. Defaults to None, which means that the dataset will try
        to obtain the task labels from the original datasets. If no task
        labels could be found for a dataset, a default task label 0 will
        be applied to all patterns of that dataset.
    :param collate_fn: The function to use when slicing to merge single
        patterns. In the future this function may become the function
        used in the data loading process, too. If None, the constructor
        will check if a `collate_fn` field exists in the first dataset. If
        no such field exists, the default collate function will be used.
        Beware that the chosen collate function will be applied to all
        the concatenated datasets even if a different collate is defined
        in different datasets.
    """
    dds = []
    per_dataset_task_labels = _split_user_def_task_label(datasets, task_labels)

    per_dataset_targets = _split_user_def_targets(
        datasets, targets, lambda x: isinstance(x, int)
    )

    # Find common "current_group" or use "train"
    if initial_transform_group is None:
        initial_transform_group = find_common_transforms_group(
            datasets, default_group="train"
        )

    supervised = True
    for dd, dataset_task_labels, dataset_targets in zip(
        datasets, per_dataset_task_labels, per_dataset_targets
    ):
        dd = _make_taskaware_classification_dataset(
            dd,
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=dataset_task_labels,
            targets=dataset_targets,
            collate_fn=collate_fn,
        )

        if not isinstance(dd, TaskAwareSupervisedClassificationDataset):
            supervised = False

        dds.append(dd)

    if len(dds) > 0:
        transform_groups_obj = _init_transform_groups(
            transform_groups,
            transform,
            target_transform,
            initial_transform_group,
            dds[0],
        )
    else:
        transform_groups_obj = None

    supervised = supervised and (
        (len(dds) > 0) or (targets is not None and task_labels is not None)
    )

    data: Union[
        TaskAwareSupervisedClassificationDataset, TaskAwareClassificationDataset
    ]
    if supervised:
        data = TaskAwareSupervisedClassificationDataset(
            dds, transform_groups=transform_groups_obj
        )
    else:
        data = TaskAwareClassificationDataset(
            dds, transform_groups=transform_groups_obj
        )
    return data.with_transforms(initial_transform_group)


def _select_targets(
    dataset: SupportedDataset, indices: Optional[List[int]]
) -> Sequence[TTargetType]:
    if hasattr(dataset, "targets"):
        # Standard supported dataset
        found_targets = dataset.targets  # type: ignore
    elif hasattr(dataset, "tensors"):
        # Support for PyTorch TensorDataset
        if len(dataset.tensors) < 2:  # type: ignore
            raise ValueError(
                "Tensor dataset has not enough tensors: " "at least 2 are required."
            )
        found_targets = dataset.tensors[1]  # type: ignore
    else:
        raise ValueError(
            "Unsupported dataset: must have a valid targets field "
            "or has to be a Tensor Dataset with at least 2 "
            "Tensors"
        )

    if indices is not None:
        found_targets = SubSequence(found_targets, indices=indices)

    return found_targets


def _concat_taskaware_classification_datasets_sequentially(
    train_dataset_list: Sequence[ISupportedClassificationDataset],
    test_dataset_list: Sequence[ISupportedClassificationDataset],
) -> Tuple[
    TaskAwareSupervisedClassificationDataset,
    TaskAwareSupervisedClassificationDataset,
    List[list],
]:
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
    remapped_train_datasets: List[TaskAwareSupervisedClassificationDataset] = []
    remapped_test_datasets: List[TaskAwareSupervisedClassificationDataset] = []
    next_remapped_idx = 0

    train_dataset_list_sup = list(
        map(_as_taskaware_supervised_classification_dataset, train_dataset_list)
    )
    test_dataset_list_sup = list(
        map(_as_taskaware_supervised_classification_dataset, test_dataset_list)
    )
    del train_dataset_list
    del test_dataset_list

    # Obtain the number of classes of each dataset
    classes_per_dataset = [
        _count_unique(
            train_dataset_list_sup[dataset_idx].targets,
            test_dataset_list_sup[dataset_idx].targets,
        )
        for dataset_idx in range(len(train_dataset_list_sup))
    ]

    new_class_ids_per_dataset = []
    for dataset_idx in range(len(train_dataset_list_sup)):
        # Get the train and test sets of the dataset
        train_set = train_dataset_list_sup[dataset_idx]
        test_set = test_dataset_list_sup[dataset_idx]

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

        a = _taskaware_classification_subset(train_set, class_mapping=class_mapping)

        # Create remapped datasets and append them to the final list
        remapped_train_datasets.append(
            _taskaware_classification_subset(train_set, class_mapping=class_mapping)
        )
        remapped_test_datasets.append(
            _taskaware_classification_subset(test_set, class_mapping=class_mapping)
        )
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return (
        _concat_taskaware_classification_datasets(remapped_train_datasets),
        _concat_taskaware_classification_datasets(remapped_test_datasets),
        new_class_ids_per_dataset,
    )


def _as_taskaware_supervised_classification_dataset(
    dataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset:
    if (
        transform is not None
        or target_transform is not None
        or transform_groups is not None
        or initial_transform_group is not None
        or task_labels is not None
        or targets is not None
        or collate_fn is not None
        or not isinstance(dataset, TaskAwareSupervisedClassificationDataset)
    ):
        result_dataset = _make_taskaware_classification_dataset(
            dataset=dataset,
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=task_labels,
            targets=targets,
            collate_fn=collate_fn,
        )

        if not isinstance(result_dataset, TaskAwareSupervisedClassificationDataset):
            raise ValueError(
                "The given dataset does not have supervision fields "
                "(targets, task_labels)."
            )

        return result_dataset

    return dataset


__all__ = [
    "SupportedDataset",
    "TaskAwareClassificationDataset",
    "TaskAwareSupervisedClassificationDataset",
    "_make_taskaware_classification_dataset",
    "_make_taskaware_tensor_classification_dataset",
    "_taskaware_classification_subset",
    "_concat_taskaware_classification_datasets",
]
