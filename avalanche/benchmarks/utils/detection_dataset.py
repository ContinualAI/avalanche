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
This module contains the implementation of the ``DetectionDataset``,
which is the dataset used for supervised continual learning benchmarks.
DetectionDatasets are ``AvalancheDatasets`` that manage targets and task
labels automatically. Concatenation and subsampling operations are optimized
to be used frequently, as is common in replay strategies.
"""
from functools import partial
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

import torch
from torch import Tensor
from torch.utils.data.dataset import Subset, ConcatDataset

from avalanche.benchmarks.utils.utils import (
    TaskSet,
    _init_task_labels,
    _init_transform_groups,
    _split_user_def_targets,
    _split_user_def_task_label,
    _traverse_supported_dataset,
)

from .collate_functions import detection_collate_fn
from .data import AvalancheDataset
from .data_attribute import DataAttribute
from .dataset_definitions import (
    IDataset,
    IDatasetWithTargets,
)
from .dataset_utils import (
    SubSequence,
)
from .flat_data import ConstantSequence
from .transform_groups import (
    TransformGroupDef,
    DefaultTransformGroups,
    TransformGroups,
    XTransform,
    YTransform,
)

T_co = TypeVar("T_co", covariant=True)
TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")
TTargetType = Dict[str, Tensor]


# Image (tensor), target dict, task label
DetectionExampleT = Tuple[Tensor, TTargetType, int]
TDetectionDataset = TypeVar("TDetectionDataset", bound="DetectionDataset")


class DetectionDataset(AvalancheDataset[T_co]):
    @property
    def task_pattern_indices(self) -> Dict[int, Sequence[int]]:
        """A dictionary mapping task ids to their sample indices."""
        return self.targets_task_labels.val_to_idx  # type: ignore

    @property
    def task_set(self: TDetectionDataset) -> TaskSet[TDetectionDataset]:
        """Returns the dataset's ``TaskSet``, which is a mapping <task-id,
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


class SupervisedDetectionDataset(DetectionDataset[T_co]):
    def __init__(
        self,
        datasets: List[IDataset[T_co]],
        *,
        indices: Optional[List[int]] = None,
        data_attributes: Optional[List[DataAttribute]] = None,
        transform_groups: Optional[TransformGroups] = None,
        frozen_transform_groups: Optional[TransformGroups] = None,
        collate_fn: Optional[Callable[[List], Any]] = None
    ):
        super().__init__(
            datasets=datasets,
            indices=indices,
            data_attributes=data_attributes,
            transform_groups=transform_groups,
            frozen_transform_groups=frozen_transform_groups,
            collate_fn=collate_fn,
        )

        assert hasattr(self, "targets"), (
            "The supervised version of the ClassificationDataset requires "
            + "the targets field"
        )
        assert hasattr(self, "targets_task_labels"), (
            "The supervised version of the ClassificationDataset requires "
            + "the targets_task_labels field"
        )

    @property
    def targets(self) -> DataAttribute[TTargetType]:
        return self._data_attributes["targets"]

    @property
    def targets_task_labels(self) -> DataAttribute[int]:
        return self._data_attributes["targets_task_labels"]


SupportedDetectionDataset = Union[
    IDatasetWithTargets, Subset, ConcatDataset, DetectionDataset
]


@overload
def make_detection_dataset(
    dataset: SupervisedDetectionDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> SupervisedDetectionDataset:
    ...


@overload
def make_detection_dataset(
    dataset: SupportedDetectionDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Union[int, Sequence[int]],
    targets: Sequence[TTargetType],
    collate_fn: Optional[Callable[[List], Any]] = None
) -> SupervisedDetectionDataset:
    ...


@overload
def make_detection_dataset(
    dataset: SupportedDetectionDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> DetectionDataset:
    ...


def make_detection_dataset(
    dataset: SupportedDetectionDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> Union[DetectionDataset, SupervisedDetectionDataset]:
    """Avalanche Detection Dataset.

    Supervised continual learning benchmarks in Avalanche return instances of
    this dataset, but it can also be used in a completely standalone manner.

    This dataset applies input/target transformations, it supports
    slicing and advanced indexing and it also contains useful fields as
    `targets`, which contains the pattern dictionaries, and
    `targets_task_labels`, which contains the pattern task labels.
    The `task_set` field can be used to obtain a the subset of patterns
    labeled with a given task label.

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
    :param targets: The dictionary of detection boxes of each pattern.
        Defaults to None, which means that the targets will be retrieved from
        the dataset (if possible).
    :param collate_fn: The function to use when slicing to merge single
        patterns. This function is the function used in the data loading
        process, too. If None, the constructor will check if a
        `collate_fn` field exists in the dataset. If no such field exists,
        the default collate function for detection will be used.
    """

    is_supervised = isinstance(dataset, SupervisedDetectionDataset)

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

    if collate_fn is None:
        collate_fn = getattr(dataset, "collate_fn", detection_collate_fn)

    data: Union[DetectionDataset, SupervisedDetectionDataset]
    if is_supervised:
        data = SupervisedDetectionDataset(
            [dataset],
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            collate_fn=collate_fn,
        )
    else:
        data = DetectionDataset(
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
        if len(targets) != len(dataset) and check_shape:
            raise ValueError(
                "Invalid amount of target labels. It must be equal to the "
                "number of patterns in the dataset. Got {}, expected "
                "{}!".format(len(targets), len(dataset))
            )
        return DataAttribute(targets, "targets")

    targets = _traverse_supported_dataset(dataset, _select_targets)

    if targets is None:
        return None

    return DataAttribute(targets, "targets")


def _detection_class_mapping_transform(class_mapping, example_target_dict):
    example_target_dict = dict(example_target_dict)

    # example_target_dict["labels"] is a tensor containing one label
    # for each bounding box in the image. We need to remap each of them
    example_target_labels = example_target_dict["labels"]
    example_mapped_labels = [class_mapping[int(el)] for el in example_target_labels]

    if isinstance(example_target_labels, Tensor):
        example_mapped_labels = torch.as_tensor(example_mapped_labels)

    example_target_dict["labels"] = example_mapped_labels

    return example_target_dict


@overload
def detection_subset(
    dataset: SupervisedDetectionDataset,
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
) -> SupervisedDetectionDataset:
    ...


@overload
def detection_subset(
    dataset: SupportedDetectionDataset,
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
) -> SupervisedDetectionDataset:
    ...


@overload
def detection_subset(
    dataset: SupportedDetectionDataset,
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
) -> DetectionDataset:
    ...


def detection_subset(
    dataset: SupportedDetectionDataset,
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
) -> Union[DetectionDataset, SupervisedDetectionDataset]:
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
    :param class_mapping: A list that, for each possible class label value,
        contains its corresponding remapped value. Can be None.
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
    :param targets: The target dictionary of each pattern. Defaults to None,
        which means that the targets will be retrieved from the dataset (if
        possible). This can either be a list of target dictionaries for the
        original dataset or the list of target dictionaries for the instances
        of the subset (an automatic detection will be made). In the
        unfortunate case in which the original dataset and the subset contain
        the same amount of instances, then this parameter is considered to
        contain the target dictionaries of the subset.
    :param collate_fn: The function to use when slicing to merge single
        patterns. This function is the function used in the data loading
        process, too. If None, the constructor will check if a
        `collate_fn` field exists in the dataset. If no such field exists,
        the default collate function for detection will be used
    """

    is_supervised = isinstance(dataset, SupervisedDetectionDataset)

    if isinstance(dataset, DetectionDataset):
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

    del task_labels
    del targets

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
            # Should not happen
            # The following line usually fails
            targets_data = dataset.targets  # type: ignore

        assert (
            targets_data is not None
        ), "To execute the class mapping, a list of targets is required."

        tgs = [
            _detection_class_mapping_transform(class_mapping, example_target_dict)
            for example_target_dict in targets_data
        ]

        targets_data = DataAttribute(tgs, "targets")

    if class_mapping is not None:
        mapping_fn = partial(_detection_class_mapping_transform, class_mapping)
        frozen_transform_groups = DefaultTransformGroups((None, mapping_fn))
    else:
        frozen_transform_groups = None

    das: List[DataAttribute] = []
    if targets_data is not None:
        das.append(targets_data)
    if task_labels_data is not None:
        das.append(task_labels_data)

    # Check if supervision data has been added
    is_supervised = is_supervised or (
        targets_data is not None and task_labels_data is not None
    )

    if collate_fn is None:
        collate_fn = detection_collate_fn

    if is_supervised:
        return SupervisedDetectionDataset(
            [dataset],
            indices=list(indices) if indices is not None else None,
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            frozen_transform_groups=frozen_transform_groups,
            collate_fn=collate_fn,
        )
    else:
        return DetectionDataset(
            [dataset],
            indices=list(indices) if indices is not None else None,
            data_attributes=das if len(das) > 0 else None,
            transform_groups=transform_gs,
            frozen_transform_groups=frozen_transform_groups,
            collate_fn=collate_fn,
        )


@overload
def concat_detection_datasets(
    datasets: Sequence[SupervisedDetectionDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
    targets: Optional[
        Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]]
    ] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> SupervisedDetectionDataset:
    ...


@overload
def concat_detection_datasets(
    datasets: Sequence[SupportedDetectionDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Union[int, Sequence[int], Sequence[Sequence[int]]],
    targets: Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]],
    collate_fn: Optional[Callable[[List], Any]] = None
) -> SupervisedDetectionDataset:
    ...


@overload
def concat_detection_datasets(
    datasets: Sequence[SupportedDetectionDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
    targets: Optional[
        Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]]
    ] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> DetectionDataset:
    ...


def concat_detection_datasets(
    datasets: Sequence[SupportedDetectionDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = None,
    targets: Optional[
        Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]]
    ] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> Union[DetectionDataset, SupervisedDetectionDataset]:
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
        patterns. This function is the function used in the data loading
        process, too. If None, the constructor will check if a `collate_fn`
        field exists in the first dataset. If no such field exists, the
        default collate function for detection  will be used.
        Beware that the chosen collate function will be applied to all
        the concatenated datasets even if a different collate is defined
        in different datasets.
    """
    dds = []
    per_dataset_task_labels = _split_user_def_task_label(datasets, task_labels)

    per_dataset_targets = _split_user_def_targets(
        datasets, targets, lambda x: isinstance(x, dict)
    )

    for dd, dataset_task_labels, dataset_targets in zip(
        datasets, per_dataset_task_labels, per_dataset_targets
    ):
        dd = make_detection_dataset(
            dd,
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=dataset_task_labels,
            targets=dataset_targets,
            collate_fn=collate_fn,
        )
        dds.append(dd)

    if (
        transform is None
        and target_transform is None
        and transform_groups is None
        and initial_transform_group is None
        and task_labels is None
        and targets is None
        and collate_fn is None
        and len(datasets) > 0
    ):
        d0 = datasets[0]
        if isinstance(d0, DetectionDataset):
            for d1 in datasets[1:]:
                d0 = d0.concat(d1)
            return d0

    das: List[DataAttribute] = []
    if len(dds) > 0:
        #######################################
        # TRANSFORMATION GROUPS
        #######################################
        transform_groups_obj = _init_transform_groups(
            transform_groups,
            transform,
            target_transform,
            initial_transform_group,
            dds[0],
        )

        # Find common "current_group" or use "train"
        if initial_transform_group is None:
            uniform_group = None
            for d_set in datasets:
                if isinstance(d_set, AvalancheDataset):
                    if uniform_group is None:
                        uniform_group = d_set._transform_groups.current_group
                    else:
                        if uniform_group != d_set._transform_groups.current_group:
                            uniform_group = None
                            break

            if uniform_group is None:
                initial_transform_group = "train"
            else:
                initial_transform_group = uniform_group

        #######################################
        # DATA ATTRIBUTES
        #######################################

        totlen = sum([len(d) for d in datasets])
        if task_labels is not None:  # User defined targets always take precedence
            all_labels: IDataset[int]
            if isinstance(task_labels, int):
                all_labels = ConstantSequence(task_labels, totlen)
            else:
                all_labels_lst = []
                for dd, dataset_task_labels in zip(dds, per_dataset_task_labels):
                    assert dataset_task_labels is not None

                    # We already checked that len(t_labels) == len(dataset)
                    # (done in _split_user_def_task_label)
                    if isinstance(dataset_task_labels, int):
                        all_labels_lst.extend([dataset_task_labels] * len(dd))
                    else:
                        all_labels_lst.extend(dataset_task_labels)
                all_labels = all_labels_lst
            das.append(
                DataAttribute(all_labels, "targets_task_labels", use_in_getitem=True)
            )

        if targets is not None:  # User defined targets always take precedence
            all_targets_lst: List[TTargetType] = []
            for dd, dataset_targets in zip(dds, per_dataset_targets):
                assert dataset_targets is not None

                # We already checked that len(targets) == len(dataset)
                # (done in _split_user_def_targets)
                all_targets_lst.extend(dataset_targets)
            das.append(DataAttribute(all_targets_lst, "targets"))
    else:
        transform_groups_obj = None
        initial_transform_group = "train"

    data = DetectionDataset(
        dds,
        transform_groups=transform_groups_obj,
        data_attributes=das if len(das) > 0 else None,
    )
    return data.with_transforms(initial_transform_group)


def _select_targets(
    dataset: SupportedDetectionDataset, indices: Optional[List[int]]
) -> Sequence[TTargetType]:
    if hasattr(dataset, "targets"):
        # Standard supported dataset
        found_targets = dataset.targets
    else:
        raise ValueError("Unsupported dataset: must have a valid targets field")

    if indices is not None:
        found_targets = SubSequence(found_targets, indices=indices)

    return found_targets


__all__ = [
    "SupportedDetectionDataset",
    "DetectionDataset",
    "make_detection_dataset",
    "detection_subset",
    "concat_detection_datasets",
]
