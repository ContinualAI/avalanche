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
import warnings
from collections import defaultdict, deque

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset, ConcatDataset, TensorDataset

from .data import make_avalanche_dataset, AvalancheDataset
from .transform_groups import TransformGroups, DefaultTransformGroups
from .data_attribute import DataAttribute
from .dataset_utils import (
    SubSequence,
    find_list_from_index,
)
from .flat_data import ConstantSequence
from .dataset_definitions import (
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
)

from typing_extensions import Protocol

T_co = TypeVar("T_co", covariant=True)
TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")
TTargetType = Union[int]


# Info: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols
class XComposedTransformDef(Protocol):
    def __call__(self, *input_values: Any) -> Any:
        pass


class XTransformDef(Protocol):
    def __call__(self, input_value: Any) -> Any:
        pass


class YTransformDef(Protocol):
    def __call__(self, input_value: Any) -> Any:
        pass


XTransform = Optional[Union[XTransformDef, XComposedTransformDef]]
YTransform = Optional[YTransformDef]
TransformGroupDef = Union[None, XTransform, Tuple[XTransform, YTransform]]


class _ClassificationAttributesMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def task_pattern_indices(self):
        """A dictionary mapping task ids to their sample indices."""
        return self.targets_task_labels.val_to_idx

    @property
    def task_set(self):
        """Returns the datasets's ``TaskSet``, which is a mapping <task-id,
        task-dataset>."""
        return TaskSet(self)


class ClassificationDataset(AvalancheDataset, _ClassificationAttributesMixin):
    def subset(self, indices):
        data = super().subset(indices)
        return data.with_transforms(self._transform_groups.current_group)

    def concat(self, other):
        data = super().concat(other)
        return data.with_transforms(self._transform_groups.current_group)

    def __hash__(self):
        return id(self)


SupportedDataset = Union[
    IDatasetWithTargets,
    ITensorDataset,
    Subset,
    ConcatDataset,
    ClassificationDataset
]


def make_classification_dataset(
    dataset: SupportedDataset,
    *,
    transform: XTransform = None,
    target_transform: YTransform = None,
    transform_groups: Dict[str, TransformGroupDef] = None,
    initial_transform_group: str = None,
    task_labels: Union[int, Sequence[int]] = None,
    targets: Sequence[TTargetType] = None,
    collate_fn: Callable[[List], Any] = None
):
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
    transform_gs = _init_transform_groups(
        transform_groups,
        transform,
        target_transform,
        initial_transform_group,
        dataset,
    )
    targets = _init_targets(dataset, targets)
    task_labels = _init_task_labels(dataset, task_labels)

    das = []
    if targets is not None:
        das.append(targets)
    if task_labels is not None:
        das.append(task_labels)
    if len(das) == 0:
        das = None

    data = ClassificationDataset(
        [dataset],
        data_attributes=das,
        transform_groups=transform_gs,
        collate_fn=collate_fn,
    )
    if initial_transform_group is not None:
        return data.with_transforms(initial_transform_group)
    else:
        return data


def _init_transform_groups(
    transform_groups,
    transform,
    target_transform,
    initial_transform_group,
    dataset,
):
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
            isinstance(dataset, ClassificationDataset)
            and dataset._transform_groups is not None
        ):
            initial_transform_group = dataset._transform_groups.current_group
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
        tgs = TransformGroups(
            transform_groups, current_group=initial_transform_group
        )
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


def _init_targets(dataset, targets, check_shape=True):
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

    if isinstance(dataset, ClassificationDataset):
        return None  # targets are initialized automatically
    else:
        targets = _traverse_supported_dataset(dataset, _select_targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

    if targets is None:
        return None
    return DataAttribute(targets, "targets")


def _init_task_labels(dataset, task_labels, check_shape=True):
    """A task label for each pattern in the dataset."""
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
        tls = SubSequence(task_labels, converter=int)
    else:
        if isinstance(dataset, ClassificationDataset):
            tls = None
        else:
            task_labels = _traverse_supported_dataset(
                dataset, _select_task_labels
            )
            tls = SubSequence(task_labels, converter=int)

    if tls is None:
        return None
    return DataAttribute(tls, "targets_task_labels", use_in_getitem=True)


def classification_subset(
    dataset: SupportedDataset,
    indices: Sequence[int] = None,
    *,
    class_mapping: Sequence[int] = None,
    transform: Callable[[Any], Any] = None,
    target_transform: Callable[[int], int] = None,
    transform_groups: Dict[str, Tuple[XTransform, YTransform]] = None,
    initial_transform_group: str = None,
    task_labels: Union[int, Sequence[int]] = None,
    targets: Sequence[TTargetType] = None,
    collate_fn: Callable[[List], Any] = None
):
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
    if isinstance(dataset, ClassificationDataset):
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

    targets = _init_targets(dataset, targets, check_shape=False)
    task_labels = _init_task_labels(dataset, task_labels, check_shape=False)
    transform_gs = _init_transform_groups(
        transform_groups,
        transform,
        target_transform,
        initial_transform_group,
        dataset,
    )

    if initial_transform_group is not None and isinstance(
        dataset, AvalancheDataset
    ):
        dataset = dataset.with_transforms(initial_transform_group)

    if class_mapping is not None:  # update targets
        if targets is None:
            tgs = [class_mapping[el] for el in dataset.targets]
        else:
            tgs = [class_mapping[el] for el in targets]
        targets = DataAttribute(tgs, "targets")

    if class_mapping is not None:
        frozen_transform_groups = DefaultTransformGroups(
            (None, lambda x: class_mapping[x])
        )
    else:
        frozen_transform_groups = None

    das = []
    if targets is not None:
        das.append(targets)
    if task_labels is not None:
        # special treatment for task labels depending on length for
        # backward compatibility
        if len(task_labels) != len(dataset):
            # task labels are already subsampled
            dataset = ClassificationDataset(
                [dataset],
                indices=indices,
                data_attributes=das,
                transform_groups=transform_gs,
                frozen_transform_groups=frozen_transform_groups,
                collate_fn=collate_fn,
            )
            # now add task labels
            return ClassificationDataset(
                [dataset],
                data_attributes=[dataset.targets, task_labels])
        else:
            das.append(task_labels)

    if len(das) == 0:
        das = None

    return ClassificationDataset(
        [dataset],
        indices=indices,
        data_attributes=das,
        transform_groups=transform_gs,
        frozen_transform_groups=frozen_transform_groups,
        collate_fn=collate_fn,
    )


def make_tensor_classification_dataset(
    *dataset_tensors: Sequence,
    transform: Callable[[Any], Any] = None,
    target_transform: Callable[[int], int] = None,
    transform_groups: Dict[str, Tuple[XTransform, YTransform]] = None,
    initial_transform_group: str = "train",
    task_labels: Union[int, Sequence[int]] = None,
    targets: Union[Sequence[TTargetType], int] = None,
    collate_fn: Callable[[List], Any] = None
):
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
        if not hasattr(tt, 'size'):
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
    targets = _init_targets(dataset, targets)
    task_labels = _init_task_labels(dataset, task_labels)

    if initial_transform_group is not None and isinstance(
        dataset, AvalancheDataset
    ):
        dataset = dataset.with_transforms(initial_transform_group)

    return ClassificationDataset(
        [dataset],
        data_attributes=[targets, task_labels],
        transform_groups=transform_gs,
        collate_fn=collate_fn,
    )


class _TensorClassificationDataset(TensorDataset):
    """we want class labels to be integers, not tensors."""

    def __getitem__(self, item):
        elem = list(super().__getitem__(item))
        elem[1] = elem[1].item()
        return tuple(elem)


def concat_classification_datasets(
    datasets: List[SupportedDataset],
    *,
    transform: Callable[[Any], Any] = None,
    target_transform: Callable[[int], int] = None,
    transform_groups: Dict[str, Tuple[XTransform, YTransform]] = None,
    initial_transform_group: str = None,
    task_labels: Union[int, Sequence[int], Sequence[Sequence[int]]] = None,
    targets: Union[
        Sequence[TTargetType], Sequence[Sequence[TTargetType]]
    ] = None,
    collate_fn: Callable[[List], Any] = None
):
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
    for dd in datasets:
        if not isinstance(dd, AvalancheDataset):
            dd = make_classification_dataset(
                dd,
                transform=transform,
                target_transform=target_transform,
                transform_groups=transform_groups,
                initial_transform_group=initial_transform_group,
                task_labels=task_labels,
                targets=targets,
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
        if isinstance(d0, ClassificationDataset):
            for d1 in datasets[1:]:
                d0 = d0.concat(d1)
            return d0

    das = []
    if len(dds) > 0:
        #######################################
        # TRANSFORMATION GROUPS
        #######################################
        transform_groups = _init_transform_groups(
            transform_groups,
            transform,
            target_transform,
            initial_transform_group,
            dds[0],
        )

        if initial_transform_group is None:
            uniform_group = None
            for d_set in datasets:
                if isinstance(d_set, AvalancheDataset):
                    if uniform_group is None:
                        uniform_group = d_set._transform_groups.current_group
                    else:
                        if (
                            uniform_group
                            != d_set._transform_groups.current_group
                        ):
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
        if (
            task_labels is not None
        ):  # User defined targets always take precedence
            if isinstance(task_labels, int):
                task_labels = ConstantSequence(task_labels, totlen)
            elif len(task_labels) != totlen:
                raise ValueError(
                    "Invalid amount of target labels. It must be equal to the "
                    "number of patterns in the dataset. Got {}, expected "
                    "{}!".format(len(task_labels), totlen)
                )
            das.append(
                DataAttribute(
                    task_labels, "targets_task_labels", use_in_getitem=True
                )
            )

        if targets is not None:  # User defined targets always take precedence
            if isinstance(targets, int):
                targets = ConstantSequence(targets, totlen)
            elif len(targets) != totlen:
                raise ValueError(
                    "Invalid amount of target labels. It must be equal to the "
                    "number of patterns in the dataset. Got {}, expected "
                    "{}!".format(len(targets), totlen)
                )
            das.append(DataAttribute(targets, "targets"))
    if len(das) == 0:
        das = None
    data = ClassificationDataset(
        dds, transform_groups=transform_groups, data_attributes=das
    )
    return data.with_transforms(initial_transform_group)


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


class TaskSet(Mapping):
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

    def __init__(self, data: AvalancheDataset):
        """Constructor.

        :param data: original data
        """
        super().__init__()
        self.data = data

    def __iter__(self):
        return iter(self.data.targets_task_labels.uniques)

    def __getitem__(self, task_label):
        tl_idx = self.data.targets_task_labels.val_to_idx[task_label]
        return classification_subset(self.data, tl_idx)

    def __len__(self):
        return len(self.data.targets_task_labels.uniques)


__all__ = [
    "SupportedDataset",
    "make_classification_dataset",
    "classification_subset",
    "make_tensor_classification_dataset",
    "concat_classification_datasets",
    "TaskSet",
]
