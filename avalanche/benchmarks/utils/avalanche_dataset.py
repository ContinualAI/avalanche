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

"""
This module contains the implementation of the Avalanche Dataset,
which is the standard Avalanche implementation of a PyTorch dataset. Despite
being a child class of the PyTorch Dataset, the AvalancheDataset (and its
derivatives) is much more powerful as it offers many more features
out-of-the-box.
"""
import copy
import warnings
from collections import OrderedDict, defaultdict, deque

from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset, Subset, ConcatDataset

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset, AvalancheConcatDataset
from .adaptive_transform import Compose, MultiParamTransform
from .dataset_utils import (
    manage_advanced_indexing,
    SequenceDataset,
    ClassificationSubset,
    LazyConcatIntTargets,
    find_list_from_index,
    ConstantSequence,
    LazyClassMapping,
    optimize_sequence,
    SubSequence,
    LazyConcatTargets,
    TupleTLabel,
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

from typing_extensions import Protocol

T_co = TypeVar("T_co", covariant=True)
TTargetType = TypeVar("TTargetType")

TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")


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


SupportedDataset = Union[
    IDatasetWithTargets, ITensorDataset, Subset, ConcatDataset
]


class AvalancheClassificationDataset(AvalancheDataset[T_co]):
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
    parameter, each pattern will be assigned a default task label "0".
    See the constructor for more details.
    """

    def __init__(
        self,
        dataset: SupportedDataset,
        *,
        transform: XTransform = None,
        target_transform: YTransform = None,
        transform_groups: Dict[str, TransformGroupDef] = None,
        initial_transform_group: str = None,
        task_labels: Union[int, Sequence[int]] = None,
        targets: Sequence[TTargetType] = None,
        collate_fn: Callable[[List], Any] = None,
        targets_adapter: Callable[[Any], TTargetType] = None,
    ):
        """Creates a ``AvalancheDataset`` instance.

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
            "0" will be applied to all instances.
        :param targets: The label of each pattern. Defaults to None, which
            means that the targets will be retrieved from the dataset (if
            possible).
        :param collate_fn: The function to use when slicing to merge single
            patterns.This function is the function
            used in the data loading process, too. If None
            the constructor will check if a
            `collate_fn` field exists in the dataset. If no such field exists,
            the default collate function will be used.
        :param targets_adapter: A function used to convert the values of the
            targets field. Defaults to int. Note: the adapter will not change
            the value of the second element returned by `__getitem__`.
            The adapter is used to adapt the values of the targets field only.
        """
        if transform_groups is not None and (
            transform is not None or target_transform is not None
        ):
            raise ValueError(
                "transform_groups can't be used with transform"
                "and target_transform values"
            )

        if transform_groups is not None:
            AvalancheClassificationDataset._check_groups_dict_format(transform_groups)

        self._dataset: SupportedDataset = dataset
        """
        The original dataset.
        """

        """
        The type of this dataset (UNDEFINED, CLASSIFICATION, ...).
        """

        self.targets: Sequence[TTargetType] = self._initialize_targets_sequence(
            dataset, targets, targets_adapter
        )
        """
        A sequence of values describing the label of each pattern contained in
        the dataset.
        """

        self.targets_task_labels: Sequence[
            int
        ] = self._initialize_task_labels_sequence(dataset, task_labels)
        """
        A sequence of ints describing the task label of each pattern contained 
        in the dataset.
        """

        self.tasks_pattern_indices: Dict[
            int, Sequence[int]
        ] = self._initialize_tasks_dict(dataset, self.targets_task_labels)
        """
        A dictionary mapping task labels to the indices of the patterns with 
        that task label. If you need to obtain the subset of patterns labeled
        with a certain task label, consider using the `task_set` field.
        """

        self.collate_fn = self._initialize_collate_fn(
            dataset, collate_fn
        )
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """

        # Compress targets and task labels to save some memory
        self._optimize_targets()
        self._optimize_task_labels()
        self._optimize_task_dict()

        self.task_set = self._make_task_set_dict()
        """
        A dictionary that can be used to obtain the subset of patterns given
        a specific task label.
        """

        if initial_transform_group is None:
            # Detect from the input dataset. If not an AvalancheDataset then
            # use 'train' as the initial transform group
            if isinstance(dataset, AvalancheClassificationDataset):
                initial_transform_group = dataset.current_transform_group
            else:
                initial_transform_group = "train"

        self.current_transform_group = initial_transform_group
        """
        The name of the transform group currently in use.
        """

        self.transform_groups: Dict[
            str, Tuple[XTransform, YTransform]
        ] = self._initialize_groups_dict(
            transform_groups, dataset, transform, target_transform
        )
        """
        A dictionary containing the transform groups. Transform groups are
        used to quickly switch between training and test (eval) transformations.
        This becomes useful when in need to test on the training dataset as test
        transformations usually don't contain random augmentations.

        AvalancheDataset natively supports switching between the 'train' and
        'eval' groups by calling the ``train()`` and ``eval()`` methods. When
        using custom groups one can use the ``with_transforms(group_name)``
        method instead.

        May be null, which means that the current transforms will be used to
        handle both 'train' and 'eval' groups.
        """

        if self.current_transform_group not in self.transform_groups:
            raise ValueError(
                "Invalid transformations group "
                + str(self.current_transform_group)
            )
        t_group = self.transform_groups[self.current_transform_group]

        self.transform: XTransform = t_group[0]
        """
        A function/transform that takes in an PIL image and returns a 
        transformed version.
        """

        self.target_transform: YTransform = t_group[1]
        """
        A function/transform that takes in the target and transforms it.
        """

        self._frozen_transforms: Dict[
            str, Tuple[XTransform, YTransform]
        ] = dict()
        """
        A dictionary containing frozen transformations.
        """

        for group_name in self.transform_groups.keys():
            self._frozen_transforms[group_name] = (None, None)

        self._set_original_dataset_transform_group(self.current_transform_group)

        self._flatten_dataset()

    def __add__(self, other: Dataset) -> "AvalancheClassificationDataset":
        return AvalancheConcatClassificationDataset([self, other])

    def __radd__(self, other: Dataset) -> "AvalancheClassificationDataset":
        return AvalancheConcatClassificationDataset([other, self])

    @staticmethod
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

            map_value = groups_dict[map_key]
            if not isinstance(map_value, tuple):
                raise ValueError(
                    'Transformations for group "'
                    + str(map_key)
                    + '" must be contained in a tuple'
                )

            if not len(map_value) == 2:
                raise ValueError(
                    'Transformations for group "' + str(map_key) + '" must be '
                    "a tuple containing 2 elements: a transformation for the X "
                    "values and a transformation for the Y values"
                )

        if "test" in groups_dict:
            warnings.warn(
                'A transformation group named "test" has been found. Beware '
                "that by default AvalancheDataset supports test transformations"
                ' through the "eval" group. Consider using that one!'
            )

    def _initialize_groups_dict(
        self,
        transform_groups: Optional[Dict[str, TransformGroupDef]],
        dataset: Any,
        transform: XTransform,
        target_transform: YTransform,
    ) -> Dict[str, Tuple[XTransform, YTransform]]:
        """
        A simple helper method that tries to fill the 'train' and 'eval'
        groups as those two groups must always exist.

        If no transform_groups are passed to the class constructor, then
        the transform and target_transform parameters are used for both groups.

        If train transformations are set and eval transformations are not, then
        train transformations will be used for the eval group.

        :param dataset: The original dataset. Will be used to detect existing
            groups.
        :param transform: The transformation passed as a parameter to the
            class constructor.
        :param target_transform: The target transformation passed as a parameter
            to the class constructor.
        """
        if transform_groups is None:
            transform_groups = {
                "train": (transform, target_transform),
                "eval": (transform, target_transform),
            }
        else:
            transform_groups = dict(transform_groups)

        for group_name, group_transforms in dict(transform_groups).items():
            if group_transforms is None:
                transform_groups[group_name] = (None, None)
            elif isinstance(group_transforms, Callable):
                # Single transformation: (safely) assume it is the X transform
                transform_groups[group_name] = (group_transforms, None)
            elif (
                isinstance(group_transforms, Sequence)
                and len(group_transforms) == 2
            ):
                # X and Y transforms
                transform_groups[group_name] = (
                    group_transforms[0],
                    group_transforms[1],
                )
            else:
                raise ValueError(
                    f"Unsupported transformations for group {group_name}. "
                    f"The transformation group may be None, a single Callable, "
                    f"or a tuple of 2 elements containing the X and Y "
                    f"transforms"
                )

        if "train" in transform_groups:
            if "eval" not in transform_groups:
                transform_groups["eval"] = transform_groups["train"]

        if "train" not in transform_groups:
            transform_groups["train"] = (None, None)

        if "eval" not in transform_groups:
            transform_groups["eval"] = (None, None)

        self._add_groups_from_original_dataset(dataset, transform_groups)

        return transform_groups

    def _initialize_targets_sequence(
        self, dataset, targets, targets_adapter
    ) -> Sequence[TTargetType]:
        if targets is not None:
            # User defined targets always take precedence
            # Note: no adapter is applied!
            if len(targets) != len(dataset):
                raise ValueError(
                    "Invalid amount of target labels. It must be equal to the "
                    "number of patterns in the dataset. Got {}, expected "
                    "{}!".format(len(targets), len(dataset))
                )
            return targets

        return _make_target_from_supported_dataset(dataset, targets_adapter)

    def _initialize_task_labels_sequence(
        self, dataset, task_labels: Optional[Sequence[int]]
    ) -> Sequence[int]:
        if task_labels is not None:
            # task_labels has priority over the dataset fields
            if isinstance(task_labels, int):
                task_labels = ConstantSequence(task_labels, len(dataset))
            elif len(task_labels) != len(dataset):
                raise ValueError(
                    "Invalid amount of task labels. It must be equal to the "
                    "number of patterns in the dataset. Got {}, expected "
                    "{}!".format(len(task_labels), len(dataset))
                )

            return SubSequence(task_labels, converter=int)

        return _make_task_labels_from_supported_dataset(dataset)

    def _initialize_tasks_dict(
        self, dataset, task_labels: Sequence[int]
    ) -> Dict[int, Sequence[int]]:
        if isinstance(task_labels, ConstantSequence) and len(task_labels) > 0:
            # Shortcut :)
            return {task_labels[0]: range(len(task_labels))}

        result = dict()
        for i, x in enumerate(task_labels):
            if x not in result:
                result[x] = []
            result[x].append(i)

        if len(result) == 1:
            result[next(iter(result.keys()))] = range(len(task_labels))

        return result


class AvalancheClassificationSubset(AvalancheClassificationDataset[T_co, TTargetType]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    This Dataset also supports transformations, slicing, advanced indexing,
    the targets field, class mapping and all the other goodies listed in
    :class:`AvalancheDataset`.
    """

    def __init__(
        self,
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
        collate_fn: Callable[[List], Any] = None,
        targets_adapter: Callable[[Any], TTargetType] = None,
    ):
        """
        Creates an ``AvalancheSubset`` instance.

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
            could be found, a default task label "0" will be applied to all
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
        :param targets_adapter: A function used to convert the values of the
            targets field. Defaults to None. Note: the adapter will not change
            the value of the second element returned by `__getitem__`.
            The adapter is used to adapt the values of the targets field only.
        """
        # TODO: remove class_mapping
        assert class_mapping is None
        data = AvalancheSubset(dataset, indices)
        super().__init__(data,
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=task_labels,
            targets=targets,
            collate_fn=collate_fn,
            targets_adapter=targets_adapter,
        )


class AvalancheTensorClassificationDataset(AvalancheClassificationDataset[T_co, TTargetType]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset from PyTorch,
    this Dataset also supports transformations, slicing, advanced indexing,
    the targets field and all the other goodies listed in
    :class:`AvalancheDataset`.
    """

    def __init__(
        self,
        *dataset_tensors: Sequence,
        transform: Callable[[Any], Any] = None,
        target_transform: Callable[[int], int] = None,
        transform_groups: Dict[str, Tuple[XTransform, YTransform]] = None,
        initial_transform_group: str = "train",
        task_labels: Union[int, Sequence[int]] = None,
        targets: Union[Sequence[TTargetType], int] = None,
        collate_fn: Callable[[List], Any] = None,
        targets_adapter: Callable[[Any], TTargetType] = None,
    ):
        """
        Creates a ``AvalancheTensorDataset`` instance.

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
            default task label "0" will be applied to all patterns.
        :param targets: The label of each pattern. Defaults to None, which
            means that the targets will be retrieved from the dataset.
            Otherwise, can be 1) a sequence of values containing as many
            elements as the number of patterns, or 2) the index of the sequence
            to use as the targets field. When using the default value of None,
            the targets field will be populated using the second
            tensor. If dataset is made of only one tensor, then that tensor will
            be used for the targets field, too.
        :param collate_fn: The function to use when slicing to merge single
            patterns. In the future this function may become the function
            used in the data loading process, too.
        :param targets_adapter: A function used to convert the values of the
            targets field. Defaults to None. Note: the adapter will not change
            the value of the second element returned by `__getitem__`.
            The adapter is used to adapt the values of the targets field only.
        """

        if len(dataset_tensors) < 1:
            raise ValueError("At least one sequence must be passed")

        if targets is None:
            targets = min(1, len(dataset_tensors))

        if isinstance(targets, int):
            base_dataset = SequenceDataset(*dataset_tensors, targets=targets)
            targets = None
        else:
            base_dataset = SequenceDataset(
                *dataset_tensors, targets=min(1, len(dataset_tensors))
            )

        super().__init__(
            base_dataset,
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=task_labels,
            targets=targets,
            collate_fn=collate_fn,
            targets_adapter=targets_adapter,
        )


class AvalancheConcatClassificationDataset(AvalancheClassificationDataset[T_co, TTargetType]):
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
        datasets: Collection[SupportedDataset],
        *,
        transform: Callable[[Any], Any] = None,
        target_transform: Callable[[int], int] = None,
        transform_groups: Dict[str, Tuple[XTransform, YTransform]] = None,
        initial_transform_group: str = None,
        task_labels: Union[int, Sequence[int], Sequence[Sequence[int]]] = None,
        targets: Union[
            Sequence[TTargetType], Sequence[Sequence[TTargetType]]
        ] = None,
        collate_fn: Callable[[List], Any] = None,
        targets_adapter: Callable[[Any], TTargetType] = None,
    ):
        """
        Creates a ``AvalancheConcatDataset`` instance.

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
            labels could be found for a dataset, a default task label "0" will
            be applied to all patterns of that dataset.
        :param collate_fn: The function to use when slicing to merge single
            patterns. In the future this function may become the function
            used in the data loading process, too. If None, the constructor
            will check if a `collate_fn` field exists in the first dataset. If
            no such field exists, the default collate function will be used.
            Beware that the chosen collate function will be applied to all
            the concatenated datasets even if a different collate is defined
            in different datasets.
        :param targets_adapter: A function used to convert the values of the
            targets field. Defaults to None. Note: the adapter will not change
            the value of the second element returned by `__getitem__`.
            The adapter is used to adapt the values of the targets field only.
        """
        super().__init__(
            AvalancheConcatDataset(datasets),
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=task_labels,
            targets=targets,
            collate_fn=collate_fn,
            targets_adapter=targets_adapter,
        )


def concat_datasets_sequentially(
    train_dataset_list: Sequence[ISupportedClassificationDataset],
    test_dataset_list: Sequence[ISupportedClassificationDataset],
) -> Tuple[AvalancheConcatClassificationDataset, AvalancheConcatClassificationDataset, List[list]]:
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
            AvalancheClassificationSubset(train_set, class_mapping=class_mapping)
        )
        remapped_test_datasets.append(
            AvalancheClassificationSubset(test_set, class_mapping=class_mapping)
        )
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return (
        AvalancheConcatClassificationDataset(remapped_train_datasets),
        AvalancheConcatClassificationDataset(remapped_test_datasets),
        new_class_ids_per_dataset,
    )


def as_avalanche_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheClassificationDataset[T_co, TTargetType]:
    if isinstance(dataset, AvalancheClassificationDataset):
        return dataset

    return AvalancheClassificationDataset(dataset)


def as_classification_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheClassificationDataset[T_co, int]:
    return as_avalanche_dataset(
        dataset
    )


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
    if isinstance(dataset, AvalancheClassificationDataset):
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
    if isinstance(dataset, AvalancheClassificationDataset):
        return dataset.targets_task_labels

    task_labels = _traverse_supported_dataset(dataset, _select_task_labels)

    return SubSequence(task_labels, converter=int)


__all__ = [
    "SupportedDataset",
    "AvalancheClassificationDataset",
    "AvalancheClassificationSubset",
    "AvalancheTensorClassificationDataset",
    "AvalancheConcatClassificationDataset",
    "concat_datasets_sequentially",
    "as_avalanche_dataset",
    "as_classification_dataset",
]
