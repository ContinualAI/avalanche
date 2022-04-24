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
from enum import Enum, auto
from typing import (Any, Callable, Collection, Dict, List, Optional, Sequence,
                    SupportsInt, Tuple, TypeVar, Union)

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import ConcatDataset, Dataset, Subset
from typing_extensions import Protocol

from .adaptive_transform import Compose, MultiParamTransform
from .dataset_definitions import (ClassificationDataset, IDatasetWithTargets,
                                  ISupportedClassificationDataset,
                                  ITensorDataset)
from .dataset_utils import (ClassificationSubset, ConstantSequence,
                            LazyClassMapping, LazyConcatIntTargets,
                            LazyConcatTargets, SequenceDataset, SubSequence,
                            TupleTLabel, find_list_from_index,
                            manage_advanced_indexing, optimize_sequence)

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


class AvalancheDatasetType(Enum):
    UNDEFINED = auto()
    CLASSIFICATION = auto()
    REGRESSION = auto()
    SEGMENTATION = auto()


class AvalancheDataset(IDatasetWithTargets[T_co, TTargetType], Dataset[T_co]):
    """
    The Dataset used as the base implementation for Avalanche.

    Instances of this dataset are usually returned from benchmarks, but it can
    also be used in a completely standalone manner. This dataset can be used
    to apply transformations before returning patterns/targets, it supports
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
        dataset_type: AvalancheDatasetType = None,
        collate_fn: Callable[[List], Any] = None,
        targets_adapter: Callable[[Any], TTargetType] = None,
    ):
        """
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
            "0" will be applied to all instances.
        :param targets: The label of each pattern. Defaults to None, which
            means that the targets will be retrieved from the dataset (if
            possible).
        :param dataset_type: The type of the dataset. Defaults to None,
            which means that the type will be inferred from the input dataset.
            When the `dataset_type` is different than UNDEFINED, a
            proper value for `collate_fn` and `targets_adapter` will be set.
            If the `dataset_type` is different than UNDEFINED, then
            `collate_fn` and `targets_adapter` must not be set.
        :param collate_fn: The function to use when slicing to merge single
            patterns. In the future this function may become the function
            used in the data loading process, too. If None and the
            `dataset_type` is UNDEFINED, the constructor will check if a
            `collate_fn` field exists in the dataset. If no such field exists,
            the default collate function will be used.
        :param targets_adapter: A function used to convert the values of the
            targets field. Defaults to None. Note: the adapter will not change
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

        detected_type = False
        if dataset_type is None:
            detected_type = True
            if isinstance(dataset, AvalancheDataset):
                dataset_type = dataset.dataset_type
            else:
                dataset_type = AvalancheDatasetType.UNDEFINED

        if dataset_type != AvalancheDatasetType.UNDEFINED and (
            collate_fn is not None or targets_adapter is not None
        ):
            if detected_type:
                raise ValueError(
                    "dataset_type {} was inferred from the input dataset. "
                    "This dataset type can't be used "
                    "with custom collate_fn or targets_adapter "
                    "parameters. Only the UNDEFINED type supports "
                    "custom collate_fn or targets_adapter "
                    "parameters".format(dataset_type)
                )
            else:
                raise ValueError(
                    "dataset_type {} can't be used with custom collate_fn "
                    "or targets_adapter. Only the UNDEFINED type supports "
                    "custom collate_fn or targets_adapter "
                    "parameters.".format(dataset_type)
                )

        if transform_groups is not None:
            AvalancheDataset._check_groups_dict_format(transform_groups)

        if not isinstance(dataset_type, AvalancheDatasetType):
            raise ValueError(
                "dataset_type must be a value of type " "AvalancheDatasetType"
            )

        self._dataset: SupportedDataset = dataset
        """
        The original dataset.
        """

        self.dataset_type = dataset_type
        """
        The type of this dataset (UNDEFINED, CLASSIFICATION, ...).
        """

        self.targets: Sequence[TTargetType] = self._initialize_targets_sequence(
            dataset, targets, dataset_type, targets_adapter
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
            dataset, dataset_type, collate_fn
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
            if isinstance(dataset, AvalancheDataset):
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

    def __add__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([self, other])

    def __radd__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([other, self])

    def __getitem__(self, idx) -> Union[T_co, Sequence[T_co]]:
        return TupleTLabel(
            manage_advanced_indexing(
                idx, self._get_single_item, len(self), self.collate_fn
            )
        )

    def __len__(self):
        return len(self._dataset)

    def train(self):
        """
        Returns a new dataset with the transformations of the 'train' group
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

    def freeze_transforms(self: TAvalancheDataset) -> TAvalancheDataset:
        """
        Returns a new dataset where the current transformations are frozen.

        Frozen transformations will be permanently glued to the original
        dataset so that they can't be changed anymore. This is usually done
        when using transformations to create derived datasets: in this way
        freezing the transformations will ensure that the user won't be able
        to inadvertently change them by directly setting the transformations
        field or by using the other transformations utility methods like
        ``replace_transforms``. Please note that transformations of all groups
        will be frozen. If you want to freeze a specific group, please use
        ``freeze_group_transforms``.

        The current dataset will not be affected.

        :return: A new dataset with the current transformations frozen.
        """

        dataset_copy = self._fork_dataset()

        for group_name in dataset_copy.transform_groups.keys():
            AvalancheDataset._freeze_dataset_group(dataset_copy, group_name)

        return dataset_copy

    def freeze_group_transforms(
        self: TAvalancheDataset, group_name: str
    ) -> TAvalancheDataset:
        """
        Returns a new dataset where the transformations for a specific group
        are frozen.

        Frozen transformations will be permanently glued to the original
        dataset so that they can't be changed anymore. This is usually done
        when using transformations to create derived datasets: in this way
        freezing the transformations will ensure that the user won't be able
        to inadvertently change them by directly setting the transformations
        field or by using the other transformations utility methods like
        ``replace_transforms``. To freeze transformations of all groups
        please use ``freeze_transforms``.

        The current dataset will not be affected.

        :return: A new dataset with the transformations frozen for the given
            group.
        """
        dataset_copy = self._fork_dataset()

        AvalancheDataset._freeze_dataset_group(dataset_copy, group_name)

        return dataset_copy

    def get_transforms(
        self: TAvalancheDataset, transforms_group: str = None
    ) -> Tuple[Any, Any]:
        """
        Returns the transformations given a group.

        Beware that this will not return the frozen transformations, nor the
        ones included in the wrapped dataset. Only transformations directly
        attached to this dataset will be returned.

        :param transforms_group: The transformations group. Defaults to None,
            which means that the current group is returned.
        :return: The transformation group, as a tuple
            (transform, target_transform).
        """
        if transforms_group is None:
            transforms_group = self.current_transform_group

        if transforms_group == self.current_transform_group:
            return self.transform, self.target_transform

        return self.transform_groups[transforms_group]

    def add_transforms(
        self: TAvalancheDataset,
        transform: Callable[[Any], Any] = None,
        target_transform: Callable[[int], int] = None,
    ) -> TAvalancheDataset:
        """
        Returns a new dataset with the given transformations added to
        the existing ones.

        The transformations will be added to the current transformations group.
        Other transformation groups will not be affected.

        The given transformations will be added "at the end" of previous
        transformations of the current transformations group. This means
        that existing transformations will be applied to the patterns first.

        The current dataset will not be affected.

        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :return: A new dataset with the added transformations.
        """

        dataset_copy = self._fork_dataset()

        if transform is not None:
            if dataset_copy.transform is not None:
                dataset_copy.transform = Compose(
                    [dataset_copy.transform, transform]
                )
            else:
                dataset_copy.transform = transform

        if target_transform is not None:
            if dataset_copy.target_transform is not None:
                dataset_copy.target_transform = Compose(
                    [dataset_copy.target_transform, target_transform]
                )
            else:
                dataset_copy.target_transform = target_transform

        return dataset_copy

    def add_transforms_to_group(
        self: TAvalancheDataset,
        group_name: str,
        transform: Callable[[Any], Any] = None,
        target_transform: Callable[[int], int] = None,
    ) -> TAvalancheDataset:
        """
        Returns a new dataset with the given transformations added to
        the existing ones for a certain group.

        The transformations will be added to the given transformations group.
        Other transformation groups will not be affected. The group must
        already exist.

        The given transformations will be added "at the end" of previous
        transformations of that group. This means that existing transformations
        will be applied to the patterns first.

        The current dataset will not be affected.

        :param group_name: The name of the group.
        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :return: A new dataset with the added transformations.
        """

        if self.current_transform_group == group_name:
            return self.add_transforms(transform, target_transform)

        if group_name not in self.transform_groups:
            raise ValueError("Invalid group name " + str(group_name))

        dataset_copy = self._fork_dataset()

        t_group: List[XTransform, YTransform] = list(
            dataset_copy.transform_groups[group_name]
        )
        if transform is not None:
            if t_group[0] is not None:
                t_group[0] = Compose([t_group[0], transform])
            else:
                t_group[0] = transform

        if target_transform is not None:
            if t_group[1] is not None:
                t_group[1] = Compose([t_group[1], target_transform])
            else:
                t_group[1] = target_transform

        # tuple(t_group) works too, but it triggers a type warning
        tuple_t_group: Tuple[XTransform, YTransform] = tuple(
            (t_group[0], t_group[1])
        )
        dataset_copy.transform_groups[group_name] = tuple_t_group

        return dataset_copy

    def replace_transforms(
        self: TAvalancheDataset,
        transform: XTransform,
        target_transform: YTransform,
        group: str = None,
    ) -> TAvalancheDataset:
        """
        Returns a new dataset with the existing transformations replaced with
        the given ones.

        The given transformations will replace the ones of the current
        transformations group. Other transformation groups will not be affected.

        If the original dataset is an instance of :class:`AvalancheDataset`,
        then transformations of the original set will be considered as well
        (the original dataset will be left untouched).

        The current dataset will not be affected.

        Note that this function will not override frozen transformations. This
        will also not affect transformations found in datasets that are not
        instances of :class:`AvalancheDataset`.

        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :param group: The transforms group to replace. Defaults to None, which
            means that the current group will be replaced.
        :return: A new dataset with the new transformations.
        """

        if group is None:
            group = self.current_transform_group

        dataset_copy = self._fork_dataset().with_transforms(group)
        dataset_copy._replace_original_dataset_group(None, None)

        dataset_copy.transform = transform
        dataset_copy.target_transform = target_transform

        return dataset_copy.with_transforms(self.current_transform_group)

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
        dataset_copy = self._fork_dataset()

        if group_name not in dataset_copy.transform_groups:
            raise ValueError("Invalid group name " + str(group_name))

        # Store current group (loaded in transform and target_transform fields)
        dataset_copy.transform_groups[dataset_copy.current_transform_group] = (
            dataset_copy.transform,
            dataset_copy.target_transform,
        )

        # Load new group in transform and target_transform fields
        switch_group = dataset_copy.transform_groups[group_name]
        dataset_copy.transform = switch_group[0]
        dataset_copy.target_transform = switch_group[1]
        dataset_copy.current_transform_group = group_name

        # Finally, align the underlying dataset
        dataset_copy._set_original_dataset_transform_group(group_name)

        return dataset_copy

    def add_transforms_group(
        self: TAvalancheDataset,
        group_name: str,
        transform: XTransform,
        target_transform: YTransform,
    ) -> TAvalancheDataset:
        """
        Returns a new dataset with a new transformations group.

        The current dataset will not be affected.

        This method raises an exception if a group with the same name already
        exists.

        :param group_name: The name of the new transformations group.
        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :return: A new dataset with the new transformations.
        """
        dataset_copy = self._fork_dataset()

        if group_name in dataset_copy.transform_groups:
            raise ValueError("A group with the same name already exists")

        dataset_copy.transform_groups[group_name] = (
            transform,
            target_transform,
        )

        AvalancheDataset._check_groups_dict_format(
            dataset_copy.transform_groups
        )

        dataset_copy._frozen_transforms[group_name] = (None, None)

        # Finally, align the underlying dataset
        dataset_copy._add_original_dataset_group(group_name)

        return dataset_copy

    def _fork_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = copy.copy(self)
        dataset_copy._frozen_transforms = dict(dataset_copy._frozen_transforms)
        dataset_copy.transform_groups = dict(dataset_copy.transform_groups)
        dataset_copy.task_set = dataset_copy._make_task_set_dict()

        return dataset_copy

    @staticmethod
    def _freeze_dataset_group(dataset_copy: TAvalancheDataset, group_name: str):
        # Freeze the current transformations. Frozen transformations are saved
        # in a separate dict.

        # This may rise an error if no group with the given name exists!
        frozen_group = dataset_copy._frozen_transforms[group_name]

        final_frozen_transform = frozen_group[0]
        final_frozen_target_transform = frozen_group[1]

        is_current_group = dataset_copy.current_transform_group == group_name

        # If the required group is not the current one, just freeze the ones
        # found in dataset_copy.transform_groups).
        to_be_glued = dataset_copy.transform_groups[group_name]

        # Now that transformations are stored in to_be_glued,
        # we can safely reset them in the transform_groups dictionary.
        dataset_copy.transform_groups[group_name] = (None, None)

        if is_current_group:
            # If the required group is the current one, use the transformations
            # already found in transform and target_transform fields (because
            # the ones stored in dataset_copy.transform_groups may be not
            # up-to-date).

            to_be_glued = (
                dataset_copy.transform,
                dataset_copy.target_transform,
            )

            # And of course, once frozen, set transformations to None
            dataset_copy.transform = None
            dataset_copy.target_transform = None

        if to_be_glued[0] is not None:
            if frozen_group[0] is None:
                final_frozen_transform = to_be_glued[0]
            else:
                final_frozen_transform = Compose(
                    [frozen_group[0], to_be_glued[0]]
                )

        if to_be_glued[1] is not None:
            if frozen_group[1] is None:
                final_frozen_target_transform = to_be_glued[1]
            else:
                final_frozen_target_transform = Compose(
                    [frozen_group[1], to_be_glued[1]]
                )

        # Set the frozen transformations
        dataset_copy._frozen_transforms[group_name] = (
            final_frozen_transform,
            final_frozen_target_transform,
        )

        # Finally, apply the freeze procedure to the original dataset
        dataset_copy._freeze_original_dataset(group_name)

    def _get_single_item(self, idx: int):
        return self._process_pattern(self._dataset[idx], idx)

    def _process_pattern(self, element: Tuple, idx: int):
        has_task_label = isinstance(element, TupleTLabel)
        if has_task_label:
            element = element[:-1]

        element = self._apply_transforms(element)

        return TupleTLabel((*element, self.targets_task_labels[idx]))

    def _apply_transforms(self, element: Sequence[Any]):
        element = list(element)
        frozen_group = self._frozen_transforms[self.current_transform_group]

        # Target transform
        if frozen_group[1] is not None:
            element[1] = frozen_group[1](element[1])

        if self.target_transform is not None:
            element[1] = self.target_transform(element[1])

        if frozen_group[0] is not None:
            element = MultiParamTransform(frozen_group[0])(*element)

        if self.transform is not None:
            element = MultiParamTransform(self.transform)(*element)

        return element

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
        self, dataset, targets, dataset_type, targets_adapter
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

        if targets_adapter is None:
            if dataset_type == AvalancheDatasetType.CLASSIFICATION:
                targets_adapter = int
            else:
                targets_adapter = None
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

    def _initialize_collate_fn(self, dataset, dataset_type, collate_fn):
        if collate_fn is not None:
            return collate_fn

        if dataset_type == AvalancheDatasetType.UNDEFINED:
            if hasattr(dataset, "collate_fn"):
                return getattr(dataset, "collate_fn")

        return default_collate

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

    def _set_original_dataset_transform_group(self, group_name: str) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            if self._dataset.current_transform_group == group_name:
                # Prevents a huge slowdown in some corner cases
                # (apart from being actually more performant)
                return

            self._dataset = self._dataset.with_transforms(group_name)

    def _freeze_original_dataset(self, group_name: str) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            self._dataset = self._dataset.freeze_group_transforms(group_name)

    def _replace_original_dataset_group(
        self, transform: XTransform, target_transform: YTransform
    ) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            self._dataset = self._dataset.replace_transforms(
                transform, target_transform
            )

    def _add_original_dataset_group(self, group_name: str) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            self._dataset = self._dataset.add_transforms_group(
                group_name, None, None
            )

    def _add_groups_from_original_dataset(
        self, dataset, transform_groups
    ) -> None:
        if isinstance(dataset, AvalancheDataset):
            for original_dataset_group in dataset.transform_groups.keys():
                if original_dataset_group not in transform_groups:
                    transform_groups[original_dataset_group] = (None, None)

    def _has_own_transformations(self):
        # Used to check if the current dataset has its own transformations
        # This method returns False if transformations are applied
        # by the wrapped dataset only.

        if self.transform is not None:
            return True

        if self.target_transform is not None:
            return True

        for transform_group in self.transform_groups.values():
            for transform in transform_group:
                if transform is not None:
                    return True

        for transform_group in self._frozen_transforms.values():
            for transform in transform_group:
                if transform is not None:
                    return True

        return False

    @staticmethod
    def _borrow_transformations(
        dataset, transform_groups, frozen_transform_groups
    ):
        if not isinstance(dataset, AvalancheDataset):
            return

        for original_dataset_group in dataset.transform_groups.keys():
            if original_dataset_group not in transform_groups:
                transform_groups[original_dataset_group] = (None, None)

        for original_dataset_group in dataset._frozen_transforms.keys():
            if original_dataset_group not in frozen_transform_groups:
                frozen_transform_groups[original_dataset_group] = (None, None)

        # Standard transforms
        for original_dataset_group in dataset.transform_groups.keys():
            other_dataset_transforms = dataset.transform_groups[
                original_dataset_group
            ]
            if dataset.current_transform_group == original_dataset_group:
                other_dataset_transforms = (
                    dataset.transform,
                    dataset.target_transform,
                )

            transform_groups[
                original_dataset_group
            ] = AvalancheDataset._prepend_transforms(
                transform_groups[original_dataset_group],
                other_dataset_transforms,
            )

        # Frozen transforms
        for original_dataset_group in dataset._frozen_transforms.keys():
            other_dataset_transforms = dataset._frozen_transforms[
                original_dataset_group
            ]

            frozen_transform_groups[
                original_dataset_group
            ] = AvalancheDataset._prepend_transforms(
                frozen_transform_groups[original_dataset_group],
                other_dataset_transforms,
            )

    @staticmethod
    def _prepend_transforms(transforms, to_be_prepended):
        if len(transforms) != 2:
            raise ValueError(
                "Transformation groups must contain exactly 2 transformations"
            )

        if len(transforms) != len(to_be_prepended):
            raise ValueError(
                "Transformation group size mismatch: {} vs {}".format(
                    len(transforms), len(to_be_prepended)
                )
            )

        result = []

        for i in range(len(transforms)):
            if to_be_prepended[i] is None:
                # Nothing to prepend
                result.append(transforms[i])
            elif transforms[i] is None:
                result.append(to_be_prepended[i])
            else:
                result.append(Compose([to_be_prepended[i], transforms[i]]))

        return tuple(result)  # Transform to tuple

    def _optimize_targets(self):
        self.targets = optimize_sequence(self.targets)

    def _optimize_task_labels(self):
        self.targets_task_labels = optimize_sequence(self.targets_task_labels)

    def _optimize_task_dict(self):
        for task_label in self.tasks_pattern_indices:
            self.tasks_pattern_indices[task_label] = optimize_sequence(
                self.tasks_pattern_indices[task_label]
            )

    def _flatten_dataset(self):
        pass

    def _make_task_set_dict(self) -> Dict[int, "AvalancheDataset"]:
        task_dict = _TaskSubsetDict()
        for task_id in sorted(self.tasks_pattern_indices.keys()):
            task_indices = self.tasks_pattern_indices[task_id]
            task_dict[task_id] = (self, task_indices)

        return task_dict


class _TaskSubsetDict(OrderedDict):
    def __getitem__(self, item):
        avl_dataset, indices = super().__getitem__(item)
        return _TaskSubsetDict._make_subset(avl_dataset, indices)

    @staticmethod
    def _make_subset(avl_dataset, indices: Sequence[int]):
        return AvalancheSubset(avl_dataset, indices=indices)


class AvalancheSubset(AvalancheDataset[T_co, TTargetType]):
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
        dataset_type: AvalancheDatasetType = None,
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
        :param dataset_type: The type of the dataset. Defaults to None,
            which means that the type will be inferred from the input dataset.
            When the `dataset_type` is different than UNDEFINED, a
            proper value for `collate_fn` and `targets_adapter` will be set.
            If the `dataset_type` is different than UNDEFINED, then
            `collate_fn` and `targets_adapter` must not be set.
            The only exception to this rule regards `class_mapping`.
            If `class_mapping` is set, the final dataset_type
            (as set by this parameter or detected from the subset) must be
            CLASSIFICATION or UNDEFINED.
        :param collate_fn: The function to use when slicing to merge single
            patterns. In the future this function may become the function
            used in the data loading process, too. If None and the
            `dataset_type` is UNDEFINED, the constructor will check if a
            `collate_fn` field exists in the dataset. If no such field exists,
            the default collate function will be used.
        :param targets_adapter: A function used to convert the values of the
            targets field. Defaults to None. Note: the adapter will not change
            the value of the second element returned by `__getitem__`.
            The adapter is used to adapt the values of the targets field only.
        """

        detected_type = False
        if dataset_type is None:
            detected_type = True
            if isinstance(dataset, AvalancheDataset):
                dataset_type = dataset.dataset_type
                if dataset_type == AvalancheDatasetType.UNDEFINED:
                    if collate_fn is None:
                        collate_fn = dataset.collate_fn
            else:
                dataset_type = AvalancheDatasetType.UNDEFINED

        if dataset_type != AvalancheDatasetType.UNDEFINED and (
            collate_fn is not None or targets_adapter is not None
        ):
            if detected_type:
                raise ValueError(
                    "dataset_type {} was inferred from the input dataset. "
                    "This dataset type can't be used "
                    "with custom collate_fn or targets_adapter "
                    "parameters. Only the UNDEFINED type supports "
                    "custom collate_fn or targets_adapter "
                    "parameters".format(dataset_type)
                )
            else:
                raise ValueError(
                    "dataset_type {} can't be used with custom collate_fn "
                    "or targets_adapter. Only the UNDEFINED type supports "
                    "custom collate_fn or targets_adapter "
                    "parameters.".format(dataset_type)
                )

        if class_mapping is not None:
            if dataset_type not in [
                AvalancheDatasetType.CLASSIFICATION,
                AvalancheDatasetType.UNDEFINED,
            ]:
                raise ValueError(
                    "class_mapping is defined but the dataset type"
                    " is neither CLASSIFICATION or UNDEFINED."
                )

        if class_mapping is not None:
            subset = ClassificationSubset(
                dataset, indices=indices, class_mapping=class_mapping
            )
        elif indices is not None:
            subset = Subset(dataset, indices=indices)
        else:
            subset = dataset  # Exactly like a plain AvalancheDataset

        self._original_dataset = dataset
        # self._indices and self._class_mapping currently not used apart from
        # initialization procedures
        self._class_mapping = class_mapping
        self._indices = indices

        if initial_transform_group is None:
            if isinstance(dataset, AvalancheDataset):
                initial_transform_group = dataset.current_transform_group
            else:
                initial_transform_group = "train"

        super().__init__(
            subset,
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=task_labels,
            targets=targets,
            dataset_type=dataset_type,
            collate_fn=collate_fn,
            targets_adapter=targets_adapter,
        )

    def _initialize_targets_sequence(
        self, dataset, targets, dataset_type, targets_adapter
    ) -> Sequence[TTargetType]:
        if targets is not None:
            # For the reasoning behind this, have a look at
            # _initialize_task_labels_sequence (it's basically the same).

            if len(targets) == len(self._original_dataset) and not len(
                targets
            ) == len(dataset):
                return SubSequence(targets, indices=self._indices)
            elif len(targets) == len(dataset):
                return targets
            else:
                raise ValueError(
                    "Invalid amount of targets. It must be equal to the "
                    "number of patterns in the subset. "
                    "Got {}, expected {}!".format(len(targets), len(dataset))
                )

        return super()._initialize_targets_sequence(
            dataset, None, dataset_type, targets_adapter
        )

    def _initialize_task_labels_sequence(
        self, dataset, task_labels: Optional[Sequence[int]]
    ) -> Sequence[int]:

        if task_labels is not None:
            # The task_labels parameter is kind of ambiguous...
            # it may either be the list of task labels of the required subset
            # or it may be the list of task labels of the original dataset.
            # Simple solution: check the length of task_labels!
            # However, if task_labels, the original dataset and this subset have
            # the same size, then task_labels is considered to contain the task
            # labels for the subset!

            if isinstance(task_labels, int):
                # Simplest case: constant task label
                return ConstantSequence(task_labels, len(dataset))
            elif len(task_labels) == len(self._original_dataset) and not len(
                task_labels
            ) == len(dataset):
                # task_labels refers to the original dataset ...
                return SubSequence(
                    task_labels, indices=self._indices, converter=int
                )
            elif len(task_labels) == len(dataset):
                # One label for each instance
                return SubSequence(task_labels, converter=int)
            else:
                raise ValueError(
                    "Invalid amount of task labels. It must be equal to the "
                    "number of patterns in the subset. "
                    "Got {}, expected {}!".format(
                        len(task_labels), len(dataset)
                    )
                )

        return super()._initialize_task_labels_sequence(dataset, None)

    def _set_original_dataset_transform_group(self, group_name: str) -> None:
        if isinstance(self._original_dataset, AvalancheDataset):
            if self._original_dataset.current_transform_group == group_name:
                # Prevents a huge slowdown in some corner cases
                # (apart from being actually more performant)
                return

            self._original_dataset = self._original_dataset.with_transforms(
                group_name
            )

            self._replace_original_dataset_reference()

    def _freeze_original_dataset(self, group_name: str) -> None:
        if isinstance(self._original_dataset, AvalancheDataset):
            self._original_dataset = (
                self._original_dataset.freeze_group_transforms(group_name)
            )

            self._replace_original_dataset_reference()

    def _replace_original_dataset_group(
        self, transform: XTransform, target_transform: YTransform
    ) -> None:
        if isinstance(self._original_dataset, AvalancheDataset):
            self._original_dataset = self._original_dataset.replace_transforms(
                transform, target_transform
            )

            self._replace_original_dataset_reference()

    def _add_original_dataset_group(self, group_name: str) -> None:
        if isinstance(self._original_dataset, AvalancheDataset):
            self._original_dataset = (
                self._original_dataset.add_transforms_group(
                    group_name, None, None
                )
            )

            self._replace_original_dataset_reference()

    def _add_groups_from_original_dataset(
        self, dataset, transform_groups
    ) -> None:
        if isinstance(self._original_dataset, AvalancheDataset):
            for (
                original_dataset_group
            ) in self._original_dataset.transform_groups.keys():
                if original_dataset_group not in transform_groups:
                    transform_groups[original_dataset_group] = (None, None)

    def _flatten_dataset(self):
        # Flattens this subset by borrowing indices and class mappings from
        # the original dataset (if it's an AvalancheSubset or PyTorch Subset)

        if isinstance(self._original_dataset, AvalancheSubset):
            # In order to flatten the subset, we have to integrate the
            # transformations (also frozen ones!)
            AvalancheDataset._borrow_transformations(
                self._original_dataset,
                self.transform_groups,
                self._frozen_transforms,
            )

            # We need to reload transformations after borrowing from the subset
            # This assumes that _flatten_dataset is called by __init__!
            self.transform, self.target_transform = self.transform_groups[
                self.current_transform_group
            ]

            forward_dataset = self._original_dataset._original_dataset
            forward_indices = self._original_dataset._indices
            forward_class_mapping = self._original_dataset._class_mapping

            if self._class_mapping is not None:

                if forward_class_mapping is not None:

                    new_class_mapping = []
                    for mapped_class in forward_class_mapping:
                        # -1 is sometimes used to mark unused classes
                        # shouldn't be a problem (if it is, is not our fault)
                        if mapped_class == -1:
                            forward_mapped = -1
                        else:
                            # forward_mapped may be -1
                            forward_mapped = self._class_mapping[mapped_class]

                        new_class_mapping.append(forward_mapped)
                else:
                    new_class_mapping = self._class_mapping
            else:
                new_class_mapping = forward_class_mapping  # May be None

            if self._indices is not None:
                if forward_indices is not None:
                    new_indices = [forward_indices[x] for x in self._indices]
                else:
                    new_indices = self._indices
            else:
                new_indices = forward_indices  # May be None

            self._original_dataset = forward_dataset
            self._indices = new_indices
            self._class_mapping = new_class_mapping
            if new_class_mapping is None:
                # Subset
                self._dataset = Subset(forward_dataset, indices=new_indices)
            else:
                # ClassificationSubset
                self._dataset = ClassificationSubset(
                    forward_dataset,
                    indices=new_indices,
                    class_mapping=new_class_mapping,
                )

        # --------
        # Flattening PyTorch Subset has been temporarily
        # disabled as the semantic of transformation groups collide
        # with the flattening process: PyTorch Subset doesn't have
        # transform groups and flattening it will expose the underlying
        # dataset, which may contain 'AvalancheDataset's.
        # --------

        # elif isinstance(self._original_dataset, Subset):
        #     # Very simple: just borrow indices (no transformations or
        #     # class mappings to adapt here!)
        #     forward_dataset = self._original_dataset.dataset
        #     forward_indices = self._original_dataset.indices
        #
        #     if self._indices is not None:
        #         new_indices = [forward_indices[x] for x in self._indices]
        #     else:
        #         new_indices = forward_indices
        #
        #     self._original_dataset = forward_dataset
        #     self._indices = new_indices
        #
        #     if self._class_mapping is not None:
        #         self._dataset = ClassificationSubset(
        #             forward_dataset, indices=new_indices,
        #             class_mapping=self._class_mapping)
        #
        #     elif self._indices is not None:
        #         self._dataset = Subset(forward_dataset, indices=new_indices)

    def _replace_original_dataset_reference(self):
        if isinstance(self._dataset, ClassificationSubset):
            self._dataset = ClassificationSubset(
                self._original_dataset,
                indices=self._indices,
                class_mapping=self._class_mapping,
            )
        elif isinstance(self._dataset, Subset):
            self._dataset = Subset(
                self._original_dataset, indices=self._indices
            )
        else:
            self._dataset = self._original_dataset


class AvalancheTensorDataset(AvalancheDataset[T_co, TTargetType]):
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
        dataset_type: AvalancheDatasetType = AvalancheDatasetType.UNDEFINED,
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
        :param dataset_type: The type of the dataset. Defaults to UNDEFINED.
            Setting this parameter will automatically set a proper value for
            `collate_fn` and `targets_adapter`. If this parameter is set to a
            value different from UNDEFINED then `collate_fn` and
            `targets_adapter` must not be set.
        :param collate_fn: The function to use when slicing to merge single
            patterns. In the future this function may become the function
            used in the data loading process, too.
        :param targets_adapter: A function used to convert the values of the
            targets field. Defaults to None. Note: the adapter will not change
            the value of the second element returned by `__getitem__`.
            The adapter is used to adapt the values of the targets field only.
        """

        if dataset_type != AvalancheDatasetType.UNDEFINED and (
            collate_fn is not None or targets_adapter is not None
        ):
            raise ValueError(
                "dataset_type {} can't be used with custom collate_fn "
                "or targets_adapter. Only the UNDEFINED type supports "
                "custom collate_fn or targets_adapter "
                "parameters.".format(dataset_type)
            )

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
            dataset_type=dataset_type,
            collate_fn=collate_fn,
            targets_adapter=targets_adapter,
        )


class AvalancheConcatDataset(AvalancheDataset[T_co, TTargetType]):
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
        dataset_type: AvalancheDatasetType = None,
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
        :param dataset_type: The type of the dataset. Defaults to None,
            which means that the type will be inferred from the list of
            input datasets. When `dataset_type` is None and the list of datasets
            contains incompatible types, an error will be raised.
            A list of datasets is compatible if they all have
            the same type. Datasets that are not instances of `AvalancheDataset`
            and instances of `AvalancheDataset` with type `UNDEFINED`
            are always compatible with other types.
            When the `dataset_type` is different than UNDEFINED, a
            proper value for `collate_fn` and `targets_adapter` will be set.
            If the `dataset_type` is different than UNDEFINED, then
            `collate_fn` and `targets_adapter` must not be set.
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
        dataset_list = list(datasets)

        (
            dataset_type,
            collate_fn,
            targets_adapter,
        ) = self._get_dataset_type_collate_and_adapter(
            dataset_list, dataset_type, collate_fn, targets_adapter
        )

        self._dataset_list = dataset_list
        self._datasets_lengths = [len(dataset) for dataset in dataset_list]
        self._datasets_cumulative_lengths = ConcatDataset.cumsum(dataset_list)
        self._overall_length = sum(self._datasets_lengths)

        if initial_transform_group is None:
            uniform_group = None
            for d_set in self._dataset_list:
                if isinstance(d_set, AvalancheDataset):
                    if uniform_group is None:
                        uniform_group = d_set.current_transform_group
                    else:
                        if uniform_group != d_set.current_transform_group:
                            uniform_group = None
                            break

            if uniform_group is None:
                initial_transform_group = "train"
            else:
                initial_transform_group = uniform_group

        if task_labels is not None:
            task_labels = self._concat_task_labels(task_labels)

        if targets is not None:
            targets = self._concat_targets(targets)

        self._adapt_concat_datasets()

        super().__init__(
            ClassificationDataset(),  # not used
            transform=transform,
            target_transform=target_transform,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
            task_labels=task_labels,
            targets=targets,
            dataset_type=dataset_type,
            collate_fn=collate_fn,
            targets_adapter=targets_adapter,
        )

    def _get_dataset_type_collate_and_adapter(
        self, datasets, dataset_type, collate_fn, targets_adapter
    ):

        if dataset_type is not None:
            return dataset_type, collate_fn, targets_adapter

        identified_types = set()
        first_collate_fn = None

        for dataset in datasets:
            if isinstance(dataset, AvalancheDataset):
                if dataset.dataset_type != AvalancheDatasetType.UNDEFINED:
                    identified_types.add(dataset.dataset_type)

            if first_collate_fn is None:
                first_collate_fn = getattr(dataset, "collate_fn", None)

        if len(identified_types) > 1:
            raise ValueError(
                "Error trying to infer a common dataset type while "
                "concatenating different datasets. "
                "Incompatible types: {}".format(list(identified_types))
            )
        elif len(identified_types) == 0:
            dataset_type = AvalancheDatasetType.UNDEFINED
        else:
            # len(identified_types) == 1
            dataset_type = next(iter(identified_types))

        if dataset_type != AvalancheDatasetType.UNDEFINED and (
            collate_fn is not None or targets_adapter is not None
        ):
            raise ValueError(
                "dataset_type {} was inferred from the list of "
                "concatenated dataset. This dataset type can't be used "
                "with custom collate_fn or targets_adapter "
                "parameters. Only the UNDEFINED type supports "
                "custom collate_fn or targets_adapter "
                "parameters.".format(dataset_type)
            )

        if (
            collate_fn is None
            and dataset_type == AvalancheDatasetType.UNDEFINED
        ):
            collate_fn = first_collate_fn

        return dataset_type, collate_fn, targets_adapter

    def __len__(self) -> int:
        return self._overall_length

    def _get_single_item(self, idx: int):
        dataset_idx, internal_idx = find_list_from_index(
            idx,
            self._datasets_lengths,
            self._overall_length,
            cumulative_sizes=self._datasets_cumulative_lengths,
        )

        single_element = self._dataset_list[dataset_idx][internal_idx]

        return self._process_pattern(single_element, idx)

    def _fork_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = super()._fork_dataset()

        dataset_copy._dataset_list = list(dataset_copy._dataset_list)
        # Note: there is no need to duplicate _datasets_lengths

        return dataset_copy

    def _initialize_targets_sequence(
        self, dataset, targets, dataset_type, targets_adapter
    ) -> Sequence[TTargetType]:
        if targets is not None:
            if len(targets) != self._overall_length:
                raise ValueError(
                    "Invalid amount of target labels. It must be equal to the "
                    "number of patterns in the dataset. Got {}, expected "
                    "{}!".format(len(targets), self._overall_length)
                )

            return targets

        targets_list = []
        # Could be easily done with a single line of code
        # This however, allows the user to better check which was the
        # problematic dataset by using a debugger.
        for dataset_idx, single_dataset in enumerate(self._dataset_list):
            targets_list.append(
                super()._initialize_targets_sequence(
                    single_dataset, None, dataset_type, targets_adapter
                )
            )

        return LazyConcatTargets(targets_list)

    def _initialize_task_labels_sequence(
        self, dataset, task_labels: Optional[Sequence[int]]
    ) -> Sequence[int]:
        if task_labels is not None:
            # task_labels has priority over the dataset fields

            if isinstance(task_labels, int):
                return ConstantSequence(task_labels, self._overall_length)
            elif len(task_labels) != self._overall_length:
                raise ValueError(
                    "Invalid amount of task labels. It must be equal to the "
                    "number of patterns in the dataset. Got {}, expected "
                    "{}!".format(len(task_labels), self._overall_length)
                )
            return SubSequence(task_labels, converter=int)

        concat_t_labels = []
        for dataset_idx, single_dataset in enumerate(self._dataset_list):
            concat_t_labels.append(
                super()._initialize_task_labels_sequence(single_dataset, None)
            )

        return LazyConcatTargets(concat_t_labels)

    def _initialize_collate_fn(self, dataset, dataset_type, collate_fn):
        if collate_fn is not None:
            return collate_fn

        if len(self._dataset_list) > 0 and hasattr(
            self._dataset_list[0], "collate_fn"
        ):
            return getattr(self._dataset_list[0], "collate_fn")
        return default_collate

    def _set_original_dataset_transform_group(self, group_name: str) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                if dataset.current_transform_group == group_name:
                    # Prevents a huge slowdown in some corner cases
                    # (apart from being actually more performant)
                    continue

                self._dataset_list[dataset_idx] = dataset.with_transforms(
                    group_name
                )

    def _freeze_original_dataset(self, group_name: str) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                self._dataset_list[
                    dataset_idx
                ] = dataset.freeze_group_transforms(group_name)

    def _replace_original_dataset_group(
        self, transform: XTransform, target_transform: YTransform
    ) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                self._dataset_list[dataset_idx] = dataset.replace_transforms(
                    transform, target_transform
                )

    def _add_original_dataset_group(self, group_name: str) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                self._dataset_list[dataset_idx] = dataset.add_transforms_group(
                    group_name, None, None
                )

    def _add_groups_from_original_dataset(
        self, dataset, transform_groups
    ) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                for original_dataset_group in dataset.transform_groups.keys():
                    if original_dataset_group not in transform_groups:
                        transform_groups[original_dataset_group] = (None, None)

    def _adapt_concat_datasets(self):
        all_groups = set()

        for dataset in self._dataset_list:
            if isinstance(dataset, AvalancheDataset):
                all_groups.update(dataset.transform_groups.keys())

        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                for group_name in all_groups:
                    if group_name not in dataset.transform_groups:
                        self._dataset_list[
                            dataset_idx
                        ] = dataset.add_transforms_group(group_name, None, None)

    @staticmethod
    def _concat_task_labels(
        task_labels: Union[int, Sequence[int], Sequence[Sequence[int]]]
    ):
        if isinstance(task_labels, int):
            # A single value has been passed -> use it for all instances
            # The value is returned as is because it's already managed when in
            # this form (in _initialize_task_labels_sequence).
            return task_labels
        elif isinstance(task_labels[0], int):
            # Flat list of task labels -> just return it.
            # The constructor will check if it has the correct size.
            return task_labels
        else:
            # One list for each dataset, concat them.
            return LazyConcatIntTargets(task_labels)

    @staticmethod
    def _concat_targets(
        targets: Union[Sequence[TTargetType], Sequence[Sequence[TTargetType]]]
    ):
        if isinstance(targets[0], Sequence):
            return LazyConcatTargets(targets)
        else:
            return targets

    def _flatten_dataset(self):
        # Flattens this subset by borrowing the list of concatenated datasets
        # from the original datasets (if they're 'AvalancheConcatSubset's or
        # PyTorch 'ConcatDataset's)

        flattened_list = []
        for dataset in self._dataset_list:
            if isinstance(dataset, AvalancheConcatDataset):
                if dataset._has_own_transformations():
                    # Can't flatten as the dataset has custom transformations
                    flattened_list.append(dataset)
                else:
                    flattened_list.extend(dataset._dataset_list)

            # PyTorch ConcatDataset doesn't have custom transformations
            # --------
            # Flattening PyTorch ConcatDatasets has been temporarily
            # disabled as the semantic of transformation groups collide
            # with the flattening process: PyTorch ConcatDataset doesn't have
            # transform groups and flattening it will expose the underlying
            # concatenated datasets list, which may contain 'AvalancheDataset's.
            # --------
            # elif isinstance(dataset, ConcatDataset):
            #    flattened_list.extend(dataset.datasets)
            elif isinstance(dataset, AvalancheSubset):
                flattened_list += self._flatten_subset_concat_branch(dataset)
            else:
                flattened_list.append(dataset)

        self._dataset_list = flattened_list
        self._datasets_lengths = [len(dataset) for dataset in flattened_list]
        self._datasets_cumulative_lengths = ConcatDataset.cumsum(flattened_list)
        self._overall_length = sum(self._datasets_lengths)

    def _flatten_subset_concat_branch(
        self, dataset: AvalancheSubset
    ) -> List[Dataset]:
        """
        Optimizes the dataset hierarchy in the corner case:

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
                concat_dataset._overall_length,
                cumulative_sizes=concat_dataset._datasets_cumulative_lengths,
            )

            if last_c_dataset is None:
                last_c_dataset = dataset_idx
            elif last_c_dataset != dataset_idx:
                # Consolidate current subset
                result.append(
                    AvalancheConcatDataset._make_similar_subset(
                        dataset,
                        concat_dataset._dataset_list[last_c_dataset],
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
                    concat_dataset._dataset_list[last_c_dataset],
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
        if subset.dataset_type == AvalancheDatasetType.UNDEFINED:
            collate_fn = subset.collate_fn

        result = AvalancheSubset(
            ref_dataset,
            indices=indices,
            class_mapping=subset._class_mapping,
            transform_groups=t_groups,
            initial_transform_group=subset.current_transform_group,
            task_labels=tasks,
            targets=targets,
            dataset_type=subset.dataset_type,
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
    dataset_type: AvalancheDatasetType = None,
) -> AvalancheDataset[T_co, TTargetType]:
    if isinstance(dataset, AvalancheDataset) and dataset_type is None:
        # There is no need to show the warning
        return dataset

    if dataset_type is None:
        warnings.warn(
            '"as_avalanche_dataset" was called without setting '
            '"dataset_type": this behaviour is deprecated. Consider '
            "setting this value or calling the specific functions "
            '"as_classification_dataset", "as_regression_dataset", '
            '"as_segmentation_dataset" or "as_undefined_dataset"'
        )
        dataset_type = AvalancheDatasetType.UNDEFINED

    if (
        isinstance(dataset, AvalancheDataset)
        and dataset.dataset_type == dataset_type
    ):
        return dataset

    return AvalancheDataset(dataset, dataset_type=dataset_type)


def as_classification_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, int]:
    return as_avalanche_dataset(
        dataset, dataset_type=AvalancheDatasetType.CLASSIFICATION
    )


def as_regression_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, Any]:
    return as_avalanche_dataset(
        dataset, dataset_type=AvalancheDatasetType.REGRESSION
    )


def as_segmentation_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, Any]:
    return as_avalanche_dataset(
        dataset, dataset_type=AvalancheDatasetType.SEGMENTATION
    )


def as_undefined_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset[T_co, Any]:
    return as_avalanche_dataset(
        dataset, dataset_type=AvalancheDatasetType.UNDEFINED
    )


def train_eval_avalanche_datasets(
    train_dataset: ISupportedClassificationDataset,
    test_dataset: ISupportedClassificationDataset,
    train_transformation,
    eval_transformation,
    dataset_type=None,
):
    train = AvalancheDataset(
        train_dataset,
        transform_groups=dict(
            train=(train_transformation, None), eval=(eval_transformation, None)
        ),
        initial_transform_group="train",
        dataset_type=dataset_type,
    )

    test = AvalancheDataset(
        test_dataset,
        transform_groups=dict(
            train=(train_transformation, None), eval=(eval_transformation, None)
        ),
        initial_transform_group="eval",
        dataset_type=dataset_type,
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
    "AvalancheDatasetType",
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
