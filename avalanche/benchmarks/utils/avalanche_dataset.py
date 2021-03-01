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
from collections import OrderedDict

import torch
from torchvision.transforms import Compose

from .dataset_utils import IDatasetWithTargets, \
    DatasetWithTargets, manage_advanced_indexing, \
    SequenceDataset, SubsetWithTargets, LazyTargetsConversion, \
    LazyConcatTargets, find_list_from_index, ITensorDataset, ConstantSequence, \
    LazyClassMapping, optimize_sequence

try:
    from typing import List, Any, Iterable, Sequence, Union, Optional, \
        TypeVar, Protocol, SupportsInt, Generic, Callable, Dict, Tuple
except ImportError:
    from typing import List, Any, Iterable, Sequence, Union, Optional, \
        TypeVar, SupportsInt, Generic, Callable, Dict, Tuple
    from typing_extensions import Protocol

T_co = TypeVar('T_co', covariant=True)
TTransform_co = TypeVar('TTransform_co', covariant=True)
TAvalancheDataset = TypeVar('TAvalancheDataset',
                            bound='AvalancheDataset')
XTransform = Optional[Callable[[T_co], Any]]
YTransform = Optional[Callable[[int], int]]

SupportedDataset = Union[IDatasetWithTargets[T_co], ITensorDataset[T_co]]


class AvalancheDataset(DatasetWithTargets[T_co],
                       Generic[T_co]):
    """
    The Dataset used as the base implementation for Avalanche.

    Instances of this dataset are usually returned from scenarios, but it can
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
    kind of data augmentation. The second one is 'test', that will contain
    transformations applied to test patterns. Having both groups can be
    useful when, for instance, in need to test on the training data (as this
    process usually involves removing data augmentation operations). Switching
    between transformations can be easily achieved by using the
    :func:`train` and :func:`eval` method.

    Moreover, arbitrary transformation groups can be added and used. For more
    info see the constructor and the :func:`with_transforms` method.

    This dataset will try to inherit the task labels from the input
    dataset. If none are available and none are given, each pattern will be
    assigned a default task label "0". See the constructor for more details.
    """
    def __init__(self,
                 dataset: SupportedDataset[T_co],
                 *,
                 transform: XTransform = None,
                 target_transform: YTransform = None,
                 transform_groups: Dict[str, Tuple[XTransform,
                                                   YTransform]] = None,
                 initial_transform_group='train',
                 task_labels: Sequence[int] = None):
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
            test transformations. This becomes useful when in need to test on
            the training dataset as test transformations usually don't contain
            random augmentations. ``AvalancheDataset`` natively supports the
            'train' and 'test' groups by calling the ``train()`` and ``eval()``
            methods. When using custom groups one can use the
            ``with_transforms(group_name)`` method instead. Defaults to None,
            which means that the current transforms will be used to
            handle both 'train' and 'test' groups (just like in standard
            ``torchvision`` datasets).
        :param task_labels: The task labels for each pattern. Must be a sequence
            of ints, one for each pattern in the dataset. Defaults to None,
            which means that the dataset will try to obtain the task labels
            from the original dataset. If no task labels could be found, a
            default task label "0" will be applied to all patterns.
        :param initial_transform_group: The name of the transform group
            to be used. Defaults to 'train'.
        """
        super().__init__()

        if transform_groups is not None and (
                transform is not None or target_transform is not None):
            raise ValueError('transform_groups can\'t be used with transform'
                             'and target_transform values')

        if transform_groups is not None:
            AvalancheDataset._check_groups_dict_format(transform_groups)

        self._dataset: SupportedDataset[T_co] = dataset
        """
        The original dataset.
        """

        # Here a conversion may be needed because we can receive
        # a torchvision dataset (in which targets may be a Tensor instead of a
        # sequence if int).
        self.targets: Sequence[int] = self._initialize_targets_sequence(dataset)
        """
        A sequence of ints describing the label of each pattern contained in the
        dataset.
        """

        self.targets_task_labels: Sequence[int] = \
            self._initialize_task_labels_sequence(dataset, task_labels)
        """
        A sequence of ints describing the task label of each pattern contained 
        in the dataset.
        """

        self.tasks_pattern_indices: Dict[int, Sequence[int]] = \
            self._initialize_tasks_dict(dataset, self.targets_task_labels)
        """
        A dictionary mapping task labels to the indices of the patterns with 
        that task label. If you need to obtain the subset of patterns labeled
        with a certain task label, consider using the `task_set` field.
        """

        # Compress targets and task labels to save some memory
        self._optimize_targets()
        self._optimize_task_labels()
        self._optimize_task_dict()

        self.task_set = TaskSubsetDict(self)
        """
        A dictionary that can be used to obtain the subset of patterns given
        a specific task label.
        """

        self.current_transform_group = initial_transform_group
        """
        The name of the transform group currently in use.
        """

        self.transform_groups: Dict[str, Tuple[XTransform, YTransform]] = \
            self._initialize_groups_dict(transform_groups, dataset,
                                         transform, target_transform)
        """
        A dictionary containing the transform groups. Transform groups are
        used to quickly switch between training and test transformations.
        This becomes useful when in need to test on the training dataset as test
        transformations usually don't contain random augmentations.

        AvalancheDataset natively supports switching between the 'train' and
        'test' groups by calling the ``train()`` and ``eval()`` methods. When
        using custom groups one can use the ``with_transforms(group_name)``
        method instead.

        May be null, which means that the current transforms will be used to
        handle both 'train' and 'test' groups.
        """

        if self.current_transform_group not in self.transform_groups:
            raise ValueError('Invalid transformations group ' +
                             str(self.current_transform_group))
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

        self._frozen_transforms: \
            Dict[str, Tuple[XTransform, YTransform]] = dict()
        """
        A dictionary containing frozen transformations.
        """

        for group_name in self.transform_groups.keys():
            self._frozen_transforms[group_name] = (None, None)

        self._set_original_dataset_transform_group(self.current_transform_group)

    def __getitem__(self, idx):
        return manage_advanced_indexing(idx, self._get_single_item, len(self))

    def __len__(self):
        return len(self._dataset)

    def train(self):
        """
        Returns a new dataset with the transformations of the 'train' group
        loaded.

        The current dataset will not be affected.

        :return: A new dataset with the training transformations loaded.
        """
        return self.with_transforms('train')

    def eval(self):
        """
        Returns a new dataset with the transformations of the 'test' group
        loaded.

        Test transformations usually don't contain augmentation procedures.
        This function may be useful when in need to test on training data
        (for instance, in order to run a validation pass).

        The current dataset will not be affected.

        :return: A new dataset with the test transformations loaded.
        """
        return self.with_transforms('test')

    def freeze_transforms(self: TAvalancheDataset) -> \
            TAvalancheDataset:
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
            AvalancheDataset._freeze_dataset_group(dataset_copy,
                                                   group_name)

        return dataset_copy

    def freeze_group_transforms(self: TAvalancheDataset,
                                group_name: str) -> TAvalancheDataset:
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

    def add_transforms(
            self: TAvalancheDataset,
            transform: Callable[[T_co], Any] = None,
            target_transform: Callable[[int], int] = None) -> \
            TAvalancheDataset:
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
                dataset_copy.transform = Compose([
                    dataset_copy.transform, transform])
            else:
                dataset_copy.transform = transform

        if target_transform is not None:
            if dataset_copy.target_transform is not None:
                dataset_copy.target_transform = Compose([
                    dataset_copy.target_transform, target_transform])
            else:
                dataset_copy.target_transform = transform

        return dataset_copy

    def replace_transforms(
            self: TAvalancheDataset,
            transform: XTransform,
            target_transform: YTransform) -> TAvalancheDataset:
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
        :return: A new dataset with the new transformations.
        """

        dataset_copy = self._fork_dataset()
        dataset_copy._replace_original_dataset_group(None, None)

        dataset_copy.transform = transform
        dataset_copy.target_transform = target_transform

        return dataset_copy

    def with_transforms(self: TAvalancheDataset, group_name: str) -> \
            TAvalancheDataset:
        """
        Returns a new dataset with the transformations of a different group
        loaded.

        The current dataset will not be affected.

        :param group_name: The name of the transformations group to use.
        :return: A new dataset with the new transformations.
        """
        dataset_copy = self._fork_dataset()

        if group_name not in dataset_copy.transform_groups:
            raise ValueError('Invalid group name ' + str(group_name))

        dataset_copy.transform_groups[dataset_copy.current_transform_group] = \
            (dataset_copy.transform, dataset_copy.target_transform)
        switch_group = dataset_copy.transform_groups[group_name]
        dataset_copy.transform = switch_group[0]
        dataset_copy.target_transform = switch_group[1]

        dataset_copy.current_transform_group = group_name

        # Finally, align the underlying dataset
        dataset_copy._set_original_dataset_transform_group(group_name)

        return dataset_copy

    def add_transforms_group(self: TAvalancheDataset,
                             group_name: str,
                             transform: XTransform,
                             target_transform: YTransform) -> \
            TAvalancheDataset:
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
            raise ValueError('A group with the same name already exists')

        dataset_copy.transform_groups[group_name] = \
            (transform, target_transform)

        AvalancheDataset._check_groups_dict_format(
            dataset_copy.transform_groups)

        dataset_copy._frozen_transforms[group_name] = (None, None)

        # Finally, align the underlying dataset
        dataset_copy._add_original_dataset_group(group_name)

        return dataset_copy

    def _fork_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = copy.copy(self)
        dataset_copy._frozen_transforms = dict(dataset_copy._frozen_transforms)
        dataset_copy.transform_groups = dict(dataset_copy.transform_groups)

        return dataset_copy

    @staticmethod
    def _freeze_dataset_group(dataset_copy: TAvalancheDataset,
                              group_name: str):
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

            to_be_glued = (dataset_copy.transform,
                           dataset_copy.target_transform)

            # And of course, once frozen, set transformations to None
            dataset_copy.transform = None
            dataset_copy.target_transform = None

        if to_be_glued[0] is not None:
            if frozen_group[0] is None:
                final_frozen_transform = to_be_glued[0]
            else:
                final_frozen_transform = Compose([
                    frozen_group[0], to_be_glued[0]])

        if to_be_glued[1] is not None:
            if frozen_group[1] is None:
                final_frozen_target_transform = to_be_glued[1]
            else:
                final_frozen_target_transform = Compose([
                    frozen_group[1], to_be_glued[1]])

        # Set the frozen transformations
        dataset_copy._frozen_transforms[group_name] = (
            final_frozen_transform, final_frozen_target_transform)

        # Finally, apply the freeze procedure to the original dataset
        dataset_copy._freeze_original_dataset(group_name)

    def _get_single_item(self, idx: int):
        single_element = self._dataset[idx]
        pattern = single_element[0]
        label = single_element[1]

        if _is_tensor_dataset(self._dataset):
            # Manages the fact that TensorDataset may return a single element
            # Tensor instead of an int.
            label = int(label)
        pattern, label = self._apply_transforms(pattern, label)

        return pattern, label, self.targets_task_labels[idx]

    def _apply_transforms(self, pattern: T_co, label: int):
        frozen_group = self._frozen_transforms[self.current_transform_group]
        if frozen_group[0] is not None:
            pattern = frozen_group[0](pattern)

        if self.transform is not None:
            pattern = self.transform(pattern)

        if frozen_group[1] is not None:
            label = frozen_group[1](label)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return pattern, label

    @staticmethod
    def _check_groups_dict_format(groups_dict):
        # The original groups_dict must be convertible to native Python dict
        groups_dict = dict(groups_dict)

        # Check if the format of the groups is correct
        for map_key in groups_dict:
            if not isinstance(map_key, str):
                raise ValueError('Every group must be identified by a string.'
                                 'Wrong key was: "' + str(map_key) + '"')

            map_value = groups_dict[map_key]
            if not isinstance(map_value, tuple):
                raise ValueError('Transformations for group "' + str(map_key) +
                                 '" must be contained in a tuple')

            if not len(map_value) == 2:
                raise ValueError(
                    'Transformations for group "' + str(map_key) + '" must be ' 
                    'a tuple containing 2 elements: a transformation for the X '
                    'values and a transformation for the Y values')

    def _initialize_groups_dict(
            self,
            transform_groups: Optional[Dict[str, Tuple[XTransform,
                                                       YTransform]]],
            dataset: Any,
            transform: XTransform,
            target_transform: YTransform) -> Dict[str, Tuple[XTransform,
                                                             YTransform]]:
        """
        A simple helper method that tries to fill the 'train' and 'test'
        groups as those two groups must always exist.

        If no transform_groups are passed to the class constructor, then
        the transform and target_transform parameters are used for both groups.

        If train transformations are set and test transformations are not, then
        train transformations will be used for the test group.

        :param dataset: The original dataset. Will be used to detect existing
            groups.
        :param transform: The transformation passed as a parameter to the
            class constructor.
        :param target_transform: The target transformation passed as a parameter
            to the class constructor.
        """
        if transform_groups is None:
            transform_groups = {
                'train': (transform, target_transform),
                'test': (transform, target_transform)
            }
        else:
            transform_groups = dict(transform_groups)

        if 'train' in transform_groups:
            if 'test' not in transform_groups:
                transform_groups['test'] = transform_groups['train']

        if 'train' not in transform_groups:
            transform_groups['train'] = (None, None)

        if 'test' not in transform_groups:
            transform_groups['test'] = (None, None)

        self._add_groups_from_original_dataset(dataset, transform_groups)

        return transform_groups

    def _initialize_targets_sequence(self, dataset) -> Sequence[int]:
        return _make_target_from_supported_dataset(dataset)

    def _initialize_task_labels_sequence(
            self, dataset, task_labels: Optional[Sequence[int]]) \
            -> Sequence[int]:
        if task_labels is not None:
            # task_labels has priority over the dataset fields
            if len(task_labels) != len(dataset):
                raise ValueError(
                    'Invalid amount of task labels. It must be equal to the '
                    'number of patterns in the dataset. Got {}, expected '
                    '{}!'.format(len(task_labels), len(dataset)))
            return task_labels

        if hasattr(dataset, 'targets_task_labels'):
            # Dataset is probably a dataset of this class
            # Suppose that it is
            return LazyTargetsConversion(dataset.targets_task_labels)

        # No task labels found. Set all task labels to 0 (in a lazy way).
        return ConstantSequence(0, len(dataset))

    def _initialize_tasks_dict(self, dataset, task_labels: Sequence[int]) \
            -> Dict[int, Sequence[int]]:
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

    def _set_original_dataset_transform_group(
            self, group_name: str) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            self._dataset = self._dataset.with_transforms(group_name)

    def _freeze_original_dataset(
            self, group_name: str) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            self._dataset = self._dataset.freeze_group_transforms(group_name)

    def _replace_original_dataset_group(
            self, transform: XTransform, target_transform: YTransform) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            self._dataset = self._dataset.replace_transforms(
                transform, target_transform)

    def _add_original_dataset_group(
            self, group_name: str) -> None:
        if isinstance(self._dataset, AvalancheDataset):
            self._dataset = self._dataset.add_transforms_group(
                group_name, None, None)

    def _add_groups_from_original_dataset(
            self, dataset, transform_groups) -> None:
        if isinstance(dataset, AvalancheDataset):
            for original_dataset_group in dataset.transform_groups.keys():
                if original_dataset_group not in transform_groups:
                    transform_groups[original_dataset_group] = (None, None)

    def _optimize_targets(self):
        self.targets = optimize_sequence(self.targets)

    def _optimize_task_labels(self):
        self.targets_task_labels = optimize_sequence(self.targets_task_labels)

    def _optimize_task_dict(self):
        for task_label in self.tasks_pattern_indices:
            self.tasks_pattern_indices[task_label] = optimize_sequence(
                self.tasks_pattern_indices[task_label])


class TaskSubsetDict(OrderedDict):

    def __init__(self, avalanche_dataset: AvalancheDataset):
        self._full_dataset = avalanche_dataset
        task_ids = self._full_dataset.tasks_pattern_indices.keys()
        task_ids = sorted(list(task_ids))
        base_dict = OrderedDict()
        for x in task_ids:
            base_dict[x] = x
        super().__init__(base_dict)

    def __getitem__(self, task_id: int):
        if task_id not in self._full_dataset.tasks_pattern_indices:
            raise KeyError('No pattern with ' + str(task_id) + ' found')
        pattern_indices = self._full_dataset.tasks_pattern_indices[task_id]
        return self._make_subset(pattern_indices)

    def or_empty(self, task_id: int):
        try:
            return self[task_id]
        except KeyError:
            return self._make_subset([])

    def _make_subset(self, indices: Sequence[int]):
        return AvalancheSubset(self._full_dataset, indices=indices)


class AvalancheSubset(AvalancheDataset[T_co]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    This Dataset also supports transformations, slicing, advanced indexing,
    the targets field and class mapping.
    """
    def __init__(self,
                 dataset: SupportedDataset[T_co],
                 *,
                 indices: Sequence[int] = None,
                 class_mapping: Sequence[int] = None,
                 transform: Callable[[T_co], Any] = None,
                 target_transform: Callable[[int], int] = None,
                 transform_groups: Dict[str, Tuple[XTransform,
                                                   YTransform]] = None,
                 initial_transform_group='train',
                 task_labels: Sequence[int] = None):
        """
        Creates a ``TransformationSubset`` instance.

        :param dataset: The whole dataset.
        :param indices: Indices in the whole set selected for subset. Can
            be None, which means that the whole dataset will be returned.
        :param class_mapping: A list that, for each possible target (Y) value,
            contains its corresponding remapped value. Can be None.
        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :param transform_groups: A dictionary containing the transform groups.
            Transform groups are used to quickly switch between training and
            test transformations. This becomes useful when in need to test on
            the training dataset as test transformations usually don't contain
            random augmentations. ``AvalancheDataset`` natively supports the
            'train' and 'test' groups by calling the ``train()`` and ``eval()``
            methods. When using custom groups one can use the
            ``with_transforms(group_name)`` method instead. Defaults to None,
            which means that the current transforms will be used to
            handle both 'train' and 'test' groups (just like in standard
            ``torchvision`` datasets).
        :param task_labels: The task labels for each pattern. Must be a sequence
            of ints, one for each pattern in the dataset. This can either be a
            list of task labels for the original dataset or the list of task
            labels for the patterns of the subset (an automatic detection will
            be made) Defaults to None, which means that the dataset will try to
            obtain the task labels from the original dataset. If no task labels
            could be found, a default task label "0" will be applied to all
            patterns.
        :param initial_transform_group: The name of the transform group
            to be used. Defaults to 'train'.
        """

        subset = SubsetWithTargets(dataset, indices=indices,
                                   class_mapping=class_mapping)
        self._original_dataset = dataset
        self._indices = indices

        super().__init__(subset,
                         transform=transform,
                         target_transform=target_transform,
                         transform_groups=transform_groups,
                         initial_transform_group=initial_transform_group,
                         task_labels=task_labels)

    def _initialize_task_labels_sequence(
            self, dataset,
            task_labels: Optional[Sequence[int]]) -> Sequence[int]:

        if task_labels is not None:
            # The task_labels parameter is kind of ambiguous...
            # it may either be the list of task labels of the required subset
            # or it may be the list of task labels of the original dataset.
            # Simple solution: check the length of task_labels!

            if len(task_labels) == len(self._original_dataset):
                # task_labels refers to the original dataset ...
                # or, corner case, len(original) == len(subset), in which
                # case the user just wants to obtain a dataset in which the
                # position of the patterns has been changed according to
                # "indices". This "if" will take care of the corner case, too.
                return LazyClassMapping(task_labels, indices=self._indices)
            elif len(task_labels) == len(dataset):
                # task_labels refers to the subset
                return task_labels
            else:
                raise ValueError(
                    'Invalid amount of task labels. It must be equal to the '
                    'number of patterns in the dataset or of the desired '
                    'subset. Got {}, expected {} or {}!'.format(
                        len(task_labels), len(self._original_dataset),
                        len(dataset)))

        if hasattr(self._original_dataset, 'targets_task_labels'):
            # The original dataset is probably a dataset of this class
            return LazyClassMapping(self._original_dataset.targets_task_labels,
                                    indices=self._indices)

        # No task labels found. Set all task labels to 0 (in a lazy way).
        return ConstantSequence(0, len(dataset))


class AvalancheTensorDataset(AvalancheDataset[T_co]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset from PyTorch,
    this Dataset also supports transformations, slicing, advanced indexing and
    the targets field.
    """
    def __init__(self,
                 dataset_x: Sequence[T_co],
                 dataset_y: Sequence[SupportsInt],
                 *,
                 transform: Callable[[T_co], Any] = None,
                 target_transform: Callable[[int], int] = None,
                 transform_groups: Dict[str, Tuple[XTransform,
                                                   YTransform]] = None,
                 initial_transform_group='train',
                 task_labels: Sequence[int] = None):
        """
        Creates a ``TransformationTensorDataset`` instance.

        :param dataset_x: An sequence, Tensor or ndarray representing the X
            values of the patterns.
        :param dataset_y: An sequence, Tensor int or ndarray of integers
            representing the Y values of the patterns.
        :param transform: A function/transform that takes in a single element
            from the ``dataset_x`` sequence and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :param transform_groups: A dictionary containing the transform groups.
            Transform groups are used to quickly switch between training and
            test transformations. This becomes useful when in need to test on
            the training dataset as test transformations usually don't contain
            random augmentations. ``AvalancheDataset`` natively supports the
            'train' and 'test' groups by calling the ``train()`` and ``eval()``
            methods. When using custom groups one can use the
            ``with_transforms(group_name)`` method instead. Defaults to None,
            which means that the current transforms will be used to
            handle both 'train' and 'test' groups (just like in standard
            ``torchvision`` datasets).
        :param task_labels: The task labels for each pattern. Must be a sequence
            of ints, one for each pattern in the dataset. Defaults to None,
            which means that a default task label "0" will be applied to all
            patterns.
        :param initial_transform_group: The name of the transform group
            to be used. Defaults to 'train'.
        """
        super().__init__(SequenceDataset(dataset_x, dataset_y),
                         transform=transform,
                         target_transform=target_transform,
                         transform_groups=transform_groups,
                         initial_transform_group=initial_transform_group,
                         task_labels=task_labels)


class AvalancheConcatDataset(AvalancheDataset[T_co]):
    """
    A Dataset that behaves like a PyTorch
    :class:`torch.utils.data.ConcatDataset`. However, this Dataset also supports
    transformations, slicing, advanced indexing and the targets field.

    This dataset guarantees that the operations involving the transformations
    and transformations groups are consistent across the concatenated dataset
    (if they are subclasses of :class:`AvalancheDataset`).
    """
    def __init__(self,
                 datasets: Sequence[SupportedDataset[T_co]],
                 *,
                 transform: Callable[[T_co], Any] = None,
                 target_transform: Callable[[int], int] = None,
                 transform_groups: Dict[str, Tuple[XTransform,
                                                   YTransform]] = None,
                 initial_transform_group='train',
                 task_labels: Union[Sequence[int],
                                    Sequence[Sequence[int]]] = None):
        """
        Creates a ``TransformationConcatDataset`` instance.

        :param datasets: An sequence of datasets.
        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :param transform_groups: A dictionary containing the transform groups.
            Transform groups are used to quickly switch between training and
            test transformations. This becomes useful when in need to test on
            the training dataset as test transformations usually don't contain
            random augmentations. ``AvalancheDataset`` natively supports the
            'train' and 'test' groups by calling the ``train()`` and ``eval()``
            methods. When using custom groups one can use the
            ``with_transforms(group_name)`` method instead. Defaults to None,
            which means that the current transforms will be used to
            handle both 'train' and 'test' groups (just like in standard
            ``torchvision`` datasets).
        :param task_labels: The task labels for each pattern. Must be a sequence
            of ints, one for each pattern in the dataset. Alternatively, task
            labels can be expressed as a sequence containing sequences of ints
            (one for each dataset to be concatenated). Defaults to None,
            which means that the dataset will try to obtain the task labels
            from the original datasets. If no task labels could be found for a
            dataset, a default task label "0" will be applied to all patterns
            of that dataset.
        :param initial_transform_group: The name of the transform group
            to be used. Defaults to 'train'.
        """

        self._dataset_list = list(datasets)
        self._datasets_lengths = [len(dataset) for dataset in datasets]
        self._overall_length = sum(self._datasets_lengths)

        if task_labels is not None:
            task_labels = self._concat_task_labels(task_labels)

        self._adapt_concat_datasets()

        super().__init__(DatasetWithTargets(),
                         transform=transform,
                         target_transform=target_transform,
                         transform_groups=transform_groups,
                         initial_transform_group=initial_transform_group,
                         task_labels=task_labels)

    def __len__(self) -> int:
        return self._overall_length

    def _get_single_item(self, idx: int):
        dataset_idx, internal_idx = find_list_from_index(
            idx, self._datasets_lengths, self._overall_length)

        single_element = self._dataset_list[dataset_idx][internal_idx]
        pattern = single_element[0]
        label = single_element[1]
        if _is_tensor_dataset(self._dataset_list[dataset_idx]):
            # Manages the fact that TensorDataset may return a single element
            # Tensor instead of an int.
            label = int(label)
        pattern, label = self._apply_transforms(pattern, label)

        return pattern, label, self.targets_task_labels[idx]

    def _fork_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = super()._fork_dataset()

        dataset_copy._dataset_list = list(dataset_copy._dataset_list)
        # Note: there is no need to duplicate _datasets_lengths

        return dataset_copy

    def _initialize_targets_sequence(self, dataset) -> Sequence[int]:
        targets_list = []
        # Could be easily done with a single line of code
        # This however, allows the user to better check which was the
        # problematic dataset by using a debugger.
        for dataset_idx, single_dataset in enumerate(self._dataset_list):
            targets_list.append(
                _make_target_from_supported_dataset(single_dataset))

        return LazyConcatTargets(targets_list)

    def _initialize_task_labels_sequence(
            self, dataset, task_labels: Optional[Sequence[int]]) \
            -> Sequence[int]:
        if task_labels is not None:
            # task_labels has priority over the dataset fields
            if len(task_labels) != len(dataset):
                raise ValueError(
                    'Invalid amount of task labels. It must be equal to the '
                    'number of patterns in the dataset. Got {}, expected '
                    '{}!'.format(len(task_labels), len(dataset)))
            return task_labels

        concat_t_labels = []
        for single_dataset in self._dataset_list:
            concat_t_labels.append(super()._initialize_task_labels_sequence(
                single_dataset, None
            ))

        return LazyConcatTargets(concat_t_labels)

    def _set_original_dataset_transform_group(
            self, group_name: str) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                self._dataset_list[dataset_idx] = \
                    dataset.with_transforms(group_name)

    def _freeze_original_dataset(
            self, group_name: str) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                self._dataset_list[dataset_idx] = \
                    dataset.freeze_group_transforms(group_name)

    def _replace_original_dataset_group(
            self, transform: XTransform, target_transform: YTransform) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                self._dataset_list[dataset_idx] = \
                    dataset.replace_transforms(transform, target_transform)

    def _add_original_dataset_group(
            self, group_name: str) -> None:
        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                self._dataset_list[dataset_idx] = \
                    dataset.add_transforms_group(group_name, None, None)

    def _add_groups_from_original_dataset(
            self, dataset, transform_groups) -> None:
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

        for dataset in self._dataset_list:
            if isinstance(dataset, AvalancheDataset):
                all_groups.update(dataset.transform_groups.keys())

        for dataset_idx, dataset in enumerate(self._dataset_list):
            if isinstance(dataset, AvalancheDataset):
                for group_name in all_groups:
                    if group_name not in dataset.transform_groups:
                        self._dataset_list[dataset_idx] = \
                            dataset.add_transforms_group(group_name, None, None)

    @staticmethod
    def _concat_task_labels(task_labels: Union[Sequence[int],
                                               Sequence[Sequence[int]]]):
        if isinstance(task_labels[0], int):
            # Flat list of task labels -> just return it.
            # The constructor will check if it has the correct size.
            return task_labels
        else:
            # One list for each dataset, concat them.
            return LazyConcatTargets(task_labels)


def concat_datasets_sequentially(
        train_dataset_list: Sequence[IDatasetWithTargets[T_co]],
        test_dataset_list: Sequence[IDatasetWithTargets[T_co]]) -> \
        Tuple[AvalancheConcatDataset[T_co],
              AvalancheConcatDataset[T_co],
              List[list]]:
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
        len(torch.unique(
            torch.cat((torch.as_tensor(train_dataset_list[dataset_idx].targets),
                      torch.as_tensor(test_dataset_list[dataset_idx].targets)))
            )) for dataset_idx in range(len(train_dataset_list))
    ]

    new_class_ids_per_dataset = []
    for dataset_idx in range(len(train_dataset_list)):
        # The class IDs for this dataset will be in range
        # [n_classes_in_previous_datasets,
        #       n_classes_in_previous_datasets + classes_in_this_dataset)
        class_mapping = list(
            range(next_remapped_idx,
                  next_remapped_idx + classes_per_dataset[dataset_idx]))
        new_class_ids_per_dataset.append(class_mapping)

        train_set = train_dataset_list[dataset_idx]
        test_set = test_dataset_list[dataset_idx]

        # TransformationSubset is used to apply the class IDs transformation.
        # Remember, the class_mapping parameter must be a list in which:
        # new_class_id = class_mapping[original_class_id]
        remapped_train_datasets.append(
            AvalancheSubset(train_set, class_mapping=class_mapping))
        remapped_test_datasets.append(
            AvalancheSubset(test_set, class_mapping=class_mapping))
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return (AvalancheConcatDataset(remapped_train_datasets),
            AvalancheConcatDataset(remapped_test_datasets),
            new_class_ids_per_dataset)


def as_transformation_dataset(dataset: IDatasetWithTargets[T_co]) -> \
        AvalancheDataset[T_co]:
    if isinstance(dataset, AvalancheDataset):
        return dataset

    return AvalancheDataset(dataset)


def train_test_transformation_datasets(
        train_dataset: IDatasetWithTargets[T_co],
        test_dataset: IDatasetWithTargets[T_co],
        train_transformation,
        test_transformation):
    train = AvalancheDataset(
        train_dataset,
        transform_groups=dict(train=(train_transformation, None),
                              test=(test_transformation, None)),
        initial_transform_group='train')

    test = AvalancheDataset(
        test_dataset,
        transform_groups=dict(train=(train_transformation, None),
                              test=(test_transformation, None)),
        initial_transform_group='test')
    return train, test


def _make_target_from_supported_dataset(dataset: SupportedDataset) -> \
        Sequence[int]:
    if hasattr(dataset, 'targets'):
        return LazyTargetsConversion(dataset.targets)

    if hasattr(dataset, 'tensors'):
        if len(dataset.tensors) < 2:
            raise ValueError('Tensor dataset has not enough tensors: '
                             'at least 2 are required.')
        return LazyTargetsConversion(dataset.tensors[1])

    raise ValueError('Unsupported dataset: must have a valid targets field'
                     'or has to be a Tensor Dataset with at least 2'
                     'Tensors')


def _is_tensor_dataset(dataset: SupportedDataset) -> bool:
    return hasattr(dataset, 'tensors') and len(dataset.tensors) >= 2


__all__ = [
    'SupportedDataset',
    'AvalancheDataset',
    'AvalancheSubset',
    'AvalancheTensorDataset',
    'AvalancheConcatDataset',
    'concat_datasets_sequentially',
    'as_transformation_dataset',
    'train_test_transformation_datasets'
]
