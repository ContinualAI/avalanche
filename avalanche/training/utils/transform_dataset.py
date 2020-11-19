################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
import copy

from torchvision.transforms import Compose

from .dataset_utils import IDatasetWithTargets, \
    DatasetWithTargets, manage_advanced_indexing, \
    SequenceDataset, ConcatDatasetWithTargets, SubsetWithTargets, \
    LazyTargetsConversion

try:
    from typing import List, Any, Iterable, Sequence, Union, Optional, \
        TypeVar, Protocol, SupportsInt, Generic, Callable, Dict, Tuple
except ImportError:
    from typing_extensions import List, Any, Iterable, Sequence, Union, \
        Optional, TypeVar, Protocol, SupportsInt, Generic, Callable, Dict, Tuple


T_co = TypeVar('T_co', covariant=True)
TTransform_co = TypeVar('TTransform_co', covariant=True)
TTransformationDataset = TypeVar('TTransformationDataset',
                                 bound='TransformationDataset')
XTransform = Optional[Callable[[T_co], Any]]
YTransform = Optional[Callable[[int], int]]


class TransformationDataset(DatasetWithTargets[T_co],
                            Generic[T_co]):
    """
    A Dataset that applies transformations before returning patterns/targets.
    Also, this Dataset supports slicing and advanced indexing.

    This dataset can also be used to apply several operations involving
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

    However, consider that arbitrary groups can be used. For more info see
    the constructor and the :func:`with_transforms` method.
    """
    def __init__(self,
                 dataset: IDatasetWithTargets[T_co],
                 *,
                 transform: XTransform = None,
                 target_transform: YTransform = None,
                 transform_groups: Dict[str, Tuple[XTransform,
                                                   YTransform]] = None,
                 initial_transform_group='train',
                 chain_transformations: bool = True):
        """
        Creates a ``TransformationDataset`` instance.

        :param dataset: The dataset to decorate. Beware that
            TransformationDataset will not overwrite transformations already
            applied by this dataset.
        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :param transform_groups: A dictionary containing the transform groups.
            Transform groups are used to quickly switch between training and
            test transformations. This becomes useful when in need to test on
            the training dataset as test transformations usually don't contain
            random augmentations. ``TransformDataset`` natively supports the
            'train' and 'test' groups by calling the ``train()`` and ``eval()``
            methods. When using custom groups one can use the
            ``with_transforms(group_name)`` method instead. Defaults to None,
            which means that the current transforms will be used to
            handle both 'train' and 'test' groups (just like in standard
            ``torchvision`` datasets).
        :param initial_transform_group: The name of the transform group
            to be used. Defaults to 'train'.
        :param chain_transformations: If True, will consider transformations
            found in the original dataset. This means that the methods used to
            manipulate transformations will also be applied to the original
            dataset. This only applies is the original dataset is an instance of
            TransformationDataset (or any subclass). Defaults to True.
        """
        super().__init__()

        if transform_groups is not None and (
                transform is not None or target_transform is not None):
            raise ValueError('transform_groups can\'t be used with transform'
                             'and target_transform values')

        if transform_groups is not None:
            TransformationDataset.__check_groups_dict_format(transform_groups)

        self.dataset: IDatasetWithTargets[T_co] = dataset
        """
        The original dataset.
        """

        # Here LazyTargetsConversion is needed because we can receive
        # a torchvision dataset (in which targets may be a Tensor instead of a
        # sequence if int).
        self.targets: Sequence[int] = LazyTargetsConversion(dataset.targets)
        """
        A sequence of ints describing the label of each pattern contained in the
        dataset.
        """

        self.transform_groups: Optional[Dict[str, Tuple[XTransform,
                                                        YTransform]]] = \
            transform_groups
        """
        A dictionary containing the transform groups. Transform groups are
        used to quickly switch between training and test transformations.
        This becomes useful when in need to test on the training dataset as test
        transformations usually don't contain random augmentations.
        
        TransformDataset natively supports switching between the 'train' and
        'test' groups by calling the ``train()`` and ``eval()`` methods. When
        using custom groups one can use the ``with_transforms(group_name)``
        method instead.
        
        May be null, which means that the current transforms will be used to
        handle both 'train' and 'test' groups.
        """

        self.current_transform_group = initial_transform_group
        """
        The name of the transform group currently in use.
        """

        self.__initialize_groups_dict(dataset, transform, target_transform)

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

        self.chain_transformations = chain_transformations
        """
        If True, will consider transformations found in the original dataset.
        This means that the methods used to manipulate transformations will
        also be applied to the original dataset. This only applies is the 
        original dataset is an instance of TransformationDataset (or any
        subclass).
        """

        self._frozen_transforms: \
            Dict[str, Tuple[XTransform, YTransform]] = dict()
        """
        A dictionary containing frozen transformations.
        """

        for group_name in self.transform_groups.keys():
            self._frozen_transforms[group_name] = (None, None)

        if isinstance(self.dataset, TransformationDataset):
            self.dataset = self.dataset.with_transforms(
                self.current_transform_group)

    def __getitem__(self, idx):
        return manage_advanced_indexing(idx, self.__get_single_item,
                                        len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def train(self):
        """
        Returns a new dataset with the transformations of a the 'train' group
        loaded.

        The current dataset will not be affected.

        :return: A new dataset with the training transformations loaded.
        """
        return self.with_transforms('train')

    def eval(self):
        """
        Returns a new dataset with the transformations of a the 'test' group
        loaded.

        Test transformations usually don't contain augmentation procedures.
        This function may be useful when in need to test on training data
        (for instance, in order to run a validation pass).

        The current dataset will not be affected.

        :return: A new dataset with the test transformations loaded.
        """
        return self.with_transforms('test')

    def freeze_transforms(self: TTransformationDataset) -> \
            TTransformationDataset:
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

        dataset_copy = self.__fork_dataset()

        # First, freeze the transformations on the original dataset
        # This can be easily accomplished by breaking the transformations chain
        # Note: this obviously works even when the original dataset is not
        # a TransformationDataset instance.
        dataset_copy.chain_transformations = False

        for group_name in dataset_copy.transform_groups.keys():
            TransformationDataset.__freeze_dataset_group(dataset_copy,
                                                         group_name)

        return dataset_copy

    def freeze_group_transforms(self: TTransformationDataset,
                                group_name: str) -> TTransformationDataset:
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
        dataset_copy = self.__fork_dataset()

        # First, freeze the transformations on the original dataset
        # This can be easily accomplished by breaking the transformations chain
        # Note: this obviously works even when the original dataset is not
        # a TransformationDataset instance.
        dataset_copy.chain_transformations = False

        TransformationDataset.__freeze_dataset_group(
            dataset_copy, group_name)

        return dataset_copy

    def add_transforms(
            self: TTransformationDataset,
            transform: Callable[[T_co], Any] = None,
            target_transform: Callable[[int], int] = None) -> \
            TTransformationDataset:
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

        dataset_copy = self.__fork_dataset()

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
            self: TTransformationDataset,
            transform: XTransform,
            target_transform: YTransform) -> TTransformationDataset:
        """
        Returns a new dataset with the existing transformations replaced with
        the given ones.

        The given transformations will replace the ones of the current
        transformations group. Other transformation groups will not be affected.

        If this dataset was created with ``chain_transformations`` set to True
        and if the original dataset is an instance of
        :class:`TransformationDataset`, then the transformations of the
        original set will be overwritten as well. This operation will create a
        copy of this dataset.

        The current dataset will not be affected.

        Note that this function will not override frozen transformations. This
        will also not affect transformations found in datasets that are not
        instances of :class:`TransformationDataset`.

        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        :return: A new dataset with the new transformations.
        """

        dataset_copy = self.__fork_dataset()
        if dataset_copy.chain_transformations and \
                isinstance(dataset_copy.dataset, TransformationDataset):
            dataset_copy.dataset = \
                dataset_copy.dataset.replace_transforms(None, None)

        dataset_copy.transform = transform
        dataset_copy.target_transform = target_transform

        return dataset_copy

    def with_transforms(self: TTransformationDataset, group_name: str) -> \
            TTransformationDataset:
        """
        Returns a new dataset with the transformations of a different group
        loaded.

        The current dataset will not be affected.

        :param group_name: The name of the transformations group to use.
        :return: A new dataset with the new transformations.
        """
        dataset_copy = self.__fork_dataset()

        if group_name not in dataset_copy.transform_groups:
            raise ValueError('Invalid group name ' + str(group_name))

        dataset_copy.transform_groups[dataset_copy.current_transform_group] = \
            (dataset_copy.transform, dataset_copy.target_transform)
        switch_group = dataset_copy.transform_groups[group_name]
        dataset_copy.transform = switch_group[0]
        dataset_copy.target_transform = switch_group[1]

        dataset_copy.current_transform_group = group_name

        # Finally, align the underlying dataset
        if isinstance(dataset_copy.dataset, TransformationDataset):
            dataset_copy.dataset = dataset_copy.dataset.with_transforms(
                group_name)

        return dataset_copy

    def add_transforms_group(self: TTransformationDataset,
                             group_name: str,
                             transform: XTransform,
                             target_transform: YTransform) -> \
            TTransformationDataset:
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
        dataset_copy = self.__fork_dataset()

        if group_name in dataset_copy.transform_groups:
            raise ValueError('A group with the same name already exists')

        dataset_copy.transform_groups[group_name] = \
            (transform, target_transform)

        TransformationDataset.__check_groups_dict_format(
            dataset_copy.transform_groups)

        dataset_copy._frozen_transforms[group_name] = (None, None)

        # Finally, align the underlying dataset
        if isinstance(dataset_copy.dataset, TransformationDataset):
            dataset_copy.dataset = dataset_copy.dataset.add_transforms_group(
                group_name, None, None)

        return dataset_copy

    def __fork_dataset(self: TTransformationDataset) -> TTransformationDataset:
        dataset_copy = copy.copy(self)
        dataset_copy._frozen_transforms = dict(dataset_copy._frozen_transforms)
        dataset_copy.transform_groups = dict(dataset_copy.transform_groups)

        return dataset_copy

    @staticmethod
    def __freeze_dataset_group(dataset_copy: TTransformationDataset,
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

        # Finally, set the frozen transformations
        dataset_copy._frozen_transforms[group_name] = (
            final_frozen_transform, final_frozen_target_transform)

    def __get_single_item(self, idx: int):
        pattern, label = self.dataset[idx]
        pattern, label = self.__apply_transforms(pattern, label)

        return pattern, label

    def __apply_transforms(self, pattern: T_co, label: int):
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
    def __check_groups_dict_format(groups_dict):
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

    def __initialize_groups_dict(self,
                                 dataset: Any,
                                 transform: XTransform,
                                 target_transform: YTransform):
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
        if self.transform_groups is None:
            self.transform_groups = {
                'train': (transform, target_transform),
                'test': (transform, target_transform)
            }
        else:
            self.transform_groups = dict(self.transform_groups)

        if 'train' in self.transform_groups:
            if 'test' not in self.transform_groups:
                self.transform_groups['test'] = self.transform_groups['train']

        if 'train' not in self.transform_groups:
            self.transform_groups['train'] = (None, None)

        if 'test' not in self.transform_groups:
            self.transform_groups['test'] = (None, None)

        if isinstance(dataset, TransformationDataset):
            for original_dataset_group in dataset.transform_groups.keys():
                if original_dataset_group not in self.transform_groups:
                    self.transform_groups[original_dataset_group] = (None, None)


class TransformationSubset(TransformationDataset[T_co]):
    """
    A Dataset that behaves like a pytorch :class:`torch.utils.data.Subset`.
    This Dataset also supports transformations, slicing, advanced indexing,
    the targets field and class mapping.
    """
    def __init__(self,
                 dataset: IDatasetWithTargets[T_co],
                 *,
                 indices: Sequence[int] = None,
                 class_mapping: Sequence[int] = None,
                 transform: Callable[[T_co], Any] = None,
                 target_transform: Callable[[int], int] = None,
                 transform_groups: Dict[str, Tuple[XTransform,
                                                   YTransform]] = None,
                 initial_transform_group='train',
                 chain_transformations: bool = True):
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
            random augmentations. ``TransformDataset`` natively supports the
            'train' and 'test' groups by calling the ``train()`` and ``eval()``
            methods. When using custom groups one can use the
            ``with_transforms(group_name)`` method instead. Defaults to None,
            which means that the current transforms will be used to
            handle both 'train' and 'test' groups (just like in standard
            ``torchvision`` datasets).
        :param initial_transform_group: The name of the transform group
            to be used. Defaults to 'train'.
        :param chain_transformations: If True, will consider transformations
            found in the original dataset. This means that the methods used to
            manipulate transformations will also be applied to the original
            dataset. This only applies is the original dataset is an instance of
            TransformationDataset (or any subclass). Defaults to True.
        """
        subset = SubsetWithTargets(dataset, indices=indices,
                                   class_mapping=class_mapping)
        super().__init__(subset,
                         transform=transform,
                         target_transform=target_transform,
                         transform_groups=transform_groups,
                         initial_transform_group=initial_transform_group,
                         chain_transformations=chain_transformations)


class TransformationConcatDataset(TransformationDataset[T_co]):
    """
    A Dataset that behaves like a pytorch
    :class:`torch.utils.data.ConcatDataset`. However, this Dataset also supports
    transformations, slicing, advanced indexing and the targets field.
    """
    def __init__(self,
                 datasets: Sequence[IDatasetWithTargets[T_co]],
                 *,
                 transform: Callable[[T_co], Any] = None,
                 target_transform: Callable[[int], int] = None,
                 transform_groups: Dict[str, Tuple[XTransform,
                                                   YTransform]] = None,
                 initial_transform_group='train',
                 chain_transformations: bool = True):
        """
        Creates a ``TransformationConcatDataset`` instance.

        :param datasets: An sequence of datasets.
        :param transform: A function/transform that takes the X value of a
            pattern from the original dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        """
        concat_dataset = ConcatDatasetWithTargets(datasets)

        super().__init__(concat_dataset, transform=transform,
                         target_transform=target_transform,
                         transform_groups=transform_groups,
                         initial_transform_group=initial_transform_group,
                         chain_transformations=chain_transformations)


class TransformationTensorDataset(TransformationDataset[T_co]):
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
                 chain_transformations: bool = True):
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
            random augmentations. ``TransformDataset`` natively supports the
            'train' and 'test' groups by calling the ``train()`` and ``eval()``
            methods. When using custom groups one can use the
            ``with_transforms(group_name)`` method instead. Defaults to None,
            which means that the current transforms will be used to
            handle both 'train' and 'test' groups (just like in standard
            ``torchvision`` datasets).
        :param initial_transform_group: The name of the transform group
            to be used. Defaults to 'train'.
        :param chain_transformations: If True, will consider transformations
            found in the original dataset. This means that the methods used to
            manipulate transformations will also be applied to the original
            dataset. This only applies is the original dataset is an instance of
            TransformationDataset (or any subclass). Defaults to True.
        """
        super().__init__(SequenceDataset(dataset_x, dataset_y),
                         transform=transform,
                         target_transform=target_transform,
                         transform_groups=transform_groups,
                         initial_transform_group=initial_transform_group,
                         chain_transformations=chain_transformations)


__all__ = ['TransformationDataset', 'TransformationSubset',
           'TransformationTensorDataset']
