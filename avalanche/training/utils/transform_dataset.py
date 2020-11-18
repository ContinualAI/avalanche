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
        TypeVar, Protocol, SupportsInt, Generic, Callable
except ImportError:
    from typing_extensions import List, Any, Iterable, Sequence, Union, \
        Optional, TypeVar, Protocol, SupportsInt, Generic, Callable


T_co = TypeVar('T_co', covariant=True)
TTransform_co = TypeVar('TTransform_co', covariant=True)
TTransformationDataset = TypeVar('TTransformationDataset',
                                 bound='TransformationDataset')


class TransformationDataset(DatasetWithTargets[T_co],
                            Generic[T_co]):
    """
    A Dataset that applies transformations before returning patterns/targets.
    Also, this Dataset supports slicing and advanced indexing.
    """
    def __init__(self,
                 dataset: IDatasetWithTargets[T_co],
                 *,
                 transform: Callable[[T_co], Any] = None,
                 target_transform: Callable[[int], int] = None,
                 chain_transformations=True):
        """
        Creates a ``TransformationDataset`` instance.

        :param dataset: The dataset to decorate. Beware that
            TransformationDataset will not overwrite transformations already
            applied by this dataset.
        :param transform: A function/transform that takes in an PIL image and
            returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        """
        super().__init__()

        self.transform = transform
        """
        A function/transform that takes in an PIL image and returns a 
        transformed version.
        """

        self.target_transform = target_transform
        """
        A function/transform that takes in the target and transforms it.
        """

        self.dataset: IDatasetWithTargets[T_co] = dataset
        """
        The original dataset.
        """

        self.chain_transformations = chain_transformations
        """
        If True, will consider transformations found in the original dataset.
        This means that the methods used to manipulate transformations will
        also be applied to the original dataset. This only applies is the 
        original dataset is an instance of TransformationDataset (or any
        subclass).
        """

        # Here LazyTargetsConversion is needed because we can receive
        # a torchvision dataset (in which targets may be a Tensor instead of a
        # sequence if int).
        self.targets: Sequence[int] = LazyTargetsConversion(dataset.targets)
        """
        A sequence of ints describing the label of each pattern contained in the
        dataset.
        """

        self._frozen_transforms = []
        """
        A list containing frozen transformations.
        """

        self._frozen_target_transforms = []
        """
        A list containing frozen target transformations.
        """

    def __getitem__(self, idx):
        return manage_advanced_indexing(idx, self.__get_single_item,
                                        len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def freeze_transforms(self: TTransformationDataset) -> \
            TTransformationDataset:
        """
        Returns a new dataset where the current transformations are frozen.

        Frozen transformations will be permanently glued to the original
        dataset so that they can't be changed anymore. This is usually done
        when using transformations to create derived datasets: in this way
        freezing the transformations will ensure that the user won't be able
        to inadvertently change them.

        The current dataset will not be affected.

        :return: A new dataset with the current transformations frozen.
        """

        dataset_copy = self.__fork_dataset()

        # First, freeze the transformations on the original dataset
        # This can be easily accomplished by breaking the transformations chain
        # Note: this obviously works even when the original dataset is not
        # a TransformationDataset instance.
        dataset_copy.chain_transformations = False

        # Then freeze the current transformations. Frozen transformations
        # are saved in a separate list.
        if dataset_copy.transform is not None:
            dataset_copy._frozen_transforms.append(dataset_copy.transform)
            dataset_copy.transform = None

        if dataset_copy.target_transform is not None:
            dataset_copy._frozen_target_transforms.append(
                dataset_copy.target_transform)
            dataset_copy.target_transform = None

        return dataset_copy

    def add_transforms(
            self: TTransformationDataset,
            transform: Callable[[T_co], Any] = None,
            target_transform: Callable[[int], int] = None) -> \
            TTransformationDataset:
        """
        Returns a new dataset with the given transformations added to
        the existing ones.

        The given transformations will be added at the end of previous
        transformations. The current dataset will not be affected.

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

    def __fork_dataset(self):
        dataset_copy = copy.copy(self)
        dataset_copy._frozen_transforms = list(dataset_copy._frozen_transforms)
        dataset_copy._frozen_target_transforms = list(
            dataset_copy._frozen_target_transforms)

        return dataset_copy

    def __get_single_item(self, idx: int):
        pattern, label = self.dataset[idx]

        pattern = self.__apply_transforms(pattern)
        label = self.__apply_target_transforms(label)

        return pattern, label

    def __apply_transforms(self, pattern: T_co) -> T_co:
        for frozen_transform in self._frozen_transforms:
            pattern = frozen_transform(pattern)

        if self.transform is not None:
            pattern = self.transform(pattern)

        return pattern

    def __apply_target_transforms(self, label: int) -> int:
        for frozen_target_transform in self._frozen_target_transforms:
            label = frozen_target_transform(label)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return label


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
                 chain_transformations: bool = True):
        """
        Creates a ``TransformationSubset`` instance.

        :param dataset: The whole dataset.
        :param indices: Indices in the whole set selected for subset. Can
            be None, which means that the whole dataset will be returned.
        :param class_mapping: A list that, for each possible target (Y) value,
            contains its corresponding remapped value. Can be None.
        :param transform: A function/transform that takes a pattern obtained
            from the dataset and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        """
        subset = SubsetWithTargets(dataset, indices=indices,
                                   class_mapping=class_mapping)
        super().__init__(subset, transform=transform,
                         target_transform=target_transform,
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
                 chain_transformations: bool = True):
        """
        Creates a ``TransformationConcatDataset`` instance.

        :param datasets: An sequence of datasets.
        :param transform: A function/transform that takes in a single element
            from the ``dataset_x`` sequence and returns a transformed version.
        :param target_transform: A function/transform that takes in the target
            and transforms it.
        """
        concat_dataset = ConcatDatasetWithTargets(datasets)

        super().__init__(concat_dataset, transform=transform,
                         target_transform=target_transform,
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
        """
        super().__init__(SequenceDataset(dataset_x, dataset_y),
                         transform=transform,
                         target_transform=target_transform,
                         chain_transformations=chain_transformations)


__all__ = ['TransformationDataset', 'TransformationSubset',
           'TransformationTensorDataset']
