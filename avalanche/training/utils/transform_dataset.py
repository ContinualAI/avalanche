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

from .dataset_utils import IDatasetWithTargets, \
    DatasetWithTargets, manage_advanced_indexing, \
    SequenceDataset, ConcatDatasetWithTargets, SubsetWithTargets, \
    LazyTargetsConversion

try:
    from typing import List, Any, Iterable, Sequence, Union, Optional, \
        TypeVar, Protocol, SupportsInt, Generic, Callable
except ImportError:
    from typing import List, Any, Iterable, Sequence, Union, Optional, \
        TypeVar, SupportsInt, Generic, Callable
    from typing_extensions import Protocol


T_co = TypeVar('T_co', covariant=True)
TTransform_co = TypeVar('TTransform_co', covariant=True)


class TransformationDataset(DatasetWithTargets[T_co],
                            Generic[T_co]):
    """
    A Dataset that applies transformations before returning patterns/targets.
    Also, this Dataset supports slicing and advanced indexing.
    """
    def __init__(self, dataset: IDatasetWithTargets[T_co],
                 transform: Callable[[T_co], Any] = None,
                 target_transform: Callable[[int], int] = None):
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

        # Here LazyTargetsConversion is needed because we can receive
        # a torchvision dataset (in which targets may be a Tensor instead of a
        # sequence if int).
        self.targets: Sequence[int] = LazyTargetsConversion(dataset.targets)
        """
        A sequence of ints describing the label of each pattern contained in the
        dataset.
        """

    def __getitem__(self, idx):
        return manage_advanced_indexing(idx, self.__get_single_item,
                                        len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __get_single_item(self, idx: int):
        pattern, label = self.dataset[idx]
        if self.transform is not None:
            pattern = self.transform(pattern)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return pattern, label


class TransformationSubset(TransformationDataset[T_co]):
    """
    A Dataset that behaves like a pytorch :class:`torch.utils.data.Subset`.
    This Dataset also supports transformations, slicing, advanced indexing,
    the targets field and class mapping.
    """
    def __init__(self, dataset: IDatasetWithTargets[T_co],
                 indices: Sequence[int] = None,
                 class_mapping: Sequence[int] = None,
                 transform=None, target_transform=None):
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
                         target_transform=target_transform)


class TransformationConcatDataset(TransformationDataset[T_co]):
    """
    A Dataset that behaves like a pytorch
    :class:`torch.utils.data.ConcatDataset`. However, this Dataset also supports
    transformations, slicing, advanced indexing and the targets field.
    """
    def __init__(self, datasets: Sequence[IDatasetWithTargets[T_co]],
                 transform=None, target_transform=None):
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
                         target_transform=target_transform)


class TransformationTensorDataset(TransformationDataset[T_co]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset from PyTorch,
    this Dataset also supports transformations, slicing, advanced indexing and
    the targets field.
    """
    def __init__(self, dataset_x: Sequence[T_co],
                 dataset_y: Sequence[SupportsInt],
                 transform=None, target_transform=None):
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
                         target_transform=target_transform)


__all__ = ['TransformationDataset', 'TransformationSubset',
           'TransformationTensorDataset']
