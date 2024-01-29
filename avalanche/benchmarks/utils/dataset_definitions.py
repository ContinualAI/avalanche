################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-07-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import TypeVar, SupportsInt, Sequence, Protocol

from torch.utils.data.dataset import Dataset

T_co = TypeVar("T_co", covariant=True)
TTargetType = TypeVar("TTargetType")
TTargetType_co = TypeVar("TTargetType_co", covariant=True)


# General rule: consume ISupportedClassificationDataset,
# produce ClassificationDataset (applies to non-classification, too).
#
# That is, accept ISupportedClassificationDataset as parameter to
# functions/constructors (when possible), but always expose/return instances of
# ClassificationDataset to the, user (no matter what). The main difference is
# that ClassificationDataset is a subclass of the PyTorch Dataset while
# ISupportedClassificationDataset is just a Protocol. This will allow the user
# to pass any custom dataset while receiving Dataset subclasses as outputs at
# the same time. This will allow popular IDEs (like PyCharm) to properly execute
# type checks and warn the user if something is wrong.


class IDataset(Protocol[T_co]):
    """
    Protocol definition of a Dataset.

    Note: no __add__ method is defined.
    """

    def __getitem__(self, index: int) -> T_co: ...

    def __len__(self) -> int: ...


class IDatasetWithTargets(IDataset[T_co], Protocol[T_co, TTargetType_co]):
    """
    Protocol definition of a Dataset that has a valid targets field.
    """

    @property
    def targets(self) -> Sequence[TTargetType_co]:
        """
        A sequence of elements describing the targets of each pattern.
        """
        ...


class ISupportedClassificationDataset(IDatasetWithTargets[T_co, SupportsInt], Protocol):
    """
    Protocol definition of a Dataset that has a valid targets field (like the
    Datasets in the torchvision package) for classification.

    For classification purposes, the targets field must be a sequence of ints.
    describing the class label of each pattern.

    This class however describes a targets field as a sequence of elements
    that can be converted to `int`. The main reason for this choice is that
    the targets field of some torchvision datasets is a Tensor. This means that
    this protocol class supports both sequence of native ints and Tensor of ints
    (or longs).

    On the contrary, class :class:`IClassificationDataset` strictly
    defines a `targets` field as sequence of native `int`s.
    """

    @property
    def targets(self) -> Sequence[SupportsInt]:
        """
        A sequence of ints or a PyTorch Tensor or a NumPy ndarray describing the
        label of each pattern contained in the dataset.
        """
        ...


class ITensorDataset(IDataset[T_co], Protocol):
    """
    Protocol definition of a Dataset that has a tensors field (like
    TensorDataset).

    A TensorDataset can be easily converted to a :class:`IDatasetWithTargets`
    by using one of the provided tensors (usually the second, which commonly
    contains the "y" values).
    """

    @property
    def tensors(self) -> Sequence[T_co]:
        """
        A sequence of PyTorch Tensors describing the contents of the Dataset.
        """
        ...


class IClassificationDataset(IDatasetWithTargets[T_co, int], Protocol):
    """
    Protocol definition of a Dataset that has a valid targets field (like the
    Datasets in the torchvision package) where the targets field is a sequence
    of native ints.

    The content of the sequence must be strictly native ints. For a more slack
    protocol see :class:`ISupportedClassificationDataset`.
    """

    targets: Sequence[int]
    """
    A sequence of ints describing the label of each pattern contained in the
    dataset.
    """


class ClassificationDataset(IClassificationDataset[T_co], Dataset):
    """
    Dataset that has a valid targets field (like the Datasets in the
    torchvision package) where the targets field is a sequence of native ints.

    The actual value of the targets field should be set by the child class.
    """

    def __init__(self):
        self.targets = []
        """
        A sequence of ints describing the label of each pattern contained in the
        dataset.
        """


__all__ = [
    "IDataset",
    "IDatasetWithTargets",
    "ISupportedClassificationDataset",
    "ITensorDataset",
    "IClassificationDataset",
    "ClassificationDataset",
]
