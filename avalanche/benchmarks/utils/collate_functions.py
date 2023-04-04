################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 21-04-2022                                                             #
# Author(s): Antonio Carta, Lorenzo Pellegrini                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    List,
    TypeVar,
    Generic,
    Sequence,
    Tuple,
    Dict,
    Union,
    overload,
)
from typing_extensions import TypeAlias

import torch
from torch import Tensor
from torch.utils.data import default_collate

BatchT = TypeVar("BatchT")
ExampleT = TypeVar("ExampleT")
BatchedFeatureT = TypeVar("BatchedFeatureT")
FeatureT = TypeVar("FeatureT")


ClassificationExampleT = Tuple[Tensor, ...]
ClassificationBatchT = Tuple[Tensor, ...]
ClassificationBatchedFeatureT: TypeAlias = Tensor
ClassificationFeatureT: TypeAlias = Tensor


DetectionExampleT = Tuple[Tensor, Dict, int]
DetectionBatchT = Tuple[Tuple[Tensor, ...], Tuple[Dict, ...], Tuple[int, ...]]
DetectionBatchedFeatureT = Union[Tuple[Tensor, ...],
                                 Tuple[Dict, ...],
                                 Tuple[int, ...]]
DetectionFeatureT = Union[Tensor, Dict, int]


def classification_collate_mbatches_fn(mbatches):
    """Combines multiple mini-batches together.

    Concatenates each tensor in the mini-batches along dimension 0 (usually
    this is the batch size).

    :param mbatches: sequence of mini-batches.
    :return: a single mini-batch
    """
    batch = []
    for i in range(len(mbatches[0])):
        t = classification_single_values_collate_mbatches_fn(
            [el[i] for el in mbatches], i
        )
        batch.append(t)
    return batch


def classification_single_values_collate_mbatches_fn(values_list, index):
    """
    Collate function used to merge the single elements (x or y or t,
    etcetera) of multiple minibatches of data from a classification dataset.

    Beware that this function expects a list of already batched values,
    which means that it accepts a list of [mb_size, X, Y, Z, ...] tensors.
    This is different from :func:`classification_single_values_collate_fn`,
    which expects a flat list of tensors [X, Y, Z, ...] to be collated.

    This function assumes that all values are tensors of the same shape
    (excluding the first dimension).

    :param values_list: The list of values to merge.
    :param index: The index of the element. 0 for x values, 1 for y values,
        etcetera. In this implementation, this parameter is ignored.
    :return: The merged values.
    """
    return torch.cat(values_list, dim=0)


def classification_single_values_collate_fn(values_list, index):
    """
    Collate function used to merge the single elements (x or y or t,
    etcetera) of a minibatch of data from a classification dataset.

    This function expects a flat list of tensors [X, Y, Z, ...] to be collated.
    For a version of the functions that can collate pre-batched values
    [mb_size, X, Y, Z, ...], refer to
    :func:`classification_single_values_collate_mbatches_fn`.

    This function assumes that all values are tensors of the same shape.

    :param values_list: The list of values to merge.
    :param index: The index of the element. 0 for x values, 1 for y values,
        etcetera. In this implementation, this parameter is ignored.
    :return: The merged values.
    """
    return torch.stack(values_list)


def detection_collate_fn(batch: Sequence[DetectionExampleT]) -> DetectionBatchT:
    """
    Collate function used when loading detection datasets using a DataLoader.

    This will merge the single samples of a batch to create a minibatch.
    This collate function follows the torchvision format for detection tasks.
    """

    # Equivalent to:
    # a = tuple([x[0] for x in batch])
    # b = tuple([x[1] for x in batch])
    # c = tuple([x[2] for x in batch])
    # return a, b, c

    return tuple(zip(*batch))  # type: ignore


def detection_collate_mbatches_fn(mbatches: Sequence[DetectionBatchT]) -> \
        DetectionBatchT:
    """
    Collate function used when loading detection datasets using a DataLoader.

    This will merge multiple batches to create a concatenated batch.

    Beware that merging multiple batches is different from creating a batch
    from single dataset elements: Batches can be created from a
    list of single dataset elements by using :func:`detection_collate_fn`.
    """

    # The code used here is equivalent to the following one:
    # images: List[Tuple[Tensor, ...]] = []
    # targets: List[Tuple[Dict, ...]] = []
    # task_labels: List[Tuple[int, ...]] = []

    # for mb in mbatches:
    #     images.append(mb[0])
    #     targets.append(mb[1])
    #     task_labels.append(mb[2])

    # batched_images = tuple(itertools.chain.from_iterable(images))
    # batched_targets = tuple(itertools.chain.from_iterable(targets))
    # batched_task_labels = tuple(itertools.chain.from_iterable(task_labels))

    # return batched_images, batched_targets, batched_task_labels

    lists_dict: Dict[int, List] = defaultdict(list)
    for mb in mbatches:
        for mb_elem_idx, mb_elem in enumerate(mb):
            lists_dict[mb_elem_idx].append(mb_elem)

    batch_elements = []
    for mb_elem_idx in range(max(lists_dict.keys()) + 1):
        batch_elements.append(
            tuple(itertools.chain.from_iterable(lists_dict[mb_elem_idx]))
        )

    return tuple(batch_elements)  # type: ignore


class Collate(ABC, Generic[ExampleT, BatchT, BatchedFeatureT, FeatureT]):

    @abstractmethod
    def collate_fn(self, batch: Sequence[ExampleT]) -> BatchT:
        """

        Merge multiple examples to create a batch.

        This function expects a list of elements as obtained from
        the dataset.

        PyTorch official documentation described the default_collate_fn as:
        "Function that takes in a batch of data and puts the elements within
        the batch into a tensor with an additional
        outer dimension - batch size."

        :param batch: The list of examples.
        :return: The batch.
        """
        pass

    @abstractmethod
    def collate_single_value_fn(
            self,
            feature_batch: Sequence[FeatureT],
            feature_idx: int) -> BatchedFeatureT:
        """
        Merge a specific feature to create a single-feature batch.

        This function expects a list of features.

        :param feature_batch: The list of features to be batched.
        :param feature_idx: The index of the feature being batched.
            This may be useful to customize how features are merged.

        :return: The batched features.
        """
        pass

    @abstractmethod
    def collate_batches_fn(self, batches: Sequence[BatchT]) -> BatchT:
        """
        Merge multiple batches.

        This function expects a list of pre-collated batches
        (as collated through :meth:`collate_fn`.)

        :param batches: A list of batches to be merged together.
        :return: A batch made by collating the input batches.
        """
        pass

    @abstractmethod
    def collate_single_value_batches_fn(
            self,
            feature_batches: Sequence[BatchedFeatureT],
            feature_idx: int) -> BatchedFeatureT:
        """
        Merge a specific feature of examples contained in multiple batches.

        This function expects a list of pre-batched features.

        :param feature_batches: A list of batched features to be merged
            together.
        :param feature_idx: The index of the feature being batched.
            This may be useful to customize how features are merged.
        :return: A batch of features made by collating the input batched
            features.
        """
        pass

    def __call__(self, batch: Sequence[ExampleT]) -> BatchT:
        """
        Merges multiple examples to create a batch.

        In practice, this will call :meth:`collate_fn`.
        """
        return self.collate_fn(batch)


class ClassificationCollate(
    Collate[ClassificationExampleT,
            ClassificationBatchT,
            ClassificationBatchedFeatureT,
            ClassificationFeatureT]):

    def collate_fn(
            self,
            batch: Sequence[ClassificationExampleT]) -> ClassificationBatchT:
        return default_collate(list(batch))

    def collate_single_value_fn(
            self,
            feature_batch: Sequence[Tensor],
            feature_idx: int) -> Tensor:
        return torch.stack(list(feature_batch))

    def collate_batches_fn(
            self, 
            batches: Sequence[ClassificationBatchT]) -> ClassificationBatchT:
        batch: List[Tensor] = []
        for i in range(len(batches[0])):
            t = self.collate_single_value_batches_fn(
                [el[i] for el in batches], i
            )
            batch.append(t)
        return tuple(batch)

    def collate_single_value_batches_fn(
            self,
            feature_batch: Sequence[Tensor],
            feature_idx: int) -> Tensor:
        return torch.cat(list(feature_batch), dim=0)


class DetectionCollate(
    Collate[DetectionExampleT, 
            DetectionBatchT,
            DetectionBatchedFeatureT,
            DetectionFeatureT]):

    def collate_fn(self, batch: Sequence[DetectionExampleT]) -> \
            DetectionBatchT:
        return detection_collate_fn(batch)

    def collate_single_value_fn(
            self,
            feature_batch,
            feature_idx):
        return tuple(feature_batch)

    def collate_batches_fn(
            self,
            batches: Sequence[DetectionBatchT]) -> DetectionBatchT:
        return detection_collate_mbatches_fn(batches)
    
    @overload
    def collate_single_value_batches_fn(
            self,
            feature_batch: Sequence[Tuple[Tensor, ...]],
            feature_idx: int) -> Tuple[Tensor, ...]:
        ...

    @overload
    def collate_single_value_batches_fn(
            self,
            feature_batch: Sequence[Tuple[Dict, ...]],
            feature_idx: int) -> Tuple[Dict, ...]:
        ...

    @overload
    def collate_single_value_batches_fn(
            self,
            feature_batch: Sequence[Tuple[int, ...]],
            feature_idx: int) -> Tuple[int, ...]:
        ...

    def collate_single_value_batches_fn(
            self,
            feature_batch: Sequence[Tuple],
            feature_idx: int) -> Tuple:

        flattened_features: List[DetectionBatchedFeatureT] = []
        for batch in feature_batch:
            flattened_features.extend(batch)
        return tuple(flattened_features)


__all__ = [
    "classification_collate_mbatches_fn",
    "classification_single_values_collate_mbatches_fn",
    "detection_collate_fn",
    "detection_collate_mbatches_fn",
    "Collate",
    "ClassificationCollate",
    "DetectionCollate"
]
