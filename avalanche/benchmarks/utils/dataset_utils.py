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
from abc import ABC, abstractmethod
import bisect
import copy
from typing import Iterator, overload, final
import numpy as np
from numpy import ndarray
from torch import Tensor

from torch.utils.data import Subset

from .dataset_definitions import (
    IDatasetWithTargets,
)


from typing import (
    Sequence,
    List,
    Iterable,
    Union,
    Optional,
    TypeVar,
    Callable,
    Generic,
)


T_co = TypeVar("T_co", covariant=True)
TTargetType = TypeVar("TTargetType")
TMappableTargetType = TypeVar("TMappableTargetType")


TData = TypeVar("TData")
TIntermediateData = TypeVar("TIntermediateData")
TSliceSequence = TypeVar("TSliceSequence", bound="SliceSequence")


class SliceSequence(Sequence[TData], Generic[TData, TIntermediateData], ABC):
    def __init__(self, slice_ids: Optional[List[int]]):
        self.slice_ids: Optional[List[int]] = (
            list(slice_ids) if slice_ids is not None else None
        )
        """
        Describes thew indices in the current slice
        (w.r.t. the original sequence). 
        Can be None, which means that this object is the original stream.
        """
        super().__init__()

    def __iter__(self) -> Iterator[TData]:
        # Iter built on __getitem__ + __len__
        for i in range(len(self)):
            el = self[i]
            yield el

    @overload
    def __getitem__(self, item: int) -> TData: ...

    @overload
    def __getitem__(self: TSliceSequence, item: slice) -> TSliceSequence: ...

    @final
    def __getitem__(
        self: TSliceSequence, item: Union[int, slice]
    ) -> Union[TSliceSequence, TData]:
        if isinstance(item, (int, np.integer)):
            item = int(item)
            if item >= len(self):
                raise IndexError("Sequence index out of bounds" + str(int(item)))

            curr_elem = item if self.slice_ids is None else self.slice_ids[item]

            el = self._make_element(curr_elem)
            el = self._post_process_element(el)
            return el
        else:
            new_slice = self._forward_slice(self.slice_ids, item)
            return self._make_slice(new_slice)

    def __len__(self) -> int:
        """
        Gets the size of this sequence.

        :return: The number of elements in this sequence.
        """
        if self.slice_ids is not None:
            return len(self.slice_ids)
        else:
            return self._full_length()

    def _forward_slice(
        self, *slices: Union[None, slice, Iterable[int]]
    ) -> Optional[Iterable[int]]:
        any_slice = False
        indices = list(range(self._full_length()))
        for sl in slices:
            if sl is None:
                continue
            any_slice = True

            slice_indices = slice_alike_object_to_indices(
                slice_alike_object=sl, max_length=len(indices)
            )

            new_indices = [indices[x] for x in slice_indices]
            indices = new_indices

        if any_slice:
            return indices
        else:
            return None  # No slicing

    @abstractmethod
    def _full_length(self) -> int:
        """
        Gets the number of elements in the originating sequence
        (that is, the non-sliced sequence).
        """
        pass

    @abstractmethod
    def _make_element(self, element_idx: int) -> TIntermediateData:
        """
        Obtain the element at the given position in the originating sequence
        (that is, the non-sliced sequence).

        This element is then passed to `_post_process_element` before
        returning it.
        """
        pass

    def _post_process_element(self, element: TIntermediateData) -> TData:
        """
        Post process the element obtained by _make_element.

        By default, returns the element as is.

        Subclasses may override this to provide post-processing.
        """
        return element  # type: ignore

    def _make_slice(
        self: TSliceSequence, sequence_slice: Optional[Iterable[int]]
    ) -> TSliceSequence:
        """
        Obtain a sub-squence given a list of indices of the elements
        to include.

        Element ids are the ones of the originating sequence
        (that is, the non-sliced sequence).
        """
        stream_copy = copy.copy(self)
        stream_copy.slice_ids = (
            list(sequence_slice) if sequence_slice is not None else None
        )
        return stream_copy

    def __str__(self):
        return "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"


class SubSequence(
    SliceSequence[TTargetType, TMappableTargetType],
    Generic[TTargetType, TMappableTargetType],
):
    """
    A utility class used to define a lazily evaluated sub-sequence.
    """

    def __init__(
        self,
        targets: Sequence[TMappableTargetType],
        *,
        indices: Optional[Sequence[int]] = None,
        converter: Optional[Callable[[TMappableTargetType], TTargetType]] = None,
    ):
        self._targets = targets
        self.converter = converter
        super().__init__(slice_ids=list(indices) if indices is not None else None)

    def _full_length(self) -> int:
        return len(self._targets)

    def _make_element(self, element_idx: int) -> TMappableTargetType:
        return self._targets[element_idx]

    def _post_process_element(self, element: TMappableTargetType) -> TTargetType:
        if self.converter is None:
            return element  # type: ignore
        return self.converter(element)


class SubsetWithTargets(Generic[T_co, TTargetType], Subset[T_co]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    However, this dataset also supports the targets field.
    """

    def __init__(
        self,
        dataset: IDatasetWithTargets[T_co, TTargetType],
        indices: Union[Sequence[int], None],
    ):
        if indices is None:
            indices = range(len(dataset))
        super().__init__(dataset, indices)
        self.targets: Sequence[TTargetType] = SubSequence(
            dataset.targets, indices=indices
        )

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


class SequenceDataset(IDatasetWithTargets[T_co, TTargetType]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset.
    """

    def __init__(
        self, *sequences: Sequence, targets: Union[int, Sequence[TTargetType]] = 1
    ):
        """
        Creates a ``SequenceDataset`` instance.

        Beware that the second sequence, will be used to fill the targets
        field without running any kind of type conversion.

        :param sequences: A sequence of sequences, Tensors or ndarrays
            representing the patterns.
        :param targets: A sequence representing the targets field of the
            dataset. Can either be 1) a sequence of values containing as many
            elements as the number of contained patterns, or 2) the index
            of the sequence to use as the targets field. Defaults to 1, which
            means that the second sequence (usually containing the "y" values)
            will be used for the targets field.
        """
        if len(sequences) < 1:
            raise ValueError("At least one sequence must be passed")

        common_size = len(sequences[0])
        for seq in sequences:
            if len(seq) != common_size:
                raise ValueError(
                    "Sequences must contain the same " "amount of elements"
                )

        self._sequences = sequences
        if isinstance(targets, int):
            targets = sequences[targets]

        self.targets: Sequence[TTargetType] = targets

    def __getitem__(self, idx):
        return tuple(seq[idx] for seq in self._sequences)

    def __len__(self) -> int:
        return len(self._sequences[0])


def find_list_from_index(
    pattern_idx: int,
    list_sizes: Sequence[int],
    max_size: int,
    cumulative_sizes=None,
):
    if pattern_idx >= max_size:
        raise IndexError()

    if cumulative_sizes is None:
        r, s = [], 0
        for list_len in list_sizes:
            r.append(list_len + s)
            s += list_len
        cumulative_sizes = r

    list_idx = bisect.bisect_right(cumulative_sizes, pattern_idx)
    if list_idx != 0:
        pattern_idx = pattern_idx - cumulative_sizes[list_idx - 1]

    if pattern_idx >= list_sizes[list_idx]:
        raise ValueError("Index out of bounds, wrong max_size parameter")
    return list_idx, pattern_idx


T = TypeVar("T")
X = TypeVar("X")


def manage_advanced_indexing(
    idx: Union[slice, int, Iterable[int]],
    single_element_getter: Callable[[int], X],
    max_length: int,
    collate_fn: Callable[[Iterable[X]], T],
) -> Union[X, T]:
    """
    Utility function used to manage the advanced indexing and slicing.

    If more than a pattern is selected, the X and Y values will be merged
    in two separate torch Tensor objects using the stack operation.

    :param idx: Either an in, a slice object or a list (including ndarrays and
        torch Tensors) of indexes.
    :param single_element_getter: A callable used to obtain a single element
        given its int index.
    :param max_length: The maximum sequence length.
    :param collate_fn: The function to use to create a batch of data from
        single elements.
    :return: A tuple consisting of two tensors containing the X and Y values
        of the patterns addressed by the idx parameter.
    """
    indexes_iterator: Iterable[int] = slice_alike_object_to_indices(
        slice_alike_object=idx, max_length=max_length
    )

    elements: List[X] = []
    for single_idx in indexes_iterator:
        single_element = single_element_getter(int(single_idx))
        elements.append(single_element)

    if len(elements) == 1:
        return elements[0]

    return collate_fn(elements)


def slice_alike_object_to_indices(
    slice_alike_object: Union[slice, int, Iterable[int], Tensor, ndarray],
    max_length: int,
) -> Iterable[int]:
    """
    Utility function used to obtain the sequence of indices given a slice
    object.

    This fuction offers some additional flexibility by also accepting generic
    Iterable[int], PyTorch Tensor and NumPy ndarray.

    Beware that this function only supports 1-D slicing.

    This will also take care of managing negative indices.

    If the input object is a native slice or int, then negative indices will be
    managed as usual (like when used on a native Python list). If a tensor or
    generic iterable is passed, then indices will be transformed as they where
    int(s).
    """

    indexes_iterator: Iterable[int]
    check_bounds = True

    # Makes dataset sliceable
    if isinstance(slice_alike_object, (int, np.integer)):
        # Single index (Python native or NumPy)
        indexes_iterator = [int(slice_alike_object)]
    elif isinstance(slice_alike_object, slice):
        # Slice object (Python native)
        indexes_iterator = range(*slice_alike_object.indices(max_length))
        check_bounds = False
    elif hasattr(slice_alike_object, "shape"):
        tensor_shape = getattr(slice_alike_object, "shape")
        if len(tensor_shape) == 0:
            # Manages 0-d ndarray / Tensor
            indexes_iterator = [int(slice_alike_object)]  # type: ignore
        else:
            if len(tensor_shape) == 1:
                if tensor_shape[0] == 0:
                    # Empty Tensor (NumPy or PyTorch)
                    indexes_iterator = []
                else:
                    # Flat Tensor (NumPy or PyTorch)
                    indexes_iterator = slice_alike_object.tolist()  # type: ignore
            else:
                # Last attempt
                indexes_iterator = [slice_alike_object.item()]  # type: ignore
            if len(indexes_iterator) > 0:  # type: ignore
                assert isinstance(indexes_iterator, int)
    else:
        # Generic iterable
        indexes_iterator = slice_alike_object  # type: ignore

    if check_bounds:
        # Executed only if slice_alike_object is not a slice
        # (because "slice" already takes care of managing negative indices)

        # This will:
        # 1) transform negative indices to positive indices
        # 2) check that max(indices) < max_length
        # 3) check that min(indices) >= 0

        iterator_as_list = []
        for idx in indexes_iterator:
            assert isinstance(idx, int)
            if idx >= 0:
                if idx >= max_length:
                    raise IndexError(
                        f"Index {idx} out of range for sequence "
                        f"of length {max_length}"
                    )
            else:
                pos_idx = max_length - idx  # Negative to positive
                if pos_idx < 0:
                    raise IndexError(
                        f"Index {idx} out of range for sequence "
                        f"of length {max_length}"
                    )
                idx = pos_idx

            iterator_as_list.append(idx)
        indexes_iterator = iterator_as_list

    return indexes_iterator  # type: ignore


__all__ = [
    "SubSequence",
    "SubsetWithTargets",
    "SequenceDataset",
    "find_list_from_index",
    "manage_advanced_indexing",
    "slice_alike_object_to_indices",
]
