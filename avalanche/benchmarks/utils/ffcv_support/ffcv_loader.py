"""
Custom version of the FFCV loader that accepts a batch sampler.
"""

from typing import Any, Callable, List, Mapping, Optional, Sequence, Type, Union
import warnings

from ffcv.fields.base import Field

import torch as ch

from torch.utils.data.sampler import BatchSampler, Sampler

from ffcv.loader.loader import (
    Loader as FFCVLoader,
    OrderOption,
    ORDER_TYPE,
    DEFAULT_OS_CACHE,
)

from ffcv.traversal_order.base import TraversalOrder
from ffcv.pipeline.operation import Operation
from ffcv.pipeline import Compiler

from avalanche.benchmarks.utils.ffcv_support.ffcv_epoch_iterator import (
    _CustomEpochIterator,
)


class _TraversalOrderAsSampler(Sampler[int]):
    def __init__(self, traversal_order: TraversalOrder):
        self.traversal_order: TraversalOrder = traversal_order
        self.current_epoch: int = 0

    def __iter__(self):
        yield from self.traversal_order.sample_order(self.current_epoch)

    def __len__(self):
        return len(self.traversal_order.indices)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch


class _CustomLoader(FFCVLoader):
    """
    Customized FFCV loader class that can be used as a drop-in replacement
    for standard (e.g. PyTorch) data loaders.

    Differently from the original FFCV loader, this version also accepts a batch sampler.

    Parameters
    ----------
    fname: str
        Full path to the location of the dataset (.beton file format).
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers used for data loading. Consider using the actual number of cores instead of the number of threads if you only use JITed augmentations as they usually don't benefit from hyper-threading.
    os_cache : bool
        Leverages the operating for caching purposes. This is beneficial when there is enough memory to cache the dataset and/or when multiple processes on the same machine training using the same dataset. See https://docs.ffcv.io/performance_guide.html for more information.
    order : Union[OrderOption, TraversalOrder]
        Traversal order, one of: SEQEUNTIAL, RANDOM, QUASI_RANDOM, or a custom TraversalOrder

        QUASI_RANDOM is a random order that tries to be as uniform as possible while minimizing the amount of data read from the disk. Note that it is mostly useful when `os_cache=False`. Currently unavailable in distributed mode.
    distributed : bool
        For distributed training (multiple GPUs). Emulates the behavior of DistributedSampler from PyTorch.
    seed : int
        Random seed for batch ordering.
    indices : Sequence[int]
        Subset of dataset by filtering only some indices.
    pipelines : Mapping[str, Sequence[Union[Operation, torch.nn.Module]]
        Dictionary defining for each field the sequence of Decoders and transforms to apply.
        Fileds with missing entries will use the default pipeline, which consists of the default decoder and `ToTensor()`,
        but a field can also be disabled by explicitly by passing `None` as its pipeline.
    custom_fields : Mapping[str, Field]
        Dictonary informing the loader of the types associated to fields that are using a custom type.
    drop_last : bool
        Drop non-full batch in each iteration.
    batches_ahead : int
        Number of batches prepared in advance; balances latency and memory.
    recompile : bool
        Recompile every iteration. This is necessary if the implementation of some augmentations are expected to change during training.
    batch_sampler : BatchSampler
        If not None, will ignore `batch_size`, `indices`, `drop_last` and will use this sampler instead.
        The batch sampler must be an iterable that outputs lists of int (the indices of examples to include in each batch).
        When running in a distributed training setup, the BatchSampler should already wrap a DistributedSampler.
    """

    def __init__(
        self,
        fname: str,
        batch_size: int,
        num_workers: int = -1,
        os_cache: bool = DEFAULT_OS_CACHE,
        order: Union[ORDER_TYPE, TraversalOrder] = OrderOption.SEQUENTIAL,
        distributed: bool = False,
        seed: Optional[int] = None,  # For ordering of samples
        indices: Optional[Sequence[int]] = None,  # For subset selection
        pipelines: Mapping[str, Sequence[Union[Operation, ch.nn.Module]]] = {},
        custom_fields: Mapping[str, Type[Field]] = {},
        drop_last: bool = True,
        batches_ahead: int = 3,
        recompile: bool = False,  # Recompile at every epoch
        batch_sampler: Optional[Sampler[List[int]]] = None,
    ):
        # Set batch sampler to an empty list so that next_traversal_order()
        # and __len__() work when running super().__init__(...)
        self.batch_sampler: Sampler[List[int]] = []

        super().__init__(
            fname=fname,
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=os_cache,
            order=order,
            distributed=distributed,
            seed=seed,
            indices=indices,
            pipelines=pipelines,
            custom_fields=custom_fields,
            drop_last=drop_last,
            batches_ahead=batches_ahead,
            recompile=recompile,
        )

        self._args["batch_sampler"] = batch_sampler

        if batch_sampler is None:
            batch_sampler = BatchSampler(
                _TraversalOrderAsSampler(self.traversal_order),
                batch_size=batch_size,
                drop_last=drop_last,
            )

        self.batch_sampler = batch_sampler

    def next_traversal_order(self):
        # Manage distributed sampler, which has to know the id of the current epoch
        self._batch_sampler_set_epoch()

        return list(self.batch_sampler)

    def __iter__(self):
        Compiler.set_num_threads(self.num_workers)
        order = self.next_traversal_order()
        self.next_epoch += 1

        # Compile at the first epoch
        if self.code is None or self.recompile:
            self.generate_code()

        return _CustomEpochIterator(self, order)

    def filter(self, field_name: str, condition: Callable[[Any], bool]) -> "FFCVLoader":
        if self._args["batch_sampler"] is not None:
            warnings.warn(
                "The original loader was created by passing a batch sampler. "
                "The filtered loader will not inherit the sampler!"
            )

        return super().filter(field_name, condition)

    def __len__(self):
        return len(self.batch_sampler)

    def _batch_sampler_set_epoch(self):
        if hasattr(self.batch_sampler, "set_epoch"):
            # Supports batch samplers with set_epoch method
            self.batch_sampler.set_epoch(self.next_epoch)
        else:
            # Standard setup: the batch sampler wraps a TraversalOrder or
            # a distributed sampler
            if hasattr(self.batch_sampler, "sampler"):
                if hasattr(self.batch_sampler.sampler, "set_epoch"):
                    self.batch_sampler.sampler.set_epoch(self.next_epoch)


__all__ = ["_CustomLoader"]
