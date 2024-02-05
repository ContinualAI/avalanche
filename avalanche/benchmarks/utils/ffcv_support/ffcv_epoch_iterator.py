"""
Custom version of the FFCV epoch iterator.
"""

from threading import Thread, Event, Lock
from queue import Queue
from typing import List, Sequence, TYPE_CHECKING

from ffcv.traversal_order.quasi_random import QuasiRandom
from ffcv.loader.epoch_iterator import (
    EpochIterator as FFCVEpochIterator,
    QUASIRANDOM_ERROR_MSG,
)

import torch

if TYPE_CHECKING:
    from avalanche.benchmarks.utils.ffcv_support.ffcv_loader import _CustomLoader

IS_CUDA = torch.cuda.is_available()


class AtomicCounter:
    """
    An atomic, thread-safe incrementing counter.

    Based on:
    https://gist.github.com/benhoyt/8c8a8d62debe8e5aa5340373f9c509c7
    """

    def __init__(self):
        """Initialize a new atomic counter to 0."""
        self.value = 0
        self._lock = Lock()

    def increment(self):
        """
        Atomically increment the counter by 1 and return the
        previous value.
        """
        with self._lock:
            prev_value = self.value
            self.value += 1
            return prev_value


class _QueueWithIndex(Queue):
    """
    A Python Queue that also returns the index of the inserted element.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counter = AtomicCounter()

    def _put(self, item):
        item_index = self._counter.increment()
        super()._put((item_index, item))


class _CustomEpochIterator(FFCVEpochIterator, Thread):
    def __init__(self, loader: "_CustomLoader", batches: Sequence[List[int]]):
        Thread.__init__(self, daemon=True)
        self.loader: "_CustomLoader" = loader
        self.metadata = loader.reader.metadata
        self.current_batch_slot = 0
        self.batches = batches
        self.iter_ixes = iter(batches)
        self.closed = False
        self.output_queue = _QueueWithIndex(self.loader.batches_ahead)
        self.terminate_event = Event()
        self.memory_context = self.loader.memory_manager.schedule_epoch(batches)

        if IS_CUDA:
            self.current_stream = torch.cuda.current_stream()

        try:
            self.memory_context.__enter__()
        except MemoryError as e:
            if not isinstance(loader.traversal_order, QuasiRandom):
                print(QUASIRANDOM_ERROR_MSG)
                print("Full error below:")

            raise e

        self.storage_state = self.memory_context.state

        self.cuda_streams = [
            (torch.cuda.Stream() if IS_CUDA else None)
            for _ in range(self.loader.batches_ahead + 2)
        ]

        max_batch_size = max(map(len, batches), default=0)

        self.memory_allocations = self.loader.graph.allocate_memory(
            max_batch_size, self.loader.batches_ahead + 2
        )

        self.start()

    def __next__(self):
        result = self.output_queue.get()
        batch_index, result = result

        if result is None:
            self.close()
            raise StopIteration()

        slot, result = result
        indices = list(self.batches[batch_index])

        if IS_CUDA:
            stream = self.cuda_streams[slot]
            # We wait for the copy to be done
            self.current_stream.wait_stream(stream)

        return indices, result


__all__ = ["_CustomEpochIterator"]
