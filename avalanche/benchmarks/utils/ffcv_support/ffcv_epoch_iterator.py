"""
Custom version of the FFCV epoch iterator.
"""
from threading import Thread, Event
from queue import Queue
from typing import List, Sequence, TYPE_CHECKING

import torch as ch

from ffcv.traversal_order.quasi_random import QuasiRandom
from ffcv.loader.epoch_iterator import (
    EpochIterator as FFCVEpochIterator,
    QUASIRANDOM_ERROR_MSG,
)

if TYPE_CHECKING:
    from avalanche.benchmarks.utils.ffcv_support.ffcv_loader import Loader

IS_CUDA = ch.cuda.is_available()


class EpochIterator(FFCVEpochIterator, Thread):
    def __init__(self, loader: "Loader", batches: Sequence[List[int]]):
        Thread.__init__(self, daemon=True)
        self.loader: "Loader" = loader
        self.metadata = loader.reader.metadata
        self.current_batch_slot = 0
        self.iter_ixes = iter(batches)
        self.closed = False
        self.output_queue = Queue(self.loader.batches_ahead)
        self.terminate_event = Event()
        self.memory_context = self.loader.memory_manager.schedule_epoch(batches)

        if IS_CUDA:
            self.current_stream = ch.cuda.current_stream()

        try:
            self.memory_context.__enter__()
        except MemoryError as e:
            if not isinstance(loader.traversal_order, QuasiRandom):
                print(QUASIRANDOM_ERROR_MSG)
                print("Full error below:")

            raise e

        self.storage_state = self.memory_context.state

        self.cuda_streams = [
            (ch.cuda.Stream() if IS_CUDA else None)
            for _ in range(self.loader.batches_ahead + 2)
        ]

        max_batch_size = max(map(len, batches), default=0)

        self.memory_allocations = self.loader.graph.allocate_memory(
            max_batch_size, self.loader.batches_ahead + 2
        )

        self.start()


__all__ = ["EpochIterator"]
